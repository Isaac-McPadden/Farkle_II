import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("sklearn")

from farkle.analysis import hgb_feat, run_hgb
from farkle.config import AppConfig, IOConfig
from farkle.simulation.strategies import STRATEGY_TUPLE_FIELDS, ThresholdStrategy, strategy_tuple
from farkle.utils.artifact_contract import (
    make_artifact_sidecar,
    validate_artifact_sidecar,
    write_artifact_with_sidecar_atomic,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.stage_completion import stage_done_path, write_stage_done


def test_hgb_external_random_state_is_direct_coordinate_owned() -> None:
    selected = run_hgb._model_random_state(32, 2, 1).bytes(64)

    assert selected == run_hgb._model_random_state(32, 2, 1).bytes(64)
    assert selected != run_hgb._model_random_state(33, 2, 1).bytes(64)
    assert selected != run_hgb._model_random_state(32, 4, 1).bytes(64)
    assert selected != run_hgb._model_random_state(32, 2, 2).bytes(64)


def _setup_cfg(tmp_path: Path) -> tuple[AppConfig, Path]:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    cfg.sim.n_players_list = [2]
    analysis_dir = cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.performance_by_k_path(2)
    frame = pd.DataFrame(
        {
            "strategy": ["Strat(300,2)[SD][FOFS][AND][H-]"],
            "k": [2],
            "win_rate": [0.5],
        }
    )
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sidecar = make_artifact_sidecar(
        cfg,
        metrics_path,
        producer="test",
        scope="by_k",
        source_scope="by_k",
        operation="aggregate_strategy_outcomes",
        consistency_columns=table.schema.names,
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
    )
    write_parquet_artifact_atomic(table, metrics_path, sidecar=sidecar)
    manifest_path = cfg.strategy_manifest_root_path()
    manifest_table = pa.table({"strategy_id": [0], "score_threshold": [300]})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(manifest_table, manifest_path)
    os.utime(metrics_path, (1000, 1000))
    return cfg, metrics_path


def _hgb_outputs(cfg: AppConfig) -> list[Path]:
    return [
        cfg.across_k_dir("hgb") / "hgb_importance.json",
        cfg.hgb_future_proposals_path(),
        cfg.concat_ks_dir("hgb") / hgb_feat._hgb.LONG_IMPORTANCE_NAME,
        cfg.across_k_dir("hgb") / hgb_feat._hgb.OVERALL_IMPORTANCE_NAME,
        cfg.hgb_importance_path(2),
        cfg.hgb_predictive_scores_path(2),
        cfg.hgb_fold_metrics_path(2),
    ]


def _publish_output_placeholders(cfg: AppConfig, *, mtime: float) -> None:
    outputs = _hgb_outputs(cfg)
    for path in outputs:
        is_per_k = "by_k" in path.parts
        scope = "by_k" if is_per_k else "concat_ks" if "concat_ks" in path.parts else "across_k"
        metadata = make_artifact_sidecar(
            cfg,
            path,
            producer="test",
            scope=scope,
            source_scope="by_k",
            operation="heldout_prediction",
            player_counts=[2],
            required_player_counts=[2],
            missing_cell_policy="fail",
        )

        def _write_placeholder(staged: Path) -> None:
            staged.write_text("{}")

        write_artifact_with_sidecar_atomic(path, metadata, _write_placeholder)
        os.utime(path, (mtime, mtime))


def test_hgb_feat_skips_when_up_to_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, _metrics = _setup_cfg(tmp_path)
    _publish_output_placeholders(cfg, mtime=1010)
    outputs = [
        cfg.across_k_dir("hgb") / "hgb_importance.json",
        cfg.hgb_future_proposals_path(),
        cfg.concat_ks_dir("hgb") / hgb_feat._hgb.LONG_IMPORTANCE_NAME,
        cfg.across_k_dir("hgb") / hgb_feat._hgb.OVERALL_IMPORTANCE_NAME,
        cfg.hgb_importance_path(2),
        cfg.hgb_predictive_scores_path(2),
        cfg.hgb_fold_metrics_path(2),
    ]
    write_stage_done(
        stage_done_path(cfg.hgb_stage_dir, "hgb"),
        inputs=[cfg.performance_by_k_path(2), cfg.strategy_manifest_root_path()],
        outputs=outputs,
        cfg=cfg,
        stage="hgb",
        freshness_key=hgb_feat._hgb_freshness_key(cfg),
        sidecar_artifacts=outputs,
    )

    def boom(**kwargs):  # pragma: no cover - should not be called
        raise AssertionError("_hgb.run_hgb should not be called when up-to-date")

    monkeypatch.setattr(hgb_feat._hgb, "run_hgb", boom)
    hgb_feat.run(cfg)


def test_hgb_feat_runs_when_outdated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, metrics = _setup_cfg(tmp_path)
    json_out = cfg.across_k_dir("hgb") / "hgb_importance.json"
    _publish_output_placeholders(cfg, mtime=900)
    os.utime(metrics, (1020, 1020))

    called = {}

    def fake_run(
        *,
        cfg: AppConfig,
        metrics_paths: list[Path],
        manifest_path: Path | None,
    ) -> None:
        assert cfg is not None
        assert metrics_paths == [cfg.performance_by_k_path(2)]
        assert manifest_path == cfg.strategy_manifest_root_path()
        called["output"] = json_out

    monkeypatch.setattr(hgb_feat._hgb, "run_hgb", fake_run)
    hgb_feat.run(cfg)
    assert called
    assert not any(cfg.analysis_dir.glob("*.pkl"))


@pytest.mark.parametrize(
    "mutation",
    ["target", "features", "parameter", "output", "code", "method", "sidecar"],
)
def test_hgb_authenticated_completion_recomputes_on_contract_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    cfg, metrics = _setup_cfg(tmp_path)
    calls = 0

    def fake_run(**_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        _publish_output_placeholders(cfg, mtime=1010 + calls)

    monkeypatch.setattr(hgb_feat._hgb, "run_hgb", fake_run)
    hgb_feat.run(cfg)
    hgb_feat.run(cfg)
    assert calls == 1, "unchanged authenticated HGB work must skip"

    if mutation == "target":
        frame = pd.DataFrame(
            {
                "strategy": ["Strat(300,2)[SD][FOFS][AND][H-]"],
                "k": [2],
                "win_rate": [0.6],
            }
        )
        table = pa.Table.from_pandas(frame, preserve_index=False)
        metadata = make_artifact_sidecar(
            cfg,
            metrics,
            producer="test",
            scope="by_k",
            source_scope="by_k",
            operation="aggregate_strategy_outcomes",
            consistency_columns=table.schema.names,
            player_counts=[2],
            required_player_counts=[2],
            missing_cell_policy="fail",
        )
        write_parquet_artifact_atomic(table, metrics, sidecar=metadata)
    elif mutation == "features":
        pq.write_table(
            pa.table({"strategy_id": [0], "score_threshold": [350]}),
            cfg.strategy_manifest_root_path(),
        )
    elif mutation == "parameter":
        cfg.hgb.max_depth += 1
    elif mutation == "output":
        with cfg.hgb_importance_path(2).open("ab") as handle:
            handle.write(b"changed")
    elif mutation == "code":
        cfg._code_identity = {
            "commit": "f" * 40,
            "policy": "development_dirty",
            "state": "development_dirty",
            "dirty_fingerprint_sha256": "e" * 64,
        }
    elif mutation == "method":
        monkeypatch.setattr(
            hgb_feat,
            "HGB_METHOD_VERSION",
            hgb_feat.HGB_METHOD_VERSION + 1,
        )
    else:
        sidecar_path = cfg.hgb_importance_path(2).with_name(
            f"{cfg.hgb_importance_path(2).name}.sidecar.json"
        )
        sidecar_path.write_text("{}", encoding="utf-8")

    hgb_feat.run(cfg)
    assert calls == 2


def test_hgb_force_recomputes_and_corrupt_bytes_cannot_be_blessed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, _metrics = _setup_cfg(tmp_path)
    calls = 0

    def fake_run(**_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        _publish_output_placeholders(cfg, mtime=1010 + calls)

    monkeypatch.setattr(hgb_feat._hgb, "run_hgb", fake_run)
    hgb_feat.run(cfg)
    hgb_feat.run(cfg, force=True)
    assert calls == 2

    corrupted = cfg.hgb_importance_path(2)
    metadata = make_artifact_sidecar(
        cfg,
        corrupted,
        producer="test",
        scope="by_k",
        source_scope="by_k",
        operation="heldout_prediction",
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
    )
    def write_corruption(staged: Path) -> None:
        staged.write_bytes(b"corrupt-but-sidecar-matches")

    write_artifact_with_sidecar_atomic(corrupted, metadata, write_corruption)
    validate_artifact_sidecar(corrupted)

    hgb_feat.run(cfg)
    assert calls == 3, "a newly matching sidecar must not bless bytes absent from completion"


def test_hgb_feat_returns_when_metrics_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    cfg.sim.n_players_list = [2]
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)

    called = False

    def boom(**kwargs):  # pragma: no cover - should not be called
        nonlocal called
        called = True
        raise AssertionError("_hgb.run_hgb should not be called when metrics are missing")

    monkeypatch.setattr(hgb_feat._hgb, "run_hgb", boom)

    hgb_feat.run(cfg)

    assert called is False


def test_hgb_feat_returns_when_canonical_performance_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    cfg.sim.n_players_list = [2]
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    called = False

    def boom(**kwargs):  # pragma: no cover - should not be called
        nonlocal called
        called = True
        raise AssertionError("_hgb.run_hgb should not be called without canonical performance")

    monkeypatch.setattr(hgb_feat._hgb, "run_hgb", boom)

    hgb_feat.run(cfg)

    assert called is False


def test_configuration_run_writes_heldout_artifacts_and_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    cfg.sim.n_players_list = [2]
    strategy_objects = [
        ThresholdStrategy(score_threshold=score, dice_threshold=dice)
        for score, dice in ((200, 1), (300, 2), (400, 1), (500, 2))
    ]
    strategy_ids = list(range(len(strategy_objects)))
    manifest = pd.DataFrame(
        [
            {
                "strategy_id": strategy_id,
                **dict(zip(STRATEGY_TUPLE_FIELDS, strategy_tuple(strategy), strict=True)),
            }
            for strategy_id, strategy in zip(strategy_ids, strategy_objects, strict=True)
        ]
    )
    manifest["favor_dice_or_score"] = manifest["favor_dice_or_score"].map(lambda value: value.value)
    manifest_path = cfg.strategy_manifest_root_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    performance = pd.DataFrame(
        {
            "root_seed": [0] * 4,
            "k": [2] * 4,
            "strategy": strategy_ids,
            "win_rate": [0.35, 0.45, 0.55, 0.65],
        }
    )
    source = cfg.performance_by_k_path(2)
    table = pa.Table.from_pandas(performance, preserve_index=False)
    source_sidecar = make_artifact_sidecar(
        cfg,
        source,
        producer="test",
        scope="by_k",
        source_scope="by_k",
        operation="aggregate_strategy_outcomes",
        consistency_columns=table.schema.names,
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
    )
    write_parquet_artifact_atomic(table, source, sidecar=source_sidecar)
    hgb_feat.run(cfg)

    outputs = [
        cfg.hgb_importance_path(2),
        cfg.hgb_predictive_scores_path(2),
        cfg.hgb_fold_metrics_path(2),
        cfg.hgb_future_proposals_path(),
        cfg.concat_ks_dir("hgb") / hgb_feat._hgb.LONG_IMPORTANCE_NAME,
        cfg.across_k_dir("hgb") / hgb_feat._hgb.OVERALL_IMPORTANCE_NAME,
        cfg.across_k_dir("hgb") / "hgb_importance.json",
    ]
    for output in outputs:
        metadata = validate_artifact_sidecar(output)
        assert metadata.code_revision != "unknown"
        assert set(metadata.source_artifacts) == {str(source), str(manifest_path)}
    concat_metadata = validate_artifact_sidecar(
        cfg.concat_ks_dir("hgb") / hgb_feat._hgb.LONG_IMPORTANCE_NAME,
        expected={
            "scope": "concat_ks",
            "operation": "concatenate",
            "weighted_quantity": "heldout_permutation_association_importance",
        },
    )
    assert concat_metadata.k_aggregation_method == "none"
    validate_artifact_sidecar(
        cfg.across_k_dir("hgb") / hgb_feat._hgb.OVERALL_IMPORTANCE_NAME,
        expected={
            "scope": "across_k",
            "operation": "equal_k_mean",
            "weighted_quantity": "heldout_permutation_association_importance",
            "conditioning": "finite_grid_predictive_association_not_causal",
        },
    )
    predictions = pd.read_parquet(cfg.hgb_predictive_scores_path(2))
    assert set(predictions["strategy"]) == set(strategy_ids)
    assert len(predictions) == len(strategy_ids)
    assert predictions.groupby("strategy")["fold"].nunique().eq(1).all()
    proposals = pd.read_parquet(cfg.hgb_future_proposals_path())
    if not proposals.empty:
        assert proposals["included_in_current_analysis"].eq(False).all()
        assert proposals["strategy_id"].isna().all()
