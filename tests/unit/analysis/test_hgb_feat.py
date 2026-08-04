import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("sklearn")

from tests.helpers.artifact_sidecars import (
    clean_test_code_identity,
    make_authenticated_v3_config,
    mutate_artifact_bytes,
    publish_v3_parquet,
    publish_v3_strategy_manifest,
)

from farkle.analysis import hgb_feat, run_hgb
from farkle.config import AppConfig, IOConfig
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils.artifact_contract import (
    make_artifact_sidecar,
    validate_artifact_sidecar,
    write_artifact_with_sidecar_atomic,
)
from farkle.utils.authenticated_contract import validate_authenticated_artifact_unbound
from farkle.utils.stage_completion import stage_done_path, write_stage_done


def test_hgb_external_random_state_is_direct_coordinate_owned() -> None:
    selected = run_hgb._model_random_state(32, 2, 1).bytes(64)

    assert selected == run_hgb._model_random_state(32, 2, 1).bytes(64)
    assert selected != run_hgb._model_random_state(33, 2, 1).bytes(64)
    assert selected != run_hgb._model_random_state(32, 4, 1).bytes(64)
    assert selected != run_hgb._model_random_state(32, 2, 2).bytes(64)


def _setup_cfg(tmp_path: Path) -> tuple[AppConfig, Path]:
    cfg = make_authenticated_v3_config(tmp_path, name="hgb", root_seed=11)
    analysis_dir = cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.performance_by_k_path(2)
    frame = pd.DataFrame(
        {
            "strategy": pd.array([0], dtype="Int32"),
            "k": [2],
            "win_rate": [0.5],
        }
    )
    table = pa.Table.from_pandas(frame, preserve_index=False)
    publish_v3_parquet(
        cfg,
        metrics_path,
        table,
        stage_key="metrics",
        producer="test",
        source_scope="by_k",
        operation="aggregate_strategy_outcomes",
    )
    publish_v3_strategy_manifest(
        cfg,
        (ThresholdStrategy(score_threshold=300, dice_threshold=2, strategy_id=0),),
    )
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
    sources = [cfg.performance_by_k_path(2), cfg.strategy_manifest_root_path()]
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
            source_artifacts=sources,
            player_counts=[2],
            required_player_counts=[2],
            missing_cell_policy="fail",
        )

        def _write_placeholder(
            staged: Path,
            *,
            is_parquet: bool = path.suffix == ".parquet",
        ) -> None:
            if is_parquet:
                pq.write_table(pa.table({"fixture_value": [1]}), staged)
            else:
                staged.write_text("{}", encoding="utf-8")

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
                "strategy": pd.array([0], dtype="Int32"),
                "k": [2],
                "win_rate": [0.6],
            }
        )
        table = pa.Table.from_pandas(frame, preserve_index=False)
        publish_v3_parquet(
            cfg,
            metrics,
            table,
            stage_key="metrics",
            producer="test",
            source_scope="by_k",
            operation="aggregate_strategy_outcomes",
        )
    elif mutation == "features":
        publish_v3_strategy_manifest(
            cfg,
            (ThresholdStrategy(score_threshold=350, dice_threshold=2, strategy_id=0),),
        )
    elif mutation == "parameter":
        cfg.hgb.max_depth += 1
    elif mutation == "output":
        with cfg.hgb_importance_path(2).open("ab") as handle:
            handle.write(b"changed")
    elif mutation == "code":
        cfg._code_identity = clean_test_code_identity("f" * 40)
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
    validate_artifact_sidecar(corrupted)
    mutate_artifact_bytes(corrupted)

    hgb_feat.run(cfg)
    assert calls == 3, "a newly matching sidecar must not bless bytes absent from completion"


@pytest.mark.parametrize("mutation", ["missing", "mutated"])
def test_hgb_rejects_invalid_strategy_manifest_sidecar_at_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    cfg, _metrics = _setup_cfg(tmp_path)
    calls = 0

    def fake_run(**_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        _publish_output_placeholders(cfg, mtime=1010 + calls)

    monkeypatch.setattr(hgb_feat._hgb, "run_hgb", fake_run)
    hgb_feat.run(cfg)
    assert calls == 1, "valid authenticated HGB control must reach the consumer"

    manifest_sidecar = cfg.strategy_manifest_root_path().with_name(
        f"{cfg.strategy_manifest_root_path().name}.sidecar.json"
    )
    if mutation == "missing":
        manifest_sidecar.unlink()
        expected = "missing sidecar"
    else:
        manifest_sidecar.write_text("{}", encoding="utf-8")
        expected = "artifact_contract_version"

    with pytest.raises(RuntimeError, match=expected):
        hgb_feat.run(cfg)
    assert calls == 1, "invalid manifest authentication must be rejected before HGB fitting"


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
    cfg = make_authenticated_v3_config(tmp_path, name="configuration", root_seed=0)
    strategy_objects = [
        ThresholdStrategy(score_threshold=score, dice_threshold=dice)
        for score, dice in ((200, 1), (300, 2), (400, 1), (500, 2))
    ]
    strategy_ids = list(range(len(strategy_objects)))
    canonical_strategies = tuple(
        replace(strategy, strategy_id=strategy_id)
        for strategy_id, strategy in zip(strategy_ids, strategy_objects, strict=True)
    )
    manifest_path = publish_v3_strategy_manifest(cfg, canonical_strategies)
    performance = pd.DataFrame(
        {
            "root_seed": [0] * 4,
            "k": [2] * 4,
            "strategy": strategy_ids,
            "win_rate": [0.35, 0.45, 0.55, 0.65],
        }
    )
    performance["strategy"] = pd.array(performance["strategy"].tolist(), dtype="Int32")
    source = cfg.performance_by_k_path(2)
    table = pa.Table.from_pandas(performance, preserve_index=False)
    publish_v3_parquet(
        cfg,
        source,
        table,
        stage_key="metrics",
        producer="test",
        source_scope="by_k",
        operation="aggregate_strategy_outcomes",
    )
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
        authenticated = validate_authenticated_artifact_unbound(
            output,
            validate_provenance=False,
        )
        assert {
            identity.artifact.location.path(cfg) for identity in authenticated.source_artifacts
        } == {source, manifest_path}
    concat_metadata = validate_artifact_sidecar(
        cfg.concat_ks_dir("hgb") / hgb_feat._hgb.LONG_IMPORTANCE_NAME,
        expected={
            "scope": "concat_ks",
            "operation": "concatenate",
            "weighted_quantity": "heldout_permutation_association_importance",
        },
    )
    performance["strategy"] = pd.array(performance["strategy"].tolist(), dtype="Int32")
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
