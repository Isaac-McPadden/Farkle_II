import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from tests.helpers.artifact_sidecars import (
    make_authenticated_v3_config,
    publish_v3_parquet,
)
from tests.helpers.diagnostic_fixtures import build_curated_fixture

from farkle.analysis import game_stats
from farkle.config import AppConfig, assign_config_sha
from farkle.utils.artifact_contract import sidecar_path, validate_artifact_sidecar


def test_rare_event_flags_cover_game_and_strategy_levels(tmp_path):
    cfg, _, per_n = build_curated_fixture(tmp_path)
    thresholds = (10, 60)

    output_path = tmp_path / "rare_events.parquet"
    rows = game_stats._rare_event_flags(
        [(2, per_n)],
        thresholds=thresholds,
        target_score=100,
        output_path=output_path,
        codec=cfg.parquet_codec,
    )

    assert rows > 0
    flags = pd.read_parquet(output_path)
    # Three game-level rows plus strategy-level and n-player summaries
    assert {"game", "strategy", "n_players"} <= set(flags["summary_level"].unique())

    aggro = flags[(flags["strategy"] == 1) & (flags["summary_level"] == "strategy")].iloc[0]
    assert aggro["observations"] == 3
    assert aggro["multi_reached_target"] == pytest.approx(2 / 3)
    assert aggro[f"margin_le_{thresholds[1]}"] == pytest.approx(1.0)


def test_summarize_rounds_handles_empty_and_values():
    empty = game_stats._summarize_rounds([])
    assert empty["observations"] == 0
    assert pd.isna(empty["mean_rounds"])

    stats = game_stats._summarize_rounds([1, 5, 9])
    assert stats["observations"] == 3
    assert stats["prob_rounds_le_5"] == pytest.approx(2 / 3)


def _cfg(
    tmp_path: Path,
    *,
    player_counts: tuple[int, ...] = (2,),
    root_seed: int = 0,
) -> AppConfig:
    return make_authenticated_v3_config(
        tmp_path,
        name="game_stats",
        root_seed=root_seed,
        player_counts=player_counts,
    )


def _canonical_strategy_columns(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    for column in result.columns:
        if column.endswith("_strategy"):
            result[column] = result[column].astype("Int32")
    return result


def _publish_curated(cfg: AppConfig, path: Path, rows: pd.DataFrame) -> Path:
    table = pa.Table.from_pandas(
        _canonical_strategy_columns(rows),
        preserve_index=False,
    ).replace_schema_metadata(None)
    return publish_v3_parquet(
        cfg,
        path,
        table,
        stage_key="curate",
        producer="curate",
        operation="curate_game_rows",
        source_scope="by_k",
    )


def _publish_combined(
    cfg: AppConfig,
    path: Path,
    rows: pd.DataFrame,
    *,
    sources: tuple[Path, ...],
) -> Path:
    table = pa.Table.from_pandas(
        _canonical_strategy_columns(rows),
        preserve_index=False,
    ).replace_schema_metadata(None)
    return publish_v3_parquet(
        cfg,
        path,
        table,
        stage_key="combine",
        producer="combine",
        operation="concatenate",
        sources=sources,
        source_scope="by_k",
    )


def _build_parquet(tmp_path: Path, cfg: AppConfig) -> tuple[Path, Path]:
    rows = pd.DataFrame(
        [
            {
                "termination_status": "completed",
                "seat_ranks": ["P1", "P2"],
                "n_rounds": 4,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 120,
                "P2_score": 110,
            },
            {
                "termination_status": "completed",
                "seat_ranks": ["P2", "P1"],
                "n_rounds": 8,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 50,
                "P2_score": 200,
            },
            {
                "termination_status": "completed",
                "seat_ranks": ["P1", "P2"],
                "n_rounds": 12,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 300,
                "P2_score": 100,
            },
        ]
    )

    per_n_path = cfg.ingested_rows_curated(2)
    _publish_curated(cfg, per_n_path, rows)

    combined_path = cfg.curated_parquet
    _publish_combined(cfg, combined_path, rows, sources=(per_n_path,))

    return per_n_path, combined_path


def test_run_generates_all_outputs(tmp_path: Path):
    cfg = _cfg(tmp_path)
    per_n_path, combined_path = _build_parquet(tmp_path, cfg)

    game_stats.run(cfg, force=True)

    game_length = cfg.game_stats_concat_path("game_length.parquet")
    margin_path = cfg.game_stats_concat_path("margin_stats.parquet")
    rare_events_path = cfg.game_stats_output_path("rare_events.parquet")

    assert game_length.exists()
    assert margin_path.exists()
    assert rare_events_path.exists()

    # Strategy summaries come from per-n inputs; global stats come from combined parquet
    game_df = pd.read_parquet(game_length)
    assert {"strategy", "n_players"} <= set(game_df.columns)
    assert any(game_df["summary_level"] == "n_players")

    margin_df = pd.read_parquet(margin_path)
    assert set(margin_df["summary_level"].unique()) == {"strategy"}
    assert all(
        col in margin_df.columns
        for col in ("mean_margin_runner_up", "median_margin_runner_up", "mean_score_spread")
    )

    per_k_stats, _ = game_stats._per_k_game_stats_paths(cfg.game_stats_stage_dir, 2)
    assert per_k_stats.exists()
    per_k_df = pd.read_parquet(per_k_stats)
    assert set(per_k_df["n_players"].dropna().astype(int).unique()) <= {2}

    rare_df = pd.read_parquet(rare_events_path)
    assert {"game", "strategy", "n_players"} <= set(rare_df["summary_level"].unique())

    stamp = cfg.game_stats_stage_dir / "game_stats.done.json"
    assert stamp.exists()
    stamp_meta = json.loads(stamp.read_text())
    assert any(
        item["artifact"]["location"]
        == {
            "scope": "by_k",
            "stage_key": "game_stats",
            "player_count": 2,
            "relative_path": per_k_stats.name,
        }
        for item in stamp_meta["outputs"]
    )

    shard_path, stats_path, _done_path = game_stats._rare_event_shard_paths(rare_events_path, 2)
    for path in (rare_events_path, shard_path, stats_path):
        validate_artifact_sidecar(path)

    original = rare_events_path.read_bytes()
    for path in (rare_events_path, shard_path, stats_path):
        sidecar_path(path).unlink()
    game_stats.run(cfg)
    assert rare_events_path.read_bytes() == original
    for path in (rare_events_path, shard_path, stats_path):
        validate_artifact_sidecar(path)


def test_run_requires_inputs(tmp_path: Path):
    cfg = _cfg(tmp_path)

    game_stats.run(cfg)

    assert not cfg.game_stats_concat_path("game_length.parquet").exists()
    assert not cfg.game_stats_concat_path("margin_stats.parquet").exists()
    assert not cfg.game_stats_output_path("rare_events.parquet").exists()


def test_compute_margins_and_aggregation(tmp_path: Path):
    cfg = _cfg(tmp_path)
    per_n_path, _ = _build_parquet(tmp_path, cfg)

    per_n_inputs = [(2, per_n_path)]
    margins = game_stats._per_strategy_margin_stats(per_n_inputs, thresholds=(100,))
    assert not margins.empty
    assert margins.loc[0, "prob_margin_runner_up_le_100"] == pytest.approx(1 / 3)
    assert margins.loc[0, "prob_score_spread_le_100"] == pytest.approx(1 / 3)

    rare_path = tmp_path / "rare_events.parquet"
    rare_rows = game_stats._rare_event_flags(
        per_n_inputs,
        thresholds=(100,),
        target_score=150,
        output_path=rare_path,
        codec=cfg.parquet_codec,
    )
    assert rare_rows > 0
    rare = pd.read_parquet(rare_path)
    assert set(rare["summary_level"].unique()) >= {"game", "strategy"}


def test_global_stats_warns_when_seat_ranks_missing(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    class DummyDataset:
        schema = type("Schema", (), {"names": ["n_rounds"]})()

        @staticmethod
        def to_table(_columns=None):
            return pa.Table.from_pandas(pd.DataFrame({"n_rounds": [1, 2, 3]}))

    monkeypatch.setattr(game_stats.ds, "dataset", lambda path: DummyDataset())

    with caplog.at_level("WARNING"):
        result = game_stats._global_stats(Path("dummy"))

    assert result.empty
    assert "Combined parquet missing seat_ranks" in caplog.text


def test_global_stats_handles_numpy_array_seat_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyDataset:
        schema = type("Schema", (), {"names": ["seat_ranks", "n_rounds"]})()

        @staticmethod
        def to_table(_columns=None):
            return pa.Table.from_pandas(
                pd.DataFrame(
                    {
                        "seat_ranks": [
                            np.array(["P1", "P2"], dtype=object),
                            np.array(["P2", "P1"], dtype=object),
                        ],
                        "n_rounds": [4, 8],
                    }
                )
            )

    monkeypatch.setattr(game_stats.ds, "dataset", lambda _path: DummyDataset())
    monkeypatch.setattr(game_stats, "n_players_from_schema", lambda _schema: 12)

    result = game_stats._global_stats(Path("dummy"))

    assert not result.empty
    assert set(result["n_players"].astype(int).tolist()) == {2}


def _write_multi_k_curated_inputs(cfg: AppConfig) -> None:
    rows_2p = pd.DataFrame(
        [
            {
                "termination_status": "completed",
                "seat_ranks": ["P1", "P2"],
                "n_rounds": 5,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 120,
                "P2_score": 100,
            },
            {
                "termination_status": "completed",
                "seat_ranks": ["P2", "P1"],
                "n_rounds": 7,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 70,
                "P2_score": 160,
            },
        ]
    )
    rows_3p = pd.DataFrame(
        [
            {
                "termination_status": "completed",
                "seat_ranks": ["P1", "P2", "P3"],
                "n_rounds": 6,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P3_strategy": 3,
                "P1_score": 210,
                "P2_score": 180,
                "P3_score": 120,
            },
            {
                "termination_status": "completed",
                "seat_ranks": ["P3", "P1", "P2"],
                "n_rounds": 10,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P3_strategy": 3,
                "P1_score": 190,
                "P2_score": 150,
                "P3_score": 230,
            },
        ]
    )

    for n_players, rows in ((2, rows_2p), (3, rows_3p)):
        per_n_path = cfg.ingested_rows_curated(n_players)
        _publish_curated(cfg, per_n_path, rows)

    combined = pd.concat([rows_2p, rows_3p], ignore_index=True, sort=False)
    combined_path = cfg.curated_parquet
    _publish_combined(
        cfg,
        combined_path,
        combined,
        sources=tuple(cfg.ingested_rows_curated(k) for k in (2, 3)),
    )


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_writes_per_k_outputs_and_is_idempotent_for_multi_k(tmp_path: Path) -> None:
    k_values = [2, 3]
    cfg = _cfg(tmp_path, player_counts=tuple(k_values), root_seed=123)
    _write_multi_k_curated_inputs(cfg)

    game_stats.run(cfg)

    combined_targets = {
        "game_length.parquet": cfg.game_stats_concat_path("game_length.parquet"),
        "margin_stats.parquet": cfg.game_stats_concat_path("margin_stats.parquet"),
    }
    for output_path in combined_targets.values():
        assert output_path.exists()

    expected_per_k_paths: list[Path] = []
    for k in k_values:
        path, _ = game_stats._per_k_game_stats_paths(cfg.game_stats_stage_dir, k)
        assert path.exists()
        expected_per_k_paths.append(path)
        per_k_df = pd.read_parquet(path)
        assert set(per_k_df["n_players"].dropna().astype(int).unique()) <= {k}

    tracked_paths = list(combined_targets.values()) + expected_per_k_paths
    before = {path: (path.stat().st_mtime_ns, _hash_file(path)) for path in tracked_paths}

    game_stats.run(cfg)

    after = {path: (path.stat().st_mtime_ns, _hash_file(path)) for path in tracked_paths}
    assert after == before


def test_run_resolves_rare_event_thresholds_from_histograms(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.analysis.rare_event_margin_quantile = 0.5
    cfg.analysis.rare_event_target_rate = 0.4
    assign_config_sha(cfg)
    _build_parquet(tmp_path, cfg)

    game_stats.run(cfg, force=True)

    rare_df = pd.read_parquet(cfg.game_stats_output_path("rare_events.parquet"))
    assert "margin_le_150" in rare_df.columns

    strat_row = rare_df[
        (rare_df["summary_level"] == "strategy") & (rare_df["strategy"] == 1.0)
    ].iloc[0]
    assert strat_row["multi_reached_target"] == pytest.approx(2 / 3)


def test_run_generates_margin_summary_columns_and_histogram_inputs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.analysis.game_stats_margin_thresholds = (25, 175)
    cfg.analysis.rare_event_margin_quantile = 0.9
    cfg.analysis.rare_event_target_rate = 0.2
    assign_config_sha(cfg)
    _build_parquet(tmp_path, cfg)

    game_stats.run(cfg, force=True)

    margin_df = pd.read_parquet(cfg.game_stats_concat_path("margin_stats.parquet"))
    combined_margin_df = pd.read_parquet(
        cfg.game_stats_output_path("margin_strategy_conditioned_equal_k_mean.parquet")
    )
    rare_df = pd.read_parquet(cfg.game_stats_output_path("rare_events.parquet"))

    for threshold in (25, 175):
        assert f"prob_margin_runner_up_le_{threshold}" in margin_df.columns
        assert f"prob_score_spread_le_{threshold}" in margin_df.columns
        assert f"prob_margin_runner_up_le_{threshold}" in combined_margin_df.columns
        assert f"prob_score_spread_le_{threshold}" in combined_margin_df.columns

    derived_margin_cols = [c for c in rare_df.columns if c.startswith("margin_le_")]
    assert derived_margin_cols == ["margin_le_200"]


def test_run_rejects_noncanonical_aggregation_method_aliases(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _build_parquet(tmp_path, cfg)
    cfg.k_aggregation.method = "equal_k"
    assign_config_sha(cfg)

    with pytest.raises(ValueError, match="Unknown aggregation scheme"):
        game_stats.run(cfg, force=True)


def test_run_raises_for_invalid_aggregation_method(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _build_parquet(tmp_path, cfg)
    cfg.k_aggregation.method = "invalid-scheme"
    assign_config_sha(cfg)

    with pytest.raises(ValueError, match="Unknown aggregation scheme"):
        game_stats.run(cfg, force=True)


def test_discover_per_n_inputs_handles_partial_layouts(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, player_counts=(2, 3))

    assert game_stats._discover_per_n_inputs(cfg) == []

    valid_dir = cfg.curate_block_dir(2)
    valid_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir = cfg.data_dir / "by_k" / "badp"
    invalid_dir.mkdir(parents=True, exist_ok=True)
    missing_file_dir = cfg.curate_block_dir(3)
    missing_file_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.DataFrame(
        [
            {
                "seat_ranks": ["P1", "P2"],
                "n_rounds": 3,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 100,
                "P2_score": 90,
            }
        ]
    )
    rows.to_parquet(valid_dir / cfg.curated_rows_name)

    discovered = game_stats._discover_per_n_inputs(cfg)
    assert discovered == [(2, valid_dir / cfg.curated_rows_name)]


def test_run_rejects_incomplete_configured_k_support(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, player_counts=(2, 3), root_seed=101)
    _build_parquet(tmp_path, cfg)

    with pytest.raises(FileNotFoundError, match=r"incomplete canonical by-k support: \[3\]"):
        game_stats.run(cfg, force=True)


def test_run_aggregation_alias_and_invalid_via_run(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _build_parquet(tmp_path, cfg)

    cfg.k_aggregation.method = "count"
    assign_config_sha(cfg)
    with pytest.raises(ValueError, match="Unknown aggregation scheme"):
        game_stats.run(cfg, force=True)

    cfg.k_aggregation.method = "definitely-bad"
    assign_config_sha(cfg)
    with pytest.raises(ValueError, match="Unknown aggregation scheme"):
        game_stats.run(cfg, force=True)
