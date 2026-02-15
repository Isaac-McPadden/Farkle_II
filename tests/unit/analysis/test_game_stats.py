from pathlib import Path
import hashlib
import json

import pandas as pd
import pyarrow as pa
import pytest
from tests.helpers.diagnostic_fixtures import build_curated_fixture

from farkle.analysis import game_stats
from farkle.config import AppConfig, IOConfig, SimConfig


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
    assert aggro["observations"] == 5
    assert aggro["multi_reached_target"] == pytest.approx(0.6)
    assert aggro[f"margin_le_{thresholds[1]}"] == pytest.approx(1.0)


def test_summarize_rounds_handles_empty_and_values():
    empty = game_stats._summarize_rounds([])
    assert empty["observations"] == 0
    assert pd.isna(empty["mean_rounds"])

    stats = game_stats._summarize_rounds([1, 5, 9])
    assert stats["observations"] == 3
    assert stats["prob_rounds_le_5"] == pytest.approx(2 / 3)


def _build_parquet(tmp_path: Path, cfg):
    rows = pd.DataFrame(
        [
            {
                "seat_ranks": ["P1", "P2"],
                "n_rounds": 4,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 120,
                "P2_score": 110,
            },
            {
                "seat_ranks": ["P2", "P1"],
                "n_rounds": 8,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 50,
                "P2_score": 200,
            },
            {
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
    per_n_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(per_n_path)

    combined_path = cfg.curated_parquet
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(combined_path)

    return per_n_path, combined_path


def test_run_generates_all_outputs(tmp_path: Path):
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2]))
    per_n_path, combined_path = _build_parquet(tmp_path, cfg)

    game_stats.run(cfg, force=True)

    game_length = cfg.game_stats_output_path("game_length.parquet")
    margin_path = cfg.game_stats_output_path("margin_stats.parquet")
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

    per_k_game_length = cfg.per_k_subdir("game_stats", 2) / "game_length.parquet"
    per_k_margin = cfg.per_k_subdir("game_stats", 2) / "margin_stats.parquet"
    assert per_k_game_length.exists()
    assert per_k_margin.exists()

    per_k_game_df = pd.read_parquet(per_k_game_length)
    per_k_margin_df = pd.read_parquet(per_k_margin)
    assert set(per_k_game_df["n_players"].dropna().astype(int).unique()) <= {2}
    assert set(per_k_margin_df["n_players"].dropna().astype(int).unique()) <= {2}

    rare_df = pd.read_parquet(rare_events_path)
    assert {"game", "strategy", "n_players"} <= set(rare_df["summary_level"].unique())

    stamp = cfg.game_stats_stage_dir / "game_stats.done.json"
    assert stamp.exists()
    stamp_meta = json.loads(stamp.read_text())
    assert str(per_k_game_length) in stamp_meta["outputs"]
    assert str(per_k_margin) in stamp_meta["outputs"]



def test_run_requires_inputs(tmp_path: Path):
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))

    game_stats.run(cfg)

    assert not cfg.game_stats_output_path("game_length.parquet").exists()
    assert not cfg.game_stats_output_path("margin_stats.parquet").exists()
    assert not cfg.game_stats_output_path("rare_events.parquet").exists()


def test_compute_margins_and_aggregation(tmp_path: Path):
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
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


def test_global_stats_warns_when_seat_ranks_missing(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch):
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


def _write_multi_k_curated_inputs(cfg: AppConfig) -> None:
    rows_2p = pd.DataFrame(
        [
            {
                "seat_ranks": ["P1", "P2"],
                "n_rounds": 5,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 120,
                "P2_score": 100,
            },
            {
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
        per_n_path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(per_n_path)

    combined = pd.concat([rows_2p, rows_3p], ignore_index=True, sort=False)
    combined_path = cfg.curated_parquet
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(combined_path)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_writes_per_k_outputs_and_is_idempotent_for_multi_k(tmp_path: Path) -> None:
    k_values = [2, 3]
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path),
        sim=SimConfig(n_players_list=k_values, seed=123, seed_list=[123]),
    )
    _write_multi_k_curated_inputs(cfg)

    game_stats.run(cfg)

    pooled_targets = {
        "game_length.parquet": cfg.game_stats_output_path("game_length.parquet"),
        "margin_stats.parquet": cfg.game_stats_output_path("margin_stats.parquet"),
    }
    for output_path in pooled_targets.values():
        assert output_path.exists()

    expected_per_k_paths: list[Path] = []
    for k in k_values:
        for filename in pooled_targets:
            path = cfg.per_k_subdir("game_stats", k) / filename
            assert path.exists()
            expected_per_k_paths.append(path)

            per_k_df = pd.read_parquet(path)
            assert set(per_k_df["n_players"].dropna().astype(int).unique()) <= {k}

            pooled_df = pd.read_parquet(pooled_targets[filename])
            expected = pooled_df.loc[pooled_df["n_players"] == k].reset_index(drop=True)
            pd.testing.assert_frame_equal(per_k_df.reset_index(drop=True), expected)

    tracked_paths = list(pooled_targets.values()) + expected_per_k_paths
    before = {path: (path.stat().st_mtime_ns, _hash_file(path)) for path in tracked_paths}

    game_stats.run(cfg)

    after = {path: (path.stat().st_mtime_ns, _hash_file(path)) for path in tracked_paths}
    assert after == before
