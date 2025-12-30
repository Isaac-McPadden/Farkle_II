from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from tests.helpers.diagnostic_fixtures import build_curated_fixture

from farkle.analysis import game_stats
from farkle.config import AppConfig, IOConfig


def test_rare_event_flags_cover_game_and_strategy_levels(tmp_path):
    cfg, _, per_n = build_curated_fixture(tmp_path)
    thresholds = (10, 60)

    flags = game_stats._rare_event_flags([(2, per_n)], thresholds=thresholds, target_score=100)

    assert not flags.empty
    # Three game-level rows plus strategy-level and n-player summaries
    assert {"game", "strategy", "n_players"} <= set(flags["summary_level"].unique())

    aggro = flags[(flags["strategy"] == "Aggro") & (flags["summary_level"] == "strategy")].iloc[0]
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
                "P1_strategy": "Aggro",
                "P2_strategy": "Control",
                "P1_score": 120,
                "P2_score": 110,
            },
            {
                "seat_ranks": ["P2", "P1"],
                "n_rounds": 8,
                "P1_strategy": "Aggro",
                "P2_strategy": "Control",
                "P1_score": 50,
                "P2_score": 200,
            },
            {
                "seat_ranks": ["P1", "P2"],
                "n_rounds": 12,
                "P1_strategy": "Aggro",
                "P2_strategy": "Control",
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
    cfg = AppConfig(io=IOConfig(results_dir=tmp_path, append_seed=False))
    per_n_path, combined_path = _build_parquet(tmp_path, cfg)

    game_stats.run(cfg, force=True)

    out_dir = cfg.game_stats_stage_dir
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
    assert all(col in margin_df.columns for col in ("mean_margin", "median_margin"))

    rare_df = pd.read_parquet(rare_events_path)
    assert {"game", "strategy", "n_players"} <= set(rare_df["summary_level"].unique())

    stamp = cfg.game_stats_stage_dir / "game_stats.done.json"
    assert stamp.exists()


def test_run_requires_inputs(tmp_path: Path):
    cfg = AppConfig(io=IOConfig(results_dir=tmp_path, append_seed=False))

    with pytest.raises(FileNotFoundError):
        game_stats.run(cfg)


def test_compute_margins_and_aggregation(tmp_path: Path):
    cfg = AppConfig(io=IOConfig(results_dir=tmp_path, append_seed=False))
    per_n_path, _ = _build_parquet(tmp_path, cfg)

    per_n_inputs = [(2, per_n_path)]
    margins = game_stats._per_strategy_margin_stats(per_n_inputs, thresholds=(100,))
    assert not margins.empty
    assert margins.loc[0, "prob_margin_le_100"] == pytest.approx(1 / 3)

    rare = game_stats._rare_event_flags(per_n_inputs, thresholds=(100,), target_score=150)
    assert not rare.empty
    assert set(rare["summary_level"].unique()) >= {"game", "strategy"}


def test_global_stats_warns_when_seat_ranks_missing(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch):
    class DummyDataset:
        schema = type("Schema", (), {"names": ["n_rounds"]})()

        @staticmethod
        def to_table(columns=None):  # noqa: ARG002
            return pa.Table.from_pandas(pd.DataFrame({"n_rounds": [1, 2, 3]}))

    monkeypatch.setattr(game_stats.ds, "dataset", lambda path: DummyDataset())

    with caplog.at_level("WARNING"):
        result = game_stats._global_stats(Path("dummy"))

    assert result.empty
    assert "Combined parquet missing seat_ranks" in caplog.text
