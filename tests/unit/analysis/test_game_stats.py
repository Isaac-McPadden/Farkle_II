import pandas as pd
import pytest

from tests.helpers.diagnostic_fixtures import build_curated_fixture

from farkle.analysis import game_stats


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
