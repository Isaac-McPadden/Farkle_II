import pandas as pd

from farkle.analysis import tiering_report


def test_weighted_winrate_with_grouping_weights():
    df = pd.DataFrame(
        [
            {"strategy": "s1", "n_players": 2, "games": 10, "wins": 6, "win_rate": 0.6},
            {"strategy": "s1", "n_players": 2, "games": 5, "wins": 1, "win_rate": 0.2},
            {"strategy": "s1", "n_players": 3, "games": 0, "wins": 0, "win_rate": 0.5},
            {"strategy": "s2", "n_players": 2, "games": 1, "wins": 1, "win_rate": 1.0},
            {"strategy": "s2", "n_players": 3, "games": 10, "wins": 5, "win_rate": 0.5},
            {"strategy": "s2", "n_players": 3, "games": 10, "wins": 7, "win_rate": 0.7},
        ]
    )

    collapsed, per_k = tiering_report._weighted_winrate(df, {2: 0.6, 3: 0.4})

    expected_per_k = pd.DataFrame(
        [
            {"strategy": "s1", "n_players": 2, "win_rate": (6 + 1) / 15},
            {"strategy": "s1", "n_players": 3, "win_rate": 0.5},
            {"strategy": "s2", "n_players": 2, "win_rate": 1.0},
            {"strategy": "s2", "n_players": 3, "win_rate": (0.5 * 10 + 0.7 * 10) / 20},
        ]
    )
    expected_per_k["w"] = expected_per_k["n_players"].map({2: 0.6, 3: 0.4})
    expected_per_k["weighted"] = expected_per_k["win_rate"] * expected_per_k["w"]

    pd.testing.assert_frame_equal(per_k.reset_index(drop=True), expected_per_k)

    expected_collapsed = pd.Series(
        {
            "s2": expected_per_k.loc[expected_per_k["strategy"] == "s2", "weighted"].sum(),
            "s1": expected_per_k.loc[expected_per_k["strategy"] == "s1", "weighted"].sum(),
        },
        name="weighted",
    )
    expected_collapsed.index.name = "strategy"
    pd.testing.assert_series_equal(collapsed, expected_collapsed)
