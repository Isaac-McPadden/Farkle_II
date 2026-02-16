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
            {"strategy": "s1", "n_players": 2, "games": 15.0, "win_rate": (0.6 * 10 + 0.2 * 5) / 15, "w_k": 0.6},
            {"strategy": "s1", "n_players": 3, "games": 1.0, "win_rate": 0.5, "w_k": 0.4},
            {"strategy": "s2", "n_players": 2, "games": 1.0, "win_rate": 1.0, "w_k": 0.6},
            {
                "strategy": "s2",
                "n_players": 3,
                "games": 20.0,
                "win_rate": (0.5 * 10 + 0.7 * 10) / 20,
                "w_k": 0.4,
            },
        ]
    )
    expected_per_k["weighted"] = expected_per_k["win_rate"] * expected_per_k["w_k"]

    pd.testing.assert_frame_equal(
        per_k.reset_index(drop=True),
        expected_per_k,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    expected_collapsed = pd.Series(
        {
            "s2": expected_per_k.loc[expected_per_k["strategy"] == "s2", "weighted"].sum(),
            "s1": expected_per_k.loc[expected_per_k["strategy"] == "s1", "weighted"].sum(),
        },
        name="weighted",
    )
    expected_collapsed.index.name = "strategy"
    pd.testing.assert_series_equal(
        collapsed,
        expected_collapsed,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_weighted_winrate_grouping_defaults_for_missing_and_extra_weights():
    df = pd.DataFrame(
        [
            {"strategy": "s1", "n_players": 2, "games": 4, "win_rate": 0.5},
            {"strategy": "s1", "n_players": 3, "games": 6, "win_rate": 0.25},
            {"strategy": "s2", "n_players": 2, "games": 8, "win_rate": 0.75},
            {"strategy": "s2", "n_players": 3, "games": 2, "win_rate": 0.5},
        ]
    )

    # Includes an extra key (4) not present in data and omits key 3 used by the data.
    collapsed, per_k = tiering_report._weighted_winrate(df, {2: 1.0, 4: 0.5})

    # Missing group weights default to 0.0; extra keys are ignored by the map/join.
    assert (per_k.loc[per_k["n_players"] == 3, "w_k"] == 0.0).all()
    assert (per_k.loc[per_k["n_players"] == 2, "w_k"] == 1.0).all()

    expected = pd.Series({"s2": 0.75, "s1": 0.5}, name="weighted")
    expected.index.name = "strategy"
    pd.testing.assert_series_equal(
        collapsed,
        expected,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
