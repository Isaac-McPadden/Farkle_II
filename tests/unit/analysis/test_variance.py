import pandas as pd

from farkle.analysis import variance


def test_compute_variance_and_components_minimums():
    seed_frame = pd.DataFrame(
        {
            "strategy_id": ["A", "A", "A", "B"],
            "players": [2, 2, 2, 2],
            "seed": [1, 2, 3, 1],
            "win_rate": [0.5, 0.7, 0.9, 0.25],
            "score_mean": [100, 110, 120, 80],
            "mean_farkles": [1.0, 1.5, 2.0, 0.5],
            "turns_mean": [8, 9, 10, 7],
        }
    )

    agg_variance = variance._compute_variance(seed_frame)
    row_a = agg_variance[agg_variance["strategy_id"] == "A"].iloc[0]
    assert row_a["n_seeds"] == 3
    assert row_a["mean_seed_win_rate"] == (0.5 + 0.7 + 0.9) / 3
    assert row_a["variance_win_rate"] == ((0.5 - row_a["mean_seed_win_rate"]) ** 2 + (0.7 - row_a["mean_seed_win_rate"]) ** 2 + (0.9 - row_a["mean_seed_win_rate"]) ** 2) / 2

    components = variance._compute_variance_components(seed_frame, min_seeds=2)
    mean_score = components[
        (components["strategy_id"] == "A") & (components["component"] == "total_score")
    ].iloc[0]
    assert mean_score["n_seeds"] == 3
    assert mean_score["mean"] == 110
    assert mean_score["variance"] == ((100 - 110) ** 2 + (110 - 110) ** 2 + (120 - 110) ** 2) / 2

    assert components[components["strategy_id"] == "B"].empty
