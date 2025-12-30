from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farkle.analysis import variance
from farkle.config import AppConfig, IOConfig, SimConfig


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


def test_discover_seed_summaries_filters(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir=tmp_path, append_seed=False))
    cfg.analysis.outputs = {}
    valid = cfg.seed_summaries_dir(2) / "strategy_summary_2p_seed1.parquet"
    legacy = cfg.analysis_dir / "strategy_summary_2p_seed2.parquet"
    duplicate = cfg.seed_summaries_stage_dir / "strategy_summary_2p_seed1.parquet"
    invalid = cfg.seed_summaries_dir(2) / "not_a_summary.parquet"

    for path in (valid, legacy, duplicate, invalid):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder")

    discovered = variance._discover_seed_summaries(cfg)

    assert legacy in discovered
    assert valid in discovered
    assert len(discovered) >= 2


def test_load_metrics_and_seed_summaries_validation(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.parquet"
    pd.DataFrame(
        [
            {"strategy": "X", "n_players": 3, "win_rate": 0.5},
            {"strategy": "Y", "n_players": 3, "win_rate": np.nan},
        ]
    ).to_parquet(metrics_path, index=False)

    metrics = variance._load_metrics(metrics_path)
    assert metrics["strategy_id"].tolist() == ["X", "Y"]
    assert metrics["players"].tolist() == [3, 3]

    with pytest.raises(FileNotFoundError):
        variance._load_metrics(tmp_path / "missing.parquet")

    summary_path = tmp_path / "strategy_summary_3p_seed1.parquet"
    pd.DataFrame(
        [
            {"strategy_id": "X", "players": 3, "seed": 1, "win_rate": "0.5"},
            {"strategy_id": "X", "players": 3, "seed": 2, "score_mean": 101},
        ]
    ).to_parquet(summary_path, index=False)

    seed_frame = variance._load_seed_summaries([summary_path])
    assert set(seed_frame.columns) >= {"win_rate", "score_mean", "turns_mean"}
    assert seed_frame["players"].dtype.kind in {"i", "u"}

    empty = variance._load_seed_summaries([])
    assert empty.empty and list(empty.columns)


def test_merge_and_summarize_variance() -> None:
    metrics_frame = pd.DataFrame(
        {
            "strategy_id": ["A", "B"],
            "players": [2, 2],
            "win_rate": [0.6, np.nan],
        }
    )
    variance_frame = pd.DataFrame(
        {
            "strategy_id": ["A", "B"],
            "players": [2, 2],
            "n_seeds": [2, 2],
            "mean_seed_win_rate": [0.6, 0.4],
            "variance_win_rate": [0.01, 0.02],
            "std_win_rate": [0.1, 0.1414],
            "se_win_rate": [0.07, 0.1],
        }
    )

    merged = variance._merge_metrics(metrics_frame, variance_frame)
    assert merged.loc[merged["strategy_id"] == "A", "signal_to_noise"].iloc[0] > 0
    assert merged.loc[merged["strategy_id"] == "B", "win_rate_mean"].iloc[0] == pytest.approx(0.4)

    summary = variance._summarize_variance(merged)
    assert summary.loc[0, "n_players"] == 2
    assert summary.loc[0, "median_variance"] == pytest.approx(0.015)


def test_run_writes_outputs(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir=tmp_path / "results", append_seed=False),
        sim=SimConfig(seed=1, n_players_list=[2]),
    )
    cfg.analysis.outputs = {}

    metrics_path = cfg.metrics_input_path()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"strategy_id": "A", "players": 2, "win_rate": 0.55},
            {"strategy_id": "B", "players": 2, "win_rate": 0.45},
        ]
    ).to_parquet(metrics_path, index=False)

    for seed in (1, 2):
        seed_path = cfg.seed_summaries_dir(2) / f"strategy_summary_2p_seed{seed}.parquet"
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "strategy_id": "A",
                    "players": 2,
                    "seed": seed,
                    "win_rate": 0.5 + 0.05 * seed,
                    "score_mean": 100 + seed,
                    "turns_mean": 8 + seed,
                },
                {
                    "strategy_id": "B",
                    "players": 2,
                    "seed": seed,
                    "win_rate": 0.45 - 0.01 * seed,
                    "score_mean": 90 + seed,
                    "turns_mean": 7 + seed,
                },
            ]
        ).to_parquet(seed_path, index=False)

    variance.run(cfg)

    variance_path = cfg.variance_output_path(variance.VARIANCE_OUTPUT)
    summary_path = cfg.variance_output_path(variance.SUMMARY_OUTPUT)
    components_path = cfg.variance_output_path(variance.COMPONENTS_OUTPUT)

    assert variance_path.exists()
    assert summary_path.exists()
    assert components_path.exists()

    variance_frame = pd.read_parquet(variance_path)
    assert set(variance_frame["strategy_id"]) == {"A", "B"}
    assert variance_frame["n_seeds"].min() >= variance.MIN_SEEDS

    components = pd.read_parquet(components_path)
    assert set(components["component"]) >= {"win_rate", "total_score", "game_length"}
