from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farkle.analysis import variance
from farkle.config import AppConfig, IOConfig, SimConfig
from farkle.utils.analysis_shared import as_float, is_na


@pytest.fixture
def constant_seed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_id": ["CONST", "CONST", "CONST"],
            "players": [2, 2, 2],
            "seed": [11, 12, 13],
            "win_rate": [0.5, 0.5, 0.5],
            "score_mean": [100.0, 100.0, 100.0],
            "turns_mean": [8.0, 8.0, 8.0],
            "mean_farkles": [1.0, 1.0, 1.0],
        }
    )


@pytest.fixture
def high_variance_seed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_id": ["SPIKY"] * 6,
            "players": [4] * 6,
            "seed": [1, 2, 3, 4, 5, 6],
            "win_rate": [0.0, 1.0, 0.05, 0.95, 0.1, 0.9],
            "score_mean": [50.0, 200.0, 60.0, 190.0, 55.0, 195.0],
            "turns_mean": [2.0, 30.0, 3.0, 28.0, 4.0, 29.0],
        }
    )


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


def test_constant_outcomes_numeric_stability(constant_seed_frame: pd.DataFrame) -> None:
    detailed = variance._compute_variance(constant_seed_frame)
    assert len(detailed) == 1
    row = detailed.iloc[0]
    assert row["variance_win_rate"] == pytest.approx(0.0, abs=1e-15)
    assert row["std_win_rate"] == pytest.approx(0.0, abs=1e-15)
    assert row["se_win_rate"] == pytest.approx(0.0, abs=1e-15)

    components = variance._compute_variance_components(constant_seed_frame)
    win_component = components[components["component"] == "win_rate"].iloc[0]
    assert win_component["variance"] == pytest.approx(0.0, abs=1e-15)
    assert win_component["ci_lower"] == pytest.approx(0.5, abs=1e-12)
    assert win_component["ci_upper"] == pytest.approx(0.5, abs=1e-12)


def test_tiny_samples_and_nan_inf_guards() -> None:
    seed_frame = pd.DataFrame(
        {
            "strategy_id": ["ONE", "ONE", "TWO", "TWO"],
            "players": [2, 2, 2, 2],
            "seed": [1, 1, 1, 2],
            "win_rate": [0.6, np.nan, np.inf, 0.4],
            "score_mean": [100.0, np.nan, np.inf, 95.0],
            "turns_mean": [9.0, np.nan, np.inf, 8.0],
        }
    )

    with pytest.warns(RuntimeWarning, match="invalid value encountered in subtract"):
        detailed = variance._compute_variance(seed_frame)
    one = detailed[detailed["strategy_id"] == "ONE"].iloc[0]
    assert one["n_seeds"] == 1
    assert one["variance_win_rate"] == pytest.approx(0.0, abs=1e-15)
    assert one["se_win_rate"] == pytest.approx(0.0, abs=1e-15)

    with pytest.warns(RuntimeWarning, match="invalid value encountered in subtract"):
        components = variance._compute_variance_components(seed_frame, min_seeds=2)
    assert set(components["strategy_id"]) == {"TWO"}
    assert components["variance"].isna().all()
    assert components["ci_lower"].isna().all()
    assert components["ci_upper"].isna().all()

    merged = variance._merge_metrics(
        pd.DataFrame(
            {
                "strategy_id": ["ONE", "TWO"],
                "players": [2, 2],
                "win_rate": [np.nan, np.inf],
            }
        ),
        detailed,
    )
    one_merged = merged[merged["strategy_id"] == "ONE"].iloc[0]
    assert is_na(one_merged["signal_to_noise"])
    two_merged = merged[merged["strategy_id"] == "TWO"].iloc[0]
    assert is_na(two_merged["signal_to_noise"])


def test_high_variance_components_confidence_interval_and_signal(
    high_variance_seed_frame: pd.DataFrame,
) -> None:
    detailed = variance._compute_variance(high_variance_seed_frame)
    row = detailed.iloc[0]
    assert row["variance_win_rate"] > 0.2
    assert row["std_win_rate"] == pytest.approx(np.sqrt(as_float(row["variance_win_rate"])), rel=1e-12)
    assert row["se_win_rate"] > 0

    components = variance._compute_variance_components(high_variance_seed_frame)
    win_component = components[components["component"] == "win_rate"].iloc[0]
    assert win_component["ci_upper"] > win_component["ci_lower"]
    assert win_component["ci_lower"] == pytest.approx(
        win_component["mean"] - 1.96 * win_component["se_mean"], rel=1e-12
    )
    assert win_component["ci_upper"] == pytest.approx(
        win_component["mean"] + 1.96 * win_component["se_mean"], rel=1e-12
    )

    merged = variance._merge_metrics(
        pd.DataFrame({"strategy_id": ["SPIKY"], "players": [4], "win_rate": [0.5]}),
        detailed,
    )
    assert merged["signal_to_noise"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_discover_seed_summaries_filters(tmp_path: Path) -> None:
    cfg = AppConfig(IOConfig(results_dir_prefix=tmp_path))
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
        IOConfig(results_dir_prefix=tmp_path / "results"),
        SimConfig(seed=1, n_players_list=[2]),
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
