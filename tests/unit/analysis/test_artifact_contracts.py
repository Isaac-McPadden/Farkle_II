from __future__ import annotations

import pandas as pd
from tests.helpers import metrics_samples as sample_data
from tests.helpers.config_factory import make_test_app_config

from farkle.analysis import (
    agreement,
    coverage_by_k,
    meta,
    metrics,
    reporting,
    run_hgb,
    run_trueskill,
    seed_summaries,
    tiering_report,
    variance,
)


def _write_metrics_parquet(cfg, frame: pd.DataFrame) -> None:
    path = cfg.metrics_output_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_metrics_artifact_matches_downstream_reader_contracts(tmp_path) -> None:
    cfg = sample_data.stage_sample_run(tmp_path, refresh_inputs=False)

    metrics.run(cfg)

    metrics_path = cfg.metrics_output_path()
    loaded_metrics, loaded_path = seed_summaries._load_metrics_frame(cfg)
    coverage = coverage_by_k._stream_metrics_counts(metrics_path, default_seed=cfg.sim.seed)
    variance_metrics = variance._load_metrics(metrics_path)

    assert loaded_path == metrics_path
    assert {"strategy_id", "players", "seed", "games", "wins"} <= set(loaded_metrics.columns)
    assert set(loaded_metrics["players"]) == {2, 3}
    assert set(loaded_metrics["seed"]) == {cfg.sim.seed}

    assert coverage.columns.tolist() == [
        "seed",
        "k",
        "games",
        "strategies",
        "missing_before_pad",
    ]
    assert set(coverage["seed"]) == {cfg.sim.seed}
    assert set(coverage["k"]) == {2, 3}

    assert set(variance_metrics.columns) == {"strategy_id", "players", "win_rate"}
    assert set(variance_metrics["players"]) == {2, 3}
    assert len(variance_metrics) == len(loaded_metrics)


def test_seed_summary_artifacts_match_meta_variance_and_reporting_readers(tmp_path) -> None:
    cfg = make_test_app_config(results_dir_prefix=tmp_path / "results")
    metrics_frame = pd.DataFrame(
        [
            {
                "strategy": 1,
                "n_players": 2,
                "games": 10,
                "wins": 6,
                "seed": 101,
                "mean_n_rounds": 15.0,
            },
            {
                "strategy": 2,
                "n_players": 2,
                "games": 10,
                "wins": 4,
                "seed": 101,
                "mean_n_rounds": 18.0,
            },
            {
                "strategy": 1,
                "n_players": 3,
                "games": 9,
                "wins": 3,
                "seed": 101,
            },
            {
                "strategy": 2,
                "n_players": 3,
                "games": 9,
                "wins": 6,
                "seed": 101,
            },
            {
                "strategy": 1,
                "n_players": 2,
                "games": 8,
                "wins": 5,
                "seed": 202,
                "mean_n_rounds": 16.0,
            },
            {
                "strategy": 2,
                "n_players": 2,
                "games": 8,
                "wins": 3,
                "seed": 202,
                "mean_n_rounds": 19.0,
            },
            {
                "strategy": 1,
                "n_players": 3,
                "games": 7,
                "wins": 2,
                "seed": 202,
            },
            {
                "strategy": 2,
                "n_players": 3,
                "games": 7,
                "wins": 5,
                "seed": 202,
            },
        ]
    )
    _write_metrics_parquet(cfg, metrics_frame)

    seed_summaries.run(cfg)

    summary_paths = [
        cfg.seed_summaries_dir(2) / "strategy_summary_2p_seed101.parquet",
        cfg.seed_summaries_dir(2) / "strategy_summary_2p_seed202.parquet",
    ]
    variance_seed_frame = variance._load_seed_summaries(summary_paths)
    reported_seed_frame = reporting._load_seed_summaries(cfg, 2)

    assert all(path.exists() for path in summary_paths)

    assert {"strategy_id", "players", "seed", "win_rate"} <= set(variance_seed_frame.columns)
    assert set(variance_seed_frame["seed"]) == {101, 202}
    assert set(variance_seed_frame["strategy_id"]) == {"1", "2"}

    assert set(reported_seed_frame.columns) == {"strategy_id", "seed", "win_rate", "ci_lo", "ci_hi"}
    assert set(reported_seed_frame["seed"]) == {101, 202}
    assert set(reported_seed_frame["strategy_id"]) == {"1", "2"}

    meta.run(cfg)

    meta_path = cfg.meta_output_path(2, "strategy_summary_2p_meta.parquet")
    pooled = pd.read_parquet(meta_path)

    assert meta_path.exists()
    assert set(pooled.columns) == {
        "strategy_id",
        "players",
        "win_rate",
        "se",
        "ci_lo",
        "ci_hi",
        "n_seeds",
    }
    assert set(pooled["strategy_id"]) == {"1", "2"}
    assert set(pooled["players"]) == {2}
    assert set(pooled["n_seeds"]) == {2}


def test_trueskill_artifacts_match_agreement_and_hgb_readers(tmp_path) -> None:
    cfg = make_test_app_config(results_dir_prefix=tmp_path / "results")
    per_k_dir = cfg.trueskill_stage_dir / "2p"
    pooled_dir = cfg.trueskill_pooled_dir
    per_k_dir.mkdir(parents=True, exist_ok=True)
    pooled_dir.mkdir(parents=True, exist_ok=True)

    run_trueskill._save_ratings_parquet(
        per_k_dir / "ratings_2.parquet",
        {"1": (30.0, 3.0), "2": (27.5, 3.5)},
    )
    run_trueskill._save_ratings_parquet(
        per_k_dir / "ratings_2_seed101.parquet",
        {"1": (31.0, 3.0), "2": (27.0, 3.5)},
    )
    run_trueskill._save_ratings_parquet(
        per_k_dir / "ratings_2_seed202.parquet",
        {"1": (29.0, 3.1), "2": (28.0, 3.4)},
    )
    run_trueskill._save_ratings_parquet(
        pooled_dir / "ratings_k_weighted_seed101.parquet",
        {"1": (31.0, 3.0), "2": (27.0, 3.5)},
    )
    run_trueskill._save_ratings_parquet(
        pooled_dir / "ratings_k_weighted_seed202.parquet",
        {"1": (29.0, 3.1), "2": (28.0, 3.4)},
    )

    trueskill_data = agreement._load_trueskill(cfg, 2, pooled_scope=False)
    seed_targets = run_hgb._load_seed_targets(pooled_dir)

    assert trueskill_data is not None
    assert trueskill_data.scores.index.tolist() == ["1", "2"]
    assert trueskill_data.scores.tolist() == [30.0, 27.5]
    assert len(trueskill_data.per_seed_scores) == 2
    assert all(series.index.tolist() == ["1", "2"] for series in trueskill_data.per_seed_scores)

    assert seed_targets.columns.tolist() == ["strategy", "mu", "seed"]
    assert {str(value) for value in seed_targets["strategy"]} == {"1", "2"}
    assert set(seed_targets["seed"]) == {101, 202}
    assert len(seed_targets) == 4


def test_frequentist_scores_artifact_matches_agreement_reader(tmp_path) -> None:
    cfg = make_test_app_config(results_dir_prefix=tmp_path / "results")
    frequentist_tiers = pd.DataFrame(
        [
            {"strategy": 1, "win_rate": 0.62, "mdd_tier": 1},
            {"strategy": 2, "win_rate": 0.38, "mdd_tier": 2},
        ]
    )
    pooled_winrates = pd.Series({1: 0.62, 2: 0.38}, name="weighted")
    winrates_by_players = pd.DataFrame(
        [
            {"strategy": 1, "n_players": 2, "games": 10.0, "win_rate": 0.60},
            {"strategy": 2, "n_players": 2, "games": 10.0, "win_rate": 0.40},
            {"strategy": 1, "n_players": 3, "games": 12.0, "win_rate": 0.64},
            {"strategy": 2, "n_players": 3, "games": 12.0, "win_rate": 0.36},
        ]
    )

    tiering_report._write_frequentist_scores(
        cfg,
        frequentist_tiers,
        pooled_winrates,
        winrates_by_players,
        weights_by_k={2: 0.5, 3: 0.5},
    )

    players_data = agreement._load_frequentist(cfg, 2)
    pooled_data = agreement._load_frequentist(cfg, 0)

    assert players_data is not None
    assert players_data.scores.index.tolist() == ["1", "2"]
    assert players_data.scores.tolist() == [0.60, 0.40]
    assert players_data.tiers == {"1": 1, "2": 2}
    assert players_data.per_seed_scores == []

    assert pooled_data is not None
    assert pooled_data.scores.index.tolist() == ["1", "2"]
    assert pooled_data.scores.tolist() == [0.62, 0.38]
    assert pooled_data.tiers == {"1": 1, "2": 2}
