import json
from pathlib import Path

import pandas as pd
import pytest

import farkle.analysis.run_bonferroni_head2head as rb
from farkle.analysis.stage_registry import StageDefinition, StageLayout, StagePlacement
from farkle.config import AppConfig
from farkle.simulation.simulation import simulate_many_games_from_seeds
from farkle.simulation.strategies import ThresholdStrategy


def test_simulate_many_games_from_seeds(monkeypatch):
    def fake_play(seed, strategies, target_score=10_000):  # noqa: ARG001
        return {
            "winner": f"P{seed}",
            "winning_score": seed,
            "n_rounds": seed,
            "P1_strategy": str(strategies[0]),
        }

    import farkle.simulation.simulation as sim

    monkeypatch.setattr(sim, "_play_game", fake_play, raising=True)
    seeds1 = [1, 2, 3]
    seeds2 = [4, 5, 6]
    strat = ThresholdStrategy(300, 1, True, True)
    df1 = simulate_many_games_from_seeds(seeds=seeds1, strategies=[strat], n_jobs=1)
    df2 = simulate_many_games_from_seeds(seeds=seeds2, strategies=[strat], n_jobs=1)

    assert len(df1) == len(seeds1)
    assert len(df2) == len(seeds2)
    assert not df1.equals(df2)


def test_run_bonferroni_head2head_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An informative error is raised when tiers.json is absent."""
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Tier file not found"):
        rb.run_bonferroni_head2head(seed=1, cfg=cfg)


def test_run_bonferroni_head2head_empty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    analysis_dir = cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "tiers.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="No tiers found"):
        rb.run_bonferroni_head2head(cfg=cfg)


def test_count_pair_wins_prefers_strategy_column() -> None:
    df = pd.DataFrame({"winner_strategy": ["A", "B", "A", "A"]})
    wins = rb._count_pair_wins(df, "A", "B")
    assert wins == (3, 1)


def test_count_pair_wins_falls_back_to_seat_column() -> None:
    df = pd.DataFrame({"winner": ["P1", "P2", "P2", "P1", "P1"]})
    wins = rb._count_pair_wins(df, "A", "B")
    assert wins == (3, 2)


def test_run_bonferroni_head2head_resumes_and_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    analysis_dir = cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "tiers.json").write_text(json.dumps({"S1": 0, "S2": 0, "S3": 0}))

    existing = pd.DataFrame(
        [
            {
                "players": 2,
                "seed": 0,
                "pair_id": 0,
                "a": "S1",
                "b": "S2",
                "games": 1,
                "wins_a": 1,
                "wins_b": 0,
                "win_rate_a": 1.0,
                "pval_one_sided": 0.5,
            }
        ]
    )
    shard_dir = analysis_dir / "bonferroni_pairwise_shards"
    shard_dir.mkdir()
    existing_path = shard_dir / "bonferroni_pairwise_shard_0000.parquet"
    existing.to_parquet(existing_path)

    call_counter = {"calls": 0}

    def fake_games_for_power(**kwargs):  # noqa: ANN001
        return 1

    def fake_simulate(seeds, strategies, n_jobs):  # noqa: ANN001,ARG001
        call_counter["calls"] += 1
        return pd.DataFrame({"winner_strategy": [str(strategies[0])] * len(seeds)})

    monkeypatch.setattr(rb, "games_for_power", fake_games_for_power)
    monkeypatch.setattr(rb, "parse_strategy", lambda name: name)
    monkeypatch.setattr(rb, "simulate_many_games_from_seeds", fake_simulate)

    rb.run_bonferroni_head2head(seed=1, cfg=cfg, shard_size=1)

    pairwise_path = cfg.head2head_path("bonferroni_pairwise.parquet")
    assert pairwise_path.exists()
    df = pd.read_parquet(pairwise_path)
    assert set(df["pair_id"]) == {0, 1, 2}

    shards = sorted(shard_dir.glob("*.parquet"))
    assert len(shards) >= 2
    # Only two pairs should be simulated because one was already completed
    assert call_counter["calls"] == 2


def test_run_bonferroni_head2head_progress_schedule_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    analysis_dir = cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "tiers.json").write_text(json.dumps({"A": 0, "B": 0}))

    monkeypatch.setattr(rb, "games_for_power", lambda **kwargs: 1)  # noqa: ANN001
    monkeypatch.setattr(rb, "parse_strategy", lambda name: name)
    monkeypatch.setattr(
        rb,
        "simulate_many_games_from_seeds",
        lambda seeds, strategies, n_jobs: pd.DataFrame({"winner_strategy": ["A"]}),
    )

    with pytest.raises(ValueError, match="progress_schedule must have three values"):
        rb.run_bonferroni_head2head(root=cfg.results_root, progress_schedule=[1, 2])


def test_run_bonferroni_limits_pair_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    analysis_dir = cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "tiers.json").write_text(json.dumps({"A": 0, "B": 0, "C": 0}))

    monkeypatch.setattr(rb, "games_for_power", lambda **kwargs: 1)  # noqa: ANN001
    monkeypatch.setattr(rb, "parse_strategy", lambda name: name)

    pair_jobs: list[int] = []

    def fake_simulate(seeds, strategies, n_jobs):  # noqa: ANN001,ARG001
        pair_jobs.append(n_jobs)
        return pd.DataFrame({"winner_strategy": [str(strategies[0])] * len(seeds)})

    monkeypatch.setattr(rb, "simulate_many_games_from_seeds", fake_simulate)

    rb.run_bonferroni_head2head(root=cfg.results_root, n_jobs=4)

    assert pair_jobs
    assert all(job == 2 for job in pair_jobs)


def test_load_top_strategies_handles_missing_and_invalid(tmp_path: Path, caplog):
    ratings = tmp_path / "ratings.parquet"
    metrics = tmp_path / "metrics.parquet"

    caplog.set_level("INFO")
    strategies = rb._load_top_strategies(ratings_path=ratings, metrics_path=metrics)

    assert strategies == []
    assert any("Fallback selection skipped" in rec.message for rec in caplog.records)

    ratings_df = pd.DataFrame({"strategy": ["S1", "S2"], "mu": [1.0, 2.0]})
    metrics_df = pd.DataFrame({"strategy": ["S2", "S3"], "win_rate": [0.2, 0.8]})
    ratings_df.to_parquet(ratings)
    metrics_df.to_parquet(metrics)

    combined = rb._load_top_strategies(ratings_path=ratings, metrics_path=metrics)
    assert combined == ["S2", "S1", "S3"]


def test_tiers_path_prefers_stage_layout_and_warns(tmp_path: Path, caplog):
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    cfg.set_stage_layout(
        StageLayout(
            placements=[
                StagePlacement(
                    definition=StageDefinition(key="head2head", group="analytics"),
                    index=0,
                    folder_name="11_head2head",
                ),
                StagePlacement(
                    definition=StageDefinition(key="tiering", group="analytics"),
                    index=1,
                    folder_name="11_head2head",
                ),
            ]
        )
    )
    stage_dir = cfg.analysis_dir / "11_head2head"
    stage_dir.mkdir(parents=True)
    legacy_tiers = cfg.analysis_dir / "tiers.json"
    legacy_tiers.write_text("{}")
    preferred = stage_dir / "tiers.json"
    preferred.write_text("{}")

    caplog.set_level("WARNING")
    path = rb._tiers_path(cfg)

    assert path == preferred


def test_count_pair_wins_errors_without_winner(monkeypatch):
    df = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(KeyError):
        rb._count_pair_wins(df, "A", "B")
