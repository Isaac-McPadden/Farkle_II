import json
from pathlib import Path

import pandas as pd
import pytest

import farkle.analysis.run_bonferroni_head2head as rb
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
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Tier file not found"):
        rb.run_bonferroni_head2head(seed=1, root=tmp_path)


def test_run_bonferroni_head2head_empty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "tiers.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="No tiers found"):
        rb.run_bonferroni_head2head(root=tmp_path)


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
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
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

    rb.run_bonferroni_head2head(seed=1, root=tmp_path, shard_size=1)

    pairwise_path = analysis_dir / "bonferroni_pairwise.parquet"
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
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "tiers.json").write_text(json.dumps({"A": 0, "B": 0}))

    monkeypatch.setattr(rb, "games_for_power", lambda **kwargs: 1)  # noqa: ANN001
    monkeypatch.setattr(rb, "parse_strategy", lambda name: name)
    monkeypatch.setattr(
        rb,
        "simulate_many_games_from_seeds",
        lambda seeds, strategies, n_jobs: pd.DataFrame({"winner_strategy": ["A"]}),
    )

    with pytest.raises(ValueError, match="progress_schedule must have three values"):
        rb.run_bonferroni_head2head(root=tmp_path, progress_schedule=[1, 2])


def test_run_bonferroni_limits_pair_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "tiers.json").write_text(json.dumps({"A": 0, "B": 0, "C": 0}))

    monkeypatch.setattr(rb, "games_for_power", lambda **kwargs: 1)  # noqa: ANN001
    monkeypatch.setattr(rb, "parse_strategy", lambda name: name)

    pair_jobs: list[int] = []

    def fake_simulate(seeds, strategies, n_jobs):  # noqa: ANN001,ARG001
        pair_jobs.append(n_jobs)
        return pd.DataFrame({"winner_strategy": [str(strategies[0])] * len(seeds)})

    monkeypatch.setattr(rb, "simulate_many_games_from_seeds", fake_simulate)

    rb.run_bonferroni_head2head(root=tmp_path, n_jobs=4)

    assert pair_jobs
    assert all(job == 1 for job in pair_jobs)
