import json

import pandas as pd
import pytest

import farkle.run_bonferroni_head2head as rb
from farkle.simulation import simulate_many_games_from_seeds
from farkle.strategies import ThresholdStrategy


def test_simulate_many_games_from_seeds(monkeypatch):
    def fake_play(seed, strategies, target_score=10_000):  # noqa: ARG001
        return {
            "winner": f"P{seed}",
            "winning_score": seed,
            "n_rounds": seed,
            "P1_strategy": str(strategies[0]),
        }

    import farkle.simulation as sim

    monkeypatch.setattr(sim, "_play_game", fake_play, raising=True)
    seeds1 = [1, 2, 3]
    seeds2 = [4, 5, 6]
    strat = ThresholdStrategy(300, 1, True, True)
    df1 = simulate_many_games_from_seeds(seeds=seeds1, strategies=[strat], n_jobs=1)
    df2 = simulate_many_games_from_seeds(seeds=seeds2, strategies=[strat], n_jobs=1)

    assert len(df1) == len(seeds1)
    assert len(df2) == len(seeds2)
    assert not df1.equals(df2)


def test_run_bonferroni_head2head_writes_csv(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tiers_path = data_dir / "tiers.json"
    tiers_path.write_text(json.dumps({"A": 1, "B": 1}))
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        rb,
        "games_for_power",
        lambda n, method="bonferroni", full_pairwise=True: 1,  # noqa: ARG005
    )
    monkeypatch.setattr(
        rb,
        "bonferroni_pairs",
        lambda elites, games_needed, seed: pd.DataFrame({"a": ["A"], "b": ["B"], "seed": [seed]}),  # noqa: ARG005
    )
    monkeypatch.setattr(rb, "parse_strategy", lambda s: s)

    def fake_many_games_from_seeds(*, seeds, strategies, n_jobs):
        _ = strategies
        _ = n_jobs
        return pd.DataFrame({"winner_strategy": ["A"] * len(seeds)})

    monkeypatch.setattr(rb, "simulate_many_games_from_seeds", fake_many_games_from_seeds)

    rb.run_bonferroni_head2head(seed=0, root=data_dir)
    out_csv = data_dir / "bonferroni_pairwise.csv"
    df = pd.read_csv(out_csv)
    assert set(df.columns) == {"a", "b", "wins_a", "wins_b", "pvalue"}


def test_run_bonferroni_head2head_single_strategy(tmp_path, monkeypatch):
    """Gracefully handle tiers.json with only one strategy."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tiers_path = data_dir / "tiers.json"
    tiers_path.write_text(json.dumps({"A": 1}))
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(rb, "games_for_power", lambda *a, **k: 0)  # noqa: ARG005
    monkeypatch.setattr(
        rb, "bonferroni_pairs", lambda *a, **k: pd.DataFrame(columns=["a", "b", "seed"])  # noqa: ARG005
    )
    monkeypatch.setattr(rb, "parse_strategy", lambda s: s)
    monkeypatch.setattr(
        rb,
        "simulate_many_games_from_seeds",
        lambda **k: pd.DataFrame({"winner_strategy": []}),  # noqa: ARG005
    )

    rb.run_bonferroni_head2head(seed=0, root=data_dir)
    out_csv = data_dir / "bonferroni_pairwise.csv"
    assert out_csv.exists()
    assert out_csv.read_text() == "\n"


def test_run_bonferroni_head2head_missing_file(tmp_path, monkeypatch):
    """An informative error is raised when tiers.json is absent."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Tier file not found"):
        rb.run_bonferroni_head2head(seed=1, root=tmp_path / "data")


def test_run_bonferroni_head2head_empty_file(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "tiers.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="No tiers found"):
        rb.run_bonferroni_head2head(root=data_dir)


def test_main_delegates_to_runner(monkeypatch):
    captured = {}

    def fake_run(seed: int = 0, root=None) -> None:  # noqa: ANN001
        _ = root
        captured["seed"] = seed

    monkeypatch.setattr(rb, "run_bonferroni_head2head", fake_run)
    rb.main(["--seed", "42", "--root", "d"])
    assert captured["seed"] == 42
