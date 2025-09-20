import json

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


def test_run_bonferroni_head2head_writes_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path
    analysis_dir = root / "analysis"
    analysis_dir.mkdir()
    tiers_path = analysis_dir / "tiers.json"
    tiers_path.write_text(json.dumps({"A": 1, "B": 1}))
    monkeypatch.chdir(root)

    monkeypatch.setattr(
        rb,
        "games_for_power",
        lambda n, method="bonferroni", full_pairwise=True: 1,  # noqa: ARG005
    )
    monkeypatch.setattr(
        rb,
        "bonferroni_pairs",
        lambda elites, games_needed, seed: pd.DataFrame(  # noqa: ARG005
            {"a": ["A"], "b": ["B"], "seed": [seed]}
        ),  # noqa: ARG005
    )
    monkeypatch.setattr(rb, "parse_strategy", lambda s: s)

    def fake_many_games_from_seeds(*, seeds, strategies, n_jobs):
        _ = strategies
        assert n_jobs == 2
        return pd.DataFrame({"winner_strategy": ["A"] * len(seeds)})

    monkeypatch.setattr(rb, "simulate_many_games_from_seeds", fake_many_games_from_seeds)

    rb.run_bonferroni_head2head(seed=0, root=root, n_jobs=2)
    out_csv = analysis_dir / "bonferroni_pairwise.parquet"
    df = pd.read_parquet(out_csv)
    assert set(df.columns) == {"a", "b", "wins_a", "wins_b", "pvalue"}


def test_run_bonferroni_head2head_single_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Gracefully handle tiers.json with only one strategy."""

    root = tmp_path
    analysis_dir = root / "analysis"
    analysis_dir.mkdir()
    tiers_path = analysis_dir / "tiers.json"
    tiers_path.write_text(json.dumps({"A": 1}))
    monkeypatch.chdir(root)

    monkeypatch.setattr(rb, "games_for_power", lambda *a, **k: 0)  # noqa: ARG005
    monkeypatch.setattr(
        rb,
        "bonferroni_pairs",
        lambda *a, **k: pd.DataFrame(columns=["a", "b", "seed"]),  # noqa: ARG005
    )
    monkeypatch.setattr(rb, "parse_strategy", lambda s: s)
    monkeypatch.setattr(
        rb,
        "simulate_many_games_from_seeds",
        lambda **k: pd.DataFrame({"winner_strategy": []}),  # noqa: ARG005
    )

    with caplog.at_level("INFO"):
        rb.run_bonferroni_head2head(seed=0, root=root)

    assert "Bonferroni head-to-head: no games needed" in caplog.text

    out_csv = analysis_dir / "bonferroni_pairwise.parquet"
    assert not out_csv.exists()


def test_run_bonferroni_head2head_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An informative error is raised when tiers.json is absent."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="Tier file not found"):
        rb.run_bonferroni_head2head(seed=1, root=tmp_path)


def test_run_bonferroni_head2head_empty_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "tiers.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="No tiers found"):
        rb.run_bonferroni_head2head(root=tmp_path)
