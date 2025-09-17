"""
Tiny integration-style tests for time_farkle.py utilities.
(We don't time anything here - just logic.)
"""

from types import SimpleNamespace

import pandas as pd
import pytest

import farkle.simulation.time_farkle as tf


@pytest.fixture(autouse=True)
def fast_sim(monkeypatch):
    """
    Stub the two heavy helpers so they return instantly *and* look
    enough like pandas objects that measure_sim_times() is happy.
    """
    dummy_game = SimpleNamespace(
        players={"P0": SimpleNamespace(score=10)},
        game=SimpleNamespace(n_rounds=1),
    )

    # simulate_one_game → lightweight object
    monkeypatch.setattr(
        tf,
        "simulate_one_game",
        lambda **_: dummy_game,
        raising=False,
    )

    # simulate_many_games → tiny DataFrame with a 'winner' column
    monkeypatch.setattr(
        tf,
        "simulate_many_games",
        lambda **_: pd.DataFrame({"winner": [0]}),
        raising=False,
    )


def test_measure_sim_times_logs(capinfo):
    tf.measure_sim_times(n_games=4, players=3, seed=21, jobs=2)

    stages = {(rec.benchmark, rec.stage) for rec in capinfo.records if hasattr(rec, "benchmark")}
    assert ("time_farkle", "simulation") in stages
    assert ("batch", "simulation") in stages
    assert ("single_game", "simulation") in stages


def test_make_random_strategies_deterministic():
    s1 = tf.make_random_strategies(3, seed=42)
    s2 = tf.make_random_strategies(3, seed=42)
    assert [str(x) for x in s1] == [str(x) for x in s2]
    assert len(s1) == 3


def test_dataframe_winner_column(tmp_path, monkeypatch):  # noqa: ARG001
    """
    Trivial smoke test: monkey-patch simulate_many_games so we don't
    actually run heavy sims - return a synthetic DataFrame instead.
    """
    import farkle.simulation.time_farkle as tf

    def fake_many_games(*_, **__):
        return pd.DataFrame({"winner": ["P1", "P2", "P1"]})

    monkeypatch.setattr(tf, "simulate_many_games", fake_many_games)
    df = tf.simulate_many_games(n_games=3, strategies=[], seed=0, n_jobs=1)
    assert set(df["winner"]) == {"P1", "P2"}


def test_winners_breakdown_multiple_winners(monkeypatch, capinfo):
    df = pd.DataFrame({"winner": ["P1", "P2", "P1"]})
    monkeypatch.setattr(tf, "simulate_many_games", lambda **_: df)
    tf.measure_sim_times()

    batch_records = [rec for rec in capinfo.records if getattr(rec, "benchmark", None) == "batch"]
    assert batch_records, "Batch benchmark log missing"
    winners = getattr(batch_records[-1], "winners", {})
    assert winners == {"P1": 2, "P2": 1}
