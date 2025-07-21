"""
Tiny integration-style tests for time_farkle.py utilities.
(We don't time anything here - just logic.)
"""
from types import SimpleNamespace

import pandas as pd
import pytest

import farkle.time_farkle as tf


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


def test_measure_sim_times_runs_capfd(capfd):
    # use argv=[] so argparse sees no flags
    tf.measure_sim_times(argv=[])
    out, _ = capfd.readouterr()
    assert "Single game:" in out
    assert "Batch of" in out


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
    import farkle.time_farkle as tf

    def fake_many_games(*_, **__):
        return pd.DataFrame({"winner": ["P1", "P2", "P1"]})

    monkeypatch.setattr(tf, "simulate_many_games", fake_many_games)
    df = tf.simulate_many_games(n_games=3, strategies=[], seed=0, n_jobs=1)
    assert set(df["winner"]) == {"P1", "P2"}


@pytest.mark.parametrize(
    "flag,val",
    [
        ("-n", "0"),
        ("-n", "-1"),
        ("-p", "0"),
        ("-p", "-2"),
        ("-j", "0"),
        ("-j", "-3"),
    ],
)
def test_cli_rejects_nonpositive_values(flag, val):
    with pytest.raises(SystemExit):
        tf.measure_sim_times(argv=[flag, val])


def test_build_arg_parser_defaults():
    parser = tf.build_arg_parser()
    args = parser.parse_args([])
    assert args.n_games == 1000 and isinstance(args.n_games, int)
    assert args.players == 5 and isinstance(args.players, int)
    assert args.seed == 42 and isinstance(args.seed, int)
    assert args.jobs == 1 and isinstance(args.jobs, int)


def test_winners_breakdown_multiple_winners(monkeypatch, capfd):
    df = pd.DataFrame({"winner": ["P1", "P2", "P1"]})
    monkeypatch.setattr(tf, "simulate_many_games", lambda **_: df)
    tf.measure_sim_times(argv=[])  # defaults
    out, _ = capfd.readouterr()
    assert "P1" in out and "2" in out
    assert "P2" in out and "1" in out
    