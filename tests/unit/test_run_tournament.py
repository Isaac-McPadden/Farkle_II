"""
Fast, side-effect-free sanity checks for run_tournament.py
(all run in < 1 s).

We monkey-patch the heavy helpers so no real games are played.
"""

from __future__ import annotations

import types  # noqa: F401
from collections import Counter

import numpy as np  # noqa: F401 | Potentially imports something that needs it
import pytest

import farkle.run_tournament as rt
from farkle.strategies import ThresholdStrategy

# --------------------------------------------------------------------------- #
# Mini test doubles – replace expensive pieces with cheap determinism
# --------------------------------------------------------------------------- #


def _mini_strats(n: int = 6):
    """Return deterministic Strategy objects with distinct __str__()."""
    return [ThresholdStrategy(50 + 50 * i, i % 3, True, True) for i in range(n)]


@pytest.fixture(autouse=True)
def fast_helpers(monkeypatch):
    """
    1) Replace generate_strategy_grid() with 12 toy strategies.
    2) Make _play_shuffle() return a trivial deterministic Counter.
    3) Skip the ProcessPool - we call _run_chunk() directly.
    """
    strats = _mini_strats(12)
    monkeypatch.setattr(rt, "generate_strategy_grid", lambda *a, **kw: (strats, None), raising=True)  # noqa: ARG005

    def fake_play_shuffle(seed: int) -> Counter[str]:
        # pretend player at index (seed % len(strats)) always wins
        return Counter({str(strats[seed % len(strats)]): 1})

    monkeypatch.setattr(rt, "_play_shuffle", fake_play_shuffle, raising=True)


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #
def test_run_chunk_counts_games():
    seeds = [1, 2, 3, 4]
    wins = rt._run_chunk(seeds)
    assert sum(wins.values()) == len(seeds)


def test_checkpoint_timer(monkeypatch, tmp_path):
    """
    Drive run_tournament() with a fake time source so we hit exactly
    one checkpoint save and one logging call.
    """

    # ── 1 · deterministic, instantly-returning ProcessPool substitute ─────────

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):  # mimic concurrent.futures.Future
            return self._result

        def __hash__(self):  # make it usable as a dict key
            return id(self)

        def __eq__(self, other):
            return self is other

    class DummyPool:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, arg):
            # run the work *eagerly* and wrap the result
            return DummyFuture(fn(arg))

    monkeypatch.setattr(rt, "ProcessPoolExecutor", lambda *a, **k: DummyPool())  # noqa: ARG005

    # ── 2 · small config so only TWO chunks total ────────────────────────────
    cfg = rt.TournamentConfig(num_shuffles=2, ckpt_every_sec=30)

    # ── 3 · fake wall-clock (t jumps +31 s before the 2nd chunk) ─────────────
    t = 0.0

    def fake_perf():  # our replacement for time.perf_counter()
        return t

    monkeypatch.setattr(rt.time, "perf_counter", fake_perf, raising=True)

    def fake_as_completed(futures):  # yield first future, then advance time
        nonlocal t
        it = iter(futures)
        fut1 = next(it)
        yield fut1  # after first chunk (t = 0)
        t += 31  # 31 s later → timer should fire
        for fut in it:
            yield fut

    monkeypatch.setattr(rt, "as_completed", fake_as_completed, raising=True)

    # ── 3 · skip the real throughput probe (no timing, no games) ──────────────
    monkeypatch.setattr(
        rt,
        "_measure_throughput",
        lambda *a, **k: 1,  # small → 1 shuffle / chunk  # noqa: ARG005
        raising=True,
    )

    # ── 4 · count how many times we attempt to write a checkpoint ─────────────
    saves = {"n": 0}

    def fake_write_bytes(self, data):  # noqa: ARG001
        saves["n"] += 1

    monkeypatch.setattr(rt.Path, "write_bytes", fake_write_bytes, raising=True)

    # also silence logging noise
    monkeypatch.setattr(rt.logging, "info", lambda *a, **k: None, raising=False)  # noqa: ARG005

    # ── 6 · run – we expect 1 mid-run save + 1 final save ─────────────────────
    rt.run_tournament(
        config=cfg,
        global_seed=0,
        checkpoint_path=tmp_path / "chk.pkl",
        n_jobs=None,
    )

    assert saves["n"] == 2  # exactly one inside the loop, one at end


def test_init_worker_valid_and_invalid(monkeypatch):
    strats = _mini_strats(8)
    cfg = rt.TournamentConfig(n_players=4)
    rt._init_worker(strats, cfg)
    assert rt.N_PLAYERS == 4
    assert rt.GAMES_PER_SHUFFLE == 8_160 // 4

    with pytest.raises(ValueError):
        rt._init_worker(strats, rt.TournamentConfig(n_players=7))

    rt._init_worker(strats, rt.TournamentConfig(n_players=5))


def test_run_tournament_player_count(monkeypatch, tmp_path):
    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class DummyPool:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, arg):
            return DummyFuture(fn(arg))

    monkeypatch.setattr(rt, "ProcessPoolExecutor", lambda *a, **k: DummyPool())
    monkeypatch.setattr(rt, "_measure_throughput", lambda *a, **k: 1, raising=True)
    monkeypatch.setattr(rt, "NUM_SHUFFLES", 1, raising=False)
    monkeypatch.setattr(rt.Path, "write_bytes", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(rt.logging, "info", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(rt, "as_completed", lambda d: list(d.keys()), raising=True)

    cfg_ok = rt.TournamentConfig(n_players=6)
    rt.run_tournament(config=cfg_ok, checkpoint_path=tmp_path / "c.pkl", n_jobs=None)

    with pytest.raises(ValueError):
        cfg_bad = rt.TournamentConfig(n_players=7)
        rt.run_tournament(config=cfg_bad, checkpoint_path=tmp_path / "d.pkl", n_jobs=None)


def test_run_tournament_num_shuffles_override(monkeypatch, tmp_path):
    """run_tournament should honour the ``num_shuffles`` argument."""

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class DummyPool:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, arg):
            return DummyFuture(fn(arg))

    monkeypatch.setattr(rt, "ProcessPoolExecutor", lambda *a, **k: DummyPool())
    monkeypatch.setattr(rt, "_measure_throughput", lambda *a, **k: 1, raising=True)
    monkeypatch.setattr(rt.Path, "write_bytes", lambda *a, **k: None, raising=True)
    monkeypatch.setattr(rt.logging, "info", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(rt, "as_completed", lambda l: list(l), raising=True)

    called = []

    def fake_run_chunk(batch):
        called.append(list(batch))
        return Counter()

    monkeypatch.setattr(rt, "_run_chunk", fake_run_chunk, raising=True)

    cfg = rt.TournamentConfig(num_shuffles=5)
    rt.run_tournament(config=cfg, checkpoint_path=tmp_path / "x.pkl", num_shuffles=2)

    assert len(called) == 2
