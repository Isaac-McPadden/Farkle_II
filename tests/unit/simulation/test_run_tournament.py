"""
Fast, side-effect-free sanity checks for run_tournament.py
(all run in < 1 s).

We monkey-patch the heavy helpers so no real games are played.
"""

from __future__ import annotations

import logging
import types  # noqa: F401
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast
from unittest.mock import ANY

import numpy as np  # noqa: F401 | Potentially imports something that needs it
import pytest

import farkle.simulation.run_tournament as rt
from farkle.simulation.strategies import ThresholdStrategy

# --------------------------------------------------------------------------- #
# Mini test doubles – replace expensive pieces with cheap determinism
# --------------------------------------------------------------------------- #


def _mini_strats(n: int = 6):
    """Return deterministic Strategy objects with distinct __str__()."""
    return [ThresholdStrategy(50 + 50 * i, i % 3, True, True) for i in range(n)]


def fake_play_shuffle(seed: int) -> Counter[str]:
        # pretend player at index (seed % len(strats)) always wins
        strats = _mini_strats(12)
        return Counter({str(strats[seed % len(strats)]): 1})


@pytest.fixture(autouse=True)
def fast_helpers(monkeypatch):
    """
    Keep tests lightning-fast **and** remember the real ``_play_shuffle`` so
    individual tests can restore it when they *need* the real behavior.
    """
    strats = _mini_strats(12)
    monkeypatch.setattr(
        rt, "generate_strategy_grid", lambda *a, **kw: (strats, None), raising=True  # noqa: ARG005
    )

    # save the genuine function on the module *once*
    if not hasattr(rt, "_ORIG_PLAY_SHUFFLE"):
        rt._ORIG_PLAY_SHUFFLE = rt._play_shuffle # type: ignore

    monkeypatch.setattr(rt, "_play_shuffle", fake_play_shuffle, raising=True)


@pytest.fixture(autouse=True)
def silence_logging():
    """Silence run_tournament logs by default while allowing tests to override."""
    level = rt.LOGGER.level
    rt.LOGGER.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        rt.LOGGER.setLevel(level)


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


    # ── 6 · run – we expect 1 mid-run save + 1 final save ─────────────────────
    rt.run_tournament(
        config=cfg,
        global_seed=0,
        checkpoint_path=tmp_path / "chk.pkl",
        n_jobs=None,
    )

    assert saves["n"] == 2  # exactly one inside the loop, one at end


def test_init_worker_valid_and_invalid():
    strats = _mini_strats(8)
    cfg = rt.TournamentConfig(n_players=4)
    rt._init_worker(strats, cfg, None)
    assert rt._STATE is not None
    assert rt._STATE.cfg.n_players == 4
    assert rt._STATE.cfg.games_per_shuffle == 8_160 // 4

    with pytest.raises(ValueError):
        rt._init_worker(strats, rt.TournamentConfig(n_players=7), None)

    rt._init_worker(strats, rt.TournamentConfig(n_players=5), None)
    assert rt._STATE is not None and rt._STATE.cfg.n_players == 5


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

    monkeypatch.setattr(rt, "ProcessPoolExecutor", lambda *a, **k: DummyPool())  # noqa: ARG005
    monkeypatch.setattr(rt, "_measure_throughput", lambda *a, **k: 1, raising=True)  # noqa: ARG005
    monkeypatch.setattr(rt, "NUM_SHUFFLES", 1, raising=False)
    monkeypatch.setattr(rt.Path, "write_bytes", lambda *a, **k: None, raising=True)  # noqa: ARG005
    monkeypatch.setattr(rt, "as_completed", lambda d: list(d.keys()), raising=True)

    cfg_ok = rt.TournamentConfig(n_players=6)
    rt.run_tournament(config=cfg_ok, checkpoint_path=tmp_path / "c.pkl", n_jobs=None)

    with pytest.raises(ValueError):
        cfg_bad = rt.TournamentConfig(n_players=7)
        rt.run_tournament(config=cfg_bad, checkpoint_path=tmp_path / "d.pkl", n_jobs=None)


def test_run_tournament_emits_logging(monkeypatch, caplog, tmp_path):
    caplog.set_level(logging.INFO, logger="farkle.simulation.run_tournament")

    def fake_process_map(fn, iterable, **kwargs):  # noqa: ANN001
        for item in iterable:
            yield fn(item)

    monkeypatch.setattr(rt.parallel, "process_map", fake_process_map)
    monkeypatch.setattr(rt, "_measure_throughput", lambda *a, **k: 1, raising=True)
    monkeypatch.setattr(rt.urandom, "spawn_seeds", lambda count, seed=0: range(seed, seed + count))

    cfg = rt.TournamentConfig(num_shuffles=2, ckpt_every_sec=1)
    rt.run_tournament(
        config=cfg,
        global_seed=0,
        checkpoint_path=tmp_path / "logchk.pkl",
        n_jobs=1,
    )

    messages = [rec.message for rec in caplog.records]
    assert any("Tournament run start" in msg for msg in messages)
    assert any("Tournament run complete" in msg for msg in messages)


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

    monkeypatch.setattr(rt, "ProcessPoolExecutor", lambda *a, **k: DummyPool())  # noqa: ARG005
    monkeypatch.setattr(rt, "_measure_throughput", lambda *a, **k: 1, raising=True)  # noqa: ARG005
    monkeypatch.setattr(rt.Path, "write_bytes", lambda *a, **k: None, raising=True)  # noqa: ARG005
    monkeypatch.setattr(rt, "as_completed", lambda x: list(x), raising=True)

    called = []

    def fake_run_chunk(batch):
        called.append(list(batch))
        return Counter()

    monkeypatch.setattr(rt, "_run_chunk", fake_run_chunk, raising=True)

    cfg = rt.TournamentConfig(num_shuffles=5)
    rt.run_tournament(config=cfg, checkpoint_path=tmp_path / "x.pkl", num_shuffles=2)

    assert len(called) == 2


def test_run_tournament_config_overrides(monkeypatch, tmp_path):
    """n_players and num_shuffles args should update the passed config."""

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

    monkeypatch.setattr(rt, "ProcessPoolExecutor", lambda *a, **k: DummyPool())  # noqa: ARG005
    monkeypatch.setattr(rt, "_measure_throughput", lambda *a, **k: 1, raising=True)  # noqa: ARG005
    monkeypatch.setattr(rt.Path, "write_bytes", lambda *a, **k: None, raising=True)  # noqa: ARG005
    monkeypatch.setattr(rt, "as_completed", lambda x: list(x), raising=True)
    monkeypatch.setattr(rt, "_run_chunk", lambda batch: Counter(), raising=True)  # noqa: ARG005

    cfg = rt.TournamentConfig(n_players=5, num_shuffles=3, ckpt_every_sec=99)
    rt.run_tournament(
        config=cfg,
        n_players=4,
        num_shuffles=2,
        checkpoint_path=tmp_path / "c.pkl",
    )

    assert cfg.n_players == 4
    assert cfg.num_shuffles == 2


def test_extract_winner_metrics():
    row = {
        "winning_score": 100,
        "n_rounds": 5,
        "A_farkles": 1,
        "A_rolls": 2,
        "A_highest_turn": 3,
        "A_smart_five_uses": 4,
        "A_n_smart_five_dice": 5,
        "A_smart_one_uses": 6,
        "A_n_smart_one_dice": 7,
        "A_hot_dice": 8,
    }
    assert rt._extract_winner_metrics(row, "A") == [100, 5, 1, 2, 3, 4, 5, 6, 7, 8]


def test_play_one_shuffle_collects(monkeypatch):
    strats = _mini_strats(4)
    cfg = rt.TournamentConfig(n_players=2)
    rt._init_worker(strats, cfg, None)
    rt._STATE = rt.WorkerState(
        list(strats),
        cast(rt.TournamentConfig, types.SimpleNamespace(n_players=2, games_per_shuffle=2)),
        None,
    )

    class DummyRNG:
        def permutation(self, n): return rt.np.arange(n)
        def integers(self, *a, size): return rt.np.arange(1, size + 1)  # noqa: ARG002

    monkeypatch.setattr(rt.np.random, "default_rng", lambda _: DummyRNG(), raising=True)

    def fake_play_game(seed, players):  # noqa: ARG002
        return {
            "winner": "P0",
            "P0_strategy": str(players[0]),
            "winning_score": seed,
            "n_rounds": seed + 1,
            "P0_farkles": seed + 2,
            "P0_rolls": seed + 3,
            "P0_highest_turn": seed + 4,
            "P0_smart_five_uses": seed + 5,
            "P0_n_smart_five_dice": seed + 6,
            "P0_smart_one_uses": seed + 7,
            "P0_n_smart_one_dice": seed + 8,
            "P0_hot_dice": seed + 9,
        }

    monkeypatch.setattr(rt, "_play_game", fake_play_game, raising=True)

    wins, sums, sqs, rows = rt._play_one_shuffle(0, collect_rows=True)
    assert wins == Counter({str(strats[0]): 1, str(strats[2]): 1})
    first, second = range(1, 11), range(2, 12)
    for lab, v1, v2 in zip(rt.METRIC_LABELS, first, second, strict=True):
        assert sums[lab][str(strats[0])] == v1
        assert sums[lab][str(strats[2])] == v2
        assert sqs[lab][str(strats[0])] == v1 * v1
        assert sqs[lab][str(strats[2])] == v2 * v2
    assert rows[0]["game_seed"] == 1 and rows[1]["game_seed"] == 2


def test_run_chunk_metrics_queue(monkeypatch):
    def fake_play_one_shuffle(seed, *, collect_rows=False):  # noqa: ARG002
        w = Counter({f"S{seed}": 1})
        sums = {m: defaultdict(float, {f"S{seed}": float(seed)}) for m in rt.METRIC_LABELS}
        sqs = {m: defaultdict(float, {f"S{seed}": float(seed * seed)}) for m in rt.METRIC_LABELS}
        rows = [{"game_seed": seed}] if collect_rows else []
        return w, sums, sqs, rows

    monkeypatch.setattr(rt, "_play_one_shuffle", fake_play_one_shuffle, raising=True)

    sent = []

    class DummyQueue:
        def put(self, item):
            sent.append(item)

    monkeypatch.setattr(
        rt,
        "_STATE",
        rt.WorkerState(
            [],
            cast(rt.TournamentConfig, types.SimpleNamespace(games_per_shuffle=0)),
            cast(rt.mp.Queue, DummyQueue()),
        ),
        raising=False,
    )

    wins, sums, sqs = rt._run_chunk_metrics([1, 2], collect_rows=True)
    assert sent == [{"game_seed": 1}, {"game_seed": 2}]
    assert wins == Counter({"S1": 1, "S2": 1})
    for m in rt.METRIC_LABELS:
        assert sums[m]["S1"] == 1 and sums[m]["S2"] == 2
        assert sqs[m]["S1"] == 1 and sqs[m]["S2"] == 4


def test_measure_throughput(monkeypatch):
    calls = []
    monkeypatch.setattr(rt, "_play_game", lambda s, p: calls.append(s), raising=True)  # noqa: ARG005
    monkeypatch.setattr(rt.time, "perf_counter", lambda: 0 if not calls else 1, raising=True)
    assert rt._measure_throughput(_mini_strats(2), sample_games=3, seed=0) == 3
    assert len(calls) == 3


def test_play_shuffle_wrapper(monkeypatch):
    import farkle.simulation.run_tournament as rt

    called = {}

    # 1 · stub that records the seed and returns correct tuple
    def fake_play_one_shuffle(seed, *, collect_rows=False):  # noqa: ARG001, ARG002
        called.setdefault("seed", seed)
        return Counter(), {}, {}, []

    monkeypatch.setattr(rt, "_play_one_shuffle", fake_play_one_shuffle, raising=True)

    # 2 · restore the genuine wrapper just for this test
    monkeypatch.setattr(rt, "_play_shuffle", rt._ORIG_PLAY_SHUFFLE, raising=True)  # type:ignore

    rt._play_shuffle(123)
    assert called["seed"] == 123


def test_run_tournament_cli(monkeypatch):
    called = {}
    monkeypatch.setattr(rt, "run_tournament", lambda **kw: called.update(kw), raising=True)

    class DummyArgs:
        config, seed, checkpoint, jobs, ckpt_sec, metrics, num_shuffles = (
            ANY,
            7,
            Path("out.pkl"),
            2,
            5,
            True,
            3,
        )
        row_dir = Path("rows")
        metric_chunk_dir = Path("chunks")
        log_level = "INFO"

    monkeypatch.setattr(rt.argparse, "ArgumentParser", lambda *a, **k: type("P", (), {"add_argument": lambda *a, **k: None, "parse_args": lambda _: DummyArgs})())  # noqa: ARG005

    rt.main()
    assert called == {
        "config": ANY,
        "global_seed": 7,
        "checkpoint_path": Path("out.pkl"),
        "n_jobs": 2,
        "collect_metrics": True,
        "row_output_directory": Path("rows"),
        "metric_chunk_directory": Path("chunks"),
        "num_shuffles": 3,
    }


def test_run_tournament_collect_rows(monkeypatch, tmp_path):
    class DummyFuture:
        def __init__(self, r): self._r = r
        def result(self): return self._r
        def __hash__(self): return id(self)

    class DummyPool:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def submit(self, fn, arg): return DummyFuture(fn(arg))

    monkeypatch.setattr(rt, "ProcessPoolExecutor", lambda *a, **k: DummyPool())  # noqa: ARG005
    monkeypatch.setattr(rt, "_measure_throughput", lambda *a, **k: 1, raising=True)  # noqa: ARG005
    monkeypatch.setattr(rt.Path, "write_bytes", lambda *a, **k: None, raising=True)  # noqa: ARG005
    monkeypatch.setattr(rt, "as_completed", lambda d: list(d.keys()), raising=True)
    monkeypatch.setattr(
        rt,
        "_run_chunk_metrics",
        lambda *a, **k: (Counter(), {m: defaultdict(float) for m in rt.METRIC_LABELS}, {m: defaultdict(float) for m in rt.METRIC_LABELS}),  # noqa: ARG005
        raising=True,
    )

    row_dir = tmp_path / "rows"
    started = {"n": 0}

    class DummyProcess:
        def __init__(self, *a, **k): pass
        def start(self): started["n"] += 1

    class DummyCtx:
        def Queue(self, maxsize): return type("Q", (), {"put": lambda *a, **k: None})()  # noqa: ARG002, ARG005
        def Process(self, *a, **k): return DummyProcess()  # noqa: ARG002

    monkeypatch.setattr(rt.mp, "get_context", lambda *a, **k: DummyCtx())  # noqa: ARG005

    cfg = rt.TournamentConfig(num_shuffles=1, ckpt_every_sec=30)
    rt.run_tournament(
        config=cfg,
        global_seed=0,
        checkpoint_path=tmp_path / "chk.pkl",
        n_jobs=None,
        collect_metrics=True,
        row_output_directory=row_dir,
    )
    assert started["n"] == 1


def test_run_tournament_metric_chunking(monkeypatch, tmp_path):
    class DummyFuture:
        def __init__(self, r): self._r = r
        def result(self): return self._r
        def __hash__(self): return id(self)

    class DummyPool:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def submit(self, fn, arg): return DummyFuture(fn(arg))

    monkeypatch.setattr(rt, "ProcessPoolExecutor", lambda *a, **k: DummyPool())  # noqa: ARG005
    monkeypatch.setattr(rt, "_measure_throughput", lambda *a, **k: 1, raising=True)  # noqa: ARG005
    monkeypatch.setattr(rt, "as_completed", lambda d: list(d.keys()), raising=True)

    def fake_run_chunk_metrics(batch, collect_rows=False, row_dir=None):  # noqa: ARG001
        w = Counter({"S": 1})
        sums = {m: defaultdict(float, {"S": 1.0}) for m in rt.METRIC_LABELS}
        sqs = {m: defaultdict(float, {"S": 1.0}) for m in rt.METRIC_LABELS}
        return w, sums, sqs

    monkeypatch.setattr(rt, "_run_chunk_metrics", fake_run_chunk_metrics, raising=True)

    captured = {}

    def fake_save_checkpoint(path, wins, sums, sqs):  # noqa: ARG001
        captured["wins"] = wins
        captured["sums"] = sums
        captured["sqs"] = sqs

    monkeypatch.setattr(rt, "_save_checkpoint", fake_save_checkpoint, raising=True)

    cfg = rt.TournamentConfig(num_shuffles=2)
    metric_dir = tmp_path / "chunks"

    rt.run_tournament(
        config=cfg,
        global_seed=0,
        checkpoint_path=tmp_path / "chk.pkl",
        n_jobs=None,
        collect_metrics=True,
        metric_chunk_directory=metric_dir,
        num_shuffles=2,
    )

    files = sorted(metric_dir.glob("metrics_*.parquet"))
    assert len(files) == 2
    for m in rt.METRIC_LABELS:
        assert captured["sums"][m]["S"] == 2.0
        assert captured["sqs"][m]["S"] == 2.0
