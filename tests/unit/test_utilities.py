import sys
from pathlib import Path

import pytest
import yaml

from farkle import farkle_cli  # imports the module, not the exe
from farkle.farkle_io import simulate_many_games_stream
import queue
import threading
import csv
import farkle.farkle_io as farkle_io
from farkle.stats import games_for_power
from farkle.strategies import ThresholdStrategy


def test_games_for_power_monotonic():
    n_small_delta = games_for_power(n_strategies=2, delta=0.05)
    n_large_delta = games_for_power(n_strategies=2, delta=0.10)
    # Larger effect size ⇒ fewer games required
    assert n_large_delta < n_small_delta


def test_cli_run(tmp_path, monkeypatch):
    cfg = {
        "strategy_grid": {
            "score_thresholds": [300],
            "dice_thresholds": [2],
            "smart_five_opts": [False],
            "smart_one_opts": [False],
            "consider_score_opts": [True],
            "consider_dice_opts": [True],
            "auto_hot_opts": [False],
        },
        "sim": {
            "n_games": 2,
            "out_csv": str(tmp_path / "out.csv"),
            "seed": 42,
            "n_jobs": 1,
        },
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(sys, "argv", ["farkle", "run", str(cfg_path)])
    farkle_cli.main()

    assert Path(cfg["sim"]["out_csv"]).exists()


def test_stream_writer(tmp_path):
    out_csv = tmp_path / "results.csv"
    strat = [ThresholdStrategy(score_threshold=300, dice_threshold=2)]
    simulate_many_games_stream(
        n_games=10, strategies=strat,
        out_csv=str(out_csv), seed=123, n_jobs=1
    )
    lines = out_csv.read_text().splitlines()
    assert len(lines) == 11  # header + 10 rows
    header = lines[0].split(",")
    assert header == ["game_id", "winner", "winning_score", "winner_strategy", "n_rounds"]
    
@pytest.mark.parametrize(
    "method,pairwise", [("bh", True), ("bonferroni", True), ("bonferroni", False)]
)
def test_games_for_power_branches(method, pairwise):
    n = games_for_power(n_strategies=3, method=method, pairwise=pairwise)
    assert n > 0


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_stream_parallel(tmp_path, n_jobs):
    # tiny run hits both serial & MP code paths
    out = tmp_path / "w.csv"
    strategies = [ThresholdStrategy(score_threshold=0, dice_threshold=6)]
    simulate_many_games_stream(
        n_games=4, strategies=strategies, out_csv=str(out),
        seed=7, n_jobs=n_jobs
    )
    rows = out.read_text().splitlines()
    assert len(rows) == 5  # header + 4


def test_stream_custom_tmpdir(tmp_path, monkeypatch):
    tmpdir = tmp_path / "mptmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))

    out_csv = tmp_path / "tmpdir.csv"
    strategies = [ThresholdStrategy(score_threshold=0, dice_threshold=6)]
    simulate_many_games_stream(
        n_games=4, strategies=strategies, out_csv=str(out_csv), seed=5, n_jobs=2
    )
    rows = out_csv.read_text().splitlines()
    assert len(rows) == 5


def test_stream_buffer_queue_limits(tmp_path, monkeypatch):
    buffer_size = 3
    queue_size = 2

    def writer_worker(q, outpath, header):
        first = not Path(outpath).exists()
        with open(outpath, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            if first:
                w.writeheader()
            buf = []
            while True:
                row = q.get()
                if row is None:
                    break
                buf.append(row)
                if len(buf) >= buffer_size:
                    w.writerows(buf)
                    fh.flush()
                    buf.clear()
            if buf:
                w.writerows(buf)

    monkeypatch.setattr(farkle_io, "_writer_worker", writer_worker)

    monkeypatch.setattr(
        farkle_io.mp,
        "Queue",
        lambda *a, **k: queue.Queue(maxsize=queue_size),
    )

    class ThreadProcess:
        def __init__(self, target, args=()):
            self._thread = threading.Thread(target=target, args=args)

        def start(self):
            self._thread.start()

        def join(self):
            self._thread.join()

    monkeypatch.setattr(farkle_io.mp, "Process", ThreadProcess)

    class DummyPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def imap_unordered(self, func, iterable, chunksize=1):
            for item in iterable:
                yield func(item)

    monkeypatch.setattr(farkle_io.mp, "Pool", DummyPool)

    out_csv = tmp_path / "limits.csv"
    strategies = [ThresholdStrategy(score_threshold=0, dice_threshold=6)]
    n_games = max(buffer_size, queue_size) + 2
    simulate_many_games_stream(
        n_games=n_games, strategies=strategies, out_csv=str(out_csv), seed=9, n_jobs=2
    )

    rows = out_csv.read_text().splitlines()
    assert len(rows) == n_games + 1
