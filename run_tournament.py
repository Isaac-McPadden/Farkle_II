#!/usr/bin/env python3
"""Parallel tournament runner for Farkle strategies.

This script generates the default strategy grid and plays a powered number of
round‐robin games using multiprocessing. Progress is periodically written to a
checkpoint file so long runs can be resumed.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Locate project root and ensure the package is importable
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    try:
        start = Path(__file__).resolve()
    except NameError:  # interactive session
        start = Path.cwd()
    for p in (start, *start.parents):
        if (p / "Src" / "Farkle").is_dir():
            return p
    return Path.cwd()

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from farkle.simulation import generate_strategy_grid, _play_game
from farkle.stats import games_for_power

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNKSIZE = 200
FLUSH_EVERY = 1000
PROCESSES = 16
QUEUE_MAXSIZE = 50_000
REPORT_EVERY = 100_000
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = CHECKPOINT_DIR / "win_counter.chk"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class FirstNFilter(logging.Filter):
    """Limit DEBUG spam to the first *n* occurrences per callsite."""

    def __init__(self, n: int = 10_000) -> None:
        super().__init__()
        self.n = n
        self.seen: Counter[Tuple[str, int]] = Counter()

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - bool
        key = (record.pathname, record.lineno)
        self.seen[key] += 1
        return self.seen[key] <= self.n

root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(levelname)-5s | %(processName)s | %(message)s",
        datefmt="%H:%M:%S",
    )
)
handler.addFilter(FirstNFilter())
root.handlers[:] = [handler]
log = logging.getLogger("tournament")
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def producer(task_q: mp.Queue, n_games_per_player: int, num_strats: int) -> None:
    """Enqueue (seed, idx tuple) tasks for all tables."""
    perm_rng = np.random.default_rng(999)
    seed_rng = np.random.default_rng(1234)
    for _ in range(n_games_per_player):
        perm = perm_rng.permutation(num_strats)
        for j in range(0, num_strats, 5):
            if j + 5 > num_strats:
                break
            seed = int(seed_rng.integers(2**32))
            task_q.put((seed, tuple(int(x) for x in perm[j:j + 5])))
    for _ in range(PROCESSES):
        task_q.put(None)


def worker(strat_list: List, task_q: mp.Queue, result_q: mp.Queue) -> None:
    strategies = strat_list
    local_counter: Counter[str] = Counter()
    processed = 0
    while True:
        batch: List[Tuple[int, Tuple[int, ...]]] = []
        sentinel = False
        for _ in range(CHUNKSIZE):
            task = task_q.get()
            if task is None:
                sentinel = True
                break
            batch.append(task)
        if not batch and sentinel:
            break
        for seed, idxs in batch:
            row = _play_game(seed, [strategies[i] for i in idxs], 10_000)
            winner_key = str(row[f"{row['winner']}_strategy"])
            local_counter[winner_key] += 1
            processed += 1
            if processed % FLUSH_EVERY == 0:
                result_q.put(local_counter)
                local_counter = Counter()
        if sentinel:
            break
    if local_counter:
        result_q.put(local_counter)
    result_q.put(None)


def collector(result_q: mp.Queue, num_strats: int, strategies: List) -> None:
    win_counter: Counter[str] = Counter()
    done_batches = 0
    active_workers = PROCESSES
    start = time.perf_counter()
    while active_workers:
        msg = result_q.get()
        if msg is None:
            active_workers -= 1
            continue
        win_counter.update(msg)
        done_batches += 1
        if done_batches % REPORT_EVERY == 0:
            hrs = (time.perf_counter() - start) / 3600
            log.info("[batch %d] %.2f h elapsed", done_batches, hrs)
            with CHECKPOINT_FILE.open("wb") as f:
                pickle.dump({"done": done_batches, "counter": dict(win_counter)}, f)
    # final write
    with CHECKPOINT_FILE.open("wb") as f:
        pickle.dump({"done": done_batches, "counter": dict(win_counter)}, f)
    df = pd.DataFrame({
        "strategy_idx": range(num_strats),
        "str_repr": [str(s) for s in strategies],
    })
    df["wincount"] = df["str_repr"].map(win_counter).fillna(0).astype(int)
    df.to_csv("wincounts.csv", index=False)
    log.info("collector CSV written")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    strategies, _ = generate_strategy_grid()
    num_strats = len(strategies)
    n_games_per_player = games_for_power(
        n_strategies=num_strats,
        delta=0.03,
        alpha=0.025,
        power=0.90,
        method="bh",
        pairwise=True,
    )
    total_tasks = num_strats * n_games_per_player // 5
    log.info(
        "Grid: %d strategies, %d games/strat → %d tasks.",
        num_strats,
        n_games_per_player,
        total_tasks,
    )

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=QUEUE_MAXSIZE)
    result_q = ctx.Queue(maxsize=QUEUE_MAXSIZE)

    prod = threading.Thread(
        target=producer,
        args=(task_q, n_games_per_player, num_strats),
        daemon=False,
    )
    coll = threading.Thread(
        target=collector,
        args=(result_q, num_strats, strategies),
        daemon=False,
    )
    prod.start()
    coll.start()

    processes = [
        ctx.Process(target=worker, args=(strategies, task_q, result_q))
        for _ in range(PROCESSES)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    prod.join()
    coll.join()
    log.info("All workers joined – tournament complete.")
