# src/farkle/run_tournament_2.py
from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence

import numpy as np

from farkle.simulation import _play_game, generate_strategy_grid
from farkle.strategies import ThresholdStrategy

# ── Constants ───────────────────────────────────────────────────────────────
N_PLAYERS            = 5
NUM_SHUFFLES         = 10_223                   # games-per-strategy
GAMES_PER_SHUFFLE    = 8_160 // N_PLAYERS       # 1 632
DESIRED_SEC_PER_CHUNK = 10

# ── Globals shared by worker processes ──────────────────────────────────────
_STRATS: List[ThresholdStrategy] = []           # filled by _init_worker

# ----------------------------------------------------------------------------
# 0.  Worker initialiser – runs once per fork
# ----------------------------------------------------------------------------
def _init_worker(strategies: Sequence[ThresholdStrategy]) -> None:
    """Store the strategies in a module-level variable (zero copy per task)."""
    global _STRATS
    _STRATS = list(strategies)                  # plain list → fast index lookup


# ----------------------------------------------------------------------------
# 1.  Fast helpers that use the global strategy list
# ----------------------------------------------------------------------------
def _play_table(seed: int, idxs: Sequence[int]) -> str:
    """Play one game given *indices* into _STRATS, return the winner’s repr."""
    table = [_STRATS[i] for i in idxs]
    res   = _play_game(seed, table)
    return res[f"{res['winner']}_strategy"]


def _play_shuffle(seed: int) -> Counter[str]:
    """Play all 1 632 games that make up one shuffle."""
    rng    = np.random.default_rng(seed)
    perm   = rng.permutation(len(_STRATS))      # ndarray[int64]
    seeds  = rng.integers(0, 2**32 - 1, size=GAMES_PER_SHUFFLE)

    wins: Counter[str] = Counter()
    offset = 0
    for s in seeds.tolist():                    # cast → list[int]  (Ruff happy)
        idxs = perm[offset : offset + N_PLAYERS].tolist()
        wins[_play_table(int(s), idxs)] += 1
        offset += N_PLAYERS
    return wins


# ----------------------------------------------------------------------------
# 2.  Chunk wrappers
# ----------------------------------------------------------------------------
def _run_chunk(shuffle_seed_slice: Sequence[int]) -> Counter[str]:
    """Aggregate results from several shuffles (one process-pool task)."""
    ctr: Counter[str] = Counter()
    for sd in shuffle_seed_slice:
        ctr.update(_play_shuffle(int(sd)))
    return ctr


# ----------------------------------------------------------------------------
# 3.  Utility: empirical games-per-second
# ----------------------------------------------------------------------------
def _measure_throughput(strategies: Sequence[ThresholdStrategy],
                        test_games: int = 2_000,
                        seed: int = 0) -> float:
    rng   = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=test_games)
    t0    = time.perf_counter()
    for s in seeds.tolist():
        _play_game(int(s), strategies[:N_PLAYERS])
    return test_games / (time.perf_counter() - t0)


# ----------------------------------------------------------------------------
# 4.  Main driver
# ----------------------------------------------------------------------------
def run_tournament(global_seed: int = 0,
                   checkpoint_path: str | Path = "checkpoint.pkl",
                   n_jobs: int | None = None) -> None:

    strategies, _ = generate_strategy_grid()          # 8 160 objects
    gps  = _measure_throughput(strategies[:N_PLAYERS])
    shuffles_per_chunk = max(1,
        int(DESIRED_SEC_PER_CHUNK * gps // GAMES_PER_SHUFFLE))

    master         = np.random.default_rng(global_seed)
    shuffle_seeds  = master.integers(0, 2**32 - 1, size=NUM_SHUFFLES).tolist()

    # split into roughly equal-sized chunks
    chunks = [
        shuffle_seeds[i : i + shuffles_per_chunk]
        for i in range(0, NUM_SHUFFLES, shuffles_per_chunk)
    ]

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    win_counter: Counter[str] = Counter()

    with ProcessPoolExecutor(max_workers=n_jobs,
                             initializer=_init_worker,
                             initargs=(strategies,)) as pool:

        futures = {pool.submit(_run_chunk, c): idx
                   for idx, c in enumerate(chunks)}
        for done, fut in enumerate(as_completed(futures), 1):
            win_counter.update(fut.result())
            logging.info("after %5d/%d chunks: %d games played",
                         done, len(chunks), sum(win_counter.values()))

            if done % 10 == 0:
                Path(checkpoint_path).write_bytes(pickle.dumps(win_counter))
                logging.info("checkpoint saved → %s", checkpoint_path)

    Path(checkpoint_path).write_bytes(pickle.dumps(win_counter))
    logging.info("All done! Final win counts:\n%s", win_counter)


# ----------------------------------------------------------------------------
# 5.  Convenience wrapper
# ----------------------------------------------------------------------------
def main(global_seed: int = 0,
         checkpoint_path: str | Path = "checkpoint.pkl",
         n_jobs: int = 16) -> None:
    mp.set_start_method("spawn", force=True)
    run_tournament(global_seed, checkpoint_path, n_jobs)


if __name__ == "__main__":
    main()
