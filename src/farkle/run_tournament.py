# src/farkle/run_tournament.py
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

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Globals & tuning knobs  (can be patched from tests or CLI)
# ─────────────────────────────────────────────────────────────────────────────
N_PLAYERS             = 5
NUM_SHUFFLES          = 10_223
GAMES_PER_SHUFFLE     = 8_160 // N_PLAYERS          # 1 632
DESIRED_SEC_PER_CHUNK = 10
CKPT_EVERY_SEC        = 30                          # default wall-clock cadence
LOG_TS_FMT            = "%Y-%m-%d %H:%M:%S"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Fast helpers – strategy list lives in a module-level var in each worker
# ─────────────────────────────────────────────────────────────────────────────
_STRATS: List[ThresholdStrategy] = []               # filled by _init_worker


def _init_worker(strategies: Sequence[ThresholdStrategy]) -> None:
    """One-off initialiser for every forked worker process."""
    global _STRATS
    _STRATS = list(strategies)                      # zero-copy index look-ups


def _play_table(seed: int, idxs: Sequence[int]) -> str:
    """Play a single 5-player game given *indices* into `_STRATS`."""
    table = [_STRATS[i] for i in idxs]
    res   = _play_game(seed, table)
    return res[f"{res['winner']}_strategy"]          # string repr → Counter key


def _play_shuffle(seed: int) -> Counter[str]:
    """Play the 1 632 games that make up **one** shuffle → Counter of wins."""
    rng    = np.random.default_rng(seed)
    perm   = rng.permutation(len(_STRATS))
    seeds  = rng.integers(0, 2**32 - 1, size=GAMES_PER_SHUFFLE)

    wins: Counter[str] = Counter()
    off = 0
    for s in seeds.tolist():                        # Ruff: prefer list[int]
        idxs = perm[off : off + N_PLAYERS].tolist()
        wins[_play_table(int(s), idxs)] += 1
        off += N_PLAYERS
    return wins


def _run_chunk(shuffle_seed_slice: Sequence[int]) -> Counter[str]:
    """Aggregate several shuffles – one ProcessPool task."""
    ctr: Counter[str] = Counter()
    for sd in shuffle_seed_slice:
        ctr.update(_play_shuffle(int(sd)))
    return ctr


def _measure_throughput(strats: Sequence[ThresholdStrategy],
                        test_games: int = 2_000,
                        seed: int = 0) -> float:
    """Rough games / sec so we can pick a sensible chunk size."""
    rng   = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=test_games)
    t0    = time.perf_counter()
    for s in seeds.tolist():
        _play_game(int(s), strats[:N_PLAYERS])
    return test_games / (time.perf_counter() - t0)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Main driver
# ─────────────────────────────────────────────────────────────────────────────
def run_tournament(
    *,
    global_seed: int = 0,
    checkpoint_path: str | Path = "checkpoint.pkl",
    n_jobs: int | None = None,
    ckpt_every_sec: int = CKPT_EVERY_SEC,
) -> None:

    strategies, _ = generate_strategy_grid()                 # 8 160 objects
    gps = _measure_throughput(strategies[:N_PLAYERS])
    shuffles_per_chunk = max(1,
        int(DESIRED_SEC_PER_CHUNK * gps // GAMES_PER_SHUFFLE))

    master        = np.random.default_rng(global_seed)
    shuffle_seeds = master.integers(0, 2**32 - 1,
                                    size=NUM_SHUFFLES).tolist()

    chunks = [
        shuffle_seeds[i : i + shuffles_per_chunk]
        for i in range(0, NUM_SHUFFLES, shuffles_per_chunk)
    ]

    # --- logging ---------------------------------------------------------
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    win_totals: Counter[str] = Counter()
    last_save = time.perf_counter()                 # wall-clock book-keeping
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=n_jobs,
                             initializer=_init_worker,
                             initargs=(strategies,)) as pool:

        fut2idx = {pool.submit(_run_chunk, c): i for i, c in enumerate(chunks)}
        for done, fut in enumerate(as_completed(fut2idx), 1):
            win_totals.update(fut.result())

            now = time.perf_counter()
            if now - last_save >= ckpt_every_sec:
                ckpt_path.write_bytes(pickle.dumps(win_totals,
                                                   protocol=pickle.HIGHEST_PROTOCOL))
                ts = time.strftime(LOG_TS_FMT, time.localtime())
                logging.info("%s  %5d/%d chunks  %d games played",
                             ts, done, len(chunks), sum(win_totals.values()))
                last_save = now

    ckpt_path.write_bytes(pickle.dumps(win_totals,
                                       protocol=pickle.HIGHEST_PROTOCOL))
    logging.info("All done! Final win counts:\n%s", win_totals)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLI convenience – lets user override checkpoint cadence
# ─────────────────────────────────────────────────────────────────────────────
def main(
    global_seed: int = 0,
    checkpoint_path: str | Path = "checkpoint.pkl",
    n_jobs: int = 16,
    ckpt_every_sec: int = CKPT_EVERY_SEC,
) -> None:
    mp.set_start_method("spawn", force=True)
    run_tournament(global_seed=global_seed,
                   checkpoint_path=checkpoint_path,
                   n_jobs=n_jobs,
                   ckpt_every_sec=ckpt_every_sec)


if __name__ == "__main__":
    main()
