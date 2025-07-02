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
    """Initialise the worker process with a strategy list.

    Inputs
    ------
    strategies : Sequence[ThresholdStrategy]
        Strategy objects created in the parent process.

    Returns
    -------
    None
        _STRATS is populated for fast lookups in the worker.
    """
    global _STRATS
    _STRATS = list(strategies)                      # zero-copy index look-ups


def _play_table(seed: int, idxs: Sequence[int]) -> str:
    """Run one table of five players using stored strategies.

    Inputs
    ------
    seed : int
        RNG seed for the game engine.
    idxs : Sequence[int]
        Indices into the global _STRATS list.

    Returns
    -------
    str
        Strategy string representation of the winning player.
    """
    table = [_STRATS[i] for i in idxs]
    res   = _play_game(seed, table)
    return res[f"{res['winner']}_strategy"]          # string repr → Counter key


def _play_shuffle(seed: int) -> Counter[str]:
    """Play all games that comprise a single shuffle.

    Inputs
    ------
    seed : int
        Seed used to create permutations and per-game seeds.

    Returns
    -------
    Counter[str]
        Mapping of strategy strings to win counts for this shuffle.
    """
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
    """Aggregate multiple shuffles in one worker call.

    Inputs
    ------
    shuffle_seed_slice : Sequence[int]
        Collection of seeds, one for each shuffle to execute.

    Returns
    -------
    Counter[str]
        Combined win counts across all shuffles in shuffle_seed_slice.
    """
    ctr: Counter[str] = Counter()
    for sd in shuffle_seed_slice:
        ctr.update(_play_shuffle(int(sd)))
    return ctr


def _measure_throughput(strats: Sequence[ThresholdStrategy],
                        test_games: int = 2_000,
                        seed: int = 0) -> float:
    """Estimate processing speed in games per second.

    Inputs
    ------
    strats : Sequence[ThresholdStrategy]
        Strategy objects used in the benchmark games.
    test_games : int, default 2000
        Number of games to run for the estimate.
    seed : int, default 0
        RNG seed for reproducibility.

    Returns
    -------
    float
        Approximate games processed per second.
    """
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
    """Run the full tournament across many worker processes.

    Inputs
    ------
    global_seed : int, default 0
        Seed controlling the overall reproducibility of the tournament.
    checkpoint_path : str | Path, default "checkpoint.pkl"
        File path to which intermediate results are periodically saved.
    n_jobs : int | None, optional
        Number of worker processes, None uses CPU count.
    ckpt_every_sec : int, default CKPT_EVERY_SEC
        Seconds between checkpoint writes.

    Returns
    -------
    None
        Progress is logged and the final win counts are written to checkpoint_path.
    """
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
    """main()

    Inputs
    ------
    global_seed : int, default 0
        Master seed for reproducibility.
    checkpoint_path : str | Path, default "checkpoint.pkl"
        Location to store tournament checkpoints.
    n_jobs : int, default 16
        Number of worker processes to spawn.
    ckpt_every_sec : int, default CKPT_EVERY_SEC
        Seconds between checkpoint writes.

    Returns
    -------
    None
        The function runs run_tournament with the provided parameters.
    """
    mp.set_start_method("spawn", force=True)
    run_tournament(global_seed=global_seed,
                   checkpoint_path=checkpoint_path,
                   n_jobs=n_jobs,
                   ckpt_every_sec=ckpt_every_sec)


if __name__ == "__main__":
    main()
