import logging
import multiprocessing as mp
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from farkle.simulation import (
    _play_game,
    generate_strategy_grid,  # <- replace with your real loader
)
from farkle.strategies import ThresholdStrategy

# ── Configuration ───────────────────────────────────────────────────────────
N_PLAYERS          = 5
NUM_SHUFFLES       = 10_223                 # what you called “GAMES_PER_STRAT”
GAMES_PER_SHUFFLE  = 8160 // N_PLAYERS      # = 1 632
TOTAL_GAMES        = NUM_SHUFFLES * GAMES_PER_SHUFFLE   # = 16 683 936
DESIRED_SEC_PER_CHUNK = 10

# ── Throughput estimation ────────────────────────────────────────────────────
def measure_throughput(
    strategies: Sequence[ThresholdStrategy],
    test_games: int = 2000,
    seed: int = 0
) -> float:
    """
    Simulate a small batch to estimate games/sec.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=test_games)
    t0 = time.perf_counter()
    for s in seeds:
        _play_game(int(s), strategies[:N_PLAYERS])
    dt = time.perf_counter() - t0
    return test_games / dt

# def compute_chunk_size(
#     strategies: Sequence[ThresholdStrategy],
#     desired_sec: int = DESIRED_SEC_PER_CHUNK,
#     min_chunk: int = 1000,
#     max_chunk: int = 50000
# ) -> int:
#     """
#     Choose a chunk size so each batch takes about `desired_sec` seconds,
#     clamped between min_chunk and max_chunk.
#     """
#     gps = measure_throughput(strategies)
#     chunk = int(desired_sec * gps)
#     return max(min_chunk, min(chunk, max_chunk))

# ── Core simulation ──────────────────────────────────────────────────────────

def simulate_chunk(
    seeds: NDArray[np.int64],
    strategies: Sequence[ThresholdStrategy]
) -> Counter[str]:
    ctr: Counter[str] = Counter()
    for s in seeds.tolist():           # convert to Python int
        result = _play_game(int(s), strategies)
        winner_strat = result[f"{result['winner']}_strategy"]
        ctr[winner_strat] += 1
    return ctr


def run_chunk(shuffle_seeds: Sequence[int],
              strategies: Sequence[ThresholdStrategy]) -> Counter[str]:
    ctr = Counter()
    for sd in shuffle_seeds:
        ctr.update(play_shuffle(sd, strategies))
    return ctr


def play_shuffle(seed: int,
                 strategies: Sequence[ThresholdStrategy]
                 ) -> Counter[str]:
    """Plays *all* 1 632 games in one shuffle and returns a win Counter."""
    rng     = np.random.default_rng(seed)
    perm    = rng.permutation(np.asarray(strategies, dtype=object)) 
    wins    = Counter()

    game_seed_iter = rng.integers(0, 2**32 - 1,
                                  size=GAMES_PER_SHUFFLE).tolist()

    for i in range(0, len(perm), N_PLAYERS):
        table = perm[i : i + N_PLAYERS]
        row   = _play_game(game_seed_iter[i // N_PLAYERS], table)
        winner_strat = row[f"{row['winner']}_strategy"]
        wins[winner_strat] += 1

    return wins

# ── Main entry point ─────────────────────────────────────────────────────────
def run_tournament(
    global_seed: int,
    checkpoint_path: str,
    n_jobs: int | None = None
) -> None:
    """
    1) Load strategies
    2) Auto-tune a chunk size
    3) Spawn workers to run all games in parallel, checkpointing every 10 chunks
    4) Load or finalize the win counter
    """
    # load your strategies however you like
    strategies, _ = generate_strategy_grid()
    gps           = measure_throughput(strategies[:N_PLAYERS])  # quick probe
    shuffles_per_chunk = max(
        1,
        int(DESIRED_SEC_PER_CHUNK * gps // GAMES_PER_SHUFFLE)
    )

    # Pre-bake the seeds that define each shuffle
    master  = np.random.default_rng(global_seed)
    shuffle_seeds = master.integers(0, 2**32 - 1, size=NUM_SHUFFLES)

    # Cut into chunks of shuffles
    chunks = [
        shuffle_seeds[i : i + shuffles_per_chunk]
        for i in range(0, NUM_SHUFFLES, shuffles_per_chunk)
    ]

    # 4) set up logging & run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    win_counter: Counter[str] = Counter()

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {ex.submit(run_chunk, c, strategies): k
                for k, c in enumerate(chunks)}
        chunks_done = 0
        for fut in as_completed(futures):

            chunk_result = fut.result()
            win_counter.update(chunk_result)
            logging.info(
                f"total wins={sum(win_counter.values())}"
            )
            chunks_done += 1                # NEW  # noqa: SIM113 - generic counter, enumerate() overcomplicates
            if chunks_done % 10 == 0:       # every ten finished chunks
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(win_counter, f)
                logging.info("Checkpoint written (%d chunks).", chunks_done)


    # 5) final load or fallback
    chk = Path(checkpoint_path)
    if chk.exists():
        with chk.open("rb") as f:
            final_counter = pickle.load(f)
    else:
        final_counter = win_counter

    logging.info(f"Final win counts: {final_counter}")
    
    
def main(global_seed=0, checkpoint_path="checkpoint.pkl", n_jobs=16):
    run_tournament(global_seed, checkpoint_path, n_jobs)

# ── Script guard ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main(global_seed=0, checkpoint_path="checkpoint.pkl", n_jobs=16)
