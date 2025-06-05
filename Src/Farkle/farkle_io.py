import csv
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from farkle.simulation import _play_game
from farkle.strategies import ThresholdStrategy


# ------------------------------------------------------------
def _writer_worker(queue: mp.Queue, outpath: str, header: Sequence[str]) -> None:
    """Runs in its own process; pulls rows off queue and appends to CSV."""
    first = not Path(outpath).exists()
    with open(outpath, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        if first:
            w.writeheader()
        buffer = []
        while True:
            row = queue.get()
            if row is None:
                break
            buffer.append(row)
            if len(buffer) >= 10_000:
                w.writerows(buffer)
                fh.flush()
                buffer.clear()
        # after loop
        if buffer:
            w.writerows(buffer)

# ------------------------------------------------------------
def simulate_many_games_stream(
    *,
    n_games: int,
    strategies: Sequence[ThresholdStrategy],
    target_score: int = 10_000,
    out_csv: str = "winners.csv",
    seed: int | None = None,
    n_jobs: int = 1,
) -> None:
    """Runs games **without** building a huge DataFrame.

    Each finished game immediately lands as one row in *out_csv*:
        game_id, winner, winning_score, winner_strategy, n_rounds
    """
    master = np.random.default_rng(seed)
    seeds = master.integers(0, 2**32 - 1, size=n_games)

    # We will write only five tiny columns per game
    header = ["game_id", "winner", "winning_score", "winner_strategy", "n_rounds"]

    if n_jobs == 1:
        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            for gid, s in enumerate(seeds, 1):
                row = _single_game_row(gid, s, strategies, target_score)
                writer.writerow(row)
    else:
        queue: mp.Queue = mp.Queue(maxsize=2_000)
        writer = mp.Process(target=_writer_worker, args=(queue, out_csv, header))
        writer.start()

        # map-reduce style parallel play
        with mp.Pool(processes=n_jobs) as pool:
            for _, row in pool.imap_unordered(
                _single_game_row_mp,
                [(i + 1, sd, strategies, target_score) for i, sd in enumerate(seeds)],
                chunksize=50,
            ):
                queue.put(row)

        queue.put(None)    # poison pill
        writer.join()


# ------- helpers shared by both execution paths --------------------------
def _single_game_row(
    game_id: int,
    seed: int,
    strategies: Sequence[ThresholdStrategy],
    target_score: int,
) -> Dict[str, Any]:
    gm = _play_game(seed, strategies, target_score)       # re-use existing helper
    winner_name = gm["winner"]
    winner_strategy = gm[f"{winner_name}_strategy"]
    return {
        "game_id": game_id,
        "winner": winner_name,
        "winning_score": gm["winning_score"],
        "winner_strategy": winner_strategy,
        "n_rounds": gm["n_rounds"],
    }

# pickle-friendly wrapper for mp.Pool
def _single_game_row_mp(args_tuple):
    return args_tuple[0], _single_game_row(*args_tuple)