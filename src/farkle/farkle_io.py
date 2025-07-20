import csv
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from farkle.simulation import _play_game
from farkle.strategies import ThresholdStrategy


# ------------------------------------------------------------
def _writer_worker(queue: mp.Queue, outpath: str, header: Sequence[str]) -> None:
    """Summary: write queued rows to outpath in a separate process.

    Inputs:
        queue: multiprocessing.Queue containing row dictionaries and a
            None sentinel to stop the worker.
        outpath: Destination CSV file.
        header: Column names for the csv.DictWriter.

    Returns:
        None. Rows are appended to outpath until the sentinel is
        received.
    """
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

    Inputs:
        n_games: Number of games to simulate.
        strategies: Strategies to assign to the players.
        target_score: Score needed to trigger the final round.
        out_csv: Path to the output CSV file.
        seed: Optional seed for deterministic runs.
        n_jobs: Number of processes to use; 1 runs serially.

    Returns:
        None. Metrics for each game are written incrementally to
        out_csv.
    """
    master = np.random.default_rng(seed)
    seeds = master.integers(0, 2**32 - 1, size=n_games)

    # We will write only five tiny columns per game
    header = ["game_id", "winner", "winning_score", "winner_strategy", "n_rounds"]

    # --- truncate file & write header once, upfront --------------------
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()

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

        queue.put(None)  # poison pill
        writer.join()


# ------- helpers shared by both execution paths --------------------------
def _single_game_row(
    game_id: int,
    seed: int,
    strategies: Sequence[ThresholdStrategy],
    target_score: int,
) -> Dict[str, Any]:
    """Summary: play one game and format metrics for CSV output.

    Inputs:
        game_id: Sequential identifier for the game.
        seed: Random seed used for the game's RNGs.
        strategies: Strategies applied to the players.
        target_score: Score required to win the game.

    Returns:
        Mapping of column names to values for a single row of the
        simulate_many_games_stream output.
    """
    gm = _play_game(seed, strategies, target_score)  # re-use existing helper
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
    """
    multiprocessing helper that forwards to _single_game_row.

    Inputs:
        args_tuple: Tuple (game_id, seed, strategies, target_score) as
            expected by _single_game_row.

    Returns:
        (game_id, row_dict) so that the game id survives pool ordering.
    """
    return args_tuple[0], _single_game_row(*args_tuple)
