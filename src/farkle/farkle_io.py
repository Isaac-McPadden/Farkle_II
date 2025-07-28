# src/farkle/farkle_io.py
"""Collection of read/write data helper functions available for project-wide use"""
import csv
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from farkle.simulation import _play_game
from farkle.strategies import ThresholdStrategy

# Batching and queue sizes for file I/O
BUFFER_SIZE = 10_000
QUEUE_SIZE = 2_000


# ------------------------------------------------------------
def _writer_worker(
    queue: mp.Queue,
    out_csv: str,
    header: Sequence[str],
) -> None:
    """Summary: write queued rows to ``out_csv`` in a separate process.

    Rows are buffered until :data:`BUFFER_SIZE` rows accumulate before
    being flushed to disk.

    Inputs:
        queue: multiprocessing.Queue containing row dictionaries and a
            None sentinel to stop the worker.
        out_csv: Destination CSV file.
        header: Column names for the csv.DictWriter.

    Returns:
        None. Rows are appended to ``out_csv`` until the sentinel is
        received.
    """
    first = True
    with open(out_csv, "a", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=header)
        if first:
            writer.writeheader()
        buffer = []
        while True:
            row = queue.get()
            if row is None:
                break
            buffer.append(row)
            if len(buffer) >= BUFFER_SIZE:
                writer.writerows(buffer)
                file_handle.flush()
                buffer.clear()
        # after loop
        if buffer:
            writer.writerows(buffer)


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
        out_csv: Path to the output CSV file. The file will be
            overwritten at the start of the run. Its parent directory
            must exist or will be created.
        seed: Optional seed for deterministic runs.
        n_jobs: Number of processes to use; 1 runs serially. When greater than
            one, results are sent through a queue limited to
            :data:`QUEUE_SIZE` items.

    Returns:
        None. Metrics for each game are written incrementally to
        out_csv.
    """
    master = np.random.default_rng(seed)
    seeds = master.integers(0, 2**32 - 1, size=n_games)

    # ensure target directory exists
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    # We will write only five tiny columns per game
    header = [
        "game_id",
        "winner",
        "winning_score",
        "winner_strategy",
        "n_rounds",
    ]

    if n_jobs == 1:
        # open once, write header, and stream rows serially
        with open(out_csv, "w", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=header)
            writer.writeheader()
            for gid, s in enumerate(seeds, 1):
                row = _single_game_row(gid, int(s), strategies, target_score)
                writer.writerow(row)
    else:
        # truncate the file – the writer-process will write the header once
        open(out_csv, "w").close()

        queue: mp.Queue = mp.Queue(maxsize=QUEUE_SIZE)
        writer_process = mp.Process(
            target=_writer_worker,
            args=(queue, out_csv, header),
        )
        writer_process.start()

        # map-reduce style parallel play
        with mp.Pool(processes=n_jobs) as pool:
            for _, row in pool.imap_unordered(
                _single_game_row_mp,
                [
                    (i + 1, int(sd), strategies, target_score)
                    for i, sd in enumerate(seeds)
                ],
                chunksize=50,
            ):
                queue.put(row)

        queue.put(None)  # poison pill
        writer_process.join()


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
def _single_game_row_mp(
    args_tuple: tuple[int, int, Sequence[ThresholdStrategy], int],
) -> tuple[int, Dict[str, Any]]:
    """
    Multiprocessing helper that forwards to ``_single_game_row``.

    Inputs:
        args_tuple: tuple[int, int, Sequence[ThresholdStrategy], int]
            ``(game_id, seed, strategies, target_score)``.

    Returns:
        tuple[int, Dict[str, Any]] containing the game id and the row
        dictionary produced by ``_single_game_row``.
    """
    return args_tuple[0], _single_game_row(*args_tuple)
