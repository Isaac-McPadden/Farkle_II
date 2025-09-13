"""Parallel execution helpers.

The :func:`simulate_many_games_stream` function mirrors the behaviour of
the old ``farkle_io`` module.  It coordinates a pool of worker processes
that play games and stream the results to a CSV file without ever
building an intermediate :class:`pandas.DataFrame`.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from farkle.simulation.simulation import _play_game
from farkle.simulation.strategies import ThresholdStrategy

from .files import QUEUE_SIZE, _writer_worker

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_many_games_stream(
    *,
    n_games: int,
    strategies: Sequence[ThresholdStrategy],
    target_score: int = 10_000,
    out_csv: str = "winners.csv",
    seed: int | None = None,
    n_jobs: int = 1,
) -> None:
    """Stream ``n_games`` games to ``out_csv``.

    This variant of the simulator writes each completed game directly to a
    CSV file.  When ``n_jobs`` is greater than one it uses a
    multiprocessing pool and a separate writer process to avoid GIL
    contention.
    """

    master = np.random.default_rng(seed)
    seeds = master.integers(0, 2**32 - 1, size=n_games)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    header = [
        "game_id",
        "winner",
        "winning_score",
        "winner_strategy",
        "n_rounds",
    ]

    if n_jobs == 1:
        with open(out_csv, "w", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=header)
            writer.writeheader()
            for gid, s in enumerate(seeds, 1):
                row = _single_game_row(gid, int(s), strategies, target_score)
                writer.writerow(row)
    else:
        Path(out_csv).unlink(missing_ok=True)

        queue: mp.Queue = mp.Queue(maxsize=QUEUE_SIZE)
        writer_process = mp.Process(
            target=_writer_worker, args=(queue, out_csv, header)
        )
        writer_process.start()

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

        queue.put(None)
        writer_process.join()


# ---------------------------------------------------------------------------
# Helpers shared by serial and parallel execution paths
# ---------------------------------------------------------------------------


def _single_game_row(
    game_id: int,
    seed: int,
    strategies: Sequence[ThresholdStrategy],
    target_score: int,
) -> Dict[str, Any]:
    """Play one game and format metrics for CSV output."""

    gm = _play_game(seed, strategies, target_score)
    winner_name = gm["winner"]
    winner_strategy = gm[f"{winner_name}_strategy"]
    return {
        "game_id": game_id,
        "winner": winner_name,
        "winning_score": gm["winning_score"],
        "winner_strategy": winner_strategy,
        "n_rounds": gm["n_rounds"],
    }


def _single_game_row_mp(
    args_tuple: Tuple[int, int, Sequence[ThresholdStrategy], int]
) -> Tuple[int, Dict[str, Any]]:
    """Multiprocessing wrapper around :func:`_single_game_row`."""

    return args_tuple[0], _single_game_row(*args_tuple)


__all__ = ["simulate_many_games_stream"]

