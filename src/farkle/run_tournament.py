# src/farkle/run_tournament.py
"""Parallel Monte-Carlo tournament driver.

This version keeps the original fast win-count loop used in the unit tests but
adds optional collection of richer statistics. When enabled the worker
processes accumulate running sums and sum-of-squares for a small set of metrics
so that per-strategy means and variances can be computed without storing every
row.  A parquet dump of all rows can also be requested via --row-dir.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import pickle
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

import numpy as np

from farkle.simulation import _play_game, generate_strategy_grid
from farkle.strategies import ThresholdStrategy

# ---------------------------------------------------------------------------
# Configuration constants (patched by tests/CLI)
# ---------------------------------------------------------------------------
N_PLAYERS: int = 5
NUM_SHUFFLES: int = 10_223
GAMES_PER_SHUFFLE: int = 8_160 // N_PLAYERS  # 1 632
DESIRED_SEC_PER_CHUNK: int = 10
CKPT_EVERY_SEC: int = 30

# metric fields tracked per winning strategy
METRIC_LABELS: Tuple[str, ...] = (
    "winning_score",
    "n_rounds",
    "winner_farkles",
    "winner_rolls",
    "winner_highest_turn",
)

# ---------------------------------------------------------------------------
# Worker initialisation and helpers
# ---------------------------------------------------------------------------
_STRATS: List[ThresholdStrategy] = []


def _init_worker(strategies: Sequence[ThresholdStrategy]) -> None:
    """Store the strategy list in each worker process."""

    global _STRATS
    _STRATS = list(strategies)


def _play_single_game(seed: int, strat_indices: Sequence[int]) -> Tuple[str, List[int]]:
    """Run one game and return the winning strategy string and metrics."""

    table = [_STRATS[i] for i in strat_indices]
    res: Dict[str, Any] = _play_game(seed, table)

    winner = res["winner"]
    strat_repr = res[f"{winner}_strategy"]
    metrics = [
        res["winning_score"],
        res["n_rounds"],
        res[f"{winner}_farkles"],
        res[f"{winner}_rolls"],
        res[f"{winner}_highest_turn"],
    ]
    return strat_repr, metrics


# ---------------------------------------------------------------------------
# Shuffle-level helpers
# ---------------------------------------------------------------------------


def _play_one_shuffle(seed: int, *, collect_rows: bool = False) -> Tuple[
    Counter[str],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    List[Dict[str, Any]],
]:
    """Play GAMES_PER_SHUFFLE games and aggregate the results."""

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(_STRATS))
    game_seeds = rng.integers(0, 2**32 - 1, size=GAMES_PER_SHUFFLE)

    wins: Counter[str] = Counter()
    sums: Dict[str, Dict[str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    sq_sums: Dict[str, Dict[str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    rows: List[Dict[str, Any]] = []

    offset = 0
    for gseed in game_seeds.tolist():
        idxs = perm[offset : offset + N_PLAYERS].tolist()
        offset += N_PLAYERS

        strat_repr, metrics = _play_single_game(int(gseed), idxs)
        wins[strat_repr] += 1
        for label, value in zip(METRIC_LABELS, metrics, strict=True):
            sums[label][strat_repr] += value
            sq_sums[label][strat_repr] += value * value
        if collect_rows:
            rows.append(
                {
                    "game_seed": int(gseed),
                    "winner_strategy": strat_repr,
                    **dict(zip(METRIC_LABELS, metrics, strict=True)),
                }
            )

    return wins, sums, sq_sums, rows


# Legacy helper retained for unit tests --------------------------------------


def _play_shuffle(seed: int) -> Counter[str]:
    """Compatibility wrapper returning only win counts for one shuffle."""

    wins, _, _, _ = _play_one_shuffle(seed, collect_rows=False)
    return wins


def _run_chunk(shuffle_seed_batch: Sequence[int]) -> Counter[str]:
    """Aggregate win counts for a batch of shuffles (legacy behaviour)."""

    total: Counter[str] = Counter()
    for sd in shuffle_seed_batch:
        total.update(_play_shuffle(int(sd)))
    return total


# Rich metrics variant -------------------------------------------------------


def _run_chunk_metrics(
    shuffle_seed_batch: Sequence[int],
    *,
    collect_rows: bool = False,
    row_dir: Path | None = None,
) -> Tuple[
    Counter[str],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    """Same as :func:`_run_chunk` but also accumulates metric sums."""

    wins_total: Counter[str] = Counter()
    sums_total: Dict[str, Dict[str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    sq_total: Dict[str, Dict[str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    all_rows: List[Dict[str, Any]] = []

    for seed in shuffle_seed_batch:
        w, s, sq, rows = _play_one_shuffle(int(seed), collect_rows=collect_rows)
        wins_total.update(w)
        for label in METRIC_LABELS:
            for k, v in s[label].items():
                sums_total[label][k] += v
            for k, v in sq[label].items():
                sq_total[label][k] += v
        if collect_rows:
            all_rows.extend(rows)

    if collect_rows and all_rows:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            assert row_dir is not None
            out = row_dir / f"rows_{mp.current_process().pid}_{time.time_ns()}.parquet"
            pq.write_table(pa.Table.from_pylist(all_rows), out)
        except Exception:  # pragma: no cover - optional dependency
            logging.warning("pyarrow not installed - row logging skipped")

    return wins_total, sums_total, sq_total


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _measure_throughput(
    sample_strategies: Sequence[ThresholdStrategy],
    sample_games: int = 2_000,
    seed: int = 0,
) -> float:
    """Quick benchmark returning games processed per second."""

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**32 - 1, size=sample_games)
    start = time.perf_counter()
    for gs in seeds.tolist():
        _play_game(int(gs), sample_strategies)
    return sample_games / (time.perf_counter() - start)


def _save_checkpoint(
    path: Path,
    wins: Counter[str],
    sums: Dict[str, Dict[str, float]] | None,
    sq_sums: Dict[str, Dict[str, float]] | None,
) -> None:
    """Pickle the current aggregates to path."""

    payload: Dict[str, Any] = {"win_totals": wins}
    if sums is not None and sq_sums is not None:
        payload["metric_sums"] = sums
        payload["metric_square_sums"] = sq_sums
    path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_tournament(
    *,
    global_seed: int = 0,
    checkpoint_path: Path | str = "checkpoint.pkl",
    n_jobs: int | None = None,
    ckpt_every_sec: int = CKPT_EVERY_SEC,
    collect_metrics: bool = False,
    row_output_directory: Path | None = None, # None if --row-dir omitted
) -> None:
    """Orchestrate the multi-process tournament."""

    strategies, _ = generate_strategy_grid()  # 8 160 strategies

    games_per_sec = _measure_throughput(strategies[:N_PLAYERS])
    shuffles_per_chunk = max(1, int(DESIRED_SEC_PER_CHUNK * games_per_sec // GAMES_PER_SHUFFLE))

    master_rng = np.random.default_rng(global_seed)
    shuffle_seeds = master_rng.integers(0, 2**32 - 1, size=NUM_SHUFFLES).tolist()
    chunks = [
        shuffle_seeds[i : i + shuffles_per_chunk]
        for i in range(0, NUM_SHUFFLES, shuffles_per_chunk)
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    win_totals: Counter[str] = Counter()
    metric_sums: Dict[str, Dict[str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    metric_sq_sums: Dict[str, Dict[str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}

    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    last_ckpt = time.perf_counter()

    collect_rows = row_output_directory is not None
    if collect_rows:
        row_output_directory.mkdir(parents=True, exist_ok=True)

    if collect_metrics or collect_rows:
        chunk_fn = partial(
            _run_chunk_metrics, collect_rows=collect_rows, row_dir=row_output_directory
        )
    else:
        chunk_fn = _run_chunk

    with ProcessPoolExecutor(
        max_workers=n_jobs, initializer=_init_worker, initargs=(strategies,)
    ) as pool:
        future_to_index = {pool.submit(chunk_fn, c): i for i, c in enumerate(chunks)}

        for done, fut in enumerate(as_completed(future_to_index), 1):
            result = fut.result()
            if collect_metrics or collect_rows:
                wins, sums, sqs = cast(
                    Tuple[Counter[str], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]],
                    result,
                )
                win_totals.update(wins)
                for label in METRIC_LABELS:
                    for k, v in sums[label].items():
                        metric_sums[label][k] += v
                    for k, v in sqs[label].items():
                        metric_sq_sums[label][k] += v
            else:
                win_totals.update(cast(Counter[str], result))

            now = time.perf_counter()
            if now - last_ckpt >= ckpt_every_sec:
                _save_checkpoint(
                    ckpt_path,
                    win_totals,
                    metric_sums if collect_metrics or collect_rows else None,
                    metric_sq_sums if collect_metrics or collect_rows else None,
                )
                logging.info(
                    "checkpoint … %d/%d chunks, %d games",
                    done,
                    len(chunks),
                    sum(win_totals.values()),
                )
                last_ckpt = now

    _save_checkpoint(
        ckpt_path,
        win_totals,
        metric_sums if collect_metrics or collect_rows else None,
        metric_sq_sums if collect_metrics or collect_rows else None,
    )
    logging.info("finished - %d games", sum(win_totals.values()))


# ---------------------------------------------------------------------------
# Command line entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mp.set_start_method("spawn", force=True)

    p = argparse.ArgumentParser(description="Run a Monte-Carlo Farkle tournament")
    p.add_argument("--seed", type=int, default=0, help="global RNG seed")
    p.add_argument("--checkpoint", type=Path, default="checkpoint.pkl", help="pickle output")
    p.add_argument("--jobs", type=int, default=None, help="worker processes")
    p.add_argument("--ckpt-sec", type=int, default=CKPT_EVERY_SEC, help="seconds between saves")
    p.add_argument(
        "--metrics",
        action="store_true",
        help="collect per-strategy means/variances",
    )
    p.add_argument(
        "--row-dir",
        type=Path,
        metavar="DIR",
        help="write full per-game rows to DIR as parquet",
    )
    args = p.parse_args()

    run_tournament(
        global_seed=args.seed,
        checkpoint_path=args.checkpoint,
        n_jobs=args.jobs,
        ckpt_every_sec=args.ckpt_sec,
        collect_metrics=args.metrics,
        row_output_directory=args.row_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

# from __future__ import annotations

# import logging
# import multiprocessing as mp
# import pickle
# import time
# from collections import Counter
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from pathlib import Path
# from typing import List, Sequence

# import numpy as np

# from farkle.simulation import _play_game, generate_strategy_grid
# from farkle.strategies import ThresholdStrategy

# # ─────────────────────────────────────────────────────────────────────────────
# # 0.  Globals & tuning knobs  (can be patched from tests or CLI)
# # ─────────────────────────────────────────────────────────────────────────────
# N_PLAYERS             = 5
# NUM_SHUFFLES          = 10_223
# GAMES_PER_SHUFFLE     = 8_160 // N_PLAYERS          # 1 632
# DESIRED_SEC_PER_CHUNK = 10
# CKPT_EVERY_SEC        = 30                          # default wall-clock cadence
# LOG_TS_FMT            = "%Y-%m-%d %H:%M:%S"

# # ─────────────────────────────────────────────────────────────────────────────
# # 1.  Fast helpers - strategy list lives in a module-level var in each worker
# # ─────────────────────────────────────────────────────────────────────────────
# _STRATS: List[ThresholdStrategy] = []               # filled by _init_worker


# def _init_worker(strategies: Sequence[ThresholdStrategy]) -> None:
#     """Initialise the worker process with a strategy list.

#     Inputs
#     ------
#     strategies : Sequence[ThresholdStrategy]
#         Strategy objects created in the parent process.

#     Returns
#     -------
#     None
#         _STRATS is populated for fast lookups in the worker.
#     """
#     global _STRATS
#     _STRATS = list(strategies)                      # zero-copy index look-ups


# def _play_table(seed: int, idxs: Sequence[int]) -> str:
#     """Run one table of five players using stored strategies.

#     Inputs
#     ------
#     seed : int
#         RNG seed for the game engine.
#     idxs : Sequence[int]
#         Indices into the global _STRATS list.

#     Returns
#     -------
#     str
#         Strategy string representation of the winning player.
#     """
#     table = [_STRATS[i] for i in idxs]
#     res   = _play_game(seed, table)
#     return res[f"{res['winner']}_strategy"]          # string repr → Counter key


# def _play_shuffle(seed: int) -> Counter[str]:
#     """Play all games that comprise a single shuffle.

#     Inputs
#     ------
#     seed : int
#         Seed used to create permutations and per-game seeds.

#     Returns
#     -------
#     Counter[str]
#         Mapping of strategy strings to win counts for this shuffle.
#     """
#     rng    = np.random.default_rng(seed)
#     perm   = rng.permutation(len(_STRATS))
#     seeds  = rng.integers(0, 2**32 - 1, size=GAMES_PER_SHUFFLE)

#     wins: Counter[str] = Counter()
#     off = 0
#     for s in seeds.tolist():                        # Ruff: prefer list[int]
#         idxs = perm[off : off + N_PLAYERS].tolist()
#         wins[_play_table(int(s), idxs)] += 1
#         off += N_PLAYERS
#     return wins


# def _run_chunk(shuffle_seed_slice: Sequence[int]) -> Counter[str]:
#     """Aggregate multiple shuffles in one worker call.

#     Inputs
#     ------
#     shuffle_seed_slice : Sequence[int]
#         Collection of seeds, one for each shuffle to execute.

#     Returns
#     -------
#     Counter[str]
#         Combined win counts across all shuffles in shuffle_seed_slice.
#     """
#     ctr: Counter[str] = Counter()
#     for sd in shuffle_seed_slice:
#         ctr.update(_play_shuffle(int(sd)))
#     return ctr


# def _measure_throughput(strats: Sequence[ThresholdStrategy],
#                         test_games: int = 2_000,
#                         seed: int = 0) -> float:
#     """Estimate processing speed in games per second.

#     Inputs
#     ------
#     strats : Sequence[ThresholdStrategy]
#         Strategy objects used in the benchmark games.
#     test_games : int, default 2000
#         Number of games to run for the estimate.
#     seed : int, default 0
#         RNG seed for reproducibility.

#     Returns
#     -------
#     float
#         Approximate games processed per second.
#     """
#     rng   = np.random.default_rng(seed)
#     seeds = rng.integers(0, 2**32 - 1, size=test_games)
#     t0    = time.perf_counter()
#     for s in seeds.tolist():
#         _play_game(int(s), strats[:N_PLAYERS])
#     return test_games / (time.perf_counter() - t0)

# # ─────────────────────────────────────────────────────────────────────────────
# # 2.  Main driver
# # ─────────────────────────────────────────────────────────────────────────────
# def run_tournament(
#     *,
#     global_seed: int = 0,
#     checkpoint_path: str | Path = "checkpoint.pkl",
#     n_jobs: int | None = None,
#     ckpt_every_sec: int = CKPT_EVERY_SEC,
#     ) -> None:
#     """Run the full tournament across many worker processes.

#     Inputs
#     ------
#     global_seed : int, default 0
#         Seed controlling the overall reproducibility of the tournament.
#     checkpoint_path : str | Path, default "checkpoint.pkl"
#         File path to which intermediate results are periodically saved.
#     n_jobs : int | None, optional
#         Number of worker processes, None uses CPU count.
#     ckpt_every_sec : int, default CKPT_EVERY_SEC
#         Seconds between checkpoint writes.

#     Returns
#     -------
#     None
#         Progress is logged and the final win counts are written to checkpoint_path.
#     """
#     strategies, _ = generate_strategy_grid()                 # 8 160 objects
#     gps = _measure_throughput(strategies[:N_PLAYERS])
#     shuffles_per_chunk = max(1,
#         int(DESIRED_SEC_PER_CHUNK * gps // GAMES_PER_SHUFFLE))

#     master        = np.random.default_rng(global_seed)
#     shuffle_seeds = master.integers(0, 2**32 - 1,
#                                     size=NUM_SHUFFLES).tolist()

#     chunks = [
#         shuffle_seeds[i : i + shuffles_per_chunk]
#         for i in range(0, NUM_SHUFFLES, shuffles_per_chunk)
#     ]

#     # --- logging ---------------------------------------------------------
#     logging.basicConfig(level=logging.INFO,
#                         format="%(asctime)s  %(message)s",
#                         datefmt="%H:%M:%S")

#     win_totals: Counter[str] = Counter()
#     last_save = time.perf_counter()                 # wall-clock book-keeping
#     ckpt_path = Path(checkpoint_path)
#     ckpt_path.parent.mkdir(parents=True, exist_ok=True)

#     with ProcessPoolExecutor(max_workers=n_jobs,
#                              initializer=_init_worker,
#                              initargs=(strategies,)) as pool:

#         fut2idx = {pool.submit(_run_chunk, c): i for i, c in enumerate(chunks)}
#         for done, fut in enumerate(as_completed(fut2idx), 1):
#             win_totals.update(fut.result())

#             now = time.perf_counter()
#             if now - last_save >= ckpt_every_sec:
#                 ckpt_path.write_bytes(pickle.dumps(win_totals,
#                                                    protocol=pickle.HIGHEST_PROTOCOL))
#                 ts = time.strftime(LOG_TS_FMT, time.localtime())
#                 logging.info("%s  %5d/%d chunks  %d games played",
#                              ts, done, len(chunks), sum(win_totals.values()))
#                 last_save = now

#     ckpt_path.write_bytes(pickle.dumps(win_totals,
#                                        protocol=pickle.HIGHEST_PROTOCOL))
#     logging.info("All done! Final win counts:\n%s", win_totals)


# # ─────────────────────────────────────────────────────────────────────────────
# # 3.  CLI convenience - lets user override checkpoint cadence
# # ─────────────────────────────────────────────────────────────────────────────
# def main(
#     global_seed: int = 0,
#     checkpoint_path: str | Path = "checkpoint.pkl",
#     n_jobs: int = 16,
#     ckpt_every_sec: int = CKPT_EVERY_SEC,
# ) -> None:
#     """main()

#     Inputs
#     ------
#     global_seed : int, default 0
#         Master seed for reproducibility.
#     checkpoint_path : str | Path, default "checkpoint.pkl"
#         Location to store tournament checkpoints.
#     n_jobs : int, default 16
#         Number of worker processes to spawn.
#     ckpt_every_sec : int, default CKPT_EVERY_SEC
#         Seconds between checkpoint writes.

#     Returns
#     -------
#     None
#         The function runs run_tournament with the provided parameters.
#     """
#     mp.set_start_method("spawn", force=True)
#     run_tournament(global_seed=global_seed,
#                    checkpoint_path=checkpoint_path,
#                    n_jobs=n_jobs,
#                    ckpt_every_sec=ckpt_every_sec)


# if __name__ == "__main__":
#     main()
