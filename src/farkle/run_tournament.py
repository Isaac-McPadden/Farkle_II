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
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, cast

import numpy as np

from farkle.simulation import _play_game, generate_strategy_grid
from farkle.strategies import ThresholdStrategy

# ---------------------------------------------------------------------------
# Configuration constants (patched by tests/CLI)
# ---------------------------------------------------------------------------
N_PLAYERS: int = 5  # default – can be overwritten at runtime
NUM_SHUFFLES: int = 10_223
GAMES_PER_SHUFFLE: int = 8_160 // N_PLAYERS  # 1 632
DESIRED_SEC_PER_CHUNK: int = 10
CKPT_EVERY_SEC: int = 30

# ---------------------------------------------------------------------------
# Dataclass configuration
# ---------------------------------------------------------------------------


@dataclass
class TournamentConfig:
    """Runtime configuration for :func:`run_tournament`."""

    n_players: int = N_PLAYERS
    num_shuffles: int = NUM_SHUFFLES
    desired_sec_per_chunk: int = DESIRED_SEC_PER_CHUNK
    ckpt_every_sec: int = CKPT_EVERY_SEC

    @property
    def games_per_shuffle(self) -> int:
        return 8_160 // self.n_players


# metric fields tracked per winning strategy
METRIC_LABELS: Tuple[str, ...] = (
    "winning_score",
    "n_rounds",
    "winner_farkles",
    "winner_rolls",
    "winner_highest_turn",
    "winner_smart_five_uses",
    "winner_n_smart_five_dice",
    "winner_smart_one_uses",
    "winner_n_smart_one_dice",
    "winner_hot_dice",
)


def _extract_winner_metrics(row: Mapping[str, Any], winner: str) -> List[int]:
    """Return the metric vector for the winning player."""

    return [
        row["winning_score"],
        row["n_rounds"],
        row[f"{winner}_farkles"],
        row[f"{winner}_rolls"],
        row[f"{winner}_highest_turn"],
        row[f"{winner}_smart_five_uses"],
        row[f"{winner}_n_smart_five_dice"],
        row[f"{winner}_smart_one_uses"],
        row[f"{winner}_n_smart_one_dice"],
        row[f"{winner}_hot_dice"],
    ]


# ---------------------------------------------------------------------------
# Worker initialisation and helpers
# ---------------------------------------------------------------------------
_STRATS: List[ThresholdStrategy] = []
_CFG: TournamentConfig = TournamentConfig()


def _init_worker(
    strategies: Sequence[ThresholdStrategy],
    config: TournamentConfig,
) -> None:
    """Initialise per-process globals.

    Parameters
    ----------
    strategies : Sequence[ThresholdStrategy]
        Strategy objects to copy into the worker.
    config: TournamentConfig
        Config rules sent with worker.

    Returns
    -------
    None
    """

    global _STRATS, _CFG, N_PLAYERS, GAMES_PER_SHUFFLE
    if 8_160 % config.n_players != 0:
        raise ValueError("n_players must divide 8,160")
    _STRATS = list(strategies)
    _CFG = config
    N_PLAYERS = config.n_players
    GAMES_PER_SHUFFLE = config.games_per_shuffle


# ---------------------------------------------------------------------------
# Shuffle-level helpers
# ---------------------------------------------------------------------------


def _play_one_shuffle(
    seed: int, *, collect_rows: bool = False
) -> Tuple[
    Counter[str],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
    List[Dict[str, Any]],
]:
    """Play all games for one shuffle and aggregate the results."""

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

        row = _play_game(int(gseed), [_STRATS[i] for i in idxs])
        winner = row["winner"]
        strat_repr = row[f"{winner}_strategy"]
        metrics = _extract_winner_metrics(row, winner)
        wins[strat_repr] += 1
        for label, value in zip(METRIC_LABELS, metrics, strict=True):
            sums[label][strat_repr] += value
            sq_sums[label][strat_repr] += value * value
        if collect_rows:
            rows.append({"game_seed": int(gseed), **row})

    return wins, sums, sq_sums, rows


# Legacy helper retained for unit tests --------------------------------------


def _play_shuffle(seed: int) -> Counter[str]:
    """Compatibility wrapper returning only win counts for one shuffle."""

    wins, _, _, _ = _play_one_shuffle(seed, collect_rows=False)
    return wins


def _run_chunk(shuffle_seed_batch: Sequence[int]) -> Counter[str]:
    """Play a batch of shuffles and tally wins.

    Parameters
    ----------
    shuffle_seed_batch : Sequence[int]
        RNG seeds for each shuffle processed by this worker.

    Returns
    -------
    collections.Counter[str]
        Mapping of strategy strings to win counts for the batch.
    """

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
    """Play shuffles and accumulate metrics.

    Parameters
    ----------
    shuffle_seed_batch : Sequence[int]
        RNG seeds for each shuffle in this batch.
    collect_rows : bool, default False
        If ``True`` return and optionally persist full per-game rows.
    row_dir : Path | None, default None
        Directory used to write parquet files when ``collect_rows`` is ``True``.

    Returns
    -------
    tuple
        ``(wins, sums, square_sums)`` where each element aggregates the
        respective values over the batch.

    Side Effects
    ------------
    When ``collect_rows`` is ``True`` a parquet file containing all rows for the
    current worker is written to ``row_dir``.
    """

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
        except ImportError:  # pragma: no cover - optional dependency
            logging.warning("pyarrow not installed - row logging skipped")
        else:
            assert row_dir is not None
            out = row_dir / f"rows_{mp.current_process().pid}_{time.time_ns()}.parquet"
            pq.write_table(pa.Table.from_pylist(all_rows), out)

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
    sums: Mapping[str, Mapping[str, float]] | None,
    sq_sums: Mapping[str, Mapping[str, float]] | None,
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
    config: TournamentConfig | None = None,
    global_seed: int = 0,
    checkpoint_path: Path | str = "checkpoint.pkl",
    n_jobs: int | None = None,
    collect_metrics: bool = False,
    row_output_directory: Path | None = None,  # None if --row-dir omitted
    num_shuffles: int = NUM_SHUFFLES,  # noqa: ARG001
) -> None:
    """Run a multi-process Monte-Carlo Farkle tournament.

    Parameters
    ----------
    config : TournamentConfig, optional
        Encapsulates all tunable constants (number of players, shuffle count,
        checkpoint cadence, etc.). If ``None`` a default instance is created
        from the module-level constants.
    global_seed : int, default 0
        Seed for the master RNG that generates per-shuffle seeds -- make this
        different to obtain a fresh tournament.
    checkpoint_path : str | pathlib.Path, default "checkpoint.pkl"
        Location for periodic checkpoint pickles. Parent directories are
        created automatically.
    n_jobs : int | None, default None
        Worker processes to spawn. ``None`` lets
        :class:`~concurrent.futures.ProcessPoolExecutor` decide
        (usually ``os.cpu_count()``).
    collect_metrics : bool, default False
        If ``True``, per-strategy means/variances for several game metrics are
        accumulated in addition to raw win counts.
    row_output_directory : pathlib.Path | None, default None
        When supplied, every worker writes complete per-game rows to this
        directory as Parquet files (requires *pyarrow*).

    Notes
    -----
    *Old keyword arguments such as ``num_shuffles`` and ``ckpt_every_sec`` are
    now fields of :class:`TournamentConfig`. Provide a custom config if you need
    to override them.*

    Side Effects
    ------------
    Creates/updates ``checkpoint_path`` and, if ``row_output_directory`` is
    given, a set of Parquet files containing raw game rows.
    """
    strategies, _ = generate_strategy_grid()  # 8_160 strategies

    cfg = config or TournamentConfig()
    if num_shuffles != cfg.num_shuffles:
        cfg.num_shuffles = num_shuffles
    if cfg.n_players < 2:
        raise ValueError("n_players must be ≥2")

    games_per_sec = _measure_throughput(strategies[: cfg.n_players])
    shuffles_per_chunk = max(
        1,
        int(cfg.desired_sec_per_chunk * games_per_sec // cfg.games_per_shuffle),
    )

    master_rng = np.random.default_rng(global_seed)
    shuffle_seeds = master_rng.integers(0, 2**32 - 1, size=cfg.num_shuffles).tolist()
    chunks = [
        shuffle_seeds[i : i + shuffles_per_chunk]
        for i in range(0, cfg.num_shuffles, shuffles_per_chunk)
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
        assert row_output_directory is not None
        row_output_directory.mkdir(parents=True, exist_ok=True)

    if collect_metrics or collect_rows:
        chunk_fn: Callable[[Sequence[int]], object] = partial(
            _run_chunk_metrics, collect_rows=collect_rows, row_dir=row_output_directory
        )
    else:
        chunk_fn = _run_chunk

    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_init_worker,
        initargs=(strategies, cfg),
    ) as pool:
        futures = [pool.submit(chunk_fn, c) for c in chunks]

        for done, fut in enumerate(as_completed(futures), 1):
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
            if now - last_ckpt >= cfg.ckpt_every_sec:
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
        "--num-shuffles",
        type=int,
        default=NUM_SHUFFLES,
        help="number of shuffles to simulate",
    )
    p.add_argument(
        "--row-dir",
        type=Path,
        metavar="DIR",
        help="write full per-game rows to DIR as parquet",
    )
    args = p.parse_args()

    cfg = TournamentConfig(
        n_players=N_PLAYERS,
        num_shuffles=NUM_SHUFFLES,
        desired_sec_per_chunk=DESIRED_SEC_PER_CHUNK,
        ckpt_every_sec=args.ckpt_sec,
    )

    run_tournament(
        config=cfg,
        global_seed=args.seed,
        checkpoint_path=args.checkpoint,
        n_jobs=args.jobs,
        collect_metrics=args.metrics,
        row_output_directory=args.row_dir,
        num_shuffles=args.num_shuffles,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
