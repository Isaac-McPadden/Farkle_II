# src/farkle/run_tournament.py
"""Parallel Monte-Carlo tournament driver.

This version keeps the original fast win-count loop used in the unit tests but
adds optional collection of richer statistics. When enabled the worker
processes accumulate running sums and sum-of-squares for a small set of metrics
so that per-strategy means and variances can be computed without storing every
row.  A parquet dump of all rows can also be requested via --row-dir.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from os import getpid
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, cast

import numpy as np
import pyarrow as pa

from farkle.simulation.simulation import _play_game, generate_strategy_grid
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils import parallel, random as urandom
from farkle.utils.manifest import append_manifest_line
from farkle.utils.writer import ParquetShardWriter

# from farkle.utils.logging import setup_info_logging, setup_warning_logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants (patched by tests/CLI)
# ---------------------------------------------------------------------------
NUM_SHUFFLES: int = 5_907  # BH-power calculation for default
# Default result of NUM_SHUFFLES * games_per_shuffle is 9_640_224 games total

# Counting metrics in pkl file
DESIRED_SEC_PER_CHUNK: int = 10
CKPT_EVERY_SEC: int = 30

# ---------------------------------------------------------------------------
# Dataclass configuration
# ---------------------------------------------------------------------------


@dataclass
class TournamentConfig:
    """Runtime configuration for :func:`run_tournament`."""

    n_players: int = 5
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
# Worker state and helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WorkerState:
    strats: list[ThresholdStrategy]
    cfg: TournamentConfig


_STATE: WorkerState | None = None


def _init_worker(
    strategies: Sequence[ThresholdStrategy],
    config: TournamentConfig,
) -> None:
    """Initialise per-process state."""

    global _STATE
    if 8_160 % config.n_players != 0:
        raise ValueError("n_players must divide 8,160")
    _STATE = WorkerState(list(strategies), config)


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

    state = _STATE
    assert state is not None

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(state.strats))
    game_seeds = urandom.spawn_seeds(state.cfg.games_per_shuffle, seed=seed)

    wins: Counter[str] = Counter()
    sums: Dict[str, Dict[str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    sq_sums: Dict[str, Dict[str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    rows: List[Dict[str, Any]] = []

    offset = 0
    for gseed in game_seeds:
        idxs = perm[offset : offset + state.cfg.n_players].tolist()
        offset += state.cfg.n_players

        row = _play_game(int(gseed), [state.strats[i] for i in idxs])
        winner = row.get("winner_seat") or row.get("winner")
        strat_repr = row[f"{winner}_strategy"]
        metrics = _extract_winner_metrics(row, winner)   # pyright: ignore[reportArgumentType]
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
        log.info("Shuffle %s started", sd)
        try:
            result = _play_shuffle(int(sd))
        except Exception:
            log.error("Shuffle %s failed", sd, exc_info=True)
            raise
        else:
            state = _STATE
            games = state.cfg.games_per_shuffle if state is not None else 0
            log.info("Shuffle %s finished: %d games", sd, games)
            total.update(result)
    return total


# Rich metrics variant -------------------------------------------------------


def _run_chunk_metrics(
    shuffle_seed_batch: Sequence[int],
    *,
    collect_rows: bool = False,
    row_dir: Path | None = None,
    manifest_path: Path | None = None,
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
    manifest_path : Path | None, default None
        When provided, append one NDJSON record per shard.

    Returns
    -------
    tuple
        ``(wins, sums, square_sums)`` where each element aggregates the
        respective values over the batch.
    """

    wins_total: Counter[str] = Counter()
    sums_total: Dict[str, Dict[str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    sq_total: Dict[str, Dict[str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}

    for seed in shuffle_seed_batch:
        log.info("Shuffle %s started", seed)
        wins, sums, sqs, rows = _play_one_shuffle(int(seed), collect_rows=collect_rows)
        wins_total.update(wins)
        for label in METRIC_LABELS:
            for k, v in sums[label].items():
                sums_total[label][k] += v
            for k, v in sqs[label].items():
                sq_total[label][k] += v

        if row_dir is not None and collect_rows:
            out = row_dir / f"rows_{getpid()}_{seed}.parquet"
            tbl = pa.Table.from_pylist(rows)
            writer = ParquetShardWriter(out)
            writer.write_batch(tbl)
            writer.close()
            if manifest_path is not None:
                append_manifest_line(
                    manifest_path,
                    {
                        "path": out.name,
                        "rows": writer.rows_written,
                        "n_players": _STATE.cfg.n_players if _STATE else None,
                        "shuffle_seed": int(seed),
                        "pid": getpid(),
                    },
                )

        # free memory for this shuffle
        rows.clear()

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

    seeds = urandom.spawn_seeds(sample_games, seed=seed)
    start = time.perf_counter()
    for gs in seeds:
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
    n_players: int | None = None,
    global_seed: int = 0,
    checkpoint_path: Path | str = "checkpoint.pkl",
    n_jobs: int | None = None,
    collect_metrics: bool = False,
    row_output_directory: Path | None = None,  # None if --row-dir omitted
    metric_chunk_directory: Path | None = None,
    num_shuffles: int = NUM_SHUFFLES,
) -> None:
    """Run a multi-process Monte-Carlo Farkle tournament.

    Parameters
    ----------
    config : TournamentConfig, optional
        Encapsulates all tunable constants (number of players, shuffle count,
        checkpoint cadence, etc.). If ``None`` a default instance is created
        from the module-level constants.
    n_players : int | None, optional
        Deprecated override for ``config.n_players``. If provided it replaces
        the value inside ``config`` for this run.
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
    metric_chunk_directory : pathlib.Path | None, default None
        Persist per-chunk metric aggregates to this directory instead of
        keeping them all in memory.

    Notes
    -----
    *Old keyword arguments such as ``num_shuffles`` and ``ckpt_every_sec`` are
    now fields of :class:`TournamentConfig`. Provide a custom config if you need
    to override them.*

    Side Effects
    ------------
    Creates/updates ``checkpoint_path`` and, if ``row_output_directory`` is
    given, a set of Parquet files containing raw game rows. When
    ``metric_chunk_directory`` is set, per-chunk metric aggregates are also
    written there.
    """
    strategies, _ = generate_strategy_grid()  # 8_160 strategies

    cfg = config or TournamentConfig()
    if n_players is not None:
        cfg.n_players = n_players
    if num_shuffles != cfg.num_shuffles:
        cfg.num_shuffles = num_shuffles
    if cfg.n_players < 2:
        raise ValueError("n_players must be ≥2")
    if 8_160 % cfg.n_players != 0:
        raise ValueError("n_players must divide 8,160")

    games_per_sec = _measure_throughput(strategies[: cfg.n_players])
    shuffles_per_chunk = max(
        1,
        int(cfg.desired_sec_per_chunk * games_per_sec // cfg.games_per_shuffle),
    )

    shuffle_seeds = list(urandom.spawn_seeds(cfg.num_shuffles, seed=global_seed))
    chunks = [
        shuffle_seeds[i : i + shuffles_per_chunk]
        for i in range(0, cfg.num_shuffles, shuffles_per_chunk)
    ]

    win_totals: Counter[str] = Counter()
    metric_sums: Dict[str, Dict[str, float]] | None
    metric_sq_sums: Dict[str, Dict[str, float]] | None
    if metric_chunk_directory is None:
        metric_sums = {m: defaultdict(float) for m in METRIC_LABELS}
        metric_sq_sums = {m: defaultdict(float) for m in METRIC_LABELS}
    else:
        metric_sums = None
        metric_sq_sums = None

    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    last_ckpt = time.perf_counter()

    collect_rows = row_output_directory is not None
    if collect_rows:
        assert row_output_directory is not None
        row_output_directory.mkdir(parents=True, exist_ok=True)

    if metric_chunk_directory is not None:
        metric_chunk_directory.mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")
    row_queue = None  # per-worker shards; no central writer

    if collect_metrics or collect_rows:
        manifest_path = (row_output_directory / "manifest.jsonl") if row_output_directory else None
        chunk_fn: Callable[[Sequence[int]], object] = partial(
            _run_chunk_metrics,
            collect_rows=collect_rows,
            row_dir=row_output_directory,
            manifest_path=manifest_path,
        )
    else:
        chunk_fn = _run_chunk

    try:
        for done, result in enumerate(
            parallel.process_map(
                chunk_fn,
                chunks,
                n_jobs=n_jobs,
                initializer=_init_worker,
                initargs=(strategies, cfg),
                window=4 * (n_jobs or 1),
            ),
            1,
        ):
            if collect_metrics or collect_rows:
                wins, sums, sqs = cast(
                    Tuple[Counter[str], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]],
                    result,
                )
                win_totals.update(wins)
                if metric_chunk_directory is not None:
                    chunk_path = metric_chunk_directory / f"metrics_{done:06d}.parquet"
                    rows = [
                        {
                            "metric": label,
                            "strategy": strat,
                            "sum": val,
                            "square_sum": sqs[label][strat],
                        }
                        for label in METRIC_LABELS
                        for strat, val in sums[label].items()
                    ]
                    pq.write_table(pa.Table.from_pylist(rows), chunk_path)
                else:
                    assert metric_sums is not None and metric_sq_sums is not None
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
                    None
                    if metric_chunk_directory is not None
                    else (metric_sums if collect_metrics or collect_rows else None),
                    None
                    if metric_chunk_directory is not None
                    else (metric_sq_sums if collect_metrics or collect_rows else None),
                )
                log.info(
                    "checkpoint … %d/%d chunks, %d games",
                    done,
                    len(chunks),
                    sum(win_totals.values()),
                )
                last_ckpt = now

        if metric_chunk_directory is not None and (collect_metrics or collect_rows):
            metric_sums = {m: defaultdict(float) for m in METRIC_LABELS}
            metric_sq_sums = {m: defaultdict(float) for m in METRIC_LABELS}
            for path in sorted(metric_chunk_directory.glob("metrics_*.parquet")):
                table = pq.read_table(path)
                for row in table.to_pylist():
                    label = row["metric"]
                    strat = row["strategy"]
                    metric_sums[label][strat] += row["sum"]
                    metric_sq_sums[label][strat] += row["square_sum"]

    finally:
        # no central writer to tear down
        pass

    _save_checkpoint(
        ckpt_path,
        win_totals,
        metric_sums if (collect_metrics or collect_rows) else None,
        metric_sq_sums if (collect_metrics or collect_rows) else None,
    )
    log.info("finished - %d games", sum(win_totals.values()))

