# src/farkle/simulation/run_tournament.py
"""Parallel Monte-Carlo tournament driver.

This version keeps the original fast win-count loop used in the unit tests but
adds optional collection of richer statistics. When enabled the worker
processes accumulate running sums and sum-of-squares for a small set of metrics
so that per-strategy means and variances can be computed without storing every
row.  A parquet dump of all rows can also be requested via --row-dir.
"""

from __future__ import annotations

import logging
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
from pyarrow import parquet as pq

from farkle.simulation.simulation import _play_game, generate_strategy_grid
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils import parallel
from farkle.utils import random as urandom
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.manifest import iter_manifest
from farkle.utils.streaming_loop import run_streaming_shard
from farkle.utils.writer import atomic_path

# from farkle.utils.logging import setup_info_logging, setup_warning_logging

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants (patched by tests/CLI)
# ---------------------------------------------------------------------------
NUM_SHUFFLES: int = 5_907  # BH-power calculation for default (can be overridden)

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
    n_strategies: int = 7_140  # overridden when strategies are provided

    @property
    def games_per_shuffle(self) -> int:
        """Number of unique games produced for a full shuffle of strategies."""
        return self.n_strategies // self.n_players


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
    "winner_hit_max_rounds",
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
        row[f"{winner}_hit_max_rounds"],
    ]


# ---------------------------------------------------------------------------
# Worker state and helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WorkerState:
    """State container shared between tournament worker processes."""

    strats: list[ThresholdStrategy]
    cfg: TournamentConfig


_STATE: WorkerState | None = None


def _init_worker(
    strategies: Sequence[ThresholdStrategy],
    config: TournamentConfig,
) -> None:
    """Initialise per-process state."""

    global _STATE
    if len(strategies) % config.n_players != 0:
        raise ValueError(f"n_players must divide {len(strategies):,}")
    _STATE = WorkerState(list(strategies), config)


# ---------------------------------------------------------------------------
# Shuffle-level helpers
# ---------------------------------------------------------------------------


def _play_one_shuffle(seed: int, *, collect_rows: bool = False) -> Tuple[
    Counter[int | str],
    Dict[str, Dict[int | str, float]],
    Dict[str, Dict[int | str, float]],
    List[Dict[str, Any]],
]:
    """Play all games for one shuffle and aggregate the results."""

    state = _STATE

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(state.strats))  # type: ignore
    game_seeds = urandom.spawn_seeds(state.cfg.games_per_shuffle, seed=seed)  # type: ignore

    wins: Counter[int | str] = Counter()
    sums: Dict[str, Dict[int | str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    sq_sums: Dict[str, Dict[int | str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    rows: List[Dict[str, Any]] = []

    offset = 0
    for gseed in game_seeds:
        idxs = perm[offset : offset + state.cfg.n_players].tolist()  # type: ignore
        offset += state.cfg.n_players  # type: ignore

        row = _play_game(int(gseed), [state.strats[i] for i in idxs])  # type: ignore
        winner = row.get("winner_seat") or row.get("winner")
        strat_repr = row[f"{winner}_strategy"]
        winner = cast(str, winner)
        metrics = _extract_winner_metrics(row, winner)  # pyright: ignore[reportArgumentType]
        wins[strat_repr] += 1
        for label, value in zip(METRIC_LABELS, metrics, strict=True):
            sums[label][strat_repr] += value
            sq_sums[label][strat_repr] += value * value
        if collect_rows:
            rows.append({"game_seed": int(gseed), **row})

    return wins, sums, sq_sums, rows


def _play_shuffle(seed: int) -> Counter[int | str]:
    """Compatibility wrapper returning only win counts for one shuffle."""

    wins, _, _, _ = _play_one_shuffle(seed, collect_rows=False)
    return wins


def _run_chunk(shuffle_seed_batch: Sequence[int]) -> Counter[int | str]:
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

    total: Counter[int | str] = Counter()
    batch_size = len(shuffle_seed_batch)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "Processing shuffle batch",
            extra={
                "stage": "simulation",
                "batch_size": batch_size,
                "first_seed": int(shuffle_seed_batch[0]) if batch_size else None,
                "last_seed": int(shuffle_seed_batch[-1]) if batch_size else None,
            },
        )
    for sd in shuffle_seed_batch:
        try:
            result = _play_shuffle(int(sd))
        except Exception:
            LOGGER.exception(
                "Shuffle failed",
                extra={"stage": "simulation", "shuffle_seed": int(sd)},
            )
            raise
        else:
            total.update(result)
    return total


def _run_chunk_item(
    item: Tuple[int, Sequence[int]],
    *,
    chunk_fn: Callable[[Sequence[int]], object],
) -> Tuple[int, object]:
    chunk_index, seeds = item
    return chunk_index, chunk_fn(seeds)


# Rich metrics variant -------------------------------------------------------


def _run_chunk_metrics(
    shuffle_seed_batch: Sequence[int],
    *,
    collect_rows: bool = False,
    row_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> Tuple[
    Counter[int | str],
    Dict[str, Dict[int | str, float]],
    Dict[str, Dict[int | str, float]],
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

    wins_total: Counter[int | str] = Counter()
    sums_total: Dict[str, Dict[int | str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    sq_total: Dict[str, Dict[int | str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}

    batch_size = len(shuffle_seed_batch)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "Processing metrics shuffle batch",
            extra={
                "stage": "simulation",
                "batch_size": batch_size,
                "first_seed": int(shuffle_seed_batch[0]) if batch_size else None,
                "last_seed": int(shuffle_seed_batch[-1]) if batch_size else None,
            },
        )
    for seed in shuffle_seed_batch:
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
            manifest_file = manifest_path or (row_dir / "manifest.jsonl")
            run_streaming_shard(
                out_path=str(out),
                manifest_path=str(manifest_file),
                schema=tbl.schema,
                batch_iter=(tbl,),
                manifest_extra={
                    "path": out.name,
                    "n_players": _STATE.cfg.n_players if _STATE else None,
                    "shuffle_seed": int(seed),
                    "pid": getpid(),
                },
            )
            LOGGER.info(
                "Row shard written",
                extra={
                    "stage": "simulation",
                    "shuffle_seed": int(seed),
                    "rows": len(rows),
                    "path": str(out),
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
    wins: Counter[int | str],
    sums: Mapping[str, Mapping[int | str, float]] | None,
    sq_sums: Mapping[str, Mapping[int | str, float]] | None,
    *,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Pickle the current aggregates to path."""

    payload: Dict[str, Any] = {"win_totals": wins}
    if sums is not None and sq_sums is not None:
        payload["metric_sums"] = sums
        payload["metric_square_sums"] = sq_sums
    if meta:
        payload["meta"] = dict(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as tmp_path:
        Path(tmp_path).write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def _coerce_counter(raw: object) -> Counter[int | str]:
    if isinstance(raw, Counter):
        return Counter(raw)
    if isinstance(raw, Mapping):
        return Counter({int(k) if str(k).isdigit() else k: int(v) for k, v in raw.items()})
    raise TypeError(f"Unexpected win_totals payload type: {type(raw)}")


def _coerce_metric_sums(
    raw: Mapping[str, Mapping[int | str, float]] | None,
) -> Dict[str, Dict[int | str, float]] | None:
    if raw is None:
        return None
    return {
        label: defaultdict(
            float,
            {int(k) if str(k).isdigit() else k: float(v) for k, v in raw.get(label, {}).items()},
        )
        for label in METRIC_LABELS
    }


def _manifest_int_set(manifest_path: Path, key: str) -> set[int]:
    values: set[int] = set()
    for record in iter_manifest(manifest_path):
        raw = record.get(key)
        if raw is None:
            continue
        try:
            values.add(int(raw))
        except (TypeError, ValueError):
            continue
    return values


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
    strategies: Sequence[ThresholdStrategy] | None = None,
    resume: bool = True,
    checkpoint_metadata: Mapping[str, Any] | None = None,
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
    resume : bool, default True
        When ``True`` (default), resume from existing checkpoints and manifests
        to avoid recomputing completed work.
    checkpoint_metadata : Mapping[str, Any] | None, default None
        Optional metadata to store alongside checkpoints for resume validation.

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
    if strategies is None:
        strategies, _ = generate_strategy_grid()  # 7_140 strategies

    cfg = config or TournamentConfig()
    if n_players is not None:
        cfg.n_players = n_players
    if num_shuffles != cfg.num_shuffles:
        cfg.num_shuffles = num_shuffles
    if cfg.n_players < 2:
        raise ValueError("n_players must be ≥2")
    cfg.n_strategies = len(strategies)
    if cfg.n_strategies % cfg.n_players != 0:
        raise ValueError(f"n_players must divide {cfg.n_strategies:,}")

    checkpoint_meta = {
        "n_players": cfg.n_players,
        "num_shuffles": cfg.num_shuffles,
        "global_seed": global_seed,
        "n_strategies": cfg.n_strategies,
    }
    if checkpoint_metadata:
        checkpoint_meta.update(checkpoint_metadata)

    games_per_sec = _measure_throughput(strategies[: cfg.n_players])
    shuffles_per_chunk = max(
        1,
        int(cfg.desired_sec_per_chunk * games_per_sec // cfg.games_per_shuffle),
    )
    LOGGER.debug(
        "Derived chunk sizing",
        extra={
            "stage": "simulation",
            "n_players": cfg.n_players,
            "games_per_shuffle": cfg.games_per_shuffle,
            "shuffles_per_chunk": shuffles_per_chunk,
        },
    )

    shuffle_seeds = list(urandom.spawn_seeds(cfg.num_shuffles, seed=global_seed))

    win_totals: Counter[int | str] = Counter()
    games_completed = 0
    metric_sums: Dict[str, Dict[int | str, float]] | None
    metric_sq_sums: Dict[str, Dict[int | str, float]] | None

    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    last_ckpt = time.perf_counter()
    payload: dict[str, Any] | None = None
    if resume and ckpt_path.exists():
        payload = pickle.loads(ckpt_path.read_bytes())
        assert payload is not None
        raw_counts = payload.get("win_totals", payload)
        win_totals = _coerce_counter(raw_counts)
        games_completed = int(sum(win_totals.values()))

    collect_rows = row_output_directory is not None
    if collect_rows:
        assert row_output_directory is not None
        row_output_directory.mkdir(parents=True, exist_ok=True)

    metrics_manifest_path: Path | None = None
    if metric_chunk_directory is not None:
        metric_chunk_directory.mkdir(parents=True, exist_ok=True)
        metrics_manifest_path = metric_chunk_directory / "metrics_manifest.jsonl"

    if collect_metrics or collect_rows:
        metric_sums = _coerce_metric_sums(payload.get("metric_sums") if payload else None)
        metric_sq_sums = _coerce_metric_sums(
            payload.get("metric_square_sums")
            if payload
            else None
        )
        if metric_sq_sums is None and payload:
            metric_sq_sums = _coerce_metric_sums(payload.get("metric_sq_sums"))
    else:
        metric_sums = None
        metric_sq_sums = None

    if metric_chunk_directory is None and (collect_metrics or collect_rows):
        if metric_sums is None:
            metric_sums = {m: defaultdict(float) for m in METRIC_LABELS}
        if metric_sq_sums is None:
            metric_sq_sums = {m: defaultdict(float) for m in METRIC_LABELS}
    elif metric_chunk_directory is not None:
        if metric_sums is None or metric_sq_sums is None:
            metric_sums = None
            metric_sq_sums = None

    mp_context = parallel.resolve_mp_context(cfg.mp_start_method)

    if resume and payload is not None and row_output_directory is not None:
        manifest_file = row_output_directory / "manifest.jsonl"
        if manifest_file.exists():
            completed_shuffles = _manifest_int_set(manifest_file, "shuffle_seed")
            if completed_shuffles:
                shuffle_seeds = [s for s in shuffle_seeds if s not in completed_shuffles]
                LOGGER.info(
                    "Filtered completed shuffle seeds from row manifest",
                    extra={
                        "stage": "simulation",
                        "completed_shuffles": len(completed_shuffles),
                        "remaining_shuffles": len(shuffle_seeds),
                        "manifest_path": str(manifest_file),
                    },
                )

    chunks = [
        shuffle_seeds[i : i + shuffles_per_chunk]
        for i in range(0, len(shuffle_seeds), shuffles_per_chunk)
    ]
    chunk_items = list(enumerate(chunks, start=1))
    if (
        resume
        and payload is not None
        and metrics_manifest_path is not None
        and metrics_manifest_path.exists()
    ):
        completed_chunk_indices = _manifest_int_set(metrics_manifest_path, "chunk_index")
        if completed_chunk_indices:
            chunk_items = [
                (idx, seeds) for idx, seeds in chunk_items if idx not in completed_chunk_indices
            ]
            LOGGER.info(
                "Filtered completed metric chunks from manifest",
                extra={
                    "stage": "simulation",
                    "completed_chunks": len(completed_chunk_indices),
                    "remaining_chunks": len(chunk_items),
                    "manifest_path": str(metrics_manifest_path),
                },
            )

    LOGGER.info(
        "Tournament run start",
        extra={
            "stage": "simulation",
            "n_players": cfg.n_players,
            "num_shuffles": cfg.num_shuffles,
            "global_seed": global_seed,
            "n_jobs": n_jobs,
            "chunks": len(chunk_items),
            "collect_metrics": collect_metrics,
            "collect_rows": collect_rows,
            "row_dir": str(row_output_directory) if row_output_directory else None,
            "checkpoint_path": str(ckpt_path),
            "resume": resume,
        },
    )

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

    new_metric_chunks: list[Path] = []
    chunk_wrapper = partial(_run_chunk_item, chunk_fn=chunk_fn)

    try:
        for chunk_index, result in parallel.process_map(
            chunk_wrapper,
            chunk_items,
            n_jobs=n_jobs,
            initializer=_init_worker,
            initargs=(strategies, cfg),
            window=4 * (n_jobs or 1),
            mp_context=mp_context,
        ):
            if collect_metrics or collect_rows:
                wins, sums, sqs = cast(
                    Tuple[
                        Counter[int | str],
                        Dict[str, Dict[int | str, float]],
                        Dict[str, Dict[int | str, float]],
                    ],
                    result,
                )
                win_totals.update(wins)
                chunk_games = int(sum(wins.values()))
                games_completed += chunk_games
                if metric_chunk_directory is not None:
                    chunk_path = metric_chunk_directory / f"metrics_{chunk_index:06d}.parquet"
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
                    tbl = pa.Table.from_pylist(rows)
                    manifest_file = metrics_manifest_path or (
                        metric_chunk_directory / "metrics_manifest.jsonl"
                    )
                    run_streaming_shard(
                        out_path=str(chunk_path),
                        manifest_path=str(manifest_file),
                        schema=tbl.schema,
                        batch_iter=(tbl,),
                        manifest_extra={
                            "path": chunk_path.name,
                            "chunk_index": chunk_index,
                            "n_players": cfg.n_players,
                        },
                    )
                    new_metric_chunks.append(chunk_path)
                    LOGGER.info(
                        "Metrics chunk written",
                        extra={
                            "stage": "simulation",
                            "chunk_index": chunk_index,
                            "rows": len(rows),
                            "path": str(chunk_path),
                        },
                    )
                    if metric_sums is not None and metric_sq_sums is not None:
                        for label in METRIC_LABELS:
                            for k, v in sums[label].items():
                                metric_sums[label][k] += v
                            for k, v in sqs[label].items():
                                metric_sq_sums[label][k] += v
                else:
                    assert metric_sums is not None and metric_sq_sums is not None
                    for label in METRIC_LABELS:
                        for k, v in sums[label].items():
                            metric_sums[label][k] += v
                        for k, v in sqs[label].items():
                            metric_sq_sums[label][k] += v
                LOGGER.debug(
                    "Chunk processed",
                    extra={
                        "stage": "simulation",
                        "chunk_index": chunk_index,
                        "wins": chunk_games,
                    },
                )
            else:
                chunk_wins = cast(Counter[int | str], result)
                win_totals.update(chunk_wins)
                chunk_games = int(sum(chunk_wins.values()))
                games_completed += chunk_games
                LOGGER.debug(
                    "Chunk processed",
                    extra={
                        "stage": "simulation",
                        "chunk_index": chunk_index,
                        "wins": chunk_games,
                    },
                )

            now = time.perf_counter()
            if now - last_ckpt >= cfg.ckpt_every_sec:
                _save_checkpoint(
                    ckpt_path,
                    win_totals,
                    (metric_sums if (collect_metrics or collect_rows) else None),
                    (metric_sq_sums if (collect_metrics or collect_rows) else None),
                    meta=checkpoint_meta,
                )
                LOGGER.info(
                    "Checkpoint written after %d games",
                    games_completed,
                    extra={
                        "stage": "simulation",
                        "chunk_index": chunk_index,
                        "chunks_total": len(chunk_items),
                        "games": games_completed,
                        "games_completed": games_completed,
                        "path": str(ckpt_path),
                    },
                )
                last_ckpt = now

        if metric_chunk_directory is not None and (collect_metrics or collect_rows):
            if metric_sums is None or metric_sq_sums is None:
                metric_sums = {m: defaultdict(float) for m in METRIC_LABELS}
                metric_sq_sums = {m: defaultdict(float) for m in METRIC_LABELS}
                chunk_paths = sorted(metric_chunk_directory.glob("metrics_*.parquet"))
            else:
                chunk_paths = new_metric_chunks
            for path in sorted(chunk_paths):
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
        meta=checkpoint_meta,
    )
    if (collect_metrics or collect_rows) and metric_sums is not None and metric_sq_sums is not None:
        metrics_rows = []
        for label in METRIC_LABELS:
            sums_for_label = metric_sums.get(label, {})
            squares_for_label = metric_sq_sums.get(label, {})
            for strat, total in sums_for_label.items():
                metrics_rows.append(
                    {
                        "metric": label,
                        "strategy": strat,
                        "sum": float(total),
                        "square_sum": float(squares_for_label.get(strat, 0.0)),
                    }
                )

        fields: list[pa.Field] = [
            pa.field("metric", pa.string()),
            pa.field("strategy", pa.int32()),
            pa.field("sum", pa.float64()),
            pa.field("square_sum", pa.float64()),
        ]
        metrics_table = pa.Table.from_pylist(
            metrics_rows,
            schema=pa.schema(fields),
        )
        metrics_path = ckpt_path.with_name(f"{cfg.n_players}p_metrics.parquet")
        write_parquet_atomic(metrics_table, metrics_path)
        LOGGER.info(
            "Per-table metrics written",
            extra={
                "stage": "simulation",
                "n_players": cfg.n_players,
                "rows": metrics_table.num_rows,
                "path": str(metrics_path),
            },
        )
    LOGGER.info(
        "Tournament run complete after %d games",
        games_completed,
        extra={
            "stage": "simulation",
            "games": games_completed,
            "games_completed": games_completed,
            "n_players": cfg.n_players,
            "chunks": len(chunk_items),
            "checkpoint_path": str(ckpt_path),
        },
    )
