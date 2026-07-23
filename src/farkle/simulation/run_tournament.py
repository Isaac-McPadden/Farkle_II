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
from dataclasses import dataclass, field
from functools import partial
from itertools import islice
from os import getpid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Sequence, Tuple, cast

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq

from farkle.game.engine import TerminationStatus
from farkle.simulation.simulation import (
    PlayerRngCoordinates,
    _play_game,
    generate_strategy_grid,
    simulation_rows_to_table,
)
from farkle.simulation.strategies import ThresholdStrategy
from farkle.simulation.workload_planner import (
    TournamentWorkloadPlan,
    WorkloadCapExceeded,
    write_workload_plan,
)
from farkle.utils import parallel
from farkle.utils import random as urandom
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.manifest import iter_manifest
from farkle.utils.progress import ProgressLogConfig, ScheduledProgressLogger
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION, TOURNAMENT_METHOD_VERSION
from farkle.utils.streaming_loop import run_streaming_shard
from farkle.utils.writer import atomic_path

# from farkle.utils.logging import setup_info_logging, setup_warning_logging

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants (patched by tests/CLI)
# ---------------------------------------------------------------------------
NUM_SHUFFLES: int = 5_907  # Direct low-level API default; runner supplies its resolved plan.

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
    progress_logging: ProgressLogConfig = field(default_factory=ProgressLogConfig)
    n_strategies: int = 7_140  # overridden when strategies are provided
    mp_start_method: str | None = None
    deterministic_batch_size: int = 30

    @property
    def games_per_shuffle(self) -> int:
        """Number of unique games produced for a full shuffle of strategies."""
        return self.n_strategies // self.n_players


@dataclass(frozen=True, slots=True)
class ShuffleTask:
    """Stable coordinate identity for one complete tournament shuffle."""

    root_seed: int
    k: int
    shuffle_index: int
    shuffle_seed: int
    deterministic_batch_id: int


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


def _require_outcome(row: Mapping[str, Any], *, source: str) -> TerminationStatus:
    """Validate and return one explicit attempted-game outcome."""

    status = row.get("termination_status")
    version = row.get("outcome_schema_version")
    if version != OUTCOME_SCHEMA_VERSION:
        raise RuntimeError(
            f"{source} is not outcome-schema-v{OUTCOME_SCHEMA_VERSION} compatible; "
            "explicit-outcome tournament aggregation is disabled for this row"
        )
    if status not in {member.value for member in TerminationStatus}:
        raise RuntimeError(f"{source} has unsupported termination_status={status!r}")
    outcome = TerminationStatus(status)
    winner_seat = row.get("winner_seat")
    winner_strategy = row.get("winner_strategy")
    if outcome is TerminationStatus.SAFETY_LIMIT:
        if winner_seat is not None or winner_strategy is not None:
            raise RuntimeError(f"{source} fabricates a winner for a safety-limit attempt")
    elif winner_seat is None or winner_strategy is None:
        raise RuntimeError(f"{source} is completed but has no canonical winner")
    return outcome


class OutcomeCounter(Counter[int | str]):
    """Win counter carrying additive attempted/completed exposure conservation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.attempted_exposures: Counter[int | str] = Counter()
        self.completed_exposures: Counter[int | str] = Counter()
        self.safety_limit_exposures: Counter[int | str] = Counter()
        self.games_attempted = 0
        self.games_completed = 0
        self.games_safety_limit = 0
        super().__init__(*args, **kwargs)

    def record_row(self, row: Mapping[str, Any], *, k: int, source: str) -> TerminationStatus:
        """Record one validated row without fabricating a winner."""

        status = _require_outcome(row, source=source)
        for seat in range(1, k + 1):
            strategy = row.get(f"P{seat}_strategy")
            if strategy is None:
                raise ValueError(f"{source} is missing strategy exposure for P{seat}")
            self.attempted_exposures[strategy] += 1
            if status is TerminationStatus.COMPLETED:
                self.completed_exposures[strategy] += 1
            else:
                self.safety_limit_exposures[strategy] += 1
        self.games_attempted += 1
        if status is TerminationStatus.COMPLETED:
            self.games_completed += 1
        else:
            self.games_safety_limit += 1
        return status

    def absorb(self, other: Counter[int | str]) -> None:
        """Add a worker counter, accepting legacy test doubles as completed games."""

        super().update(other)
        if isinstance(other, OutcomeCounter):
            self.attempted_exposures.update(other.attempted_exposures)
            self.completed_exposures.update(other.completed_exposures)
            self.safety_limit_exposures.update(other.safety_limit_exposures)
            self.games_attempted += other.games_attempted
            self.games_completed += other.games_completed
            self.games_safety_limit += other.games_safety_limit
            return
        completed = int(sum(other.values()))
        self.attempted_exposures.update(other)
        self.completed_exposures.update(other)
        self.games_attempted += completed
        self.games_completed += completed

    def outcome_payload(self) -> dict[str, Any]:
        """Return explicit checkpoint/report counts."""

        return {
            "games_attempted": self.games_attempted,
            "games_completed": self.games_completed,
            "games_safety_limit": self.games_safety_limit,
            "attempted_exposures": dict(self.attempted_exposures),
            "completed_exposures": dict(self.completed_exposures),
            "safety_limit_exposures": dict(self.safety_limit_exposures),
        }

    def __reduce__(self) -> tuple[object, tuple[dict[int | str, int], dict[str, Any]]]:
        """Preserve explicit outcome state across process boundaries."""

        return (_restore_outcome_counter, (dict(self), self.outcome_payload()))


def _restore_outcome_counter(
    counts: dict[int | str, int], outcome_counts: dict[str, Any]
) -> OutcomeCounter:
    """Rebuild a worker result without losing non-Counter conservation fields."""

    restored = OutcomeCounter(counts)
    restored.attempted_exposures.update(outcome_counts.get("attempted_exposures", {}))
    restored.completed_exposures.update(outcome_counts.get("completed_exposures", {}))
    restored.safety_limit_exposures.update(outcome_counts.get("safety_limit_exposures", {}))
    restored.games_attempted = int(outcome_counts.get("games_attempted", 0))
    restored.games_completed = int(outcome_counts.get("games_completed", 0))
    restored.games_safety_limit = int(outcome_counts.get("games_safety_limit", 0))
    return restored


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


def _coerce_shuffle_task(task: ShuffleTask | int) -> ShuffleTask:
    """Support direct low-level calls while production uses complete coordinates."""

    if isinstance(task, ShuffleTask):
        return task
    state = _STATE
    k = state.cfg.n_players if state is not None else 0
    return ShuffleTask(
        root_seed=int(task),
        k=k,
        shuffle_index=0,
        shuffle_seed=int(task),
        deterministic_batch_id=0,
    )


def _play_one_shuffle(task: ShuffleTask | int, *, collect_rows: bool = False) -> Tuple[
    Counter[int | str],
    Dict[str, Dict[int | str, float]],
    Dict[str, Dict[int | str, float]],
    List[Dict[str, Any]],
]:
    """Play all games for one shuffle and aggregate the results."""

    state = _STATE
    work = _coerce_shuffle_task(task)

    rng = urandom.coordinate_rng(
        urandom.RandomPurpose.SHUFFLE_PERMUTATION,
        root_seed=work.root_seed,
        k=work.k,
        shuffle_index=work.shuffle_index,
    )
    perm = rng.permutation(len(state.strats))  # type: ignore
    game_seeds = [
        urandom.coordinate_seed(
            urandom.RandomPurpose.TOURNAMENT_GAME,
            root_seed=work.root_seed,
            k=work.k,
            shuffle_index=work.shuffle_index,
            game_index=game_index,
            dtype=np.uint32,
        )
        for game_index in range(state.cfg.games_per_shuffle)  # type: ignore[union-attr]
    ]

    wins = OutcomeCounter()
    sums: Dict[str, Dict[int | str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    sq_sums: Dict[str, Dict[int | str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    rows: List[Dict[str, Any]] = []

    offset = 0
    for game_index, gseed in enumerate(game_seeds):
        idxs = perm[offset : offset + state.cfg.n_players].tolist()  # type: ignore
        offset += state.cfg.n_players  # type: ignore

        row = _play_game(  # type: ignore[union-attr]
            int(gseed),
            [state.strats[i] for i in idxs],  # type: ignore[union-attr]
            provenance={
                "root_seed": work.root_seed,
                "k": work.k,
                "shuffle_index": work.shuffle_index,
                "game_index": game_index,
                "deterministic_batch_id": work.deterministic_batch_id,
                "shuffle_seed": work.shuffle_seed,
                "game_seed": int(gseed),
                "rng_scheme_version": urandom.RNG_SCHEME_VERSION,
                "rng_purpose_namespace": int(urandom.RandomPurpose.TOURNAMENT_GAME),
            },
            player_rng_coordinates=PlayerRngCoordinates(
                purpose=urandom.RandomPurpose.TOURNAMENT_PLAYER,
                root_seed=work.root_seed,
                k=work.k,
                shuffle_index=work.shuffle_index,
                game_index=game_index,
            ),
        )
        status = wins.record_row(row, k=work.k, source="Simulation row")
        if status is TerminationStatus.SAFETY_LIMIT:
            if collect_rows:
                rows.append(dict(row))
            continue
        winner = row.get("winner_seat")
        if winner is None:
            raise ValueError("Simulation row missing canonical winner_seat")
        strat_repr = row[f"{winner}_strategy"]
        winner = cast(str, winner)
        metrics = _extract_winner_metrics(row, winner)  # pyright: ignore[reportArgumentType]
        wins[strat_repr] += 1
        for label, value in zip(METRIC_LABELS, metrics, strict=True):
            sums[label][strat_repr] += value
            sq_sums[label][strat_repr] += value * value
        if collect_rows:
            rows.append(dict(row))

    return wins, sums, sq_sums, rows


def _play_shuffle(task: ShuffleTask | int) -> Counter[int | str]:
    """Compatibility wrapper returning only win counts for one shuffle."""

    wins, _, _, _ = _play_one_shuffle(task, collect_rows=False)
    return wins


def _run_chunk(shuffle_tasks: Sequence[ShuffleTask | int]) -> Counter[int | str]:
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

    tasks = [_coerce_shuffle_task(task) for task in shuffle_tasks]
    total = OutcomeCounter()
    batch_size = len(tasks)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "Processing shuffle batch",
            extra={
                "stage": "simulation",
                "batch_size": batch_size,
                "first_seed": tasks[0].shuffle_seed if batch_size else None,
                "last_seed": tasks[-1].shuffle_seed if batch_size else None,
            },
        )
    for task in tasks:
        try:
            result = _play_shuffle(task)
        except Exception:
            LOGGER.exception(
                "Shuffle failed",
                extra={
                    "stage": "simulation",
                    "shuffle_seed": task.shuffle_seed,
                    "shuffle_index": task.shuffle_index,
                },
            )
            raise
        else:
            total.absorb(result)
    return total


def _run_chunk_item(
    item: Tuple[int, Sequence[ShuffleTask]],
    *,
    chunk_fn: Callable[[Sequence[ShuffleTask]], object],
) -> Tuple[int, tuple[ShuffleTask, ...], object]:
    chunk_index, seeds = item
    tasks = tuple(seeds)
    return chunk_index, tasks, chunk_fn(tasks)


# Rich metrics variant -------------------------------------------------------


def _run_chunk_metrics(
    shuffle_tasks: Sequence[ShuffleTask | int],
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
    shuffle_tasks : Sequence[ShuffleTask]
        Stable coordinate tasks for each shuffle in this process-executor batch.
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

    tasks = [_coerce_shuffle_task(task) for task in shuffle_tasks]
    wins_total = OutcomeCounter()
    sums_total: Dict[str, Dict[int | str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}
    sq_total: Dict[str, Dict[int | str, float]] = {m: defaultdict(float) for m in METRIC_LABELS}

    batch_size = len(tasks)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "Processing metrics shuffle batch",
            extra={
                "stage": "simulation",
                "batch_size": batch_size,
                "first_seed": tasks[0].shuffle_seed if batch_size else None,
                "last_seed": tasks[-1].shuffle_seed if batch_size else None,
            },
        )
    for task in tasks:
        wins, sums, sqs, rows = _play_one_shuffle(task, collect_rows=collect_rows)
        wins_total.absorb(wins)
        for label in METRIC_LABELS:
            for k, v in sums[label].items():
                sums_total[label][k] += v
            for k, v in sqs[label].items():
                sq_total[label][k] += v

        if row_dir is not None and collect_rows:
            out = row_dir / f"rows_{task.root_seed}_{task.k}p_{task.shuffle_index:012d}.parquet"
            tbl = simulation_rows_to_table(rows, task.k)
            manifest_file = manifest_path or (row_dir / "manifest.jsonl")
            run_streaming_shard(
                out_path=str(out),
                manifest_path=str(manifest_file),
                schema=tbl.schema,
                batch_iter=(tbl,),
                manifest_extra={
                    "path": out.name,
                    "root_seed": task.root_seed,
                    "n_players": task.k,
                    "shuffle_index": task.shuffle_index,
                    "shuffle_seed": task.shuffle_seed,
                    "deterministic_batch_id": task.deterministic_batch_id,
                    "rng_scheme_version": urandom.RNG_SCHEME_VERSION,
                    "rng_purpose_namespace": int(urandom.RandomPurpose.TOURNAMENT_SHUFFLE),
                    "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
                    "tournament_method_version": TOURNAMENT_METHOD_VERSION,
                    "pid": getpid(),
                },
            )
            LOGGER.info(
                "Row shard written",
                extra={
                    "stage": "simulation",
                    "shuffle_seed": task.shuffle_seed,
                    "shuffle_index": task.shuffle_index,
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
    for game_index, game_seed in enumerate(seeds):
        _play_game(
            int(game_seed),
            sample_strategies,
            player_rng_coordinates=PlayerRngCoordinates(
                purpose=urandom.RandomPurpose.TOURNAMENT_PLAYER,
                root_seed=seed,
                k=len(sample_strategies),
                shuffle_index=0,
                game_index=game_index,
            ),
        )
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
    if isinstance(wins, OutcomeCounter):
        payload["outcome_counts"] = wins.outcome_payload()
    if sums is not None and sq_sums is not None:
        payload["metric_sums"] = sums
        payload["metric_square_sums"] = sq_sums
    if meta:
        payload["meta"] = dict(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as tmp_path:
        Path(tmp_path).write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def _coerce_counter(raw: object, outcome_counts: Mapping[str, Any] | None = None) -> OutcomeCounter:
    """Coerce checkpointed win totals into a normalized ``Counter``.

    Args:
        raw: Raw win-total payload loaded from checkpoint metadata.

    Returns:
        Counter keyed by normalized strategy identifiers.
    """
    normalized = OutcomeCounter()
    if isinstance(raw, Counter):
        Counter.update(normalized, raw)
    elif isinstance(raw, Mapping):
        Counter.update(
            normalized,
            {int(k) if str(k).isdigit() else k: int(v) for k, v in raw.items()},
        )
    else:
        raise TypeError(f"Unexpected win_totals payload type: {type(raw)}")
    if isinstance(raw, OutcomeCounter) and outcome_counts is None:
        normalized.attempted_exposures.update(raw.attempted_exposures)
        normalized.completed_exposures.update(raw.completed_exposures)
        normalized.safety_limit_exposures.update(raw.safety_limit_exposures)
        normalized.games_attempted = raw.games_attempted
        normalized.games_completed = raw.games_completed
        normalized.games_safety_limit = raw.games_safety_limit
        return normalized
    if outcome_counts is None:
        normalized.attempted_exposures.update(normalized)
        normalized.completed_exposures.update(normalized)
        normalized.games_completed = int(sum(normalized.values()))
        normalized.games_attempted = normalized.games_completed
        return normalized
    for name, target in (
        ("attempted_exposures", normalized.attempted_exposures),
        ("completed_exposures", normalized.completed_exposures),
        ("safety_limit_exposures", normalized.safety_limit_exposures),
    ):
        values = outcome_counts.get(name, {})
        if not isinstance(values, Mapping):
            raise TypeError(f"Unexpected {name} payload type: {type(values)}")
        target.update({int(k) if str(k).isdigit() else k: int(v) for k, v in values.items()})
    normalized.games_attempted = int(outcome_counts.get("games_attempted", 0))
    normalized.games_completed = int(
        outcome_counts.get("games_completed", sum(normalized.values()))
    )
    normalized.games_safety_limit = int(outcome_counts.get("games_safety_limit", 0))
    if min(
        (
            normalized.games_attempted,
            normalized.games_completed,
            normalized.games_safety_limit,
        )
    ) < 0 or any(value < 0 for value in normalized.values()):
        raise ValueError("checkpoint win counts must be nonnegative")
    strategies = (
        set(normalized)
        | set(normalized.attempted_exposures)
        | set(normalized.completed_exposures)
        | set(normalized.safety_limit_exposures)
    )
    for strategy in strategies:
        attempted = normalized.attempted_exposures[strategy]
        completed = normalized.completed_exposures[strategy]
        safety_limit = normalized.safety_limit_exposures[strategy]
        if min(attempted, completed, safety_limit) < 0:
            raise ValueError("checkpoint exposure counts must be nonnegative")
        if attempted != completed + safety_limit or normalized[strategy] > completed:
            raise ValueError("checkpoint strategy exposure conservation failed")
    if normalized.games_attempted != normalized.games_completed + normalized.games_safety_limit:
        raise ValueError("checkpoint outcome counts violate attempted = completed + safety_limit")
    if sum(normalized.values()) != normalized.games_completed:
        raise ValueError("checkpoint wins must equal completed games")
    if normalized.games_attempted:
        attempted_exposures = sum(normalized.attempted_exposures.values())
        if attempted_exposures % normalized.games_attempted:
            raise ValueError("checkpoint attempted exposures do not identify an integer k")
        k = attempted_exposures // normalized.games_attempted
        if k < 2 or sum(normalized.completed_exposures.values()) != k * normalized.games_completed:
            raise ValueError("checkpoint completed exposure conservation failed")
        if sum(normalized.safety_limit_exposures.values()) != k * normalized.games_safety_limit:
            raise ValueError("checkpoint safety-limit exposure conservation failed")
    elif any(
        (
            normalized.games_completed,
            normalized.games_safety_limit,
            sum(normalized.attempted_exposures.values()),
        )
    ):
        raise ValueError("zero attempted games require zero outcome counts")
    return normalized


def _coerce_metric_sums(
    raw: Mapping[str, Mapping[int | str, float]] | None,
) -> Dict[str, Dict[int | str, float]] | None:
    """Coerce checkpointed metric aggregates into normalized nested mappings.

    Args:
        raw: Raw metric aggregate payload loaded from checkpoint metadata.

    Returns:
        Nested default-dict mapping keyed by metric label and strategy identifier.
    """
    if raw is None:
        return None
    return {
        label: defaultdict(
            float,
            {int(k) if str(k).isdigit() else k: float(v) for k, v in raw.get(label, {}).items()},
        )
        for label in METRIC_LABELS
    }


def _empty_metric_aggregates() -> Tuple[
    Dict[str, Dict[int | str, float]],
    Dict[str, Dict[int | str, float]],
]:
    return (
        {label: defaultdict(float) for label in METRIC_LABELS},
        {label: defaultdict(float) for label in METRIC_LABELS},
    )


def _manifest_int_set(manifest_path: Path, key: str) -> set[int]:
    """Collect integer values for one key across manifest records.

    Args:
        manifest_path: Manifest file to scan.
        key: Record key whose integer-like values should be collected.

    Returns:
        Set of successfully parsed integer values.
    """
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


def _resolve_manifest_record_path(manifest_path: Path, record: Mapping[str, Any]) -> Path:
    """Resolve a manifest record's relative or absolute shard path.

    Args:
        manifest_path: Manifest file containing the record.
        record: Manifest record with a ``path`` field.

    Returns:
        Absolute shard path referenced by the record.
    """
    raw_path = record.get("path")
    if raw_path in (None, ""):
        raise ValueError(f"Manifest record missing path at {manifest_path}")
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _load_row_manifest_aggregates(
    manifest_path: Path,
) -> Tuple[
    OutcomeCounter,
    Dict[str, Dict[int | str, float]],
    Dict[str, Dict[int | str, float]],
]:
    """Rebuild win and metric aggregates by replaying row shards from a manifest.

    Args:
        manifest_path: Manifest listing row shard parquet files.

    Returns:
        Tuple of win totals, metric sums, and metric squared sums.
    """
    win_totals = OutcomeCounter()
    metric_sums, metric_sq_sums = _empty_metric_aggregates()

    for record in iter_manifest(manifest_path):
        row_path = _resolve_manifest_record_path(manifest_path, record)
        if not row_path.exists():
            raise FileNotFoundError(f"Missing row shard listed in manifest: {row_path}")
        parquet_file = pq.ParquetFile(row_path)
        for batch in parquet_file.iter_batches():
            for row in batch.to_pylist():
                status = win_totals.record_row(row, k=int(row["k"]), source=f"Row shard {row_path}")
                if status is TerminationStatus.SAFETY_LIMIT:
                    continue
                winner = row.get("winner_seat")
                if winner is None:
                    raise ValueError(f"Row shard missing canonical winner_seat: {row_path}")
                winner_name = cast(str, winner)
                strategy = row.get(f"{winner_name}_strategy")
                if strategy is None:
                    raise ValueError(
                        f"Row shard missing winner strategy for {winner_name!r}: {row_path}"
                    )
                metrics = _extract_winner_metrics(row, winner_name)
                win_totals[strategy] += 1
                for label, value in zip(METRIC_LABELS, metrics, strict=True):
                    metric_sums[label][strategy] += float(value)
                    metric_sq_sums[label][strategy] += float(value) * float(value)

    return win_totals, metric_sums, metric_sq_sums


def _load_metric_chunk_aggregates(
    manifest_path: Path,
) -> Tuple[
    OutcomeCounter | None,
    Dict[str, Dict[int | str, float]],
    Dict[str, Dict[int | str, float]],
]:
    """Rebuild aggregates from metric chunk shards listed in a manifest.

    Args:
        manifest_path: Manifest listing metric chunk parquet files.

    Returns:
        Tuple of optional win totals, metric sums, and metric squared sums.
    """
    win_totals = OutcomeCounter()
    metric_sums, metric_sq_sums = _empty_metric_aggregates()
    wins_available = True
    outcomes_available = True

    for record in iter_manifest(manifest_path):
        chunk_path = _resolve_manifest_record_path(manifest_path, record)
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing metric chunk listed in manifest: {chunk_path}")
        parquet_file = pq.ParquetFile(chunk_path)
        column_names = set(parquet_file.schema_arrow.names)
        file_has_wins = "wins" in column_names
        wins_available = wins_available and file_has_wins
        columns = ["metric", "strategy", "sum", "square_sum"]
        if file_has_wins:
            columns.append("wins")
        outcome_columns = [
            "attempted_exposures",
            "completed_exposures",
            "safety_limit_exposures",
        ]
        file_has_outcomes = set(outcome_columns).issubset(column_names)
        outcomes_available = outcomes_available and file_has_outcomes
        if file_has_outcomes:
            columns.extend(outcome_columns)
        for batch in parquet_file.iter_batches(columns=columns):
            for row in batch.to_pylist():
                label = row["metric"]
                strategy = row["strategy"]
                metric_sums[label][strategy] += float(row["sum"])
                metric_sq_sums[label][strategy] += float(row["square_sum"])
                if wins_available and file_has_wins and label == METRIC_LABELS[0]:
                    win_totals[strategy] += int(row["wins"])
                    if file_has_outcomes:
                        win_totals.attempted_exposures[strategy] += int(row["attempted_exposures"])
                        win_totals.completed_exposures[strategy] += int(row["completed_exposures"])
                        win_totals.safety_limit_exposures[strategy] += int(
                            row["safety_limit_exposures"]
                        )
                    else:
                        wins = int(row["wins"])
                        win_totals.attempted_exposures[strategy] += wins
                        win_totals.completed_exposures[strategy] += wins

    if wins_available:
        win_totals.games_completed = int(sum(win_totals.values()))
        win_totals.games_attempted = int(sum(win_totals.attempted_exposures.values()))
        if outcomes_available and win_totals.games_attempted:
            # Every game contributes exactly k exposures; manifests own one k.
            player_counts = _manifest_int_set(manifest_path, "n_players")
            if len(player_counts) != 1:
                raise ValueError(f"Metric manifest must identify one player count: {player_counts}")
            k = next(iter(player_counts))
            win_totals.games_attempted //= k
            win_totals.games_safety_limit = (
                int(sum(win_totals.safety_limit_exposures.values())) // k
            )
        elif not outcomes_available:
            win_totals.games_attempted = win_totals.games_completed

    return (win_totals if wins_available else None), metric_sums, metric_sq_sums


def _iter_original_chunk_items(
    *,
    num_shuffles: int,
    shuffles_per_chunk: int,
    global_seed: int,
    k: int = 0,
    deterministic_batch_size: int = 30,
) -> Iterator[Tuple[int, List[ShuffleTask]]]:
    """Yield deterministic shuffle-seed chunks in original chunk order.

    Args:
        num_shuffles: Total number of shuffle seeds to generate.
        shuffles_per_chunk: Maximum seeds per emitted chunk.
        global_seed: Root seed used to spawn deterministic chunk seeds.

    Yields:
        ``(chunk_index, chunk_seeds)`` pairs in original execution order.
    """
    task_iter = iter(
        ShuffleTask(
            root_seed=global_seed,
            k=k,
            shuffle_index=shuffle_index,
            shuffle_seed=urandom.coordinate_seed(
                urandom.RandomPurpose.TOURNAMENT_SHUFFLE,
                root_seed=global_seed,
                k=k,
                shuffle_index=shuffle_index,
                dtype=np.uint32,
            ),
            deterministic_batch_id=shuffle_index // deterministic_batch_size,
        )
        for shuffle_index in range(num_shuffles)
    )
    chunk_index = 0
    while True:
        chunk = list(islice(task_iter, shuffles_per_chunk))
        if not chunk:
            return
        chunk_index += 1
        yield chunk_index, chunk


def _iter_pending_chunk_items(
    *,
    num_shuffles: int,
    shuffles_per_chunk: int,
    global_seed: int,
    k: int = 0,
    deterministic_batch_size: int = 30,
    completed_shuffle_indices: set[int],
    completed_chunk_indices: set[int],
) -> Iterator[Tuple[int, List[ShuffleTask]]]:
    """Yield only the unfinished shuffle chunks needed for a resumed run.

    Args:
        num_shuffles: Total number of shuffle seeds to generate.
        shuffles_per_chunk: Maximum seeds per emitted chunk.
        global_seed: Root seed used to spawn deterministic chunk seeds.
        completed_shuffle_indices: Semantic shuffle indices already processed in prior runs.
        completed_chunk_indices: Chunk indices already fully processed.

    Yields:
        ``(chunk_index, pending_seeds)`` pairs for remaining work only.
    """
    for chunk_index, chunk in _iter_original_chunk_items(
        num_shuffles=num_shuffles,
        shuffles_per_chunk=shuffles_per_chunk,
        global_seed=global_seed,
        k=k,
        deterministic_batch_size=deterministic_batch_size,
    ):
        if chunk_index in completed_chunk_indices:
            continue
        pending = [task for task in chunk if task.shuffle_index not in completed_shuffle_indices]
        if pending:
            yield chunk_index, pending


def _reduce_metric_chunk_payloads(
    collected_chunks: Mapping[
        int,
        Tuple[
            Dict[str, Dict[int | str, float]],
            Dict[str, Dict[int | str, float]],
        ],
    ],
    metric_sums: Dict[str, Dict[int | str, float]],
    metric_sq_sums: Dict[str, Dict[int | str, float]],
) -> None:
    """Reduce chunk-level metric payloads in deterministic chunk order."""

    for chunk_index in sorted(collected_chunks):
        sums, sqs = collected_chunks[chunk_index]
        for label in METRIC_LABELS:
            for strategy, value in sums[label].items():
                metric_sums[label][strategy] += value
            for strategy, value in sqs[label].items():
                metric_sq_sums[label][strategy] += value


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
    metric_chunk_directory: Path | None = None,
    num_shuffles: int = NUM_SHUFFLES,
    strategies: Sequence[ThresholdStrategy] | None = None,
    resume: bool = True,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    workload_plan: TournamentWorkloadPlan | None = None,
    workload_plan_path: Path | None = None,
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
        Worker processes to spawn after normalization via
        :func:`farkle.utils.parallel.normalize_n_jobs`: ``None`` uses ``1``,
        ``0`` uses ``os.cpu_count()``, and positive values are used directly.
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
    if num_shuffles != cfg.num_shuffles:
        cfg.num_shuffles = num_shuffles
    if cfg.n_players < 2:
        raise ValueError("n_players must be ≥2")
    cfg.n_strategies = len(strategies)
    if cfg.n_strategies % cfg.n_players != 0:
        raise ValueError(f"n_players must divide {cfg.n_strategies:,}")
    if cfg.deterministic_batch_size < 1:
        raise ValueError("deterministic_batch_size must be positive")

    checkpoint_meta = {
        "n_players": cfg.n_players,
        "num_shuffles": cfg.num_shuffles,
        "global_seed": global_seed,
        "n_strategies": cfg.n_strategies,
        "rng_scheme_version": urandom.RNG_SCHEME_VERSION,
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        "tournament_method_version": TOURNAMENT_METHOD_VERSION,
        "rng_bit_generator": "PCG64DXSM",
        "coordinate_contract_version": 2,
        "shuffle_purpose_namespace": int(urandom.RandomPurpose.TOURNAMENT_SHUFFLE),
        "shuffle_permutation_purpose_namespace": int(urandom.RandomPurpose.SHUFFLE_PERMUTATION),
        "game_purpose_namespace": int(urandom.RandomPurpose.TOURNAMENT_GAME),
        "player_purpose_namespace": int(urandom.RandomPurpose.TOURNAMENT_PLAYER),
        "deterministic_batch_size": cfg.deterministic_batch_size,
    }
    if checkpoint_metadata:
        checkpoint_meta.update(checkpoint_metadata)
    if workload_plan is not None:
        checkpoint_meta.update(
            {
                "workload_plan_version": workload_plan.plan_version,
                "screening_resolution_delta": workload_plan.resolution_delta,
                "screening_interval_confidence": workload_plan.confidence,
                "batch_count": workload_plan.batch_count,
                "shuffles_per_batch": workload_plan.shuffles_per_batch,
                "batch_construction": workload_plan.batch_construction,
            }
        )

    games_per_sec = _measure_throughput(strategies[: cfg.n_players])
    if workload_plan is not None:
        if (
            workload_plan.k != cfg.n_players
            or workload_plan.strategy_count != cfg.n_strategies
            or workload_plan.required_shuffles != cfg.num_shuffles
            or workload_plan.shuffles_per_batch != cfg.deterministic_batch_size
        ):
            raise ValueError(
                "Tournament workload plan does not match the resolved run configuration"
            )
        workload_plan = workload_plan.with_games_per_second(games_per_sec)
        if workload_plan_path is not None:
            write_workload_plan(workload_plan_path, workload_plan)
        LOGGER.info(
            "Tournament workload projection complete",
            extra={"stage": "simulation", **workload_plan.to_dict()},
        )
        if workload_plan.cap_exceeded:
            raise WorkloadCapExceeded(workload_plan)
    shuffles_per_chunk = cfg.deterministic_batch_size
    LOGGER.debug(
        "Resolved immutable process-block sizing",
        extra={
            "stage": "simulation",
            "n_players": cfg.n_players,
            "games_per_shuffle": cfg.games_per_shuffle,
            "shuffles_per_process_block": shuffles_per_chunk,
            "projected_games_per_second": games_per_sec,
        },
    )

    win_totals = OutcomeCounter()
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
        persisted_meta = payload.get("meta", {}) if isinstance(payload, Mapping) else {}
        if (
            isinstance(persisted_meta, Mapping)
            and "rng_scheme_version" in persisted_meta
            and persisted_meta.get("rng_scheme_version") != urandom.RNG_SCHEME_VERSION
        ):
            raise ValueError(
                "Checkpoint RNG scheme is stale or unsupported; restart from a v2 output root"
            )
        raw_counts = payload.get("win_totals", payload)
        outcome_counts = payload.get("outcome_counts")
        win_totals = _coerce_counter(
            raw_counts, outcome_counts if isinstance(outcome_counts, Mapping) else None
        )
        games_completed = win_totals.games_completed
    can_resume_from_artifacts = resume and payload is not None

    collect_rows = row_output_directory is not None
    if collect_rows:
        assert row_output_directory is not None
        row_output_directory.mkdir(parents=True, exist_ok=True)
    row_manifest_path = (row_output_directory / "manifest.jsonl") if row_output_directory else None

    metrics_manifest_path: Path | None = None
    if metric_chunk_directory is not None:
        metric_chunk_directory.mkdir(parents=True, exist_ok=True)
        metrics_manifest_path = metric_chunk_directory / "metrics_manifest.jsonl"

    if collect_metrics or collect_rows:
        metric_sums = _coerce_metric_sums(payload.get("metric_sums") if payload else None)
        metric_sq_sums = _coerce_metric_sums(payload.get("metric_square_sums") if payload else None)
        if metric_sq_sums is None and payload:
            metric_sq_sums = _coerce_metric_sums(payload.get("metric_sq_sums"))
    else:
        metric_sums = None
        metric_sq_sums = None

    checkpoint_payload_meta = payload.get("meta", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(checkpoint_payload_meta, Mapping):
        checkpoint_payload_meta = {}
    completed_shuffle_indices = {
        int(value) for value in checkpoint_payload_meta.get("completed_shuffle_indices", [])
    }
    if can_resume_from_artifacts and row_manifest_path is not None and row_manifest_path.exists():
        completed_shuffle_indices |= _manifest_int_set(row_manifest_path, "shuffle_index")
        if completed_shuffle_indices:
            LOGGER.info(
                "Recovered completed semantic shuffle coordinates from row manifest",
                extra={
                    "stage": "simulation",
                    "completed_shuffles": len(completed_shuffle_indices),
                    "remaining_shuffles": max(0, cfg.num_shuffles - len(completed_shuffle_indices)),
                    "manifest_path": str(row_manifest_path),
                },
            )

    completed_chunk_indices = {
        int(value) for value in checkpoint_payload_meta.get("completed_process_block_indices", [])
    }
    if (
        can_resume_from_artifacts
        and metrics_manifest_path is not None
        and metrics_manifest_path.exists()
    ):
        completed_chunk_indices |= _manifest_int_set(metrics_manifest_path, "process_block_index")
        completed_chunk_indices |= _manifest_int_set(metrics_manifest_path, "chunk_index")
        if completed_chunk_indices:
            total_chunks = (cfg.num_shuffles + shuffles_per_chunk - 1) // shuffles_per_chunk
            LOGGER.info(
                "Recovered completed metric chunks from manifest",
                extra={
                    "stage": "simulation",
                    "completed_chunks": len(completed_chunk_indices),
                    "remaining_chunks": max(0, total_chunks - len(completed_chunk_indices)),
                    "manifest_path": str(metrics_manifest_path),
                },
            )

    checkpoint_meta["completed_shuffle_indices"] = sorted(completed_shuffle_indices)
    checkpoint_meta["completed_process_block_indices"] = sorted(completed_chunk_indices)

    recovered_metric_state = False
    if can_resume_from_artifacts and row_manifest_path is not None and row_manifest_path.exists():
        win_totals, metric_sums, metric_sq_sums = _load_row_manifest_aggregates(row_manifest_path)
        games_completed = win_totals.games_completed
        recovered_metric_state = collect_metrics or collect_rows
        LOGGER.info(
            "Recovered aggregates from row shards",
            extra={
                "stage": "simulation",
                "games_completed": games_completed,
                "manifest_path": str(row_manifest_path),
            },
        )
    elif (
        can_resume_from_artifacts
        and metrics_manifest_path is not None
        and metrics_manifest_path.exists()
    ):
        recovered_wins, recovered_sums, recovered_sq_sums = _load_metric_chunk_aggregates(
            metrics_manifest_path
        )
        metric_sums = recovered_sums
        metric_sq_sums = recovered_sq_sums
        recovered_metric_state = collect_metrics or collect_rows
        if recovered_wins is not None:
            win_totals = recovered_wins
            games_completed = win_totals.games_completed
        else:
            LOGGER.warning(
                "Metric chunk recovery could not rebuild win totals; falling back to checkpoint counts",
                extra={
                    "stage": "simulation",
                    "manifest_path": str(metrics_manifest_path),
                    "checkpoint_path": str(ckpt_path),
                },
            )
        LOGGER.info(
            "Recovered aggregates from metric chunks",
            extra={
                "stage": "simulation",
                "games_completed": games_completed,
                "manifest_path": str(metrics_manifest_path),
            },
        )

    if metric_chunk_directory is None and (collect_metrics or collect_rows):
        if metric_sums is None:
            metric_sums, metric_sq_sums = _empty_metric_aggregates()
    elif (
        metric_chunk_directory is not None
        and not recovered_metric_state
        and (metric_sums is None or metric_sq_sums is None)
    ):
        metric_sums = None
        metric_sq_sums = None

    mp_context = parallel.resolve_mp_context(cfg.mp_start_method)
    chunk_items = _iter_pending_chunk_items(
        num_shuffles=cfg.num_shuffles,
        shuffles_per_chunk=shuffles_per_chunk,
        global_seed=global_seed,
        k=cfg.n_players,
        deterministic_batch_size=cfg.deterministic_batch_size,
        completed_shuffle_indices=completed_shuffle_indices,
        completed_chunk_indices=completed_chunk_indices,
    )
    remaining_chunk_count = sum(
        1
        for _ in _iter_pending_chunk_items(
            num_shuffles=cfg.num_shuffles,
            shuffles_per_chunk=shuffles_per_chunk,
            global_seed=global_seed,
            k=cfg.n_players,
            deterministic_batch_size=cfg.deterministic_batch_size,
            completed_shuffle_indices=completed_shuffle_indices,
            completed_chunk_indices=completed_chunk_indices,
        )
    )

    resolved_n_jobs = parallel.normalize_n_jobs(n_jobs)

    LOGGER.info(
        "Tournament run start",
        extra={
            "stage": "simulation",
            "n_players": cfg.n_players,
            "num_shuffles": cfg.num_shuffles,
            "global_seed": global_seed,
            "n_jobs": resolved_n_jobs,
            "chunks": remaining_chunk_count,
            "collect_metrics": collect_metrics,
            "collect_rows": collect_rows,
            "row_dir": str(row_output_directory) if row_output_directory else None,
            "checkpoint_path": str(ckpt_path),
            "resume": resume,
        },
    )
    total_games = cfg.num_shuffles * cfg.games_per_shuffle
    checkpoint_progress = ScheduledProgressLogger(
        LOGGER,
        label=f"Tournament {cfg.n_players}p checkpoint",
        schedule=cfg.progress_logging,
        unit="games",
        total=total_games,
    )

    if collect_metrics or collect_rows:
        manifest_path = row_manifest_path
        chunk_fn: Callable[[Sequence[ShuffleTask]], object] = partial(
            _run_chunk_metrics,
            collect_rows=collect_rows,
            row_dir=row_output_directory,
            manifest_path=manifest_path,
        )
    else:
        chunk_fn = _run_chunk

    collected_metric_chunks: dict[
        int,
        Tuple[
            Dict[str, Dict[int | str, float]],
            Dict[str, Dict[int | str, float]],
        ],
    ] = {}
    chunk_wrapper = partial(_run_chunk_item, chunk_fn=chunk_fn)

    try:
        for chunk_index, block_tasks, result in parallel.process_map(
            chunk_wrapper,
            chunk_items,
            n_jobs=resolved_n_jobs,
            initializer=_init_worker,
            initargs=(strategies, cfg),
            window=4 * resolved_n_jobs,
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
                win_totals.absorb(wins)
                chunk_games = (
                    wins.games_completed
                    if isinstance(wins, OutcomeCounter)
                    else int(sum(wins.values()))
                )
                games_completed = win_totals.games_completed
                if metric_chunk_directory is not None:
                    chunk_path = metric_chunk_directory / f"metrics_{chunk_index:06d}.parquet"
                    rows = [
                        {
                            "metric": label,
                            "strategy": strat,
                            "sum": val,
                            "square_sum": sqs[label][strat],
                            "wins": int(wins.get(strat, 0)),
                        }
                        for label in METRIC_LABELS
                        for strat in sorted(
                            set(sums[label])
                            | set(
                                wins.attempted_exposures
                                if isinstance(wins, OutcomeCounter)
                                else wins
                            ),
                            key=str,
                        )
                        for val in [sums[label].get(strat, 0.0)]
                    ]
                    for row in rows:
                        strategy = cast(int | str, row["strategy"])
                        row["attempted_exposures"] = int(
                            wins.attempted_exposures.get(strategy, 0)
                            if isinstance(wins, OutcomeCounter)
                            else wins.get(strategy, 0)
                        )
                        row["completed_exposures"] = int(
                            wins.completed_exposures.get(strategy, 0)
                            if isinstance(wins, OutcomeCounter)
                            else wins.get(strategy, 0)
                        )
                        row["safety_limit_exposures"] = int(
                            wins.safety_limit_exposures.get(strategy, 0)
                            if isinstance(wins, OutcomeCounter)
                            else 0
                        )
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
                            "process_block_index": chunk_index,
                            "root_seed": global_seed,
                            "n_players": cfg.n_players,
                            "deterministic_batch_id": block_tasks[0].deterministic_batch_id,
                            "shuffle_index_start": block_tasks[0].shuffle_index,
                            "shuffle_index_end": block_tasks[-1].shuffle_index,
                            "shuffle_count": len(block_tasks),
                            "shuffle_indices": [task.shuffle_index for task in block_tasks],
                            "shuffle_seeds": [task.shuffle_seed for task in block_tasks],
                            "rng_scheme_version": urandom.RNG_SCHEME_VERSION,
                            "rng_purpose_namespace": int(urandom.RandomPurpose.TOURNAMENT_SHUFFLE),
                            "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
                            "tournament_method_version": TOURNAMENT_METHOD_VERSION,
                        },
                    )
                    LOGGER.info(
                        "Metrics chunk written",
                        extra={
                            "stage": "simulation",
                            "chunk_index": chunk_index,
                            "rows": len(rows),
                            "path": str(chunk_path),
                        },
                    )
                    collected_metric_chunks[chunk_index] = (sums, sqs)
                else:
                    collected_metric_chunks[chunk_index] = (sums, sqs)
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
                win_totals.absorb(chunk_wins)
                chunk_games = int(sum(chunk_wins.values()))
                games_completed = win_totals.games_completed
                LOGGER.debug(
                    "Chunk processed",
                    extra={
                        "stage": "simulation",
                        "chunk_index": chunk_index,
                        "wins": chunk_games,
                    },
                )

            completed_shuffle_indices.update(task.shuffle_index for task in block_tasks)
            completed_chunk_indices.add(chunk_index)
            checkpoint_meta["completed_shuffle_indices"] = sorted(completed_shuffle_indices)
            checkpoint_meta["completed_process_block_indices"] = sorted(completed_chunk_indices)
            now = time.perf_counter()
            if now - last_ckpt >= cfg.ckpt_every_sec:
                _save_checkpoint(
                    ckpt_path,
                    win_totals,
                    (metric_sums if (collect_metrics or collect_rows) else None),
                    (metric_sq_sums if (collect_metrics or collect_rows) else None),
                    meta=checkpoint_meta,
                )
                checkpoint_progress.maybe_log(
                    win_totals.games_attempted,
                    detail=f"chunk {chunk_index + 1}/{remaining_chunk_count}, checkpoint {ckpt_path.name}",
                    extra={
                        "stage": "simulation",
                        "chunk_index": chunk_index,
                        "chunks_total": remaining_chunk_count,
                        "games": games_completed,
                        "games_completed": games_completed,
                        "games_attempted": win_totals.games_attempted,
                        "games_safety_limit": win_totals.games_safety_limit,
                        "path": str(ckpt_path),
                    },
                )
                last_ckpt = now

        if (
            (collect_metrics or collect_rows)
            and metric_sums is not None
            and metric_sq_sums is not None
        ):
            _reduce_metric_chunk_payloads(collected_metric_chunks, metric_sums, metric_sq_sums)

        if (
            metric_chunk_directory is not None
            and (collect_metrics or collect_rows)
            and (metric_sums is None or metric_sq_sums is None)
        ):
            if metrics_manifest_path is None or not metrics_manifest_path.exists():
                raise FileNotFoundError(
                    f"Missing metrics manifest required for resume-safe recovery: {metrics_manifest_path}"
                )
            recovered_wins, metric_sums, metric_sq_sums = _load_metric_chunk_aggregates(
                metrics_manifest_path
            )
            if recovered_wins is not None:
                win_totals = recovered_wins
                games_completed = win_totals.games_completed

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
        "Tournament run complete after %d attempted games",
        win_totals.games_attempted,
        extra={
            "stage": "simulation",
            "games": win_totals.games_attempted,
            "games_attempted": win_totals.games_attempted,
            "games_completed": games_completed,
            "games_safety_limit": win_totals.games_safety_limit,
            "n_players": cfg.n_players,
            "chunks": remaining_chunk_count,
            "checkpoint_path": str(ckpt_path),
        },
    )
