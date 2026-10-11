"""Exact outcome arithmetic for bounded curated-row slices (Task 1A).

These are partial reductions, not certified publications. Task 1B must merge
them, authenticate the complete workload/roster, and certify support before
publishing verified zeros. No missing observation is filled with a zero here.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from farkle.config import AppConfig, ArtifactScope
from farkle.utils.arrow_batches import iter_parquet_tables_by_bytes
from farkle.utils.artifact_contract import validate_artifact_sidecar
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION, expected_schema_for
from farkle.utils.strategy_ids import STRATEGY_ID_ARROW_TYPE

OUTCOME_METHOD_VERSION: Final = 1
DISTRIBUTIONS: Final = (
    "completed_score",
    "competition_rank",
    "losing_margin",
    "midrank",
    "stopped_score",
    "winning_margin",
)
_COUNTS: Final = (
    "attempted",
    "completed",
    "safety",
    "wins",
    "losses_completed",
    "deficit_sum",
    "victory_margin_sum",
)
_ROOT_FIELDS: Final[list[pa.Field]] = [
    pa.field("root_seed", pa.int64(), nullable=False),
    pa.field("k", pa.int16(), nullable=False),
]
_SEAT_FIELDS: Final[list[pa.Field]] = [
    pa.field("strategy", STRATEGY_ID_ARROW_TYPE, nullable=False),
    pa.field("seat", pa.int16(), nullable=False),
]
_BATCH_FIELD: Final[pa.Field] = pa.field("deterministic_batch_id", pa.int32(), nullable=False)
_BK: Final[list[pa.Field]] = [*_ROOT_FIELDS, _BATCH_FIELD, *_SEAT_FIELDS]
_K: Final[list[pa.Field]] = [*_ROOT_FIELDS, *_SEAT_FIELDS]
_FREQUENCY_FIELDS: Final[list[pa.Field]] = [
    pa.field("distribution", pa.string(), nullable=False),
    pa.field("value", pa.int64(), nullable=False),
    pa.field("value_scale", pa.int8(), nullable=False),
    pa.field("count", pa.int64(), nullable=False),
]
_STAT_FIELDS: Final[list[pa.Field]] = [
    *(pa.field(name, pa.int64(), nullable=False) for name in _COUNTS),
    pa.field("support_status", pa.string(), nullable=False),
]
_FREQUENCY_DTYPE: Final = np.dtype(
    [
        ("root_seed", "<i8"),
        ("k", "<i2"),
        ("deterministic_batch_id", "<i4"),
        ("strategy", "<i4"),
        ("seat", "<i2"),
        ("distribution", "i1"),
        ("value", "<i8"),
    ]
)
_STATS_DTYPE: Final = np.dtype(
    [
        ("root_seed", "<i8"),
        ("k", "<i2"),
        ("deterministic_batch_id", "<i4"),
        ("strategy", "<i4"),
        ("seat", "<i2"),
        *((name, "<i8") for name in _COUNTS),
    ]
)
_INT64_MAX: Final = np.iinfo(np.int64).max


def outcome_frequency_schema(*, batch: bool = False) -> pa.Schema:
    """Return F, or its batch-indexed partial schema; midranks use scale two."""
    return pa.schema([*(_BK if batch else _K), *_FREQUENCY_FIELDS])


def outcome_support_schema(*, batch: bool = False) -> pa.Schema:
    """Return S, or B. Partial reductions contain observed actual-seat cells."""
    return pa.schema([*(_BK if batch else _K), *_STAT_FIELDS])


def outcome_coordinate_schema() -> pa.Schema:
    """Retain consumed game keys for later exact batch-coverage certification."""
    return pa.schema(
        [
            *_ROOT_FIELDS,
            _BATCH_FIELD,
            pa.field("shuffle_index", pa.int64(), nullable=False),
            pa.field("game_index", pa.int32(), nullable=False),
            pa.field("termination_status", pa.string(), nullable=False),
        ]
    )


def outcome_source_columns(k: int) -> list[str]:
    """Project existing outcome fields from actual seats only; ignore padding."""
    return [
        "root_seed",
        "k",
        "deterministic_batch_id",
        "shuffle_index",
        "game_index",
        "outcome_schema_version",
        "termination_status",
        "hit_safety_limit",
        "winner_seat",
        "winner_strategy",
        "winning_score",
        "victory_margin",
        "seat_ranks",
        *(
            f"P{seat}_{suffix}"
            for seat in range(1, k + 1)
            for suffix in ("strategy", "score", "rank", "loss_margin", "hit_max_rounds")
        ),
    ]


def _slice_row_limit(k: int, max_bytes: int) -> int:
    # Reserve Q/4 each for projection, fixed records, grouping, and output views.
    # 4096 bytes/exposure conservatively admits Arrow/Python conversion and
    # simultaneous typed views; it is not a claimed process RSS ceiling.
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("outcome byte allowance must be a positive integer")
    limit = max_bytes // (4 * 4096 * k)
    if limit < 1:
        raise MemoryError(f"outcome slice needs at least {4 * 4096 * k} bytes for one game")
    return limit


def _validate_design(root_seed: int, k: int, shuffles_per_batch: int) -> None:
    for name, value in (
        ("root_seed", root_seed),
        ("k", k),
        ("shuffles_per_batch", shuffles_per_batch),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if not 0 <= root_seed <= _INT64_MAX or not 2 <= k <= 127 or shuffles_per_batch < 1:
        raise ValueError(
            "outcome design requires a nonnegative root, k=2..127, and positive batch width"
        )


def _project(table: pa.Table, k: int) -> pa.Table:
    columns = outcome_source_columns(k)
    expected = expected_schema_for(k)
    for name in columns:
        if table.schema.names.count(name) != 1:
            raise ValueError(f"outcome source requires exactly one {name} column")
        if table.schema.field(name).type != expected.field(name).type:
            raise ValueError(f"outcome source has incompatible type for {name}")
    return table.select(columns)


def _checked_group(table: pa.Table, schema: pa.Schema, sums: tuple[str, ...]) -> pa.Table:
    """Reduce one admitted slice with exact integer sums and stable key order.

    This is not the external frequency merge required for production histories.
    """
    if not table.num_rows:
        return pa.Table.from_batches([], schema=schema)
    for name in sums:
        values = table[name].to_numpy()
        if np.any(values < 0):
            raise ValueError("outcome counts and sums must be nonnegative")
        # The fast bound avoids Python integer summation for ordinary counts.
        if int(values.max()) > _INT64_MAX // len(values) and sum(map(int, values)) > _INT64_MAX:
            raise OverflowError(f"outcome {name} sum exceeds int64")
    keys = [name for name in schema.names if name not in sums]
    grouped = table.group_by(keys, use_threads=False).aggregate([(name, "sum") for name in sums])
    grouped = grouped.rename_columns([*keys, *sums]).select(schema.names)
    return grouped.sort_by([(name, "ascending") for name in keys]).cast(schema)


def _views(table: pa.Table, schema: pa.Schema, sums: tuple[str, ...]) -> pa.Table:
    actual = table.drop(["deterministic_batch_id"])
    overall = actual.set_column(
        actual.schema.get_field_index("seat"),
        schema.field("seat"),
        pa.array(np.zeros(actual.num_rows, dtype=np.int16)),
    )
    return _checked_group(pa.concat_tables([actual, overall]), schema, sums)


@dataclass(frozen=True)
class OutcomeSlice:
    """Partial exact counts; none of these tables certifies complete support.

    Batch tables keep actual seats only. Views add seat=0 by summing seats,
    never by averaging rates. Empty conditional histograms are interpreted
    through explicit attempted/completed/safety/win/loss support counts.
    """

    batch_frequencies: pa.Table
    batch_stats: pa.Table
    coordinates: pa.Table

    def frequency_views(self) -> pa.Table:
        """Return partial F for actual seats and the count-weighted overall view."""
        return _views(self.batch_frequencies, outcome_frequency_schema(), ("count",))

    def support_views(self) -> pa.Table:
        """Return partial S; absent cells await certification, never zero-fill."""
        return _views(self.batch_stats, outcome_support_schema(), _COUNTS)


def _outcome(row: dict[str, Any], k: int) -> tuple[list[int], list[int], int | None]:
    """Validate recorded outcomes without imposing distinct-strategy seating.

    The simulation-row validator requires distinct strategies; these analysis
    contracts also cover repeated-strategy exposures. Official ordinal ranks
    and tie order are checked here and never replaced by descriptive ranks.
    """
    if row["outcome_schema_version"] != OUTCOME_SCHEMA_VERSION:
        raise ValueError("incompatible outcome_schema_version")
    strategies = [row[f"P{s}_strategy"] for s in range(1, k + 1)]
    scores = [row[f"P{s}_score"] for s in range(1, k + 1)]
    if any(x is None or x < 0 for x in [*strategies, *scores]):
        raise ValueError("missing or negative actual-seat strategy/score")
    ranks = [row[f"P{s}_rank"] for s in range(1, k + 1)]
    margins = [row[f"P{s}_loss_margin"] for s in range(1, k + 1)]
    status = row["termination_status"]
    if status not in {"completed", "safety_limit"}:
        raise ValueError("missing or invalid termination_status")
    stopped = status == "safety_limit"
    if row["hit_safety_limit"] is not stopped or any(
        row[f"P{s}_hit_max_rounds"] is not stopped for s in range(1, k + 1)
    ):
        raise ValueError("termination status and safety flags disagree")
    if stopped:
        if any(
            row[name] is not None
            for name in ("winner_seat", "winner_strategy", "winning_score", "victory_margin")
        ):
            raise ValueError("safety-limit game cannot have a winner or victory margin")
        if (
            any(value is not None for value in [*ranks, *margins])
            or row["seat_ranks"] != [None] * k
        ):
            raise ValueError("safety-limit game cannot have ranks or losing margins")
        return strategies, scores, None
    order = sorted(range(k), key=lambda i: (-scores[i], i))
    official = [0] * k
    for rank, i in enumerate(order, 1):
        official[i] = rank
    winner = order[0]
    if ranks != official or row["seat_ranks"] != [f"P{i + 1}" for i in order]:
        raise ValueError("recorded engine ranks disagree with score/seat tie order")
    if row["winner_seat"] != f"P{winner + 1}" or row["winner_strategy"] != strategies[winner]:
        raise ValueError("official winner disagrees with engine ranks or strategy")
    if (
        row["winning_score"] != scores[winner]
        or row["victory_margin"] != scores[winner] - scores[order[1]]
    ):
        raise ValueError("inconsistent recorded winning score or victory margin")
    if margins != [scores[winner] - score for score in scores]:
        raise ValueError("inconsistent recorded player loss margin")
    return strategies, scores, winner


def accumulate_outcome_slice(
    table: pa.Table,
    *,
    root_seed: int,
    k: int,
    shuffles_per_batch: int,
    max_bytes: int = 32 * 1024 * 1024,
) -> OutcomeSlice:
    """Count exact outcomes in one bounded, canonically ordered source slice.

    `shuffles_per_batch` is the resolved workload-plan width, not an I/O size.
    Partial slices may start/end inside a statistical batch. Required coverage,
    roster multiplicities, and verified-zero certification belong to Task 1B.
    """
    _validate_design(root_seed, k, shuffles_per_batch)
    limit = _slice_row_limit(k, max_bytes)
    if table.num_rows > limit:
        raise MemoryError(f"outcome slice exceeds {limit} admitted games; split the source slice")
    projected = _project(table, k)
    if projected.nbytes > max_bytes // 4:
        raise MemoryError("outcome projection exceeds its quarter-budget allocation")
    # Reject malformed variable-length lists before Python expansion.
    if projected["seat_ranks"].null_count or np.any(
        np.asarray(pc.list_value_length(projected["seat_ranks"])) != k
    ):
        raise ValueError("seat_ranks must contain exactly k entries per game")
    frequencies = np.empty(4 * k * table.num_rows, dtype=_FREQUENCY_DTYPE)
    stats = np.empty(k * table.num_rows, dtype=_STATS_DTYPE)
    used = 0
    previous: tuple[int, int] | None = None
    # Only this admitted slice becomes Python rows; expanded outcomes stay in
    # fixed-width arrays rather than an unbounded histogram dictionary.
    for game, row in enumerate(projected.to_pylist()):
        shuffle, game_index, batch = (
            row[name] for name in ("shuffle_index", "game_index", "deterministic_batch_id")
        )
        if row["root_seed"] != root_seed or row["k"] != k:
            raise ValueError("outcome slice mixes or lacks configured root/k support")
        if any(value is None or value < 0 for value in (shuffle, game_index, batch)):
            raise ValueError("missing or negative game coordinate")
        if batch != shuffle // shuffles_per_batch:
            raise ValueError("deterministic batch disagrees with workload-plan shuffle width")
        coordinate = (shuffle, game_index)
        if previous is not None and coordinate <= previous:
            raise ValueError("duplicate or out-of-order game coordinate")
        previous = coordinate
        strategies, scores, winner = _outcome(row, k)
        completed = winner is not None
        victory = row["victory_margin"] if completed else 0
        for seat, (strategy, score) in enumerate(zip(strategies, scores, strict=True), 1):
            key = (root_seed, k, batch, strategy, seat)
            win = completed and seat - 1 == winner
            loss = completed and not win
            deficit = scores[winner] - score if loss and winner is not None else 0
            stats[game * k + seat - 1] = (
                *key,
                1,
                int(completed),
                int(not completed),
                int(win),
                int(loss),
                deficit,
                victory if win else 0,
            )
            if not completed:
                frequencies[used] = (*key, 4, score)
                used += 1
                continue
            rank = 1 + sum(other > score for other in scores)
            midrank2 = 2 * rank + sum(other == score for other in scores) - 1
            for distribution, value in (
                (0, score),
                (1, rank),
                (3, midrank2),
                (5 if win else 2, victory if win else deficit),
            ):
                frequencies[used] = (*key, distribution, value)
                used += 1
    unique, counts = np.unique(frequencies[:used], return_counts=True)
    frequency_schema = outcome_frequency_schema(batch=True)
    arrays = [pa.array(unique[field.name], type=field.type) for field in _BK]
    arrays.extend(
        [
            pa.array([DISTRIBUTIONS[int(i)] for i in unique["distribution"]], type=pa.string()),
            pa.array(unique["value"], type=pa.int64()),
            pa.array(np.where(unique["distribution"] == 3, 2, 1), type=pa.int8()),
            pa.array(counts, type=pa.int64()),
        ]
    )
    batch_frequencies = pa.Table.from_arrays(arrays, schema=frequency_schema)
    stat_schema = outcome_support_schema(batch=True)
    stat_table = pa.Table.from_arrays(
        [
            *(pa.array(stats[field.name], type=field.type) for field in list(stat_schema)[:-1]),
            pa.array(["observed"] * len(stats), type=pa.string()),
        ],
        schema=stat_schema,
    )
    batch_stats = _checked_group(stat_table, stat_schema, _COUNTS)
    coordinates = projected.select(outcome_coordinate_schema().names).cast(
        outcome_coordinate_schema()
    )
    return OutcomeSlice(batch_frequencies, batch_stats, coordinates)


def iter_curated_outcome_slices(
    cfg: AppConfig,
    k: int,
    *,
    shuffles_per_batch: int,
    max_batch_rows: int | None = None,
) -> Iterator[OutcomeSlice]:
    """Read authenticated current-run curated rows into bounded partial counts.

    This read-only adapter publishes nothing. The durable producer must bind
    the supplied plan width to authenticated workload/roster/coverage inputs.
    """
    _validate_design(cfg.sim.seed, k, shuffles_per_batch)
    if k not in cfg.sim.n_players_list:
        raise ValueError("outcome k is not in the configured workload")
    source = cfg.ingested_rows_curated(k)
    if not source.is_file():
        raise FileNotFoundError(source)
    validate_artifact_sidecar(
        source,
        expected={
            "scope": ArtifactScope.BY_K.value,
            "operation": "curate_game_rows",
            "player_counts": [k],
            "required_player_counts": [k],
            "missing_cell_policy": "fail",
        },
    )
    max_bytes = int(cfg.resources.stage_batch_bytes["analysis"])
    limit = _slice_row_limit(k, max_bytes)
    if max_batch_rows is not None:
        if (
            isinstance(max_batch_rows, bool)
            or not isinstance(max_batch_rows, int)
            or max_batch_rows < 1
        ):
            raise ValueError("max_batch_rows must be positive")
        limit = min(limit, max_batch_rows)
    _project(pa.Table.from_batches([], schema=pq.read_schema(source)), k)
    previous: tuple[int, int] | None = None
    observed = False
    for _, _, table in iter_parquet_tables_by_bytes(
        source,
        columns=outcome_source_columns(k),
        max_batch_bytes=max_bytes // 4,
        max_batch_rows=limit,
    ):
        result = accumulate_outcome_slice(
            table,
            root_seed=cfg.sim.seed,
            k=k,
            shuffles_per_batch=shuffles_per_batch,
            max_bytes=max_bytes,
        )
        first = (
            int(result.coordinates["shuffle_index"][0].as_py()),
            int(result.coordinates["game_index"][0].as_py()),
        )
        if previous is not None and first <= previous:
            raise ValueError("duplicate or out-of-order game coordinate across source slices")
        previous = (
            int(result.coordinates["shuffle_index"][-1].as_py()),
            int(result.coordinates["game_index"][-1].as_py()),
        )
        observed = True
        yield result
    if not observed:
        raise ValueError("required curated source has no observed game support")


__all__ = [
    "DISTRIBUTIONS",
    "OUTCOME_METHOD_VERSION",
    "OutcomeSlice",
    "accumulate_outcome_slice",
    "iter_curated_outcome_slices",
    "outcome_coordinate_schema",
    "outcome_frequency_schema",
    "outcome_source_columns",
    "outcome_support_schema",
]
