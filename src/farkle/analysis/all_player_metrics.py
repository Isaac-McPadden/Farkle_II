"""Stream unconditional player-exposure sufficient statistics by simulation batch."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from farkle.config import AppConfig, ArtifactScope
from farkle.game.engine import TerminationStatus
from farkle.utils.arrow_batches import iter_parquet_tables_by_bytes
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.manifest import iter_manifest
from farkle.utils.parallel import (
    ProcessTreeMemoryGuard,
    apply_native_thread_limits,
    resolve_stage_parallel_policy,
)
from farkle.utils.release_identity import is_v3_config
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION, TOURNAMENT_METHOD_VERSION
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.strategy_ids import STRATEGY_ID_ARROW_TYPE
from farkle.utils.streaming_loop import run_streaming_shard

ATTEMPT_CONDITIONING: Final[str] = "all_attempted_player_game_exposures_safety_limit_is_loss"

_BEHAVIOR_SUFFIXES: Final[tuple[str, ...]] = (
    "rank",
    "loss_margin",
    "rolls",
    "farkles",
    "highest_turn",
    "hot_dice",
    "smart_five_uses",
    "n_smart_five_dice",
    "smart_one_uses",
    "n_smart_one_dice",
)

_IDENTITY_FIELDS: Final[list[pa.Field]] = [
    pa.field("root_seed", pa.int64(), nullable=False),
    pa.field("k", pa.int16(), nullable=False),
    pa.field("deterministic_batch_id", pa.int32(), nullable=False),
    pa.field("strategy", STRATEGY_ID_ARROW_TYPE, nullable=False),
]
_CORE_COUNT_FIELDS: Final[tuple[str, ...]] = (
    "raw_player_game_exposures",
    "raw_completed_player_game_exposures",
    "raw_safety_limit_player_game_exposures",
    "raw_wins",
    "raw_losses",
    "raw_turn_round_mismatch_count",
    "raw_max_round_abort_exposures",
)
_CORE_SUM_FIELDS: Final[tuple[str, ...]] = (
    "raw_final_score_sum",
    "raw_final_score_square_sum",
    "raw_n_turns_sum",
    "raw_n_turns_square_sum",
    "raw_turn_return_game_weighted_exact_sum",
    "raw_turn_return_game_weighted_exact_square_sum",
    "raw_turn_return_round_proxy_sum",
    "raw_turn_return_round_proxy_square_sum",
    "raw_turn_minus_rounds_sum",
    "raw_turn_minus_rounds_square_sum",
)
_DERIVED_FIELDS: Final[tuple[str, ...]] = (
    "turn_return_turn_weighted",
    "turn_return_game_weighted_exact",
    "turn_return_round_proxy",
    "round_proxy_gap",
    "round_proxy_relative_gap",
    "turn_round_mismatch_prevalence",
    "win_rate_per_attempt",
    "win_rate_given_completion",
    "safety_limit_exposure_rate",
)
_ACCUMULATOR_FIELDS: Final[tuple[str, ...]] = (
    *_CORE_COUNT_FIELDS,
    *_CORE_SUM_FIELDS,
    *(
        name
        for suffix in _BEHAVIOR_SUFFIXES
        for name in (
            f"raw_{suffix}_observations",
            f"raw_{suffix}_sum",
            f"raw_{suffix}_square_sum",
        )
    ),
)
_ACCUMULATOR_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(_ACCUMULATOR_FIELDS)
}


def all_player_batch_schema() -> pa.Schema:
    """Return the stable unconditional batch-metric schema."""

    behavior_fields: list[pa.Field] = []
    for suffix in _BEHAVIOR_SUFFIXES:
        behavior_fields.extend(
            [
                pa.field(f"raw_{suffix}_observations", pa.int64(), nullable=False),
                pa.field(f"raw_{suffix}_sum", pa.float64(), nullable=False),
                pa.field(f"raw_{suffix}_square_sum", pa.float64(), nullable=False),
            ]
        )
    return pa.schema(
        [
            *_IDENTITY_FIELDS,
            *(pa.field(name, pa.int64(), nullable=False) for name in _CORE_COUNT_FIELDS),
            *(pa.field(name, pa.float64(), nullable=False) for name in _CORE_SUM_FIELDS),
            *behavior_fields,
            *(pa.field(name, pa.float64()) for name in _DERIVED_FIELDS),
        ]
    )


def validate_unconditional_all_player_schema(schema: pa.Schema) -> None:
    """Reject conditional fields and incomplete unconditional metric schemas."""

    conditional = sorted(name for name in schema.names if name.startswith("win_conditioned_"))
    if conditional:
        raise ValueError(
            "unconditional all-player metrics cannot contain winner-conditioned fields: "
            f"{conditional}"
        )
    required = set(all_player_batch_schema().names)
    missing = sorted(required.difference(schema.names))
    if missing:
        raise ValueError(f"unconditional all-player metric schema is incomplete: {missing}")


class _ColumnAccumulators:
    """Dense numeric accumulators keyed by the compact strategy-ID domain."""

    def __init__(self) -> None:
        self.strategy_to_index: dict[int, int] = {}
        self.strategies: list[int] = []
        self.values = np.zeros((0, len(_ACCUMULATOR_FIELDS)), dtype=np.float64)

    def indices(self, strategies: np.ndarray[Any, np.dtype[np.int32]]) -> np.ndarray:
        new_strategies = [
            int(strategy)
            for strategy in np.unique(strategies)
            if int(strategy) not in self.strategy_to_index
        ]
        if new_strategies:
            start = len(self.strategies)
            self.strategies.extend(new_strategies)
            self.strategy_to_index.update(
                (strategy, start + offset) for offset, strategy in enumerate(new_strategies)
            )
            self.values = np.vstack(
                (
                    self.values,
                    np.zeros(
                        (len(new_strategies), len(_ACCUMULATOR_FIELDS)),
                        dtype=np.float64,
                    ),
                )
            )
        return np.fromiter(
            (self.strategy_to_index[int(strategy)] for strategy in strategies),
            dtype=np.intp,
            count=len(strategies),
        )

    def add(self, indices: np.ndarray, field: str, values: np.ndarray | float | int) -> None:
        # add.at is deliberately unbuffered. It preserves source-row addition
        # order across arbitrary execution-batch boundaries.
        np.add.at(self.values[:, _ACCUMULATOR_INDEX[field]], indices, values)

    def mapping(self, strategy: int) -> dict[str, float]:
        row = self.values[self.strategy_to_index[strategy]]
        return {name: float(row[index]) for index, name in enumerate(_ACCUMULATOR_FIELDS)}


def _required_source_columns(k: int) -> list[str]:
    columns = [
        "root_seed",
        "k",
        "deterministic_batch_id",
        "winner_seat",
        "termination_status",
        "outcome_schema_version",
        "n_rounds",
    ]
    for seat in range(1, k + 1):
        columns.extend(
            [
                f"P{seat}_strategy",
                f"P{seat}_score",
                f"P{seat}_n_turns",
                f"P{seat}_hit_max_rounds",
                *(f"P{seat}_{suffix}" for suffix in _BEHAVIOR_SUFFIXES),
            ]
        )
    return columns


def _projected_row_width(schema: pa.Schema, columns: Sequence[str]) -> int:
    """Return a conservative deterministic width used for the row ceiling."""

    width = 0
    for name in columns:
        data_type = schema.field(name).type
        if pa.types.is_boolean(data_type):
            width += 2
        elif pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
            width += 40
        elif pa.types.is_integer(data_type) or pa.types.is_floating(data_type):
            width += max(1, int(data_type.bit_width) // 8) + 1
        else:
            width += 64
    return max(1, width)


def _execution_batch_limits(cfg: AppConfig, source: Path, k: int) -> tuple[int, int]:
    """Resolve the k/schema-aware byte and row ceilings for projected reads."""

    schema = pq.read_schema(source)
    columns = _required_source_columns(k)
    max_bytes = int(
        cfg.resources.stage_batch_bytes.get(
            "all_player_metrics",
            cfg.resources.stage_batch_bytes["analysis"],
        )
    )
    estimated_width = _projected_row_width(schema, columns)
    max_rows = max(1, min(int(cfg.row_group_size), max_bytes // estimated_width))
    return max_bytes, max_rows


def _numpy_column(
    table: pa.Table,
    name: str,
    dtype: np.dtype[Any],
) -> np.ndarray:
    array = table.column(name).combine_chunks()
    return np.asarray(array.to_numpy(zero_copy_only=False), dtype=dtype)


def _boolean_comparison(table: pa.Table, name: str, value: str) -> np.ndarray:
    values = np.asarray(
        table.column(name).combine_chunks().to_numpy(zero_copy_only=False),
        dtype=object,
    )
    return np.asarray(values == value, dtype=bool)


def _seat_exposure_columns(
    table: pa.Table,
    *,
    seat: int,
    source: Path,
    completed: np.ndarray,
    safety: np.ndarray,
    n_rounds: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Decode and validate one seat into compact numeric columns."""

    prefix = f"P{seat}_"
    strategy_column = table.column(f"{prefix}strategy")
    score_column = table.column(f"{prefix}score")
    if strategy_column.null_count or score_column.null_count:
        raise ValueError(
            f"{source} lacks strategy/final-score values for seat {seat}; "
            "retired row schemas cannot satisfy unconditional all-player metrics"
        )
    strategies = _numpy_column(table, f"{prefix}strategy", np.dtype(np.int32))
    scores = _numpy_column(table, f"{prefix}score", np.dtype(np.float64))
    if table.column(f"{prefix}n_turns").null_count:
        raise ValueError(
            f"{source} is missing required {prefix}n_turns values; rerun simulation and "
            "curation under the coordinate-and-turn row contract"
        )
    turns = _numpy_column(table, f"{prefix}n_turns", np.dtype(np.float64))
    invalid_turns = ~np.isfinite(turns) | (turns < 0) | (completed & (turns < 1))
    invalid_rounds = ~np.isfinite(n_rounds) | (n_rounds < 0) | (completed & (n_rounds < 1))
    if np.any(invalid_turns):
        raise ValueError(
            f"{source} contains invalid {prefix}n_turns values under the termination contract"
        )
    if np.any(invalid_rounds):
        raise ValueError(
            f"{source} contains invalid n_rounds values under the termination contract"
        )
    if np.any(((turns == 0) | (n_rounds == 0)) & (scores != 0)):
        raise ValueError(
            f"{source} contains a positive final score with a zero turn/round denominator"
        )
    hit_max_column = table.column(f"{prefix}hit_max_rounds")
    if hit_max_column.null_count:
        raise ValueError(
            f"{source} is missing maximum-round abort status for seat {seat}; "
            "rerun simulation and curation under the turn row contract"
        )
    hit_max = _numpy_column(table, f"{prefix}hit_max_rounds", np.dtype(bool))
    won = _boolean_comparison(table, "winner_seat", f"P{seat}")
    if np.any(safety & ~pc.is_null(table.column("winner_seat")).to_numpy(zero_copy_only=False)):
        raise ValueError(f"{source} fabricates a winner for a safety-limit attempt")

    exact_return = np.divide(scores, turns, out=np.zeros_like(scores), where=turns != 0)
    proxy_return = np.divide(scores, n_rounds, out=np.zeros_like(scores), where=n_rounds != 0)
    turn_difference = turns - n_rounds
    columns = {
        "raw_player_game_exposures": np.ones(len(strategies), dtype=np.int8),
        "raw_completed_player_game_exposures": completed,
        "raw_safety_limit_player_game_exposures": safety,
        "raw_wins": won,
        "raw_losses": ~won,
        "raw_final_score_sum": scores,
        "raw_final_score_square_sum": scores * scores,
        "raw_n_turns_sum": turns,
        "raw_n_turns_square_sum": turns * turns,
        "raw_turn_return_game_weighted_exact_sum": exact_return,
        "raw_turn_return_game_weighted_exact_square_sum": exact_return * exact_return,
        "raw_turn_return_round_proxy_sum": proxy_return,
        "raw_turn_return_round_proxy_square_sum": proxy_return * proxy_return,
        "raw_turn_round_mismatch_count": turn_difference != 0,
        "raw_max_round_abort_exposures": hit_max,
        "raw_turn_minus_rounds_sum": turn_difference,
        "raw_turn_minus_rounds_square_sum": turn_difference * turn_difference,
    }
    for suffix in _BEHAVIOR_SUFFIXES:
        behavior = table.column(f"{prefix}{suffix}")
        present_bool = np.asarray(pc.is_valid(behavior).to_numpy(zero_copy_only=False), dtype=bool)
        values = _numpy_column(table, f"{prefix}{suffix}", np.dtype(np.float64))
        present = present_bool.astype(np.int8)
        numeric = np.where(present_bool, values, 0.0)
        columns[f"raw_{suffix}_observations"] = present
        columns[f"raw_{suffix}_sum"] = numeric
        columns[f"raw_{suffix}_square_sum"] = numeric * numeric
    return strategies, columns


def _update_exposure_columns(
    accumulators: _ColumnAccumulators,
    table: pa.Table,
    *,
    k: int,
    source: Path,
    completed: np.ndarray,
    safety: np.ndarray,
    n_rounds: np.ndarray,
) -> None:
    """Update all seats in stable source-row/seat order using numeric columns."""

    seat_payloads = [
        _seat_exposure_columns(
            table,
            seat=seat,
            source=source,
            completed=completed,
            safety=safety,
            n_rounds=n_rounds,
        )
        for seat in range(1, k + 1)
    ]
    strategies = np.stack([payload[0] for payload in seat_payloads], axis=1).reshape(-1)
    indices = accumulators.indices(strategies)
    for field in _ACCUMULATOR_FIELDS:
        values = np.stack([payload[1][field] for payload in seat_payloads], axis=1).reshape(-1)
        accumulators.add(indices, field, values)


def _finish_row(
    root_seed: int,
    k: int,
    deterministic_batch_id: int,
    strategy: int,
    values: Mapping[str, float],
) -> dict[str, int | float | None]:
    exposures = int(values["raw_player_game_exposures"])
    completed_exposures = int(values["raw_completed_player_game_exposures"])
    safety_exposures = int(values["raw_safety_limit_player_game_exposures"])
    wins = int(values["raw_wins"])
    losses = int(values["raw_losses"])
    if exposures != completed_exposures + safety_exposures:
        raise ValueError("attempted exposures must equal completed plus safety-limit exposures")
    if losses != exposures - wins or wins > completed_exposures:
        raise ValueError("win/loss exposure conservation failed")
    if int(values["raw_max_round_abort_exposures"]) != safety_exposures:
        raise ValueError("maximum-round exposure count disagrees with termination status")
    turns = values["raw_n_turns_sum"]
    turn_weighted = values["raw_final_score_sum"] / turns if turns else None
    game_exact = (
        values["raw_turn_return_game_weighted_exact_sum"] / exposures if exposures else None
    )
    round_proxy = values["raw_turn_return_round_proxy_sum"] / exposures if exposures else None
    gap = round_proxy - game_exact if round_proxy is not None and game_exact is not None else None
    relative_gap = gap / game_exact if gap is not None and game_exact else None
    row: dict[str, int | float | None] = {
        "root_seed": root_seed,
        "k": k,
        "deterministic_batch_id": deterministic_batch_id,
        "strategy": strategy,
        **{name: int(values[name]) for name in _CORE_COUNT_FIELDS},
        **{name: float(values[name]) for name in _CORE_SUM_FIELDS},
        "turn_return_turn_weighted": turn_weighted,
        "turn_return_game_weighted_exact": game_exact,
        "turn_return_round_proxy": round_proxy,
        "round_proxy_gap": gap,
        "round_proxy_relative_gap": relative_gap,
        "turn_round_mismatch_prevalence": (
            values["raw_turn_round_mismatch_count"] / exposures if exposures else None
        ),
        "win_rate_per_attempt": wins / exposures if exposures else None,
        "win_rate_given_completion": wins / completed_exposures if completed_exposures else None,
        "safety_limit_exposure_rate": safety_exposures / exposures if exposures else None,
    }
    for suffix in _BEHAVIOR_SUFFIXES:
        row[f"raw_{suffix}_observations"] = int(values[f"raw_{suffix}_observations"])
        row[f"raw_{suffix}_sum"] = float(values[f"raw_{suffix}_sum"])
        row[f"raw_{suffix}_square_sum"] = float(values[f"raw_{suffix}_square_sum"])
    return row


def _iter_batch_tables(
    source: Path,
    k: int,
    *,
    max_batch_bytes: int,
    max_batch_rows: int,
    memory_guard: ProcessTreeMemoryGuard,
) -> Iterator[pa.Table]:
    parquet_file = pq.ParquetFile(source)
    required = _required_source_columns(k)
    missing = sorted(set(required).difference(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(
            f"{source} cannot produce unconditional all-player metrics; missing columns: {missing}"
        )

    current_coordinate: tuple[int, int, int] | None = None
    accumulators = _ColumnAccumulators()

    def _flush() -> pa.Table | None:
        if current_coordinate is None or not accumulators.strategies:
            return None
        root_seed, row_k, batch_id = current_coordinate
        rows = [
            _finish_row(root_seed, row_k, batch_id, strategy, accumulators.mapping(strategy))
            for strategy in sorted(accumulators.strategies)
        ]
        table = pa.Table.from_pylist(rows, schema=all_player_batch_schema())
        validate_unconditional_all_player_schema(table.schema)
        return table

    stream = iter_parquet_tables_by_bytes(
        source,
        columns=required,
        max_batch_bytes=max_batch_bytes,
        max_batch_rows=max_batch_rows,
        use_threads=False,
    )
    for _row_group, _execution_batch, table in stream:
        memory_guard.check_before_schedule()
        if table.nbytes > max_batch_bytes:
            raise MemoryError("projected all-player batch crossed its configured byte ceiling")
        if any(table.column(field.name).null_count for field in _IDENTITY_FIELDS[:3]):
            raise ValueError(
                f"{source} contains rows without root/k/batch coordinates; rerun simulation"
            )
        roots = _numpy_column(table, "root_seed", np.dtype(np.int64))
        player_counts = _numpy_column(table, "k", np.dtype(np.int16))
        batch_ids = _numpy_column(table, "deterministic_batch_id", np.dtype(np.int32))
        change = (
            np.flatnonzero(
                (roots[1:] != roots[:-1])
                | (player_counts[1:] != player_counts[:-1])
                | (batch_ids[1:] != batch_ids[:-1])
            )
            + 1
        )
        boundaries = np.concatenate(([0], change, [table.num_rows]))
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            coordinate = (int(roots[start]), int(player_counts[start]), int(batch_ids[start]))
            if coordinate[1] != k:
                raise ValueError(f"{source} contains k={coordinate[1]} in canonical k={k} input")
            if current_coordinate is not None and coordinate < current_coordinate:
                raise ValueError(f"{source} is not ordered by root, k, and deterministic batch")
            if current_coordinate is not None and coordinate != current_coordinate:
                finished = _flush()
                if finished is not None:
                    yield finished
                accumulators = _ColumnAccumulators()
            current_coordinate = coordinate
            segment = table.slice(int(start), int(stop - start))
            versions = _numpy_column(segment, "outcome_schema_version", np.dtype(np.int64))
            if np.any(versions != OUTCOME_SCHEMA_VERSION):
                raise ValueError(
                    f"{source} is not outcome-schema-v{OUTCOME_SCHEMA_VERSION} compatible"
                )
            completed = _boolean_comparison(
                segment, "termination_status", TerminationStatus.COMPLETED.value
            )
            safety = _boolean_comparison(
                segment, "termination_status", TerminationStatus.SAFETY_LIMIT.value
            )
            if np.any(~(completed | safety)) or np.any(completed & safety):
                raise ValueError(f"{source} contains an invalid termination_status")
            rounds = _numpy_column(segment, "n_rounds", np.dtype(np.float64))
            _update_exposure_columns(
                accumulators,
                segment,
                k=k,
                source=source,
                completed=completed,
                safety=safety,
                n_rounds=rounds,
            )

    final_table = _flush()
    if final_table is not None:
        yield final_table


def build_all_player_batch_metrics(
    cfg: AppConfig,
    k: int,
    *,
    force: bool = False,
) -> Path:
    """Build the canonical unconditional player-exposure artifact for one k."""

    source = cfg.ingested_rows_curated(k)
    if is_v3_config(cfg):
        validate_artifact_sidecar(
            source,
            expected={
                "scope": ArtifactScope.BY_K.value,
                "operation": "curate_game_rows",
            },
        )
    if not source.exists():
        raise FileNotFoundError(source)
    output = cfg.metrics_all_player_batch_path(k)
    manifest = output.with_suffix(".manifest.jsonl")
    done = stage_done_path(output.parent, "all_player_batch_metrics")
    if not force and stage_is_up_to_date(
        done,
        inputs=[source],
        outputs=[output, manifest],
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=[output],
    ):
        validate_unconditional_all_player_schema(pq.read_schema(output))
        return output

    manifest.unlink(missing_ok=True)
    sidecar = make_artifact_sidecar(
        cfg,
        output,
        producer="metrics",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation="aggregate_player_batch_statistics",
        method_contract={
            "kind": "turn_metrics",
            "procedure": "aggregate_player_batch_statistics",
            "parameters": {
                "exposure_denominator": "player_game_exposure",
                "completed_diagnostic_denominator": "completed_player_game_exposure",
                "safety_limit_numerator": "safety_limit_player_game_exposure",
                "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
                "tournament_method_version": TOURNAMENT_METHOD_VERSION,
            },
        },
        source_artifacts=[source],
        consistency_columns=all_player_batch_schema().names,
        grouping_keys=["root_seed", "k", "deterministic_batch_id", "strategy"],
        player_counts=[k],
        required_player_counts=[k],
        missing_cell_policy="fail",
        replication_unit="deterministic_shuffle_batch",
        conditioning=ATTEMPT_CONDITIONING,
    )
    policy = resolve_stage_parallel_policy("analysis", cfg.analysis, resources=cfg.resources)
    apply_native_thread_limits(policy)
    memory_guard = ProcessTreeMemoryGuard(
        cfg.resources.rss_abort_mb,
        cfg.resources.rss_sample_interval_seconds,
    )
    memory_guard.check_before_schedule(force=True)
    max_batch_bytes, max_batch_rows = _execution_batch_limits(cfg, source, k)
    run_streaming_shard(
        out_path=str(output),
        manifest_path=str(manifest),
        schema=all_player_batch_schema(),
        batch_iter=_iter_batch_tables(
            source,
            k,
            max_batch_bytes=max_batch_bytes,
            max_batch_rows=max_batch_rows,
            memory_guard=memory_guard,
        ),
        row_group_size=cfg.row_group_size,
        compression=cfg.parquet_codec,
        manifest_extra={
            "path": output.name,
            "root_seed": cfg.sim.seed,
            "k": k,
            "grouping_keys": ["root_seed", "k", "deterministic_batch_id", "strategy"],
        },
        sidecar=sidecar,
    )
    records = list(iter_manifest(manifest))
    if len(records) != 1:
        raise RuntimeError(
            f"expected exactly one manifest entry for {output}, found {len(records)}"
        )
    validate_unconditional_all_player_schema(pq.read_schema(output))
    write_stage_done(
        done,
        inputs=[source],
        outputs=[output, manifest],
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=[output],
    )
    return output


__all__ = [
    "all_player_batch_schema",
    "ATTEMPT_CONDITIONING",
    "build_all_player_batch_metrics",
    "validate_unconditional_all_player_schema",
]
