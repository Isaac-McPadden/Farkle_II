"""Stream unconditional player-exposure sufficient statistics by simulation batch."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.config import AppConfig, ArtifactScope
from farkle.game.engine import TerminationStatus
from farkle.utils.artifact_contract import make_artifact_sidecar
from farkle.utils.manifest import iter_manifest
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION, TOURNAMENT_METHOD_VERSION
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
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
    pa.field("strategy", pa.int32(), nullable=False),
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


def _empty_accumulator() -> defaultdict[str, float]:
    values: defaultdict[str, float] = defaultdict(float)
    for name in (*_CORE_COUNT_FIELDS, *_CORE_SUM_FIELDS):
        values[name] = 0.0
    for suffix in _BEHAVIOR_SUFFIXES:
        values[f"raw_{suffix}_observations"] = 0.0
        values[f"raw_{suffix}_sum"] = 0.0
        values[f"raw_{suffix}_square_sum"] = 0.0
    return values


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


def _require_positive_int(row: Mapping[str, Any], field: str, *, source: Path) -> int:
    value = row.get(field)
    if value is None:
        raise ValueError(
            f"{source} is missing required {field!r} values; rerun simulation and curation "
            "under the coordinate-and-turn row contract"
        )
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{source} contains nonpositive {field!r}: {integer}")
    return integer


def _update_exposure(
    accumulator: defaultdict[str, float],
    row: Mapping[str, Any],
    *,
    seat: int,
    source: Path,
) -> None:
    prefix = f"P{seat}_"
    strategy_value = row.get(f"{prefix}strategy")
    score_value = row.get(f"{prefix}score")
    if strategy_value is None or score_value is None:
        raise ValueError(
            f"{source} lacks strategy/final-score values for seat {seat}; "
            "retired row schemas cannot satisfy unconditional all-player metrics"
        )
    score = float(score_value)
    n_turns = _require_positive_int(row, f"{prefix}n_turns", source=source)
    n_rounds = _require_positive_int(row, "n_rounds", source=source)
    exact_return = score / n_turns
    proxy_return = score / n_rounds
    turn_difference = n_turns - n_rounds
    hit_max_rounds = row.get(f"{prefix}hit_max_rounds")
    if hit_max_rounds is None:
        raise ValueError(
            f"{source} is missing maximum-round abort status for seat {seat}; "
            "rerun simulation and curation under the turn row contract"
        )

    if row.get("outcome_schema_version") != OUTCOME_SCHEMA_VERSION:
        raise ValueError(f"{source} is not outcome-schema-v{OUTCOME_SCHEMA_VERSION} compatible")
    try:
        status = TerminationStatus(row["termination_status"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{source} contains an invalid termination_status") from exc
    won = row.get("winner_seat") == f"P{seat}"
    if status is TerminationStatus.SAFETY_LIMIT and row.get("winner_seat") is not None:
        raise ValueError(f"{source} fabricates a winner for a safety-limit attempt")

    accumulator["raw_player_game_exposures"] += 1
    accumulator["raw_completed_player_game_exposures"] += int(
        status is TerminationStatus.COMPLETED
    )
    accumulator["raw_safety_limit_player_game_exposures"] += int(
        status is TerminationStatus.SAFETY_LIMIT
    )
    accumulator["raw_wins"] += int(won)
    accumulator["raw_losses"] += int(not won)
    accumulator["raw_final_score_sum"] += score
    accumulator["raw_final_score_square_sum"] += score * score
    accumulator["raw_n_turns_sum"] += n_turns
    accumulator["raw_n_turns_square_sum"] += n_turns * n_turns
    accumulator["raw_turn_return_game_weighted_exact_sum"] += exact_return
    accumulator["raw_turn_return_game_weighted_exact_square_sum"] += exact_return**2
    accumulator["raw_turn_return_round_proxy_sum"] += proxy_return
    accumulator["raw_turn_return_round_proxy_square_sum"] += proxy_return**2
    accumulator["raw_turn_round_mismatch_count"] += int(turn_difference != 0)
    accumulator["raw_max_round_abort_exposures"] += int(bool(hit_max_rounds))
    accumulator["raw_turn_minus_rounds_sum"] += turn_difference
    accumulator["raw_turn_minus_rounds_square_sum"] += turn_difference**2

    for suffix in _BEHAVIOR_SUFFIXES:
        value = row.get(f"{prefix}{suffix}")
        if value is None:
            continue
        numeric = float(value)
        accumulator[f"raw_{suffix}_observations"] += 1
        accumulator[f"raw_{suffix}_sum"] += numeric
        accumulator[f"raw_{suffix}_square_sum"] += numeric * numeric


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


def _iter_batch_tables(source: Path, k: int) -> Iterator[pa.Table]:
    parquet_file = pq.ParquetFile(source)
    required = _required_source_columns(k)
    missing = sorted(set(required).difference(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(
            f"{source} cannot produce unconditional all-player metrics; missing columns: {missing}"
        )

    current_coordinate: tuple[int, int, int] | None = None
    accumulators: dict[int, defaultdict[str, float]] = {}

    def _flush() -> pa.Table | None:
        if current_coordinate is None or not accumulators:
            return None
        root_seed, row_k, batch_id = current_coordinate
        rows = [
            _finish_row(root_seed, row_k, batch_id, strategy, accumulators[strategy])
            for strategy in sorted(accumulators)
        ]
        table = pa.Table.from_pylist(rows, schema=all_player_batch_schema())
        validate_unconditional_all_player_schema(table.schema)
        return table

    for record_batch in parquet_file.iter_batches(columns=required):
        for row in record_batch.to_pylist():
            root_value = row.get("root_seed")
            k_value = row.get("k")
            batch_value = row.get("deterministic_batch_id")
            if root_value is None or k_value is None or batch_value is None:
                raise ValueError(
                    f"{source} contains rows without root/k/batch coordinates; rerun simulation"
                )
            coordinate = (int(root_value), int(k_value), int(batch_value))
            if coordinate[1] != k:
                raise ValueError(f"{source} contains k={coordinate[1]} in canonical k={k} input")
            if current_coordinate is not None and coordinate < current_coordinate:
                raise ValueError(f"{source} is not ordered by root, k, and deterministic batch")
            if current_coordinate is not None and coordinate != current_coordinate:
                table = _flush()
                if table is not None:
                    yield table
                accumulators = {}
            current_coordinate = coordinate
            for seat in range(1, k + 1):
                strategy_value = row.get(f"P{seat}_strategy")
                if strategy_value is None:
                    raise ValueError(f"{source} is missing a strategy exposure for seat {seat}")
                strategy = int(strategy_value)
                _update_exposure(
                    accumulators.setdefault(strategy, _empty_accumulator()),
                    row,
                    seat=seat,
                    source=source,
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
    run_streaming_shard(
        out_path=str(output),
        manifest_path=str(manifest),
        schema=all_player_batch_schema(),
        batch_iter=_iter_batch_tables(source, k),
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
