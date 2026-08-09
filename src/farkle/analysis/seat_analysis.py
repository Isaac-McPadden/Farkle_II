"""Canonical within-k seat effects and explicitly secondary cross-k diagnostics."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.all_player_metrics import ATTEMPT_CONDITIONING
from farkle.config import AppConfig, ArtifactScope
from farkle.game.engine import TerminationStatus
from farkle.utils.arrow_batches import iter_parquet_tables_by_bytes
from farkle.utils.artifact_contract import (
    make_artifact_sidecar,
    sha256_file,
    validate_artifact_sidecar,
    write_artifact_with_sidecar_atomic,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.parallel import ProcessTreeMemoryGuard
from farkle.utils.partitioned_stage import (
    PartitionedStageIdentity,
    PartitionedUnit,
    resolved_code_identity_sha256,
    run_partitioned_stage,
    validate_final_manifest,
)
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.strategy_ids import STRATEGY_ID_ARROW_TYPE
from farkle.utils.streaming_loop import run_streaming_shard

_COUNT_SCHEMA: Final = pa.schema(
    [
        pa.field("root_seed", pa.int64(), nullable=False),
        pa.field("k", pa.int16(), nullable=False),
        pa.field("deterministic_batch_id", pa.int32(), nullable=False),
        pa.field("strategy", STRATEGY_ID_ARROW_TYPE, nullable=False),
        pa.field("seat", pa.int16(), nullable=False),
        pa.field("raw_wins", pa.int64(), nullable=False),
        pa.field("raw_exposures", pa.int64(), nullable=False),
        pa.field("raw_completed_exposures", pa.int64(), nullable=False),
        pa.field("raw_safety_limit_exposures", pa.int64(), nullable=False),
    ]
)

_STANDARDIZED_COLUMNS: Final = [
    "root_seed",
    "effect_scope",
    "strategy",
    "seat",
    "common_k_support",
    "standardized_seat_effect",
]
_MIXTURE_COLUMNS: Final = [
    "root_seed",
    "effect_scope",
    "strategy",
    "seat",
    "common_k_support",
    "raw_wins",
    "raw_exposures",
    "raw_completed_exposures",
    "raw_safety_limit_exposures",
    "exposure_weighted_baseline",
    "exposure_weighted_seat_effect",
]
_SELFPLAY_COLUMNS: Final = [
    "root_seed",
    "k",
    "strategy",
    "p1_wins",
    "games_attempted",
    "games_completed",
    "games_safety_limit",
    "p1_win_rate_per_attempt",
    "p1_win_rate_given_completion",
    "p1_effect_vs_chance",
]
_MIRRORED_COLUMNS: Final = [
    "root_seed",
    "k",
    "strategy_a",
    "strategy_b",
    "paired_mirrored_games",
    "games_attempted",
    "games_completed",
    "games_safety_limit",
    "unpaired_forward_games",
    "unpaired_reverse_games",
    "mean_p1_win_difference",
]
_MIRROR_SHARD_SCHEMA: Final = pa.schema(
    [
        pa.field("root_seed", pa.int64(), nullable=False),
        pa.field("deterministic_batch_id", pa.int32(), nullable=False),
        pa.field("strategy_a", STRATEGY_ID_ARROW_TYPE, nullable=False),
        pa.field("strategy_b", STRATEGY_ID_ARROW_TYPE, nullable=False),
        pa.field("paired_mirrored_games", pa.int64(), nullable=False),
        pa.field("p1_win_difference_sum", pa.int64(), nullable=False),
        pa.field("games_completed", pa.int64(), nullable=False),
        pa.field("games_safety_limit", pa.int64(), nullable=False),
        pa.field("unpaired_forward_games", pa.int64(), nullable=False),
        pa.field("unpaired_reverse_games", pa.int64(), nullable=False),
    ]
)
_MIRROR_METHOD_VERSION: Final = 2
_MIRROR_SCHEMA_VERSION: Final = 1


@dataclass(frozen=True)
class SeatAnalysisArtifacts:
    """Paths written by canonical seat analysis."""

    batch_counts: tuple[Path, ...]
    by_k: tuple[Path, ...]
    population_by_k: tuple[Path, ...]
    standardized_across_k: Path
    exposure_mixture_diagnostic: Path
    selfplay_diagnostic: Path
    mirrored_diagnostic: Path

    @property
    def all_paths(self) -> tuple[Path, ...]:
        return (
            *self.batch_counts,
            *self.by_k,
            *self.population_by_k,
            self.standardized_across_k,
            self.exposure_mixture_diagnostic,
            self.selfplay_diagnostic,
            self.mirrored_diagnostic,
        )


def _source_columns(k: int) -> list[str]:
    return [
        "root_seed",
        "k",
        "deterministic_batch_id",
        "shuffle_index",
        "game_index",
        "winner_seat",
        "termination_status",
        *(f"P{seat}_strategy" for seat in range(1, k + 1)),
    ]


def _iter_seat_count_tables(source: Path, k: int) -> Iterator[pa.Table]:
    parquet_file = pq.ParquetFile(source)
    columns = _source_columns(k)
    missing = sorted(set(columns).difference(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"{source} lacks canonical seat-analysis columns: {missing}")
    coordinate: tuple[int, int, int] | None = None
    counts: defaultdict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0, 0, 0])

    def _flush() -> pa.Table | None:
        if coordinate is None or not counts:
            return None
        root_seed, row_k, batch_id = coordinate
        rows = [
            {
                "root_seed": root_seed,
                "k": row_k,
                "deterministic_batch_id": batch_id,
                "strategy": strategy,
                "seat": seat,
                "raw_wins": values[0],
                "raw_exposures": values[1],
                "raw_completed_exposures": values[2],
                "raw_safety_limit_exposures": values[3],
            }
            for (strategy, seat), values in sorted(counts.items())
        ]
        return pa.Table.from_pylist(rows, schema=_COUNT_SCHEMA)

    for batch in parquet_file.iter_batches(columns=columns):
        values = batch.to_pydict()
        for index in range(batch.num_rows):
            current = (
                int(values["root_seed"][index]),
                int(values["k"][index]),
                int(values["deterministic_batch_id"][index]),
            )
            if current[1] != k:
                raise ValueError(f"{source} contains k={current[1]} in canonical k={k} input")
            if coordinate is not None and current < coordinate:
                raise ValueError(f"{source} is not ordered by root, k, and deterministic batch")
            if coordinate is not None and current != coordinate:
                table = _flush()
                if table is not None:
                    yield table
                counts = defaultdict(lambda: [0, 0, 0, 0])
            coordinate = current
            try:
                status = TerminationStatus(values["termination_status"][index])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{source} contains invalid termination status") from exc
            winner = values["winner_seat"][index]
            if status is TerminationStatus.SAFETY_LIMIT and winner is not None:
                raise ValueError(f"{source} credits a safety-limit winner")
            for seat in range(1, k + 1):
                strategy_value = values[f"P{seat}_strategy"][index]
                if strategy_value is None:
                    raise ValueError(f"{source} has a missing strategy exposure in seat {seat}")
                cell = counts[(int(strategy_value), seat)]
                cell[0] += int(winner == f"P{seat}")
                cell[1] += 1
                cell[2] += int(status is TerminationStatus.COMPLETED)
                cell[3] += int(status is TerminationStatus.SAFETY_LIMIT)
    table = _flush()
    if table is not None:
        yield table


def _write_batch_counts(cfg: AppConfig, k: int, source: Path, output: Path) -> None:
    manifest = output.with_suffix(".manifest.jsonl")
    manifest.unlink(missing_ok=True)
    sidecar = make_artifact_sidecar(
        cfg,
        output,
        producer="seat_analysis",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation="aggregate_seat_batch_exposures",
        baseline="chance_1_over_k",
        weighted_quantity="seat_win_indicator",
        support_count_role="raw_player_game_exposures",
        replication_unit="deterministic_shuffle_batch",
        conditioning=ATTEMPT_CONDITIONING,
        consistency_columns=_COUNT_SCHEMA.names,
        source_artifacts=[source],
        grouping_keys=["root_seed", "k", "deterministic_batch_id", "strategy", "seat"],
        player_counts=[k],
        required_player_counts=[k],
        missing_cell_policy="fail",
    )
    run_streaming_shard(
        out_path=str(output),
        manifest_path=str(manifest),
        schema=_COUNT_SCHEMA,
        batch_iter=_iter_seat_count_tables(source, k),
        row_group_size=cfg.row_group_size,
        compression=cfg.parquet_codec,
        manifest_extra={"root_seed": cfg.sim.seed, "k": k},
        sidecar=sidecar,
    )


def _validate_source(path: Path, k: int) -> int:
    """Validate a canonical by-k row partition and return its single root."""

    validate_artifact_sidecar(
        path,
        expected={
            "scope": ArtifactScope.BY_K.value,
            "source_scope": ArtifactScope.BY_K.value,
            "operation": "concatenate_rows_within_k",
            "player_counts": [k],
            "required_player_counts": [k],
            "missing_cell_policy": "fail",
        },
    )
    columns = _source_columns(k)
    schema = pq.read_schema(path)
    missing = sorted(set(columns).difference(schema.names))
    if missing:
        raise ValueError(f"{path} lacks canonical seat-analysis columns: {missing}")
    observed_k_set: set[int] = set()
    roots_set: set[int] = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=["root_seed", "k"]):
        values = batch.to_pydict()
        for root, observed_k_value in zip(values["root_seed"], values["k"], strict=True):
            if root is not None:
                roots_set.add(int(root))
            if observed_k_value is not None:
                observed_k_set.add(int(observed_k_value))
    observed_k = sorted(observed_k_set)
    if observed_k != [k]:
        raise ValueError(f"{path} has k support {observed_k}, expected [{k}]")
    roots = sorted(roots_set)
    if len(roots) != 1:
        raise ValueError(f"{path} must contain exactly one root, found {roots}")
    return roots[0]


def _within_k_frames(counts: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (
        counts["raw_exposures"]
        == counts["raw_completed_exposures"] + counts["raw_safety_limit_exposures"]
    ).all():
        raise ValueError("seat counts violate attempted exposure conservation")
    if (counts["raw_wins"] > counts["raw_completed_exposures"]).any():
        raise ValueError("seat counts credit a win outside completed exposure support")
    grouped = (
        counts.groupby(["root_seed", "k", "strategy", "seat"], as_index=False)
        .agg(
            raw_wins=("raw_wins", "sum"),
            raw_exposures=("raw_exposures", "sum"),
            raw_completed_exposures=("raw_completed_exposures", "sum"),
            raw_safety_limit_exposures=("raw_safety_limit_exposures", "sum"),
        )
        .sort_values(["strategy", "seat"])
    )
    grouped["chance_baseline"] = 1.0 / k
    grouped["win_rate"] = grouped["raw_wins"] / grouped["raw_exposures"]
    grouped["win_rate_per_attempt"] = grouped["win_rate"]
    grouped["win_rate_given_completion"] = grouped["raw_wins"].div(
        grouped["raw_completed_exposures"].replace(0, pd.NA)
    )
    grouped["safety_limit_exposure_rate"] = (
        grouped["raw_safety_limit_exposures"] / grouped["raw_exposures"]
    )
    grouped["raw_losses"] = grouped["raw_exposures"] - grouped["raw_wins"]
    grouped["seat_effect"] = grouped["win_rate"] - grouped["chance_baseline"]
    population = (
        counts.groupby(["root_seed", "k", "seat"], as_index=False)
        .agg(
            raw_wins=("raw_wins", "sum"),
            raw_exposures=("raw_exposures", "sum"),
            raw_completed_exposures=("raw_completed_exposures", "sum"),
            raw_safety_limit_exposures=("raw_safety_limit_exposures", "sum"),
        )
        .sort_values("seat")
    )
    population["chance_baseline"] = 1.0 / k
    population["win_rate"] = population["raw_wins"] / population["raw_exposures"]
    population["win_rate_per_attempt"] = population["win_rate"]
    population["win_rate_given_completion"] = population["raw_wins"].div(
        population["raw_completed_exposures"].replace(0, pd.NA)
    )
    population["safety_limit_exposure_rate"] = (
        population["raw_safety_limit_exposures"] / population["raw_exposures"]
    )
    population["raw_losses"] = population["raw_exposures"] - population["raw_wins"]
    population["seat_effect"] = population["win_rate"] - population["chance_baseline"]
    return grouped, population


def _declared_weights(cfg: AppConfig, ks: list[int]) -> tuple[dict[int, float], str, str]:
    if cfg.k_aggregation.method == "equal-k":
        weights = {k: 1.0 / len(ks) for k in ks}
        return weights, "equal_k_mean", "equal_k"
    configured = cfg.k_aggregation.k_weights or {}
    if set(configured) != set(ks):
        raise ValueError("declared seat standardization weights must cover every configured k")
    return (
        {int(k): float(weight) for k, weight in configured.items()},
        "declared_k_weighted_mean",
        "declared_mapping",
    )


def _standardized_frames(
    cfg: AppConfig,
    by_k: dict[int, pd.DataFrame],
    population_by_k: dict[int, pd.DataFrame],
    ks: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights, _, _ = _declared_weights(cfg, ks)
    common_seats = range(1, min(ks) + 1)
    common_strategies = set.intersection(
        *(set(frame["strategy"].astype(int)) for frame in by_k.values())
    )
    standardized: list[dict[str, Any]] = []
    mixture: list[dict[str, Any]] = []
    for strategy in sorted(common_strategies):
        for seat in common_seats:
            cells = [
                by_k[k].loc[(by_k[k]["strategy"] == strategy) & (by_k[k]["seat"] == seat)]
                for k in ks
            ]
            if any(cell.empty for cell in cells):
                continue
            effect = sum(
                float(cell.iloc[0]["seat_effect"]) * weights[k]
                for k, cell in zip(ks, cells, strict=True)
            )
            wins = sum(int(cell.iloc[0]["raw_wins"]) for cell in cells)
            exposures = sum(int(cell.iloc[0]["raw_exposures"]) for cell in cells)
            completed_exposures = sum(
                int(cell.iloc[0]["raw_completed_exposures"]) for cell in cells
            )
            safety_exposures = sum(
                int(cell.iloc[0]["raw_safety_limit_exposures"]) for cell in cells
            )
            baseline_mass = sum(
                int(cell.iloc[0]["raw_exposures"]) / k for k, cell in zip(ks, cells, strict=True)
            )
            standardized.append(
                {
                    "root_seed": int(cells[0].iloc[0]["root_seed"]),
                    "effect_scope": "strategy",
                    "strategy": strategy,
                    "seat": seat,
                    "common_k_support": ks,
                    "standardized_seat_effect": effect,
                }
            )
            mixture.append(
                {
                    "root_seed": int(cells[0].iloc[0]["root_seed"]),
                    "effect_scope": "strategy",
                    "strategy": strategy,
                    "seat": seat,
                    "common_k_support": ks,
                    "raw_wins": wins,
                    "raw_exposures": exposures,
                    "raw_completed_exposures": completed_exposures,
                    "raw_safety_limit_exposures": safety_exposures,
                    "exposure_weighted_baseline": baseline_mass / exposures,
                    "exposure_weighted_seat_effect": wins / exposures - baseline_mass / exposures,
                }
            )
    for seat in common_seats:
        cells = [population_by_k[k].loc[population_by_k[k]["seat"] == seat] for k in ks]
        if any(cell.empty for cell in cells):
            continue
        standardized.append(
            {
                "root_seed": int(cells[0].iloc[0]["root_seed"]),
                "effect_scope": "population",
                "strategy": None,
                "seat": seat,
                "common_k_support": ks,
                "standardized_seat_effect": sum(
                    float(cell.iloc[0]["seat_effect"]) * weights[k]
                    for k, cell in zip(ks, cells, strict=True)
                ),
            }
        )
        wins = sum(int(cell.iloc[0]["raw_wins"]) for cell in cells)
        exposures = sum(int(cell.iloc[0]["raw_exposures"]) for cell in cells)
        completed_exposures = sum(int(cell.iloc[0]["raw_completed_exposures"]) for cell in cells)
        safety_exposures = sum(int(cell.iloc[0]["raw_safety_limit_exposures"]) for cell in cells)
        baseline_mass = sum(
            int(cell.iloc[0]["raw_exposures"]) / k for k, cell in zip(ks, cells, strict=True)
        )
        mixture.append(
            {
                "root_seed": int(cells[0].iloc[0]["root_seed"]),
                "effect_scope": "population",
                "strategy": None,
                "seat": seat,
                "common_k_support": ks,
                "raw_wins": wins,
                "raw_exposures": exposures,
                "raw_completed_exposures": completed_exposures,
                "raw_safety_limit_exposures": safety_exposures,
                "exposure_weighted_baseline": baseline_mass / exposures,
                "exposure_weighted_seat_effect": wins / exposures - baseline_mass / exposures,
            }
        )
    standardized_frame = pd.DataFrame(standardized, columns=_STANDARDIZED_COLUMNS)
    mixture_frame = pd.DataFrame(mixture, columns=_MIXTURE_COLUMNS)
    standardized_frame["strategy"] = pd.array(
        standardized_frame["strategy"].tolist(), dtype="Int64"
    )
    mixture_frame["strategy"] = pd.array(mixture_frame["strategy"].tolist(), dtype="Int64")
    return standardized_frame, mixture_frame


def _game_diagnostics(sources: dict[int, Path]) -> pd.DataFrame:
    selfplay: defaultdict[tuple[int, int, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    for k, source in sources.items():
        columns = _source_columns(k)
        for batch in pq.ParquetFile(source).iter_batches(columns=columns):
            values = batch.to_pydict()
            for index in range(batch.num_rows):
                root = int(values["root_seed"][index])
                strategies = tuple(
                    int(values[f"P{seat}_strategy"][index]) for seat in range(1, k + 1)
                )
                status = TerminationStatus(values["termination_status"][index])
                winner = values["winner_seat"][index]
                if status is TerminationStatus.SAFETY_LIMIT and winner is not None:
                    raise ValueError(f"{source} credits a safety-limit winner")
                p1_win = int(winner == "P1")
                if len(set(strategies)) == 1:
                    cell = selfplay[(root, k, strategies[0])]
                    cell[0] += p1_win
                    cell[1] += 1
                    cell[2] += int(status is TerminationStatus.SAFETY_LIMIT)
    selfplay_rows: list[dict[str, Any]] = [
        {
            "root_seed": root,
            "k": k,
            "strategy": strategy,
            "p1_wins": values[0],
            "games_attempted": values[1],
            "games_completed": values[1] - values[2],
            "games_safety_limit": values[2],
            "p1_win_rate_per_attempt": values[0] / values[1],
            "p1_win_rate_given_completion": (
                values[0] / (values[1] - values[2]) if values[1] > values[2] else None
            ),
            "p1_effect_vs_chance": values[0] / values[1] - 1.0 / k,
        }
        for (root, k, strategy), values in sorted(selfplay.items())
    ]
    return pd.DataFrame(selfplay_rows, columns=_SELFPLAY_COLUMNS)


@dataclass(frozen=True)
class _MirroredBatch:
    root_seed: int
    batch_id: int
    rows: int


def _mirror_partition(a: int, b: int, partitions: int) -> int:
    """Return a hash-seed-independent canonical strategy-pair partition."""

    digest = hashlib.blake2b(f"{a}:{b}".encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % partitions


def _mirrored_batches(source: Path, *, max_batch_bytes: int) -> tuple[_MirroredBatch, ...]:
    """Discover ordered deterministic batches without materializing game records."""

    columns = ("root_seed", "k", "deterministic_batch_id")
    current: tuple[int, int] | None = None
    rows = 0
    result: list[_MirroredBatch] = []
    for _row_group, _batch, table in iter_parquet_tables_by_bytes(
        source,
        columns=columns,
        max_batch_bytes=max_batch_bytes,
        max_batch_rows=max(1, max_batch_bytes // 16),
    ):
        values = table.to_pydict()
        for index in range(table.num_rows):
            if int(values["k"][index]) != 2:
                continue
            coordinate = (
                int(values["root_seed"][index]),
                int(values["deterministic_batch_id"][index]),
            )
            if current is not None and coordinate < current:
                raise ValueError(f"{source} is not ordered by root and deterministic batch")
            if current is not None and coordinate != current:
                result.append(_MirroredBatch(current[0], current[1], rows))
                rows = 0
            current = coordinate
            rows += 1
    if current is not None:
        result.append(_MirroredBatch(current[0], current[1], rows))
    return tuple(result)


def _mirrored_units(
    batches: tuple[_MirroredBatch, ...], *, max_records: int
) -> tuple[PartitionedUnit, ...]:
    units: list[PartitionedUnit] = []
    for batch in batches:
        # A deliberately conservative fixed-record budget leaves room for Arrow
        # decode buffers, stable sorting, and two compact FIFO bit arrays.
        partitions = max(1, math.ceil(batch.rows / max_records))
        for partition in range(partitions):
            units.append(
                PartitionedUnit(
                    (batch.root_seed, batch.batch_id, partition, partitions),
                    (
                        f"root-{batch.root_seed}/batch-{batch.batch_id:08d}/"
                        f"part-{partition:04d}-of-{partitions:04d}.parquet"
                    ),
                )
            )
    return tuple(units)


@dataclass(frozen=True)
class _MirroredPartitionWriter:
    source: str
    columns: tuple[str, ...]
    max_batch_bytes: int
    max_records: int

    def __call__(self, unit: PartitionedUnit, output: Path) -> None:
        root, batch_id, partition, partitions = (int(value) for value in unit.key)
        records: list[tuple[int, int, int, int, int]] = []
        # The source is scanned only for a missing unit. Completed units are
        # authenticated and skipped by the shared partition stage on resume.
        for _rg, _bi, table in iter_parquet_tables_by_bytes(
            Path(self.source),
            columns=self.columns,
            max_batch_bytes=self.max_batch_bytes,
            max_batch_rows=max(1, self.max_batch_bytes // 32),
        ):
            values = table.to_pydict()
            for index in range(table.num_rows):
                if (
                    int(values["root_seed"][index]) != root
                    or int(values["deterministic_batch_id"][index]) != batch_id
                ):
                    continue
                first = values["P1_strategy"][index]
                second = values["P2_strategy"][index]
                if first is None or second is None:
                    raise ValueError("mirrored-game diagnostic has missing strategy exposure")
                first_i, second_i = int(first), int(second)
                if first_i == second_i:
                    continue
                a, b = sorted((first_i, second_i))
                if _mirror_partition(a, b, partitions) != partition:
                    continue
                status = TerminationStatus(values["termination_status"][index])
                winner = values["winner_seat"][index]
                if status is TerminationStatus.SAFETY_LIMIT:
                    if winner is not None:
                        raise ValueError("mirrored-game safety-limit observation has a winner")
                    records.append((a, b, 2, 0, len(records)))
                    continue
                if winner not in {"P1", "P2"}:
                    raise ValueError("mirrored-game completed observation has no valid winner")
                orientation = int((first_i, second_i) == (b, a))
                records.append((a, b, orientation, int(winner == "P1"), len(records)))
        if len(records) > self.max_records:
            raise MemoryError(
                "deterministic pair partition exceeds its assigned compact-record budget; "
                "increase deterministic partitioning before publication"
            )
        if not records:
            pq.write_table(pa.Table.from_pylist([], schema=_MIRROR_SHARD_SCHEMA), output)
            return
        raw = np.asarray(records, dtype=np.int64)
        order = np.lexsort((raw[:, 4], raw[:, 1], raw[:, 0]))
        raw = raw[order]
        rows: list[dict[str, int]] = []
        start = 0
        while start < raw.shape[0]:
            stop = start + 1
            while stop < raw.shape[0] and tuple(raw[stop, :2]) == tuple(raw[start, :2]):
                stop += 1
            a, b = (int(value) for value in raw[start, :2])
            forward = np.empty(stop - start, dtype=np.int8)
            reverse = np.empty(stop - start, dtype=np.int8)
            forward_head = reverse_head = forward_size = reverse_size = 0
            matched = difference = completed = safety = 0
            for orientation, p1 in raw[start:stop, 2:4]:
                if orientation == 2:
                    safety += 1
                else:
                    completed += 1
                    if orientation == 0:
                        if reverse_head < reverse_size:
                            difference += int(p1) - int(reverse[reverse_head])
                            reverse_head += 1
                            matched += 1
                        else:
                            forward[forward_size] = p1
                            forward_size += 1
                    elif forward_head < forward_size:
                        difference += int(forward[forward_head]) - int(p1)
                        forward_head += 1
                        matched += 1
                    else:
                        reverse[reverse_size] = p1
                        reverse_size += 1
            rows.append(
                {
                    "root_seed": root,
                    "deterministic_batch_id": batch_id,
                    "strategy_a": a,
                    "strategy_b": b,
                    "paired_mirrored_games": matched,
                    "p1_win_difference_sum": difference,
                    "games_completed": completed,
                    "games_safety_limit": safety,
                    "unpaired_forward_games": forward_size - forward_head,
                    "unpaired_reverse_games": reverse_size - reverse_head,
                }
            )
            start = stop
        pq.write_table(pa.Table.from_pylist(rows, schema=_MIRROR_SHARD_SCHEMA), output)


def _mirrored_identity(cfg: AppConfig, source: Path) -> PartitionedStageIdentity:
    return PartitionedStageIdentity(
        stage_name="seat_mirrored_pairing",
        root_seed=int(cfg.sim.seed),
        input_identities=(("combined_rows_k2", sha256_file(source)),),
        statistical_config_sha256=cfg.stage_config_sha("metrics"),
        code_identity_sha256=resolved_code_identity_sha256(cfg),
        schema_version=_MIRROR_SCHEMA_VERSION,
        method_version=_MIRROR_METHOD_VERSION,
    )


def _write_mirrored_diagnostic(
    cfg: AppConfig,
    *,
    source: Path,
    units_root: Path,
    units: tuple[PartitionedUnit, ...],
    manifest: Path,
    output: Path,
    guard: ProcessTreeMemoryGuard,
) -> None:
    """Stable merge authenticated batch shards without a global pair map."""

    sidecar = make_artifact_sidecar(
        cfg,
        output,
        producer="seat_analysis",
        scope=ArtifactScope.DIAGNOSTICS,
        source_scope=ArtifactScope.BY_K,
        operation="calculate_mirrored_game_diagnostics",
        baseline="chance_1_over_k",
        weighted_quantity="paired_p1_win_indicator_difference",
        support_count_role="matched_unmatched_and_termination_partition_counts",
        uncertainty_method="descriptive",
        replication_unit="within_batch_count_matched_opposite_orientation_game_pair",
        conditioning='termination_status == "completed"',
        consistency_columns=_MIRRORED_COLUMNS,
        source_artifacts=[source],
        input_manifests=[manifest],
        grouping_keys=["root_seed", "strategy_a", "strategy_b"],
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="explicit_zero_matched_support",
    )

    def write(staged: Path) -> None:
        guard.check_before_schedule(force=True)
        output_schema = pa.schema(
            [
                pa.field("root_seed", pa.int64()),
                pa.field("k", pa.int16()),
                pa.field("strategy_a", STRATEGY_ID_ARROW_TYPE),
                pa.field("strategy_b", STRATEGY_ID_ARROW_TYPE),
                pa.field("paired_mirrored_games", pa.int64()),
                pa.field("games_attempted", pa.int64()),
                pa.field("games_completed", pa.int64()),
                pa.field("games_safety_limit", pa.int64()),
                pa.field("unpaired_forward_games", pa.int64()),
                pa.field("unpaired_reverse_games", pa.int64()),
                pa.field("mean_p1_win_difference", pa.float64()),
            ]
        )
        writer = pq.ParquetWriter(staged, output_schema, compression=cfg.parquet_codec)
        try:

            def shard_rows(
                unit: PartitionedUnit,
            ) -> Iterator[tuple[tuple[int, int, int], list[int]]]:
                shard = units_root / "units" / unit.relative_output
                for batch in pq.ParquetFile(shard).iter_batches(
                    batch_size=max(1, cfg.row_group_size)
                ):
                    values = batch.to_pydict()
                    for index in range(batch.num_rows):
                        yield (
                            (
                                int(values["root_seed"][index]),
                                int(values["strategy_a"][index]),
                                int(values["strategy_b"][index]),
                            ),
                            [
                                int(values[name][index])
                                for name in (
                                    "paired_mirrored_games",
                                    "p1_win_difference_sum",
                                    "games_completed",
                                    "games_safety_limit",
                                    "unpaired_forward_games",
                                    "unpaired_reverse_games",
                                )
                            ],
                        )

            streams = [shard_rows(unit) for unit in units]
            heap: list[tuple[tuple[int, int, int], int, list[int]]] = []
            for index, stream in enumerate(streams):
                try:
                    key, value = next(stream)
                except StopIteration:
                    continue
                heapq.heappush(heap, (key, index, value))
            rows: list[dict[str, Any]] = []
            while heap:
                key, index, value = heapq.heappop(heap)
                totals = value
                try:
                    next_key, next_value = next(streams[index])
                    heapq.heappush(heap, (next_key, index, next_value))
                except StopIteration:
                    pass
                while heap and heap[0][0] == key:
                    _same_key, same_index, same_value = heapq.heappop(heap)
                    totals = [left + right for left, right in zip(totals, same_value, strict=True)]
                    try:
                        next_key, next_value = next(streams[same_index])
                        heapq.heappush(heap, (next_key, same_index, next_value))
                    except StopIteration:
                        pass
                matched, difference, completed, safety, forward, reverse = totals
                root, a, b = key
                rows.append(
                    {
                        "root_seed": root,
                        "k": 2,
                        "strategy_a": a,
                        "strategy_b": b,
                        "paired_mirrored_games": matched,
                        "games_attempted": completed + safety,
                        "games_completed": completed,
                        "games_safety_limit": safety,
                        "unpaired_forward_games": forward,
                        "unpaired_reverse_games": reverse,
                        "mean_p1_win_difference": difference / matched if matched else None,
                    }
                )
                if len(rows) >= max(1, cfg.row_group_size):
                    writer.write_table(pa.Table.from_pylist(rows, schema=output_schema))
                    rows.clear()
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=output_schema))
        finally:
            writer.close()
        guard.check_before_schedule(force=True)

    write_artifact_with_sidecar_atomic(output, sidecar, write)
    guard.check_before_schedule(force=True)


def _write_frame(
    cfg: AppConfig,
    frame: pd.DataFrame,
    path: Path,
    *,
    scope: ArtifactScope,
    operation: str,
    sources: list[Path],
    ks: list[int],
    grouping_keys: list[str],
    k_method: str = "none",
    k_weights: dict[int, float] | None = None,
    missing_cell_policy: str = "fail",
    conditioning: str = ATTEMPT_CONDITIONING,
    replication_unit: str = "attempted_player_game_exposure_grouped_by_deterministic_shuffle_batch",
) -> None:
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="seat_analysis",
        scope=scope,
        source_scope=ArtifactScope.BY_K,
        operation=operation,
        baseline="chance_1_over_k",
        weighted_quantity="seat_win_indicator",
        k_aggregation_method=k_method,
        k_weights=k_weights,
        support_count_role="raw_player_game_exposures",
        uncertainty_method="descriptive",
        replication_unit=replication_unit,
        conditioning=conditioning,
        consistency_columns=frame.columns.tolist(),
        source_artifacts=sources,
        grouping_keys=grouping_keys,
        player_counts=ks,
        required_player_counts=ks,
        missing_cell_policy=missing_cell_policy,
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        sidecar=sidecar,
        codec=cfg.parquet_codec,
    )


def build_canonical_seat_analysis(cfg: AppConfig, *, force: bool = False) -> SeatAnalysisArtifacts:
    """Build within-k seat effects and clearly labelled secondary diagnostics."""

    ks = sorted({int(k) for k in cfg.sim.n_players_list})
    sources = {k: cfg.combined_rows_by_k(k) for k in ks}
    missing = [path for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"canonical per-k seat inputs are missing: {missing}")
    roots = {_validate_source(path, k) for k, path in sources.items()}
    if roots != {int(cfg.sim.seed)}:
        raise ValueError(
            "seat analysis requires identical configured-root support; "
            f"expected [{cfg.sim.seed}], found {sorted(roots)}"
        )
    artifacts = SeatAnalysisArtifacts(
        batch_counts=tuple(cfg.seat_batch_counts_path(k) for k in ks),
        by_k=tuple(cfg.seat_effects_by_k_path(k) for k in ks),
        population_by_k=tuple(cfg.seat_population_by_k_path(k) for k in ks),
        standardized_across_k=cfg.seat_standardized_across_k_path(),
        exposure_mixture_diagnostic=cfg.seat_exposure_mixture_diagnostic_path(),
        selfplay_diagnostic=cfg.seat_selfplay_diagnostic_path(),
        mirrored_diagnostic=cfg.seat_mirrored_diagnostic_path(),
    )
    done = stage_done_path(cfg.metrics_stage_dir, "canonical_seat_analysis")
    mirror_bytes = int(
        cfg.resources.stage_batch_bytes.get(
            "partitioned_stage", cfg.resources.stage_batch_bytes["analysis"]
        )
    )
    mirror_records = max(1_024, mirror_bytes // 128)
    mirror_units: tuple[PartitionedUnit, ...] = ()
    mirror_root: Path | None = None
    mirror_cache_valid = True
    if 2 in sources:
        mirror_units = _mirrored_units(
            _mirrored_batches(sources[2], max_batch_bytes=mirror_bytes), max_records=mirror_records
        )
        mirror_root = cfg.metrics_stage_dir / "checkpoints" / "seat_mirrored_pairing_v2"
        mirror_cache_valid = (
            validate_final_manifest(
                mirror_root / "partition_manifest.jsonl",
                root=mirror_root,
                identity=_mirrored_identity(cfg, sources[2]),
                unit_source=lambda: iter(mirror_units),
            )
            is not None
        )
    if (
        not force
        and stage_is_up_to_date(
            done,
            inputs=list(sources.values()),
            outputs=list(artifacts.all_paths),
            cfg=cfg,
            stage="metrics",
            sidecar_artifacts=list(artifacts.all_paths),
        )
        and mirror_cache_valid
    ):
        return artifacts

    by_k: dict[int, pd.DataFrame] = {}
    population_by_k: dict[int, pd.DataFrame] = {}
    for k, count_path, effect_path, population_path in zip(
        ks, artifacts.batch_counts, artifacts.by_k, artifacts.population_by_k, strict=True
    ):
        _write_batch_counts(cfg, k, sources[k], count_path)
        counts = pq.read_table(count_path).to_pandas()
        effects, population = _within_k_frames(counts, k)
        by_k[k] = effects
        population_by_k[k] = population
        _write_frame(
            cfg,
            effects,
            effect_path,
            scope=ArtifactScope.BY_K,
            operation="calculate_strategy_seat_effects",
            sources=[count_path],
            ks=[k],
            grouping_keys=["root_seed", "k", "strategy", "seat"],
            conditioning=(
                "all attempted player-game exposures conditional on root_seed, k, strategy, and seat"
            ),
        )
        _write_frame(
            cfg,
            population,
            population_path,
            scope=ArtifactScope.BY_K,
            operation="calculate_population_seat_effects",
            sources=[count_path],
            ks=[k],
            grouping_keys=["root_seed", "k", "seat"],
            conditioning="all attempted player-game exposures conditional on root_seed, k, and seat",
        )

    standardized, mixture = _standardized_frames(cfg, by_k, population_by_k, ks)
    weights, operation, k_method = _declared_weights(cfg, ks)
    sidecar_weights = weights if k_method == "declared_mapping" else None
    _write_frame(
        cfg,
        standardized,
        artifacts.standardized_across_k,
        scope=ArtifactScope.ACROSS_K,
        operation=operation,
        sources=list(artifacts.by_k),
        ks=ks,
        grouping_keys=["root_seed", "effect_scope", "strategy", "seat"],
        k_method=k_method,
        k_weights=sidecar_weights,
        missing_cell_policy="declared_common_support",
        conditioning=(
            "complete declared k support conditional on root_seed, effect_scope, strategy, and seat"
        ),
        replication_unit="declared_k_cell",
    )
    _write_frame(
        cfg,
        mixture,
        artifacts.exposure_mixture_diagnostic,
        scope=ArtifactScope.DIAGNOSTICS,
        operation="within_k_exposure_combination",
        sources=list(artifacts.by_k),
        ks=ks,
        grouping_keys=["root_seed", "effect_scope", "strategy", "seat"],
        missing_cell_policy="declared_common_support",
        conditioning=(
            "all attempted player-game exposures over declared common k support conditional on "
            "root_seed, effect_scope, strategy, and seat"
        ),
    )
    selfplay = _game_diagnostics(sources)
    _write_frame(
        cfg,
        selfplay,
        artifacts.selfplay_diagnostic,
        scope=ArtifactScope.DIAGNOSTICS,
        operation="calculate_self_play_diagnostics",
        sources=list(sources.values()),
        ks=ks,
        grouping_keys=["root_seed", "k", "strategy"],
        conditioning="all attempted games conditional on every seat using the same strategy",
        replication_unit="attempted_self_play_game",
    )
    if 2 in sources:
        assert mirror_root is not None
        mirror_guard = ProcessTreeMemoryGuard(
            cfg.resources.rss_abort_mb, cfg.resources.rss_sample_interval_seconds
        )
        mirror_stage = run_partitioned_stage(
            root=mirror_root,
            identity=_mirrored_identity(cfg, sources[2]),
            unit_source=lambda: iter(mirror_units),
            writer=_MirroredPartitionWriter(
                str(sources[2]), tuple(_source_columns(2)), mirror_bytes, mirror_records
            ),
            resources=cfg.resources,
            requested_workers=cfg.analysis.n_jobs,
            mp_start_method=cfg.analysis.mp_start_method,
            force=force,
            memory_guard=mirror_guard,
        )
        if mirror_stage.required_units != len(mirror_units):
            raise RuntimeError(
                "mirrored-game partition manifest does not cover every required unit"
            )
        _write_mirrored_diagnostic(
            cfg,
            source=sources[2],
            units_root=mirror_root,
            units=mirror_units,
            manifest=mirror_stage.manifest_path,
            output=artifacts.mirrored_diagnostic,
            guard=mirror_guard,
        )
    else:
        _write_frame(
            cfg,
            pd.DataFrame(columns=_MIRRORED_COLUMNS),
            artifacts.mirrored_diagnostic,
            scope=ArtifactScope.DIAGNOSTICS,
            operation="calculate_mirrored_game_diagnostics",
            sources=list(sources.values()),
            ks=ks,
            grouping_keys=["root_seed", "strategy_a", "strategy_b"],
            conditioning='termination_status == "completed"',
            replication_unit="within_batch_count_matched_opposite_orientation_game_pair",
        )
    write_stage_done(
        done,
        inputs=list(sources.values()),
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=list(artifacts.all_paths),
    )
    return artifacts


__all__ = ["SeatAnalysisArtifacts", "build_canonical_seat_analysis"]
