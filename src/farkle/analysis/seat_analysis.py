"""Canonical within-k seat effects and explicitly secondary cross-k diagnostics."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import defaultdict
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
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
from farkle.utils.parallel import (
    ProcessTreeMemoryGuard,
    make_process_tree_memory_guard,
    process_map,
    resolve_mp_context,
    resolve_stage_parallel_policy,
)
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
from farkle.utils.writer import atomic_path

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
_COUNT_FIELDS: Final = _COUNT_SCHEMA.names[5:]


def _seat_bytes(cfg: AppConfig) -> int:
    return int(
        cfg.resources.stage_batch_bytes.get(
            "seat_analysis", cfg.resources.stage_batch_bytes["analysis"]
        )
    )


def _tables(source: Path, columns: list[str], max_bytes: int) -> Iterator[pa.Table]:
    # Bound decoded Arrow and any legacy Python conversion independently of row groups.
    for _, _, table in iter_parquet_tables_by_bytes(
        source,
        columns=columns,
        max_batch_bytes=max_bytes,
        max_batch_rows=max(1, max_bytes // (128 * len(columns))),
    ):
        yield table


def _integers(table: pa.Table, name: str) -> np.ndarray:
    column = table[name]
    if column.null_count or not pa.types.is_integer(column.type):
        raise ValueError(f"seat analysis requires non-null integer {name}")
    return np.asarray(column.to_numpy(), dtype=np.int64)


class _SeatTotals:
    """Compact observed strategy/seat counts, explicitly admitted by byte budget.

    State grows with the strategy roster, never the number of games or batches.
    The conservative allowance includes mapping objects, array growth/copies,
    and typed output buffers. Oversized rosters fail before allocation rather
    than silently exceeding the declared working allocation.
    """

    def __init__(self, k: int, max_bytes: int) -> None:
        self.k = k
        self.max_bytes = max_bytes
        self.limit = max_bytes // (512 + 4 * k * 4 * 8)
        if self.limit < 1:
            raise MemoryError("seat_analysis byte budget cannot hold one strategy")
        self.positions: dict[int, int] = {}
        self.data = np.zeros((min(128, self.limit), k, 4), dtype=np.int64)
        self.column_totals = [0, 0, 0, 0]

    def add(self, strategies: np.ndarray, seats: np.ndarray, values: np.ndarray) -> None:
        if not len(strategies):
            return
        if np.any(strategies < 0) or np.any(strategies > np.iinfo(np.int32).max):
            raise ValueError("invalid canonical strategy identifier")
        if np.any(seats < 0) or np.any(seats >= self.k) or np.any(values < 0):
            raise ValueError("invalid seat or negative seat counts")
        if np.any(values[:, 0] > values[:, 2]) or np.any(
            values[:, 1] - values[:, 2] != values[:, 3]
        ):
            raise ValueError("seat counts violate attempted/completed/win conservation")
        unique, inverse = np.unique(strategies, return_inverse=True)
        new = [int(strategy) for strategy in unique if int(strategy) not in self.positions]
        required = len(self.positions) + len(new)
        if required > self.limit:
            raise MemoryError(
                f"seat_analysis roster needs {required} strategies; byte budget "
                f"{self.max_bytes} admits {self.limit}; increase resources.stage_batch_bytes"
            )
        if required > len(self.data):
            grown = np.zeros(
                (min(self.limit, max(required, len(self.data) * 2)), self.k, 4), dtype=np.int64
            )
            grown[: len(self.positions)] = self.data[: len(self.positions)]
            self.data = grown
        for strategy in new:
            self.positions[strategy] = len(self.positions)
        positions = np.array([self.positions[int(s)] for s in unique], dtype=np.int64)[inverse]
        maximum = np.iinfo(np.int64).max
        for column in range(4):
            v = values[:, column]
            added = int(v.sum()) if int(v.max()) <= maximum // len(v) else sum(map(int, v))
            if self.column_totals[column] + added > maximum:
                raise OverflowError("seat count totals exceed int64")
            self.column_totals[column] += added
        np.add.at(self.data, (positions, seats), values)

    def tables(self, coordinate: tuple[int, int, int]) -> Iterator[pa.Table]:
        strategies = np.array(sorted(self.positions), dtype=np.int32)
        chunk = max(1, self.max_bytes // (256 * self.k))
        for start in range(0, len(strategies), chunk):
            selected = strategies[start : start + chunk]
            positions = [self.positions[int(s)] for s in selected]
            values = self.data[positions].reshape(-1, 4)
            present = values[:, 1] > 0
            values = values[present]
            if not len(values):
                continue
            columns = [
                pa.array(np.full(len(values), value), type=field.type)
                for value, field in zip(coordinate, list(_COUNT_SCHEMA)[:3], strict=True)
            ]
            columns.extend(
                [
                    pa.array(np.repeat(selected, self.k)[present], type=STRATEGY_ID_ARROW_TYPE),
                    pa.array(
                        np.tile(np.arange(1, self.k + 1), len(selected))[present], type=pa.int16()
                    ),
                    *(pa.array(values[:, i], type=pa.int64()) for i in range(4)),
                ]
            )
            yield pa.Table.from_arrays(columns, schema=_COUNT_SCHEMA)


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

    @property
    def batch_manifests(self) -> tuple[Path, ...]:
        """Return canonical streaming manifests in deterministic k order."""

        return tuple(path.with_suffix(".manifest.jsonl") for path in self.batch_counts)

    @property
    def publication_paths(self) -> tuple[Path, ...]:
        """Return every canonical artifact owned by seat-analysis completion."""

        return (*self.all_paths, *self.batch_manifests)


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


def _iter_seat_count_tables(
    source: Path, k: int, *, max_bytes: int = 32 * 1024 * 1024
) -> Iterator[pa.Table]:
    columns = _source_columns(k)
    coordinate: tuple[int, int, int] | None = None
    counts = _SeatTotals(k, max_bytes)
    for table in _tables(source, columns, max_bytes):
        roots = _integers(table, "root_seed")
        sizes = _integers(table, "k")
        batches = _integers(table, "deterministic_batch_id")
        _integers(table, "shuffle_index")
        _integers(table, "game_index")
        if np.any(sizes != k) or np.any(batches < 0):
            raise ValueError(f"{source} has invalid k/batch coordinates")
        if table["termination_status"].null_count:
            raise ValueError(f"{source} has missing termination status")
        completed = np.asarray(pc.equal(table["termination_status"], pa.scalar("completed")))
        safety = np.asarray(pc.equal(table["termination_status"], pa.scalar("safety_limit")))
        valid_winner = np.asarray(
            pc.is_in(table["winner_seat"], value_set=pa.array([f"P{i}" for i in range(1, k + 1)]))
        )
        if np.any(~(completed | safety)) or np.any(completed & ~valid_winner):
            raise ValueError(f"{source} has invalid termination status or completed winner")
        if np.any(safety & np.asarray(pc.is_valid(table["winner_seat"]))):
            raise ValueError(f"{source} credits a safety-limit winner")
        changes = (
            np.flatnonzero(
                (roots[1:] != roots[:-1])
                | (sizes[1:] != sizes[:-1])
                | (batches[1:] != batches[:-1])
            )
            + 1
        )
        boundaries = np.concatenate(([0], changes, [table.num_rows]))
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            current = (int(roots[start]), k, int(batches[start]))
            if coordinate is not None and current < coordinate:
                raise ValueError(f"{source} is not ordered by root, k, and deterministic batch")
            if coordinate is not None and current != coordinate:
                yield from counts.tables(coordinate)
                counts = _SeatTotals(k, max_bytes)
            coordinate = current
            segment = table.slice(int(start), int(stop - start))
            for seat in range(1, k + 1):
                strategies = _integers(segment, f"P{seat}_strategy")
                wins = np.asarray(
                    pc.fill_null(
                        pc.equal(segment["winner_seat"], pa.scalar(f"P{seat}")), pa.scalar(False)
                    ),
                    dtype=np.int64,
                )
                values = np.column_stack(
                    (
                        wins,
                        np.ones(len(wins), dtype=np.int64),
                        completed[start:stop],
                        safety[start:stop],
                    )
                )
                counts.add(strategies, np.full(len(wins), seat - 1), values)
    if coordinate is not None:
        yield from counts.tables(coordinate)


def _write_batch_counts(
    cfg: AppConfig, k: int, source: Path, output: Path, *, force: bool = False
) -> None:
    manifest = output.with_suffix(".manifest.jsonl")
    done = stage_done_path(output.parent, "seat_batch_counts")
    if not force and stage_is_up_to_date(
        done,
        inputs=[source],
        outputs=[output, manifest],
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=[output, manifest],
    ):
        return
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
    manifest_sidecar = make_artifact_sidecar(
        cfg,
        manifest,
        producer="seat_analysis",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation="publish_seat_batch_counts_manifest",
        weighted_quantity="authenticated_streaming_output_inventory",
        support_count_role="single_output_identity",
        uncertainty_method="none",
        replication_unit="manifest_entry",
        conditioning="unconditional",
        source_artifacts=[output],
        player_counts=[k],
        required_player_counts=[k],
        missing_cell_policy="fail",
    )
    run_streaming_shard(
        out_path=str(output),
        manifest_path=str(manifest),
        schema=_COUNT_SCHEMA,
        batch_iter=_iter_seat_count_tables(source, k, max_bytes=_seat_bytes(cfg)),
        row_group_size=cfg.row_group_size,
        compression=cfg.parquet_codec,
        manifest_extra={"root_seed": cfg.sim.seed, "k": k},
        sidecar=sidecar,
        manifest_sidecar=manifest_sidecar,
    )
    write_stage_done(
        done,
        inputs=[source],
        outputs=[output, manifest],
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=[output, manifest],
    )


def _validate_source(path: Path, k: int, *, max_bytes: int = 32 * 1024 * 1024) -> int:
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
    for table in _tables(path, ["root_seed", "k"], max_bytes):
        roots_set.update(map(int, np.unique(_integers(table, "root_seed"))))
        observed_k_set.update(map(int, np.unique(_integers(table, "k"))))
        if len(roots_set) > 1 or observed_k_set != {k}:
            raise ValueError(f"{path} has invalid root/k support")
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


def _stream_within_k_frames(
    source: Path, k: int, max_bytes: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    totals = _SeatTotals(k, max_bytes)
    root: int | None = None
    for table in _tables(source, _COUNT_SCHEMA.names, max_bytes):
        roots = _integers(table, "root_seed")
        sizes = _integers(table, "k")
        if len(np.unique(roots)) != 1 or np.any(sizes != k):
            raise ValueError("seat counts have incompatible root/k support")
        if root is not None and root != int(roots[0]):
            raise ValueError("seat counts mix roots")
        root = int(roots[0])
        values = np.column_stack([_integers(table, name) for name in _COUNT_FIELDS])
        totals.add(_integers(table, "strategy"), _integers(table, "seat") - 1, values)
    if root is None:
        raise ValueError("seat counts have no observed support")
    # Only the compact roster x seat totals enter pandas, never batch histories.
    compact = pa.concat_tables(list(totals.tables((root, k, 0)))).to_pandas()
    return _within_k_frames(compact, k)


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
    standardized: list[pd.DataFrame] = []
    mixture: list[pd.DataFrame] = []
    for scope, frames, keys in (
        ("strategy", by_k, ["strategy", "seat"]),
        ("population", population_by_k, ["seat"]),
    ):
        indexed = {
            k: frame.loc[frame["seat"] <= min(ks)].set_index(keys, verify_integrity=True)
            for k, frame in frames.items()
        }
        support = indexed[ks[0]].index
        for k in ks[1:]:
            support = support.intersection(indexed[k].index)
        support = support.sort_values()
        cells = {k: indexed[k].loc[support] for k in ks}
        base = support.to_frame(index=False)
        base.insert(0, "effect_scope", scope)
        base.insert(0, "root_seed", cells[ks[0]]["root_seed"].to_numpy())
        if scope == "population":
            base["strategy"] = None
        base["common_k_support"] = [ks] * len(base)
        effect = np.zeros(len(base), dtype=np.float64)
        baseline = np.zeros(len(base), dtype=np.float64)
        totals = np.zeros((len(base), 4), dtype=np.int64)
        for k, cell in cells.items():
            effect += cell["seat_effect"].to_numpy() * weights[k]
            counts = cell[_COUNT_FIELDS].to_numpy(dtype=np.int64)
            if np.any(counts > np.iinfo(np.int64).max - totals):
                raise OverflowError("cross-k seat totals exceed int64")
            totals += counts
            baseline += counts[:, 1] / k
        standardized_cell = base.copy()
        standardized_cell["standardized_seat_effect"] = effect
        standardized.append(standardized_cell)
        for i, column in enumerate(_COUNT_FIELDS):
            base[column] = totals[:, i]
        base["exposure_weighted_baseline"] = baseline / totals[:, 1]
        base["exposure_weighted_seat_effect"] = (totals[:, 0] - baseline) / totals[:, 1]
        mixture.append(base)
    standardized_frame = pd.concat(standardized, ignore_index=True)[_STANDARDIZED_COLUMNS]
    mixture_frame = pd.concat(mixture, ignore_index=True)[_MIXTURE_COLUMNS]
    for frame in (standardized_frame, mixture_frame):
        frame["strategy"] = pd.array(frame["strategy"].tolist(), dtype="Int32")
    return standardized_frame, mixture_frame


def _game_diagnostics(
    sources: dict[int, Path], *, max_bytes: int = 32 * 1024 * 1024
) -> pd.DataFrame:
    selfplay: defaultdict[tuple[int, int, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    for k, source in sources.items():
        columns = _source_columns(k)
        for batch in _tables(source, columns, max_bytes):
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
                    if (root, k, strategies[0]) not in selfplay and len(
                        selfplay
                    ) >= max_bytes // 512:
                        raise MemoryError("self-play roster exceeds seat_analysis byte budget")
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
        max_batch_rows=max(1, max_batch_bytes // (128 * len(columns))),
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
            max_batch_rows=max(1, self.max_batch_bytes // (128 * len(self.columns))),
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
                if len(records) >= self.max_records:
                    raise MemoryError("mirrored partition exceeds its admitted record budget")
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


_MIRROR_KEYS: Final = ["root_seed", "strategy_a", "strategy_b"]
_MIRROR_VALUES: Final = _MIRROR_SHARD_SCHEMA.names[4:]
_MERGE_SCHEMA: Final = pa.schema(
    [_MIRROR_SHARD_SCHEMA.field(name) for name in [*_MIRROR_KEYS, *_MIRROR_VALUES]]
)
_MIRROR_FANIN: Final = 16


def _merge_mirror_rows(
    paths: list[Path], max_bytes: int
) -> Iterator[tuple[tuple[int, int, int], list[int]]]:
    """Merge at most 16 sorted readers with a shared decode budget."""
    if len(paths) > _MIRROR_FANIN:
        raise ValueError("mirrored merge exceeds bounded fan-in")
    reader_bytes = max(1, max_bytes // max(1, len(paths)))

    def rows(path: Path) -> Generator[tuple[tuple[int, int, int], list[int]], None, None]:
        for table in _tables(path, _MERGE_SCHEMA.names, reader_bytes):
            arrays = [_integers(table, name) for name in _MERGE_SCHEMA.names]
            for i in range(table.num_rows):
                yield (
                    (int(arrays[0][i]), int(arrays[1][i]), int(arrays[2][i])),
                    [int(array[i]) for array in arrays[3:]],
                )

    streams = [rows(path) for path in paths]
    previous: tuple[int, int, int] | None = None
    total: list[int] = []
    try:
        for key, values in heapq.merge(*streams, key=lambda row: row[0]):
            if previous is not None and key != previous:
                yield previous, total
                total = []
            previous = key
            total = [a + b for a, b in zip(total, values, strict=True)] if total else values
        if previous is not None:
            yield previous, total
    finally:
        for stream in streams:
            stream.close()


def _reduce_mirror_fanin(paths: list[Path], scratch: Path, max_bytes: int) -> list[Path]:
    """Spill merge levels; durable authenticated source units remain resumable."""
    level = 0
    row_limit = max(1, max_bytes // 2048)
    while len(paths) > _MIRROR_FANIN:
        reduced: list[Path] = []
        for start in range(0, len(paths), _MIRROR_FANIN):
            path = scratch / f"level-{level}-part-{start}.parquet"
            with (
                atomic_path(str(path)) as staged,
                pq.ParquetWriter(staged, _MERGE_SCHEMA) as writer,
            ):
                records: list[dict[str, int]] = []
                for key, values in _merge_mirror_rows(
                    paths[start : start + _MIRROR_FANIN], max_bytes
                ):
                    records.append(dict(zip(_MERGE_SCHEMA.names, [*key, *values], strict=True)))
                    if len(records) >= row_limit:
                        writer.write_table(pa.Table.from_pylist(records, schema=_MERGE_SCHEMA))
                        records.clear()
                if records:
                    writer.write_table(pa.Table.from_pylist(records, schema=_MERGE_SCHEMA))
            reduced.append(path)
        paths = reduced
        level += 1
    return paths


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

            paths = [units_root / "units" / unit.relative_output for unit in units]
            rows: list[dict[str, Any]] = []
            max_bytes = _seat_bytes(cfg)
            row_limit = max(1, min(cfg.row_group_size, max_bytes // 2048))
            scratch_parent = cfg.io.temp_dir or units_root
            Path(scratch_parent).mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix="seat-merge-", dir=scratch_parent) as scratch:
                paths = _reduce_mirror_fanin(paths, Path(scratch), max_bytes)
                merged = _merge_mirror_rows(paths, max_bytes)
                for key, totals in merged:
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
                    if len(rows) >= row_limit:
                        writer.write_table(pa.Table.from_pylist(rows, schema=output_schema))
                        rows.clear()
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=output_schema))
        finally:
            writer.close()
        guard.check_before_schedule(force=True)

    write_artifact_with_sidecar_atomic(output, sidecar, write)
    guard.check_before_schedule(force=True)


def _frame_schema(columns: list[str]) -> pa.Schema:
    """One explicit schema for empty and nonempty canonical seat products."""
    integers = {
        "root_seed",
        "raw_wins",
        "raw_exposures",
        "raw_completed_exposures",
        "raw_safety_limit_exposures",
        "raw_losses",
        "p1_wins",
        "games_attempted",
        "games_completed",
        "games_safety_limit",
        "paired_mirrored_games",
        "unpaired_forward_games",
        "unpaired_reverse_games",
    }
    floats = {
        "chance_baseline",
        "win_rate",
        "win_rate_per_attempt",
        "win_rate_given_completion",
        "safety_limit_exposure_rate",
        "seat_effect",
        "standardized_seat_effect",
        "exposure_weighted_baseline",
        "exposure_weighted_seat_effect",
        "p1_win_rate_per_attempt",
        "p1_win_rate_given_completion",
        "p1_effect_vs_chance",
        "mean_p1_win_difference",
    }
    fields = []
    for name in columns:
        dtype: pa.DataType
        if name in integers:
            dtype = pa.int64()
        elif name in floats:
            dtype = pa.float64()
        elif name in {"strategy", "strategy_a", "strategy_b"}:
            dtype = STRATEGY_ID_ARROW_TYPE
        elif name in {"k", "seat"}:
            dtype = pa.int16()
        elif name == "common_k_support":
            dtype = pa.list_(pa.int16())
        elif name == "effect_scope":
            dtype = pa.string()
        else:
            raise ValueError(f"unregistered seat artifact field: {name}")
        fields.append(pa.field(name, dtype))
    return pa.schema(fields)


def _seat_cell_outputs(cfg: AppConfig, k: int) -> list[Path]:
    counts = cfg.seat_batch_counts_path(k)
    return [
        counts,
        counts.with_suffix(".manifest.jsonl"),
        cfg.seat_effects_by_k_path(k),
        cfg.seat_population_by_k_path(k),
    ]


def _seat_cell_valid(cfg: AppConfig, k: int) -> bool:
    outputs = _seat_cell_outputs(cfg, k)
    return stage_is_up_to_date(
        stage_done_path(outputs[0].parent, "seat_analysis_cell"),
        inputs=[cfg.combined_rows_by_k(k)],
        outputs=outputs,
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=outputs,
    )


def _build_seat_cell(task: tuple[AppConfig, int, bool]) -> int:
    cfg, k, force = task
    if not force and _seat_cell_valid(cfg, k):
        return k
    source = cfg.combined_rows_by_k(k)
    root = _validate_source(source, k, max_bytes=_seat_bytes(cfg))
    if root != cfg.sim.seed:
        raise ValueError(
            f"seat analysis requires configured-root support {cfg.sim.seed}; found {root}"
        )
    counts, _manifest, effects_path, population_path = _seat_cell_outputs(cfg, k)
    _write_batch_counts(cfg, k, source, counts, force=force)
    effects, population = _stream_within_k_frames(counts, k, _seat_bytes(cfg))
    for frame, path, operation, groups, conditioning in (
        (
            effects,
            effects_path,
            "calculate_strategy_seat_effects",
            ["root_seed", "k", "strategy", "seat"],
            "all attempted player-game exposures conditional on root_seed, k, strategy, and seat",
        ),
        (
            population,
            population_path,
            "calculate_population_seat_effects",
            ["root_seed", "k", "seat"],
            "all attempted player-game exposures conditional on root_seed, k, and seat",
        ),
    ):
        _write_frame(
            cfg,
            frame,
            path,
            scope=ArtifactScope.BY_K,
            operation=operation,
            sources=[counts],
            ks=[k],
            grouping_keys=groups,
            conditioning=conditioning,
        )
    outputs = _seat_cell_outputs(cfg, k)
    write_stage_done(
        stage_done_path(counts.parent, "seat_analysis_cell"),
        inputs=[source],
        outputs=outputs,
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=outputs,
    )
    return k


def _compact_frame(path: Path, max_bytes: int) -> pd.DataFrame:
    """Admit a small roster summary; never use this reader for batch histories."""
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows * 256 > max_bytes:
        raise MemoryError(f"seat summary {path} exceeds configured compact-frame budget")
    tables = list(_tables(path, parquet.schema_arrow.names, max_bytes))
    table = (
        pa.concat_tables(tables)
        if tables
        else pa.Table.from_batches([], schema=parquet.schema_arrow)
    )
    return table.to_pandas()


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
        pa.Table.from_pandas(
            frame, schema=_frame_schema(frame.columns.tolist()), preserve_index=False
        ),
        path,
        sidecar=sidecar,
        codec=cfg.parquet_codec,
    )


def build_canonical_seat_analysis(cfg: AppConfig, *, force: bool = False) -> SeatAnalysisArtifacts:
    """Build within-k seat effects and clearly labeled secondary diagnostics."""

    ks = sorted({int(k) for k in cfg.sim.n_players_list})
    if not ks:
        raise ValueError("seat analysis requires at least one configured k")
    sources = {k: cfg.combined_rows_by_k(k) for k in ks}
    missing = [path for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"canonical per-k seat inputs are missing: {missing}")
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
            outputs=list(artifacts.publication_paths),
            cfg=cfg,
            stage="metrics",
            sidecar_artifacts=list(artifacts.publication_paths),
        )
        and mirror_cache_valid
    ):
        return artifacts

    pending = [(cfg, k, force) for k in ks if force or not _seat_cell_valid(cfg, k)]
    if pending:
        policy = resolve_stage_parallel_policy("analysis", cfg.analysis, resources=cfg.resources)
        workers = min(policy.process_workers, len(pending))
        for _k in process_map(
            _build_seat_cell,
            pending,
            n_jobs=workers,
            window=workers * cfg.resources.max_in_flight_per_worker,
            mp_context=resolve_mp_context(cfg.analysis.mp_start_method),
            memory_guard=make_process_tree_memory_guard(cfg.resources),
        ):
            pass
    by_k: dict[int, pd.DataFrame] = {}
    population_by_k: dict[int, pd.DataFrame] = {}
    for k in ks:
        effects = _compact_frame(cfg.seat_effects_by_k_path(k), _seat_bytes(cfg))
        # Cross-k diagnostics use only seats common to every k.
        by_k[k] = effects.loc[effects["seat"] <= min(ks)].copy()
        population_by_k[k] = _compact_frame(cfg.seat_population_by_k_path(k), _seat_bytes(cfg))

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
    selfplay = _game_diagnostics(sources, max_bytes=_seat_bytes(cfg))
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
        mirror_guard = make_process_tree_memory_guard(cfg.resources)
        mirror_stage = run_partitioned_stage(
            root=mirror_root,
            identity=_mirrored_identity(cfg, sources[2]),
            unit_source=lambda: iter(mirror_units),
            writer=_MirroredPartitionWriter(
                str(sources[2]), tuple(_source_columns(2)), mirror_bytes, mirror_records
            ),
            resources=cfg.resources,
            temp_dir=cfg.io.temp_dir,
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
        outputs=list(artifacts.publication_paths),
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=list(artifacts.publication_paths),
    )
    return artifacts


__all__ = ["SeatAnalysisArtifacts", "build_canonical_seat_analysis"]
