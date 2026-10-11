"""Authenticated, resumable exact distributions and complete-support summaries."""

from __future__ import annotations

import hashlib
import heapq
import json
import shutil
import struct
from collections.abc import Callable, Generator, Iterator
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from itertools import groupby
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.outcome_distributions import (
    _COUNTS,
    DISTRIBUTIONS,
    OUTCOME_METHOD_VERSION,
    _slice_row_limit,
    accumulate_outcome_slice,
    outcome_frequency_schema,
    outcome_source_columns,
    outcome_support_schema,
)
from farkle.config import AppConfig, ArtifactScope
from farkle.simulation.runner import simulation_is_complete
from farkle.utils.arrow_batches import iter_parquet_tables_by_bytes
from farkle.utils.artifact_contract import (
    ArtifactSidecar,
    make_artifact_sidecar,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.artifacts import write_json_artifact_atomic
from farkle.utils.authenticated_contract import identity_sha256
from farkle.utils.partitioned_stage import (
    PartitionedStageIdentity,
    PartitionedUnit,
    resolved_code_identity_sha256,
    run_partitioned_stage,
    validate_final_manifest,
)
from farkle.utils.release_identity import CapturedV3Inputs, capture_v3_inputs
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.writer import ParquetShardWriter

# One fixed-width intermediate format avoids separate scans for each product.
# Metrics 0..6 are B sums, 7..12 are histogram bins, 13 is per-shuffle roster
# multiplicity, and 14 is game termination support (value 0 completed/1 safety).
_RUN_DTYPE = np.dtype(
    [
        ("batch", "<i4"),
        ("strategy", "<i4"),
        ("seat", "<i2"),
        ("metric", "i1"),
        ("value", "<i8"),
        ("count", "<i8"),
    ]
)
_RUN_SCHEMA = pa.schema(
    [
        pa.field(name, pa.from_numpy_dtype(_RUN_DTYPE[name]), nullable=False)
        for name in _RUN_DTYPE.names or ()
    ]
)
_MAX = np.iinfo(np.int64).max
_DENOMINATORS = ("completed", "completed", "losses_completed", "completed", "safety", "wins")
_CONDITIONING = (
    "completed",
    "completed",
    "completed_loss",
    "completed",
    "safety_limit",
    "completed_win",
)


def outcome_inventory_schema() -> pa.Schema:
    integer_types: dict[str, pa.DataType] = {
        "root_seed": pa.int64(),
        "k": pa.int16(),
        "deterministic_batch_id": pa.int32(),
    }
    names = [
        "root_seed",
        "k",
        "deterministic_batch_id",
        "shuffle_start",
        "shuffle_stop_exclusive",
        "games_per_shuffle",
        "expected_games",
        "observed_games",
        "completed_games",
        "safety_games",
        "expected_support_rows",
        "emitted_support_rows",
    ]
    return pa.schema(
        [
            *(pa.field(n, integer_types.get(n, pa.int64()), nullable=False) for n in names),
            *(
                pa.field(n, pa.string(), nullable=False)
                for n in (
                    "coordinate_digest",
                    "source_identity_sha256",
                    "roster_identity_sha256",
                    "support_status",
                )
            ),
        ]
    )


def outcome_summary_schema() -> pa.Schema:
    return pa.schema(
        [
            *list(outcome_frequency_schema())[:4],
            *(
                pa.field(n, pa.string(), nullable=False)
                for n in ("quantity", "statistic", "conditioning")
            ),
            pa.field("denominator", pa.int64(), nullable=False),
            pa.field("estimate", pa.float64()),
            pa.field("status", pa.string(), nullable=False),
        ]
    )


def outcome_publication_paths(cfg: AppConfig, k: int) -> tuple[Path, ...]:
    return (
        cfg.outcome_frequencies_path(k),
        cfg.outcome_batch_stats_path(k),
        cfg.outcome_batch_inventory_path(k),
        cfg.outcome_support_path(k),
        cfg.outcome_summaries_path(k),
        cfg.outcome_units_dir(k) / "plan.json",
        cfg.outcome_checkpoint_inventory_path(k),
    )


def _budget(cfg: AppConfig) -> int:
    return int(cfg.resources.stage_batch_bytes["analysis"])


def _admit_source_footers(cfg: AppConfig) -> None:
    """Bound serialized metadata before Arrow/authentication decodes it.

    Projection limits decoded rows, not the footer that describes all row
    groups. Reserve a quarter of the configured allowance for serialized source
    metadata; native parser overhead remains part of measured process RSS.
    """
    paths = [cfg.strategy_manifest_root_path()]
    for k in sorted(set(cfg.sim.n_players_list)):
        paths.extend((cfg.ingested_rows_raw(k), cfg.ingested_rows_curated(k)))
    for path in paths:
        with path.open("rb") as stream:
            stream.seek(-8, 2)
            size, magic = struct.unpack("<I4s", stream.read(8))
        if magic != b"PAR1":
            raise ValueError(f"invalid required Parquet source footer: {path}")
        if size > _budget(cfg) // 4:
            raise MemoryError(
                f"source Parquet footer exceeds metadata admission: {path}; increase analysis stage_batch_bytes"
            )


def _checked(value: int) -> int:
    if value < 0 or value > _MAX:
        raise OverflowError("outcome integer count/sum outside nonnegative int64")
    return value


@dataclass(frozen=True)
class _Design:
    root: int
    k: int
    batches: int
    width: int
    games_per_shuffle: int
    roster: tuple[tuple[int, int], ...]
    source: str
    inputs: tuple[str, ...]
    source_hash: str
    roster_hash: str

    @property
    def games_per_batch(self) -> int:
        return self.width * self.games_per_shuffle


def _resolve_design(cfg: AppConfig, k: int) -> _Design:
    source = cfg.ingested_rows_curated(k)
    workload = cfg.simulation_workload_plan_path(k)
    roster_path = cfg.strategy_manifest_root_path()
    if not simulation_is_complete(cfg, k):
        raise ValueError("outcomes require authenticated complete simulation support")
    ks = sorted(set(cfg.sim.n_players_list))
    if not stage_is_up_to_date(
        stage_done_path(cfg.curate_stage_dir, "curate"),
        inputs=[*(cfg.ingested_rows_raw(n) for n in ks), *(cfg.ingest_manifest(n) for n in ks)],
        outputs=[*(cfg.ingested_rows_curated(n) for n in ks), *(cfg.manifest_for(n) for n in ks)],
        cfg=cfg,
        stage="curate",
    ):
        raise ValueError("outcomes require authenticated complete curate support")
    for path in (source, workload, roster_path):
        validate_artifact_sidecar(path)
    validate_artifact_sidecar(source, expected={"operation": "curate_game_rows", "scope": "by_k"})
    plan = json.loads(workload.read_text(encoding="utf-8"))
    fields = (
        "root_seed",
        "k",
        "batch_count",
        "shuffles_per_batch",
        "games_per_shuffle",
        "required_shuffles",
        "required_games",
        "strategy_count",
    )
    if any(type(plan.get(name)) is not int for name in fields):
        raise ValueError("invalid integer workload plan")
    b, w, g = (plan[n] for n in ("batch_count", "shuffles_per_batch", "games_per_shuffle"))
    if (
        plan["root_seed"] != cfg.sim.seed
        or plan["k"] != k
        or min(b, w, g) < 1
        or plan["required_shuffles"] != b * w
        or plan["required_games"] != b * w * g
        or plan["strategy_count"] != g * k
        or plan.get("cap_exceeded") is not False
    ):
        raise ValueError("inconsistent complete outcome workload")
    parquet = pq.ParquetFile(roster_path)
    if parquet.metadata.num_rows * 512 > _budget(cfg):
        raise MemoryError("outcome roster exceeds admitted metadata budget")
    columns = ["strategy_id"]
    if "multiplicity" in parquet.schema_arrow.names:
        columns.append("multiplicity")
    roster: dict[int, int] = {}
    for chunk in parquet.iter_batches(columns=columns, batch_size=max(1, _budget(cfg) // 512)):
        for row in chunk.to_pylist():
            sid, count = row["strategy_id"], row.get("multiplicity", 1)
            if type(sid) is not int or not 0 <= sid <= np.iinfo(np.int32).max or sid in roster:
                raise ValueError("invalid/duplicate strategy identity in roster")
            if type(count) is not int or count < 1:
                raise ValueError("invalid authenticated roster multiplicity")
            roster[sid] = count
    if sum(roster.values()) != plan["strategy_count"]:
        raise ValueError(
            "roster multiplicities do not match plan; do not infer them from surviving games"
        )
    if pq.ParquetFile(source).metadata.num_rows != b * w * g:
        raise ValueError("missing or excess required game records")
    inputs = (
        source,
        workload,
        roster_path,
        cfg.n_dir(k) / "simulation.done.json",
        stage_done_path(cfg.curate_stage_dir, "curate"),
    )
    return _Design(
        cfg.sim.seed,
        k,
        b,
        w,
        g,
        tuple(sorted(roster.items())),
        str(source),
        tuple(map(str, inputs)),
        sha256_file(source),
        sha256_file(roster_path),
    )


def _metadata(
    cfg: AppConfig, k: int, path: Path, operation: str, sources: list[Path]
) -> ArtifactSidecar:
    schemas = {
        "outcome_frequencies.parquet": outcome_frequency_schema(),
        "outcome_batch_stats.parquet": outcome_support_schema(batch=True),
        "outcome_batch_inventory.parquet": outcome_inventory_schema(),
        "outcome_support.parquet": outcome_support_schema(),
        "outcome_summaries.parquet": outcome_summary_schema(),
    }
    columns: list[str] = []
    if path.suffix == ".parquet":
        columns = schemas.get(
            path.name,
            (
                outcome_support_schema(batch=True)
                if "certified_batches" in path.parts
                else _RUN_SCHEMA
            ),
        ).names
    elif path.name == "plan.json":
        columns = [
            "method_version",
            "design_hash",
            "code_hash",
            "rows_per_unit",
            "minimum_budget",
            "fan_in",
        ]
    elif path.name == "publication.json":
        columns = ["method_version", "stages"]
    return make_artifact_sidecar(
        cfg,
        path,
        producer="outcome_distributions",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation=operation,
        baseline="none",
        weighted_quantity="exact_empirical_outcome_counts",
        support_count_role="certified_player_game_exposures",
        uncertainty_method="none",
        replication_unit="deterministic_shuffle_batch",
        conditioning="explicit_per_distribution_completed_or_safety_limit",
        source_artifacts=sources,
        player_counts=[k],
        required_player_counts=[k],
        missing_cell_policy="fail",
        consistency_columns=columns,
        method_contract=cast(
            Any,
            {
                "kind": "operation",
                "procedure": operation,
                "parameters": {"outcome_distribution_method_version": OUTCOME_METHOD_VERSION},
            },
        ),
    )


@dataclass(frozen=True)
class _Sidecars:
    cfg: AppConfig
    k: int
    operation: str
    sources: tuple[str, ...]
    captured: CapturedV3Inputs | None = None
    unit_sources: bool = False

    def __call__(self, unit: PartitionedUnit, path: Path) -> ArtifactSidecar:
        sources = [Path(str(p)) for p in (unit.key[2:] if self.unit_sources else self.sources)]
        metadata = _metadata(self.cfg, self.k, path, self.operation, sources)
        return replace(metadata, _captured_v3_inputs=self.captured) if self.captured else metadata


def _run_table(records: np.ndarray) -> pa.Table:
    if not len(records):
        return pa.Table.from_batches([], schema=_RUN_SCHEMA)
    keys, inverse = np.unique(records[list(_RUN_SCHEMA.names[:-1])], return_inverse=True)
    counts = np.zeros(len(keys), dtype=np.int64)
    _checked(sum(map(int, records["count"])))
    np.add.at(counts, inverse, records["count"])
    return pa.Table.from_arrays(
        [
            *(pa.array(keys[n], type=_RUN_SCHEMA.field(n).type) for n in _RUN_SCHEMA.names[:-1]),
            pa.array(counts),
        ],
        schema=_RUN_SCHEMA,
    )


def _read_rows(
    path: Path, budget: int, schema: pa.Schema = _RUN_SCHEMA
) -> Generator[tuple[int, ...], None, None]:
    for _, _, table in iter_parquet_tables_by_bytes(
        path,
        columns=schema.names,
        max_batch_bytes=budget,
        max_batch_rows=max(1, budget // (128 * len(schema))),
    ):
        columns = [table[n].to_numpy() for n in schema.names]
        for i in range(table.num_rows):
            yield tuple(int(c[i]) for c in columns)


def _merged(paths: list[Path], budget: int) -> Iterator[tuple[int, ...]]:
    if len(paths) > 16:
        raise ValueError("outcome merge fan-in exceeds 16")
    streams = [_read_rows(p, max(1, budget // (4 * max(1, len(paths))))) for p in paths]
    previous: tuple[int, ...] | None = None
    count = 0
    try:
        for row in heapq.merge(*streams, key=lambda r: r[:-1]):
            key = row[:-1]
            if previous is not None and key != previous:
                yield (*previous, count)
                count = 0
            previous = key
            count = _checked(count + row[-1])
        if previous is not None:
            yield (*previous, count)
    finally:
        for stream in streams:
            stream.close()


def _write_rows(
    path: Path,
    schema: pa.Schema,
    rows: Iterator[dict[str, Any]],
    budget: int,
    metadata: dict[str, str] | None = None,
) -> None:
    with pq.ParquetWriter(path, schema) as writer:
        coalescer = _CoalescedWriter(writer.write_table, schema, max(1, budget // 4))
        buffer: list[dict[str, Any]] = []
        for row in rows:
            buffer.append(row)
            if len(buffer) >= max(1, budget // 8192):
                coalescer.write_batch(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
        if buffer:
            coalescer.write_batch(pa.Table.from_pylist(buffer, schema=schema))
        coalescer.flush()
        if metadata is not None:
            writer.add_key_value_metadata(metadata)


class _CoalescedWriter:
    """Bound Arrow buffers separately from Python rows and Parquet footer state."""

    def __init__(self, write: Callable[[pa.Table], Any], schema: pa.Schema, budget: int):
        self.write, self.schema, self.budget = write, schema, budget
        self.tables: list[pa.Table] = []
        self.bytes = 0
        self.footer_allowance_used = 0

    def write_batch(self, table: pa.Table) -> None:
        if self.tables and self.bytes + table.nbytes > self.budget:
            self.flush()
        if table.nbytes > self.budget:
            raise MemoryError("outcome output chunk exceeds its Arrow buffer allowance")
        self.tables.append(table)
        self.bytes += table.nbytes

    def flush(self) -> None:
        if self.tables:
            table = pa.concat_tables(self.tables)
            # All owned writers use groups of at least 200,000 rows or one
            # smaller coalesced table. Admit conservative per-column metadata
            # too; unusually broad exact support must not grow a footer forever.
            self.footer_allowance_used += (
                max(1, (table.num_rows + 199999) // 200000) * len(self.schema) * 1024
            )
            if self.footer_allowance_used > self.budget:
                raise MemoryError(
                    "outcome Parquet footer exceeds admitted metadata budget; increase analysis stage_batch_bytes"
                )
            self.write(table)
            self.tables.clear()
            self.bytes = 0


@dataclass(frozen=True)
class _SliceWriter:
    design: _Design
    budget: int

    def __call__(self, unit: PartitionedUnit, output: Path) -> None:
        batch, _index, start, stop, rg, rg_start = map(int, unit.key)
        d = self.design
        capacity = (stop - start) * (12 * d.k + 2)
        records = np.empty(capacity, dtype=_RUN_DTYPE)
        used, position = 0, rg_start

        def push(strategy: int, seat: int, metric: int, value: int, count: int) -> None:
            nonlocal used
            if used >= capacity:
                raise MemoryError("outcome source slice exceeded admitted record count")
            records[used] = (batch, strategy, seat, metric, value, count)
            used += 1

        for _, _, table in iter_parquet_tables_by_bytes(
            Path(d.source),
            columns=outcome_source_columns(d.k),
            max_batch_bytes=self.budget // 4,
            max_batch_rows=_slice_row_limit(d.k, self.budget),
            start_row_group=rg,
        ):
            end = position + table.num_rows
            left, right = max(start, position), min(stop, end)
            if right > left:
                selected = table.slice(left - position, right - left)
                partial = accumulate_outcome_slice(
                    selected,
                    root_seed=d.root,
                    k=d.k,
                    shuffles_per_batch=d.width,
                    max_bytes=self.budget,
                )
                for i, coord in enumerate(partial.coordinates.to_pylist(), left):
                    if (coord["shuffle_index"], coord["game_index"]) != divmod(
                        i, d.games_per_shuffle
                    ):
                        raise ValueError(
                            "missing, duplicate, or reordered required game coordinate"
                        )
                    push(-1, 0, 14, int(coord["termination_status"] == "safety_limit"), 1)
                for row in partial.batch_frequencies.to_pylist():
                    push(
                        row["strategy"],
                        row["seat"],
                        7 + DISTRIBUTIONS.index(row["distribution"]),
                        row["value"],
                        row["count"],
                    )
                for row in partial.batch_stats.to_pylist():
                    for metric, field in enumerate(_COUNTS):
                        push(row["strategy"], row["seat"], metric, 0, row[field])
                shuffles = selected["shuffle_index"].to_numpy()
                for seat in range(1, d.k + 1):
                    for shuffle, sid in zip(
                        shuffles, selected[f"P{seat}_strategy"].to_numpy(), strict=True
                    ):
                        push(int(sid), 0, 13, int(shuffle), 1)
            position = end
            if position >= stop:
                break
        if position < stop:
            raise ValueError("source ended before planned slice boundary")
        pq.write_table(_run_table(records[:used]), output)


@dataclass(frozen=True)
class _MergeWriter:
    budget: int

    def __call__(self, unit: PartitionedUnit, output: Path) -> None:
        rows = (
            dict(zip(_RUN_SCHEMA.names, row, strict=True))
            for row in _merged([Path(str(p)) for p in unit.key[2:]], self.budget)
        )
        _write_rows(output, _RUN_SCHEMA, rows, self.budget)


@dataclass(frozen=True)
class _NormalizeWriter:
    budget: int

    def __call__(self, unit: PartitionedUnit, output: Path) -> None:
        source = Path(str(unit.key[2]))
        # Input is one byte-admitted source unit, not a merged history.
        count = pq.ParquetFile(source).metadata.num_rows
        if count * _RUN_DTYPE.itemsize * 2 > self.budget // 2:
            raise MemoryError("normalized slice exceeds its record allowance")
        records = np.empty(count * 2, dtype=_RUN_DTYPE)
        used = 0
        for _, sid, seat, metric, value, amount in _read_rows(source, self.budget // 4):
            if metric > 12:
                continue
            records[used] = (-1, sid, seat, metric, value, amount)
            records[used + 1] = (-1, sid, 0, metric, value, amount)
            used += 2
        pq.write_table(_run_table(records[:used]), output)


def _coordinate_digest(d: _Design, batch: int) -> str:
    digest = hashlib.sha256()
    dtype = np.dtype([("root", "<i8"), ("k", "<i2"), ("shuffle", "<i8"), ("game", "<i4")])
    start, stop = batch * d.games_per_batch, (batch + 1) * d.games_per_batch
    for left in range(start, stop, 4096):
        ordinal = np.arange(left, min(stop, left + 4096), dtype=np.int64)
        keys = np.empty(len(ordinal), dtype=dtype)
        keys["root"], keys["k"] = d.root, d.k
        keys["shuffle"], keys["game"] = (
            ordinal // d.games_per_shuffle,
            ordinal % d.games_per_shuffle,
        )
        digest.update(keys.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class _CertifyWriter:
    design: _Design
    budget: int

    def __call__(self, unit: PartitionedUnit, output: Path) -> None:
        batch = int(unit.key[0])
        d = self.design
        metadata: dict[str, str] = {}

        def support_rows() -> Iterator[dict[str, Any]]:
            grouped = iter(
                groupby(_read_rows(Path(str(unit.key[2])), self.budget // 4), key=lambda r: r[1])
            )
            first = next(grouped, None)
            if first is None or first[0] != -1:
                raise ValueError("missing batch game support")
            game_counts = [0, 0]
            for b, _, seat, metric, value, count in first[1]:
                if b != batch or seat != 0 or metric != 14 or value not in (0, 1):
                    raise ValueError("invalid batch game-support record")
                game_counts[value] = _checked(game_counts[value] + count)
            if sum(game_counts) != d.games_per_batch:
                raise ValueError("missing required batch games")
            totals = [0] * 7
            emitted = 0
            for strategy, multiplicity in d.roster:
                group = next(grouped, None)
                if group is None or group[0] != strategy:
                    raise ValueError("missing or unexpected scheduled strategy support")
                values = [[0] * 7 for _ in range(d.k)]
                masses = [[0] * 6 for _ in range(d.k)]
                margins = [[0, 0] for _ in range(d.k)]
                next_shuffle = batch * d.width
                for b, _, seat, metric, value, count in group[1]:
                    if b != batch:
                        raise ValueError("batch run crosses its certified boundary")
                    if seat == 0:
                        if metric != 13 or value != next_shuffle or count != multiplicity:
                            raise ValueError("scheduled per-shuffle strategy multiplicity mismatch")
                        next_shuffle += 1
                    elif not 1 <= seat <= d.k:
                        raise ValueError("invalid actual seat")
                    elif metric < 7:
                        if value != 0:
                            raise ValueError("invalid sufficient-statistic value key")
                        values[seat - 1][metric] = _checked(values[seat - 1][metric] + count)
                    elif metric < 13:
                        index = metric - 7
                        masses[seat - 1][index] = _checked(masses[seat - 1][index] + count)
                        if index in (2, 5):
                            j = int(index == 5)
                            margins[seat - 1][j] = _checked(margins[seat - 1][j] + value * count)
                    else:
                        raise ValueError("unexpected run metric")
                if (
                    next_shuffle != (batch + 1) * d.width
                    or sum(v[0] for v in values) != multiplicity * d.width
                ):
                    raise ValueError("incomplete strategy coverage in batch")
                for seat, counts in enumerate(values, 1):
                    a, c, safety, wins, losses, deficit, victory = counts
                    if (
                        a != c + safety
                        or c != wins + losses
                        or masses[seat - 1] != [c, c, losses, c, safety, wins]
                        or margins[seat - 1] != [deficit, victory]
                    ):
                        raise ValueError("outcome frequencies and support do not conserve counts")
                    totals = [_checked(x + y) for x, y in zip(totals, counts, strict=True)]
                    emitted += 1
                    yield {
                        "root_seed": d.root,
                        "k": d.k,
                        "deterministic_batch_id": batch,
                        "strategy": strategy,
                        "seat": seat,
                        **dict(zip(_COUNTS, counts, strict=True)),
                        "support_status": "observed" if a else "verified_zero",
                    }
            if next(grouped, None) is not None:
                raise ValueError("unexpected strategy outside required roster")
            if totals[:4] != [
                d.k * d.games_per_batch,
                d.k * game_counts[0],
                d.k * game_counts[1],
                game_counts[0],
            ]:
                raise ValueError("batch game/player/winner conservation failed")
            inventory = {
                "root_seed": d.root,
                "k": d.k,
                "deterministic_batch_id": batch,
                "shuffle_start": batch * d.width,
                "shuffle_stop_exclusive": (batch + 1) * d.width,
                "games_per_shuffle": d.games_per_shuffle,
                "expected_games": d.games_per_batch,
                "observed_games": sum(game_counts),
                "completed_games": game_counts[0],
                "safety_games": game_counts[1],
                "expected_support_rows": len(d.roster) * d.k,
                "emitted_support_rows": emitted,
                "coordinate_digest": _coordinate_digest(d, batch),
                "source_identity_sha256": d.source_hash,
                "roster_identity_sha256": d.roster_hash,
                "support_status": "complete_valid",
            }
            metadata["outcome_inventory"] = json.dumps(inventory, sort_keys=True)

        _write_rows(
            output, outcome_support_schema(batch=True), support_rows(), self.budget, metadata
        )


def _source_units(d: _Design, rows_per_unit: int) -> list[PartitionedUnit]:
    parquet = pq.ParquetFile(d.source)
    if parquet.metadata.num_rows != d.batches * d.games_per_batch:
        raise ValueError("source row count does not cover required coordinates")
    group, group_start = 0, 0
    group_stop = parquet.metadata.row_group(0).num_rows
    units = []
    for batch in range(d.batches):
        end = (batch + 1) * d.games_per_batch
        for index, start in enumerate(range(batch * d.games_per_batch, end, rows_per_unit)):
            while group_stop <= start:
                group_start = group_stop
                group += 1
                group_stop += parquet.metadata.row_group(group).num_rows
            units.append(
                PartitionedUnit(
                    (batch, index, start, min(end, start + rows_per_unit), group, group_start),
                    f"batch_{batch:06d}/slice_{index:08d}.parquet",
                )
            )
    return units


def _unit_from_json(value: dict[str, Any]) -> PartitionedUnit:
    return PartitionedUnit(
        tuple(value["key"]),
        value["relative_output"],
        tuple(tuple(pair) for pair in value["input_identities"]),
    )


def _identity_from_json(value: dict[str, Any]) -> PartitionedStageIdentity:
    return PartitionedStageIdentity(
        **{**value, "input_identities": tuple(tuple(pair) for pair in value["input_identities"])}
    )


def _inputs(cfg: AppConfig, k: int) -> list[Path]:
    return [
        cfg.ingested_rows_curated(k),
        cfg.simulation_workload_plan_path(k),
        cfg.strategy_manifest_root_path(),
        cfg.n_dir(k) / "simulation.done.json",
        stage_done_path(cfg.curate_stage_dir, "curate"),
    ]


def outcome_is_up_to_date(cfg: AppConfig, k: int) -> bool:
    """Require both canonical publication and every retained durable unit."""
    paths = outcome_publication_paths(cfg, k)
    if not stage_is_up_to_date(
        stage_done_path(cfg.metrics_per_k_dir(k), "outcome_distributions"),
        inputs=_inputs(cfg, k),
        outputs=paths,
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=paths,
    ):
        return False
    return validate_outcome_checkpoints(cfg.outcome_checkpoint_inventory_path(k))


def validate_outcome_checkpoints(publication: Path) -> bool:
    """Authenticate every declared unit/stamp, including at the final audit."""
    try:
        validate_artifact_sidecar(publication)
        payload = json.loads(publication.read_text(encoding="utf-8"))
        if payload["method_version"] != OUTCOME_METHOD_VERSION or not payload["stages"]:
            return False
        for descriptor in payload["stages"]:
            name = descriptor["name"]
            if (
                not isinstance(name, str)
                or not name
                or any(c not in "abcdefghijklmnopqrstuvwxyz_0123456789" for c in name)
            ):
                return False
            root = publication.parent / name
            units = [_unit_from_json(value) for value in descriptor["units"]]
            if (
                validate_final_manifest(
                    root / "manifest.jsonl",
                    root=root,
                    identity=_identity_from_json(descriptor["identity"]),
                    unit_source=units.__iter__,
                    require_sidecar=True,
                    validator=_validate_outcome_unit,
                )
                is None
            ):
                return False
        return True
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return False


def _validate_outcome_unit(_unit: PartitionedUnit, path: Path) -> dict[str, str]:
    validate_artifact_sidecar(path, expected={"operation": "outcome_durable_unit"})
    return {"sidecar_sha256": sha256_file(sidecar_path(path))}


class _Stages:
    """Small, byte-admitted plan metadata; game data remain on disk."""

    def __init__(self, cfg: AppConfig, d: _Design, plan: Path, *, force: bool):
        self.cfg, self.design, self.plan, self.force = cfg, d, plan, force
        self.descriptors: list[dict[str, Any]] = []
        self.manifests: list[Path] = []
        self.identity = PartitionedStageIdentity(
            "analysis",
            d.root,
            (("plan", sha256_file(plan)),),
            identity_sha256(asdict(d)),
            resolved_code_identity_sha256(cfg),
            1,
            OUTCOME_METHOD_VERSION,
        )

    def run(
        self,
        name: str,
        units: list[PartitionedUnit],
        writer: Callable[[PartitionedUnit, Path], None],
        *,
        source: bool = False,
    ) -> list[Path]:
        cfg, d = self.cfg, self.design
        descriptor = {
            "name": name,
            "identity": asdict(self.identity),
            "units": [asdict(u) for u in units],
        }
        # Metadata also has a budget. Fail explicitly instead of silently growing
        # a Python plan proportional to an arbitrarily large source history.
        if len(json.dumps([*self.descriptors, descriptor])) * 4 > _budget(cfg):
            raise MemoryError("outcome checkpoint inventory exceeds admitted metadata budget")
        root = cfg.outcome_units_dir(d.k) / name
        sources = [Path(p) for p in d.inputs] + [self.plan] if source else [self.plan]
        template = _metadata(cfg, d.k, root / "manifest.jsonl", "outcome_unit_manifest", sources)
        factory = _Sidecars(
            cfg,
            d.k,
            "outcome_durable_unit",
            tuple(map(str, sources)),
            capture_v3_inputs(template) if source else None,
            not source,
        )
        run_partitioned_stage(
            root=root,
            identity=self.identity,
            unit_source=lambda: iter(units),
            writer=writer,
            resources=cfg.resources,
            requested_workers=cfg.analysis.n_jobs,
            mp_start_method=cfg.analysis.mp_start_method,
            force=self.force,
            sidecar_factory=factory,
            validator=_validate_outcome_unit,
            manifest_path=root / "manifest.jsonl",
            manifest_sidecar=template,
            temp_dir=cfg.io.temp_dir,
        )
        self.descriptors.append(descriptor)
        self.manifests.append(root / "manifest.jsonl")
        return [root / "units" / u.relative_output for u in units]

    def merge(self, name: str, groups: dict[int, list[Path]], fan_in: int) -> dict[int, Path]:
        level = 0
        while any(len(paths) > 1 for paths in groups.values()):
            units = []
            next_groups: dict[int, list[Path]] = {}
            for batch, paths in sorted(groups.items()):
                if len(paths) == 1:
                    next_groups[batch] = paths
                    continue
                for index, start in enumerate(range(0, len(paths), fan_in)):
                    chunk = paths[start : start + fan_in]
                    units.append(_dependent_unit(batch, index, chunk))
            results = self.run(f"{name}_{level:03d}", units, _MergeWriter(_budget(self.cfg)))
            for unit, path in zip(units, results, strict=True):
                next_groups.setdefault(int(unit.key[0]), []).append(path)
            groups = next_groups
            level += 1
        return {batch: paths[0] for batch, paths in groups.items()}


def _dependent_unit(batch: int, index: int, paths: list[Path]) -> PartitionedUnit:
    identities = tuple(
        (
            f"input_{i:04d}",
            identity_sha256({"data": sha256_file(p), "sidecar": sha256_file(sidecar_path(p))}),
        )
        for i, p in enumerate(paths)
    )
    return PartitionedUnit(
        (batch, index, *map(str, paths)), f"batch_{batch:06d}/unit_{index:08d}.parquet", identities
    )


def _publish_batches(
    cfg: AppConfig, d: _Design, certificates: list[Path], publication: Path
) -> None:
    for output, schema, inventory in (
        (cfg.outcome_batch_stats_path(d.k), outcome_support_schema(batch=True), False),
        (cfg.outcome_batch_inventory_path(d.k), outcome_inventory_schema(), True),
    ):
        metadata = _metadata(
            cfg,
            d.k,
            output,
            "outcome_batch_inventory" if inventory else "outcome_batch_statistics",
            [publication, *certificates],
        )
        with ParquetShardWriter(
            str(output), schema=schema, sidecar=metadata, temp_dir=cfg.io.temp_dir
        ) as writer:
            for path in certificates:
                if inventory:
                    footer = pq.ParquetFile(path).metadata.metadata
                    if footer is None:
                        raise ValueError("certified batch lacks its inventory")
                    row = json.loads(footer[b"outcome_inventory"])
                    writer.write_batch(pa.Table.from_pylist([row], schema=schema))
                else:
                    for _, _, table in iter_parquet_tables_by_bytes(
                        path,
                        columns=schema.names,
                        max_batch_bytes=_budget(cfg) // 4,
                        max_batch_rows=max(1, _budget(cfg) // 8192),
                    ):
                        writer.write_batch(table.replace_schema_metadata(None).cast(schema))


def _status(counts: dict[str, int], denominator: int, conditioning: str) -> str:
    if counts["attempted"] == 0:
        return "verified_zero"
    if conditioning.startswith("completed") and counts["completed"] == 0:
        return "no_completed_support"
    return "ok" if denominator else "no_conditional_support"


def _publish_summaries(cfg: AppConfig, d: _Design, merged: Path, publication: Path) -> None:
    """One forward scan, exact inverse-CDF quantiles, no materialized histograms."""
    schemas = (outcome_frequency_schema(), outcome_support_schema(), outcome_summary_schema())
    paths = (
        cfg.outcome_frequencies_path(d.k),
        cfg.outcome_support_path(d.k),
        cfg.outcome_summaries_path(d.k),
    )
    sources = [
        merged,
        cfg.outcome_batch_stats_path(d.k),
        cfg.outcome_batch_inventory_path(d.k),
        publication,
    ]
    rows = _read_rows(merged, _budget(cfg) // 4)
    current = next(rows, None)
    buffers: list[list[dict[str, Any]]] = [[], [], []]
    limit = max(1, _budget(cfg) // (4 * 3 * 8192))
    with ExitStack() as stack:
        writers = [
            stack.enter_context(
                ParquetShardWriter(
                    str(path),
                    schema=schema,
                    sidecar=_metadata(cfg, d.k, path, "outcome_" + name, sources),
                    temp_dir=cfg.io.temp_dir,
                )
            )
            for path, schema, name in zip(
                paths, schemas, ("frequencies", "support", "summaries"), strict=True
            )
        ]
        coalescers = [
            _CoalescedWriter(writer.write_batch, schema, _budget(cfg) // 12)
            for writer, schema in zip(writers, schemas, strict=True)
        ]

        def emit(index: int, row: dict[str, Any]) -> None:
            buffers[index].append(row)
            if len(buffers[index]) >= limit:
                coalescers[index].write_batch(
                    pa.Table.from_pylist(buffers[index], schema=schemas[index])
                )
                buffers[index].clear()

        for strategy, _multiplicity in d.roster:
            for seat in range(d.k + 1):
                key = (-1, strategy, seat)
                base = {"root_seed": d.root, "k": d.k, "strategy": strategy, "seat": seat}
                cell_summaries: list[dict[str, Any]] = []
                counts: dict[str, int] = dict.fromkeys(_COUNTS, 0)
                while current is not None and current[:3] == key and current[3] < 7:
                    counts[_COUNTS[current[3]]] = current[5]
                    current = next(rows, None)
                a, c, safety, wins, losses, deficit, victory = (counts[n] for n in _COUNTS)
                if a != c + safety or c != wins + losses:
                    raise ValueError("all-batch sufficient statistics do not conserve exposures")
                emit(1, {**base, **counts, "support_status": "observed" if a else "verified_zero"})
                for index, distribution in enumerate(DISTRIBUTIONS):
                    denominator = counts[_DENOMINATORS[index]]
                    conditioning = _CONDITIONING[index]
                    scale = 2 if distribution == "midrank" else 1
                    cumulative = total = 0
                    quantiles: dict[int, int] = {}
                    while current is not None and current[:4] == (*key, index + 7):
                        value, amount = current[4:]
                        if amount <= 0:
                            raise ValueError("frequency bins must have positive counts")
                        cumulative = _checked(cumulative + amount)
                        total = _checked(total + value * amount)
                        emit(
                            0,
                            {
                                **base,
                                "distribution": distribution,
                                "value": value,
                                "value_scale": scale,
                                "count": amount,
                            },
                        )
                        for percentile in (5, 25, 50, 75, 95):
                            if (
                                percentile not in quantiles
                                and cumulative >= (denominator * percentile + 99) // 100
                            ):
                                quantiles[percentile] = value
                        current = next(rows, None)
                    if (
                        cumulative != denominator
                        or (index == 2 and total != deficit)
                        or (index == 5 and total != victory)
                    ):
                        raise ValueError("all-batch frequency mass/margins disagree with support")
                    summary = {
                        **base,
                        "quantity": distribution,
                        "conditioning": conditioning,
                        "denominator": denominator,
                        "status": _status(counts, denominator, conditioning),
                    }
                    cell_summaries.append(
                        {
                            **summary,
                            "statistic": "mean",
                            "estimate": total / denominator / scale if denominator else None,
                        },
                    )
                    for percentile in (5, 25, 50, 75, 95):
                        cell_summaries.append(
                            {
                                **summary,
                                "statistic": f"q{percentile:02d}",
                                "estimate": quantiles[percentile] / scale if denominator else None,
                            },
                        )
                for quantity, numerator, denominator, conditioning in (
                    ("completed_win_rate", wins, c, "completed"),
                    ("completed_loss_rate", losses, c, "completed"),
                    ("mean_deficit_completed", deficit, c, "completed"),
                    ("mean_losing_margin", deficit, losses, "completed_loss"),
                    ("mean_winning_margin", victory, wins, "completed_win"),
                    ("attempted_win_rate", wins, a, "all_attempted"),
                    ("attempted_loss_rate", a - wins, a, "all_attempted"),
                    ("safety_rate", safety, a, "all_attempted"),
                ):
                    cell_summaries.append(
                        {
                            **base,
                            "quantity": quantity,
                            "statistic": "estimate",
                            "conditioning": conditioning,
                            "denominator": denominator,
                            "estimate": numerator / denominator if denominator else None,
                            "status": _status(counts, denominator, conditioning),
                        },
                    )
                for summary_row in sorted(
                    cell_summaries, key=lambda r: (r["quantity"], r["statistic"], r["conditioning"])
                ):
                    emit(2, summary_row)
        if current is not None:
            raise ValueError("unexpected all-batch strategy/seat/metric outside certified support")
        for index in range(3):
            # The empty table also establishes a schema when no conditional bins exist.
            coalescers[index].write_batch(
                pa.Table.from_pylist(buffers[index], schema=schemas[index])
            )
            coalescers[index].flush()
    rows.close()


def build_outcome_distributions(cfg: AppConfig, k: int, *, force: bool = False) -> tuple[Path, ...]:
    """Certify complete support and publish bounded, resumable exact outcomes."""
    if k not in cfg.sim.n_players_list:
        raise ValueError("outcome player count is outside configured support")
    _admit_source_footers(cfg)
    if not force and outcome_is_up_to_date(cfg, k):
        return outcome_publication_paths(cfg, k)
    d = _resolve_design(cfg, k)
    budget = _budget(cfg)
    _slice_row_limit(k, budget)  # Fail before publication if even one row cannot be admitted.
    plan_path = cfg.outcome_units_dir(k) / "plan.json"
    design_hash = identity_sha256(
        {
            "design": asdict(d),
            "inputs": [
                (
                    p,
                    sha256_file(Path(p)),
                    sha256_file(sidecar_path(p)) if sidecar_path(p).exists() else None,
                )
                for p in d.inputs
            ],
        }
    )
    code_hash = resolved_code_identity_sha256(cfg)
    plan: dict[str, Any] | None = None
    if plan_path.exists():
        try:
            validate_artifact_sidecar(plan_path)
            candidate = json.loads(plan_path.read_text(encoding="utf-8"))
            if (
                candidate["design_hash"] == design_hash
                and candidate["code_hash"] == code_hash
                and candidate["method_version"] == OUTCOME_METHOD_VERSION
                and not (force and budget < candidate["minimum_budget"])
            ):
                plan = candidate
        except (OSError, RuntimeError, ValueError, KeyError):
            pass
    if plan is None:
        checkpoint_root = cfg.outcome_units_dir(k).resolve()
        expected_root = cfg.metrics_per_k_dir(k).resolve() / "_outcome_units"
        if checkpoint_root != expected_root or not checkpoint_root.is_relative_to(
            cfg.results_root.resolve()
        ):
            raise ValueError("outcome checkpoint path escaped its configured owner")
        # A changed source/method/code cannot reuse old checkpoints. Remove only
        # this producer's generated state so stale graph members cannot survive.
        if checkpoint_root.exists():
            shutil.rmtree(checkpoint_root)
        rows_per_unit = max(1, budget // (16 * _RUN_DTYPE.itemsize * (12 * k + 2)))
        plan = {
            "method_version": OUTCOME_METHOD_VERSION,
            "design_hash": design_hash,
            "code_hash": code_hash,
            "rows_per_unit": rows_per_unit,
            "minimum_budget": budget,
            "fan_in": 16,
        }
        write_json_artifact_atomic(
            plan,
            plan_path,
            sidecar=_metadata(
                cfg, k, plan_path, "outcome_partition_plan", [Path(p) for p in d.inputs]
            ),
        )
    if budget < plan["minimum_budget"]:
        raise MemoryError("resume needs the original outcome byte budget or --force to repartition")
    # Conservative metadata admission precedes construction of the unit list.
    unit_count = d.batches * (
        (d.games_per_batch + plan["rows_per_unit"] - 1) // plan["rows_per_unit"]
    )
    if unit_count * 2048 > budget:
        raise MemoryError(
            "outcome source plan exceeds metadata budget; increase analysis stage_batch_bytes"
        )
    stages = _Stages(cfg, d, plan_path, force=force)
    units = _source_units(d, plan["rows_per_unit"])
    slices = stages.run("source", units, _SliceWriter(d, budget), source=True)
    groups: dict[int, list[Path]] = {}
    for unit, path in zip(units, slices, strict=True):
        groups.setdefault(int(unit.key[0]), []).append(path)
    batch_runs = stages.merge("batch_merge", groups, plan["fan_in"])
    certificates = stages.run(
        "certified_batches",
        [_dependent_unit(batch, 0, [path]) for batch, path in sorted(batch_runs.items())],
        _CertifyWriter(d, budget),
    )
    normalized = stages.run(
        "normalized",
        [
            _dependent_unit(int(unit.key[0]), int(unit.key[1]), [path])
            for unit, path in zip(units, slices, strict=True)
        ],
        _NormalizeWriter(budget),
    )
    merged = stages.merge("all_batch_merge", {0: normalized}, plan["fan_in"])[0]
    publication = cfg.outcome_checkpoint_inventory_path(k)
    write_json_artifact_atomic(
        {"method_version": OUTCOME_METHOD_VERSION, "stages": stages.descriptors},
        publication,
        sidecar=_metadata(cfg, k, publication, "outcome_checkpoint_inventory", stages.manifests),
    )
    _publish_batches(cfg, d, certificates, publication)
    _publish_summaries(cfg, d, merged, publication)
    paths = outcome_publication_paths(cfg, k)
    write_stage_done(
        stage_done_path(cfg.metrics_per_k_dir(k), "outcome_distributions"),
        inputs=_inputs(cfg, k),
        outputs=paths,
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=paths,
    )
    return paths
