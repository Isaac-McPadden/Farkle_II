"""Selected pointwise batch precision under first_release_task0_v1.

The four ratios share root/k/replicate draws. Completed loss is the exact
complement of completed win. These are approximate Monte Carlo intervals,
conditional on the fixed design and exchangeable independent batches; they
are neither simultaneous intervals nor tests, and do not certify rare tails.
"""

from __future__ import annotations

import json
import shutil
import struct
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.outcome_distributions import _COUNTS, outcome_support_schema
from farkle.analysis.outcome_publication import (
    _admit_source_footers,
    _checked,
    _CoalescedWriter,
    _Design,
    _identity_from_json,
    _resolve_design,
    _unit_from_json,
    _write_rows,
    outcome_inventory_schema,
    outcome_is_up_to_date,
)
from farkle.config import AppConfig, ArtifactScope
from farkle.protocols import SCREENING_INTERVAL_CONFIDENCE
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
from farkle.utils.random import RandomPurpose, coordinate_rng
from farkle.utils.release_identity import CapturedV3Inputs, capture_v3_inputs
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.writer import ParquetShardWriter

OUTCOME_PRECISION_METHOD_VERSION = 1
METHOD_ID = "outcome_batch_percentile_v1"
CONTRACT_ID = "first_release_task0_v1"
SUMMARY_IDS = (
    "completed_win_rate",
    "mean_deficit_completed",
    "mean_losing_margin",
    "mean_winning_margin",
)
_NUM = (3, 5, 5, 6)
_DEN = (1, 1, 4, 3)
_RANGE_WIDTH = 100


def outcome_precision_schema() -> pa.Schema:
    """The frozen U schema, including nulls for withheld precision."""
    fields = list(outcome_support_schema())[:4]
    fields += [pa.field("summary_id", pa.string(), nullable=False)]
    fields += [pa.field("estimate", pa.float64())]
    fields += [pa.field("denominator", pa.int64(), nullable=False)]
    fields += [
        pa.field(n, pa.int32(), nullable=False) for n in ("batch_count", "contributing_batches")
    ]
    fields += [pa.field("confidence", pa.float64(), nullable=False)]
    fields += [
        pa.field(n, pa.int32(), nullable=False)
        for n in ("replicates_requested", "replicates_valid", "replicates_invalid")
    ]
    fields += [
        pa.field(n, pa.float64())
        for n in ("ci_low", "ci_high", "interval_width", "mcse", "replicate_min", "replicate_max")
    ]
    fields += [pa.field(n, pa.string(), nullable=False) for n in ("status", "method_id")]
    return pa.schema(fields)


def _range_schema() -> pa.Schema:
    return pa.schema(
        [
            *list(outcome_support_schema())[:4],
            pa.field("summary_id", pa.string(), nullable=False),
            pa.field("replicate_start", pa.int32(), nullable=False),
            pa.field("replicate_stop_exclusive", pa.int32(), nullable=False),
            pa.field("estimates", pa.list_(pa.float64()), nullable=False),
        ]
    )


def precision_publication_paths(cfg: AppConfig, k: int) -> tuple[Path, ...]:
    return (
        cfg.outcome_precision_path(k),
        cfg.outcome_precision_ranges_dir(k) / "plan.json",
        cfg.outcome_precision_inventory_path(k),
    )


def _inputs(cfg: AppConfig, k: int) -> list[Path]:
    return [
        cfg.outcome_batch_stats_path(k),
        cfg.outcome_batch_inventory_path(k),
        cfg.outcome_support_path(k),
        cfg.outcome_checkpoint_inventory_path(k),
        cfg.simulation_workload_plan_path(k),
        cfg.strategy_manifest_root_path(),
        stage_done_path(cfg.metrics_per_k_dir(k), "outcome_distributions"),
    ]


def _specification(cfg: AppConfig) -> dict[str, Any]:
    return {
        "contract_id": CONTRACT_ID,
        "method_id": METHOD_ID,
        "outcome_precision_method_version": OUTCOME_PRECISION_METHOD_VERSION,
        "summary_ids": [*SUMMARY_IDS, "completed_loss_rate"],
        "confidence": SCREENING_INTERVAL_CONFIDENCE,
        "replicates": cfg.screening.bootstrap_replicates,
        "minimum_batches": 30,
        "minimum_contributing_batches": 30,
        "minimum_replicates": 2000,
        "undefined_resamples": "withhold_all",
        "degenerate_resamples": "withhold_all",
        "quantile_method": "linear",
        "mcse_ddof": 1,
        "rng_purpose": int(RandomPurpose.OUTCOME_BOOTSTRAP),
        "conditioning": ["completed", "completed", "completed_loss", "completed_win"],
        "claim": "pointwise_approximate_monte_carlo_precision_fixed_design",
        "assumption": "exchangeable_independent_deterministic_batches",
        "limits": "not_simultaneous_or_tests; support_policy_does_not_certify_tail_precision",
    }


def _metadata(
    cfg: AppConfig, k: int, path: Path, operation: str, sources: list[Path]
) -> ArtifactSidecar:
    columns: list[str] = []
    if path.suffix == ".parquet":
        schema = (
            outcome_support_schema(batch=True)
            if "vectors" in path.parts
            else (
                outcome_precision_schema()
                if path.name == "outcome_precision.parquet" or "summaries" in path.parts
                else _range_schema()
            )
        )
        columns = schema.names
    elif path.name == "plan.json":
        columns = [
            "contract_id",
            "method_version",
            "design_hash",
            "code_hash",
            "minimum_budget",
            "tile_size",
            "roster_size",
            "range_width",
            "specification",
        ]
    elif path.name == "publication.json":
        columns = ["method_version", "stages"]
    return make_artifact_sidecar(
        cfg,
        path,
        producer="outcome_precision",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation=operation,
        baseline="none",
        weighted_quantity="ratios_of_resampled_sums",
        support_count_role="certified_completed_player_game_exposures",
        uncertainty_method=METHOD_ID,
        replication_unit="deterministic_shuffle_batch",
        conditioning="explicit_per_summary_completed_or_official_loss_or_win",
        source_artifacts=sources,
        player_counts=[k],
        required_player_counts=[k],
        missing_cell_policy="fail",
        consistency_columns=columns,
        method_contract=cast(
            Any, {"kind": "operation", "procedure": operation, "parameters": _specification(cfg)}
        ),
    )


def _tables(path: Path, schema: pa.Schema, budget: int) -> Iterator[pa.Table]:
    for _, _, table in iter_parquet_tables_by_bytes(
        path,
        columns=schema.names,
        max_batch_bytes=budget // 4,
        max_batch_rows=max(1, budget // 8192),
    ):
        if any(table[n].null_count for n in schema.names if not schema.field(n).nullable):
            raise ValueError("null required outcome precision input")
        yield table


def _admit_parquet(path: Path, schema: pa.Schema, budget: int) -> pq.ParquetFile:
    with path.open("rb") as handle:
        handle.seek(-8, 2)
        size, magic = struct.unpack("<I4s", handle.read(8))
    if magic != b"PAR1" or size > budget // 4:
        raise MemoryError("precision source footer exceeds metadata admission or is invalid")
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow.remove_metadata() != schema:
        raise ValueError("incompatible outcome precision input schema")
    return parquet


def _validate_support(cfg: AppConfig, d: _Design, budget: int) -> None:
    """Validate the complete published grid; a missing row is never a zero."""
    bpath, ipath, spath = _inputs(cfg, d.k)[:3]
    n = len(d.roster)
    for path, schema, rows in (
        (bpath, outcome_support_schema(batch=True), d.batches * n * d.k),
        (ipath, outcome_inventory_schema(), d.batches),
        (spath, outcome_support_schema(), n * (d.k + 1)),
    ):
        validate_artifact_sidecar(path)
        if _admit_parquet(path, schema, budget).metadata.num_rows != rows:
            raise ValueError("missing or duplicate required outcome support row")
    inventory = iter(
        row for t in _tables(ipath, outcome_inventory_schema(), budget) for row in t.to_pylist()
    )
    current = next(inventory, None)
    totals = [0] * 5
    position = 0
    for table in _tables(bpath, outcome_support_schema(batch=True), budget):
        for row in table.to_pylist():
            batch, cell = divmod(position, n * d.k)
            index, seat0 = divmod(cell, d.k)
            expected = (d.root, d.k, batch, d.roster[index][0], seat0 + 1)
            if (
                tuple(
                    row[c] for c in ("root_seed", "k", "deterministic_batch_id", "strategy", "seat")
                )
                != expected
            ):
                raise ValueError("missing, duplicate, or reordered required outcome support cell")
            counts = [_checked(row[c]) for c in _COUNTS]
            a, c, s, w, losses, deficit, victory = counts
            if a != c + s or c != w + losses or (not losses and deficit) or (not w and victory):
                raise ValueError("inconsistent outcome batch counts")
            if row["support_status"] != ("observed" if a else "verified_zero"):
                raise ValueError("uncertified zero or invalid outcome support status")
            totals = [_checked(x + y) for x, y in zip(totals, counts[:5], strict=True)]
            position += 1
            if position % (n * d.k) == 0:
                if current is None or any(
                    current[key] != value
                    for key, value in {
                        "root_seed": d.root,
                        "k": d.k,
                        "deterministic_batch_id": batch,
                        "shuffle_start": batch * d.width,
                        "shuffle_stop_exclusive": (batch + 1) * d.width,
                        "games_per_shuffle": d.games_per_shuffle,
                        "expected_games": d.games_per_batch,
                        "observed_games": d.games_per_batch,
                        "expected_support_rows": n * d.k,
                        "emitted_support_rows": n * d.k,
                        "support_status": "complete_valid",
                        "source_identity_sha256": d.source_hash,
                        "roster_identity_sha256": d.roster_hash,
                    }.items()
                ):
                    raise ValueError("incomplete or incompatible batch inventory")
                completed, safety = current["completed_games"], current["safety_games"]
                if totals != [
                    d.k * d.games_per_batch,
                    d.k * completed,
                    d.k * safety,
                    completed,
                    (d.k - 1) * completed,
                ]:
                    raise ValueError("batch statistics disagree with certified inventory")
                totals = [0] * 5
                current = next(inventory, None)
    if current is not None or position != d.batches * n * d.k:
        raise ValueError("incomplete batch support")


def _read_span(parquet: pq.ParquetFile, start: int, stop: int, budget: int) -> Iterator[pa.Table]:
    """Read just intersecting row groups; never decode a whole large group."""
    offset = 0
    for rg in range(parquet.num_row_groups):
        size = parquet.metadata.row_group(rg).num_rows
        if offset >= stop:
            break
        if offset + size > start:
            position = offset
            for batch in parquet.iter_batches(
                row_groups=[rg], batch_size=max(1, budget // 8192), use_threads=False
            ):
                end = position + batch.num_rows
                left, right = max(start, position), min(stop, end)
                if right > left:
                    yield pa.Table.from_batches([batch.slice(left - position, right - left)])
                position = end
                if position >= stop:
                    break
        offset += size


@dataclass(frozen=True)
class _VectorWriter:
    cfg: AppConfig
    design: _Design
    tile_size: int
    budget: int

    def __call__(self, unit: PartitionedUnit, output: Path) -> None:
        d = self.design
        tile = int(unit.key[0])
        start = tile * self.tile_size
        roster = d.roster[start : start + self.tile_size]
        values = np.zeros((len(roster), d.k + 1, d.batches, 7), dtype=np.int64)
        parquet = pq.ParquetFile(self.cfg.outcome_batch_stats_path(d.k))
        for b in range(d.batches):
            first = (b * len(d.roster) + start) * d.k
            position = 0
            for table in _read_span(parquet, first, first + len(roster) * d.k, self.budget):
                for row in table.to_pylist():
                    i, seat0 = divmod(position, d.k)
                    if (row["strategy"], row["seat"], row["deterministic_batch_id"]) != (
                        roster[i][0],
                        seat0 + 1,
                        b,
                    ):
                        raise ValueError("batch tile input coordinate mismatch")
                    values[i, seat0 + 1, b] = [row[c] for c in _COUNTS]
                    position += 1
            if position != len(roster) * d.k:
                raise ValueError("missing required batch tile rows")
        for i in range(len(roster)):
            for c in range(7):
                _checked(sum(map(int, values[i, 1:, :, c].flat)))
        values[:, 0] = values[:, 1:].sum(axis=1)
        # S is an independent retained total for every actual/overall view.
        support = pq.ParquetFile(self.cfg.outcome_support_path(d.k))
        position = 0
        for table in _read_span(
            support, start * (d.k + 1), (start + len(roster)) * (d.k + 1), self.budget
        ):
            for row in table.to_pylist():
                i, seat = divmod(position, d.k + 1)
                if (row["root_seed"], row["k"], row["strategy"], row["seat"]) != (
                    d.root,
                    d.k,
                    roster[i][0],
                    seat,
                ):
                    raise ValueError("required outcome total support coordinate mismatch")
                if list(values[i, seat].sum(axis=0)) != [row[c] for c in _COUNTS]:
                    raise ValueError("batch vectors disagree with outcome support totals")
                position += 1
        if position != len(roster) * (d.k + 1):
            raise ValueError("missing required outcome total support")
        schema = outcome_support_schema(batch=True)
        flat = values.reshape(-1, 7)
        count = len(flat)
        arrays = [
            pa.array(np.full(count, d.root), type=pa.int64()),
            pa.array(np.full(count, d.k), type=pa.int16()),
            pa.array(np.tile(np.arange(d.batches), len(roster) * (d.k + 1)), type=pa.int32()),
            pa.array(np.repeat([s for s, _ in roster], (d.k + 1) * d.batches), type=pa.int32()),
            pa.array(
                np.tile(np.repeat(np.arange(d.k + 1), d.batches), len(roster)), type=pa.int16()
            ),
            *(pa.array(flat[:, c]) for c in range(7)),
            pa.array(np.where(flat[:, 0] > 0, "observed", "verified_zero")),
        ]
        pq.write_table(pa.Table.from_arrays(arrays, schema=schema), output)


def _load_vectors(path: Path, batches: int) -> tuple[pa.Table, np.ndarray]:
    table = pq.ParquetFile(path).read()
    values = np.column_stack([table[c].to_numpy() for c in _COUNTS]).reshape(-1, batches, 7)
    return table, values


def _support_status(values: np.ndarray, metric: int, replicates: int) -> str:
    totals = values.sum(axis=0)
    if not totals[0]:
        return "verified_zero"
    if not totals[1]:
        return "no_completed_support"
    denominator = _DEN[metric]
    if not totals[denominator]:
        return "no_conditional_support"
    if len(values) < 30 or np.count_nonzero(values[:, denominator]) < 30:
        return "insufficient_batches"
    if replicates < 2000:
        return "insufficient_replicates"
    return "resample"


def batch_multiplicities(root: int, k: int, replicate: int, batches: int) -> np.ndarray:
    rng = coordinate_rng(
        RandomPurpose.OUTCOME_BOOTSTRAP, root_seed=root, k=k, replicate_index=replicate
    )
    return np.bincount(rng.integers(0, batches, size=batches), minlength=batches)


@dataclass(frozen=True)
class _RangeWriter:
    cfg: AppConfig
    k: int
    batches: int
    budget: int

    def __call__(self, unit: PartitionedUnit, output: Path) -> None:
        start, stop, _tile = map(int, unit.key[:3])
        table, vectors = _load_vectors(Path(str(unit.key[3])), self.batches)
        eligible = np.array(
            [
                [
                    _support_status(v, m, self.cfg.screening.bootstrap_replicates) == "resample"
                    for m in range(4)
                ]
                for v in vectors
            ]
        )
        estimates = np.full((len(vectors), 4, stop - start), np.nan)
        # One multiplicity vector per root/k/j, used by every cell and metric.
        for j in range(start, stop) if eligible.any() else ():
            weights = batch_multiplicities(self.cfg.sim.seed, self.k, j, self.batches).astype(
                np.float64
            )
            # Fixed batch-axis reduction, independent of tile boundaries. Float
            # products avoid int64 overflow from repeated large batch sums.
            totals = np.einsum("b,cbm->cm", weights, vectors, dtype=np.float64, optimize=False)
            denominator = totals[:, _DEN]
            np.divide(
                totals[:, _NUM],
                denominator,
                out=estimates[:, :, j - start],
                where=eligible & (denominator > 0),
            )

        def rows() -> Iterator[dict[str, Any]]:
            for cell in range(len(vectors)):
                base = {
                    n: table[n][cell * self.batches].as_py()
                    for n in ("root_seed", "k", "strategy", "seat")
                }
                for m, summary in enumerate(SUMMARY_IDS):
                    yield {
                        **base,
                        "summary_id": summary,
                        "replicate_start": start,
                        "replicate_stop_exclusive": stop,
                        "estimates": (
                            [None if np.isnan(x) else float(x) for x in estimates[cell, m]]
                            if eligible[cell, m]
                            else []
                        ),
                    }

        _write_rows(output, _range_schema(), rows(), self.budget)


def precision_row(
    values: np.ndarray, metric: int, samples: np.ndarray, *, confidence: float, replicates: int
) -> dict[str, Any]:
    """Apply the frozen policy in order, without dropping undefined resamples."""
    denominator = int(values[:, _DEN[metric]].sum())
    numerator = int(values[:, _NUM[metric]].sum())
    row: dict[str, Any] = {
        "summary_id": SUMMARY_IDS[metric],
        "estimate": numerator / denominator if denominator else None,
        "denominator": denominator,
        "batch_count": len(values),
        "contributing_batches": int(np.count_nonzero(values[:, _DEN[metric]])),
        "confidence": confidence,
        "replicates_requested": replicates,
        "replicates_valid": 0,
        "replicates_invalid": 0,
        **dict.fromkeys(
            ("ci_low", "ci_high", "interval_width", "mcse", "replicate_min", "replicate_max")
        ),
        "status": _support_status(values, metric, replicates),
        "method_id": METHOD_ID,
    }
    if row["status"] != "resample":
        if len(samples):
            raise ValueError("unexpected resamples for a withheld-support cell")
        return row
    if len(samples) != replicates or np.isinf(samples).any():
        raise ValueError("incomplete or invalid precision replicate inventory")
    valid = samples[np.isfinite(samples)]
    row.update(replicates_valid=len(valid), replicates_invalid=replicates - len(valid))
    if len(valid):
        row.update(replicate_min=float(valid.min()), replicate_max=float(valid.max()))
    if len(valid) != replicates:
        row["status"] = "undefined_resamples"
    elif row["replicate_min"] == row["replicate_max"]:
        row["status"] = "degenerate_resamples"
    else:
        low, high = np.quantile(
            samples, [(1 - confidence) / 2, 1 - (1 - confidence) / 2], method="linear"
        )
        row.update(
            status="ok",
            ci_low=float(low),
            ci_high=float(high),
            interval_width=float(high - low),
            mcse=float(np.std(samples, ddof=1)),
        )
    return row


def _loss_complement(win: dict[str, Any]) -> dict[str, Any]:
    row = {**win, "summary_id": "completed_loss_rate"}
    for target, source in (
        ("estimate", "estimate"),
        ("ci_low", "ci_high"),
        ("ci_high", "ci_low"),
        ("replicate_min", "replicate_max"),
        ("replicate_max", "replicate_min"),
    ):
        row[target] = None if win[source] is None else 1.0 - win[source]
    return row


@dataclass(frozen=True)
class _SummaryWriter:
    cfg: AppConfig
    batches: int
    budget: int

    def __call__(self, unit: PartitionedUnit, output: Path) -> None:
        _tile, vector_path, *range_paths = unit.key
        table, vectors = _load_vectors(Path(str(vector_path)), self.batches)
        r = self.cfg.screening.bootstrap_replicates
        eligible = [[_support_status(v, m, r) == "resample" for m in range(4)] for v in vectors]
        # A disk spool, not a resident R x all-cells array or a memory map whose
        # touched pages can grow with the tile. It is disposable execution scratch.
        with tempfile.TemporaryFile(dir=output.parent) as spool:
            spool.truncate(len(vectors) * 4 * r * 8)
            expected_start = 0
            for path in range_paths:
                position = 0
                stop = min(r, expected_start + _RANGE_WIDTH)
                for chunk in _tables(Path(str(path)), _range_schema(), self.budget):
                    for row in chunk.to_pylist():
                        cell, m = divmod(position, 4)
                        expected = (
                            table["root_seed"][cell * self.batches].as_py(),
                            table["k"][cell * self.batches].as_py(),
                            table["strategy"][cell * self.batches].as_py(),
                            table["seat"][cell * self.batches].as_py(),
                            SUMMARY_IDS[m],
                            expected_start,
                            stop,
                        )
                        if tuple(row[n] for n in _range_schema().names[:-1]) != expected:
                            raise ValueError(
                                "missing, duplicate, or reordered precision replicate cell"
                            )
                        samples = np.asarray(row["estimates"], dtype=np.float64)
                        if len(samples) != (stop - expected_start if eligible[cell][m] else 0):
                            raise ValueError("incomplete precision replicate range")
                        if len(samples):
                            spool.seek(((cell * 4 + m) * r + expected_start) * 8)
                            spool.write(samples.astype("<f8").tobytes())
                        position += 1
                if position != len(vectors) * 4:
                    raise ValueError("missing precision range support rows")
                expected_start = stop
            if expected_start != r:
                raise ValueError("incomplete precision range inventory")

            def rows() -> Iterator[dict[str, Any]]:
                for cell, values in enumerate(vectors):
                    base = {
                        n: table[n][cell * self.batches].as_py()
                        for n in ("root_seed", "k", "strategy", "seat")
                    }
                    for m in range(4):
                        samples = np.array([], dtype=np.float64)
                        if eligible[cell][m]:
                            spool.seek((cell * 4 + m) * r * 8)
                            samples = np.frombuffer(spool.read(r * 8), dtype="<f8")
                        result = {
                            **base,
                            **precision_row(
                                values,
                                m,
                                samples,
                                confidence=SCREENING_INTERVAL_CONFIDENCE,
                                replicates=r,
                            ),
                        }
                        if m == 0:
                            yield _loss_complement(result)
                        yield result

            _write_rows(output, outcome_precision_schema(), rows(), self.budget)


@dataclass(frozen=True)
class _Sidecars:
    cfg: AppConfig
    k: int
    sources: tuple[str, ...]
    captured: CapturedV3Inputs | None = None
    source_start: int | None = None

    def __call__(self, unit: PartitionedUnit, path: Path) -> ArtifactSidecar:
        sources = [Path(p) for p in self.sources]
        if self.source_start is not None:
            sources += [Path(str(p)) for p in unit.key[self.source_start :]]
        metadata = _metadata(self.cfg, self.k, path, "outcome_precision_unit", sources)
        return replace(metadata, _captured_v3_inputs=self.captured) if self.captured else metadata


def _validate_unit(_unit: PartitionedUnit, path: Path) -> dict[str, str]:
    validate_artifact_sidecar(path, expected={"operation": "outcome_precision_unit"})
    return {"sidecar_sha256": sha256_file(sidecar_path(path))}


def validate_precision_checkpoints(publication: Path) -> bool:
    """Freshly authenticate every range, vector, reduction and typed stamp."""
    try:
        validate_artifact_sidecar(publication)
        payload = json.loads(publication.read_text(encoding="utf-8"))
        plan_path = publication.parent / "plan.json"
        validate_artifact_sidecar(plan_path)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if payload["method_version"] != OUTCOME_PRECISION_METHOD_VERSION or [
            d["name"] for d in payload["stages"]
        ] != ["vectors", ".", "summaries"]:
            return False
        if (
            plan["contract_id"] != CONTRACT_ID
            or plan["method_version"] != OUTCOME_PRECISION_METHOD_VERSION
            or plan["range_width"] != _RANGE_WIDTH
        ):
            return False
        tiles = (plan["roster_size"] + plan["tile_size"] - 1) // plan["tile_size"]
        r = plan["specification"]["replicates"]
        if tiles < 1 or r < 1:
            return False
        vectors = [
            publication.parent / "vectors" / "units" / f"tile-{t:08d}.parquet" for t in range(tiles)
        ]
        ranges = [
            (start, min(r, start + _RANGE_WIDTH), t, vector)
            for start in range(0, r, _RANGE_WIDTH)
            for t, vector in enumerate(vectors)
        ]
        range_paths = [
            publication.parent / "units" / f"range-{start:08d}-{stop:08d}/tile-{t:08d}.parquet"
            for start, stop, t, _ in ranges
        ]
        expected: list[list[tuple[tuple[int | str, ...], str]]] = [
            [((t,), f"tile-{t:08d}.parquet") for t in range(tiles)],
            [
                (
                    (start, stop, t, str(vector)),
                    f"range-{start:08d}-{stop:08d}/tile-{t:08d}.parquet",
                )
                for start, stop, t, vector in ranges
            ],
            [
                (
                    (
                        t,
                        str(vector),
                        *(str(range_paths[i]) for i in range(t, len(range_paths), tiles)),
                    ),
                    f"tile-{t:08d}.parquet",
                )
                for t, vector in enumerate(vectors)
            ],
        ]
        for descriptor, required in zip(payload["stages"], expected, strict=True):
            root = publication.parent / descriptor["name"]
            units = [_unit_from_json(u) for u in descriptor["units"]]
            identity = _identity_from_json(descriptor["identity"])
            # Keys retain their original spelling for exact manifest/stamp
            # authentication. Compare source paths by their resolved identity
            # so a relative producer and absolute audit name the same files.
            source_start = {"vectors": 1, ".": 3, "summaries": 1}[descriptor["name"]]
            actual_keys = [
                (
                    (
                        *u.key[:source_start],
                        *(Path(cast(str, p)).resolve() for p in u.key[source_start:]),
                    ),
                    u.relative_output,
                )
                for u in units
            ]
            required_keys = [
                (
                    (
                        *key[:source_start],
                        *(Path(cast(str, p)).resolve() for p in key[source_start:]),
                    ),
                    output,
                )
                for key, output in required
            ]
            if (
                actual_keys != required_keys
                or identity.input_identities != (("plan", sha256_file(plan_path)),)
                or identity.statistical_config_sha256 != plan["design_hash"]
                or identity.code_identity_sha256 != plan["code_hash"]
            ):
                return False
            if (
                not units
                or validate_final_manifest(
                    root / "manifest.jsonl",
                    root=root,
                    identity=identity,
                    unit_source=units.__iter__,
                    require_sidecar=True,
                    validator=_validate_unit,
                )
                is None
            ):
                return False
        return True
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return False


def precision_is_up_to_date(cfg: AppConfig, k: int) -> bool:
    paths = precision_publication_paths(cfg, k)
    return stage_is_up_to_date(
        stage_done_path(cfg.metrics_per_k_dir(k), "outcome_precision"),
        inputs=_inputs(cfg, k),
        outputs=paths,
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=paths,
    ) and validate_precision_checkpoints(cfg.outcome_precision_inventory_path(k))


def build_outcome_precision(cfg: AppConfig, k: int, *, force: bool = False) -> tuple[Path, ...]:
    """Publish U without changing simulation, batches, or requested replicates."""
    if k not in cfg.sim.n_players_list:
        raise ValueError("outcome precision k outside configured support")
    if not force and precision_is_up_to_date(cfg, k) and outcome_is_up_to_date(cfg, k):
        return precision_publication_paths(cfg, k)
    _admit_source_footers(cfg)
    if not outcome_is_up_to_date(cfg, k):
        raise ValueError("precision requires authenticated complete outcome inputs")
    d = _resolve_design(cfg, k)
    budget = int(cfg.resources.stage_batch_bytes["analysis"])
    r = cfg.screening.bootstrap_replicates
    if r < 1 or r > np.iinfo(np.int32).max:
        raise ValueError("invalid configured precision replicates")
    # Reserve separate quarters for projected rows, vectors/typed copies,
    # replicate working arrays, and buffered output. Quantiles hold one cell.
    bytes_per_strategy = (k + 1) * max(d.batches * 256, _RANGE_WIDTH * 4 * 32)
    tile_size = min(len(d.roster), budget // (4 * bytes_per_strategy))
    if tile_size < 1 or r * 8 * 8 > budget // 4:
        raise MemoryError(
            "precision needs a larger analysis stage_batch_bytes for one strategy or replicate vector"
        )
    _validate_support(cfg, d, budget)
    root = cfg.outcome_precision_ranges_dir(k)
    plan_path = root / "plan.json"
    inputs = _inputs(cfg, k)
    design_hash = identity_sha256(
        {
            "design": asdict(d),
            "specification": _specification(cfg),
            "inputs": [
                (
                    str(p),
                    sha256_file(p),
                    sha256_file(sidecar_path(p)) if sidecar_path(p).exists() else None,
                )
                for p in inputs
            ],
        }
    )
    code_hash = resolved_code_identity_sha256(cfg)
    plan: dict[str, Any] | None = None
    if plan_path.exists():
        try:
            validate_artifact_sidecar(plan_path)
            old = json.loads(plan_path.read_text(encoding="utf-8"))
            if (
                old["design_hash"] == design_hash
                and old["code_hash"] == code_hash
                and not (force and budget < old["minimum_budget"])
            ):
                plan = old
        except (OSError, RuntimeError, ValueError, KeyError):
            pass
    if plan is None:
        resolved = root.resolve()
        if resolved != cfg.metrics_per_k_dir(
            k
        ).resolve() / "_outcome_precision_ranges" or not resolved.is_relative_to(
            cfg.results_root.resolve()
        ):
            raise ValueError("precision checkpoint path escaped configured owner")
        if root.exists():
            shutil.rmtree(resolved)
        plan = {
            "contract_id": CONTRACT_ID,
            "method_version": OUTCOME_PRECISION_METHOD_VERSION,
            "design_hash": design_hash,
            "code_hash": code_hash,
            "minimum_budget": budget,
            "tile_size": tile_size,
            "roster_size": len(d.roster),
            "range_width": _RANGE_WIDTH,
            "specification": _specification(cfg),
        }
        write_json_artifact_atomic(
            plan, plan_path, sidecar=_metadata(cfg, k, plan_path, "outcome_precision_plan", inputs)
        )
    if budget < plan["minimum_budget"]:
        raise MemoryError(
            "precision resume needs original byte allowance or --force to repartition"
        )
    tile_size = int(plan["tile_size"])
    tiles = (len(d.roster) + tile_size - 1) // tile_size
    ranges = (r + _RANGE_WIDTH - 1) // _RANGE_WIDTH
    if tiles * (ranges + 2) * 4096 > budget:
        raise MemoryError("precision unit inventory exceeds metadata allowance")
    identity = PartitionedStageIdentity(
        "analysis",
        d.root,
        (("plan", sha256_file(plan_path)),),
        design_hash,
        code_hash,
        1,
        OUTCOME_PRECISION_METHOD_VERSION,
    )
    descriptors: list[dict[str, Any]] = []
    manifests: list[Path] = []

    def run(
        name: str,
        units: list[PartitionedUnit],
        writer: Callable[[PartitionedUnit, Path], None],
        sources: list[Path],
        source_start: int | None = None,
    ) -> list[Path]:
        directory = root / name
        descriptor = {
            "name": name,
            "identity": asdict(identity),
            "units": [asdict(u) for u in units],
        }
        if len(json.dumps([*descriptors, descriptor])) * 4 > budget:
            raise MemoryError("precision checkpoint metadata exceeds analysis allowance")
        metadata = _metadata(
            cfg, k, directory / "manifest.jsonl", "outcome_precision_manifest", sources
        )
        factory = _Sidecars(
            cfg,
            k,
            tuple(map(str, sources)),
            capture_v3_inputs(metadata) if source_start is None else None,
            source_start,
        )
        run_partitioned_stage(
            root=directory,
            identity=identity,
            unit_source=units.__iter__,
            writer=writer,
            resources=cfg.resources,
            requested_workers=cfg.analysis.n_jobs,
            mp_start_method=cfg.analysis.mp_start_method,
            force=force,
            sidecar_factory=factory,
            validator=_validate_unit,
            manifest_path=directory / "manifest.jsonl",
            manifest_sidecar=metadata,
            temp_dir=cfg.io.temp_dir,
        )
        descriptors.append(descriptor)
        manifests.append(directory / "manifest.jsonl")
        return [directory / "units" / u.relative_output for u in units]

    vectors = run(
        "vectors",
        [PartitionedUnit((t,), f"tile-{t:08d}.parquet") for t in range(tiles)],
        _VectorWriter(cfg, d, tile_size, budget),
        [*inputs, plan_path],
    )
    range_units = [
        PartitionedUnit(
            (start, min(r, start + _RANGE_WIDTH), t, str(vector)),
            f"range-{start:08d}-{min(r, start + _RANGE_WIDTH):08d}/tile-{t:08d}.parquet",
            (
                ("vector", sha256_file(vector)),
                ("vector_sidecar", sha256_file(sidecar_path(vector))),
            ),
        )
        for start in range(0, r, _RANGE_WIDTH)
        for t, vector in enumerate(vectors)
    ]
    shards = run(".", range_units, _RangeWriter(cfg, k, d.batches, budget), [plan_path], 3)
    summary_units = []
    for t, vector in enumerate(vectors):
        tile_paths = [vector, *(shards[i] for i in range(t, len(shards), tiles))]
        identities = tuple(
            sorted(
                (f"{i:08d}:{kind}", sha256_file(p if kind == "data" else sidecar_path(p)))
                for i, p in enumerate(tile_paths)
                for kind in ("data", "sidecar")
            )
        )
        summary_units.append(
            PartitionedUnit((t, *map(str, tile_paths)), f"tile-{t:08d}.parquet", identities)
        )
    summaries = run(
        "summaries", summary_units, _SummaryWriter(cfg, d.batches, budget), [plan_path], 1
    )
    publication = cfg.outcome_precision_inventory_path(k)
    write_json_artifact_atomic(
        {"method_version": OUTCOME_PRECISION_METHOD_VERSION, "stages": descriptors},
        publication,
        sidecar=_metadata(cfg, k, publication, "outcome_precision_inventory", manifests),
    )
    if not validate_precision_checkpoints(publication):
        raise ValueError("precision cannot publish without complete range/tile inventory")
    output = cfg.outcome_precision_path(k)
    schema = outcome_precision_schema()
    with ParquetShardWriter(
        str(output),
        schema=schema,
        sidecar=_metadata(cfg, k, output, "outcome_precision", [publication, *summaries]),
        temp_dir=cfg.io.temp_dir,
    ) as writer:
        coalescer = _CoalescedWriter(writer.write_batch, schema, budget // 4)
        for path in summaries:
            for table in _tables(path, schema, budget):
                coalescer.write_batch(table.replace_schema_metadata(None))
        coalescer.flush()
    paths = precision_publication_paths(cfg, k)
    write_stage_done(
        stage_done_path(cfg.metrics_per_k_dir(k), "outcome_precision"),
        inputs=inputs,
        outputs=paths,
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=paths,
    )
    return paths
