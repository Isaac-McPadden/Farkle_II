"""Bounded benchmark for Task 3B RNG route-unit coarsening.

This diagnostic benchmark invokes the production count/stats route writers and
reducers directly against deterministic Parquet row groups.  Every scenario has
an isolated disposable directory.  Canonical pipeline roots are never read or
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import shutil
import statistics
import sys
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis import rng_diagnostics
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose
from farkle.utils.telemetry import sample_process_resource_state
from farkle.utils.writer import atomic_path

BENCHMARK_VERSION: Final = 2
SAFE_PREFIX: Final = "farkle-task3b-"
MARKER_NAME: Final = ".farkle-task3b-owned.json"
PRODUCTION_GAMES_BY_K: Final = {
    2: 11_094_000,
    3: 7_396_000,
    4: 5_547_000,
    5: 4_437_600,
    6: 3_698_000,
    8: 2_773_500,
    10: 2_218_800,
    12: 1_849_000,
}


@dataclass(frozen=True, slots=True)
class Scenario:
    source_row_groups: int
    row_groups_per_unit: int
    workers: int
    repetition: int
    order_position: int
    warmup: bool = False
    rows_per_group: int = 1
    diagnostic_partitions: int = 8
    production_k_mix: bool = False


@dataclass(frozen=True, slots=True)
class _WriterTask:
    writer: Any
    unit: rng_diagnostics.PartitionedUnit
    output: str


@dataclass(slots=True)
class _SampledResources:
    peak_process_tree_rss_bytes: int = 0
    minimum_host_available_memory_bytes: int | None = None
    peak_native_threads: int = 0


class _Sampler:
    def __init__(self) -> None:
        self.result = _SampledResources()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> _Sampler:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            self._sample()

    def _sample(self) -> None:
        state = sample_process_resource_state()
        self.result.peak_process_tree_rss_bytes = max(
            self.result.peak_process_tree_rss_bytes,
            _numeric_int(state.get("process_tree_rss_bytes")),
        )
        available = state.get("host_available_memory_bytes")
        if isinstance(available, int):
            self.result.minimum_host_available_memory_bytes = (
                available
                if self.result.minimum_host_available_memory_bytes is None
                else min(self.result.minimum_host_available_memory_bytes, available)
            )
        self.result.peak_native_threads = max(
            self.result.peak_native_threads, _numeric_int(state.get("native_threads"))
        )


def _numeric_int(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _execute_writer(task: _WriterTask) -> dict[str, Any]:
    counters: defaultdict[str, int | float] = defaultdict(int)

    def capture(
        _phase: str,
        *,
        counters: dict[str, int | float] | None = None,
        state: str = "working",
    ) -> bool:
        del state
        for key, value in (counters or {}).items():
            captured[key] += value
        return True

    captured = counters
    original = rng_diagnostics.report_worker_progress
    started_cpu = time.process_time()
    try:
        rng_diagnostics.report_worker_progress = capture
        task.writer(task.unit, Path(task.output))
    finally:
        rng_diagnostics.report_worker_progress = original
    return {"cpu_seconds": time.process_time() - started_cpu, "counters": dict(counters)}


def _run_tasks(tasks: list[_WriterTask], workers: int) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    if workers == 1:
        results = [_execute_writer(task) for task in tasks]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            results = list(executor.map(_execute_writer, tasks))
    return results, time.perf_counter() - started


def _fixture_table(
    row_groups: int,
    rows_per_group: int = 1,
    *,
    k: int = 2,
    max_players: int = 2,
) -> pa.Table:
    rows = []
    for shuffle_index in range(row_groups):
        for game_index in range(rows_per_group):
            coordinate = shuffle_index * rows_per_group + game_index
            if k == 2 and max_players == 2 and rows_per_group == 1:
                matchups = ((1, 2), (1, 3), (2, 3), (4, 5))
                p1, p2 = matchups[shuffle_index % len(matchups)]
                participants = [p1, p2]
                winner = p2 if shuffle_index % 3 == 0 else p1
                n_rounds = 2 + (shuffle_index * 7) % 13
            else:
                participants = [
                    (coordinate * 104_729 + shuffle_index * 31 + seat * 997 + k) % 5_160
                    for seat in range(k)
                ]
                winner = participants[coordinate % k]
                n_rounds = 2 + (coordinate * 7 + k) % 13
            row = {
                "root_seed": 9,
                "k": k,
                "shuffle_index": shuffle_index,
                "game_index": game_index,
                "rng_scheme_version": RNG_SCHEME_VERSION,
                "rng_purpose_namespace": int(RandomPurpose.TOURNAMENT_GAME),
                "n_rounds": n_rounds,
                "winner_strategy": winner,
            }
            row.update(
                {
                    f"P{seat + 1}_strategy": (
                        participants[seat] if seat < k else rng_diagnostics._MISSING_STRATEGY
                    )
                    for seat in range(max_players)
                }
            )
            rows.append(row)
    schema = pa.schema(
        [
            pa.field("root_seed", pa.int64(), nullable=False),
            pa.field("k", pa.int16(), nullable=False),
            pa.field("shuffle_index", pa.int64(), nullable=False),
            pa.field("game_index", pa.int64(), nullable=False),
            pa.field("rng_scheme_version", pa.int16(), nullable=False),
            pa.field("rng_purpose_namespace", pa.int16(), nullable=False),
            pa.field("n_rounds", pa.int32(), nullable=False),
            pa.field("winner_strategy", pa.int32(), nullable=False),
            *[
                pa.field(f"P{seat + 1}_strategy", pa.int32(), nullable=False)
                for seat in range(max_players)
            ],
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _write_source(path: Path, table: pa.Table, rows_per_group: int = 1) -> None:
    with pq.ParquetWriter(path, table.schema, compression="zstd") as writer:
        for offset in range(0, table.num_rows, rows_per_group):
            writer.write_table(table.slice(offset, rows_per_group))


def _production_row_group_counts(total: int) -> dict[int, int]:
    """Allocate a bounded fixture across k using the production game proportions."""

    if total < len(PRODUCTION_GAMES_BY_K):
        raise ValueError("production-k-mix scenarios require at least one row group per k")
    remaining = total - len(PRODUCTION_GAMES_BY_K)
    weight_total = sum(PRODUCTION_GAMES_BY_K.values())
    exact = {k: remaining * games / weight_total for k, games in PRODUCTION_GAMES_BY_K.items()}
    counts = {k: 1 + math.floor(value) for k, value in exact.items()}
    unassigned = total - sum(counts.values())
    for k in sorted(exact, key=lambda value: (-(exact[value] % 1), value))[:unassigned]:
        counts[k] += 1
    return counts


def _write_selection(
    path: Path, table: pa.Table, partition_count: int, strat_cols: tuple[str, ...]
) -> None:
    arrays = rng_diagnostics._extract_batch_arrays(
        table.combine_chunks().to_batches(max_chunksize=max(1, table.num_rows))[0],
        winner_col="winner_strategy",
        strat_cols=strat_cols,
        expected_root_seed=9,
    )
    counts = rng_diagnostics._count_records(arrays)
    dtype = rng_diagnostics._selection_key_dtype(len(strat_cols))
    selected = np.empty(counts.size, dtype=dtype)
    for name in dtype.names or ():
        selected[name] = counts[name]
    partitions = rng_diagnostics._stable_partitions(counts, partition_count)
    columns = [pa.array(partitions, type=pa.int16())]
    fields = [pa.field("partition", pa.int16(), nullable=False)]
    for name in dtype.names or ():
        arrow_type = {
            "group_type": pa.uint8(),
            "k": pa.int16(),
            "group_id": pa.uint64(),
        }.get(name, pa.int32())
        columns.append(pa.array(selected[name], type=arrow_type))
        fields.append(pa.field(name, arrow_type, nullable=False))
    pq.write_table(pa.Table.from_arrays(columns, schema=pa.schema(fields)), path)


def _aggregate(results: list[dict[str, Any]]) -> tuple[float, dict[str, int | float]]:
    cpu = 0.0
    counters: defaultdict[str, int | float] = defaultdict(int)
    for result in results:
        cpu += float(result["cpu_seconds"])
        for key, value in dict(result["counters"]).items():
            counters[key] += value
    return cpu, dict(counters)


def _digest_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _projected_source_bytes(path: Path, columns: tuple[str, ...]) -> int:
    with pq.ParquetFile(path) as parquet:
        indices = [parquet.schema_arrow.get_field_index(name) for name in columns]
        return sum(
            parquet.metadata.row_group(row_group).column(index).total_compressed_size
            for row_group in range(parquet.num_row_groups)
            for index in indices
        )


def run_scenario(root: Path, scenario: Scenario) -> dict[str, Any]:
    root.mkdir(parents=True)
    row_groups_by_k = (
        _production_row_group_counts(scenario.source_row_groups)
        if scenario.production_k_mix
        else {2: scenario.source_row_groups}
    )
    max_players = max(row_groups_by_k)
    tables = {
        k: _fixture_table(
            row_groups,
            scenario.rows_per_group,
            k=k,
            max_players=max_players,
        )
        for k, row_groups in row_groups_by_k.items()
    }
    table = pa.concat_tables(list(tables.values()))
    source_paths: list[Path] = []
    source_inventory: list[tuple[int, str, int]] = []
    for k, source_table in tables.items():
        source = root / f"source-{k}p.parquet"
        _write_source(source, source_table, scenario.rows_per_group)
        source_paths.append(source)
        for row_group in range(row_groups_by_k[k]):
            source_inventory.append((len(source_inventory), str(source), row_group))
    selection = root / "selection.parquet"
    partition_count = scenario.diagnostic_partitions
    strat_cols = tuple(f"P{seat + 1}_strategy" for seat in range(max_players))
    source_batch_bytes = 32 * 1024 * 1024 if scenario.production_k_mix else 4096
    reduce_run_bytes = 256 * 1024 * 1024 if scenario.production_k_mix else 0
    count_batch_rows = (
        reduce_run_bytes // rng_diagnostics._count_dtype(max_players).itemsize
        if scenario.production_k_mix
        else 256
    )
    stats_batch_rows = (
        reduce_run_bytes // rng_diagnostics._observation_dtype(max_players).itemsize
        if scenario.production_k_mix
        else 128
    )
    _write_selection(selection, table, partition_count, strat_cols)
    sources = tuple(source_inventory)
    units = tuple(
        rng_diagnostics._row_group_units(sources, row_groups_per_unit=scenario.row_groups_per_unit)
    )
    columns = (
        "root_seed",
        "k",
        "shuffle_index",
        "game_index",
        "rng_scheme_version",
        "rng_purpose_namespace",
        "n_rounds",
        "winner_strategy",
        *strat_cols,
    )
    count_dir = root / "count_route"
    eligibility_dir = root / "eligibility"
    stats_route_dir = root / "stats_route"
    stats_dir = root / "stats"
    for directory in (count_dir, eligibility_dir, stats_route_dir, stats_dir):
        directory.mkdir()

    phase_wall: dict[str, float] = {}
    all_results: list[dict[str, object]] = []
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    with _Sampler() as sampler:
        count_writer = rng_diagnostics._CountRouteWriter(
            sources,
            columns,
            "winner_strategy",
            strat_cols,
            partition_count,
            source_batch_bytes,
            9,
        )
        tasks = [
            _WriterTask(count_writer, unit, str(count_dir / unit.relative_output)) for unit in units
        ]
        results, phase_wall["count_route"] = _run_tasks(tasks, scenario.workers)
        all_results.extend(results)
        count_paths = tuple(str(count_dir / unit.relative_output) for unit in units)

        eligibility_writer = rng_diagnostics._EligibilityWriter(
            count_paths, partition_count, max_players, 3, 9, count_batch_rows
        )
        tasks = [
            _WriterTask(
                eligibility_writer,
                rng_diagnostics.PartitionedUnit((partition,), f"part-{partition:03d}.parquet"),
                str(eligibility_dir / f"part-{partition:03d}.parquet"),
            )
            for partition in range(partition_count)
        ]
        results, phase_wall["count_reduce"] = _run_tasks(tasks, scenario.workers)
        all_results.extend(results)

        stats_writer = rng_diagnostics._StatsRouteWriter(
            sources,
            str(selection),
            columns,
            "winner_strategy",
            strat_cols,
            partition_count,
            source_batch_bytes,
            9,
            64 * 1024 * 1024,
        )
        tasks = [
            _WriterTask(stats_writer, unit, str(stats_route_dir / unit.relative_output))
            for unit in units
        ]
        results, phase_wall["stats_route"] = _run_tasks(tasks, scenario.workers)
        all_results.extend(results)
        stats_paths = tuple(str(stats_route_dir / unit.relative_output) for unit in units)

        stats_partition_writer = rng_diagnostics._StatsPartitionWriter(
            stats_paths, partition_count, max_players, (1,), stats_batch_rows
        )
        tasks = [
            _WriterTask(
                stats_partition_writer,
                rng_diagnostics.PartitionedUnit((partition,), f"part-{partition:03d}.parquet"),
                str(stats_dir / f"part-{partition:03d}.parquet"),
            )
            for partition in range(partition_count)
        ]
        results, phase_wall["stats_reduce"] = _run_tasks(tasks, scenario.workers)
        all_results.extend(results)

        route_files = sorted(count_dir.glob("*.arrow")) + sorted(stats_route_dir.glob("*.arrow"))
        hash_started = time.perf_counter()
        for path in route_files:
            sha256_file(path)
        hash_seconds = time.perf_counter() - hash_started

    worker_cpu, counters = _aggregate(all_results)
    total_wall = time.perf_counter() - started_wall
    parent_cpu = time.process_time() - started_cpu
    total_cpu = parent_cpu if scenario.workers == 1 else parent_cpu + worker_cpu
    route_bytes = sum(path.stat().st_size for path in route_files)
    final_files = sorted(eligibility_dir.glob("*.parquet")) + sorted(stats_dir.glob("*.parquet"))
    return {
        "scenario": asdict(scenario),
        "route_layout_version": rng_diagnostics._ROUTE_LAYOUT_VERSION,
        "partition_schema_version": rng_diagnostics._PARTITION_SCHEMA_VERSION,
        "diagnostic_method_version": rng_diagnostics._DIAGNOSTIC_METHOD_VERSION,
        "diagnostic_partitions": partition_count,
        "source_batch_bytes": source_batch_bytes,
        "reduce_run_bytes": reduce_run_bytes,
        "source_rows_by_k": {str(k): value.num_rows for k, value in tables.items()},
        "source_rows": table.num_rows,
        "source_row_groups": scenario.source_row_groups,
        "source_file_bytes": sum(path.stat().st_size for path in source_paths),
        "source_read_bytes": sum(_projected_source_bytes(path, columns) for path in source_paths)
        * 2,
        "source_read_bytes_definition": (
            "selected-column compressed bytes across count and stats route scans; "
            "excludes repeated Parquet footer bytes"
        ),
        "durable_route_units_per_phase": len(units),
        "durable_route_files": len(route_files),
        "durable_route_unit_stamps": len(route_files),
        "route_bytes": route_bytes,
        "source_arrow_batches": int(counters.get("source_arrow_batches", 0)),
        "route_arrow_batches": int(counters.get("route_arrow_batches", 0)),
        "raw_count_records": int(counters.get("raw_count_records", 0)),
        "batch_reduced_count_records": int(counters.get("batch_reduced_count_records", 0)),
        "routed_count_records": int(counters.get("routed_count_records", 0)),
        "raw_stats_records": int(counters.get("raw_stats_records", 0)),
        "routed_stats_records": int(counters.get("routed_stats_records", 0)),
        "reducer_route_unit_opens": len(units) * partition_count * 2,
        "selection_membership_loads": len(units),
        "initial_spill_runs": int(counters.get("spill_runs_created", 0)),
        "initial_spill_bytes": int(counters.get("spill_bytes_written", 0)),
        "merge_passes": int(counters.get("merge_passes_completed", 0)),
        "merge_outputs": int(counters.get("merge_runs_created", 0)),
        "merge_bytes": int(counters.get("merge_bytes_written", 0)),
        "hashing_authentication_seconds": hash_seconds,
        "hashing_authentication_bytes": route_bytes,
        "hashing_authentication_files": len(route_files),
        "phase_wall_seconds": phase_wall,
        "total_wall_seconds": total_wall,
        "parent_cpu_seconds": parent_cpu,
        "worker_cpu_seconds": worker_cpu,
        "total_cpu_seconds": total_cpu,
        "peak_process_tree_rss_bytes": sampler.result.peak_process_tree_rss_bytes,
        "minimum_host_available_memory_bytes": (sampler.result.minimum_host_available_memory_bytes),
        "peak_native_threads": sampler.result.peak_native_threads,
        "requested_workers": scenario.workers,
        "effective_workers": min(scenario.workers, max(1, len(units))),
        "interruption_recovery_row_groups": min(
            scenario.row_groups_per_unit, scenario.source_row_groups
        ),
        "retry_events": 0,
        "downshift_events": 0,
        "memory_pause_events": 0,
        "failure_events": 0,
        "cleanup_failures": 0,
        "exact_equivalence_digest": _digest_files(final_files),
    }


def summarize(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for measurement in measurements:
        scenario = dict(measurement["scenario"])
        if not scenario.get("warmup"):
            grouped[
                (
                    int(scenario["source_row_groups"]),
                    int(scenario["row_groups_per_unit"]),
                    int(scenario["workers"]),
                )
            ].append(measurement)
    rows = []
    for (scale, range_size, workers), values in sorted(grouped.items()):
        digests = {str(value["exact_equivalence_digest"]) for value in values}
        rows.append(
            {
                "source_row_groups": scale,
                "row_groups_per_unit": range_size,
                "workers": workers,
                "repetitions": len(values),
                "median_wall_seconds": statistics.median(
                    float(value["total_wall_seconds"]) for value in values
                ),
                "median_cpu_seconds": statistics.median(
                    float(value["total_cpu_seconds"]) for value in values
                ),
                "maximum_peak_process_tree_rss_bytes": max(
                    int(value["peak_process_tree_rss_bytes"]) for value in values
                ),
                "durable_route_files": values[0]["durable_route_files"],
                "durable_route_unit_stamps": values[0]["durable_route_unit_stamps"],
                "reducer_route_unit_opens": values[0]["reducer_route_unit_opens"],
                "selection_membership_loads": values[0]["selection_membership_loads"],
                "median_initial_spill_runs": statistics.median(
                    int(value["initial_spill_runs"]) for value in values
                ),
                "equivalence_digest": next(iter(digests)),
                "repetition_digests_identical": len(digests) == 1,
            }
        )
    by_scale: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        if int(row["workers"]) == 1:
            by_scale[int(row["source_row_groups"])].add(str(row["equivalence_digest"]))
    return {
        "groups": rows,
        "exact_equivalence_across_range_sizes": {
            str(scale): len(digests) == 1 for scale, digests in sorted(by_scale.items())
        },
    }


def _safe_root(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.name.startswith(SAFE_PREFIX):
        raise ValueError(f"benchmark root name must start with {SAFE_PREFIX!r}")
    if resolved.exists():
        raise FileExistsError(f"benchmark root must not already exist: {resolved}")
    return resolved


def _prepare_root(path: Path, settings: Mapping[str, object], *, resume: bool) -> Path:
    resolved = path.resolve()
    if not resolved.name.startswith(SAFE_PREFIX):
        raise ValueError(f"benchmark root name must start with {SAFE_PREFIX!r}")
    marker = resolved / MARKER_NAME
    expected = {"benchmark_version": BENCHMARK_VERSION, "settings": settings}
    if resolved.exists():
        if not resume:
            raise FileExistsError(f"benchmark root must not already exist: {resolved}")
        try:
            actual = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("benchmark resume root lacks a valid ownership marker") from exc
        if actual != expected:
            raise ValueError("benchmark resume settings do not match the ownership marker")
        return resolved
    resolved.mkdir(parents=True)
    with atomic_path(str(marker)) as staged:
        Path(staged).write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")
    return resolved


def _recorded_scenario(root: Path, name: str, scenario: Scenario) -> dict[str, Any]:
    scenario_root = root / name
    checkpoint = scenario_root / "measurement.json"
    if checkpoint.is_file():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if payload.get("scenario") != asdict(scenario):
            raise ValueError(f"benchmark scenario checkpoint identity mismatch: {name}")
        return payload
    if scenario_root.exists():
        if scenario_root.parent.resolve() != root.resolve():
            raise ValueError("partial benchmark scenario escaped its owned root")
        shutil.rmtree(scenario_root)
    payload = run_scenario(scenario_root, scenario)
    with atomic_path(str(checkpoint)) as staged:
        Path(staged).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
    return payload


def _remove_owned_root(path: Path, settings: Mapping[str, object]) -> None:
    resolved = path.resolve()
    if not resolved.name.startswith(SAFE_PREFIX):
        raise ValueError("refusing to remove a benchmark root without the safe prefix")
    marker = resolved / MARKER_NAME
    expected = {"benchmark_version": BENCHMARK_VERSION, "settings": settings}
    try:
        actual = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("refusing to remove a benchmark root without its marker") from exc
    if actual != expected:
        raise ValueError("refusing to remove a benchmark root with a mismatched marker")
    shutil.rmtree(resolved)


def run_benchmark(
    root: Path,
    *,
    scales: tuple[int, ...],
    range_sizes: tuple[int, ...],
    repetitions: int,
    include_two_workers: bool,
    resume: bool = False,
    rows_per_group: int = 1,
    diagnostic_partitions: int = 8,
    production_k_mix: bool = False,
) -> dict[str, Any]:
    settings: dict[str, object] = {
        "scales": list(scales),
        "range_sizes": list(range_sizes),
        "repetitions": repetitions,
        "diagnostic_partitions": diagnostic_partitions,
        "alternating_order": True,
        "warmup": True,
        "include_two_workers": include_two_workers,
        "rows_per_group": rows_per_group,
        "production_k_mix": production_k_mix,
    }
    root = _prepare_root(root, settings, resume=resume)
    measurements: list[dict[str, Any]] = []
    warmup = Scenario(
        min(scales),
        max(range_sizes),
        1,
        0,
        0,
        True,
        rows_per_group,
        diagnostic_partitions,
        production_k_mix,
    )
    measurements.append(_recorded_scenario(root, "warmup", warmup))
    for scale in scales:
        for repetition in range(1, repetitions + 1):
            order = range_sizes if repetition % 2 else tuple(reversed(range_sizes))
            for position, range_size in enumerate(order, start=1):
                scenario = Scenario(
                    scale,
                    range_size,
                    1,
                    repetition,
                    position,
                    False,
                    rows_per_group,
                    diagnostic_partitions,
                    production_k_mix,
                )
                measurements.append(
                    _recorded_scenario(
                        root,
                        f"rg{scale}-range{range_size}-w1-r{repetition}",
                        scenario,
                    )
                )
        if include_two_workers:
            scenario = Scenario(
                scale,
                max(range_sizes),
                2,
                1,
                1,
                False,
                rows_per_group,
                diagnostic_partitions,
                production_k_mix,
            )
            measurements.append(
                _recorded_scenario(root, f"rg{scale}-range{max(range_sizes)}-w2-r1", scenario)
            )
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "python": sys.version,
        "platform": sys.platform,
        "settings": settings,
        "measurements": measurements,
    }
    payload["summary"] = summarize(measurements)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=int, nargs="+", default=(256, 1024))
    parser.add_argument("--range-sizes", type=int, nargs="+", default=(1, 16, 32))
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--rows-per-group", type=int, default=1)
    parser.add_argument("--diagnostic-partitions", type=int, default=8)
    parser.add_argument("--production-k-mix", action="store_true")
    parser.add_argument("--skip-two-workers", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.repetitions < 1
        or args.rows_per_group < 1
        or args.diagnostic_partitions < 1
        or (args.production_k_mix and min(args.scales) < len(PRODUCTION_GAMES_BY_K))
        or any(value < 1 for value in (*args.scales, *args.range_sizes))
    ):
        parser.error(
            "scales, range sizes, rows per group, partitions, and repetitions must be positive; "
            "production-k-mix scales must cover every configured k"
        )
    owned_temporary = args.work_root is None
    root = (
        Path(tempfile.gettempdir()) / f"{SAFE_PREFIX}{time.time_ns()}"
        if owned_temporary
        else args.work_root
    )
    payload = run_benchmark(
        root,
        scales=tuple(args.scales),
        range_sizes=tuple(args.range_sizes),
        repetitions=args.repetitions,
        include_two_workers=not args.skip_two_workers,
        resume=args.resume,
        rows_per_group=args.rows_per_group,
        diagnostic_partitions=args.diagnostic_partitions,
        production_k_mix=args.production_k_mix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(args.output)) as staged:
        Path(staged).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if owned_temporary and not args.keep_work:
        settings = payload["settings"]
        if not isinstance(settings, dict):
            raise TypeError("benchmark settings must be an object")
        _remove_owned_root(root, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
