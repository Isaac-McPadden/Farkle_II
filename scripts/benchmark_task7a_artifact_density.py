"""Task 7A physical artifact-density inventory and bounded benchmark.

The driver reads accepted result trees without mutating them and writes only to
an explicitly owned, noncanonical Task 7A root.  It does not implement a new
production layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.utils.artifact_contract import sha256_file, validate_artifact_sidecar
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.schema_helpers import expected_schema_for, raw_simulation_schema_for
from farkle.utils.writer import atomic_path

SCHEMA_VERSION: Final = 1
MARKER_NAME: Final = "task7a_ownership.json"
CHECKPOINT_NAME: Final = "task7a_measurements.json"
SAFE_ROOT_NAME: Final = "farkle-task7a-artifact-density-v1"
DEFAULT_ROOT: Final = Path("data") / SAFE_ROOT_NAME
DEFAULT_CURRENT_ROOT: Final = Path("data/results_rng_opt_fast_seed_pair_56_57")
DEFAULT_HISTORICAL_ROOT: Final = Path("data/results_efficiency_updates_4D_fast_seed_pair_54_55")
DEFAULT_ROWS: Final = 16_384
DEFAULT_REPETITIONS: Final = 3


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as staged:
        Path(staged).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _settings(
    current_root: Path,
    historical_root: Path,
    *,
    rows: int,
    repetitions: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "current_root": str(current_root.resolve()),
        "historical_root": str(historical_root.resolve()),
        "rows": rows,
        "repetitions": repetitions,
        "many_files": 64,
        "few_files": 4,
        "row_group_size": 64_000,
        "compression": "snappy",
    }


def prepare_owned_root(
    root: Path,
    settings: Mapping[str, Any],
    *,
    force: bool = False,
) -> Path:
    """Create or validate the exact Task 7A owned root."""

    resolved = root.resolve()
    if resolved.name != SAFE_ROOT_NAME:
        raise ValueError(f"Task 7A root must have exact name {SAFE_ROOT_NAME!r}")
    marker = resolved / MARKER_NAME
    expected = {
        "task": "7A",
        "schema_version": SCHEMA_VERSION,
        "noncanonical": True,
        "owned_root": str(resolved),
        "settings": dict(settings),
        "owned_children": ["fixtures"],
    }
    if resolved.exists():
        actual = _read_object(marker)
        if actual != expected:
            raise ValueError("Task 7A ownership marker or settings mismatch")
        if force:
            fixtures = (resolved / "fixtures").resolve()
            if fixtures.parent != resolved or fixtures.name != "fixtures":
                raise ValueError("refusing unsafe Task 7A fixture cleanup")
            if fixtures.exists():
                _remove_owned_tree(fixtures)
            (resolved / CHECKPOINT_NAME).unlink(missing_ok=True)
        return resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.mkdir()
    _atomic_json(marker, expected)
    return resolved


def _remove_owned_tree(path: Path) -> None:
    """Remove an already ownership-checked tree, clearing provider read-only bits."""

    def _clear_and_retry(function: Any, target: str, _error: BaseException) -> None:
        os.chmod(target, stat.S_IWRITE)
        function(target)

    shutil.rmtree(path, onexc=_clear_and_retry)


def _role(path: Path) -> str:
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    if name.endswith(".sidecar.json"):
        return "sidecar"
    if name.endswith(".unit.done.json"):
        return "partition_stamp"
    if name.endswith(".done.json"):
        return "completion"
    if "manifest" in name:
        return "manifest"
    if "checkpoint" in parts or "checkpoints" in parts or name.endswith(".pkl"):
        return "checkpoint"
    if name.startswith(("._", "tmp", ".tmp")) or any(part.startswith("._") for part in parts):
        return "temporary"
    return "canonical_or_support"


def _stage(relative: Path) -> str:
    parts = relative.parts
    for part in parts:
        if len(part) >= 3 and part[:2].isdigit() and part[2] == "_":
            return part
    if any("_rows" in part for part in parts):
        return "simulation_rows"
    if relative.name in {
        "pipeline_health.json",
        "pipeline_telemetry.json",
        "run_context.json",
    }:
        return "root_lifecycle"
    return "other"


def _percentiles(values: Sequence[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
    }


def _walk_json(value: object, keys: Counter[str], strings: Counter[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            keys[str(key)] += 1
            _walk_json(child, keys, strings)
    elif isinstance(value, list):
        for child in value:
            _walk_json(child, keys, strings)
    elif isinstance(value, str) and len(value) >= 8:
        strings[value] += 1


def _infer_k(path: Path, parquet: pq.ParquetFile) -> int | None:
    for part in reversed(path.parts):
        for suffix in ("_players", "p", "_rows"):
            prefix = part.split(suffix, 1)[0]
            if suffix in part and prefix.isdigit():
                return int(prefix)
    if "k" in parquet.schema_arrow.names and parquet.metadata.num_rows:
        try:
            column = pq.read_table(path, columns=["k"]).column(0)
            unique = column.unique().to_pylist()
            if len(unique) == 1 and unique[0] is not None:
                return int(unique[0])
        except (OSError, pa.ArrowException, ValueError):
            return None
    return None


def _parquet_record(path: Path, root: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    file_bytes = path.stat().st_size
    row_groups: list[int] = []
    compressed_data_bytes = 0
    padded_bytes = 0
    padded_columns: set[str] = set()
    k = _infer_k(path, parquet)
    for group_index in range(metadata.num_row_groups):
        group = metadata.row_group(group_index)
        row_groups.append(int(group.num_rows))
        for column_index in range(group.num_columns):
            column = group.column(column_index)
            compressed = int(column.total_compressed_size)
            compressed_data_bytes += compressed
            name = str(column.path_in_schema).split(".", 1)[0]
            if k is not None and name.startswith("P") and "_" in name:
                seat = name[1:].split("_", 1)[0]
                if seat.isdigit() and int(seat) > k:
                    padded_bytes += compressed
                    padded_columns.add(name)
    schema = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in parquet.schema_arrow
    ]
    return {
        "path": str(path.relative_to(root)),
        "file_bytes": file_bytes,
        "rows": int(metadata.num_rows),
        "row_groups": int(metadata.num_row_groups),
        "row_group_rows": row_groups,
        "footer_metadata_bytes": int(metadata.serialized_size),
        "compressed_column_bytes": compressed_data_bytes,
        "inferred_k": k,
        "padded_player_column_bytes": padded_bytes,
        "padded_player_columns": sorted(padded_columns),
        "schema": schema if metadata.num_rows == 0 else None,
    }


def inventory_tree(root: Path, *, detailed_parquets: bool = False) -> dict[str, Any]:
    """Return a streaming structural inventory of one immutable evidence tree."""

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    by_type: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    by_role: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    by_stage: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    keys: Counter[str] = Counter()
    strings: Counter[str] = Counter()
    jsonl_records: list[dict[str, Any]] = []
    parquet_details: list[dict[str, Any]] = []
    parquet_rows: list[int] = []
    parquet_sizes: list[int] = []
    row_group_rows: list[int] = []
    zero_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    total_files = 0
    total_bytes = 0
    footer_bytes = 0
    compressed_column_bytes = 0
    padded_bytes = 0
    start = time.perf_counter()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        size = path.stat().st_size
        suffix = path.suffix.lower() or "[none]"
        role = _role(relative)
        stage = _stage(relative)
        total_files += 1
        total_bytes += size
        for bucket, key in ((by_type, suffix), (by_role, role), (by_stage, stage)):
            bucket[key]["files"] += 1
            bucket[key]["bytes"] += size
        if suffix == ".parquet":
            try:
                record = _parquet_record(path, root)
                parquet_rows.append(int(record["rows"]))
                parquet_sizes.append(size)
                row_group_rows.extend(record.pop("row_group_rows"))
                footer_bytes += int(record["footer_metadata_bytes"])
                compressed_column_bytes += int(record["compressed_column_bytes"])
                padded_bytes += int(record["padded_player_column_bytes"])
                if int(record["rows"]) == 0:
                    zero_rows.append(record)
                elif detailed_parquets:
                    parquet_details.append(record)
            except (OSError, pa.ArrowException, ValueError) as exc:
                errors.append({"path": str(relative), "error": repr(exc)})
        elif suffix in {".json", ".jsonl"}:
            try:
                text = path.read_text(encoding="utf-8")
                if suffix == ".jsonl":
                    physical = [line for line in text.splitlines() if line.strip()]
                    parsed_lines = 0
                    for line in physical:
                        try:
                            json.loads(line)
                            parsed_lines += 1
                        except json.JSONDecodeError:
                            pass
                    structurally_valid = parsed_lines == len(physical)
                    jsonl_records.append(
                        {
                            "path": str(relative),
                            "bytes": size,
                            "nonblank_lines": len(physical),
                            "parsed_lines": parsed_lines,
                            "structurally_valid": structurally_valid,
                        }
                    )
                value = json.loads(text)
                _walk_json(value, keys, strings)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append({"path": str(relative), "error": repr(exc)})
    parquet_count = by_type[".parquet"]["files"]
    json_bytes = sum(value["bytes"] for key, value in by_type.items() if key in {".json", ".jsonl"})
    return {
        "root": str(root),
        "read_only": True,
        "inventory_wall_seconds": time.perf_counter() - start,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "by_type": dict(sorted(by_type.items())),
        "by_role": dict(sorted(by_role.items())),
        "by_stage": dict(sorted(by_stage.items())),
        "parquet": {
            "files": parquet_count,
            "rows": sum(parquet_rows),
            "row_groups": len(row_group_rows),
            "footer_metadata_bytes": footer_bytes,
            "footer_fraction_of_parquet_bytes": (
                footer_bytes / sum(parquet_sizes) if parquet_sizes else None
            ),
            "compressed_column_bytes": compressed_column_bytes,
            "padded_player_column_bytes": padded_bytes,
            "file_rows_distribution": _percentiles(parquet_rows),
            "file_bytes_distribution": _percentiles(parquet_sizes),
            "row_group_rows_distribution": _percentiles(row_group_rows),
            "zero_row_files": zero_rows,
            "details": parquet_details,
        },
        "json": {
            "files": by_type[".json"]["files"] + by_type[".jsonl"]["files"],
            "bytes": json_bytes,
            "jsonl": jsonl_records,
            "invalid_jsonl": [item for item in jsonl_records if not item["structurally_valid"]],
            "top_repeated_keys": keys.most_common(30),
            "top_repeated_strings": strings.most_common(30),
        },
        "errors": errors,
    }


def _fixture_table(rows: int, n_players: int, *, pad_to: int | None = None) -> pa.Table:
    schema = raw_simulation_schema_for(n_players)
    arrays: list[pa.Array] = []
    index = np.arange(rows)
    for field in schema:
        name = field.name
        dtype = field.type
        if name == "termination_status":
            values: object = np.full(rows, "completed", dtype=object)
        elif name == "winner_seat":
            values = np.full(rows, "P1", dtype=object)
        elif name == "seat_ranks":
            values = [[f"P{i}" for i in range(1, n_players + 1)] for _ in range(rows)]
        elif pa.types.is_boolean(dtype):
            values = np.zeros(rows, dtype=bool)
        elif pa.types.is_string(dtype):
            values = np.full(rows, "value", dtype=object)
        elif pa.types.is_integer(dtype):
            if name == "k":
                values = np.full(rows, n_players)
            elif name in {"outcome_schema_version", "rng_scheme_version"}:
                values = np.full(rows, 2)
            elif name.startswith("P") and name.endswith("_strategy"):
                seat = int(name[1:].split("_", 1)[0])
                values = (index + seat) % 5_160
            else:
                bit_width = dtype.bit_width
                # Keep generated values valid for narrow int8/int16 production
                # fields while retaining deterministic variation.
                values = index % min(10_000, (1 << (bit_width - 1)) - 1)
        else:
            raise TypeError(f"unsupported fixture type: {name}: {dtype}")
        arrays.append(pa.array(values, type=dtype))
    table = pa.Table.from_arrays(arrays, schema=schema)
    if pad_to is None or pad_to == n_players:
        return table
    target = expected_schema_for(pad_to)
    padded: list[pa.Array | pa.ChunkedArray] = []
    for field in target:
        if field.name in table.schema.names:
            padded.append(table[field.name].cast(field.type))
        else:
            padded.append(pa.nulls(rows, type=field.type))
    return pa.Table.from_arrays(padded, schema=target)


class _ResourceSampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self.peak_rss = 0
        self.peak_threads = 0
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.wait(0.02):
            processes = [process, *process.children(recursive=True)]
            rss = 0
            threads = 0
            for child in processes:
                try:
                    rss += child.memory_info().rss
                    threads += child.num_threads()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak_rss = max(self.peak_rss, rss)
            self.peak_threads = max(self.peak_threads, threads)

    def __enter__(self) -> _ResourceSampler:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join()


def _cleanup_case(case_root: Path, owned_root: Path) -> float:
    resolved = case_root.resolve()
    fixture_root = (owned_root / "fixtures").resolve()
    if resolved.parent != fixture_root:
        raise ValueError("refusing cleanup outside the exact fixture root")
    start = time.perf_counter()
    if resolved.exists():
        _remove_owned_tree(resolved)
    return time.perf_counter() - start


def _measure_layout(
    fixture_root: Path,
    owned_root: Path,
    table: pa.Table,
    *,
    label: str,
    file_count: int,
    repetition: int,
) -> dict[str, Any]:
    case_root = fixture_root / f"{label}-{file_count}-r{repetition}"
    cleanup_seconds = _cleanup_case(case_root, owned_root)
    case_root.mkdir(parents=True)
    chunks = [
        table.slice(start, stop - start) for start, stop in _ranges(table.num_rows, file_count)
    ]
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    with _ResourceSampler() as resources:
        paths: list[Path] = []
        for index, chunk in enumerate(chunks):
            path = case_root / f"part-{index:05d}.parquet"
            write_parquet_atomic(chunk, path, codec="snappy")
            paths.append(path)
        write_wall = time.perf_counter() - wall_start
        write_cpu = time.process_time() - cpu_start

        footer_start = time.perf_counter()
        footer_cpu_start = time.process_time()
        footer_bytes = 0
        row_groups = 0
        for path in paths:
            metadata = pq.ParquetFile(path).metadata
            footer_bytes += int(metadata.serialized_size)
            row_groups += int(metadata.num_row_groups)
        footer_wall = time.perf_counter() - footer_start
        footer_cpu = time.process_time() - footer_cpu_start

        scan_start = time.perf_counter()
        scan_cpu_start = time.process_time()
        read_tables = [pq.read_table(path) for path in paths]
        restored = pa.concat_tables(read_tables).combine_chunks()
        scan_wall = time.perf_counter() - scan_start
        scan_cpu = time.process_time() - scan_cpu_start

        hash_start = time.perf_counter()
        hash_cpu_start = time.process_time()
        digests = [sha256_file(path) for path in paths]
        hash_wall = time.perf_counter() - hash_start
        hash_cpu = time.process_time() - hash_cpu_start
    if not restored.equals(table.combine_chunks()):
        raise ValueError(f"logical fixture mismatch for {label}/{file_count}")
    logical = hashlib.sha256()
    for batch in restored.to_batches(max_chunksize=65_536):
        logical.update(batch.serialize().to_pybytes())
    return {
        "label": label,
        "file_count": file_count,
        "repetition": repetition,
        "rows": table.num_rows,
        "columns": table.num_columns,
        "schema": str(table.schema),
        "file_bytes": sum(path.stat().st_size for path in paths),
        "footer_metadata_bytes": footer_bytes,
        "row_groups": row_groups,
        "write_wall_seconds": write_wall,
        "write_cpu_seconds": write_cpu,
        "footer_open_wall_seconds": footer_wall,
        "footer_open_cpu_seconds": footer_cpu,
        "scan_wall_seconds": scan_wall,
        "scan_cpu_seconds": scan_cpu,
        "hash_wall_seconds": hash_wall,
        "hash_cpu_seconds": hash_cpu,
        "hash_calls": len(paths),
        "hash_bytes": sum(path.stat().st_size for path in paths),
        "physical_digest": hashlib.sha256("".join(digests).encode()).hexdigest(),
        "logical_digest": logical.hexdigest(),
        "peak_process_tree_rss_bytes": resources.peak_rss,
        "peak_native_threads": resources.peak_threads,
        "cleanup_seconds_before_case": cleanup_seconds,
        "cleanup_failures": 0,
        "case_root": str(case_root),
    }


def _ranges(rows: int, count: int) -> Iterable[tuple[int, int]]:
    for index in range(count):
        start = rows * index // count
        stop = rows * (index + 1) // count
        if stop > start:
            yield start, stop


def _layout_benchmarks(
    root: Path,
    current_root: Path,
    *,
    rows: int,
    repetitions: int,
) -> list[dict[str, Any]]:
    fixture_root = root / "fixtures"
    fixture_root.mkdir(exist_ok=True)
    tables = {
        "k2_native": _fixture_table(rows, 2),
        "k2_padded12": _fixture_table(rows, 2, pad_to=12),
        "k12_native": _fixture_table(rows, 12),
    }
    # One unreported warm-up uses the production schema and writer.
    warm = fixture_root / "warmup"
    warm.mkdir(exist_ok=True)
    write_parquet_atomic(tables["k2_native"].slice(0, min(rows, 512)), warm / "warm.parquet")
    _cleanup_case(warm, root)
    results: list[dict[str, Any]] = []
    for label, table in tables.items():
        for repetition in range(1, repetitions + 1):
            order = (64, 4) if repetition % 2 else (4, 64)
            for file_count in order:
                results.append(
                    _measure_layout(
                        fixture_root,
                        root,
                        table,
                        label=label,
                        file_count=file_count,
                        repetition=repetition,
                    )
                )

    block_root = current_root / "seed_pair_analysis" / "04_h2h_execute" / "h2h_2p" / "blocks"
    block_paths = sorted(block_root.glob("pair_*_root_*_order_*.parquet"))
    if block_paths:
        block_tables = [pq.read_table(path) for path in block_paths]
        h2h_table = pa.concat_tables(block_tables).combine_chunks()
        bundled_files = (len(block_paths) + 15) // 16
        for repetition in range(1, repetitions + 1):
            order = (
                (len(block_paths), bundled_files)
                if repetition % 2
                else (bundled_files, len(block_paths))
            )
            for file_count in order:
                results.append(
                    _measure_layout(
                        fixture_root,
                        root,
                        h2h_table,
                        label="h2h_actual_blocks",
                        file_count=file_count,
                        repetition=repetition,
                    )
                )
    return results


def _authentication_benchmark(
    current_root: Path,
    *,
    repetitions: int,
) -> list[dict[str, Any]]:
    candidates = sorted(
        (
            path
            for path in current_root.rglob("*")
            if path.is_file()
            and not path.name.endswith(".sidecar.json")
            and path.with_name(f"{path.name}.sidecar.json").is_file()
        ),
        key=lambda path: (path.stat().st_size, str(path)),
    )
    results: list[dict[str, Any]] = []
    for count in (8, 32, 128):
        selected = candidates[: min(count, len(candidates))]
        if not selected:
            continue
        validate_artifact_sidecar(selected[0])  # warm-up
        for repetition in range(1, repetitions + 1):
            start = time.perf_counter()
            cpu_start = time.process_time()
            with _ResourceSampler() as resources:
                for path in selected:
                    validate_artifact_sidecar(path)
            results.append(
                {
                    "artifact_count": len(selected),
                    "artifact_bytes": sum(path.stat().st_size for path in selected),
                    "sidecar_bytes": sum(
                        path.with_name(f"{path.name}.sidecar.json").stat().st_size
                        for path in selected
                    ),
                    "repetition": repetition,
                    "wall_seconds": time.perf_counter() - start,
                    "cpu_seconds": time.process_time() - cpu_start,
                    "authentication_calls": len(selected),
                    "hash_calls": len(selected),
                    "hash_bytes": sum(path.stat().st_size for path in selected),
                    "minimum_file_bytes": min(path.stat().st_size for path in selected),
                    "maximum_file_bytes": max(path.stat().st_size for path in selected),
                    "peak_process_tree_rss_bytes": resources.peak_rss,
                    "peak_native_threads": resources.peak_threads,
                }
            )
    return results


def _summarize_measurements(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["label"]), int(row["file_count"]))].append(row)
    return {
        f"{label}:{count}": {
            "repetitions": len(group),
            "median_write_wall_seconds": statistics.median(
                float(item["write_wall_seconds"]) for item in group
            ),
            "median_footer_open_wall_seconds": statistics.median(
                float(item["footer_open_wall_seconds"]) for item in group
            ),
            "median_scan_wall_seconds": statistics.median(
                float(item["scan_wall_seconds"]) for item in group
            ),
            "median_hash_wall_seconds": statistics.median(
                float(item["hash_wall_seconds"]) for item in group
            ),
            "median_file_bytes": statistics.median(int(item["file_bytes"]) for item in group),
            "median_footer_metadata_bytes": statistics.median(
                int(item["footer_metadata_bytes"]) for item in group
            ),
            "logical_digests": sorted({str(item["logical_digest"]) for item in group}),
        }
        for (label, count), group in sorted(grouped.items())
    }


def run(
    output_root: Path = DEFAULT_ROOT,
    *,
    current_root: Path = DEFAULT_CURRENT_ROOT,
    historical_root: Path = DEFAULT_HISTORICAL_ROOT,
    rows: int = DEFAULT_ROWS,
    repetitions: int = DEFAULT_REPETITIONS,
    force: bool = False,
) -> dict[str, Any]:
    if rows < 1 or repetitions < 3:
        raise ValueError("rows must be positive and repetitions must be at least three")
    settings = _settings(current_root, historical_root, rows=rows, repetitions=repetitions)
    root = prepare_owned_root(output_root, settings, force=force)
    checkpoint = root / CHECKPOINT_NAME
    if checkpoint.is_file() and not force:
        return _read_object(checkpoint)
    current_inventory = inventory_tree(current_root, detailed_parquets=True)
    historical_inventory = inventory_tree(historical_root, detailed_parquets=True)
    layout = _layout_benchmarks(
        root,
        current_root,
        rows=rows,
        repetitions=repetitions,
    )
    authentication = _authentication_benchmark(current_root, repetitions=repetitions)
    payload: dict[str, Any] = {
        "task": "7A",
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "bounded": True,
        "production_run_performed": False,
        "production_modules_changed": False,
        "settings": settings,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "pyarrow": pa.__version__,
            "filesystem_root": Path.cwd().anchor,
            "host_available_memory_bytes": psutil.virtual_memory().available,
        },
        "inventories": {
            "post_task6_seed_56_57": current_inventory,
            "historical_seed_54_55": historical_inventory,
        },
        "layout_measurements": layout,
        "layout_summary": _summarize_measurements(layout),
        "authentication_measurements": authentication,
        "retention": {
            "owned_root": str(root),
            "retained": [MARKER_NAME, CHECKPOINT_NAME, "fixtures"],
            "historical_roots_modified": False,
        },
    }
    _atomic_json(checkpoint, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--current-root", type=Path, default=DEFAULT_CURRENT_ROOT)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run(
        args.output_root,
        current_root=args.current_root,
        historical_root=args.historical_root,
        rows=args.rows,
        repetitions=args.repetitions,
        force=args.force,
    )
    if args.output is not None:
        _atomic_json(args.output, payload)
    print(json.dumps(payload["layout_summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
