"""Bounded Task 3A synchronized-tree versus local-storage benchmark.

The benchmark is intentionally noncanonical and diagnostic.  It exercises
production-shaped RNG route/reduce/merge publication and H2H checkpoint
publication with deterministic fixtures, the repository's atomic writer,
authenticated sidecars, bounded process scheduling, and Task 1 telemetry.

Only explicitly supplied, absent directories whose names begin with
``farkle-task3a-`` may be created or removed.  Historical result roots are
therefore outside the executable cleanup surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np

import farkle.utils.artifact_contract as artifact_contract
from farkle.utils.artifact_contract import (
    ARTIFACT_CONTRACT_VERSION,
    ArtifactSidecar,
    publish_staged_artifact_with_sidecar,
    sidecar_path,
)
from farkle.utils.authenticated_contract import canonical_json_bytes, identity_sha256
from farkle.utils.parallel import (
    StageParallelPolicy,
    apply_native_thread_limits,
    process_map,
)
from farkle.utils.random import RNG_SCHEME_VERSION
from farkle.utils.telemetry import sample_process_resource_state
from farkle.utils.writer import atomic_path

BENCHMARK_VERSION: Final = 1
SAFE_DIRECTORY_PREFIX: Final = "farkle-task3a-"
MARKER_NAME: Final = ".farkle-task3a-owned.json"
MATERIAL_WALL_RATIO: Final = 1.10
_RECORD_DTYPE: Final = np.dtype(
    [
        ("group_id", "<u8"),
        ("coordinate", "<u8"),
        ("value", "<f8"),
        ("count", "<u8"),
    ]
)


@dataclass(frozen=True, slots=True)
class WorkloadSettings:
    """Exact bounded workload and execution policy."""

    seed: int = 30_048_049
    workers: int = 2
    native_threads_per_process: int = 1
    rng_route_units: int = 32
    rng_records_per_unit: int = 4_096
    rng_partitions: int = 8
    rng_merge_fan_in: int = 4
    h2h_blocks: int = 16
    h2h_chunks_per_block: int = 4
    h2h_games_per_chunk: int = 4_000
    h2h_payload_records_per_chunk: int = 512
    h2h_execution_state_cadence_rounds: int = 1

    @property
    def rng_record_count(self) -> int:
        return self.rng_route_units * self.rng_records_per_unit


@dataclass(slots=True)
class OperationMetrics:
    """Additive operation counters and timings for one workload."""

    source_bytes_read: int = 0
    route_bytes_read: int = 0
    temporary_bytes_written: int = 0
    durable_bytes_written: int = 0
    file_creates: int = 0
    file_opens: int = 0
    file_closes: int = 0
    measured_open_seconds: float = 0.0
    measured_close_seconds: float = 0.0
    spill_runs_created: int = 0
    spill_bytes_written: int = 0
    merge_passes: int = 0
    merge_runs_created: int = 0
    merge_input_bytes: int = 0
    merge_output_bytes: int = 0
    merge_seconds: float = 0.0
    hash_calls: int = 0
    hash_bytes: int = 0
    hash_seconds: float = 0.0
    sidecar_publications: int = 0
    sidecar_bytes_written: int = 0
    sidecar_publication_seconds: float = 0.0
    authentication_calls: int = 0
    checkpoint_writes: int = 0
    checkpoint_bytes_written: int = 0
    checkpoint_rewrite_seconds: float = 0.0
    checkpoint_queue_seconds: float = 0.0
    cleanup_seconds: float = 0.0
    cleanup_failures: int = 0
    retry_events: int = 0
    downshift_events: int = 0
    memory_pause_events: int = 0
    worker_failure_events: int = 0
    scheduler_events: int = 0
    worker_cpu_seconds: float = 0.0

    def add(self, other: OperationMetrics) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass(frozen=True, slots=True)
class _RouteTask:
    source: str
    staged: str
    unit: int
    records_per_unit: int
    submitted_at: float


@dataclass(frozen=True, slots=True)
class _ReduceTask:
    route_paths: tuple[str, ...]
    work_dir: str
    staged: str
    partition: int
    partition_count: int
    merge_fan_in: int
    submitted_at: float


@dataclass(frozen=True, slots=True)
class _H2HTask:
    source: str
    block: int
    chunk: int
    games: int
    payload_records: int
    seed: int
    submitted_at: float


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    coordinate: int
    secondary_coordinate: int
    staged: str
    payload: bytes
    metrics: OperationMetrics
    queue_seconds: float


@dataclass(slots=True)
class _ResourceTracker:
    samples: int = 0
    peak_process_tree_rss_bytes: int = 0
    peak_aggregate_memory_bytes: int = 0
    windows_job_committed_memory_peak_bytes: int | None = None
    aggregate_memory_hard_limit_bytes: int = 0
    aggregate_memory_source: str | None = None
    minimum_host_available_memory_bytes: int | None = None
    peak_native_threads: int = 0
    monitoring_errors: list[str] = field(default_factory=list)

    def sample(self) -> None:
        state = sample_process_resource_state()
        self.samples += 1
        self.peak_process_tree_rss_bytes = max(
            self.peak_process_tree_rss_bytes,
            _as_int(state.get("process_tree_rss_bytes")),
        )
        self.peak_aggregate_memory_bytes = max(
            self.peak_aggregate_memory_bytes,
            _as_int(state.get("peak_aggregate_memory_bytes")),
        )
        job_peak = state.get("windows_job_committed_memory_peak_bytes")
        if isinstance(job_peak, int):
            self.windows_job_committed_memory_peak_bytes = max(
                self.windows_job_committed_memory_peak_bytes or 0,
                job_peak,
            )
        self.aggregate_memory_hard_limit_bytes = max(
            self.aggregate_memory_hard_limit_bytes,
            _as_int(state.get("aggregate_memory_hard_limit_bytes")),
        )
        source = state.get("aggregate_memory_source")
        if isinstance(source, str):
            self.aggregate_memory_source = source
        available = _as_int(state.get("host_available_memory_bytes"))
        if available:
            self.minimum_host_available_memory_bytes = (
                available
                if self.minimum_host_available_memory_bytes is None
                else min(self.minimum_host_available_memory_bytes, available)
            )
        self.peak_native_threads = max(
            self.peak_native_threads,
            _as_int(state.get("native_threads")),
        )
        error = state.get("monitoring_error")
        if error:
            self.monitoring_errors.append(str(error))


class _HashProbe:
    """Measure the actual hashes issued by the artifact helper in this process."""

    def __init__(self, metrics: OperationMetrics) -> None:
        self.metrics = metrics
        self._original = artifact_contract.sha256_file

    def __enter__(self) -> _HashProbe:
        original = self._original
        metrics = self.metrics

        def measured(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
            source = Path(path)
            size = source.stat().st_size
            started = time.perf_counter()
            digest = original(source, chunk_size=chunk_size)
            metrics.hash_calls += 1
            metrics.hash_bytes += size
            metrics.hash_seconds += time.perf_counter() - started
            metrics.file_opens += 1
            metrics.file_closes += 1
            return digest

        artifact_contract.sha256_file = measured
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        artifact_contract.sha256_file = self._original


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return 0


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"expected numeric benchmark value, found {type(value).__name__}")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _onedrive_roots() -> tuple[Path, ...]:
    roots: set[Path] = set()
    for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        value = os.environ.get(name)
        if value:
            roots.add(Path(value).resolve())
    return tuple(sorted(roots, key=str))


def validate_disposable_roots(
    onedrive_root: Path,
    local_root: Path,
    *,
    repository_root: Path,
) -> tuple[Path, Path]:
    """Resolve and reject unsafe roots before anything is created or deleted."""

    resolved: list[Path] = []
    for label, candidate in (("onedrive", onedrive_root), ("local", local_root)):
        if not candidate.is_absolute():
            raise ValueError(f"{label} benchmark root must be absolute: {candidate}")
        target = candidate.resolve(strict=False)
        if not target.name.lower().startswith(SAFE_DIRECTORY_PREFIX):
            raise ValueError(
                f"{label} benchmark directory name must start with {SAFE_DIRECTORY_PREFIX!r}"
            )
        if target.exists():
            raise FileExistsError(f"{label} benchmark root must not already exist: {target}")
        if not target.parent.is_dir():
            raise FileNotFoundError(f"{label} benchmark parent does not exist: {target.parent}")
        if target.parent.resolve(strict=True) != target.parent:
            raise ValueError(f"{label} benchmark parent does not resolve stably: {target.parent}")
        resolved.append(target)

    sync_target, local_target = resolved
    if os.path.normcase(str(sync_target)) == os.path.normcase(str(local_target)):
        raise ValueError("benchmark roots must be distinct")
    repository = repository_root.resolve(strict=True)
    if not _is_within(sync_target, repository):
        raise ValueError("OneDrive benchmark root must remain inside the repository tree")
    providers = _onedrive_roots()
    if not providers or not any(_is_within(sync_target, root) for root in providers):
        raise ValueError("OneDrive benchmark root is not beneath a configured OneDrive root")
    if any(_is_within(local_target, root) for root in providers):
        raise ValueError("local benchmark root must remain outside configured OneDrive roots")
    if _is_within(local_target, repository):
        raise ValueError("local benchmark root must remain outside the repository tree")
    return sync_target, local_target


def _owned_marker(root: Path, token: str) -> dict[str, object]:
    return {
        "benchmark": "farkle-task3a-storage",
        "benchmark_version": BENCHMARK_VERSION,
        "resolved_root": str(root),
        "ownership_token": token,
    }


def _create_owned_directory(root: Path, token: str) -> None:
    root.mkdir(parents=False, exist_ok=False)
    marker = _owned_marker(root, token)
    (root / MARKER_NAME).write_bytes(canonical_json_bytes(marker) + b"\n")


def _validate_owned_directory(root: Path, token: str) -> None:
    if root.resolve(strict=True) != root:
        raise ValueError(f"benchmark root no longer resolves stably: {root}")
    marker_path = root / MARKER_NAME
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    if payload != _owned_marker(root, token):
        raise ValueError(f"benchmark ownership marker is invalid: {marker_path}")


def _remove_owned_directory(root: Path, token: str) -> tuple[float, int]:
    started = time.perf_counter()
    failures = 0
    try:
        _validate_owned_directory(root, token)
        shutil.rmtree(root)
    except Exception:
        failures = 1
        raise
    return time.perf_counter() - started, failures


def _open_read(path: Path, metrics: OperationMetrics) -> bytes:
    started = time.perf_counter()
    handle = path.open("rb")
    metrics.measured_open_seconds += time.perf_counter() - started
    metrics.file_opens += 1
    try:
        return handle.read()
    finally:
        started = time.perf_counter()
        handle.close()
        metrics.measured_close_seconds += time.perf_counter() - started
        metrics.file_closes += 1


def _open_write(path: Path, payload: bytes, metrics: OperationMetrics, *, temporary: bool) -> None:
    started = time.perf_counter()
    handle = path.open("xb")
    metrics.measured_open_seconds += time.perf_counter() - started
    metrics.file_creates += 1
    metrics.file_opens += 1
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        started = time.perf_counter()
        handle.close()
        metrics.measured_close_seconds += time.perf_counter() - started
        metrics.file_closes += 1
    if temporary:
        metrics.temporary_bytes_written += len(payload)
    else:
        metrics.durable_bytes_written += len(payload)


def _write_atomic(path: Path, payload: bytes, metrics: OperationMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as temporary:
        # atomic_path's mkstemp create/close is counted; its latency is internal.
        metrics.file_creates += 1
        metrics.file_opens += 1
        metrics.file_closes += 1
        temporary_path = Path(temporary)
        temporary_path.unlink()
        _open_write(temporary_path, payload, metrics, temporary=False)


def _stage_bytes(path: Path, payload: bytes, metrics: OperationMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _open_write(path, payload, metrics, temporary=True)


def _sidecar_template(path: Path, operation: str, *, kind: str = "operation") -> ArtifactSidecar:
    config_hash = identity_sha256(
        {"benchmark": "task3a_storage", "version": BENCHMARK_VERSION, "operation": operation}
    )
    return ArtifactSidecar(
        artifact_contract_version=ARTIFACT_CONTRACT_VERSION,
        estimand_version=1,
        schema_version=1,
        artifact_name=path.name,
        producer="task3a_storage_benchmark",
        scope="diagnostics",
        source_scope="diagnostics",
        operation=operation,
        method_contract={"kind": kind, "procedure": operation},  # type: ignore[typeddict-item]
        baseline="none",
        weighted_quantity="none",
        k_aggregation_method="none",
        k_weights=None,
        support_count_role="bounded_fixture_records",
        uncertainty_method="none",
        replication_unit="deterministic_benchmark_coordinate",
        conditioning="unconditional",
        consistency_columns=[],
        source_artifacts=[],
        grouping_keys=[],
        player_counts=[],
        required_player_counts=[],
        missing_cell_policy="not_applicable",
        seed_scope="not_applicable",
        rng_scheme_version=RNG_SCHEME_VERSION,
        config_hash=config_hash,
        input_manifest_hashes=[],
        code_revision=f"task3a-benchmark-v{BENCHMARK_VERSION}",
    )


def _publish_staged(
    staged: Path,
    final: Path,
    operation: str,
    metrics: OperationMetrics,
    *,
    kind: str = "operation",
    checkpoint: bool = False,
) -> None:
    artifact_bytes = staged.stat().st_size
    started = time.perf_counter()
    publish_staged_artifact_with_sidecar(
        staged, final, _sidecar_template(final, operation, kind=kind)
    )
    elapsed = time.perf_counter() - started
    published_sidecar = sidecar_path(final)
    sidecar_bytes = published_sidecar.stat().st_size
    metrics.sidecar_publications += 1
    metrics.sidecar_bytes_written += sidecar_bytes
    metrics.sidecar_publication_seconds += elapsed
    metrics.authentication_calls += 1
    metrics.durable_bytes_written += artifact_bytes + sidecar_bytes
    # Known v2 helper operations not visible through the benchmark's explicit
    # open wrappers: sidecar temp write plus two sidecar reads.
    metrics.file_creates += 1
    metrics.file_opens += 3
    metrics.file_closes += 3
    if checkpoint:
        metrics.checkpoint_writes += 1
        metrics.checkpoint_bytes_written += artifact_bytes + sidecar_bytes
        metrics.checkpoint_rewrite_seconds += elapsed


def _worker_policy(settings: WorkloadSettings) -> StageParallelPolicy:
    cores = os.cpu_count() or 1
    return StageParallelPolicy(
        total_cores=cores,
        process_workers=settings.workers,
        python_threads=1,
        arrow_threads=1,
        native_threads_per_process=settings.native_threads_per_process,
        configured_cpu_budget=settings.workers,
        cpu_worker_cap=settings.workers,
        memory_worker_cap=settings.workers,
    )


def _initialize_worker(policy: StageParallelPolicy) -> None:
    apply_native_thread_limits(policy)


def _stable_partition(values: np.ndarray, partitions: int) -> np.ndarray:
    mixed = values.astype(np.uint64, copy=True)
    mixed += np.uint64(0x9E3779B97F4A7C15)
    mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    return mixed % np.uint64(partitions)


def _route_worker(task: _RouteTask) -> _WorkerResult:
    metrics = OperationMetrics()
    cpu_started = time.process_time()
    queue_seconds = max(0.0, time.perf_counter() - task.submitted_at)
    source = Path(task.source)
    started = time.perf_counter()
    handle = source.open("rb")
    metrics.measured_open_seconds += time.perf_counter() - started
    metrics.file_opens += 1
    try:
        offset = task.unit * task.records_per_unit * _RECORD_DTYPE.itemsize
        handle.seek(offset)
        payload = handle.read(task.records_per_unit * _RECORD_DTYPE.itemsize)
    finally:
        started = time.perf_counter()
        handle.close()
        metrics.measured_close_seconds += time.perf_counter() - started
        metrics.file_closes += 1
    metrics.source_bytes_read += len(payload)
    records = np.frombuffer(payload, dtype=_RECORD_DTYPE).copy()
    order = np.lexsort((records["coordinate"], records["group_id"]))
    routed = records[order]
    _open_write(Path(task.staged), routed.tobytes(), metrics, temporary=True)
    metrics.worker_cpu_seconds = time.process_time() - cpu_started
    return _WorkerResult(task.unit, 0, task.staged, b"", metrics, queue_seconds)


def _reduce_worker(task: _ReduceTask) -> _WorkerResult:
    metrics = OperationMetrics()
    cpu_started = time.process_time()
    queue_seconds = max(0.0, time.perf_counter() - task.submitted_at)
    work = Path(task.work_dir)
    work.mkdir(parents=True, exist_ok=False)
    runs: list[Path] = []
    for index, text_path in enumerate(task.route_paths):
        payload = _open_read(Path(text_path), metrics)
        metrics.route_bytes_read += len(payload)
        records = np.frombuffer(payload, dtype=_RECORD_DTYPE)
        selected = records[
            _stable_partition(records["group_id"], task.partition_count) == task.partition
        ]
        order = np.lexsort((selected["coordinate"], selected["group_id"]))
        spill = selected[order].tobytes()
        spill_path = work / f"spill-{index:04d}.bin"
        _open_write(spill_path, spill, metrics, temporary=True)
        metrics.spill_runs_created += 1
        metrics.spill_bytes_written += len(spill)
        runs.append(spill_path)

    generation = 0
    while len(runs) > 1:
        pass_started = time.perf_counter()
        next_runs: list[Path] = []
        for start in range(0, len(runs), task.merge_fan_in):
            inputs = runs[start : start + task.merge_fan_in]
            arrays: list[np.ndarray] = []
            for input_path in inputs:
                payload = _open_read(input_path, metrics)
                metrics.merge_input_bytes += len(payload)
                arrays.append(np.frombuffer(payload, dtype=_RECORD_DTYPE).copy())
            merged = np.concatenate(arrays) if arrays else np.empty(0, dtype=_RECORD_DTYPE)
            if merged.size:
                order = np.lexsort((merged["coordinate"], merged["group_id"]))
                merged = merged[order]
            output = work / f"merge-{generation:02d}-{len(next_runs):04d}.bin"
            output_payload = merged.tobytes()
            _open_write(output, output_payload, metrics, temporary=True)
            metrics.merge_runs_created += 1
            metrics.merge_output_bytes += len(output_payload)
            next_runs.append(output)
        metrics.merge_passes += 1
        metrics.merge_seconds += time.perf_counter() - pass_started
        for old in runs:
            started = time.perf_counter()
            old.unlink()
            metrics.cleanup_seconds += time.perf_counter() - started
        runs = next_runs
        generation += 1
    if len(runs) != 1:
        raise RuntimeError("bounded RNG reducer did not produce exactly one final run")
    os.replace(runs[0], Path(task.staged))
    started = time.perf_counter()
    work.rmdir()
    metrics.cleanup_seconds += time.perf_counter() - started
    metrics.worker_cpu_seconds = time.process_time() - cpu_started
    return _WorkerResult(task.partition, 0, task.staged, b"", metrics, queue_seconds)


def _h2h_worker(task: _H2HTask) -> _WorkerResult:
    cpu_started = time.process_time()
    queue_seconds = max(0.0, time.perf_counter() - task.submitted_at)
    metrics = OperationMetrics()
    source_payload = _open_read(Path(task.source), metrics)
    metrics.source_bytes_read += len(source_payload)
    source_seed = int.from_bytes(hashlib.sha256(source_payload).digest()[:8], "little")
    state = (task.seed ^ source_seed ^ ((task.block + 1) * 0xD1B54A32D192ED03)) & 0xFFFFFFFFFFFFFFFF
    digest = hashlib.sha256()
    wins_a = 0
    completed = 0
    for game in range(task.games):
        coordinate = task.chunk * task.games + game
        state = (state + 0x9E3779B97F4A7C15 + coordinate) & 0xFFFFFFFFFFFFFFFF
        state = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        state = ((state ^ (state >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 31
        wins_a += state & 1
        completed += int((state & 0xFF) != 0)
        digest.update(state.to_bytes(8, "little"))
    generator = np.random.Generator(
        np.random.PCG64DXSM(task.seed + task.block * 10_000 + task.chunk)
    )
    payload = generator.integers(
        0,
        256,
        size=task.payload_records * 32,
        dtype=np.uint8,
    ).tobytes()
    header = canonical_json_bytes(
        {
            "block": task.block,
            "chunk": task.chunk,
            "completed": completed,
            "games": task.games,
            "simulation_sha256": digest.hexdigest(),
            "wins_a": wins_a,
        }
    )
    metrics.worker_cpu_seconds = time.process_time() - cpu_started
    return _WorkerResult(
        task.block, task.chunk, "", header + b"\n" + payload, metrics, queue_seconds
    )


def _fixture_bytes(settings: WorkloadSettings, *, salt: int) -> bytes:
    generator = np.random.Generator(np.random.PCG64DXSM(settings.seed + salt))
    records = np.empty(settings.rng_record_count, dtype=_RECORD_DTYPE)
    records["group_id"] = generator.integers(0, 2**48, size=records.size, dtype=np.uint64)
    records["coordinate"] = np.arange(records.size, dtype=np.uint64)
    records["value"] = generator.standard_normal(records.size)
    records["count"] = generator.integers(1, 8, size=records.size, dtype=np.uint64)
    return records.tobytes()


def _scheduler_counter(metrics: OperationMetrics, tracker: _ResourceTracker):
    def progress(event: Mapping[str, object]) -> None:
        metrics.scheduler_events += 1
        name = str(event.get("event", ""))
        if name in {"schedule_check"}:
            metrics.memory_pause_events += 0
        elif name == "worker_exception":
            metrics.worker_failure_events += 1
        tracker.sample()

    return progress


def _digest_group(paths: Iterable[Path]) -> str:
    entries = []
    for path in sorted(paths, key=lambda item: item.as_posix()):
        entries.append({"name": path.name, "sha256": artifact_contract.sha256_file(path)})
    return identity_sha256(entries)


def _finish_measurement(
    *,
    workload: str,
    metrics: OperationMetrics,
    tracker: _ResourceTracker,
    wall_started: float,
    cpu_started: float,
    correctness: Mapping[str, str],
    settings: WorkloadSettings,
) -> dict[str, object]:
    wall = time.perf_counter() - wall_started
    parent_cpu = time.process_time() - cpu_started
    total_cpu = parent_cpu + metrics.worker_cpu_seconds
    return {
        "workload": workload,
        "wall_seconds": wall,
        "parent_cpu_seconds": parent_cpu,
        "worker_cpu_seconds": metrics.worker_cpu_seconds,
        "total_cpu_seconds": total_cpu,
        "cpu_to_wall_ratio": total_cpu / wall if wall else None,
        "requested_workers": settings.workers,
        "effective_workers": settings.workers,
        "native_threads_per_process": settings.native_threads_per_process,
        "metrics": asdict(metrics),
        "resources": asdict(tracker),
        "correctness": dict(correctness),
        "merge_throughput_bytes_per_second": (
            metrics.merge_input_bytes / metrics.merge_seconds if metrics.merge_seconds else None
        ),
        "checkpoint_throughput_bytes_per_second": (
            metrics.checkpoint_bytes_written / metrics.checkpoint_rewrite_seconds
            if metrics.checkpoint_rewrite_seconds
            else None
        ),
    }


def run_rng_workload(run_dir: Path, settings: WorkloadSettings, token: str) -> dict[str, object]:
    """Run one bounded RNG route/reduce/publication workload."""

    del token  # ownership is validated at the enclosing run-directory boundary
    metrics = OperationMetrics()
    tracker = _ResourceTracker()
    fixture = _fixture_bytes(settings, salt=1)
    source = run_dir / "source-rng.bin"
    _write_atomic(source, fixture, metrics)
    source_sha = hashlib.sha256(fixture).hexdigest()
    metrics = OperationMetrics()  # fixture setup is intentionally outside the measured workload
    policy = _worker_policy(settings)
    apply_native_thread_limits(policy)
    context = mp.get_context("spawn")
    route_root = run_dir / "rng" / "route"
    result_root = run_dir / "rng" / "result"
    temp_root = run_dir / "rng" / "temporary"
    route_root.mkdir(parents=True)
    result_root.mkdir(parents=True)
    temp_root.mkdir(parents=True)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    tracker.sample()
    with _HashProbe(metrics):
        route_tasks = (
            _RouteTask(
                str(source),
                str(route_root / f"route-{unit:04d}.staged"),
                unit,
                settings.rng_records_per_unit,
                time.perf_counter(),
            )
            for unit in range(settings.rng_route_units)
        )
        routed = list(
            process_map(
                _route_worker,
                route_tasks,
                n_jobs=settings.workers,
                window=settings.workers * 2,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(policy,),
                progress_callback=_scheduler_counter(metrics, tracker),
            )
        )
        route_paths: list[Path] = []
        for result in sorted(routed, key=lambda item: item.coordinate):
            metrics.add(result.metrics)
            metrics.checkpoint_queue_seconds += result.queue_seconds
            staged = Path(result.staged)
            final = route_root / f"route-{result.coordinate:04d}.bin"
            _publish_staged(staged, final, "task3a_rng_route", metrics)
            route_paths.append(final)

        reduce_tasks = (
            _ReduceTask(
                tuple(str(path) for path in route_paths),
                str(temp_root / f"partition-{partition:02d}"),
                str(result_root / f"partition-{partition:02d}.staged"),
                partition,
                settings.rng_partitions,
                settings.rng_merge_fan_in,
                time.perf_counter(),
            )
            for partition in range(settings.rng_partitions)
        )
        reduced = list(
            process_map(
                _reduce_worker,
                reduce_tasks,
                n_jobs=settings.workers,
                window=settings.workers * 2,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(policy,),
                progress_callback=_scheduler_counter(metrics, tracker),
            )
        )
        result_paths: list[Path] = []
        manifest_entries: list[dict[str, object]] = []
        for result in sorted(reduced, key=lambda item: item.coordinate):
            metrics.add(result.metrics)
            metrics.checkpoint_queue_seconds += result.queue_seconds
            staged = Path(result.staged)
            final = result_root / f"partition-{result.coordinate:02d}.bin"
            _publish_staged(staged, final, "task3a_rng_partition", metrics)
            result_paths.append(final)
            manifest_entries.append(
                {
                    "partition": result.coordinate,
                    "relative_path": f"result/{final.name}",
                    "records": final.stat().st_size // _RECORD_DTYPE.itemsize,
                    "sha256": artifact_contract.sha256_file(final),
                    "size_bytes": final.stat().st_size,
                }
            )

        manifest = run_dir / "rng" / "rng-manifest.json"
        staged_manifest = manifest.with_name("rng-manifest.staged")
        _stage_bytes(staged_manifest, canonical_json_bytes(manifest_entries) + b"\n", metrics)
        _publish_staged(staged_manifest, manifest, "task3a_rng_manifest", metrics)
        checkpoint = run_dir / "rng" / "rng-checkpoint.json"
        checkpoint_payload = (
            canonical_json_bytes(
                {
                    "route_units": settings.rng_route_units,
                    "partitions": settings.rng_partitions,
                    "source_sha256": source_sha,
                    "status": "complete",
                }
            )
            + b"\n"
        )
        checkpoint_started = time.perf_counter()
        _write_atomic(checkpoint, checkpoint_payload, metrics)
        metrics.checkpoint_writes += 1
        metrics.checkpoint_bytes_written += len(checkpoint_payload)
        metrics.checkpoint_rewrite_seconds += time.perf_counter() - checkpoint_started
        completion = run_dir / "rng" / "rng-completion.json"
        _write_atomic(
            completion,
            canonical_json_bytes(
                {
                    "manifest_sha256": artifact_contract.sha256_file(manifest),
                    "partitions": settings.rng_partitions,
                    "status": "complete",
                }
            )
            + b"\n",
            metrics,
        )
        sidecars = [sidecar_path(path) for path in [*route_paths, *result_paths, manifest]]
        correctness = {
            "canonical_outputs_sha256": _digest_group(result_paths),
            "manifest_sha256": artifact_contract.sha256_file(manifest),
            "sidecars_sha256": _digest_group(sidecars),
            "completion_sha256": artifact_contract.sha256_file(completion),
            "checkpoint_state_sha256": artifact_contract.sha256_file(checkpoint),
            "source_fixture_sha256": source_sha,
        }
        tracker.sample()
    return _finish_measurement(
        workload="rng",
        metrics=metrics,
        tracker=tracker,
        wall_started=wall_started,
        cpu_started=cpu_started,
        correctness=correctness,
        settings=settings,
    )


def run_h2h_workload(run_dir: Path, settings: WorkloadSettings, token: str) -> dict[str, object]:
    """Run one bounded H2H simulation/checkpoint/publication workload."""

    del token  # ownership is validated at the enclosing run-directory boundary
    metrics = OperationMetrics()
    tracker = _ResourceTracker()
    fixture = (
        canonical_json_bytes(
            {
                "blocks": settings.h2h_blocks,
                "chunks": settings.h2h_chunks_per_block,
                "games_per_chunk": settings.h2h_games_per_chunk,
                "seed": settings.seed,
            }
        )
        + b"\n"
    )
    source = run_dir / "source-h2h.json"
    _write_atomic(source, fixture, metrics)
    source_sha = hashlib.sha256(fixture).hexdigest()
    metrics = OperationMetrics()
    policy = _worker_policy(settings)
    apply_native_thread_limits(policy)
    context = mp.get_context("spawn")
    root = run_dir / "h2h"
    blocks_root = root / "blocks"
    blocks_root.mkdir(parents=True)
    block_payloads: dict[int, list[bytes]] = {block: [] for block in range(settings.h2h_blocks)}
    block_paths = [blocks_root / f"block-{block:04d}.bin" for block in range(settings.h2h_blocks)]
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    tracker.sample()
    with _HashProbe(metrics):
        for chunk in range(settings.h2h_chunks_per_block):
            tasks = (
                _H2HTask(
                    str(source),
                    block,
                    chunk,
                    settings.h2h_games_per_chunk,
                    settings.h2h_payload_records_per_chunk,
                    settings.seed,
                    time.perf_counter(),
                )
                for block in range(settings.h2h_blocks)
            )
            results = list(
                process_map(
                    _h2h_worker,
                    tasks,
                    n_jobs=settings.workers,
                    window=settings.workers * 2,
                    mp_context=context,
                    initializer=_initialize_worker,
                    initargs=(policy,),
                    progress_callback=_scheduler_counter(metrics, tracker),
                )
            )
            for result in sorted(results, key=lambda item: item.coordinate):
                metrics.add(result.metrics)
                metrics.checkpoint_queue_seconds += result.queue_seconds
                block_payloads[result.coordinate].append(result.payload)
                final = block_paths[result.coordinate]
                staged = final.with_name(f"{final.name}.staged")
                payload = b"".join(block_payloads[result.coordinate])
                _stage_bytes(staged, payload, metrics)
                _publish_staged(
                    staged,
                    final,
                    "task3a_h2h_block_checkpoint",
                    metrics,
                    kind="h2h",
                    checkpoint=True,
                )
            if (chunk + 1) % settings.h2h_execution_state_cadence_rounds == 0:
                state = root / "h2h-execution-state.json"
                state_payload = (
                    canonical_json_bytes(
                        {
                            "completed_chunks_per_block": chunk + 1,
                            "blocks": settings.h2h_blocks,
                            "source_sha256": source_sha,
                            "status": (
                                "complete"
                                if chunk + 1 == settings.h2h_chunks_per_block
                                else "partial_resumable"
                            ),
                        }
                    )
                    + b"\n"
                )
                checkpoint_started = time.perf_counter()
                _write_atomic(state, state_payload, metrics)
                metrics.checkpoint_writes += 1
                metrics.checkpoint_bytes_written += len(state_payload)
                metrics.checkpoint_rewrite_seconds += time.perf_counter() - checkpoint_started

        manifest_entries = [
            {
                "block": block,
                "relative_path": f"blocks/{path.name}",
                "sha256": artifact_contract.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for block, path in enumerate(block_paths)
        ]
        manifest = root / "h2h-manifest.json"
        staged_manifest = root / "h2h-manifest.staged"
        _stage_bytes(staged_manifest, canonical_json_bytes(manifest_entries) + b"\n", metrics)
        _publish_staged(staged_manifest, manifest, "task3a_h2h_manifest", metrics, kind="h2h")
        aggregate = root / "h2h-aggregate.json"
        staged_aggregate = root / "h2h-aggregate.staged"
        _stage_bytes(
            staged_aggregate,
            canonical_json_bytes(
                {
                    "blocks": settings.h2h_blocks,
                    "completed_games": settings.h2h_blocks
                    * settings.h2h_chunks_per_block
                    * settings.h2h_games_per_chunk,
                    "manifest_sha256": artifact_contract.sha256_file(manifest),
                }
            )
            + b"\n",
            metrics,
        )
        _publish_staged(staged_aggregate, aggregate, "task3a_h2h_aggregate", metrics, kind="h2h")
        completion = root / "h2h-completion.json"
        _write_atomic(
            completion,
            canonical_json_bytes(
                {
                    "aggregate_sha256": artifact_contract.sha256_file(aggregate),
                    "manifest_sha256": artifact_contract.sha256_file(manifest),
                    "status": "complete",
                }
            )
            + b"\n",
            metrics,
        )
        state = root / "h2h-execution-state.json"
        sidecars = [sidecar_path(path) for path in [*block_paths, manifest, aggregate]]
        correctness = {
            "canonical_outputs_sha256": artifact_contract.sha256_file(aggregate),
            "manifest_sha256": artifact_contract.sha256_file(manifest),
            "sidecars_sha256": _digest_group(sidecars),
            "completion_sha256": artifact_contract.sha256_file(completion),
            "checkpoint_state_sha256": artifact_contract.sha256_file(state),
            "source_fixture_sha256": source_sha,
            "block_checkpoints_sha256": _digest_group(block_paths),
        }
        tracker.sample()
    return _finish_measurement(
        workload="h2h",
        metrics=metrics,
        tracker=tracker,
        wall_started=wall_started,
        cpu_started=cpu_started,
        correctness=correctness,
        settings=settings,
    )


def _run_location_once(
    root: Path,
    *,
    location: str,
    repetition: int,
    position: int,
    warmup: bool,
    settings: WorkloadSettings,
    root_token: str,
) -> dict[str, object]:
    run_name = f"{'warmup' if warmup else 'measured'}-{repetition:02d}-{position}-{location}"
    run_dir = root / f"{SAFE_DIRECTORY_PREFIX}{run_name}"
    run_token = identity_sha256({"root_token": root_token, "run": run_name})
    _create_owned_directory(run_dir, run_token)
    rng = run_rng_workload(run_dir, settings, run_token)
    h2h = run_h2h_workload(run_dir, settings, run_token)
    cleanup_seconds, cleanup_failures = _remove_owned_directory(run_dir, run_token)
    each_cleanup = cleanup_seconds / 2.0
    for measurement in (rng, h2h):
        metrics = measurement["metrics"]
        assert isinstance(metrics, dict)
        metrics["cleanup_seconds"] = float(metrics["cleanup_seconds"]) + each_cleanup
        metrics["cleanup_failures"] = int(metrics["cleanup_failures"]) + cleanup_failures
    return {
        "location": location,
        "repetition": repetition,
        "position": position,
        "warmup": warmup,
        "rng": rng,
        "h2h": h2h,
        "run_cleanup_seconds": cleanup_seconds,
        "run_cleanup_failures": cleanup_failures,
    }


def _path_attributes(path: Path) -> dict[str, object]:
    stat = path.stat()
    raw = int(getattr(stat, "st_file_attributes", 0))
    flags = {
        "archive": 0x20,
        "reparse_point": 0x400,
        "pinned": 0x80000,
        "unpinned": 0x100000,
        "recall_on_open": 0x40000,
        "recall_on_data_access": 0x400000,
    }
    return {"raw": raw, "flags": [name for name, bit in flags.items() if raw & bit]}


def _windows_volume_device(path: Path) -> dict[str, object]:
    if os.name != "nt":
        return {"available": False, "reason": "not_windows"}
    drive = path.drive.rstrip(":")
    command = (
        f"$v=Get-Volume -DriveLetter '{drive}'; "
        f"$p=Get-Partition -DriveLetter '{drive}'; "
        "$d=$p | Get-Disk; "
        "[pscustomobject]@{"
        "DriveLetter=$v.DriveLetter;FileSystem=$v.FileSystem;FileSystemLabel=$v.FileSystemLabel;"
        "VolumePath=$v.Path;VolumeSize=$v.Size;VolumeFree=$v.SizeRemaining;"
        "DiskNumber=$d.Number;DiskFriendlyName=$d.FriendlyName;DiskSerialNumber=$d.SerialNumber;"
        "DiskBusType=[string]$d.BusType;DiskSize=$d.Size;PartitionNumber=$p.PartitionNumber"
        "} | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode or not completed.stdout.strip():
        return {
            "available": False,
            "reason": completed.stderr.strip() or f"exit_{completed.returncode}",
        }
    return {"available": True, **json.loads(completed.stdout)}


def describe_location(path: Path) -> dict[str, object]:
    parent = path.parent.resolve(strict=True)
    providers = _onedrive_roots()
    provider_root = next((root for root in providers if _is_within(path, root)), None)
    onedrive_processes: list[dict[str, object]] = []
    try:
        import psutil

        for process in psutil.process_iter(["pid", "name", "exe"]):
            name = str(process.info.get("name") or "")
            if name.lower() == "onedrive.exe":
                onedrive_processes.append(
                    {
                        "pid": int(process.info["pid"]),
                        "name": name,
                        "exe": process.info.get("exe"),
                    }
                )
    except Exception as exc:  # environment evidence is best effort
        onedrive_processes.append({"inspection_error": f"{type(exc).__name__}: {exc}"})
    return {
        "resolved_disposable_path": str(path),
        "resolved_existing_parent": str(parent),
        "volume": _windows_volume_device(parent),
        "parent_file_attributes": _path_attributes(parent),
        "synchronization_provider": "OneDrive" if provider_root is not None else "none_detected",
        "provider_root": str(provider_root) if provider_root is not None else None,
        "onedrive_processes": onedrive_processes,
        "provider_operating_state": (
            "normal_not_paused_by_benchmark" if onedrive_processes else "process_not_detected"
        ),
    }


def _median_range(values: Sequence[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _measurement_number(repetition: Mapping[str, object], workload: str, field_name: str) -> float:
    measurement = repetition[workload]
    if not isinstance(measurement, Mapping):
        raise TypeError(f"{workload} measurement must be a mapping")
    return _as_float(measurement[field_name])


def summarize(measured: Sequence[Mapping[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for workload in ("rng", "h2h"):
        by_location: dict[str, list[Mapping[str, object]]] = {"onedrive": [], "local": []}
        for repetition in measured:
            location = str(repetition["location"])
            value = repetition[workload]
            assert isinstance(value, Mapping)
            by_location[location].append(value)
        location_summary: dict[str, object] = {}
        for location, values in by_location.items():
            wall = [_as_float(value["wall_seconds"]) for value in values]
            cpu = [_as_float(value["total_cpu_seconds"]) for value in values]
            location_summary[location] = {
                "wall_seconds": _median_range(wall),
                "total_cpu_seconds": _median_range(cpu),
            }
        onedrive_wall = [_as_float(value["wall_seconds"]) for value in by_location["onedrive"]]
        local_wall = [_as_float(value["wall_seconds"]) for value in by_location["local"]]
        ratio = statistics.median(onedrive_wall) / statistics.median(local_wall)
        pair_ratios: list[float] = []
        for repetition_number in sorted({_as_int(item["repetition"]) for item in measured}):
            items = {
                str(item["location"]): item
                for item in measured
                if _as_int(item["repetition"]) == repetition_number
            }
            onedrive_measure = items["onedrive"][workload]
            local_measure = items["local"][workload]
            assert isinstance(onedrive_measure, Mapping)
            assert isinstance(local_measure, Mapping)
            pair_ratios.append(
                _as_float(onedrive_measure["wall_seconds"])
                / _as_float(local_measure["wall_seconds"])
            )
        first = [
            _measurement_number(item, workload, "wall_seconds")
            for item in measured
            if _as_int(item["position"]) == 1
        ]
        second = [
            _measurement_number(item, workload, "wall_seconds")
            for item in measured
            if _as_int(item["position"]) == 2
        ]
        summary[workload] = {
            "locations": location_summary,
            "onedrive_to_local_median_wall_ratio": ratio,
            "paired_wall_ratios": pair_ratios,
            "first_to_second_position_median_wall_ratio": (
                statistics.median(first) / statistics.median(second)
            ),
            "material_threshold_ratio": MATERIAL_WALL_RATIO,
            "material_and_repeatable": ratio >= MATERIAL_WALL_RATIO
            and sum(value > 1.0 for value in pair_ratios) >= 2,
        }
    return summary


def _correctness(measured: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"all_required_bytes_identical": True, "workloads": {}}
    workloads = result["workloads"]
    assert isinstance(workloads, dict)
    for workload in ("rng", "h2h"):
        keys: dict[str, set[str]] = {}
        for repetition in measured:
            measurement = repetition[workload]
            assert isinstance(measurement, Mapping)
            correctness = measurement["correctness"]
            assert isinstance(correctness, Mapping)
            for name, digest in correctness.items():
                keys.setdefault(str(name), set()).add(str(digest))
        normalized = {
            name: {"identical": len(digests) == 1, "digests": sorted(digests)}
            for name, digests in sorted(keys.items())
        }
        workloads[workload] = normalized
        if any(not bool(value["identical"]) for value in normalized.values()):
            result["all_required_bytes_identical"] = False
    return result


def _physical_device_confounder(locations: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    identities: dict[str, tuple[object, object]] = {}
    for name, location in locations.items():
        volume = location.get("volume")
        assert isinstance(volume, Mapping)
        identities[name] = (volume.get("DiskNumber"), volume.get("DiskSerialNumber"))
    available = all(
        any(value is not None for value in identity) for identity in identities.values()
    )
    return {
        "device_identity_available": available,
        "different_physical_devices": available and len(set(identities.values())) > 1,
        "identities": {name: list(identity) for name, identity in identities.items()},
    }


def run_benchmark(
    *,
    onedrive_root: Path,
    local_root: Path,
    repository_root: Path,
    settings: WorkloadSettings,
) -> dict[str, object]:
    onedrive_root, local_root = validate_disposable_roots(
        onedrive_root,
        local_root,
        repository_root=repository_root,
    )
    locations = {
        "onedrive": describe_location(onedrive_root),
        "local": describe_location(local_root),
    }
    roots = {"onedrive": onedrive_root, "local": local_root}
    root_tokens = {
        name: identity_sha256({"benchmark": BENCHMARK_VERSION, "root": str(root)})
        for name, root in roots.items()
    }
    created: list[str] = []
    warmups: list[dict[str, object]] = []
    measured: list[dict[str, object]] = []
    root_cleanup: dict[str, object] = {}
    try:
        for name, root in roots.items():
            _create_owned_directory(root, root_tokens[name])
            created.append(name)
        for position, location in enumerate(("onedrive", "local"), start=1):
            warmups.append(
                _run_location_once(
                    roots[location],
                    location=location,
                    repetition=0,
                    position=position,
                    warmup=True,
                    settings=settings,
                    root_token=root_tokens[location],
                )
            )
        for repetition, order in enumerate(
            (("onedrive", "local"), ("local", "onedrive"), ("onedrive", "local")),
            start=1,
        ):
            for position, location in enumerate(order, start=1):
                measured.append(
                    _run_location_once(
                        roots[location],
                        location=location,
                        repetition=repetition,
                        position=position,
                        warmup=False,
                        settings=settings,
                        root_token=root_tokens[location],
                    )
                )
    finally:
        for name in reversed(created):
            root = roots[name]
            if root.exists():
                try:
                    seconds, failures = _remove_owned_directory(root, root_tokens[name])
                    root_cleanup[name] = {
                        "seconds": seconds,
                        "failures": failures,
                        "exists_after_cleanup": root.exists(),
                    }
                except Exception as exc:
                    root_cleanup[name] = {
                        "seconds": None,
                        "failures": 1,
                        "exists_after_cleanup": root.exists(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
    if len(measured) != 6:
        raise RuntimeError("benchmark did not complete all six measured location runs")
    correctness = _correctness(measured)
    summary = summarize(measured)
    physical = _physical_device_confounder(locations)
    material = any(bool(summary[name]["material_and_repeatable"]) for name in ("rng", "h2h"))  # type: ignore[index]
    return {
        "benchmark": "farkle-task3a-storage",
        "benchmark_version": BENCHMARK_VERSION,
        "started_and_completed_utc": _utc_now(),
        "repository_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "repository_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
            "multiprocessing_start_method": "spawn",
            "locations": locations,
            "physical_device_confounder": physical,
        },
        "workload": asdict(settings),
        "execution_order": [
            {"repetition": 1, "order": ["onedrive", "local"]},
            {"repetition": 2, "order": ["local", "onedrive"]},
            {"repetition": 3, "order": ["onedrive", "local"]},
        ],
        "measurement_scope": {
            "fixture_preparation_included": False,
            "cleanup_included_in_workload_wall": False,
            "file_counts": (
                "benchmark-owned logical operations plus known v2 sidecar helper operations"
            ),
            "file_latency": "explicit Python opens/closes only; helper latency is publication time",
            "hashing": "actual sha256_file calls made by authentication and correctness checks",
        },
        "warmups": warmups,
        "measurements": measured,
        "summary": summary,
        "correctness": correctness,
        "root_cleanup": root_cleanup,
        "confounders": [
            "OneDrive provider activity is observed, not controlled; the client was not paused.",
            "Three paired repetitions bound runtime but do not estimate long-horizon provider variance.",
            "Python-level operation timing is not kernel ETW tracing.",
            *(
                ["Locations are on different physical devices."]
                if physical["different_physical_devices"]
                else [
                    "Locations resolve to the same reported physical device; no device confounder detected."
                ]
            ),
        ],
        "decision": {
            "material_ratio_threshold": MATERIAL_WALL_RATIO,
            "synchronized_tree_overhead_material_and_repeatable": material,
            "local_working_root_design_justified": material,
            "design_implemented": False,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onedrive-root", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output.resolve(strict=False)
    if output.exists() and not args.force:
        print(f"Task 3A output already exists; skipping without --force: {output}")
        return 0
    repository_root = Path(__file__).resolve().parents[1]
    result = run_benchmark(
        onedrive_root=args.onedrive_root,
        local_root=args.local_root,
        repository_root=repository_root,
        settings=WorkloadSettings(),
    )
    if not bool(result["correctness"]["all_required_bytes_identical"]):  # type: ignore[index]
        raise RuntimeError("Task 3A correctness digests differ between locations")
    output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(output)) as temporary:
        Path(temporary).write_bytes(canonical_json_bytes(result) + b"\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"Task 3A evidence written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
