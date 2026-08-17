"""Deterministic, authenticated, bounded-memory partitioned stage execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from itertools import chain, islice
from pathlib import Path, PurePosixPath
from typing import Any, TypeAlias

from farkle.config import ResourcesConfig
from farkle.utils.artifact_contract import (
    ArtifactSidecar,
    publish_staged_artifact_with_sidecar,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.authenticated_contract import (
    CodeIdentityPolicy,
    identity_sha256,
    resolve_code_identity,
)
from farkle.utils.parallel import (
    ProcessTreeMemoryGuard,
    ResourceFailureError,
    ResourceSafetyError,
    StageParallelPolicy,
    apply_native_thread_limits,
    classify_resource_exception,
    process_map,
    resolve_mp_context,
    resolve_stage_parallel_policy,
)
from farkle.utils.telemetry import (
    current_supervisor_recorder,
    current_supervisor_scope,
    install_worker_progress_endpoint,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
UnitCoordinate: TypeAlias = int | str
UnitWriter: TypeAlias = Callable[["PartitionedUnit", Path], None]
UnitSidecarFactory: TypeAlias = Callable[["PartitionedUnit", Path], ArtifactSidecar | None]
UnitPostPublisher: TypeAlias = Callable[["PartitionedUnit", Path], None]
UnitValidator: TypeAlias = Callable[["PartitionedUnit", Path], bool | Mapping[str, Any]]


class PartitionedStageError(RuntimeError):
    """Raised when partitioned work or its authenticated lifecycle is invalid."""


def resolved_code_identity_sha256(cfg: Any) -> str:
    """Return the full repository code identity for resumable unit freshness."""

    identity = getattr(cfg, "_code_identity", None)
    if identity is None:
        identity = resolve_code_identity(
            Path(__file__).resolve().parents[3],
            policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY,
        )
    return identity_sha256(asdict(identity))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _identity_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_identity(identity: "PartitionedStageIdentity") -> dict[str, Any]:
    payload = json.loads(_canonical_bytes(asdict(identity)))
    assert isinstance(payload, dict)
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_atomic(path: Path, content: bytes, *, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=prefix, dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class PartitionedStageIdentity:
    """Statistical and code identity shared by every required unit."""

    stage_name: str
    root_seed: int
    input_identities: tuple[tuple[str, str], ...]
    statistical_config_sha256: str
    code_identity_sha256: str
    schema_version: int
    method_version: int

    def __post_init__(self) -> None:
        if not self.stage_name.strip():
            raise ValueError("partitioned stage name must be non-empty")
        if self.input_identities != tuple(sorted(self.input_identities)):
            raise ValueError("partitioned input identities must be sorted by logical role")
        if len({role for role, _digest in self.input_identities}) != len(self.input_identities):
            raise ValueError("partitioned input identity roles must be unique")
        hashes = [
            self.statistical_config_sha256,
            self.code_identity_sha256,
            *(digest for _role, digest in self.input_identities),
        ]
        if any(_SHA256.fullmatch(digest) is None for digest in hashes):
            raise ValueError("partitioned stage identities require lowercase SHA-256 digests")
        if self.schema_version < 1 or self.method_version < 1:
            raise ValueError("partitioned schema and method versions must be positive")

    @property
    def sha256(self) -> str:
        return _identity_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class PartitionedUnit:
    """One deterministically ordered and independently restartable work unit."""

    key: tuple[UnitCoordinate, ...]
    relative_output: str
    input_identities: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.key or any(not isinstance(item, (int, str)) for item in self.key):
            raise ValueError("partitioned unit keys require integer/string semantic coordinates")
        relative = PurePosixPath(self.relative_output)
        if (
            not self.relative_output
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != self.relative_output.replace("\\", "/")
        ):
            raise ValueError("partitioned unit output must be a normalized relative path")
        if self.input_identities != tuple(sorted(self.input_identities)):
            raise ValueError("partitioned unit input identities must be sorted by logical role")
        if len({role for role, _digest in self.input_identities}) != len(self.input_identities):
            raise ValueError("partitioned unit input identity roles must be unique")
        if any(_SHA256.fullmatch(digest) is None for _role, digest in self.input_identities):
            raise ValueError("partitioned unit input identities require lowercase SHA-256 digests")

    @property
    def order_bytes(self) -> bytes:
        return _canonical_bytes(
            {
                "key": self.key,
                "relative_output": self.relative_output,
                "input_identities": self.input_identities,
            }
        )

    @property
    def order_key(self) -> tuple[tuple[int, int | str], ...]:
        coordinates = tuple((0, item) if isinstance(item, int) else (1, item) for item in self.key)
        return (*coordinates, (2, self.relative_output))


@dataclass(frozen=True, slots=True)
class PartitionedStageResult:
    """Execution and reuse summary for a validated final manifest."""

    manifest_path: Path
    manifest_sha256: str
    required_units: int
    reused_units: int
    completed_units: int
    peak_sampled_rss_mb: float
    policy: StageParallelPolicy
    execution_attempts: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class _UnitTask:
    root: Path
    unit: PartitionedUnit
    identity: PartitionedStageIdentity
    writer: UnitWriter
    policy: StageParallelPolicy
    output_prefix: str
    sidecar_factory: UnitSidecarFactory | None
    post_publisher: UnitPostPublisher | None
    validator: UnitValidator | None


def _output_path(root: Path, unit: PartitionedUnit, *, output_prefix: str = "units") -> Path:
    # ``PartitionedUnit`` already rejects absolute and parent-traversal paths.
    # Avoid resolving not-yet-created OneDrive paths: Windows providers can
    # transiently return differently cased aliases in concurrent children.
    return root / output_prefix / Path(unit.relative_output)


def _stamp_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.unit.done.json")


def _stamp_payload(
    unit: PartitionedUnit,
    identity: PartitionedStageIdentity,
    output: Path,
    *,
    unit_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "unit_stamp_schema_version": 1,
        "completion_state": "complete_valid",
        "stage_name": identity.stage_name,
        "stage_identity_sha256": identity.sha256,
        "root_seed": identity.root_seed,
        "input_identities": [list(item) for item in identity.input_identities],
        "statistical_config_sha256": identity.statistical_config_sha256,
        "code_identity_sha256": identity.code_identity_sha256,
        "output_schema_version": identity.schema_version,
        "method_version": identity.method_version,
        "unit_key": list(unit.key),
        "relative_output": unit.relative_output,
        "unit_input_identities": [list(item) for item in unit.input_identities],
        "output_size_bytes": output.stat().st_size,
        "output_sha256": _sha256_file(output),
        "unit_metadata": dict(unit_metadata),
    }
    payload["stamp_sha256"] = _identity_sha256(payload)
    return payload


def _validated_unit_metadata(
    unit: PartitionedUnit,
    output: Path,
    validator: UnitValidator | None,
) -> dict[str, Any] | None:
    if validator is None:
        return {}
    try:
        result = validator(unit, output)
    except Exception as exc:  # noqa: BLE001 - invalid auxiliary evidence is not reusable
        classification = classify_resource_exception(exc)
        if classification is not None:
            if isinstance(exc, ResourceFailureError):
                raise
            raise ResourceFailureError(classification, str(exc)) from exc
        return None
    if result is False:
        return None
    if result is True:
        return {}
    try:
        normalized = json.loads(_canonical_bytes(dict(result)))
    except (TypeError, ValueError):
        return None
    return normalized if isinstance(normalized, dict) else None


def _validate_unit(
    root: Path,
    unit: PartitionedUnit,
    identity: PartitionedStageIdentity,
    *,
    output_prefix: str = "units",
    validator: UnitValidator | None = None,
) -> dict[str, Any] | None:
    output = _output_path(root, unit, output_prefix=output_prefix)
    stamp_path = _stamp_path(output)
    if not output.is_file() or not stamp_path.is_file():
        return None
    unit_metadata = _validated_unit_metadata(unit, output, validator)
    if unit_metadata is None:
        return None
    try:
        payload = json.loads(stamp_path.read_text(encoding="utf-8"))
        recorded = payload.pop("stamp_sha256")
        valid = (
            recorded == _identity_sha256(payload)
            and payload["unit_stamp_schema_version"] == 1
            and payload["completion_state"] == "complete_valid"
            and payload["stage_name"] == identity.stage_name
            and payload["stage_identity_sha256"] == identity.sha256
            and payload["root_seed"] == identity.root_seed
            and payload["input_identities"] == [list(item) for item in identity.input_identities]
            and payload["statistical_config_sha256"] == identity.statistical_config_sha256
            and payload["code_identity_sha256"] == identity.code_identity_sha256
            and payload["output_schema_version"] == identity.schema_version
            and payload["method_version"] == identity.method_version
            and payload["unit_key"] == list(unit.key)
            and payload["relative_output"] == unit.relative_output
            and payload["unit_input_identities"] == [list(item) for item in unit.input_identities]
            and payload["output_size_bytes"] == output.stat().st_size
            and payload["output_sha256"] == _sha256_file(output)
            and payload["unit_metadata"] == unit_metadata
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not valid:
        return None
    payload["stamp_sha256"] = recorded
    return payload


def _quarantine_paths(root: Path, paths: Iterable[Path]) -> None:
    quarantine = root / "quarantine"
    for path in paths:
        if not path.exists():
            continue
        quarantine.mkdir(parents=True, exist_ok=True)
        token = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
        destination = quarantine / f"{time.time_ns()}_{token}_{path.name}"
        os.replace(path, destination)


def _quarantine_invalid_unit(
    root: Path, unit: PartitionedUnit, *, output_prefix: str = "units"
) -> None:
    output = _output_path(root, unit, output_prefix=output_prefix)
    _quarantine_paths(root, (output, _stamp_path(output)))


def _quarantine_temporary_files(root: Path) -> None:
    if not root.exists():
        return
    _quarantine_paths(
        root,
        sorted(
            (path for path in root.rglob("._partition_*") if path.is_file()),
            key=lambda path: path.as_posix(),
        ),
    )


def _execute_unit(task: _UnitTask) -> tuple[UnitCoordinate, ...]:
    apply_native_thread_limits(task.policy)
    output = _output_path(task.root, task.unit, output_prefix=task.output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="._partition_output_", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        task.writer(task.unit, temporary)
        if not temporary.is_file():
            raise PartitionedStageError(f"unit writer did not create {temporary}")
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        metadata = (
            task.sidecar_factory(task.unit, output) if task.sidecar_factory is not None else None
        )
        if metadata is None:
            os.replace(temporary, output)
        else:
            publish_staged_artifact_with_sidecar(temporary, output, metadata)
        _fsync_directory(output.parent)
        if task.post_publisher is not None:
            task.post_publisher(task.unit, output)
        unit_metadata = _validated_unit_metadata(task.unit, output, task.validator)
        if unit_metadata is None:
            raise PartitionedStageError(
                f"published unit failed output validation: {task.unit.key!r}"
            )
        stamp = _stamp_payload(
            task.unit,
            task.identity,
            output,
            unit_metadata=unit_metadata,
        )
        _write_bytes_atomic(
            _stamp_path(output),
            _canonical_bytes(stamp) + b"\n",
            prefix="._partition_stamp_",
        )
        if (
            _validate_unit(
                task.root,
                task.unit,
                task.identity,
                output_prefix=task.output_prefix,
                validator=task.validator,
            )
            is None
        ):
            raise PartitionedStageError(f"published unit failed validation: {task.unit.key!r}")
        return task.unit.key
    finally:
        temporary.unlink(missing_ok=True)


def _iter_ordered_units(
    unit_source: Callable[[], Iterable[PartitionedUnit]],
) -> Iterable[PartitionedUnit]:
    previous: tuple[tuple[int, int | str], ...] | None = None
    for unit in unit_source():
        current = unit.order_key
        if previous is not None and current <= previous:
            raise PartitionedStageError("partitioned units must be strictly increasing")
        previous = current
        yield unit


def _manifest_entry(unit: PartitionedUnit, stamp: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "unit",
        "unit_key": list(unit.key),
        "relative_output": unit.relative_output,
        "unit_input_identities": [list(item) for item in unit.input_identities],
        "output_size_bytes": stamp["output_size_bytes"],
        "output_sha256": stamp["output_sha256"],
        "stamp_sha256": stamp["stamp_sha256"],
        "unit_metadata": stamp["unit_metadata"],
    }


def _publish_final_manifest(
    path: Path,
    *,
    root: Path,
    identity: PartitionedStageIdentity,
    unit_source: Callable[[], Iterable[PartitionedUnit]],
    output_prefix: str = "units",
    validator: UnitValidator | None = None,
    sidecar: ArtifactSidecar | None = None,
) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="._partition_manifest_", dir=path.parent)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            header = (
                _canonical_bytes(
                    {
                        "type": "header",
                        "manifest_schema_version": 1,
                        "stage_identity": _json_identity(identity),
                        "stage_identity_sha256": identity.sha256,
                    }
                )
                + b"\n"
            )
            handle.write(header)
            digest.update(header)
            for unit in _iter_ordered_units(unit_source):
                stamp = _validate_unit(
                    root,
                    unit,
                    identity,
                    output_prefix=output_prefix,
                    validator=validator,
                )
                if stamp is None:
                    raise PartitionedStageError(
                        f"required unit is missing or invalid during finalization: {unit.key!r}"
                    )
                line = _canonical_bytes(_manifest_entry(unit, stamp)) + b"\n"
                handle.write(line)
                digest.update(line)
                count += 1
            manifest_sha = digest.hexdigest()
            trailer = (
                _canonical_bytes(
                    {"type": "completion", "required_units": count, "manifest_sha256": manifest_sha}
                )
                + b"\n"
            )
            handle.write(trailer)
            handle.flush()
            os.fsync(handle.fileno())
        if sidecar is None:
            os.replace(temporary, path)
        else:
            publish_staged_artifact_with_sidecar(temporary, path, sidecar)
        _fsync_directory(path.parent)
        return manifest_sha, count
    finally:
        temporary.unlink(missing_ok=True)


def validate_final_manifest(
    path: Path,
    *,
    root: Path,
    identity: PartitionedStageIdentity,
    unit_source: Callable[[], Iterable[PartitionedUnit]],
    output_prefix: str = "units",
    validator: UnitValidator | None = None,
    require_sidecar: bool = False,
) -> tuple[str, int] | None:
    """Validate the manifest identity, ordering, every stamp, and every output."""

    if not path.is_file():
        return None
    try:
        if require_sidecar:
            validate_artifact_sidecar(path)
        with path.open("rb") as handle:
            header_line = handle.readline()
            header = json.loads(header_line)
            if (
                header.get("type") != "header"
                or header.get("manifest_schema_version") != 1
                or header.get("stage_identity") != _json_identity(identity)
                or header.get("stage_identity_sha256") != identity.sha256
            ):
                return None
            digest = hashlib.sha256(header_line)
            count = 0
            for unit in _iter_ordered_units(unit_source):
                line = handle.readline()
                entry = json.loads(line)
                stamp = _validate_unit(
                    root,
                    unit,
                    identity,
                    output_prefix=output_prefix,
                    validator=validator,
                )
                if stamp is None or entry != _manifest_entry(unit, stamp):
                    return None
                digest.update(line)
                count += 1
            trailer = json.loads(handle.readline())
            if handle.read(1) or trailer != {
                "type": "completion",
                "required_units": count,
                "manifest_sha256": digest.hexdigest(),
            }:
                return None
            return digest.hexdigest(), count
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return None


def run_partitioned_stage(
    *,
    root: Path,
    identity: PartitionedStageIdentity,
    unit_source: Callable[[], Iterable[PartitionedUnit]],
    writer: UnitWriter,
    resources: ResourcesConfig,
    requested_workers: int | None,
    mp_start_method: str | None = None,
    force: bool = False,
    memory_guard: ProcessTreeMemoryGuard | None = None,
    output_prefix: str = "units",
    sidecar_factory: UnitSidecarFactory | None = None,
    post_publisher: UnitPostPublisher | None = None,
    validator: UnitValidator | None = None,
    manifest_path: Path | None = None,
    manifest_sidecar: ArtifactSidecar | None = None,
    progress_total_units: int | None = None,
    progress_phase: str | None = None,
    enable_worker_progress: bool = False,
) -> PartitionedStageResult:
    """Run/reuse units and publish a final manifest only after complete validation."""

    root = Path(root)
    normalized_prefix = PurePosixPath(output_prefix.replace("\\", "/"))
    if (
        normalized_prefix.is_absolute()
        or ".." in normalized_prefix.parts
        or normalized_prefix.as_posix() != output_prefix.replace("\\", "/")
    ):
        raise ValueError("partitioned output_prefix must be a normalized relative path")
    output_prefix = normalized_prefix.as_posix()
    resources_stage = (
        identity.stage_name
        if identity.stage_name in resources.estimated_worker_memory_mb
        else "partitioned_stage"
    )
    policy_cfg = type("_PolicyConfig", (), {"n_jobs": requested_workers})()
    policy = resolve_stage_parallel_policy(
        resources_stage,
        policy_cfg,
        n_jobs_override=requested_workers,
        resources=resources,
    )
    apply_native_thread_limits(policy)
    guard = memory_guard or ProcessTreeMemoryGuard(
        resources.aggregate_memory_hard_limit_mb,
        rss_warning_mb=resources.process_tree_warning_threshold_mb,
        minimum_system_available_memory_mb=resources.minimum_system_available_memory_mb,
        sample_interval_seconds=resources.rss_sample_interval_seconds,
    )
    guard.check_before_schedule(force=True)
    recorder = current_supervisor_recorder()
    scope = current_supervisor_scope()
    scope_name = (
        scope.scope
        if scope is not None
        else "partitioned:"
        f"{identity.stage_name}:{hashlib.sha256(str(root).encode()).hexdigest()[:12]}"
    )
    execution_phase = progress_phase or "partition_execution"
    completion_scope = f"{scope_name}:{identity.stage_name}"
    if scope is not None:
        scope.update(
            phase="resume_scan",
            state="authenticating",
            progress={
                "total_units": progress_total_units,
                "requested_workers": requested_workers,
                "effective_workers": policy.process_workers,
            },
        )
    root.mkdir(parents=True, exist_ok=True)
    _quarantine_temporary_files(root)
    manifest_path = (
        root / "partition_manifest.jsonl" if manifest_path is None else Path(manifest_path)
    )
    try:
        manifest_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("partition manifest must be inside the partitioned-stage root") from exc
    if not force:
        current = validate_final_manifest(
            manifest_path,
            root=root,
            identity=identity,
            unit_source=unit_source,
            output_prefix=output_prefix,
            validator=validator,
            require_sidecar=manifest_sidecar is not None,
        )
        if current is not None:
            manifest_sha, count = current
            guard.check_before_schedule(force=True)
            result = PartitionedStageResult(
                manifest_path,
                manifest_sha,
                count,
                count,
                0,
                guard.peak_rss_bytes / (1024 * 1024),
                policy,
            )
            if recorder is not None:
                recorder.record_completion_summary(
                    completion_scope,
                    stage=identity.stage_name,
                    summary={
                        "total_units": count,
                        "reused_units": count,
                        "completed_units": 0,
                        "requested_workers": requested_workers,
                        "effective_workers": 0,
                        "reconciled_from": "authenticated_partition_manifest",
                    },
                )
            return result
        if manifest_path.exists():
            _quarantine_paths(root, (manifest_path, sidecar_path(manifest_path)))

    reused = 0
    scheduled = 0
    execution_attempts: list[dict[str, Any]] = []
    telemetry_path = root / "_execution" / "execution_telemetry.json"

    def pending_tasks(
        attempt_policy: StageParallelPolicy,
        *,
        honor_force: bool,
        count_for_result: bool,
    ) -> Iterable[_UnitTask]:
        nonlocal reused, scheduled
        for unit in _iter_ordered_units(unit_source):
            valid = (
                None
                if honor_force and force
                else _validate_unit(
                    root,
                    unit,
                    identity,
                    output_prefix=output_prefix,
                    validator=validator,
                )
            )
            if valid is not None:
                if count_for_result:
                    reused += 1
                continue
            _quarantine_invalid_unit(root, unit, output_prefix=output_prefix)
            if count_for_result:
                scheduled += 1
            yield _UnitTask(
                root,
                unit,
                identity,
                writer,
                attempt_policy,
                output_prefix,
                sidecar_factory,
                post_publisher,
                validator,
            )

    def write_execution_telemetry(final_outcome: str) -> None:
        payload = {
            "telemetry_schema_version": 1,
            "stage_name": identity.stage_name,
            "stage_identity_sha256": identity.sha256,
            "original_policy": asdict(policy),
            "attempts": execution_attempts,
            "final_outcome": final_outcome,
            "warning_crossings": int(getattr(guard, "warning_crossings", 0)),
            "backpressure_seconds": float(getattr(guard, "backpressure_seconds", 0.0)),
            "high_water_timeout_seconds": float(getattr(guard, "high_water_timeout_seconds", 30.0)),
            "aggregate_memory_source": getattr(guard, "aggregate_memory_source", None),
            "peak_aggregate_memory_bytes": int(getattr(guard, "peak_aggregate_memory_bytes", 0)),
        }
        _write_bytes_atomic(
            telemetry_path,
            _canonical_bytes(payload) + b"\n",
            prefix="._partition_execution_telemetry_",
        )

    attempt_policy = policy
    for attempt_index in range(2):
        task_iterator = iter(
            pending_tasks(
                attempt_policy,
                honor_force=attempt_index == 0,
                count_for_result=attempt_index == 0,
            )
        )
        task_head = list(islice(task_iterator, attempt_policy.process_workers))
        effective_workers = len(task_head)
        execution_policy = replace(
            attempt_policy,
            process_workers=max(1, effective_workers),
        )
        tasks = chain(
            (replace(task, policy=execution_policy) for task in task_head),
            (replace(task, policy=execution_policy) for task in task_iterator),
        )
        attempt_record: dict[str, Any] = {
            "attempt": attempt_index + 1,
            "worker_count": effective_workers,
            "pending_units": (
                effective_workers if effective_workers < attempt_policy.process_workers else None
            ),
            "policy": asdict(execution_policy),
            "retry": attempt_index == 1,
        }
        execution_attempts.append(attempt_record)
        endpoint = None
        try:
            if effective_workers == 0:
                attempt_record["outcome"] = "complete"
                break
            window = effective_workers * resources.max_in_flight_per_worker
            if scope is not None:
                scope.update(
                    phase=execution_phase,
                    state="working" if attempt_index == 0 else "retrying_downshifted",
                    progress={
                        "total_units": progress_total_units,
                        "reused_units": reused,
                        "scheduled_units": scheduled,
                        "attempt": attempt_index + 1,
                        "requested_workers": requested_workers,
                        "effective_workers": effective_workers,
                        "window": window,
                    },
                )
            mp_context = resolve_mp_context(mp_start_method)
            if (
                recorder is not None
                and enable_worker_progress
                and effective_workers > 1
            ):
                endpoint = recorder.create_worker_progress_endpoint(
                    scope_name,
                    mp_context=mp_context,
                )
            attempt_started_at = time.monotonic()

            def scheduler_progress(
                event: Mapping[str, object],
                attempt_number: int = attempt_index + 1,
                workers: int = effective_workers,
                attempt_started: float = attempt_started_at,
            ) -> None:
                if scope is None:
                    return
                event_name = str(event.get("event", "working"))
                state = {
                    "schedule_check": "waiting_for_memory",
                    "waiting_on_futures": "waiting_on_futures",
                    "cancelling": "cancelling",
                }.get(event_name, "working")
                completed_value = event.get("completed", 0)
                completed_now = completed_value if isinstance(completed_value, int) else 0
                elapsed_now = max(0.0, time.monotonic() - attempt_started)
                rate = completed_now / elapsed_now if elapsed_now > 0 else 0.0
                remaining = (
                    max(0, progress_total_units - reused - completed_now)
                    if progress_total_units is not None
                    else None
                )
                pending_value = event.get("pending", 0)
                pending_now = pending_value if isinstance(pending_value, int) else 0
                scope.update(
                    phase=execution_phase,
                    state=state,
                    progress={
                        "total_units": progress_total_units,
                        "reused_units": reused,
                        "scheduled_units": scheduled,
                        "attempt": attempt_number,
                        "requested_workers": requested_workers,
                        "effective_workers": workers,
                        "active_workers": min(workers, pending_now),
                        "queued_units": max(0, pending_now - workers),
                        "units_per_second": rate,
                        "eta_seconds": (
                            remaining / rate
                            if remaining is not None and rate > 0
                            else None
                        ),
                        **event,
                    },
                )

            for _completed_key in process_map(
                _execute_unit,
                tasks,
                n_jobs=effective_workers,
                initializer=(install_worker_progress_endpoint if endpoint is not None else None),
                initargs=((endpoint,) if endpoint is not None else ()),
                window=window,
                mp_context=mp_context,
                memory_guard=guard,
                progress_callback=scheduler_progress,
            ):
                guard.check_before_schedule()
            attempt_record["outcome"] = "complete"
            break
        except ResourceSafetyError as exc:
            classification = (
                exc.classification if isinstance(exc, ResourceFailureError) else "resource_safety"
            )
            attempt_record.update(
                outcome="resource_failure",
                failure_classification=classification,
                error=f"{type(exc).__name__}: {exc}",
            )
            if scope is not None:
                scope.update(
                    phase=execution_phase,
                    state=(
                        "resource_retry_pending"
                        if attempt_index == 0
                        else "resource_failure"
                    ),
                    progress={
                        "attempt": attempt_index + 1,
                        "effective_workers": effective_workers,
                        "failure_classification": classification,
                        "next_worker_count": (
                            max(1, effective_workers // 2)
                            if attempt_index == 0
                            else None
                        ),
                    },
                )
            write_execution_telemetry("retrying" if attempt_index == 0 else "resource_failure")
            if attempt_index == 1:
                raise
            retry_workers = max(1, effective_workers // 2)
            attempt_policy = resolve_stage_parallel_policy(
                resources_stage,
                policy_cfg,
                n_jobs_override=retry_workers,
                resources=resources,
            )
            apply_native_thread_limits(attempt_policy)
        except BaseException as exc:
            attempt_record.update(
                outcome="non_resource_failure",
                error=f"{type(exc).__name__}: {exc}",
            )
            write_execution_telemetry("non_resource_failure")
            raise
        finally:
            if recorder is not None:
                recorder.close_worker_progress_endpoint(endpoint)

    guard.check_before_schedule(force=True)
    if scope is not None:
        scope.update(phase="manifest_publication", state="publishing")
    manifest_sha, count = _publish_final_manifest(
        manifest_path,
        root=root,
        identity=identity,
        unit_source=unit_source,
        output_prefix=output_prefix,
        validator=validator,
        sidecar=manifest_sidecar,
    )
    validated = validate_final_manifest(
        manifest_path,
        root=root,
        identity=identity,
        unit_source=unit_source,
        output_prefix=output_prefix,
        validator=validator,
        require_sidecar=manifest_sidecar is not None,
    )
    if validated != (manifest_sha, count):
        _quarantine_paths(root, (manifest_path, sidecar_path(manifest_path)))
        raise PartitionedStageError("final partition manifest failed post-publication validation")
    write_execution_telemetry("complete")
    result = PartitionedStageResult(
        manifest_path,
        manifest_sha,
        count,
        reused,
        scheduled,
        guard.peak_rss_bytes / (1024 * 1024),
        policy,
        tuple(execution_attempts),
    )
    if recorder is not None:
        recorder.record_completion_summary(
            completion_scope,
            stage=identity.stage_name,
            summary={
                "total_units": count,
                "reused_units": reused,
                "completed_units": scheduled,
                "requested_workers": requested_workers,
                "effective_workers": max(
                    (int(item.get("worker_count", 0)) for item in execution_attempts),
                    default=0,
                ),
                "attempt_count": len(execution_attempts),
                "retry_downshifted": len(execution_attempts) > 1,
                "reconciled_from": "authenticated_partition_manifest",
            },
        )
    return result


__all__ = [
    "PartitionedStageError",
    "PartitionedStageIdentity",
    "PartitionedStageResult",
    "PartitionedUnit",
    "run_partitioned_stage",
    "resolved_code_identity_sha256",
    "validate_final_manifest",
]
