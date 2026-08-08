"""Deterministic, authenticated, bounded-memory partitioned stage execution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeAlias

from farkle.config import ResourcesConfig
from farkle.utils.parallel import (
    ProcessTreeMemoryGuard,
    StageParallelPolicy,
    apply_native_thread_limits,
    process_map,
    resolve_mp_context,
    resolve_stage_parallel_policy,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
UnitCoordinate: TypeAlias = int | str
UnitWriter: TypeAlias = Callable[["PartitionedUnit", Path], None]


class PartitionedStageError(RuntimeError):
    """Raised when partitioned work or its authenticated lifecycle is invalid."""


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

    @property
    def order_bytes(self) -> bytes:
        return _canonical_bytes({"key": self.key, "relative_output": self.relative_output})

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


@dataclass(frozen=True, slots=True)
class _UnitTask:
    root: Path
    unit: PartitionedUnit
    identity: PartitionedStageIdentity
    writer: UnitWriter
    policy: StageParallelPolicy


def _output_path(root: Path, unit: PartitionedUnit) -> Path:
    # ``PartitionedUnit`` already rejects absolute and parent-traversal paths.
    # Avoid resolving not-yet-created OneDrive paths: Windows providers can
    # transiently return differently cased aliases in concurrent children.
    return root / "units" / Path(unit.relative_output)


def _stamp_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.unit.done.json")


def _stamp_payload(
    unit: PartitionedUnit,
    identity: PartitionedStageIdentity,
    output: Path,
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
        "output_size_bytes": output.stat().st_size,
        "output_sha256": _sha256_file(output),
    }
    payload["stamp_sha256"] = _identity_sha256(payload)
    return payload


def _validate_unit(
    root: Path,
    unit: PartitionedUnit,
    identity: PartitionedStageIdentity,
) -> dict[str, Any] | None:
    output = _output_path(root, unit)
    stamp_path = _stamp_path(output)
    if not output.is_file() or not stamp_path.is_file():
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
            and payload["output_size_bytes"] == output.stat().st_size
            and payload["output_sha256"] == _sha256_file(output)
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


def _quarantine_invalid_unit(root: Path, unit: PartitionedUnit) -> None:
    output = _output_path(root, unit)
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
    output = _output_path(task.root, task.unit)
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
        os.replace(temporary, output)
        _fsync_directory(output.parent)
        stamp = _stamp_payload(task.unit, task.identity, output)
        _write_bytes_atomic(
            _stamp_path(output),
            _canonical_bytes(stamp) + b"\n",
            prefix="._partition_stamp_",
        )
        if _validate_unit(task.root, task.unit, task.identity) is None:
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
        "output_size_bytes": stamp["output_size_bytes"],
        "output_sha256": stamp["output_sha256"],
        "stamp_sha256": stamp["stamp_sha256"],
    }


def _publish_final_manifest(
    path: Path,
    *,
    root: Path,
    identity: PartitionedStageIdentity,
    unit_source: Callable[[], Iterable[PartitionedUnit]],
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
                stamp = _validate_unit(root, unit, identity)
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
        os.replace(temporary, path)
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
) -> tuple[str, int] | None:
    """Validate the manifest identity, ordering, every stamp, and every output."""

    if not path.is_file():
        return None
    try:
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
                stamp = _validate_unit(root, unit, identity)
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
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
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
) -> PartitionedStageResult:
    """Run/reuse units and publish a final manifest only after complete validation."""

    root = Path(root)
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
        resources.rss_abort_mb,
        resources.rss_sample_interval_seconds,
    )
    guard.check_before_schedule(force=True)
    root.mkdir(parents=True, exist_ok=True)
    _quarantine_temporary_files(root)
    manifest_path = root / "partition_manifest.jsonl"
    if not force:
        current = validate_final_manifest(
            manifest_path,
            root=root,
            identity=identity,
            unit_source=unit_source,
        )
        if current is not None:
            manifest_sha, count = current
            return PartitionedStageResult(
                manifest_path,
                manifest_sha,
                count,
                count,
                0,
                guard.peak_rss_bytes / (1024 * 1024),
                policy,
            )
        if manifest_path.exists():
            _quarantine_paths(root, (manifest_path,))

    reused = 0
    scheduled = 0

    def pending_tasks() -> Iterable[_UnitTask]:
        nonlocal reused, scheduled
        for unit in _iter_ordered_units(unit_source):
            valid = None if force else _validate_unit(root, unit, identity)
            if valid is not None:
                reused += 1
                continue
            _quarantine_invalid_unit(root, unit)
            scheduled += 1
            yield _UnitTask(root, unit, identity, writer, policy)

    window = policy.process_workers * resources.max_in_flight_per_worker
    for _completed_key in process_map(
        _execute_unit,
        pending_tasks(),
        n_jobs=policy.process_workers,
        window=window,
        mp_context=resolve_mp_context(mp_start_method),
        memory_guard=guard,
    ):
        guard.check_before_schedule()

    manifest_sha, count = _publish_final_manifest(
        manifest_path,
        root=root,
        identity=identity,
        unit_source=unit_source,
    )
    validated = validate_final_manifest(
        manifest_path,
        root=root,
        identity=identity,
        unit_source=unit_source,
    )
    if validated != (manifest_sha, count):
        _quarantine_paths(root, (manifest_path,))
        raise PartitionedStageError("final partition manifest failed post-publication validation")
    return PartitionedStageResult(
        manifest_path,
        manifest_sha,
        count,
        reused,
        scheduled,
        guard.peak_rss_bytes / (1024 * 1024),
        policy,
    )


__all__ = [
    "PartitionedStageError",
    "PartitionedStageIdentity",
    "PartitionedStageResult",
    "PartitionedUnit",
    "run_partitioned_stage",
    "validate_final_manifest",
]
