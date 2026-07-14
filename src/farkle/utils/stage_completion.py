"""Versioned stage freshness and resumable lifecycle state."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from farkle.utils.artifact_contract import validate_artifact_sidecar
from farkle.utils.writer import atomic_path

__all__ = [
    "CompletionState",
    "freshness_sha256",
    "read_stage_done",
    "resolve_stage_state",
    "stage_done_path",
    "stage_is_up_to_date",
    "write_stage_done",
]

_DEFAULT_STATUS = "success"
_ALLOWED_STATUSES = {
    "success",
    "failed",
    "skipped",
    "partial_resumable",
    "blocked_by_cap",
}
_SCHEMA_VERSION = 3
_DEFAULT_CACHE_KEY_VERSION = 3


class CompletionState(str, Enum):
    """Canonical lifecycle states for resumable stage work."""

    NOT_STARTED = "not_started"
    PARTIAL_RESUMABLE = "partial_resumable"
    COMPLETE_VALID = "complete_valid"
    COMPLETE_STALE = "complete_stale"
    BLOCKED_BY_CAP = "blocked_by_cap"


def stage_done_path(stage_dir: Path, name: str) -> Path:
    """Return a ``.done.json`` marker for a named stage directory."""

    return stage_dir / f"{name}.done.json"


def _coerce_paths(paths: Iterable[Path]) -> list[Path]:
    return [Path(path) for path in paths]


def freshness_sha256(freshness_key: Mapping[str, Any]) -> str:
    """Return the deterministic digest for a statistical freshness contract."""

    canonical = json.dumps(
        dict(freshness_key),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_stage_done(done_path: Path) -> dict[str, object]:
    """Return normalized stage metadata from a completion-state stamp."""

    payload: dict[str, object] = {
        "schema_version": None,
        "stage": None,
        "config_sha": None,
        "stage_config_sha": None,
        "cache_key_version": None,
        "freshness_key": None,
        "freshness_sha256": None,
        "completion_state": CompletionState.NOT_STARTED.value,
        "inputs": [],
        "input_fingerprints": [],
        "outputs": [],
        "status": "missing",
        "reason": None,
        "blocking_dependency": None,
        "upstream_stage": None,
    }
    if not done_path.exists():
        return payload
    try:
        metadata = json.loads(done_path.read_text(encoding="utf-8"))
    except Exception:
        payload["status"] = "invalid"
        payload["reason"] = "invalid json"
        payload["completion_state"] = CompletionState.PARTIAL_RESUMABLE.value
        return payload
    for key in payload:
        if key in metadata:
            payload[key] = metadata[key]
    payload["inputs"] = list(metadata.get("inputs") or [])
    payload["outputs"] = list(metadata.get("outputs") or [])
    payload["status"] = metadata.get("status", _DEFAULT_STATUS)
    return payload


def _resolve_expected_contract(
    *,
    cfg: Any | None,
    stage: str | None,
    config_sha: str | None,
    stage_config_sha: str | None,
    cache_key_version: int,
    freshness_key: Mapping[str, Any] | None,
) -> tuple[str | None, str | None, int, Mapping[str, Any] | None]:
    if cfg is not None and stage is not None:
        if config_sha is None:
            config_sha = getattr(cfg, "config_sha", None)
        if stage_config_sha is None:
            stage_config_sha = cfg.stage_config_sha(stage)
        cache_key_version = int(cfg.stage_cache_key_version(stage))
        if freshness_key is None:
            freshness_key = cfg.freshness_key()
    return config_sha, stage_config_sha, cache_key_version, freshness_key


def _path_fingerprint(path: Path) -> dict[str, object]:
    """Return a low-I/O identity for one file or recursively tracked directory."""

    if not path.exists():
        return {"path": str(path), "kind": "missing"}
    stat = path.stat()
    if path.is_file():
        return {
            "path": str(path),
            "kind": "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    children = []
    for child in sorted((item for item in path.rglob("*") if item.is_file()), key=str):
        child_stat = child.stat()
        children.append(
            {
                "path": child.relative_to(path).as_posix(),
                "size": child_stat.st_size,
                "mtime_ns": child_stat.st_mtime_ns,
            }
        )
    return {
        "path": str(path),
        "kind": "directory",
        "mtime_ns": stat.st_mtime_ns,
        "children": children,
    }


def resolve_stage_state(
    done_path: Path,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    *,
    config_sha: str | None = None,
    stage_config_sha: str | None = None,
    cache_key_version: int = _DEFAULT_CACHE_KEY_VERSION,
    stage: str | None = None,
    cfg: Any | None = None,
    freshness_key: Mapping[str, Any] | None = None,
    sidecar_artifacts: Iterable[Path] = (),
    partial_paths: Iterable[Path] = (),
    cap_reached: bool = False,
) -> CompletionState:
    """Resolve one stage into a canonical resumable lifecycle state."""

    input_paths = _coerce_paths(inputs)
    output_paths = _coerce_paths(outputs)
    partial = _coerce_paths(partial_paths)
    has_materialized_work = any(path.exists() for path in (*output_paths, *partial))
    if cap_reached:
        return CompletionState.BLOCKED_BY_CAP
    if not done_path.exists():
        return (
            CompletionState.PARTIAL_RESUMABLE
            if has_materialized_work
            else CompletionState.NOT_STARTED
        )

    metadata = read_stage_done(done_path)
    recorded_state = metadata.get("completion_state")
    if (
        recorded_state == CompletionState.BLOCKED_BY_CAP.value
        or metadata.get("status") == "blocked_by_cap"
    ):
        return CompletionState.BLOCKED_BY_CAP
    if metadata.get("status") in {"failed", "skipped", "partial_resumable", "invalid"}:
        return CompletionState.PARTIAL_RESUMABLE
    if metadata.get("status") != _DEFAULT_STATUS:
        return CompletionState.COMPLETE_STALE
    if not input_paths:
        recorded_inputs = metadata.get("inputs")
        if isinstance(recorded_inputs, list):
            input_paths = [Path(str(value)) for value in recorded_inputs]

    config_sha, stage_config_sha, cache_key_version, freshness_key = _resolve_expected_contract(
        cfg=cfg,
        stage=stage,
        config_sha=config_sha,
        stage_config_sha=stage_config_sha,
        cache_key_version=cache_key_version,
        freshness_key=freshness_key,
    )
    if metadata.get("schema_version") != _SCHEMA_VERSION:
        return CompletionState.COMPLETE_STALE
    if stage is not None and metadata.get("stage") != stage:
        return CompletionState.COMPLETE_STALE
    if config_sha is not None and metadata.get("config_sha") not in (None, config_sha):
        return CompletionState.COMPLETE_STALE
    expected_stage_sha = stage_config_sha if stage_config_sha is not None else config_sha
    if expected_stage_sha is not None and metadata.get("stage_config_sha") != expected_stage_sha:
        return CompletionState.COMPLETE_STALE
    if metadata.get("cache_key_version") != cache_key_version:
        return CompletionState.COMPLETE_STALE

    if freshness_key is not None:
        expected_freshness_sha = freshness_sha256(freshness_key)
        recorded_freshness = metadata.get("freshness_key")
        if recorded_freshness is not None:
            if recorded_freshness != dict(freshness_key):
                return CompletionState.COMPLETE_STALE
            if metadata.get("freshness_sha256") != expected_freshness_sha:
                return CompletionState.COMPLETE_STALE
        elif expected_stage_sha is None:
            # Stamps without an explicit freshness payload are valid only when
            # their stage hash already binds the complete freshness contract.
            return CompletionState.COMPLETE_STALE

    recorded_fingerprints = metadata.get("input_fingerprints")
    if recorded_fingerprints and recorded_fingerprints != [
        _path_fingerprint(path) for path in input_paths
    ]:
        return CompletionState.COMPLETE_STALE
    done_mtime = done_path.stat().st_mtime
    if any(not path.exists() or path.stat().st_mtime > done_mtime for path in input_paths):
        return CompletionState.COMPLETE_STALE
    if not all(path.exists() for path in output_paths):
        return CompletionState.COMPLETE_STALE
    try:
        for artifact in _coerce_paths(sidecar_artifacts):
            validate_artifact_sidecar(artifact)
    except RuntimeError:
        return CompletionState.COMPLETE_STALE
    return CompletionState.COMPLETE_VALID


def stage_is_up_to_date(
    done_path: Path,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    **kwargs: Any,
) -> bool:
    """Return whether the stage resolves to :attr:`COMPLETE_VALID`."""

    return (
        resolve_stage_state(done_path, inputs, outputs, **kwargs) is CompletionState.COMPLETE_VALID
    )


def write_stage_done(
    done_path: Path,
    *,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    config_sha: str | None = None,
    stage_config_sha: str | None = None,
    cache_key_version: int = _DEFAULT_CACHE_KEY_VERSION,
    stage: str | None = None,
    cfg: Any | None = None,
    freshness_key: Mapping[str, Any] | None = None,
    status: str = _DEFAULT_STATUS,
    reason: str | None = None,
    blocking_dependency: str | None = None,
    upstream_stage: str | None = None,
    sidecar_artifacts: Iterable[Path] = (),
) -> None:
    """Persist lifecycle state after declared artifact sidecars validate."""

    if status not in _ALLOWED_STATUSES:
        raise ValueError(
            f"Unsupported stage status {status!r}; expected one of {_ALLOWED_STATUSES}"
        )
    if status in {"failed", "skipped"} and (blocking_dependency is None or upstream_stage is None):
        raise ValueError(
            "blocking_dependency and upstream_stage are required when status is failed/skipped"
        )
    config_sha, stage_config_sha, cache_key_version, freshness_key = _resolve_expected_contract(
        cfg=cfg,
        stage=stage,
        config_sha=config_sha,
        stage_config_sha=stage_config_sha,
        cache_key_version=cache_key_version,
        freshness_key=freshness_key,
    )

    input_paths = _coerce_paths(inputs)
    output_paths = _coerce_paths(outputs)
    for artifact in _coerce_paths(sidecar_artifacts):
        validate_artifact_sidecar(artifact)

    if status == "success":
        completion_state = CompletionState.COMPLETE_VALID
    elif status == "blocked_by_cap":
        completion_state = CompletionState.BLOCKED_BY_CAP
    else:
        completion_state = CompletionState.PARTIAL_RESUMABLE
    normalized_freshness = None if freshness_key is None else dict(freshness_key)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "stage": stage,
        "config_sha": config_sha,
        "stage_config_sha": stage_config_sha if stage_config_sha is not None else config_sha,
        "cache_key_version": cache_key_version,
        "freshness_key": normalized_freshness,
        "freshness_sha256": (
            None if normalized_freshness is None else freshness_sha256(normalized_freshness)
        ),
        "completion_state": completion_state.value,
        "inputs": [str(path) for path in input_paths],
        "input_fingerprints": [_path_fingerprint(path) for path in input_paths],
        "outputs": [str(path) for path in output_paths],
        "status": status,
        "reason": reason,
        "blocking_dependency": blocking_dependency,
        "upstream_stage": upstream_stage,
    }
    done_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(done_path)) as temporary_path:
        Path(temporary_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
