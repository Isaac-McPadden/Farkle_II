"""Content-authenticated stage freshness and resumable lifecycle state.

The completion stamp is intentionally format-agnostic.  Artifact sidecars own
semantic schema validation; this module binds the exact bytes of every declared
input, output, and adjacent sidecar together with scoped config, method/version,
code, and run-lineage identities.  Paths and mtimes are diagnostic only.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
_SCHEMA_VERSION = 4
_DEFAULT_CACHE_KEY_VERSION = 4
_LIFECYCLE_CONTRACT_VERSION = 1


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
        "lifecycle_contract_version": None,
        "code_identity": None,
        "run_lineage_sha256": None,
        "stage_identity_sha256": None,
        "completion_state": CompletionState.NOT_STARTED.value,
        "inputs": [],
        "input_identities": [],
        "outputs": [],
        "output_identities": [],
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adjacent_sidecar(path: Path) -> Path:
    return path.with_name(f"{path.name}.sidecar.json")


def _path_content_identity(path: Path, *, logical_role: str) -> dict[str, object]:
    """Return a path-independent exact-byte identity for one file or directory."""

    if not path.exists():
        return {"logical_role": logical_role, "kind": "missing"}
    if path.is_file():
        sidecar = _adjacent_sidecar(path)
        return {
            "logical_role": logical_role,
            "kind": "file",
            "byte_length": path.stat().st_size,
            "content_sha256": _sha256_file(path),
            "sidecar_sha256": _sha256_file(sidecar) if sidecar.is_file() else None,
        }
    entries: list[dict[str, object]] = []
    for child in sorted(
        (item for item in path.rglob("*") if item.is_file()), key=lambda p: p.as_posix()
    ):
        entries.append(
            {
                "relative_path": child.relative_to(path).as_posix(),
                "byte_length": child.stat().st_size,
                "content_sha256": _sha256_file(child),
            }
        )
    return {
        "logical_role": logical_role,
        "kind": "directory",
        "entry_count": len(entries),
        "tree_sha256": freshness_sha256({"entries": entries}),
    }


def _path_identities(paths: Sequence[Path], *, prefix: str) -> list[dict[str, object]]:
    return [
        _path_content_identity(path, logical_role=f"{prefix}_{index:04d}")
        for index, path in enumerate(paths)
    ]


def _code_identity_payload(cfg: Any | None, supplied: Any | None) -> dict[str, object]:
    identity = supplied or (getattr(cfg, "_code_identity", None) if cfg is not None else None)
    if identity is None and cfg is not None:
        from farkle.utils.authenticated_contract import (
            CodeIdentityPolicy,
            resolve_code_identity,
        )

        repo_root = Path(__file__).resolve().parents[3]
        identity = resolve_code_identity(repo_root, policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY)
    if identity is None:
        return {"state": "unscoped_test", "commit": None, "dirty_fingerprint_sha256": None}
    if hasattr(identity, "commit"):
        return {
            "commit": str(identity.commit),
            "policy": str(identity.policy),
            "state": str(identity.state),
            "dirty_fingerprint_sha256": identity.dirty_fingerprint_sha256,
        }
    if not isinstance(identity, Mapping):
        raise TypeError("code_identity must be a CodeIdentity or mapping")
    return dict(identity)


def _stage_identity_sha256(
    *,
    stage: str | None,
    stage_config_sha: str | None,
    cache_key_version: int,
    freshness_key: Mapping[str, Any] | None,
    code_identity: Mapping[str, object],
    run_lineage_sha256: str | None,
    input_identities: Sequence[Mapping[str, object]],
) -> str:
    return freshness_sha256(
        {
            "lifecycle_contract_version": _LIFECYCLE_CONTRACT_VERSION,
            "stage_key": stage,
            "stage_cache_key_version": cache_key_version,
            "stage_config_identity": stage_config_sha,
            "method_versions": dict(freshness_key or {}),
            "code_identity": dict(code_identity),
            "run_lineage_sha256": run_lineage_sha256,
            "upstream_identities": list(input_identities),
        }
    )


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
    code_identity: Any | None = None,
    run_lineage_sha256: str | None = None,
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

    if metadata.get("lifecycle_contract_version") != _LIFECYCLE_CONTRACT_VERSION:
        return CompletionState.COMPLETE_STALE
    recorded_outputs = metadata.get("outputs")
    if isinstance(recorded_outputs, list):
        recorded_output_paths = [Path(str(value)) for value in recorded_outputs]
        if output_paths and not set(output_paths).issubset(recorded_output_paths):
            return CompletionState.COMPLETE_STALE
        output_paths = recorded_output_paths
    current_inputs = _path_identities(input_paths, prefix="input")
    current_outputs = _path_identities(output_paths, prefix="output")
    if metadata.get("input_identities") != current_inputs:
        return CompletionState.COMPLETE_STALE
    if metadata.get("output_identities") != current_outputs:
        return CompletionState.COMPLETE_STALE
    resolved_code = _code_identity_payload(cfg, code_identity)
    if metadata.get("code_identity") != resolved_code:
        return CompletionState.COMPLETE_STALE
    resolved_lineage = run_lineage_sha256 or (
        getattr(cfg, "_run_lineage_sha256", None) if cfg is not None else None
    )
    if metadata.get("run_lineage_sha256") != resolved_lineage:
        return CompletionState.COMPLETE_STALE
    expected_identity = _stage_identity_sha256(
        stage=stage,
        stage_config_sha=expected_stage_sha,
        cache_key_version=cache_key_version,
        freshness_key=freshness_key,
        code_identity=resolved_code,
        run_lineage_sha256=resolved_lineage,
        input_identities=current_inputs,
    )
    if metadata.get("stage_identity_sha256") != expected_identity:
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
    code_identity: Any | None = None,
    run_lineage_sha256: str | None = None,
    metadata: Mapping[str, Any] | None = None,
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
    if status == "success":
        missing_inputs = [path for path in input_paths if not path.exists()]
        missing_outputs = [path for path in output_paths if not path.exists()]
        if missing_inputs or missing_outputs:
            raise FileNotFoundError(
                "cannot publish successful completion with missing paths: "
                f"inputs={missing_inputs}, outputs={missing_outputs}"
            )
    for artifact in _coerce_paths(sidecar_artifacts):
        validate_artifact_sidecar(artifact)

    if status == "success":
        completion_state = CompletionState.COMPLETE_VALID
    elif status == "blocked_by_cap":
        completion_state = CompletionState.BLOCKED_BY_CAP
    else:
        completion_state = CompletionState.PARTIAL_RESUMABLE
    normalized_freshness = None if freshness_key is None else dict(freshness_key)
    input_identities = _path_identities(input_paths, prefix="input")
    output_identities = _path_identities(output_paths, prefix="output")
    resolved_code = _code_identity_payload(cfg, code_identity)
    resolved_lineage = run_lineage_sha256 or (
        getattr(cfg, "_run_lineage_sha256", None) if cfg is not None else None
    )
    resolved_stage_sha = stage_config_sha if stage_config_sha is not None else config_sha
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "lifecycle_contract_version": _LIFECYCLE_CONTRACT_VERSION,
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
        "input_identities": input_identities,
        "outputs": [str(path) for path in output_paths],
        "output_identities": output_identities,
        "code_identity": resolved_code,
        "run_lineage_sha256": resolved_lineage,
        "stage_identity_sha256": _stage_identity_sha256(
            stage=stage,
            stage_config_sha=resolved_stage_sha,
            cache_key_version=cache_key_version,
            freshness_key=normalized_freshness,
            code_identity=resolved_code,
            run_lineage_sha256=resolved_lineage,
            input_identities=input_identities,
        ),
        "status": status,
        "reason": reason,
        "blocking_dependency": blocking_dependency,
        "upstream_stage": upstream_stage,
    }
    if metadata:
        collisions = set(payload).intersection(metadata)
        if collisions:
            raise ValueError(
                f"completion metadata collides with reserved fields: {sorted(collisions)}"
            )
        payload.update(dict(metadata))
    done_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(done_path)) as temporary_path:
        Path(temporary_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
