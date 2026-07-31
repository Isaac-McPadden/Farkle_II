"""Read-only authenticated-v3 release graph audits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

from farkle.config import ArtifactScope, load_app_config
from farkle.utils.authenticated_contract import (
    canonical_json_bytes,
    load_immutable_manifest_sidecar,
    validate_authenticated_artifact_unbound,
)

_SIDECAR_SUFFIX: Final = ".sidecar.json"
_STATE_SUFFIXES: Final = (".checkpoint.json", ".done.json")
_DERIVED_SUFFIXES: Final = {
    ".json",
    ".jsonl",
    ".md",
    ".parquet",
    ".pkl",
    ".png",
    ".txt",
    ".yaml",
    ".yml",
}
_RETIRED_ENTRY_POINTS: Final = (
    "src/farkle/analysis/agreement.py",
    "src/farkle/analysis/coverage_by_k.py",
    "src/farkle/analysis/interseed_analysis.py",
    "src/farkle/analysis/meta.py",
    "src/farkle/analysis/reporting.py",
    "src/farkle/analysis/variance.py",
    "src/farkle/orchestration/pipeline.py",
    "src/farkle/utils/stage_io.py",
    "src/farkle/utils/tiers.py",
    "src/pipeline.py",
)
_CANONICAL_SCOPE_PARTS: Final = frozenset(scope.value for scope in ArtifactScope)
_ACCEPTED_GLOBAL_IDENTITY: Final = (3, 2, 2, 2, 2, 2)


def _sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.name}{_SIDECAR_SUFFIX}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_runnable_configs(config_paths: Iterable[Path]) -> list[str]:
    """Return failures from loading and validating runnable config files."""

    failures: list[str] = []
    for path in sorted(Path(item) for item in config_paths):
        try:
            cfg = load_app_config(path)
            cfg.validate_statistical_contract(require_two_roots=False)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
    return failures


def audit_retired_entry_points(repository_root: Path) -> list[str]:
    """Return retired source entry points that still exist after migration."""

    root = repository_root.resolve()
    return [relative for relative in _RETIRED_ENTRY_POINTS if (root / relative).exists()]


def _is_scoped_artifact(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        path.is_file()
        and not path.name.endswith(_SIDECAR_SUFFIX)
        and not path.name.endswith(_STATE_SUFFIXES)
        and not path.name.startswith("._")
        and path.suffix.lower() in _DERIVED_SUFFIXES
        and bool(_CANONICAL_SCOPE_PARTS.intersection(relative.parts))
    )


def _global_identity(versions: Any) -> tuple[int, int, int, int, int, int]:
    return (
        int(versions.artifact_contract_version),
        int(versions.rng_scheme_version),
        int(versions.outcome_schema_version),
        int(versions.schema_version),
        int(versions.estimand_version),
        int(versions.conditioning_version),
    )


def _validate_v3_sidecar(path: Path) -> None:
    payload = json.loads(_sidecar_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("artifact_contract_version") != 3:
        raise ValueError("release descendants must use artifact contract exactly 3")
    if "manifest_contract_version" in payload:
        manifest = load_immutable_manifest_sidecar(path)
        if _global_identity(manifest.stage_identity.versions) != _ACCEPTED_GLOBAL_IDENTITY:
            raise ValueError("immutable manifest has a mixed release identity")
        if _sha256_file(path) != manifest.manifest_sha256:
            raise ValueError("immutable manifest bytes do not match its sidecar")
        return
    metadata = validate_authenticated_artifact_unbound(path, validate_provenance=False)
    if _global_identity(metadata.versions) != _ACCEPTED_GLOBAL_IDENTITY:
        raise ValueError("artifact has a mixed release identity")
    if not metadata.versions.method_versions:
        raise ValueError("artifact is missing applicable named method versions")


def _completion_artifact_path(
    root: Path,
    location: Mapping[str, Any],
) -> Path | None:
    relative = Path(str(location["relative_path"]))
    scope = str(location["scope"])
    suffix = (
        Path(scope) / f"{int(location['player_count'])}p" / relative
        if scope == ArtifactScope.BY_K.value
        else Path(scope) / relative
    )
    matches = [
        path
        for path in root.rglob(relative.name)
        if path.is_file() and tuple(path.parts[-len(suffix.parts) :]) == suffix.parts
    ]
    return matches[0] if len(matches) == 1 else None


def _audit_completion(path: Path, root: Path) -> list[str]:
    failures: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, Mapping)
            or payload.get("lifecycle_contract_version") != 1
            or payload.get("state") != "complete_valid"
            or not isinstance(payload.get("outputs"), list)
            or not payload["outputs"]
        ):
            raise ValueError("stage completion is not authenticated-v3 complete_valid")
        locations: list[Mapping[str, Any]] = []
        for output in payload["outputs"]:
            identity = output.get("artifact") or output.get("manifest")
            location = identity["location"]
            locations.append(location)
            artifact = _completion_artifact_path(root, location)
            if artifact is None:
                raise ValueError(f"completion output cannot be resolved uniquely: {location}")
            if _sha256_file(_sidecar_path(artifact)) != output["sidecar_sha256"]:
                raise ValueError(f"completion sidecar identity is stale: {artifact}")
            _validate_v3_sidecar(artifact)
        if locations != sorted(locations, key=canonical_json_bytes):
            raise ValueError("completion inventory is not in canonical order")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"incompatible completion: {path}: {exc}")
    return failures


def audit_sidecar_completeness(audit_root: Path) -> list[str]:
    """Audit an explicit artifact root as a complete authenticated-v3 graph."""

    root = audit_root.resolve()
    if not root.is_dir():
        return [f"artifact root does not exist: {root}"]
    failures: list[str] = []
    data_paths = sorted(path for path in root.rglob("*") if _is_scoped_artifact(path, root))
    observed_sidecars = sorted(root.rglob(f"*{_SIDECAR_SUFFIX}"))
    expected_sidecars = {_sidecar_path(path) for path in data_paths}
    for path in data_paths:
        adjacent = _sidecar_path(path)
        if not adjacent.is_file():
            failures.append(f"missing sidecar: {path}")
            continue
        try:
            _validate_v3_sidecar(path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"incompatible sidecar: {path}: {exc}")
    for adjacent in observed_sidecars:
        artifact = Path(str(adjacent)[: -len(_SIDECAR_SUFFIX)])
        if adjacent not in expected_sidecars or not artifact.is_file():
            failures.append(f"orphan sidecar: {adjacent}")
    for stage_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for completion in sorted(stage_dir.glob("*.done.json")):
            failures.extend(_audit_completion(completion, root))
    return sorted(failures)


def run_release_audits(
    repository_root: Path,
    *,
    config_paths: Iterable[Path],
    artifact_roots: Iterable[Path] = (),
) -> list[str]:
    """Return every release-audit failure in deterministic order."""

    failures = audit_runnable_configs(config_paths)
    failures.extend(audit_retired_entry_points(repository_root))
    roots = tuple(Path(path) for path in artifact_roots)
    if not roots:
        failures.append("release audit requires at least one explicit fresh artifact root")
    for artifact_root in roots:
        failures.extend(audit_sidecar_completeness(artifact_root))
    return sorted(failures)


__all__ = [
    "audit_retired_entry_points",
    "audit_runnable_configs",
    "audit_sidecar_completeness",
    "run_release_audits",
]
