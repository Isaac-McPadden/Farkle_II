"""Read-only authenticated-v3 release graph audits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from farkle.config import ArtifactScope, load_app_config
from farkle.utils.authenticated_contract import (
    AuthenticatedSidecar,
    CanonicalArtifactLocation,
    ImmutableManifestSidecar,
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


@dataclass(slots=True)
class _IndexedArtifact:
    path: Path
    sidecar_path: Path
    location: CanonicalArtifactLocation
    raw_identity: Mapping[str, Any]
    is_manifest: bool
    metadata: AuthenticatedSidecar | ImmutableManifestSidecar | None = None
    sidecar_sha256: str | None = None


@dataclass(slots=True)
class _AuditIndex:
    root: Path
    files: tuple[Path, ...]
    data_paths: tuple[Path, ...]
    sidecar_paths: tuple[Path, ...]
    completion_paths: tuple[Path, ...]
    by_location: dict[CanonicalArtifactLocation, _IndexedArtifact]
    by_path: dict[Path, _IndexedArtifact]
    invalid_paths: set[Path] = field(default_factory=set)
    lookup_count: int = 0

    def resolve(self, location: CanonicalArtifactLocation) -> _IndexedArtifact | None:
        """Resolve one already-canonical location without touching the filesystem."""

        self.lookup_count += 1
        return self.by_location.get(location)


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


def _validate_v3_sidecar(
    path: Path,
) -> AuthenticatedSidecar | ImmutableManifestSidecar:
    payload = json.loads(_sidecar_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("artifact_contract_version") != 3:
        raise ValueError("release descendants must use artifact contract exactly 3")
    if "manifest_contract_version" in payload:
        manifest = load_immutable_manifest_sidecar(path)
        if manifest.location.stage_key != manifest.stage_identity.stage_key:
            raise ValueError("manifest location and stage identity disagree")
        if _global_identity(manifest.stage_identity.versions) != _ACCEPTED_GLOBAL_IDENTITY:
            raise ValueError("immutable manifest has a mixed release identity")
        if _sha256_file(path) != manifest.manifest_sha256:
            raise ValueError("immutable manifest bytes do not match its sidecar")
        return manifest
    metadata = validate_authenticated_artifact_unbound(path, validate_provenance=False)
    if metadata.artifact.location.stage_key != metadata.stage_identity.stage_key:
        raise ValueError("artifact location and stage identity disagree")
    if _global_identity(metadata.versions) != _ACCEPTED_GLOBAL_IDENTITY:
        raise ValueError("artifact has a mixed release identity")
    if not metadata.versions.method_versions:
        raise ValueError("artifact is missing applicable named method versions")
    return metadata


def _canonical_location(value: Any) -> CanonicalArtifactLocation:
    if not isinstance(value, Mapping):
        raise TypeError("canonical location must be an object")
    raw = dict(value)
    location = CanonicalArtifactLocation(**raw)
    relative = Path(location.relative_path)
    if (
        not isinstance(raw.get("relative_path"), str)
        or location.relative_path in {"", "."}
        or relative.drive
        or location.relative_path != relative.as_posix()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("relative_path must use one unambiguous canonical spelling")
    return location


def _canonical_suffix(location: CanonicalArtifactLocation) -> Path:
    parts = [location.scope]
    if location.scope == ArtifactScope.BY_K.value:
        assert location.player_count is not None
        parts.append(f"{location.player_count}p")
    return Path(*parts) / location.relative_path


def _require_canonical_suffix(path: Path, location: CanonicalArtifactLocation) -> None:
    suffix = _canonical_suffix(location)
    if tuple(path.parts[-len(suffix.parts) :]) != suffix.parts:
        raise ValueError(f"artifact does not realize canonical scope/k/path {location}: {path}")


def _declared_index_identity(
    sidecar_path: Path,
) -> tuple[CanonicalArtifactLocation, Mapping[str, Any], bool]:
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("artifact_contract_version") != 3:
        raise ValueError("release descendants must use artifact contract exactly 3")
    is_manifest = "manifest_contract_version" in payload
    if is_manifest:
        location = _canonical_location(payload.get("location"))
        raw_identity: Mapping[str, Any] = payload
    else:
        artifact = payload.get("artifact")
        if not isinstance(artifact, Mapping):
            raise TypeError("authenticated sidecar is missing its artifact identity")
        location = _canonical_location(artifact.get("location"))
        raw_identity = artifact
    return location, raw_identity, is_manifest


def _build_audit_index(root: Path) -> tuple[_AuditIndex, list[str]]:
    """Traverse *root* once and construct its fail-closed canonical index."""

    files = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
    data_paths = tuple(path for path in files if _is_scoped_artifact(path, root))
    sidecar_paths = tuple(path for path in files if path.name.endswith(_SIDECAR_SUFFIX))
    completion_paths = tuple(
        path for path in files if path.name.endswith(".done.json") and path.parent.parent == root
    )
    failures: list[str] = []
    by_location: dict[CanonicalArtifactLocation, _IndexedArtifact] = {}
    by_path: dict[Path, _IndexedArtifact] = {}
    invalid_paths: set[Path] = set()
    for path in data_paths:
        adjacent = _sidecar_path(path)
        if not adjacent.is_file():
            failures.append(f"missing sidecar: {path}")
            invalid_paths.add(path)
            continue
        try:
            location, raw_identity, is_manifest = _declared_index_identity(adjacent)
            _require_canonical_suffix(path, location)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"incompatible sidecar: {path}: {exc}")
            invalid_paths.add(path)
            continue
        record = _IndexedArtifact(
            path=path,
            sidecar_path=adjacent,
            location=location,
            raw_identity=raw_identity,
            is_manifest=is_manifest,
        )
        previous = by_location.get(location)
        if previous is not None:
            failures.append(
                "duplicate canonical artifact location: " f"{location}: {previous.path} and {path}"
            )
            invalid_paths.update((previous.path, path))
            continue
        by_location[location] = record
        by_path[path] = record
    expected_sidecars = {_sidecar_path(path) for path in data_paths}
    for adjacent in sidecar_paths:
        artifact = Path(str(adjacent)[: -len(_SIDECAR_SUFFIX)])
        if adjacent not in expected_sidecars or not artifact.is_file():
            failures.append(f"orphan sidecar: {adjacent}")
    return (
        _AuditIndex(
            root=root,
            files=files,
            data_paths=data_paths,
            sidecar_paths=sidecar_paths,
            completion_paths=completion_paths,
            by_location=by_location,
            by_path=by_path,
            invalid_paths=invalid_paths,
        ),
        failures,
    )


def _validate_indexed_artifacts(index: _AuditIndex) -> list[str]:
    failures: list[str] = []
    for path in index.data_paths:
        if path in index.invalid_paths:
            continue
        record = index.by_path[path]
        try:
            metadata = _validate_v3_sidecar(path)
            if (isinstance(metadata, ImmutableManifestSidecar)) != record.is_manifest:
                raise ValueError("indexed artifact kind changed during validation")
            location = (
                metadata.location
                if isinstance(metadata, ImmutableManifestSidecar)
                else metadata.artifact.location
            )
            if location != record.location:
                raise ValueError("indexed canonical location changed during validation")
            record.metadata = metadata
            record.sidecar_sha256 = _sha256_file(record.sidecar_path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"incompatible sidecar: {path}: {exc}")
            index.invalid_paths.add(path)

    # Authenticate exact in-root source and manifest bindings with O(1) index
    # resolution. Cross-root sources remain authenticated by their immutable
    # identities and by the separately audited owning root.
    for record in index.by_location.values():
        record_metadata = record.metadata
        if not isinstance(record_metadata, AuthenticatedSidecar):
            continue
        try:
            for source in record_metadata.source_artifacts:
                source_record = index.by_location.get(source.artifact.location)
                if source_record is None:
                    continue
                source_metadata = source_record.metadata
                if (
                    not isinstance(source_metadata, AuthenticatedSidecar)
                    or source_metadata.artifact != source.artifact
                    or source_record.sidecar_sha256 != source.sidecar_sha256
                    or source_metadata.sidecar_contract_sha256 != source.sidecar_contract_sha256
                ):
                    raise ValueError(f"source artifact identity changed: {source.logical_role}")
            for manifest in record_metadata.manifest_roots:
                manifest_record = index.by_location.get(manifest.location)
                if manifest_record is None:
                    continue
                manifest_metadata = manifest_record.metadata
                if (
                    not isinstance(manifest_metadata, ImmutableManifestSidecar)
                    or manifest_metadata.manifest_sha256 != manifest.manifest_sha256
                    or manifest_record.sidecar_sha256 != manifest.sidecar_sha256
                    or manifest_metadata.sidecar_contract_sha256 != manifest.sidecar_contract_sha256
                    or manifest_metadata.summary != manifest.summary
                ):
                    raise ValueError(f"manifest root identity changed: {manifest.logical_role}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"incompatible provenance: {record.path}: {exc}")
    return failures


def _audit_completion(path: Path, index: _AuditIndex) -> list[str]:
    failures: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, Mapping)
            or set(payload)
            != {
                "lifecycle_contract_version",
                "stage_identity_sha256",
                "state",
                "outputs",
            }
            or payload.get("lifecycle_contract_version") != 1
            or payload.get("state") != "complete_valid"
            or not isinstance(payload.get("stage_identity_sha256"), str)
            or len(payload["stage_identity_sha256"]) != 64
            or not isinstance(payload.get("outputs"), list)
            or not payload["outputs"]
        ):
            raise ValueError("stage completion is not authenticated-v3 complete_valid")
        locations: list[Mapping[str, Any]] = []
        canonical_locations: list[CanonicalArtifactLocation] = []
        for output in payload["outputs"]:
            if not isinstance(output, Mapping) or set(output) != {
                "artifact",
                "manifest",
                "sidecar_sha256",
            }:
                raise TypeError("completion output must be an object")
            artifact_identity = output.get("artifact")
            manifest_identity = output.get("manifest")
            if (artifact_identity is None) == (manifest_identity is None):
                raise TypeError("completion output requires exactly one identity")
            identity = artifact_identity or manifest_identity
            if not isinstance(identity, Mapping):
                raise TypeError("completion output requires exactly one identity")
            location = identity["location"]
            locations.append(location)
            canonical_location = _canonical_location(location)
            canonical_locations.append(canonical_location)
            record = index.resolve(canonical_location)
            if record is None:
                raise ValueError(f"completion output is missing: {location}")
            expected_path = path.parent / _canonical_suffix(canonical_location)
            if record.path.resolve() != expected_path.resolve():
                raise ValueError(
                    "completion output is outside its canonical stage layout: " f"{record.path}"
                )
            if record.metadata is None or record.sidecar_sha256 is None:
                raise ValueError(f"completion output failed artifact validation: {record.path}")
            if record.sidecar_sha256 != output.get("sidecar_sha256"):
                raise ValueError(f"completion sidecar identity is stale: {record.path}")
            if output.get("artifact") is not None:
                if record.is_manifest or identity != record.raw_identity:
                    raise ValueError(f"completion artifact identity is stale: {record.path}")
            else:
                if not record.is_manifest:
                    raise ValueError(f"completion manifest identity is stale: {record.path}")
                manifest = record.metadata
                assert isinstance(manifest, ImmutableManifestSidecar)
                if (
                    identity.get("location") != record.raw_identity.get("location")
                    or identity.get("manifest_sha256") != manifest.manifest_sha256
                    or identity.get("sidecar_sha256") != record.sidecar_sha256
                    or identity.get("sidecar_contract_sha256") != manifest.sidecar_contract_sha256
                    or identity.get("summary") != record.raw_identity.get("summary")
                ):
                    raise ValueError(f"completion manifest identity is stale: {record.path}")
        if locations != sorted(locations, key=canonical_json_bytes):
            raise ValueError("completion inventory is not in canonical order")
        if len(canonical_locations) != len(set(canonical_locations)):
            raise ValueError("completion inventory has duplicate canonical locations")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"incompatible completion: {path}: {exc}")
    return failures


def _validate_audit_index(index: _AuditIndex) -> list[str]:
    failures = _validate_indexed_artifacts(index)
    for completion in index.completion_paths:
        failures.extend(_audit_completion(completion, index))
    return failures


def audit_sidecar_completeness(audit_root: Path) -> list[str]:
    """Audit an explicit artifact root as a complete authenticated-v3 graph."""

    root = audit_root.resolve()
    if not root.is_dir():
        return [f"artifact root does not exist: {root}"]
    index, failures = _build_audit_index(root)
    failures.extend(_validate_audit_index(index))
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
