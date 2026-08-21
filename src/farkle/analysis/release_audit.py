"""Read-only authenticated-v3 release graph audits."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final

from farkle.config import AppConfig, ArtifactScope, compute_config_sha, load_app_config
from farkle.orchestration.run_contexts import load_run_context
from farkle.utils.artifact_contract import read_json_file_with_retry, sha256_file
from farkle.utils.authenticated_contract import (
    AuthenticatedSidecar,
    CanonicalArtifactLocation,
    ImmutableManifestSidecar,
    canonical_json_bytes,
    load_immutable_manifest_sidecar,
    validate_authenticated_artifact_unbound,
)
from farkle.utils.authenticated_graph import (
    AuthenticatedGraphSnapshot,
    SnapshotGeneration,
    SnapshotUseError,
)
from farkle.utils.authentication_telemetry import active_authentication_telemetry
from farkle.utils.completion_files import CompletionFileKind, CompletionNamespace

_SIDECAR_SUFFIX: Final = ".sidecar.json"
_STATE_SUFFIXES: Final = (".checkpoint.json", ".done.json")
_DERIVED_SUFFIXES: Final = {
    ".json",
    ".jsonl",
    ".md",
    ".npy",
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
        and not any(part.startswith("_") for part in relative.parts)
        and path.suffix.lower() in _DERIVED_SUFFIXES
        and (
            bool(_CANONICAL_SCOPE_PARTS.intersection(relative.parts))
            or _sidecar_path(path).is_file()
        )
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
    payload = read_json_file_with_retry(_sidecar_path(path))
    if not isinstance(payload, Mapping) or payload.get("artifact_contract_version") != 3:
        raise ValueError("release descendants must use artifact contract exactly 3")
    if "manifest_contract_version" in payload:
        manifest = load_immutable_manifest_sidecar(path)
        if manifest.location.stage_key != manifest.stage_identity.stage_key:
            raise ValueError("manifest location and stage identity disagree")
        if _global_identity(manifest.stage_identity.versions) != _ACCEPTED_GLOBAL_IDENTITY:
            raise ValueError("immutable manifest has a mixed release identity")
        if sha256_file(path) != manifest.manifest_sha256:
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
    if location.stage_key == "simulation":
        return Path(location.relative_path)
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
    payload = read_json_file_with_retry(sidecar_path)
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


def _build_audit_index(
    root: Path,
    *,
    completion_paths: Iterable[Path] | None = None,
    analysis_root: Path | None = None,
) -> tuple[_AuditIndex, list[str]]:
    """Traverse *root* once and construct its fail-closed canonical index."""

    collector = active_authentication_telemetry()
    if collector is not None:
        collector.bump("directory_traversals")
        collector.bump("graph_root_traversals")
    files = tuple(sorted(path for path in root.rglob("*") if path.is_file()))
    data_paths = tuple(path for path in files if _is_scoped_artifact(path, root))
    sidecar_paths = tuple(
        path
        for path in files
        if path.name.endswith(_SIDECAR_SUFFIX)
        and not any(part.startswith("_") for part in path.relative_to(root).parts)
    )
    explicit_completion_paths = (
        tuple(sorted(Path(path).resolve() for path in completion_paths))
        if completion_paths is not None
        else ()
    )
    resolved_analysis_root = (
        Path(analysis_root).resolve()
        if analysis_root is not None
        else ((root / "analysis").resolve() if (root / "analysis").is_dir() else root.resolve())
    )
    namespace = CompletionNamespace.build(
        graph_root=root,
        analysis_root=resolved_analysis_root,
        canonical_paths=explicit_completion_paths,
    )
    if completion_paths is None:
        resolved_completion_paths = tuple(
            path for path in files if namespace.classify(path) is CompletionFileKind.CANONICAL_STAGE
        )
    else:
        resolved_completion_paths = explicit_completion_paths
    failures: list[str] = []
    if completion_paths is not None:
        discovered_completion_paths = {
            path.resolve()
            for path in files
            if namespace.classify(path) is CompletionFileKind.CANONICAL_STAGE
        }
        expected_completion_paths = set(resolved_completion_paths)
        if discovered_completion_paths != expected_completion_paths:
            missing = sorted(expected_completion_paths - discovered_completion_paths)
            added = sorted(discovered_completion_paths - expected_completion_paths)
            if missing:
                failures.append(f"completion inventory omissions: {missing}")
            if added:
                failures.append(f"completion inventory additions: {added}")
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
            completion_paths=resolved_completion_paths,
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
            record.sidecar_sha256 = sha256_file(record.sidecar_path)
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
        payload = read_json_file_with_retry(path)
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
            expected_root = (
                path.parent.parent if canonical_location.stage_key == "simulation" else path.parent
            )
            expected_path = expected_root / _canonical_suffix(canonical_location)
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
    collector = active_authentication_telemetry()
    if collector is not None:
        collector.bump("graph_audit_invocations")
    if not root.is_dir():
        return [f"artifact root does not exist: {root}"]
    index, failures = _build_audit_index(root)
    failures.extend(_validate_audit_index(index))
    return sorted(failures)


@dataclass(frozen=True, slots=True)
class AuthenticatedReleaseAuditTarget:
    """One current context and its process-local expected graph snapshot."""

    cfg: AppConfig
    snapshot: AuthenticatedGraphSnapshot
    generation: SnapshotGeneration


def _expected_output_by_location(
    snapshot: AuthenticatedGraphSnapshot,
) -> dict[CanonicalArtifactLocation, Any]:
    return {output.location: output for output in snapshot.graph_inventory}


def _compare_snapshot_index(
    snapshot: AuthenticatedGraphSnapshot,
    index: _AuditIndex,
) -> list[str]:
    failures: list[str] = []
    expected = _expected_output_by_location(snapshot)
    observed = set(index.by_location)
    if observed != set(expected):
        missing = sorted(set(expected) - observed, key=canonical_json_bytes)
        added = sorted(observed - set(expected), key=canonical_json_bytes)
        if missing:
            failures.append(f"snapshot inventory omissions: {missing}")
        if added:
            failures.append(f"snapshot inventory additions: {added}")
    for location in sorted(set(expected).intersection(observed), key=canonical_json_bytes):
        wanted = expected[location]
        record = index.by_location[location]
        if wanted.artifact is not None:
            metadata = record.metadata
            if (
                not isinstance(metadata, AuthenticatedSidecar)
                or metadata.artifact != wanted.artifact
                or record.sidecar_sha256 != wanted.sidecar_sha256
            ):
                failures.append(f"snapshot artifact identity changed: {record.path}")
        else:
            metadata = record.metadata
            manifest = wanted.manifest
            if (
                manifest is None
                or not isinstance(metadata, ImmutableManifestSidecar)
                or metadata.manifest_sha256 != manifest.manifest_sha256
                or metadata.sidecar_contract_sha256 != manifest.sidecar_contract_sha256
                or metadata.summary != manifest.summary
                or record.sidecar_sha256 != wanted.sidecar_sha256
            ):
                failures.append(f"snapshot manifest identity changed: {record.path}")
    expected_completions = {item.path: item for item in snapshot.completions}
    if set(index.completion_paths) != set(expected_completions):
        failures.append("snapshot completion inventory changed")
    for path in sorted(set(index.completion_paths).intersection(expected_completions)):
        if not path.is_file() or sha256_file(path) != expected_completions[path].sha256:
            failures.append(f"snapshot completion identity changed: {path}")
    return failures


def _validate_cross_context_provenance(indexes: Iterable[_AuditIndex]) -> list[str]:
    records = [record for index in indexes for record in index.by_location.values()]
    failures: list[str] = []
    for record in records:
        metadata = record.metadata
        if not isinstance(metadata, AuthenticatedSidecar):
            continue
        for source in metadata.source_artifacts:
            candidates = [
                item
                for item in records
                if isinstance(item.metadata, AuthenticatedSidecar)
                and item.metadata.artifact == source.artifact
                and item.sidecar_sha256 == source.sidecar_sha256
                and item.metadata.sidecar_contract_sha256 == source.sidecar_contract_sha256
            ]
            if len(candidates) != 1:
                failures.append(
                    f"unresolved cross-context source binding: {record.path}: "
                    f"{source.logical_role}"
                )
        for manifest in metadata.manifest_roots:
            candidates = [
                item
                for item in records
                if isinstance(item.metadata, ImmutableManifestSidecar)
                and item.metadata.location == manifest.location
                and item.metadata.manifest_sha256 == manifest.manifest_sha256
                and item.sidecar_sha256 == manifest.sidecar_sha256
                and item.metadata.sidecar_contract_sha256 == manifest.sidecar_contract_sha256
                and item.metadata.summary == manifest.summary
            ]
            if len(candidates) != 1:
                failures.append(
                    f"unresolved cross-context manifest binding: {record.path}: "
                    f"{manifest.logical_role}"
                )
    return failures


def audit_authenticated_release_graphs(
    targets: Iterable[AuthenticatedReleaseAuditTarget],
    *,
    expected_code_identity: Any,
    current_code_identity: Any,
    boundary_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one top-level, fresh-byte release audit across every context graph."""

    ordered_targets = tuple(
        sorted(
            targets,
            key=lambda target: (
                0 if target.snapshot.scope == "root" else 1,
                target.snapshot.roots,
            ),
        )
    )
    collector = active_authentication_telemetry()
    if collector is not None:
        collector.bump("graph_audit_invocations")
    failures: list[str] = []
    run_contexts: dict[str, dict[str, str]] = {}
    root_metrics: list[dict[str, Any]] = []
    indexes: list[_AuditIndex] = []
    if not ordered_targets:
        failures.append("final byte-deep release audit requires authenticated graph snapshots")
    if current_code_identity != expected_code_identity:
        failures.append("repository code identity changed after snapshot construction")

    for target in ordered_targets:
        snapshot = target.snapshot
        label = "pair" if snapshot.scope == "pair" else f"root_{snapshot.roots[0]}"
        try:
            target.generation.validate(
                snapshot,
                expected_scope=snapshot.scope,
                expected_roots=snapshot.roots,
                expected_run_context_path=snapshot.run_context_path,
            )
            if (
                compute_config_sha(target.cfg) != snapshot.public_config_sha256
                or target.cfg._run_lineage_sha256 != snapshot.run_lineage_sha256
                or target.cfg._code_identity != snapshot.code_identity
            ):
                raise SnapshotUseError("current context identity differs from its snapshot")
            persisted = load_run_context(
                snapshot.run_context_path,
                active_config_path=snapshot.active_config_path,
            )
            context_sha = sha256_file(snapshot.run_context_path)
            config_sha = sha256_file(snapshot.active_config_path)
            if (
                context_sha != snapshot.run_context_sha256
                or config_sha != snapshot.active_config_sha256
                or persisted.get("run_context_sha256") != snapshot.run_context_identity_sha256
            ):
                raise SnapshotUseError("run-context or active-configuration bytes changed")
            run_contexts[label] = {
                "path": str(snapshot.run_context_path),
                "sha256": context_sha,
                "identity_sha256": snapshot.run_context_identity_sha256,
            }
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label} run context: {type(exc).__name__}: {exc}")

        index, root_failures = _build_audit_index(
            snapshot.graph_root,
            completion_paths=(item.path for item in snapshot.completions),
            analysis_root=snapshot.analysis_root,
        )
        root_failures.extend(_validate_audit_index(index))
        root_failures.extend(_compare_snapshot_index(snapshot, index))
        indexes.append(index)
        failures.extend(f"{label} authenticated graph: {failure}" for failure in root_failures)
        root_metrics.append(
            {
                "label": label,
                "root": str(snapshot.graph_root),
                "files": len(index.files),
                "artifacts": len(index.data_paths),
                "sidecars": len(index.sidecar_paths),
                "completions": len(index.completion_paths),
                "completion_lookups": index.lookup_count,
            }
        )
        if boundary_hook is not None:
            boundary_hook(f"context:{label}")

    failures.extend(_validate_cross_context_provenance(indexes))
    if boundary_hook is not None:
        boundary_hook("complete")
    return {
        "status": "passed" if not failures else "failed",
        "failures": sorted(failures),
        "run_contexts": run_contexts,
        "internal_roots": root_metrics,
        "top_level_invocations": 1,
    }


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
    "AuthenticatedReleaseAuditTarget",
    "audit_authenticated_release_graphs",
    "run_release_audits",
]
