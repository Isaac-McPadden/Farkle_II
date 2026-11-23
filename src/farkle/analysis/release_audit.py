"""Read-only authenticated-v3 release graph audits."""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Callable, Final

from farkle.config import AppConfig, ArtifactScope, compute_config_sha, load_app_config
from farkle.orchestration.execution_history import (
    audit_invocation,
    audit_parent_lifecycle,
    audit_published_launch,
    invocation_dir,
)
from farkle.orchestration.run_contexts import load_run_context, resume_namespace, run_lineage_digest
from farkle.utils.artifact_contract import read_json_file_with_retry, sha256_file
from farkle.utils.authenticated_contract import (
    AuthenticatedSidecar,
    CanonicalArtifactLocation,
    CodeIdentity,
    CodeIdentityPolicy,
    ImmutableManifestSidecar,
    ManifestEntry,
    canonical_json_bytes,
    compute_manifest_root,
    identity_sha256,
    load_immutable_manifest_sidecar,
    resolve_code_identity,
    validate_authenticated_artifact_unbound,
)
from farkle.utils.authenticated_graph import (
    AuthenticatedGraphSnapshot,
    SnapshotGeneration,
    SnapshotUseError,
    excluded_graph_path,
)
from farkle.utils.authentication_telemetry import active_authentication_telemetry
from farkle.utils.completion_files import CompletionFileKind, CompletionNamespace
from farkle.utils.manifest import iter_manifest

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
REVISION_RELEASE_POLICY_VERSION: Final = 1


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
    completion_producers: dict[Path, CodeIdentity] = field(default_factory=dict)

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
        and not excluded_graph_path(path, root)
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
        if path.name.endswith(_SIDECAR_SUFFIX) and not excluded_graph_path(path, root)
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


def _audit_frozen_mapping(design: Mapping[str, Any], path: Path, digest: str) -> None:
    from farkle.utils.frozen_h2h import read_authenticated_mapping

    mapping = design["strategy_mapping"]
    ids = [row["strategy_id"] for row in mapping]
    if (
        digest != design.get("strategy_artifact_sha256")
        or read_authenticated_mapping(path, digest, ids) != mapping
    ):
        raise ValueError("frozen mapping differs from the audited strategy manifest")


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
            if location.stage_key == "h2h_power" and path.name == "power_plan.json":
                from farkle.utils.frozen_h2h import (
                    BUILTIN_ENGINE,
                    FROZEN_DESIGN_VERSION,
                )

                plan = read_json_file_with_retry(path)
                if "frozen_design" in plan:
                    design = plan["frozen_design"]
                    if not isinstance(design, dict) or (
                        design.get("version"),
                        design.get("engine"),
                    ) != (FROZEN_DESIGN_VERSION, BUILTIN_ENGINE):
                        failures.append(f"H2H engine is ineligible for scientific release: {path}")
                    if isinstance(design, dict):
                        assert isinstance(metadata, AuthenticatedSidecar)
                        strategy_sources = [
                            source
                            for source in metadata.source_artifacts
                            if source.artifact.location.stage_key == "simulation"
                            and source.artifact.location.relative_path
                            == "strategy_manifest.parquet"
                        ]
                        if len(strategy_sources) != 1 or (
                            strategy_sources[0].artifact.content_sha256
                            != design.get("strategy_artifact_sha256")
                        ):
                            raise ValueError(
                                "frozen strategy design lacks its exact strategy source"
                            )
                        source = strategy_sources[0]
                        strategy = index.resolve(source.artifact.location)
                        if strategy is not None:
                            _audit_frozen_mapping(
                                design, strategy.path, source.artifact.content_sha256
                            )
            if location.stage_key == "h2h_execute" and path.name == "execution_state.json":
                from farkle.analysis.h2h_authorization import (
                    validate_authorization,
                    validate_authorization_sources,
                )

                # Resolve the plan through the existing canonical provenance index.
                assert isinstance(metadata, AuthenticatedSidecar)
                power_source = next(
                    source
                    for source in metadata.source_artifacts
                    if source.artifact.location.stage_key == "h2h_power"
                    and source.artifact.location.relative_path == "power_plan.json"
                )
                manifest_source = next(
                    source
                    for source in metadata.source_artifacts
                    if source.artifact.location.stage_key == "h2h_power"
                    and source.artifact.location.relative_path == "block_manifest.parquet"
                )
                power_record = index.resolve(power_source.artifact.location)
                manifest_record = index.resolve(manifest_source.artifact.location)
                authorization_path = path.with_name("cap_authorization.json")
                if (
                    power_record is None
                    or manifest_record is None
                    or authorization_path not in index.by_path
                ):
                    raise ValueError("H2H execution is missing its plan or cap authorization")
                authorization = read_json_file_with_retry(authorization_path)
                plan = read_json_file_with_retry(power_record.path)
                state = read_json_file_with_retry(path)
                if state.get("cap_authorization_sha256") != sha256_file(authorization_path):
                    raise ValueError("H2H execution references a different cap authorization")
                validate_authorization(authorization, plan, require_ready=True)
                validate_authorization_sources(
                    authorization, power_record.path, manifest_record.path
                )
                for field in (
                    "family_hash",
                    "schedule_hash",
                    "maximum_total_attempts",
                    "total_block_count",
                    "authorized_total_game_cap",
                    "execution_authorization",
                ):
                    if state.get(field) != authorization.get(field):
                        raise ValueError(f"H2H execution disagrees with cap authorization: {field}")
            if path.name == "publication.json" and path.parent.name == "preflight":
                from farkle.analysis.rng_monitor import validate_preflight_inventory

                if not validate_preflight_inventory(path):
                    raise ValueError("required preflight inventory is incomplete or corrupt")
            if path.name == "analysis_guide.json":
                from farkle.analysis.analysis_guide import render_guide_markdown, validate_guide

                guide = read_json_file_with_retry(path)
                validate_guide(guide)
                markdown_path = path.with_suffix(".md")
                if markdown_path.read_text(encoding="utf-8") != render_guide_markdown(guide):
                    raise ValueError("analysis guide Markdown differs from its JSON rendering")
            if path.name == "publication.json" and path.parent.name == "_seat_behavior_units":
                from farkle.analysis.seat_behavior import validate_behavior_checkpoints

                if not validate_behavior_checkpoints(path):
                    raise ValueError("seat behavior checkpoint inventory is incomplete or corrupt")
            if path.name == "publication.json" and path.parent.name == "_outcome_units":
                from farkle.analysis.outcome_publication import validate_outcome_checkpoints

                if not validate_outcome_checkpoints(path):
                    raise ValueError("outcome checkpoint inventory is incomplete or corrupt")
            if path.name == "publication.json" and path.parent.name == "_outcome_precision_ranges":
                from farkle.analysis.outcome_precision import validate_precision_checkpoints

                if not validate_precision_checkpoints(path):
                    raise ValueError(
                        "outcome precision checkpoint inventory is incomplete or corrupt"
                    )
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
            or set(payload) - {"completion_producer_v1"}
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
        if "completion_producer_v1" in payload:
            index.completion_producers[path] = CodeIdentity(**payload["completion_producer_v1"])
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
        if any(location.stage_key == "reporting" for location in canonical_locations):
            reporting_names = {
                location.relative_path
                for location in canonical_locations
                if location.stage_key == "reporting"
            }
            if not {"analysis_guide.json", "analysis_guide.md"}.issubset(reporting_names):
                raise ValueError("reporting completion omits required analysis guide products")
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
            elif (
                record.location.stage_key == "h2h_power"
                and record.path.name == "power_plan.json"
                and source.artifact.location.stage_key == "simulation"
                and source.artifact.location.relative_path == "strategy_manifest.parquet"
            ):
                try:
                    plan = read_json_file_with_retry(record.path)
                    if "frozen_design" in plan:
                        _audit_frozen_mapping(
                            plan["frozen_design"],
                            candidates[0].path,
                            source.artifact.content_sha256,
                        )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    failures.append(
                        f"invalid cross-root frozen strategy mapping: {record.path}: {exc}"
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


def _foreign_producers(index: _AuditIndex, expected: CodeIdentity) -> set[str]:
    paths = {
        str(record.path)
        for record in index.by_path.values()
        if record.metadata is not None and record.metadata.stage_identity.code != expected
    }
    paths.update(str(path) for path, code in index.completion_producers.items() if code != expected)
    return paths


@dataclass(frozen=True)
class _ProducerLaunch:
    context: dict[str, Any]
    receipt: dict[str, Any]
    event: Mapping[str, Any]
    code: CodeIdentity

    def confirms(self, name: str, digest: str) -> bool:
        return self.receipt["scientific_launch_v1"]["producers"].get(name) == digest


def _producer_launches(
    target: AuthenticatedReleaseAuditTarget, events: list[dict[str, Any]]
) -> list[_ProducerLaunch]:
    root = target.snapshot.run_context_path.parent
    namespace = resume_namespace(target.cfg, root)
    launches: list[_ProducerLaunch] = []
    seen = set()
    for event in events:
        if (
            Path(event["invocation_record"]).resolve()
            != (invocation_dir(root, event["run_id"]) / "invocation.json").resolve()
        ):
            continue
        context, receipt = audit_published_launch(root, event)
        run = receipt["run_id"]
        if run in seen or context.get("reuse_namespace_v1") != namespace:
            raise ValueError(
                f"duplicate invocation or incompatible retained namespace at {root}: {run}"
            )
        if launches and receipt["previous"] != launches[-1].receipt["current"]:
            raise ValueError(f"disconnected retained invocation history at {root}: {run}")
        if not launches and receipt["previous"]["run_context.json"] is not None:
            raise ValueError(f"missing original invocation before {run} at {root}")
        code = CodeIdentity(**context["code_identity"])
        cfg = target.cfg
        roots = list(cfg.sim.seed_list or ()) if target.snapshot.scope == "pair" else [cfg.sim.seed]
        if (
            code.policy != CodeIdentityPolicy.RELEASE_CLEAN.value
            or code.state != "clean"
            or not receipt.get("scientific_launch_v1", {}).get("producers")
            or context["resolved_stage_layout"] != cfg.stage_layout.to_resolved_layout()
            or context["run_lineage_sha256"]
            != run_lineage_digest(
                roots,
                tuple(context["parent_lifecycle_roots"]),
                context["resolved_stage_layout"],
                code,
                cfg._game_profile_sha256,
            )
        ):
            raise ValueError(f"missing clean producer launch at {root}: {run}")
        seen.add(run)
        launches.append(_ProducerLaunch(context, receipt, event, code))
    if not launches:
        raise ValueError(f"missing retained producer launches at {root}")
    history = invocation_dir(root, launches[-1].receipt["run_id"]).parent
    if seen != {entry.name for entry in history.iterdir()}:
        raise ValueError(
            f"pipeline manifest does not account for the retained invocations at {root}"
        )
    return launches


def _manifest_member_dependencies(
    cfg: AppConfig, record: _IndexedArtifact, index: _AuditIndex, code: CodeIdentity
) -> dict[str, str]:
    """Verify native coordinates against the independently byte-checked shard index."""
    from farkle.simulation.runner import _resolve_metric_chunk_dir
    from farkle.utils.release_identity import (
        native_manifest_contract,
        publication_science_digest,
        scientific_artifact_binding,
    )
    from farkle.utils.scientific_reuse import closure

    manifest = record.metadata
    assert isinstance(manifest, ImmutableManifestSidecar)
    inventories = [
        (k, operation)
        for k in cfg.sim.n_players_list
        for folder, filename, operation in (
            (cfg.simulation_row_dir(k), "manifest.jsonl", "publish_simulation_row_shard"),
            (
                _resolve_metric_chunk_dir(cfg, k),
                "metrics_manifest.jsonl",
                "publish_simulation_metric_chunk",
            ),
        )
        if folder is not None and (folder / filename).resolve() == record.path
    ]
    if record.location.stage_key != "simulation" or len(inventories) != 1:
        raise ValueError(f"unclassified manifest member contract at {record.path}")
    players, procedure = inventories[0]
    stage = manifest.stage_identity
    designs = stage.immutable_design_identities
    _, versions, method = native_manifest_contract(
        record.path,
        cfg=cfg,
        stage_key="simulation",
        player_counts=[players],
        source_paths=[],
        operation="publish_simulation_shard_manifest",
    )
    if (
        stage.upstream_identity_sha256
        or versions != stage.versions
        or method.sha256 != stage.method_contract_sha256
        or publication_science_digest(
            "simulation",
            designs["stage_dependencies_v1"],
            method,
            versions,
            designs,
            {},
            has_sources=False,
            has_manifests=False,
        )
        != designs.get("publication_science_v1")
    ):
        raise ValueError(f"incompatible native manifest producer at {record.path}")
    inherited: dict[str, str] = {}
    members = hashlib.sha256(b'{"ordered_members_v1":[')

    def entries() -> Iterable[ManifestEntry]:
        for position, row in enumerate(iter_manifest(record.path)):
            path = (record.path.parent / row["path"]).resolve()
            shard = index.by_path.get(path)
            if shard is None or not isinstance(shard.metadata, AuthenticatedSidecar):
                raise ValueError(f"missing audited manifest member: {path}")
            metadata = shard.metadata
            artifact = metadata.artifact
            schema = artifact.arrow_schema
            if (
                not closure(metadata.stage_identity)
                or metadata.stage_identity.stage_key != "simulation"
                or metadata.method_contract.procedure != procedure
                or path.parent != record.path.parent
                or schema is None
                or row["byte_length"] != artifact.byte_length
                or row["sidecar_sha256"] != shard.sidecar_sha256
                or row["data_sha256"] != artifact.content_sha256
                or row["schema_fingerprint_sha256"] != schema.fingerprint_sha256
            ):
                raise ValueError(f"incompatible native manifest member: {path}")
            inherited.update(closure(metadata.stage_identity))
            coordinate = next(
                row[k]
                for k in ("shuffle_index", "process_block_index", "deterministic_batch_id")
                if k in row
            )
            member = identity_sha256(
                {
                    "shard": scientific_artifact_binding(metadata, code),
                    "coordinates_and_counts": {
                        k: v for k, v in row.items() if k not in {"path", "sidecar_sha256"}
                    },
                }
            )
            if position:
                members.update(b",")
            members.update(canonical_json_bytes(member))
            yield ManifestEntry(
                (coordinate,),
                path.relative_to(index.root).as_posix(),
                artifact.content_sha256,
                row["sidecar_sha256"],
                schema.fingerprint_sha256,
            )

    summary = compute_manifest_root(entries())
    members.update(b"]}")
    if (
        summary != manifest.summary
        or not summary.entry_count
        or members.hexdigest()
        != manifest.stage_identity.immutable_design_identities.get("manifest_members_v1")
    ):
        raise ValueError(f"unclassified or incompatible manifest inventory at {record.path}")
    return inherited


def _audit_original_parents(
    cfg: AppConfig, launch: _ProducerLaunch, roots: Mapping[tuple[int, str], _ProducerLaunch]
) -> None:
    parents = getattr(cfg, "_source_root_configs", ())
    if len(parents) != 2 or [p.sim.seed for p in parents] != cfg.sim.seed_list:
        raise ValueError("historical pair requires both original ordered roots")
    run_id = launch.receipt["run_id"]
    for parent, digest in zip(parents, launch.context["parent_lifecycle_roots"], strict=True):
        original = roots[(parent.sim.seed, run_id)]
        audit_parent_lifecycle(
            parent.results_root, run_id, original.receipt, original.context, digest
        )


def _release_git(*args: str, input_bytes: bytes | None = None) -> bytes:
    from farkle.utils.scientific_invocation import ROOT

    result = subprocess.run(
        ["git", "--no-replace-objects", *args],
        cwd=ROOT,
        input=input_bytes,
        capture_output=True,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
    )
    if result.returncode:
        raise ValueError(
            f"cannot authenticate local Git objects: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def _committed_files(commit: str) -> dict[str, bytes]:
    """Read original path, mode and blob identities without checking out the producer."""
    if _release_git("cat-file", "-t", commit).strip() != b"commit":
        raise ValueError(f"unavailable producer commit: {commit}")
    files = {}
    for entry in filter(None, _release_git("ls-tree", "-rz", "--full-tree", commit).split(b"\0")):
        header, path = entry.split(b"\t", 1)
        files[os.fsdecode(path)] = header
    return files


def _working_blobs(paths: tuple[str, ...]) -> dict[str, bytes]:
    """Read the pinned working files themselves, allowing Git's normal text EOL conversion."""
    encoded = [os.fsencode(path) for path in paths]
    if any(b"\n" in path or path.startswith(b'"') for path in encoded):
        raise ValueError("unsupported scientific source pathname")
    attributes = _release_git(
        "check-attr",
        "-z",
        "--stdin",
        "filter",
        "working-tree-encoding",
        "ident",
        input_bytes=b"\0".join(encoded) + b"\0",
    ).split(b"\0")[:-1]
    if len(attributes) != 9 * len(paths) or any(
        value not in {b"unspecified", b"unset"} for value in attributes[2::3]
    ):
        raise ValueError("unclassified Git checkout transformation on scientific sources")
    blobs = _release_git("hash-object", "--stdin-paths", input_bytes=b"\n".join(encoded) + b"\n")
    return dict(zip(paths, blobs.splitlines(), strict=True))


def _audit_frozen_execution(cfg: AppConfig, index: _AuditIndex) -> None:
    """Check every terminal block against the exact consented schedule without dispatch."""
    import pyarrow.parquet as pq

    from farkle.analysis.candidate_family import verify_family_identity
    from farkle.analysis.h2h_authorization import (
        validate_authorization,
        validate_authorization_sources,
    )
    from farkle.analysis.h2h_schedule import (
        _TERMINAL_BLOCK_STATUSES,
        _block_path,
        _configured_roots,
        _configured_schedule_hash,
        _load_frozen_family,
        _read_authenticated_block,
        _roots_from_manifest,
        _schedule_rows,
    )
    from farkle.utils.frozen_h2h import BUILTIN_ENGINE, verify_frozen_strategy_design

    power, schedule = cfg.h2h_power_plan_path(), cfg.h2h_block_manifest_path()
    plan = read_json_file_with_retry(power)
    verify_frozen_strategy_design(cfg, plan)
    if plan["frozen_design"]["engine"] != BUILTIN_ENGINE:
        raise ValueError(f"release requires the authorized built-in engine at {power}")
    family = _load_frozen_family(cfg)
    verify_family_identity(cfg, family)
    candidates = family["candidates"]
    roots = _roots_from_manifest(family)
    block_games = plan["n_completed_required_per_root_order_block"]
    pairs = len(candidates) * (len(candidates) - 1) // 2
    alpha = cfg.head2head.family_alpha / pairs
    schedule_hash = _configured_schedule_hash(
        cfg, family["family_hash"], roots, alpha, block_games, plan["frozen_design"]
    )
    if (
        roots != _configured_roots(cfg)
        or plan["family_hash"] != family["family_hash"]
        or schedule_hash != plan["schedule_hash"]
    ):
        raise ValueError(
            f"frozen family or schedule identity differs from the scientific design: {power}"
        )
    expected = _schedule_rows(
        candidates=candidates,
        family_hash=family["family_hash"],
        schedule_hash=schedule_hash,
        roots=roots,
        block_games=block_games,
        max_attempt_multiplier=cfg.head2head.max_attempt_multiplier,
        alpha_per_pair=alpha,
        target_power=cfg.head2head.target_power,
        worst_power=plan["worst_scenario_achieved_power"],
        game_profile_sha256=cfg._game_profile_sha256,
    )
    authorization = read_json_file_with_retry(cfg.h2h_cap_authorization_path())
    validate_authorization(authorization, plan, require_ready=True)
    validate_authorization_sources(authorization, power, schedule)
    schedule_sha = index.by_path[schedule.resolve()].sidecar_sha256
    blocks = set()
    max_attempts = 0
    for batch in pq.ParquetFile(schedule).iter_batches(batch_size=256):
        for block in batch.to_pylist():
            if block != next(expected, None):
                raise ValueError(f"schedule block differs from its frozen design at {schedule}")
            path = _block_path(cfg, block).resolve()
            meta = index.by_path[path].metadata
            row = _read_authenticated_block(path, block)
            if (
                not isinstance(meta, AuthenticatedSidecar)
                or [s.sidecar_sha256 for s in meta.source_artifacts] != [schedule_sha]
                or path in blocks
                or row is None
                or row["completion_status"] not in _TERMINAL_BLOCK_STATUSES
                or any(
                    row[field] != block[field]
                    for field in ("strategy_a", "strategy_b", "seat1_strategy", "seat2_strategy")
                )
            ):
                raise ValueError(f"block differs from the authorized design: {path}")
            blocks.add(path)
            max_attempts += block["max_attempts"]
    observed = {
        r.path
        for r in index.by_path.values()
        if isinstance(r.metadata, AuthenticatedSidecar)
        and r.metadata.method_contract.procedure == "simulate_root_order_block"
    }
    if (
        next(expected, None) is not None
        or blocks != observed
        or len(blocks) != plan["total_block_count"]
        or max_attempts != plan["maximum_total_attempts"]
    ):
        raise ValueError(f"block inventory differs from authorization at {schedule}")


def _audited_scientific_inputs(
    record: _IndexedArtifact,
    meta: AuthenticatedSidecar,
    by_sidecar: Mapping[str, _IndexedArtifact],
    code: CodeIdentity,
) -> tuple[dict[str, str], dict[str, str]]:
    from farkle.utils.release_identity import scientific_artifact_binding
    from farkle.utils.scientific_reuse import closure

    bindings: dict[str, str] = {}
    upstream = {}
    for edge in meta.source_artifacts:
        source = by_sidecar[edge.sidecar_sha256].metadata
        if not isinstance(source, AuthenticatedSidecar) or not closure(source.stage_identity):
            raise ValueError(f"missing scientific input at {record.path}: {edge.logical_role}")
        upstream.update(closure(source.stage_identity))
        bindings[edge.logical_role] = scientific_artifact_binding(source, code)
    for manifest_edge in meta.manifest_roots:
        manifest = by_sidecar[manifest_edge.sidecar_sha256].metadata
        if not isinstance(manifest, ImmutableManifestSidecar) or not closure(
            manifest.stage_identity
        ):
            raise ValueError(f"incomplete manifest source at {record.path}")
        upstream.update(closure(manifest.stage_identity))
        published = manifest.stage_identity.immutable_design_identities
        bindings[manifest_edge.logical_role] = identity_sha256(
            {
                "producer": published["publication_science_v1"],
                "location": manifest.location,
                "members": published["manifest_members_v1"],
            }
        )
    return bindings, upstream


def _audit_scientific_leaf(
    cfg: AppConfig,
    record: _IndexedArtifact,
    meta: AuthenticatedSidecar | ImmutableManifestSidecar,
    digest: str,
) -> None:
    stage = meta.stage_identity
    designs = stage.immutable_design_identities
    if (
        isinstance(meta, AuthenticatedSidecar)
        and stage.stage_key == "simulation"
        and meta.method_contract.procedure == "publish_strategy_manifest"
        and record.path == cfg.strategy_manifest_root_path()
        and designs.get("ordered_builtin_roster_v1")
    ):
        from farkle.simulation.runner import simulation_roster_manifest

        simulation_roster_manifest(cfg)
    elif (
        isinstance(meta, AuthenticatedSidecar)
        and stage.stage_key == "rng_diagnostics"
        and meta.method_contract.procedure == "first_release_rng_evidence"
        and record.path == cfg.rng_preflight_dir() / "plan.json"
    ):
        from farkle.analysis.rng_preflight import _preflight_design, environment

        plan = read_json_file_with_retry(record.path)
        if (
            plan["schema_version"] != 2
            or plan["scientific_dependencies_v1"] != digest
            or plan["code"] != asdict(stage.code)
            or plan["topology_extension"]["workers"] != cfg.sim.n_jobs
            or _preflight_design(plan)
            != _preflight_design({**plan, "environment": environment(cfg)})
            or designs.get("preflight_design_v2") != identity_sha256(plan)
        ):
            raise ValueError(f"incompatible preflight design at {record.path}")
    else:
        raise ValueError(f"unclassified scientific leaf at {record.path}")


def _audit_mixed_context(
    target: AuthenticatedReleaseAuditTarget,
    index: _AuditIndex,
    launches: list[_ProducerLaunch],
    by_sidecar: Mapping[str, _IndexedArtifact],
    code: CodeIdentity,
    tree: Callable[[str], dict[str, bytes]],
    working_blobs: Callable[[tuple[str, ...]], dict[str, bytes]],
) -> None:

    from farkle.utils.release_identity import (
        completion_producer_identity,
        publication_science_controls,
        publication_science_digest,
        require_authenticated_output_versions,
    )
    from farkle.utils.scientific_dependencies import stage_producer
    from farkle.utils.scientific_identity import producer_config_identity
    from farkle.utils.scientific_invocation import current, dependency_key
    from farkle.utils.scientific_reuse import closure, launch_matches

    cfg = target.cfg
    roots = cfg.sim.seed_list or [cfg.sim.seed]
    retained_controls = {
        (name, digest) for launch in launches for name, digest in launch.receipt["current"].items()
    }
    retained_controls.update(
        (Path(path).name, digest)
        for launch in launches
        for _, path, digest in launch.receipt["previous_completion_stamps"]
        if digest
    )

    @cache
    def control_hashes(name: str) -> set[str]:
        return {
            sha256_file(p)
            for p in index.files
            if p.name == name
            and name.endswith(".done.json")
            and not excluded_graph_path(p, index.root)
        }

    @cache
    def requirements(stage: str):
        owner = stage_producer(cfg, stage)
        return (
            f"{owner.scope}/{stage}",
            dependency_key(owner, roots),
            cfg.stage_config_sha(stage),
            producer_config_identity(cfg, stage),
        )

    @cache
    def verify_git(stage: str, commit: str) -> None:
        pin = current()
        assert pin is not None
        paths = pin.components(cfg, stage_producer(cfg, stage))["implementation"]
        original, release = tree(commit), tree(code.commit)
        for path, oid in working_blobs(tuple(sorted(paths))).items():
            if (
                path not in release
                or original.get(path) != release[path]
                or release[path] not in (b"100644 blob " + oid, b"100755 blob " + oid)
            ):
                raise ValueError(
                    f"producer {commit} has different or untracked {stage} implementation: {path}"
                )

    for record in index.by_path.values():
        meta = record.metadata
        if meta is None:
            raise ValueError(f"missing verified producer: {record.path}")
        stage = meta.stage_identity
        name, key, digest, config = requirements(record.location.stage_key)
        designs = stage.immutable_design_identities
        origins = [
            launch
            for launch in launches
            if launch.code == stage.code
            and launch.context["run_lineage_sha256"] == designs.get("run_lineage_sha256")
        ]
        if (
            stage.code.policy != CodeIdentityPolicy.RELEASE_CLEAN.value
            or stage.code.state != "clean"
            or stage.stage_config != config
            or stage.stage_cache_key_version != cfg.stage_cache_key_version(stage.stage_key)
            or record.location.path(cfg).resolve() != record.path
            or designs.get("stage_dependencies_v1") != digest
            or not designs.get("publication_science_v1")
            or not origins
            or not all(launch.confirms(name, digest) for launch in origins)
        ):
            raise ValueError(f"missing or incompatible clean producer launch at {record.path}")
        verify_git(stage.stage_key, stage.code.commit)
        if isinstance(meta, ImmutableManifestSidecar):
            upstream = _manifest_member_dependencies(cfg, record, index, code)
        else:
            require_authenticated_output_versions(cfg, record.path, meta)
            if meta.method_contract.root_seeds != tuple(sorted(roots)):
                raise ValueError(f"scientific producer roots differ at {record.path}")
            bindings, upstream = _audited_scientific_inputs(record, meta, by_sidecar, code)
            for control, sha in publication_science_controls(
                stage.stage_key,
                meta.method_contract,
                designs,
                has_sources=bool(meta.source_artifacts),
                has_manifests=bool(meta.manifest_roots),
            ).items():
                _, filename, _ = control.split(":")
                if (filename, sha) not in retained_controls and sha not in control_hashes(filename):
                    raise ValueError(f"unverified scientific control at {record.path}: {control}")
            if (
                publication_science_digest(
                    stage.stage_key,
                    digest,
                    meta.method_contract,
                    meta.versions,
                    designs,
                    bindings,
                    has_sources=bool(meta.source_artifacts),
                    has_manifests=bool(meta.manifest_roots),
                )
                != designs["publication_science_v1"]
            ):
                raise ValueError(f"scientific publication preimage differs at {record.path}")
            if (
                record.location.stage_key == "h2h_power"
                and record.path == cfg.h2h_power_plan_path().resolve()
            ):
                _audit_frozen_execution(cfg, index)
        if not upstream:
            _audit_scientific_leaf(cfg, record, meta, digest)
        if not launch_matches(stage) or closure(stage) != {**upstream, key: digest}:
            raise ValueError(f"incomplete or incompatible scientific dependencies at {record.path}")

    for completion in target.snapshot.completions:
        records = sorted(
            (index.by_location[item.location] for item in completion.outputs),
            key=lambda item: str(item.path),
        )
        stage_key = records[0].location.stage_key
        name, _, digest, _ = requirements(stage_key)
        producer = index.completion_producers.get(completion.path)
        matching = [
            launch
            for launch in launches
            if launch.code == producer and launch.confirms(name, digest)
        ]
        if (
            producer is None
            or producer.state != "clean"
            or producer.policy != CodeIdentityPolicy.RELEASE_CLEAN.value
            or not matching
        ):
            raise ValueError(f"missing clean completion producer at {completion.path}")
        verify_git(stage_key, producer.commit)
        versions = next(
            (r.metadata.versions for r in records if isinstance(r.metadata, AuthenticatedSidecar)),
            None,
        )
        if versions is None:
            first = records[0].metadata
            assert isinstance(first, ImmutableManifestSidecar)
            versions = first.stage_identity.versions
        output_hashes = {
            f"output:{r.location.relative_path}": r.sidecar_sha256
            for r in records
            if r.sidecar_sha256 is not None
        }
        reconstructed = completion_producer_identity(
            cfg, stage_key, versions, output_hashes, producer
        )
        if reconstructed.sha256 != completion.stage_identity_sha256:
            raise ValueError(f"incompatible completion producer preimage at {completion.path}")
        if producer != code:
            previous_contexts = {
                launch.receipt["current"]["run_context.json"] for launch in matching
            }
            if not any(
                launch.receipt["previous"]["run_context.json"] in previous_contexts
                and [completion.stage_key, str(completion.path), completion.sha256]
                in launch.receipt["previous_completion_stamps"]
                for launch in launches
            ):
                raise ValueError(
                    f"missing original completion in retained history: {completion.path}"
                )


def _revision_policy(
    targets: tuple[AuthenticatedReleaseAuditTarget, ...],
    indexes: list[_AuditIndex],
    events: list[dict[str, Any]],
    code: CodeIdentity,
    foreign_count: int,
    audit_passed: bool,
) -> dict[str, Any]:
    from farkle.utils.scientific_files import ScientificDrift
    from farkle.utils.scientific_invocation import ROOT, current, verify_launch

    commits: set[str] = set()
    for index in indexes:
        commits.update(
            r.metadata.stage_identity.code.commit for r in index.by_path.values() if r.metadata
        )
        commits.update(c.commit for c in index.completion_producers.values())
    evidence = []
    reasons = []
    if foreign_count:
        try:
            if code.policy != CodeIdentityPolicy.RELEASE_CLEAN.value or code.state != "clean":
                raise ValueError("mixed revisions require a release-clean invocation")
            if not audit_passed or current() is None or not events:
                raise ValueError(
                    "mixed revisions require a successful independent graph audit and retained scientific launches"
                )
            verify_launch()
            by_sidecar = {
                r.sidecar_sha256: r
                for index in indexes
                for r in index.by_path.values()
                if r.sidecar_sha256 is not None
            }
            trees = cache(_committed_files)
            roots: dict[tuple[int, str], _ProducerLaunch] = {}
            working_blobs = cache(_working_blobs)
            for target, index in zip(targets, indexes, strict=True):
                launches = _producer_launches(target, events)
                for launch in launches:
                    run_id = launch.receipt["run_id"]
                    if target.snapshot.scope == "root":
                        roots[(target.cfg.sim.seed, run_id)] = launch
                    else:
                        _audit_original_parents(target.cfg, launch, roots)
                    evidence.append(
                        {
                            "record": launch.event["invocation_record"],
                            "sha256": launch.event["invocation_sha256"],
                            "code_identity": asdict(launch.code),
                        }
                    )
                _audit_mixed_context(
                    target, index, launches, by_sidecar, code, trees, working_blobs
                )
            verify_launch()
            if resolve_code_identity(ROOT, policy=CodeIdentityPolicy.RELEASE_CLEAN) != code:
                raise ValueError("release Git identity changed during mixed-revision qualification")
        except ScientificDrift:
            raise
        except Exception as exc:
            reasons.append(f"mixed-revision release evidence: {type(exc).__name__}: {exc}")
    return {
        "version": REVISION_RELEASE_POLICY_VERSION,
        "qualifying_commit": code.commit,
        "foreign_publications": foreign_count,
        "status": (
            "ineligible"
            if reasons or not audit_passed
            else ("qualified_mixed" if foreign_count else "single_revision")
        ),
        "producer_commits": sorted(commits),
        "invocation_records": evidence,
        "reasons": reasons,
    }


def audit_authenticated_release_graphs(
    targets: Iterable[AuthenticatedReleaseAuditTarget],
    *,
    expected_code_identity: Any,
    current_code_identity: Any,
    invocation: tuple[str, Path] | None = None,
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
    launch_events: list[dict[str, Any]] = []
    published: list[tuple[Path, str]] = []
    if invocation is not None:
        run_id, manifest_path = invocation
        try:
            launch_events = [
                e for e in iter_manifest(manifest_path) if e.get("event") == "invocation_published"
            ]
            published = [
                (Path(event["invocation_record"]).resolve(), event["invocation_sha256"])
                for event in launch_events
                if event.get("run_id") == run_id
            ]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            failures.append(f"cannot read invocation records from {manifest_path}: {exc}")
    foreign_producers: set[str] = set()
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
                or snapshot.code_identity != expected_code_identity
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
            if invocation is not None:
                evidence = audit_invocation(snapshot, invocation[0])
                binding = (
                    Path(evidence["invocation_record"]).resolve(),
                    evidence["invocation_sha256"],
                )
                if published.count(binding) != 1:
                    raise ValueError(
                        f"invocation record differs from its manifest event: {binding[0]}"
                    )
                published.remove(binding)
                run_contexts[label].update(evidence)
                if snapshot.scope == "pair":
                    parents = [
                        run_contexts[f"root_{seed}"]["lifecycle_sha256"]
                        for seed in target.cfg.sim.seed_list or ()
                    ]
                    if parents != persisted["parent_lifecycle_roots"]:
                        raise ValueError("retained parent lifecycles differ from the pair context")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label} run context: {type(exc).__name__}: {exc}")

        index, root_failures = _build_audit_index(
            snapshot.graph_root,
            completion_paths=(item.path for item in snapshot.completions),
            analysis_root=snapshot.analysis_root,
        )
        root_failures.extend(_validate_audit_index(index))
        root_failures.extend(_compare_snapshot_index(snapshot, index))
        foreign_producers.update(_foreign_producers(index, expected_code_identity))
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

    if published:
        failures.append("unmatched invocation publication in pipeline manifest")
    failures.extend(_validate_cross_context_provenance(indexes))
    if boundary_hook is not None:
        boundary_hook("complete")
    return {
        "status": "passed" if not failures else "failed",
        "failures": sorted(failures),
        "run_contexts": run_contexts,
        "foreign_producer_count": len(foreign_producers),
        "revision_policy": _revision_policy(
            ordered_targets,
            indexes,
            launch_events,
            expected_code_identity,
            len(foreign_producers),
            not failures,
        ),
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
