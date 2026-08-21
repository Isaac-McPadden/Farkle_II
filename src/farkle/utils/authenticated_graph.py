"""Immutable, explicitly owned, process-local authenticated graph snapshots."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from farkle.config import AppConfig, compute_config_sha
from farkle.orchestration.run_contexts import load_run_context
from farkle.utils.artifact_contract import read_json_file_with_retry, sha256_file
from farkle.utils.authenticated_contract import (
    ArtifactIdentity,
    AuthenticatedCompletion,
    CanonicalArtifactLocation,
    CodeIdentity,
    CompletionOutputIdentity,
    ManifestRootIdentity,
    canonical_json_bytes,
    load_authenticated_completion,
    load_authenticated_sidecar,
    load_immutable_manifest_sidecar,
)
from farkle.utils.authentication_telemetry import (
    AuthenticationTelemetry,
    active_authentication_telemetry,
    record_authentication_open,
)
from farkle.utils.completion_files import CompletionFileKind, CompletionNamespace
from farkle.utils.stage_completion import CompletionState, freshness_sha256

_COMPLETE_STATUS: Final = CompletionState.COMPLETE_VALID.value


class SnapshotUseError(RuntimeError):
    """Raised when a process-local snapshot is stale or used out of scope."""


class SnapshotGeneration:
    """Explicit lifetime owner for snapshots from one quiescent run context."""

    __slots__ = ("_generation", "_process_id", "_token")

    def __init__(self) -> None:
        self._process_id = os.getpid()
        self._generation = 0
        self._token = object()

    @property
    def generation(self) -> int:
        return self._generation

    def invalidate(
        self,
        _reason: str,
        *,
        telemetry: AuthenticationTelemetry | None = None,
    ) -> None:
        """Invalidate every snapshot previously issued by this owner."""

        self._generation += 1
        self._token = object()
        collector = telemetry or active_authentication_telemetry()
        if collector is not None:
            collector.bump("snapshot_invalidations")

    def validate(
        self,
        snapshot: AuthenticatedGraphSnapshot,
        *,
        expected_scope: str,
        expected_roots: tuple[int, ...],
        expected_run_context_path: Path,
        telemetry: AuthenticationTelemetry | None = None,
    ) -> None:
        """Fail closed unless *snapshot* belongs to this live process/context."""

        collector = telemetry or active_authentication_telemetry()
        valid = (
            os.getpid() == self._process_id == snapshot.process_id
            and snapshot._generation_token is self._token
            and snapshot.generation == self._generation
            and snapshot.scope == expected_scope
            and snapshot.roots == expected_roots
            and snapshot.run_context_path == expected_run_context_path.resolve()
            and snapshot.construction_status == "complete"
            and snapshot.all_required_stages_complete
        )
        if not valid:
            if collector is not None:
                collector.bump("snapshot_rejected_uses")
            raise SnapshotUseError(
                "authenticated graph snapshot is stale, foreign, incomplete, or out of scope"
            )
        if collector is not None:
            collector.bump("snapshot_hits")

    def __reduce__(self) -> Any:
        raise TypeError("snapshot generations are process-local and cannot be serialized")


@dataclass(frozen=True, slots=True)
class SnapshotCompletion:
    """Exact completion identity captured after complete-valid authentication."""

    stage_key: str
    path: Path
    sha256: str
    stage_identity_sha256: str
    outputs: tuple[CompletionOutputIdentity, ...]


@dataclass(frozen=True, slots=True)
class SnapshotGraphIdentity:
    """Exact sidecar-declared identity for one canonical graph member."""

    location: CanonicalArtifactLocation
    artifact: ArtifactIdentity | None
    manifest: ManifestRootIdentity | None
    sidecar_sha256: str


@dataclass(frozen=True, slots=True)
class AuthenticatedGraphSnapshot:
    """Trusted only by its explicit :class:`SnapshotGeneration` in this process."""

    scope: str
    roots: tuple[int, ...]
    graph_root: Path
    analysis_root: Path
    run_context_path: Path
    active_config_path: Path
    run_context_sha256: str
    run_context_identity_sha256: str
    active_config_sha256: str
    public_config_sha256: str
    statistical_config_sha256: str
    run_lineage_sha256: str
    code_identity: CodeIdentity
    game_profile_sha256: str | None
    stage_states: tuple[tuple[str, str], ...]
    completions: tuple[SnapshotCompletion, ...]
    outputs: tuple[CompletionOutputIdentity, ...]
    graph_inventory: tuple[SnapshotGraphIdentity, ...]
    artifact_locations: tuple[CanonicalArtifactLocation, ...]
    manifest_locations: tuple[CanonicalArtifactLocation, ...]
    stage_identity_sha256: tuple[tuple[str, str], ...]
    lifecycle_sha256: str
    construction_status: str
    all_required_stages_complete: bool
    process_id: int
    generation: int
    _generation_token: object

    def __reduce__(self) -> Any:
        raise TypeError("authenticated graph snapshots are process-local and cannot be serialized")


def _code_identity_payload(identity: CodeIdentity) -> dict[str, object]:
    return {
        "commit": identity.commit,
        "policy": identity.policy,
        "state": identity.state,
        "dirty_fingerprint_sha256": identity.dirty_fingerprint_sha256,
    }


def _output_sort_key(output: CompletionOutputIdentity) -> bytes:
    return canonical_json_bytes(output.location)


def _capture_completion(stage_key: str, path: Path) -> SnapshotCompletion:
    completion: AuthenticatedCompletion = load_authenticated_completion(path)
    if completion.state != _COMPLETE_STATUS:
        raise SnapshotUseError(f"stage is not complete-valid: {stage_key}: {path}")
    return SnapshotCompletion(
        stage_key=stage_key,
        path=path.resolve(),
        sha256=sha256_file(path),
        stage_identity_sha256=completion.stage_identity_sha256,
        outputs=completion.outputs,
    )


def _capture_graph_inventory(graph_root: Path) -> tuple[SnapshotGraphIdentity, ...]:
    collector = active_authentication_telemetry()
    if collector is not None:
        collector.bump("directory_traversals")
    sidecars = sorted(
        path
        for path in graph_root.rglob("*.sidecar.json")
        if path.is_file()
        and not any(part.startswith("_") for part in path.relative_to(graph_root).parts)
    )
    inventory: dict[CanonicalArtifactLocation, SnapshotGraphIdentity] = {}
    for adjacent in sidecars:
        artifact_path = Path(str(adjacent)[: -len(".sidecar.json")])
        if not artifact_path.is_file():
            raise SnapshotUseError(f"orphan sidecar cannot enter snapshot: {adjacent}")
        payload = read_json_file_with_retry(adjacent)
        if not isinstance(payload, Mapping) or payload.get("artifact_contract_version") != 3:
            raise SnapshotUseError(f"non-v3 graph member cannot enter snapshot: {artifact_path}")
        sidecar_sha = sha256_file(adjacent)
        if "manifest_contract_version" in payload:
            manifest_metadata = load_immutable_manifest_sidecar(artifact_path)
            location = manifest_metadata.location
            manifest = ManifestRootIdentity(
                logical_role=(
                    f"manifest:{location.stage_key}:{location.scope}:"
                    f"{location.player_count}:{location.relative_path}"
                ),
                location=location,
                manifest_sha256=manifest_metadata.manifest_sha256,
                sidecar_sha256=sidecar_sha,
                sidecar_contract_sha256=manifest_metadata.sidecar_contract_sha256,
                summary=manifest_metadata.summary,
            )
            item = SnapshotGraphIdentity(
                location=location,
                artifact=None,
                manifest=manifest,
                sidecar_sha256=sidecar_sha,
            )
        else:
            artifact_metadata = load_authenticated_sidecar(artifact_path)
            location = artifact_metadata.artifact.location
            item = SnapshotGraphIdentity(
                location=location,
                artifact=artifact_metadata.artifact,
                manifest=None,
                sidecar_sha256=sidecar_sha,
            )
        previous = inventory.get(location)
        if previous is not None:
            raise SnapshotUseError(f"duplicate canonical graph location: {location}")
        inventory[location] = item
    return tuple(sorted(inventory.values(), key=lambda item: canonical_json_bytes(item.location)))


def capture_authenticated_graph_snapshot(
    *,
    cfg: AppConfig,
    scope: str,
    roots: tuple[int, ...],
    graph_root: Path,
    analysis_root: Path,
    run_context_path: Path,
    active_config_path: Path,
    stage_states: Mapping[str, str],
    completion_paths: Sequence[tuple[str, Path]],
    generation: SnapshotGeneration,
    code_identity: CodeIdentity,
    boundary_hook: Callable[[str], None] | None = None,
) -> AuthenticatedGraphSnapshot:
    """Capture a complete graph after the caller authenticated every stage.

    The function performs no publication.  An exception at any boundary returns
    no object and therefore cannot leave trusted partial state.
    """

    if scope not in {"root", "pair"}:
        raise ValueError("snapshot scope must be root or pair")
    if not roots or tuple(sorted(set(roots))) != roots:
        raise ValueError("snapshot roots must be unique and canonically ordered")
    ordered_states = tuple(sorted((str(key), str(value)) for key, value in stage_states.items()))
    if not ordered_states or any(value != _COMPLETE_STATUS for _, value in ordered_states):
        raise SnapshotUseError("cannot capture an incomplete or stale authenticated graph")
    if len(completion_paths) != len(ordered_states):
        raise SnapshotUseError("completion inventory does not cover every required stage")
    if len({name for name, _ in completion_paths}) != len(completion_paths):
        raise SnapshotUseError("completion inventory contains duplicate stage keys")
    if set(stage_states) != {name for name, _ in completion_paths}:
        raise SnapshotUseError("stage-state and completion inventories disagree")
    namespace = CompletionNamespace.build(
        graph_root=graph_root,
        analysis_root=analysis_root,
        canonical_paths=(path for _stage, path in completion_paths),
    )
    invalid_completion_paths = [
        path
        for _stage, path in completion_paths
        if namespace.classify(path) is not CompletionFileKind.CANONICAL_STAGE
    ]
    if invalid_completion_paths:
        raise SnapshotUseError(
            f"completion inventory contains noncanonical paths: {invalid_completion_paths}"
        )

    persisted = load_run_context(run_context_path, active_config_path=active_config_path)
    record_authentication_open(run_context_path)
    physical_context_sha = sha256_file(run_context_path)
    if boundary_hook is not None:
        boundary_hook("run_context_authenticated")
    record_authentication_open(active_config_path)
    active_config_sha = sha256_file(active_config_path)
    if boundary_hook is not None:
        boundary_hook("active_config_authenticated")

    expected_config_sha = compute_config_sha(cfg)
    expected_lineage = cfg._run_lineage_sha256
    if expected_lineage is None:
        raise SnapshotUseError("run context has no authenticated lineage")
    if (
        persisted.get("public_config_sha256") != expected_config_sha
        or persisted.get("run_lineage_sha256") != expected_lineage
        or persisted.get("code_identity") != _code_identity_payload(code_identity)
        or persisted.get("resolved_paths", {}).get("analysis_root") != str(analysis_root)
    ):
        raise SnapshotUseError("run context does not match the completed in-process context")

    captured: list[SnapshotCompletion] = []
    for stage_key, completion_path in completion_paths:
        captured.append(_capture_completion(stage_key, completion_path))
        if boundary_hook is not None:
            boundary_hook(f"completion:{stage_key}")
    completions = tuple(sorted(captured, key=lambda item: item.path.as_posix()))

    outputs_by_location: dict[CanonicalArtifactLocation, CompletionOutputIdentity] = {}
    for completion in completions:
        for output in completion.outputs:
            previous = outputs_by_location.get(output.location)
            if previous is not None and previous != output:
                raise SnapshotUseError(f"conflicting completion identities for {output.location}")
            outputs_by_location[output.location] = output
    outputs = tuple(sorted(outputs_by_location.values(), key=_output_sort_key))
    if not outputs:
        raise SnapshotUseError("complete graph snapshot requires authenticated outputs")
    graph_inventory = _capture_graph_inventory(graph_root.resolve())
    graph_by_location = {item.location: item for item in graph_inventory}
    for output in outputs:
        graph_item = graph_by_location.get(output.location)
        if graph_item is None or graph_item.sidecar_sha256 != output.sidecar_sha256:
            raise SnapshotUseError(
                f"completion output is absent or stale in graph inventory: {output.location}"
            )
        if output.artifact is not None and graph_item.artifact != output.artifact:
            raise SnapshotUseError(f"completion artifact identity is stale: {output.location}")
        if output.manifest is not None and (
            graph_item.manifest is None
            or graph_item.manifest.manifest_sha256 != output.manifest.manifest_sha256
            or graph_item.manifest.sidecar_contract_sha256
            != output.manifest.sidecar_contract_sha256
            or graph_item.manifest.summary != output.manifest.summary
        ):
            raise SnapshotUseError(f"completion manifest identity is stale: {output.location}")
    artifact_locations = tuple(
        output.location for output in graph_inventory if output.artifact is not None
    )
    manifest_locations = tuple(
        output.location for output in graph_inventory if output.manifest is not None
    )
    if boundary_hook is not None:
        boundary_hook("inventory_complete")

    ordered_completion_sha = [
        next(item.sha256 for item in captured if item.stage_key == stage_key)
        for stage_key, _ in completion_paths
    ]
    lifecycle_sha = freshness_sha256(
        {
            "run_lineage_sha256": expected_lineage,
            "completion_stamps": ordered_completion_sha,
        }
    )
    snapshot = AuthenticatedGraphSnapshot(
        scope=scope,
        roots=roots,
        graph_root=graph_root.resolve(),
        analysis_root=analysis_root.resolve(),
        run_context_path=run_context_path.resolve(),
        active_config_path=active_config_path.resolve(),
        run_context_sha256=physical_context_sha,
        run_context_identity_sha256=str(persisted["run_context_sha256"]),
        active_config_sha256=active_config_sha,
        public_config_sha256=expected_config_sha,
        statistical_config_sha256=expected_config_sha,
        run_lineage_sha256=expected_lineage,
        code_identity=code_identity,
        game_profile_sha256=cfg._game_profile_sha256,
        stage_states=ordered_states,
        completions=completions,
        outputs=outputs,
        graph_inventory=graph_inventory,
        artifact_locations=artifact_locations,
        manifest_locations=manifest_locations,
        stage_identity_sha256=tuple(
            sorted((item.stage_key, item.stage_identity_sha256) for item in completions)
        ),
        lifecycle_sha256=lifecycle_sha,
        construction_status="complete",
        all_required_stages_complete=True,
        process_id=os.getpid(),
        generation=generation.generation,
        _generation_token=generation._token,
    )
    collector = active_authentication_telemetry()
    if collector is not None:
        collector.bump("snapshot_builds")
    if boundary_hook is not None:
        boundary_hook("snapshot_ready")
    return snapshot


__all__ = [
    "AuthenticatedGraphSnapshot",
    "SnapshotCompletion",
    "SnapshotGeneration",
    "SnapshotGraphIdentity",
    "SnapshotUseError",
    "capture_authenticated_graph_snapshot",
]
