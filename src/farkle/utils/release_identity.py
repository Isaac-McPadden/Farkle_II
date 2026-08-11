"""Release-path adapters for authenticated artifact contract version 3.

The analysis code historically assembled rich semantic metadata before calling
the shared atomic writers.  This module translates that already-implemented
behavior into the typed v3 identities at the publication boundary.  Contract-2
sidecars are never upgraded or accepted as v3 sources.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from farkle.config import AppConfig, ArtifactScope, effective_config_dict
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    ArtifactSidecar,
    read_json_file_with_retry,
    sha256_file,
    sidecar_path,
)
from farkle.utils.authenticated_contract import (
    ARTIFACT_CONTRACT_VERSION,
    LIFECYCLE_CONTRACT_VERSION,
    ArtifactFormatIdentity,
    AuthenticatedCompletion,
    AuthenticatedSidecar,
    CanonicalArtifactLocation,
    CodeIdentityPolicy,
    CompletionOutputIdentity,
    ManifestEntry,
    ManifestRootIdentity,
    MethodContract,
    SourceArtifactIdentity,
    VersionIdentity,
    canonical_json_bytes,
    capture_source_artifact_unbound,
    classify_authenticated_lifecycle,
    derive_canonical_location,
    identity_sha256,
    load_immutable_manifest_sidecar,
    make_stage_identity,
    publish_immutable_manifest_bytes_atomic,
    publish_staged_authenticated_artifact_atomic,
    resolve_code_identity,
    stage_config_identity,
    validate_authenticated_artifact_metadata,
    validate_authenticated_artifact_unbound,
    write_authenticated_completion_atomic,
)
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION, TOURNAMENT_METHOD_VERSION
from farkle.utils.stage_completion import CompletionState

_GLOBAL_VERSION_KEYS = {
    "artifact_contract_version",
    "rng_scheme_version",
    "outcome_schema_version",
    "schema_version",
    "estimand_version",
    "conditioning_version",
}
_CONTROL_SUFFIXES = (".done.json", ".yaml", ".yml")


@dataclass(frozen=True, slots=True)
class CapturedV3Inputs:
    """Pickle-safe upstream identities resolved once by the parent process."""

    sources: tuple[SourceArtifactIdentity, ...]
    manifests: tuple[ManifestRootIdentity, ...]
    source_paths: tuple[tuple[str, str], ...]
    manifest_paths: tuple[tuple[str, str, str], ...]
    controls: tuple[tuple[str, str, str], ...]

    @property
    def designs(self) -> dict[str, str]:
        return {role: digest for role, _path, digest in self.controls}


def is_v3_config(cfg: AppConfig | None) -> bool:
    """Return whether *cfg* explicitly requests authenticated contract v3."""

    return (
        cfg is not None
        and cfg.artifact_contract.artifact_contract_version == ARTIFACT_CONTRACT_VERSION
    )


def _flatten_field_paths(value: Mapping[str, Any], prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for key, item in sorted(value.items()):
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            paths.extend(_flatten_field_paths(item, path))
        else:
            paths.append(path)
    return tuple(paths)


def _stage_field_paths(cfg: AppConfig, stage_key: str) -> tuple[str, ...]:
    """Expand the registry-owned cache scope into explicit public fields."""

    from farkle.analysis.stage_registry import resolve_stage_definition

    public = effective_config_dict(cfg)
    paths: list[str] = []
    for scoped in resolve_stage_definition(stage_key).cache_scope:
        cursor: Any = public
        for part in scoped.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                raise ArtifactContractError(
                    f"stage cache scope {scoped!r} is absent for {stage_key}"
                )
            cursor = cursor[part]
        if isinstance(cursor, Mapping):
            paths.extend(_flatten_field_paths(cursor, scoped))
        else:
            paths.append(scoped)
    if not paths:
        paths = list(_flatten_field_paths(public))
    return tuple(sorted(set(paths)))


def _stage_key_for_path(cfg: AppConfig, path: Path, producer: str) -> str:
    resolved = path.resolve()
    if resolved == cfg.results_root.resolve() or cfg.results_root.resolve() in resolved.parents:
        analysis_root = cfg.analysis_dir.resolve()
        if resolved != analysis_root and analysis_root not in resolved.parents:
            return "simulation"
    matches: list[tuple[int, str]] = []
    for placement in cfg.stage_layout.placements:
        root = (cfg.analysis_dir / placement.folder_name).resolve()
        if resolved == root or root in resolved.parents:
            matches.append((len(root.parts), placement.definition.key))
    if matches:
        return max(matches)[1]
    normalized = producer.lower()
    aliases = {
        "trueskill_screening": "trueskill",
        "hgb": "hgb",
        "metrics": "metrics",
        "candidate_family": "candidate_freeze",
    }
    stage_key = aliases.get(normalized, normalized)
    try:
        cfg.stage_dir(stage_key, create=False)
    except KeyError as exc:
        raise ArtifactContractError(f"cannot derive canonical v3 stage for {path}") from exc
    return stage_key


def _method_versions(
    cfg: AppConfig,
    *,
    stage_key: str,
    metadata: ArtifactSidecar,
) -> tuple[dict[str, int], int]:
    versions: dict[str, int] = {}
    for key, value in cfg.freshness_key().items():
        if (
            key.endswith("_version")
            and key not in _GLOBAL_VERSION_KEYS
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            versions[key] = int(value)
    parameters = metadata.method_contract.get("parameters") or {}
    for key, value in parameters.items():
        if (
            key.endswith("_version")
            and key not in _GLOBAL_VERSION_KEYS
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            versions[key] = int(value)
    hgb_versions: dict[str, int] = {}
    if stage_key == "hgb":
        # Import lazily to avoid making the authenticated-contract primitives
        # depend on analysis modules during module initialization.  HGB owns
        # these current method identities; completion classification must use
        # the same values as the stage wrapper so a method bump is stale.
        from farkle.analysis import hgb_feat

        hgb_versions = {
            "hgb_method_version": hgb_feat.HGB_METHOD_VERSION,
            "hgb_rng_method_version": hgb_feat.HGB_RNG_METHOD_VERSION,
            "hgb_fold_construction_version": hgb_feat.HGB_FOLD_CONSTRUCTION_VERSION,
        }
    stage_versions = {
        "simulation": {"tournament_method_version": TOURNAMENT_METHOD_VERSION},
        "ingest": {"tournament_method_version": TOURNAMENT_METHOD_VERSION},
        "rng_diagnostics": {"rng_diagnostic_method_version": 4},
        "trueskill": {
            "trueskill_method_version": 3,
            "trueskill_diagnostic_method_version": 2,
        },
        "hgb": hgb_versions,
        "root_stability": {"root_stability_method_version": 3},
        "candidate_freeze": {
            "candidate_family_version": cfg.artifact_contract.candidate_family_version
        },
        "h2h_power": {"h2h_method_version": 2, "h2h_power_method_version": 2},
        "h2h_execute": {"h2h_method_version": 2, "h2h_execution_method_version": 2},
        "h2h_inference": {
            "h2h_method_version": 2,
            "h2h_inference_method_version": 2,
        },
        "h2h_digest": {"h2h_method_version": 2, "dominance_method_version": 1},
        "agreement": {"structure_agreement_method_version": 1},
        "reporting": {
            "structure_report_contract_version": 4,
            "migration_report_version": 3,
            "reporting_method_version": 1,
        },
    }
    versions.update(stage_versions.get(stage_key, {}))
    explicit = parameters.get("method_version")
    if isinstance(explicit, int) and not isinstance(explicit, bool):
        method_version = explicit
    else:
        candidates = [
            value
            for key, value in versions.items()
            if key.startswith(stage_key) and key.endswith("_method_version")
        ]
        method_version = candidates[0] if candidates else 1
    versions.setdefault(f"{stage_key}_operation_method_version", method_version)
    return dict(sorted(versions.items())), method_version


def _versions(
    cfg: AppConfig,
    *,
    stage_key: str,
    metadata: ArtifactSidecar,
) -> tuple[VersionIdentity, int]:
    contract = cfg.artifact_contract
    accepted = {
        "artifact contract": contract.artifact_contract_version,
        "RNG scheme": cfg.rng.scheme_version,
        "outcome schema": OUTCOME_SCHEMA_VERSION,
        "derived schema": contract.schema_version,
        "estimand": contract.estimand_version,
        "conditioning": contract.conditioning_version,
    }
    expected = {
        "artifact contract": 3,
        "RNG scheme": 2,
        "outcome schema": 2,
        "derived schema": 2,
        "estimand": 2,
        "conditioning": 2,
    }
    if accepted != expected:
        raise ArtifactContractError(
            f"incomplete authenticated-v3 version identity: {accepted}; expected {expected}"
        )
    method_versions, method_version = _method_versions(
        cfg,
        stage_key=stage_key,
        metadata=metadata,
    )
    return (
        VersionIdentity(
            artifact_contract_version=3,
            lifecycle_contract_version=LIFECYCLE_CONTRACT_VERSION,
            rng_scheme_version=2,
            outcome_schema_version=2,
            schema_version=2,
            estimand_version=2,
            conditioning_version=2,
            method_versions=method_versions,
        ),
        method_version,
    )


def _typed_method(
    cfg: AppConfig,
    metadata: ArtifactSidecar,
    *,
    method_version: int,
) -> MethodContract:
    parameters = metadata.method_contract.get("parameters") or {}

    def _optional_int(name: str) -> int | None:
        value = parameters.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
        return None

    raw_lags = parameters.get("normalized_lags")
    rng_lags = (
        tuple(int(value) for value in raw_lags)
        if isinstance(raw_lags, (list, tuple))
        and all(isinstance(value, int) and not isinstance(value, bool) for value in raw_lags)
        else ()
    )

    def _hash(name: str) -> str | None:
        value = parameters.get(name)
        return value if isinstance(value, str) and len(value) == 64 else None

    weights = (
        None
        if metadata.k_weights is None
        else tuple((int(key), float(value)) for key, value in sorted(metadata.k_weights.items()))
    )
    roots = tuple(sorted({int(value) for value in cfg.sim.seed_list or [cfg.sim.seed]}))
    return MethodContract(
        procedure=metadata.operation,
        method_version=method_version,
        baseline=metadata.baseline,
        replication_unit=metadata.replication_unit,
        k_weights=weights,
        multiplicity=str(parameters.get("multiplicity", cfg.freshness_key()["multiplicity"])),
        family_hash=_hash("family_hash"),
        schedule_hash=_hash("schedule_hash"),
        practical_margin=(
            float(parameters["practical_margin"])
            if isinstance(parameters.get("practical_margin"), (int, float))
            else None
        ),
        equivalence_margin=(
            float(parameters["equivalence_margin"])
            if isinstance(parameters.get("equivalence_margin"), (int, float))
            else None
        ),
        ordinary_alpha=(
            float(parameters["ordinary_alpha"])
            if isinstance(parameters.get("ordinary_alpha"), (int, float))
            else None
        ),
        simultaneous_alpha=(
            float(parameters["simultaneous_alpha"])
            if isinstance(parameters.get("simultaneous_alpha"), (int, float))
            else None
        ),
        conditioning=metadata.conditioning,
        source_scope=metadata.source_scope,
        root_seeds=roots,
        player_counts=tuple(metadata.player_counts),
        required_player_counts=tuple(metadata.required_player_counts),
        weighted_quantity=metadata.weighted_quantity,
        support_count_role=metadata.support_count_role,
        uncertainty_method=metadata.uncertainty_method,
        k_aggregation_method=metadata.k_aggregation_method,
        missing_cell_policy=metadata.missing_cell_policy,
        seed_scope=metadata.seed_scope,
        consistency_columns=tuple(metadata.consistency_columns),
        grouping_keys=tuple(metadata.grouping_keys),
        semantic_contract_sha256=identity_sha256(metadata.method_contract),
        rng_effective_matchup_group_cap=_optional_int("effective_matchup_group_cap"),
        rng_diagnostic_lags=rng_lags,
        rng_tracked_matchup_group_count=_optional_int("tracked_matchup_group_count"),
        rng_skipped_matchup_group_count=_optional_int("skipped_matchup_group_count"),
        rng_skipped_matchup_row_count=_optional_int("skipped_matchup_row_count"),
    )


def _source_role(source: AuthenticatedSidecar) -> str:
    location = source.artifact.location
    root_bits = "_".join(str(value) for value in source.method_contract.root_seeds) or "none"
    k = "all" if location.player_count is None else str(location.player_count)
    relative = location.relative_path.replace("/", ".").replace("\\", ".")
    return f"artifact.{location.stage_key}.{location.scope}." f"k_{k}.roots_{root_bits}.{relative}"


def _manifest_role(source: Any) -> str:
    location = source.location
    k = "all" if location.player_count is None else str(location.player_count)
    relative = location.relative_path.replace("/", ".").replace("\\", ".")
    return f"manifest.{location.stage_key}.{location.scope}.k_{k}.{relative}"


def _capture_inputs(
    metadata: ArtifactSidecar,
) -> tuple[
    tuple[SourceArtifactIdentity, ...],
    tuple[ManifestRootIdentity, ...],
    dict[str, Path],
    dict[str, tuple[Path, Path]],
    dict[str, str],
]:
    captured = metadata._captured_v3_inputs
    if captured is not None:
        if not isinstance(captured, CapturedV3Inputs):
            raise ArtifactContractError("invalid captured v3 input snapshot")
        if metadata._cfg is None:
            raise ArtifactContractError("captured v3 inputs require their owning config")
        captured_source_paths = {role: Path(path) for role, path in captured.source_paths}
        captured_manifest_paths = {
            role: (Path(path), Path(adjacent)) for role, path, adjacent in captured.manifest_paths
        }
        for source in captured.sources:
            path = captured_source_paths.get(source.logical_role)
            if path is None:
                raise ArtifactContractError("captured source path inventory is incomplete")
            loaded = validate_authenticated_artifact_metadata(
                path,
                cfg=metadata._cfg,
                expected_sidecar_sha256=source.sidecar_sha256,
            )
            if (
                loaded.artifact != source.artifact
                or loaded.sidecar_contract_sha256 != source.sidecar_contract_sha256
            ):
                raise ArtifactContractError(
                    f"captured source identity changed: {source.logical_role}"
                )
        for captured_manifest_identity in captured.manifests:
            paths = captured_manifest_paths.get(captured_manifest_identity.logical_role)
            if paths is None:
                raise ArtifactContractError("captured manifest path inventory is incomplete")
            path, adjacent = paths
            loaded_manifest = load_immutable_manifest_sidecar(path)
            if (
                loaded_manifest.location != captured_manifest_identity.location
                or loaded_manifest.manifest_sha256 != captured_manifest_identity.manifest_sha256
                or loaded_manifest.sidecar_contract_sha256
                != captured_manifest_identity.sidecar_contract_sha256
                or loaded_manifest.summary != captured_manifest_identity.summary
                or sha256_file(path) != captured_manifest_identity.manifest_sha256
                or sha256_file(adjacent) != captured_manifest_identity.sidecar_sha256
            ):
                raise ArtifactContractError(
                    "captured manifest identity changed: "
                    f"{captured_manifest_identity.logical_role}"
                )
        for role, raw_path, digest in captured.controls:
            path = Path(raw_path)
            if not path.is_file() or sha256_file(path) != digest:
                raise ArtifactContractError(f"captured control identity changed: {role}")
        return (
            captured.sources,
            captured.manifests,
            captured_source_paths,
            captured_manifest_paths,
            captured.designs,
        )

    sources: list[SourceArtifactIdentity] = []
    manifests: list[ManifestRootIdentity] = []
    source_paths: dict[str, Path] = {}
    manifest_paths: dict[str, tuple[Path, Path]] = {}
    designs: dict[str, str] = {}
    if metadata._cfg is None:
        raise ArtifactContractError("authenticated source capture requires its owning config")
    for raw_path in metadata.source_artifacts:
        path = Path(raw_path)
        adjacent = sidecar_path(path)
        if adjacent.exists():
            payload = read_json_file_with_retry(adjacent)
            version = (
                payload.get("artifact_contract_version") if isinstance(payload, dict) else None
            )
            if version != 3:
                raise ArtifactContractError(
                    f"contract-v2 source cannot satisfy a v3 publication: {path}"
                )
            if "manifest_contract_version" in payload:
                manifest = load_immutable_manifest_sidecar(path)
                manifest_versions = manifest.stage_identity.versions
                if (
                    manifest_versions.artifact_contract_version,
                    manifest_versions.rng_scheme_version,
                    manifest_versions.outcome_schema_version,
                    manifest_versions.schema_version,
                    manifest_versions.estimand_version,
                    manifest_versions.conditioning_version,
                ) != (3, 2, 2, 2, 2, 2):
                    raise ArtifactContractError(
                        f"manifest has incompatible v3 version identity: {path}"
                    )
                role = _manifest_role(manifest)
                captured_manifest = ManifestRootIdentity(
                    logical_role=role,
                    location=manifest.location,
                    manifest_sha256=manifest.manifest_sha256,
                    sidecar_sha256=sha256_file(adjacent),
                    sidecar_contract_sha256=manifest.sidecar_contract_sha256,
                    summary=manifest.summary,
                )
                manifests.append(captured_manifest)
                manifest_paths[role] = (path, adjacent)
            else:
                loaded = validate_authenticated_artifact_unbound(
                    path,
                    validate_provenance=False,
                )
                expected_global = (
                    3,
                    2,
                    2,
                    2,
                    2,
                    2,
                )
                actual_global = (
                    loaded.versions.artifact_contract_version,
                    loaded.versions.rng_scheme_version,
                    loaded.versions.outcome_schema_version,
                    loaded.versions.schema_version,
                    loaded.versions.estimand_version,
                    loaded.versions.conditioning_version,
                )
                if actual_global != expected_global:
                    raise ArtifactContractError(
                        f"source has incompatible v3 version identity: {path}"
                    )
                role = _source_role(loaded)
                captured_source = capture_source_artifact_unbound(path, logical_role=role)
                sources.append(captured_source)
                source_paths[role] = path
            continue
        if path.name.endswith(_CONTROL_SUFFIXES):
            if not path.is_file():
                raise ArtifactContractError(f"required control identity is missing: {path}")
            designs[f"control:{path.name}:{len(designs):04d}"] = sha256_file(path)
            continue
        raise ArtifactContractError(
            f"v3 source artifact is missing an authenticated sidecar: {path}"
        )
    sources.sort(key=lambda item: item.logical_role)
    manifests.sort(key=lambda item: item.logical_role)
    source_paths = {item.logical_role: source_paths[item.logical_role] for item in sources}
    manifest_paths = {item.logical_role: manifest_paths[item.logical_role] for item in manifests}
    return (
        tuple(sources),
        tuple(manifests),
        source_paths,
        manifest_paths,
        designs,
    )


def capture_v3_inputs(metadata: ArtifactSidecar) -> CapturedV3Inputs:
    """Resolve and authenticate a metadata template's inputs exactly once."""

    if metadata._captured_v3_inputs is not None:
        captured = metadata._captured_v3_inputs
        if not isinstance(captured, CapturedV3Inputs):
            raise ArtifactContractError("invalid captured v3 input snapshot")
        return captured
    sources, manifests, source_paths, manifest_paths, designs = _capture_inputs(metadata)
    control_paths: list[tuple[str, str, str]] = []
    control_index = 0
    for raw_path in metadata.source_artifacts:
        path = Path(raw_path)
        if sidecar_path(path).exists() or not path.name.endswith(_CONTROL_SUFFIXES):
            continue
        role = f"control:{path.name}:{control_index:04d}"
        digest = designs.get(role)
        if digest is None:
            raise ArtifactContractError(f"captured control identity is missing: {path}")
        control_paths.append((role, str(path), digest))
        control_index += 1
    return CapturedV3Inputs(
        sources=sources,
        manifests=manifests,
        source_paths=tuple((role, str(path)) for role, path in source_paths.items()),
        manifest_paths=tuple(
            (role, str(paths[0]), str(paths[1])) for role, paths in manifest_paths.items()
        ),
        controls=tuple(control_paths),
    )


def _format_identity(path: Path, staged_path: Path) -> ArtifactFormatIdentity | None:
    if path.suffix.lower() == ".parquet":
        return None
    suffix = path.suffix.lower()
    media_type = {
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".md": "text/markdown; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
        ".png": "image/png",
        ".npy": "application/x-npy",
        ".pkl": "application/x-python-pickle",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
    }.get(suffix, "application/octet-stream")
    structural: str | None = None
    if suffix == ".json":
        payload = json.loads(staged_path.read_text(encoding="utf-8"))
        structural = identity_sha256(_json_shape(payload))
    elif suffix == ".jsonl":
        text = staged_path.read_text(encoding="utf-8")
        try:
            shapes = {json.dumps(_json_shape(json.loads(text)), sort_keys=True)}
        except json.JSONDecodeError:
            shapes = {
                json.dumps(_json_shape(json.loads(line)), sort_keys=True)
                for line in text.splitlines()
                if line.strip()
            }
        structural = identity_sha256(sorted(shapes))
    elif suffix in {".md", ".txt", ".yaml", ".yml"}:
        staged_path.read_text(encoding="utf-8")
    elif suffix == ".png":
        if not staged_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ArtifactContractError(f"invalid PNG artifact: {path}")
    elif suffix == ".npy":
        try:
            array = np.load(staged_path, mmap_mode="r", allow_pickle=False)
            structural = identity_sha256({"dtype": array.dtype.descr, "shape": list(array.shape)})
            del array
        except (OSError, TypeError, ValueError) as exc:
            raise ArtifactContractError(f"invalid NumPy artifact: {path}") from exc
    return ArtifactFormatIdentity(
        media_type=media_type,
        format_version=1,
        structural_schema_sha256=structural,
    )


def _json_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_json_shape(value[0])] if value else []
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    raise ArtifactContractError(f"unsupported JSON value {type(value).__name__}")


def publish_staged_v3_from_metadata(
    staged_path: Path,
    final_path: Path,
    metadata: ArtifactSidecar,
) -> ArtifactSidecar:
    """Translate one legacy metadata builder into an authenticated v3 write."""

    cfg = metadata._cfg
    if not is_v3_config(cfg):
        raise ArtifactContractError("v3 publication requires its owning AppConfig")
    assert cfg is not None
    stage_key = _stage_key_for_path(cfg, final_path, metadata.producer)
    location = derive_canonical_location(cfg, final_path, stage_key=stage_key)
    if metadata.scope != location.scope:
        raise ArtifactContractError(
            f"declared scope {metadata.scope!r} does not match canonical "
            f"location scope {location.scope!r}: {final_path}"
        )
    if location.scope == ArtifactScope.BY_K.value:
        declared_counts = {int(value) for value in metadata.player_counts}
        if location.player_count not in declared_counts:
            raise ArtifactContractError(
                "canonical by_k location is absent from the declared player-count support"
            )
    versions, method_version = _versions(cfg, stage_key=stage_key, metadata=metadata)
    method = _typed_method(cfg, metadata, method_version=method_version)
    sources, manifests, source_paths, manifest_paths, designs = _capture_inputs(metadata)
    if cfg._run_lineage_sha256 is not None:
        designs["run_lineage_sha256"] = cfg._run_lineage_sha256
    if cfg._game_profile_sha256 is not None:
        designs["game_profile_sha256"] = cfg._game_profile_sha256
    for name in ("family_hash", "schedule_hash"):
        value = getattr(method, name)
        if value is not None:
            designs[name] = value
    effective_config_dict(cfg)
    config_identity = stage_config_identity(
        cfg,
        stage_key=stage_key,
        field_paths=_stage_field_paths(cfg, stage_key),
    )
    code = cfg._code_identity
    if code is None:
        code = resolve_code_identity(
            Path(__file__).resolve().parents[3],
            policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY,
        )
    upstream = tuple(item.sha256 for item in sources) + tuple(item.sha256 for item in manifests)
    stage_identity = make_stage_identity(
        stage_key=stage_key,
        stage_cache_key_version=cfg.stage_cache_key_version(stage_key),
        stage_config=config_identity,
        versions=versions,
        code=code,
        method_contract=method,
        upstream_identity_sha256=upstream,
        immutable_design_identities=dict(sorted(designs.items())),
    )
    authenticated = publish_staged_authenticated_artifact_atomic(
        staged_path,
        final_path,
        cfg=cfg,
        location=location,
        stage_identity=stage_identity,
        method_contract=method,
        format_identity=_format_identity(final_path, staged_path),
        sources=sources,
        manifest_roots=manifests,
        source_paths=source_paths,
        manifest_paths=manifest_paths,
        validate_unbound_sources=True,
    )
    return _compatibility_view(final_path, metadata=metadata, authenticated=authenticated)


def publish_native_manifest_v3(
    path: Path,
    *,
    cfg: AppConfig,
    stage_key: str,
    entries: list[ManifestEntry],
    source_paths: list[Path],
    operation: str,
    player_counts: list[int],
    native_records: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Publish native manifest bytes with a coordinate-sorted immutable root."""

    synthetic = ArtifactSidecar(
        artifact_contract_version=3,
        estimand_version=2,
        schema_version=2,
        artifact_name=path.name,
        producer=stage_key,
        scope=(
            ArtifactScope.DIAGNOSTICS.value
            if stage_key == "simulation"
            else ArtifactScope.BY_K.value
        ),
        source_scope=ArtifactScope.DIAGNOSTICS.value,
        operation=operation,
        method_contract={"kind": "operation", "procedure": operation},
        baseline="coordinate_identity",
        weighted_quantity="immutable_shard_inventory",
        k_aggregation_method="none",
        k_weights=None,
        support_count_role="coordinate_sorted_shards",
        uncertainty_method="none",
        replication_unit="manifest_coordinate",
        conditioning="unconditional",
        consistency_columns=[],
        source_artifacts=[str(item) for item in source_paths],
        grouping_keys=[],
        player_counts=player_counts,
        required_player_counts=player_counts,
        missing_cell_policy="fail",
        seed_scope="single_root",
        rng_scheme_version=2,
        config_hash=cfg.config_sha or "",
        input_manifest_hashes=[],
        code_revision="authenticated_v3",
        _cfg=cfg,
    )
    versions, method_version = _versions(
        cfg,
        stage_key=stage_key,
        metadata=synthetic,
    )
    method = _typed_method(cfg, synthetic, method_version=method_version)
    sources, manifests, _, _, designs = _capture_inputs(synthetic)
    if cfg._run_lineage_sha256 is not None:
        designs["run_lineage_sha256"] = cfg._run_lineage_sha256
    if cfg._game_profile_sha256 is not None:
        designs["game_profile_sha256"] = cfg._game_profile_sha256
    if manifests:
        raise ArtifactContractError("native shard manifest sources must be ordinary artifacts")
    config_identity = stage_config_identity(
        cfg,
        stage_key=stage_key,
        field_paths=_stage_field_paths(cfg, stage_key),
    )
    code = cfg._code_identity or resolve_code_identity(
        Path(__file__).resolve().parents[3],
        policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY,
    )
    stage_identity = make_stage_identity(
        stage_key=stage_key,
        stage_cache_key_version=cfg.stage_cache_key_version(stage_key),
        stage_config=config_identity,
        versions=versions,
        code=code,
        method_contract=method,
        upstream_identity_sha256=tuple(item.sha256 for item in sources),
        immutable_design_identities=dict(sorted(designs.items())),
    )

    def _write_native(staged: Path) -> None:
        if native_records is not None:
            with staged.open("wb") as handle:
                for record in native_records:
                    handle.write(canonical_json_bytes(dict(record)) + b"\n")
            return
        with path.open("rb") as source, staged.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)

    publish_immutable_manifest_bytes_atomic(
        path,
        cfg=cfg,
        location=derive_canonical_location(cfg, path, stage_key=stage_key),
        stage_identity=stage_identity,
        entries=entries,
        write_data=_write_native,
    )


def _compatibility_view(
    path: Path,
    *,
    metadata: ArtifactSidecar | None = None,
    authenticated: AuthenticatedSidecar | None = None,
) -> ArtifactSidecar:
    loaded = authenticated or validate_authenticated_artifact_unbound(
        path, validate_provenance=False
    )
    method = loaded.method_contract
    location = loaded.artifact.location
    template = metadata
    return ArtifactSidecar(
        artifact_contract_version=3,
        estimand_version=loaded.versions.estimand_version,
        schema_version=loaded.versions.schema_version,
        artifact_name=path.name,
        producer=location.stage_key if template is None else template.producer,
        scope=location.scope,
        source_scope=method.source_scope or location.scope,
        operation=loaded.artifact.logical_operation,
        method_contract={
            "kind": "operation",
            "procedure": loaded.artifact.logical_operation,
            "parameters": {"method_version": method.method_version},
        },
        baseline=method.baseline,
        weighted_quantity=method.weighted_quantity,
        k_aggregation_method=method.k_aggregation_method,
        k_weights=(
            None
            if method.k_weights is None
            else {str(key): value for key, value in method.k_weights}
        ),
        support_count_role=method.support_count_role,
        uncertainty_method=method.uncertainty_method,
        replication_unit=method.replication_unit,
        conditioning=method.conditioning,
        consistency_columns=list(method.consistency_columns),
        source_artifacts=[],
        grouping_keys=list(method.grouping_keys),
        player_counts=list(method.player_counts),
        required_player_counts=list(method.required_player_counts),
        missing_cell_policy=method.missing_cell_policy,
        seed_scope=method.seed_scope,
        rng_scheme_version=loaded.versions.rng_scheme_version,
        config_hash=loaded.stage_identity.stage_config.sha256,
        input_manifest_hashes=[item.manifest_sha256 for item in loaded.manifest_roots],
        code_revision=loaded.stage_identity.code.commit,
        artifact_sha256=loaded.artifact.content_sha256,
        artifact_size_bytes=loaded.artifact.byte_length,
        _cfg=None,
    )


def validate_v3_compat(
    path: Path | str,
    *,
    expected: Mapping[str, Any] | None = None,
) -> ArtifactSidecar:
    """Fully validate v3 bytes/schema, then expose the retiring v2 read view."""

    artifact = Path(path)
    metadata = _compatibility_view(artifact)
    for key, wanted in (expected or {}).items():
        if key == "method_contract" and isinstance(wanted, Mapping):
            loaded = validate_authenticated_artifact_unbound(
                artifact,
                validate_provenance=False,
            )
            if loaded.method_contract.semantic_contract_sha256 != identity_sha256(wanted):
                raise ArtifactContractError(
                    f"incompatible sidecar for {artifact}: method contract differs"
                )
            continue
        if not hasattr(metadata, key):
            raise ArtifactContractError(f"unknown sidecar expectation: {key}")
        actual = getattr(metadata, key)
        if actual != wanted:
            raise ArtifactContractError(
                f"incompatible sidecar for {artifact}: {key}={actual!r}, expected {wanted!r}"
            )
    return metadata


def _expanded_files(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(
                child
                for child in sorted(path.rglob("*"), key=lambda item: item.as_posix())
                if child.is_file()
                and not child.name.endswith(".sidecar.json")
                and not child.name.endswith(".done.json")
                and not child.name.startswith("._")
            )
        else:
            expanded.append(path)
    unique = {path.resolve(): path for path in expanded}
    return [unique[key] for key in sorted(unique, key=str)]


def _input_designs(paths: list[Path]) -> dict[str, str]:
    designs: dict[str, str] = {}
    for index, path in enumerate(_expanded_files(paths)):
        if path.is_file():
            designs[f"input:{index:04d}:{path.name}"] = sha256_file(path)
    return designs


def _root_owned_candidate(
    cfg: AppConfig,
    root: Path,
    location: CanonicalArtifactLocation,
) -> Path | None:
    if location.stage_key == "simulation":
        return root / location.relative_path
    folder = cfg.root_input_stage_folder(location.stage_key)
    if folder is None:
        return None
    scope_parts = [location.scope]
    if location.scope == ArtifactScope.BY_K.value:
        if location.player_count is None:
            return None
        scope_parts.append(f"{location.player_count}p")
    return root / cfg.io.analysis_subdir / folder / Path(*scope_parts) / location.relative_path


def _find_root_owned_source(
    cfg: AppConfig,
    source: SourceArtifactIdentity,
) -> Path | None:
    """Resolve an authenticated root source only within the declared pair root."""

    pair_root = cfg.analysis_dir.parent
    roots = [path for path in pair_root.iterdir() if path.is_dir()] if pair_root.is_dir() else []
    candidates: list[Path] = []
    for root in roots:
        candidate = _root_owned_candidate(cfg, root, source.artifact.location)
        if candidate is None or not candidate.is_file():
            continue
        try:
            current = capture_source_artifact_unbound(
                candidate,
                logical_role=source.logical_role,
            )
        except Exception:
            continue
        if current == source:
            candidates.append(candidate)
    return candidates[0] if len(candidates) == 1 else None


def _find_root_owned_manifest(
    cfg: AppConfig,
    manifest: ManifestRootIdentity,
) -> tuple[Path, Path] | None:
    pair_root = cfg.analysis_dir.parent
    roots = [path for path in pair_root.iterdir() if path.is_dir()] if pair_root.is_dir() else []
    candidates: list[tuple[Path, Path]] = []
    for root in roots:
        candidate = _root_owned_candidate(cfg, root, manifest.location)
        if candidate is None:
            continue
        adjacent = sidecar_path(candidate)
        if not candidate.is_file() or not adjacent.is_file():
            continue
        try:
            loaded = load_immutable_manifest_sidecar(candidate)
        except Exception:
            continue
        if (
            loaded.location == manifest.location
            and sha256_file(candidate) == manifest.manifest_sha256
            and sha256_file(adjacent) == manifest.sidecar_sha256
            and loaded.sidecar_contract_sha256 == manifest.sidecar_contract_sha256
            and loaded.summary == manifest.summary
        ):
            candidates.append((candidate, adjacent))
    return candidates[0] if len(candidates) == 1 else None


def _completion_contract(
    cfg: AppConfig,
    *,
    stage_key: str,
    inputs: list[Path],
    outputs: list[Path],
    deep_verify_outputs: bool = True,
) -> tuple[
    Any,
    tuple[CompletionOutputIdentity, ...],
    tuple[CanonicalArtifactLocation, ...],
    dict[str, Path],
    dict[str, tuple[Path, Path]],
]:
    current_global = (
        cfg.artifact_contract.artifact_contract_version,
        cfg.rng.scheme_version,
        OUTCOME_SCHEMA_VERSION,
        cfg.artifact_contract.schema_version,
        cfg.artifact_contract.estimand_version,
        cfg.artifact_contract.conditioning_version,
    )
    if current_global != (3, 2, 2, 2, 2, 2):
        raise ArtifactContractError(
            f"completion requires accepted release identity; found {current_global}"
        )
    output_files = _expanded_files(outputs)
    if not output_files:
        raise ArtifactContractError("authenticated completion requires stage outputs")
    output_identities: list[CompletionOutputIdentity] = []
    locations: list[CanonicalArtifactLocation] = []
    sidecars: list[AuthenticatedSidecar] = []
    # Every ordinary input or immutable manifest is already bound by each
    # output sidecar's typed provenance. Completion identity must be
    # reconstructible from its authoritative output inventory alone because
    # stage-health consumers intentionally do not rediscover ad-hoc input
    # lists.
    designs: dict[str, str] = {}
    if cfg._run_lineage_sha256 is not None:
        designs["run_lineage_sha256"] = cfg._run_lineage_sha256
    if cfg._game_profile_sha256 is not None:
        designs["game_profile_sha256"] = cfg._game_profile_sha256
    for path in output_files:
        adjacent = sidecar_path(path)
        if not adjacent.is_file():
            raise ArtifactContractError(
                f"authenticated completion requires a valid sidecar for {path}"
            )
        payload = read_json_file_with_retry(adjacent)
        if not isinstance(payload, dict) or payload.get("artifact_contract_version") != 3:
            raise ArtifactContractError(
                f"contract-v2 artifact cannot satisfy v3 completion: {path}"
            )
        if "manifest_contract_version" in payload:
            manifest = load_immutable_manifest_sidecar(path)
            location = derive_canonical_location(cfg, path, stage_key=stage_key)
            if manifest.location != location:
                raise ArtifactContractError(f"manifest scope/path identity mismatch: {path}")
            root = ManifestRootIdentity(
                logical_role=_manifest_role(manifest),
                location=manifest.location,
                manifest_sha256=manifest.manifest_sha256,
                sidecar_sha256=sha256_file(adjacent),
                sidecar_contract_sha256=manifest.sidecar_contract_sha256,
                summary=manifest.summary,
            )
            output_identities.append(
                CompletionOutputIdentity(
                    artifact=None,
                    manifest=root,
                    sidecar_sha256=sha256_file(adjacent),
                )
            )
            locations.append(location)
            designs[f"output:{location.relative_path}"] = sha256_file(adjacent)
            continue
        sidecar = (
            validate_authenticated_artifact_unbound(path, validate_provenance=False)
            if deep_verify_outputs
            else validate_authenticated_artifact_metadata(path)
        )
        if stage_key == "hgb":
            expected_hgb_versions, _method_version = _method_versions(
                cfg,
                stage_key=stage_key,
                metadata=_compatibility_view(path),
            )
            governed = {
                key: expected_hgb_versions[key]
                for key in (
                    "hgb_method_version",
                    "hgb_rng_method_version",
                    "hgb_fold_construction_version",
                )
            }
            recorded = sidecar.versions.method_versions
            if any(recorded.get(key) != value for key, value in governed.items()):
                raise ArtifactContractError(
                    f"artifact method/version identity is stale for {stage_key}: {path}"
                )
        location = derive_canonical_location(cfg, path, stage_key=stage_key)
        if sidecar.artifact.location != location:
            raise ArtifactContractError(f"artifact scope/path identity mismatch: {path}")
        sidecars.append(sidecar)
        output_identities.append(
            CompletionOutputIdentity(
                artifact=sidecar.artifact,
                sidecar_sha256=sha256_file(adjacent),
            )
        )
        locations.append(location)
        designs[f"output:{location.relative_path}"] = sha256_file(adjacent)

    versions = (
        sidecars[0].versions
        if sidecars
        else load_immutable_manifest_sidecar(output_files[0]).stage_identity.versions
    )
    config_identity = stage_config_identity(
        cfg,
        stage_key=stage_key,
        field_paths=_stage_field_paths(cfg, stage_key),
    )
    code = cfg._code_identity
    if code is None:
        code = resolve_code_identity(
            Path(__file__).resolve().parents[3],
            policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY,
        )
    method_version = next(iter(versions.method_versions.values()))
    method = MethodContract(
        procedure=f"{stage_key}_authenticated_completion",
        method_version=method_version,
        baseline="stage_output_inventory",
        replication_unit="stage_run",
        conditioning="authenticated_complete_outputs",
        root_seeds=tuple(sorted({int(value) for value in cfg.sim.seed_list or [cfg.sim.seed]})),
        player_counts=tuple(sorted({int(value) for value in cfg.sim.n_players_list})),
        required_player_counts=tuple(sorted({int(value) for value in cfg.sim.n_players_list})),
    )
    stage_identity = make_stage_identity(
        stage_key=stage_key,
        stage_cache_key_version=cfg.stage_cache_key_version(stage_key),
        stage_config=config_identity,
        versions=versions,
        code=code,
        method_contract=method,
        upstream_identity_sha256=(),
        immutable_design_identities=dict(sorted(designs.items())),
    )

    candidate_paths = _expanded_files([*inputs, *outputs])
    source_paths: dict[str, Path] = {}
    manifest_paths: dict[str, tuple[Path, Path]] = {}
    for path in candidate_paths:
        adjacent = sidecar_path(path)
        if not adjacent.is_file():
            continue
        payload = read_json_file_with_retry(adjacent)
        if not isinstance(payload, dict) or payload.get("artifact_contract_version") != 3:
            continue
        if "manifest_contract_version" in payload:
            manifest = load_immutable_manifest_sidecar(path)
            manifest_paths[_manifest_role(manifest)] = (path, adjacent)
        else:
            sidecar = (
                validate_authenticated_artifact_unbound(path, validate_provenance=False)
                if deep_verify_outputs
                else validate_authenticated_artifact_metadata(path)
            )
            source_paths[_source_role(sidecar)] = path

    for output_sidecar in sidecars:
        for source in output_sidecar.source_artifacts:
            if source.logical_role in source_paths:
                continue
            try:
                candidate = source.artifact.location.path(cfg)
            except (KeyError, ValueError):
                candidate = None
            if candidate is not None and candidate.is_file():
                current = capture_source_artifact_unbound(
                    candidate,
                    logical_role=source.logical_role,
                )
                if current == source:
                    source_paths[source.logical_role] = candidate
                    continue
            root_candidate = _find_root_owned_source(cfg, source)
            if root_candidate is not None:
                source_paths[source.logical_role] = root_candidate
        for manifest_root in output_sidecar.manifest_roots:
            if manifest_root.logical_role in manifest_paths:
                continue
            try:
                manifest_candidate: Path | None = manifest_root.location.path(cfg)
            except (KeyError, ValueError):
                manifest_candidate = None
            manifest_adjacent = (
                sidecar_path(manifest_candidate) if manifest_candidate is not None else None
            )
            if (
                manifest_candidate is not None
                and manifest_adjacent is not None
                and manifest_candidate.is_file()
                and manifest_adjacent.is_file()
            ):
                manifest_paths[manifest_root.logical_role] = (
                    manifest_candidate,
                    manifest_adjacent,
                )
                continue
            root_manifest_candidate = _find_root_owned_manifest(cfg, manifest_root)
            if root_manifest_candidate is not None:
                manifest_paths[manifest_root.logical_role] = root_manifest_candidate

    ordered = tuple(sorted(output_identities, key=lambda item: canonical_json_bytes(item.location)))
    ordered_locations = tuple(sorted(locations, key=canonical_json_bytes))
    return (
        stage_identity,
        ordered,
        ordered_locations,
        source_paths,
        manifest_paths,
    )


def write_v3_stage_completion(
    done_path: Path,
    *,
    cfg: AppConfig,
    stage_key: str,
    inputs: list[Path],
    outputs: list[Path],
    status: str,
) -> None:
    """Publish completion last, after every declared output authenticates."""

    if status not in {"success", "blocked_by_cap"}:
        raise ArtifactContractError(
            "v3 release stages publish only authenticated success or blocked-by-cap state"
        )
    deep_verify_outputs = stage_key != "simulation"
    stage_identity, identities, locations, source_paths, manifest_paths = _completion_contract(
        cfg,
        stage_key=stage_key,
        inputs=inputs,
        outputs=outputs,
        deep_verify_outputs=deep_verify_outputs,
    )
    completion = AuthenticatedCompletion(
        lifecycle_contract_version=LIFECYCLE_CONTRACT_VERSION,
        stage_identity_sha256=stage_identity.sha256,
        state=(
            CompletionState.COMPLETE_VALID.value
            if status == "success"
            else CompletionState.BLOCKED_BY_CAP.value
        ),
        outputs=identities,
    )
    # Validate the complete graph before making success observable.
    temporary = done_path.with_name(f".{done_path.name}.validation")
    try:
        write_authenticated_completion_atomic(temporary, completion)
        state = classify_authenticated_lifecycle(
            temporary,
            cfg=cfg,
            expected_stage_identity=stage_identity,
            required_locations=locations,
            source_paths=source_paths,
            manifest_paths=manifest_paths,
            deep_verify_artifacts=deep_verify_outputs,
        )
        expected_state = (
            CompletionState.COMPLETE_VALID
            if status == "success"
            else CompletionState.BLOCKED_BY_CAP
        )
        if state is not expected_state:
            raise ArtifactContractError(
                f"stage outputs did not authenticate before completion: {stage_key}"
            )
    finally:
        temporary.unlink(missing_ok=True)
    write_authenticated_completion_atomic(done_path, completion)


def resolve_v3_stage_state(
    done_path: Path,
    *,
    cfg: AppConfig,
    stage_key: str,
    inputs: list[Path],
    outputs: list[Path],
    partial_paths: list[Path],
    cap_reached: bool,
) -> CompletionState:
    """Classify a v3 lifecycle; old schema-4/v2 evidence is always stale."""

    if cap_reached:
        return CompletionState.BLOCKED_BY_CAP
    if done_path.exists() and not outputs:
        try:
            payload = read_json_file_with_retry(done_path)
            recorded_locations = [
                (item.get("artifact") or item.get("manifest"))["location"]
                for item in payload["outputs"]
            ]
            outputs = [
                CanonicalArtifactLocation(**location).path(cfg) for location in recorded_locations
            ]
        except (KeyError, TypeError, ValueError):
            return CompletionState.COMPLETE_STALE
    materialized = any(path.exists() for path in [*outputs, *partial_paths])
    if not done_path.exists():
        return CompletionState.PARTIAL_RESUMABLE if materialized else CompletionState.NOT_STARTED
    try:
        deep_verify_outputs = stage_key != "simulation"
        stage_identity, _, locations, source_paths, manifest_paths = _completion_contract(
            cfg,
            stage_key=stage_key,
            inputs=inputs,
            outputs=outputs,
            deep_verify_outputs=deep_verify_outputs,
        )
    except Exception:
        return CompletionState.COMPLETE_STALE
    return classify_authenticated_lifecycle(
        done_path,
        cfg=cfg,
        expected_stage_identity=stage_identity,
        required_locations=locations,
        partial_paths=partial_paths,
        source_paths=source_paths,
        manifest_paths=manifest_paths,
        deep_verify_artifacts=deep_verify_outputs,
    )


__all__ = [
    "CapturedV3Inputs",
    "capture_v3_inputs",
    "is_v3_config",
    "publish_staged_v3_from_metadata",
    "publish_native_manifest_v3",
    "resolve_v3_stage_state",
    "validate_v3_compat",
    "write_v3_stage_completion",
]
