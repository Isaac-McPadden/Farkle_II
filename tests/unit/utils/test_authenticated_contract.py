from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.config import AppConfig
from farkle.utils.artifact_contract import sha256_file, sidecar_path
from farkle.utils.authenticated_contract import (
    ARTIFACT_CONTRACT_VERSION,
    LIFECYCLE_CONTRACT_VERSION,
    ArtifactIdentity,
    ArtifactMismatchError,
    AuthenticatedCompletion,
    CanonicalArtifactLocation,
    CodeIdentity,
    CodeIdentityError,
    CodeIdentityPolicy,
    CompletionOutputIdentity,
    CorruptSidecarError,
    ManifestEntry,
    MethodContract,
    MissingSidecarError,
    VersionIdentity,
    arrow_schema_identity,
    capture_manifest_root,
    capture_source_artifact,
    classify_authenticated_lifecycle,
    compute_manifest_root,
    derive_canonical_location,
    finalize_missing_sidecar_atomic,
    identity_sha256,
    make_authenticated_sidecar,
    make_stage_identity,
    publish_authenticated_parquet_atomic,
    publish_immutable_manifest_atomic,
    resolve_code_identity,
    stage_config_identity,
    validate_authenticated_artifact,
    write_authenticated_completion_atomic,
)
from farkle.utils.stage_completion import CompletionState

_COMMIT = "a" * 40
_HASH_A = "a" * 64
_HASH_B = "b" * 64


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    config = AppConfig()
    config.io.results_dir_prefix = tmp_path / "results"
    config.sim.n_players_list = [2, 4]
    return config


def _versions(**changes: object) -> VersionIdentity:
    values: dict[str, object] = {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "lifecycle_contract_version": LIFECYCLE_CONTRACT_VERSION,
        "rng_scheme_version": 2,
        "outcome_schema_version": 2,
        "schema_version": 2,
        "estimand_version": 2,
        "conditioning_version": 2,
        "method_versions": {"test_method_version": 2},
    }
    values.update(changes)
    return VersionIdentity(**values)  # type: ignore[arg-type]


def _method(**changes: object) -> MethodContract:
    values: dict[str, object] = {
        "procedure": "hand_count",
        "method_version": 2,
        "baseline": "chance_rate_by_k",
        "replication_unit": "attempted_game",
        "k_weights": ((2, 0.5), (4, 0.5)),
        "multiplicity": "holm_all_frozen_pairs",
        "family_hash": _HASH_A,
        "schedule_hash": _HASH_B,
        "practical_margin": 0.03,
        "equivalence_margin": 0.01,
        "ordinary_alpha": 0.05,
        "simultaneous_alpha": 0.025,
    }
    values.update(changes)
    return MethodContract(**values)  # type: ignore[arg-type]


def _stage(
    cfg: AppConfig,
    *,
    method: MethodContract | None = None,
    versions: VersionIdentity | None = None,
    upstream: tuple[str, ...] = (),
) -> tuple[MethodContract, VersionIdentity, object]:
    resolved_method = method or _method()
    resolved_versions = versions or _versions()
    config_identity = stage_config_identity(
        cfg,
        stage_key="metrics",
        field_paths=("sim.n_players_list", "screening.resolution_delta"),
    )
    stage = make_stage_identity(
        stage_key="metrics",
        stage_cache_key_version=4,
        stage_config=config_identity,
        versions=resolved_versions,
        code=CodeIdentity(
            commit=_COMMIT,
            policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
            state="clean",
            dirty_fingerprint_sha256=None,
        ),
        method_contract=resolved_method,
        upstream_identity_sha256=upstream,
        immutable_design_identities={"family_hash": _HASH_A, "schedule_hash": _HASH_B},
    )
    return resolved_method, resolved_versions, stage


def _location(name: str = "summary.parquet", *, scope: str = "by_k") -> CanonicalArtifactLocation:
    return CanonicalArtifactLocation(
        stage_key="metrics",
        scope=scope,
        player_count=2 if scope == "by_k" else None,
        relative_path=name,
    )


def _publish(
    cfg: AppConfig,
    *,
    location: CanonicalArtifactLocation | None = None,
    table: pa.Table | None = None,
    method: MethodContract | None = None,
    versions: VersionIdentity | None = None,
    sources: tuple[object, ...] = (),
    source_paths: dict[str, Path] | None = None,
    source_configs: dict[str, AppConfig] | None = None,
    manifest_roots: tuple[object, ...] = (),
    manifest_paths: dict[str, tuple[Path, Path]] | None = None,
    manifest_configs: dict[str, AppConfig] | None = None,
):
    resolved_location = location or _location()
    upstream = tuple(source.sha256 for source in sources) + tuple(
        manifest.sha256 for manifest in manifest_roots
    )
    resolved_method, _, stage = _stage(
        cfg,
        method=method,
        versions=versions,
        upstream=upstream,
    )
    resolved_table = table or pa.table(
        {"strategy_id": pa.array([1, 2], type=pa.int32()), "wins": [3, 4]}
    )
    path = resolved_location.path(cfg)
    metadata = publish_authenticated_parquet_atomic(
        path,
        cfg=cfg,
        location=resolved_location,
        stage_identity=stage,  # type: ignore[arg-type]
        method_contract=resolved_method,
        sources=sources,  # type: ignore[arg-type]
        manifest_roots=manifest_roots,  # type: ignore[arg-type]
        source_paths=source_paths,
        source_configs=source_configs,
        manifest_paths=manifest_paths,
        manifest_configs=manifest_configs,
        write_data=lambda temporary: pq.write_table(resolved_table, temporary),
    )
    return path, metadata


def _rewrite_sidecar_with_declared_schema(
    artifact: Path,
    metadata,
    schema: pa.Schema,
) -> None:
    declared_schema = arrow_schema_identity(schema, schema_version=metadata.versions.schema_version)
    declared_artifact = replace(metadata.artifact, arrow_schema=declared_schema)
    altered = make_authenticated_sidecar(
        artifact=declared_artifact,
        stage_identity=metadata.stage_identity,
        method_contract=metadata.method_contract,
        sources=metadata.source_artifacts,
        manifest_roots=metadata.manifest_roots,
    )
    sidecar_path(artifact).write_bytes(altered.canonical_bytes())


def test_canonical_scope_is_derived_and_valid_bytes_fail_in_wrong_scope(cfg: AppConfig) -> None:
    artifact, _ = _publish(cfg)
    assert derive_canonical_location(cfg, artifact, stage_key="metrics") == _location()

    wrong_location = _location(scope="across_k")
    wrong_path = wrong_location.path(cfg)
    wrong_path.parent.mkdir(parents=True)
    shutil.copyfile(artifact, wrong_path)
    shutil.copyfile(sidecar_path(artifact), sidecar_path(wrong_path))

    with pytest.raises(ArtifactMismatchError, match="different canonical artifact location"):
        validate_authenticated_artifact(
            wrong_path,
            cfg=cfg,
            expected_location=wrong_location,
        )


@pytest.mark.parametrize(
    "declared_schema",
    [
        pa.schema(
            [
                pa.field("strategy_id", pa.int32(), nullable=True),
                pa.field("wins", pa.int64(), nullable=True),
                pa.field("fictitious", pa.string(), nullable=True),
            ]
        ),
        pa.schema(
            [
                pa.field("strategy_id", pa.int64(), nullable=True),
                pa.field("wins", pa.int64(), nullable=True),
            ]
        ),
        pa.schema(
            [
                pa.field("strategy_id", pa.int32(), nullable=False),
                pa.field("wins", pa.int64(), nullable=True),
            ]
        ),
    ],
    ids=("fictitious-column", "wrong-dtype", "wrong-nullability"),
)
def test_actual_arrow_schema_rejects_false_declarations(
    cfg: AppConfig, declared_schema: pa.Schema
) -> None:
    artifact, metadata = _publish(cfg)
    _rewrite_sidecar_with_declared_schema(artifact, metadata, declared_schema)

    with pytest.raises(ArtifactMismatchError, match="actual Arrow schema"):
        validate_authenticated_artifact(artifact, cfg=cfg, expected_location=_location())


def test_changed_source_bytes_and_sidecar_both_invalidate_derived_artifact(cfg: AppConfig) -> None:
    source_location = _location("source.parquet")
    source_path, _ = _publish(cfg, location=source_location)
    source = capture_source_artifact(
        source_path,
        cfg=cfg,
        expected_location=source_location,
        logical_role="attempt_rows",
    )
    derived_location = _location("derived.parquet")
    derived_path, _ = _publish(
        cfg,
        location=derived_location,
        sources=(source,),
        source_paths={"attempt_rows": source_path},
        source_configs={"attempt_rows": cfg},
    )

    original_source = source_path.read_bytes()
    source_path.write_bytes(original_source + b"changed")
    with pytest.raises(ArtifactMismatchError, match="source artifact bytes/schema changed"):
        validate_authenticated_artifact(
            derived_path,
            cfg=cfg,
            expected_location=derived_location,
            source_paths={"attempt_rows": source_path},
            source_configs={"attempt_rows": cfg},
        )

    source_path.write_bytes(original_source)
    sidecar_path(source_path).write_bytes(sidecar_path(source_path).read_bytes() + b" ")
    with pytest.raises(ArtifactMismatchError, match="source sidecar changed"):
        validate_authenticated_artifact(
            derived_path,
            cfg=cfg,
            expected_location=derived_location,
            source_paths={"attempt_rows": source_path},
            source_configs={"attempt_rows": cfg},
        )


@pytest.mark.parametrize(
    "expected_change",
    [
        "method",
        "rng",
        "outcome",
    ],
)
def test_wrong_method_rng_or_outcome_version_fails(cfg: AppConfig, expected_change: str) -> None:
    artifact, metadata = _publish(cfg)
    if expected_change == "method":
        expected_method = replace(metadata.method_contract, method_version=3)
        with pytest.raises(ArtifactMismatchError, match="method contract"):
            validate_authenticated_artifact(
                artifact,
                cfg=cfg,
                expected_location=_location(),
                expected_method_contract=expected_method,
            )
    else:
        field = "rng_scheme_version" if expected_change == "rng" else "outcome_schema_version"
        expected_versions = replace(metadata.versions, **{field: 99})
        with pytest.raises(ArtifactMismatchError, match="version identity"):
            validate_authenticated_artifact(
                artifact,
                cfg=cfg,
                expected_location=_location(),
                expected_versions=expected_versions,
            )


@pytest.mark.parametrize(
    "method",
    [
        _method(family_hash="c" * 64),
        _method(schedule_hash="d" * 64),
        _method(multiplicity="bonferroni_decisions"),
        _method(k_weights=((2, 0.25), (4, 0.75))),
    ],
    ids=("family", "schedule", "multiplicity", "k-weights"),
)
def test_wrong_typed_method_parameter_fails(cfg: AppConfig, method: MethodContract) -> None:
    artifact, _ = _publish(cfg)
    with pytest.raises(ArtifactMismatchError, match="method contract"):
        validate_authenticated_artifact(
            artifact,
            cfg=cfg,
            expected_location=_location(),
            expected_method_contract=method,
        )


def test_altered_artifact_cannot_be_blessed_by_sidecar_republication(cfg: AppConfig) -> None:
    artifact, metadata = _publish(cfg)
    completion_output = CompletionOutputIdentity(
        artifact=metadata.artifact,
        sidecar_sha256=sha256_file(sidecar_path(artifact)),
    )
    sidecar_path(artifact).unlink()
    pq.write_table(pa.table({"strategy_id": [999], "wins": [999]}), artifact)

    with pytest.raises(ArtifactMismatchError, match="does not bind current artifact"):
        finalize_missing_sidecar_atomic(
            artifact,
            cfg=cfg,
            expected_sidecar=metadata,
            completion_output=completion_output,
        )
    assert not sidecar_path(artifact).exists()


def test_missing_sidecar_finalizes_only_from_exact_completion_identity(cfg: AppConfig) -> None:
    artifact, metadata = _publish(cfg)
    completion_output = CompletionOutputIdentity(
        artifact=metadata.artifact,
        sidecar_sha256=sha256_file(sidecar_path(artifact)),
    )
    sidecar_path(artifact).unlink()

    restored = finalize_missing_sidecar_atomic(
        artifact,
        cfg=cfg,
        expected_sidecar=metadata,
        completion_output=completion_output,
    )

    assert restored == metadata
    assert sha256_file(sidecar_path(artifact)) == completion_output.sidecar_sha256


def test_missing_and_corrupt_sidecars_have_distinct_failures(cfg: AppConfig) -> None:
    artifact, _ = _publish(cfg)
    sidecar_path(artifact).unlink()
    with pytest.raises(MissingSidecarError):
        validate_authenticated_artifact(artifact, cfg=cfg, expected_location=_location())

    sidecar_path(artifact).write_text("{not json", encoding="utf-8")
    with pytest.raises(CorruptSidecarError):
        validate_authenticated_artifact(artifact, cfg=cfg, expected_location=_location())


def test_unavailable_code_identity_is_rejected_in_release_mode(tmp_path: Path) -> None:
    with pytest.raises(CodeIdentityError, match="unable to determine Git code identity"):
        resolve_code_identity(tmp_path, policy=CodeIdentityPolicy.RELEASE_CLEAN)


def test_git_code_identity_enforces_clean_release_and_fingerprints_dirty_development(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "src" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    for command in (
        ("init",),
        ("config", "user.email", "test@example.invalid"),
        ("config", "user.name", "Contract Test"),
        ("add", "src/module.py"),
        ("commit", "-m", "initial"),
    ):
        subprocess.run(["git", *command], cwd=repo, check=True, capture_output=True)

    clean = resolve_code_identity(repo, policy=CodeIdentityPolicy.RELEASE_CLEAN)
    assert clean.state == "clean"
    assert len(clean.commit) == 40

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(CodeIdentityError, match="clean Git worktree"):
        resolve_code_identity(repo, policy=CodeIdentityPolicy.RELEASE_CLEAN)
    dirty = resolve_code_identity(repo, policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY)
    repeated = resolve_code_identity(repo, policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY)
    assert dirty.state == "development_dirty"
    assert dirty.dirty_fingerprint_sha256 == repeated.dirty_fingerprint_sha256

    source.write_text("VALUE = 3\n", encoding="utf-8")
    changed = resolve_code_identity(repo, policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY)
    assert changed.dirty_fingerprint_sha256 != dirty.dirty_fingerprint_sha256


def test_stage_config_identity_is_scoped_to_explicit_public_fields(cfg: AppConfig) -> None:
    original = stage_config_identity(
        cfg,
        stage_key="metrics",
        field_paths=("screening.resolution_delta",),
    )
    cfg.analysis.n_jobs = 99
    assert (
        stage_config_identity(
            cfg,
            stage_key="metrics",
            field_paths=("screening.resolution_delta",),
        )
        == original
    )
    cfg.screening.resolution_delta = 0.04
    assert (
        stage_config_identity(
            cfg,
            stage_key="metrics",
            field_paths=("screening.resolution_delta",),
        ).sha256
        != original.sha256
    )
    with pytest.raises(ValueError, match="private configuration"):
        stage_config_identity(cfg, stage_key="metrics", field_paths=("_stage_layout",))


def test_manifest_root_is_streaming_ordered_and_binds_all_entry_identities() -> None:
    entries = [
        ManifestEntry((0, 2), "by_k/2p/0.parquet", _HASH_A, _HASH_B, "c" * 64),
        ManifestEntry((1, 2), "by_k/2p/1.parquet", _HASH_B, _HASH_A, "d" * 64),
    ]
    original = compute_manifest_root(iter(entries))
    changed = compute_manifest_root(iter([entries[0], replace(entries[1], data_sha256="e" * 64)]))
    assert original.entry_count == 2
    assert original.root_sha256 != changed.root_sha256
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_manifest_root(iter(reversed(entries)))


def test_authenticated_manifest_root_avoids_rehashing_shards_but_binds_root_files(
    cfg: AppConfig,
) -> None:
    entries = [
        ManifestEntry((0, 2), "by_k/2p/0.parquet", _HASH_A, _HASH_B, "c" * 64),
        ManifestEntry((1, 2), "by_k/2p/1.parquet", _HASH_B, _HASH_A, "d" * 64),
    ]
    manifest_location = _location("shards.manifest.jsonl", scope="diagnostics")
    _, _, manifest_stage = _stage(cfg)
    manifest_path = manifest_location.path(cfg)
    publish_immutable_manifest_atomic(
        manifest_path,
        cfg=cfg,
        location=manifest_location,
        stage_identity=manifest_stage,  # type: ignore[arg-type]
        entries=iter(entries),
    )
    manifest = capture_manifest_root(
        logical_role="simulation_shards",
        manifest_path=manifest_path,
        cfg=cfg,
        expected_location=manifest_location,
        expected_stage_identity=manifest_stage,  # type: ignore[arg-type]
    )
    derived_location = _location("manifest-derived.parquet")
    derived_path, _ = _publish(
        cfg,
        location=derived_location,
        manifest_roots=(manifest,),
        manifest_paths={"simulation_shards": (manifest_path, sidecar_path(manifest_path))},
        manifest_configs={"simulation_shards": cfg},
    )
    validate_authenticated_artifact(
        derived_path,
        cfg=cfg,
        expected_location=derived_location,
        manifest_paths={"simulation_shards": (manifest_path, sidecar_path(manifest_path))},
        manifest_configs={"simulation_shards": cfg},
    )

    sidecar_path(manifest_path).write_bytes(sidecar_path(manifest_path).read_bytes() + b" ")
    with pytest.raises(ArtifactMismatchError, match="manifest sidecar changed"):
        validate_authenticated_artifact(
            derived_path,
            cfg=cfg,
            expected_location=derived_location,
            manifest_paths={"simulation_shards": (manifest_path, sidecar_path(manifest_path))},
            manifest_configs={"simulation_shards": cfg},
        )


def test_authenticated_lifecycle_classifies_all_five_states(cfg: AppConfig) -> None:
    location = _location()
    completion_path = cfg.stage_dir("metrics") / "metrics.v3.done.json"
    _, _, stage = _stage(cfg)
    assert (
        classify_authenticated_lifecycle(
            completion_path,
            cfg=cfg,
            expected_stage_identity=stage,  # type: ignore[arg-type]
            required_locations=[location],
        )
        is CompletionState.NOT_STARTED
    )

    checkpoint = cfg.stage_dir("metrics") / "checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("partial", encoding="utf-8")
    assert (
        classify_authenticated_lifecycle(
            completion_path,
            cfg=cfg,
            expected_stage_identity=stage,  # type: ignore[arg-type]
            required_locations=[location],
            partial_paths=[checkpoint],
        )
        is CompletionState.PARTIAL_RESUMABLE
    )

    artifact, metadata = _publish(cfg)
    output = CompletionOutputIdentity(
        artifact=metadata.artifact,
        sidecar_sha256=sha256_file(sidecar_path(artifact)),
    )
    complete = AuthenticatedCompletion(
        lifecycle_contract_version=LIFECYCLE_CONTRACT_VERSION,
        stage_identity_sha256=metadata.stage_identity.sha256,
        state=CompletionState.COMPLETE_VALID.value,
        outputs=(output,),
    )
    write_authenticated_completion_atomic(completion_path, complete)
    assert (
        classify_authenticated_lifecycle(
            completion_path,
            cfg=cfg,
            expected_stage_identity=metadata.stage_identity,
            required_locations=[location],
        )
        is CompletionState.COMPLETE_VALID
    )
    _, _, stale_stage = _stage(cfg, method=_method(multiplicity="different_family_rule"))
    assert (
        classify_authenticated_lifecycle(
            completion_path,
            cfg=cfg,
            expected_stage_identity=stale_stage,  # type: ignore[arg-type]
            required_locations=[location],
        )
        is CompletionState.COMPLETE_STALE
    )

    blocked = replace(
        complete,
        state=CompletionState.BLOCKED_BY_CAP.value,
        outputs=(),
    )
    write_authenticated_completion_atomic(completion_path, blocked)
    assert (
        classify_authenticated_lifecycle(
            completion_path,
            cfg=cfg,
            expected_stage_identity=metadata.stage_identity,
            required_locations=[location],
        )
        is CompletionState.BLOCKED_BY_CAP
    )


def test_sidecar_contract_hash_is_not_a_self_asserted_free_field(cfg: AppConfig) -> None:
    artifact, metadata = _publish(cfg)
    payload = json.loads(sidecar_path(artifact).read_text(encoding="utf-8"))
    payload["sidecar_contract_sha256"] = hashlib.sha256(b"fiction").hexdigest()
    sidecar_path(artifact).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorruptSidecarError, match="digest does not match"):
        validate_authenticated_artifact(artifact, cfg=cfg, expected_location=_location())


def test_artifact_identity_hash_changes_with_arrow_nullability() -> None:
    location = _location()
    nullable = ArtifactIdentity(
        location=location,
        byte_length=10,
        content_sha256=_HASH_A,
        arrow_schema=arrow_schema_identity(
            pa.schema([pa.field("x", pa.int32(), nullable=True)]), schema_version=2
        ),
        logical_operation="hand_count",
    )
    nonnullable = replace(
        nullable,
        arrow_schema=arrow_schema_identity(
            pa.schema([pa.field("x", pa.int32(), nullable=False)]), schema_version=2
        ),
    )
    assert nullable.sha256 != nonnullable.sha256
    assert identity_sha256(nullable) != identity_sha256(nonnullable)
