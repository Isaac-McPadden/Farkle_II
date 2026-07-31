from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.config import AppConfig, ArtifactScope
from farkle.simulation.runner import simulation_is_complete, write_simulation_done
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    make_artifact_sidecar,
    sidecar_path,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.authenticated_contract import (
    ArtifactMismatchError,
    CanonicalArtifactLocation,
    CodeIdentity,
    CodeIdentityPolicy,
    validate_authenticated_artifact,
    validate_authenticated_artifact_unbound,
)
from farkle.utils.stage_completion import (
    CompletionState,
    resolve_stage_state,
    stage_done_path,
    write_stage_done,
)


@pytest.fixture
def v3_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    cfg.sim.seed = 17
    cfg.sim.seed_list = [17]
    cfg.sim.n_players_list = [2]
    cfg.artifact_contract.artifact_contract_version = 3
    cfg.artifact_contract.schema_version = 2
    cfg.artifact_contract.estimand_version = 2
    cfg.artifact_contract.conditioning_version = 2
    cfg._code_identity = CodeIdentity(
        commit="a" * 40,
        policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
        state="clean",
        dirty_fingerprint_sha256=None,
    )
    return cfg


def _publish(
    cfg: AppConfig,
    name: str,
    *,
    sources: list[Path] | None = None,
    method_contract: dict[str, object] | None = None,
) -> Path:
    path = cfg.scope_path("metrics", ArtifactScope.BY_K, name, k=2)
    metadata = make_artifact_sidecar(
        cfg,
        path,
        producer="metrics",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation=f"publish_{Path(name).stem}",
        baseline="chance_rate_by_k",
        replication_unit="attempted_game",
        conditioning="all_attempted_games",
        source_artifacts=sources or [],
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        method_contract=cast(
            Any,
            method_contract
            or {
                "kind": "operation",
                "procedure": f"publish_{Path(name).stem}",
                "parameters": {"method_version": 2},
            },
        ),
    )
    write_parquet_artifact_atomic(
        pa.table({"strategy": pa.array([1, 2], type=pa.int32()), "wins": [2, 1]}),
        path,
        sidecar=metadata,
    )
    return path


def test_changed_source_bytes_invalidate_cached_descendant(v3_cfg: AppConfig) -> None:
    source = _publish(v3_cfg, "source.parquet")
    derived = _publish(v3_cfg, "derived.parquet", sources=[source])
    done = stage_done_path(v3_cfg.stage_dir("metrics"), "metrics")
    write_stage_done(
        done,
        inputs=[source],
        outputs=[derived],
        cfg=v3_cfg,
        stage="metrics",
    )
    assert (
        resolve_stage_state(
            done,
            [source],
            [derived],
            cfg=v3_cfg,
            stage="metrics",
        )
        is CompletionState.COMPLETE_VALID
    )

    source.write_bytes(source.read_bytes() + b"mutation")

    assert (
        resolve_stage_state(
            done,
            [source],
            [derived],
            cfg=v3_cfg,
            stage="metrics",
        )
        is CompletionState.COMPLETE_STALE
    )


def test_wrong_scope_and_schema_are_rejected(v3_cfg: AppConfig) -> None:
    artifact = _publish(v3_cfg, "scoped.parquet")
    wrong_location = CanonicalArtifactLocation(
        stage_key="metrics",
        scope=ArtifactScope.ACROSS_K.value,
        relative_path="scoped.parquet",
    )
    wrong_path = wrong_location.path(v3_cfg)
    wrong_path.parent.mkdir(parents=True)
    shutil.copyfile(artifact, wrong_path)
    shutil.copyfile(sidecar_path(artifact), sidecar_path(wrong_path))
    with pytest.raises(ArtifactMismatchError, match="different canonical"):
        validate_authenticated_artifact_unbound(
            wrong_path,
            expected_location=wrong_location,
            validate_provenance=False,
        )

    pq.write_table(pa.table({"strategy": ["not-an-int"], "wins": [2]}), artifact)
    with pytest.raises(ArtifactMismatchError, match="bytes or actual"):
        validate_authenticated_artifact_unbound(
            artifact,
            validate_provenance=False,
        )


@pytest.mark.parametrize(
    "field",
    (
        "rng_scheme_version",
        "outcome_schema_version",
        "schema_version",
        "estimand_version",
        "conditioning_version",
    ),
)
def test_wrong_complete_version_identity_is_rejected(
    v3_cfg: AppConfig,
    field: str,
) -> None:
    artifact = _publish(v3_cfg, "versions.parquet")
    current = validate_authenticated_artifact_unbound(
        artifact,
        validate_provenance=False,
    )
    expected = replace(current.versions, **{field: 99})
    location = current.artifact.location
    with pytest.raises(ArtifactMismatchError, match="version identity"):
        validate_authenticated_artifact(
            artifact,
            cfg=v3_cfg,
            expected_location=location,
            expected_versions=expected,
        )


def test_v2_h2h_block_cannot_satisfy_v3_plan(
    v3_cfg: AppConfig,
    tmp_path: Path,
) -> None:
    v2_cfg = AppConfig()
    v2_cfg.artifact_contract.artifact_contract_version = 2
    v2_cfg.artifact_contract.schema_version = 1
    v2_cfg.artifact_contract.estimand_version = 1
    v2_cfg.artifact_contract.conditioning_version = 1
    v2_cfg.io.results_dir_prefix = tmp_path / "legacy"
    v2_cfg.sim.n_players_list = [2]
    legacy = _publish(v2_cfg, "h2h_block_v2.parquet")

    with pytest.raises(ArtifactContractError, match="contract-v2 source"):
        _publish(v3_cfg, "h2h_plan_v3.parquet", sources=[legacy])


def test_valid_artifact_with_wrong_source_identity_is_rejected(
    v3_cfg: AppConfig,
) -> None:
    expected_source = _publish(v3_cfg, "expected_source.parquet")
    wrong_source = _publish(v3_cfg, "wrong_source.parquet")
    derived = _publish(v3_cfg, "derived_from_expected.parquet", sources=[expected_source])
    metadata = validate_authenticated_artifact_unbound(
        derived,
        validate_provenance=False,
    )
    role = metadata.source_artifacts[0].logical_role

    with pytest.raises(ArtifactMismatchError, match="source artifact bytes or sidecar changed"):
        validate_authenticated_artifact_unbound(
            derived,
            source_paths={role: wrong_source},
        )


@pytest.mark.parametrize("field", ("family_hash", "schedule_hash"))
def test_valid_artifact_with_wrong_h2h_design_identity_is_rejected(
    v3_cfg: AppConfig,
    field: str,
) -> None:
    artifact = _publish(
        v3_cfg,
        "h2h_design_bound.parquet",
        method_contract={
            "kind": "h2h",
            "procedure": "simulate_root_order_block",
            "parameters": {
                "method_version": 2,
                "family_hash": "a" * 64,
                "schedule_hash": "c" * 64,
            },
        },
    )
    metadata = validate_authenticated_artifact_unbound(
        artifact,
        validate_provenance=False,
    )
    expected_method = replace(
        metadata.method_contract,
        **{field: "b" * 64},
    )

    with pytest.raises(ArtifactMismatchError, match="method contract does not match"):
        validate_authenticated_artifact_unbound(
            artifact,
            expected_method_contract=expected_method,
            validate_provenance=False,
        )


def test_completion_is_not_published_without_valid_required_artifact(
    v3_cfg: AppConfig,
) -> None:
    raw = v3_cfg.scope_path("metrics", ArtifactScope.BY_K, "raw.parquet", k=2)
    raw.parent.mkdir(parents=True)
    pq.write_table(pa.table({"x": [1]}), raw)
    done = stage_done_path(v3_cfg.stage_dir("metrics"), "metrics")

    with pytest.raises(ArtifactContractError, match="valid sidecar"):
        write_stage_done(
            done,
            inputs=[],
            outputs=[raw],
            cfg=v3_cfg,
            stage="metrics",
        )
    assert not done.exists()


def test_completion_marker_cannot_outlive_a_required_artifact(v3_cfg: AppConfig) -> None:
    artifact = _publish(v3_cfg, "required.parquet")
    done = stage_done_path(v3_cfg.stage_dir("metrics"), "metrics")
    write_stage_done(
        done,
        inputs=[],
        outputs=[artifact],
        cfg=v3_cfg,
        stage="metrics",
    )

    artifact.unlink()

    assert (
        resolve_stage_state(
            done,
            [],
            [artifact],
            cfg=v3_cfg,
            stage="metrics",
        )
        is CompletionState.COMPLETE_STALE
    )


def test_stage_completion_fails_when_one_required_sidecar_is_missing(
    v3_cfg: AppConfig,
) -> None:
    first = _publish(v3_cfg, "first_required.parquet")
    second = _publish(v3_cfg, "second_required.parquet")
    done = stage_done_path(v3_cfg.stage_dir("metrics"), "metrics")
    write_stage_done(
        done,
        inputs=[],
        outputs=[first, second],
        cfg=v3_cfg,
        stage="metrics",
    )

    sidecar_path(second).unlink()

    assert (
        resolve_stage_state(
            done,
            [],
            [first, second],
            cfg=v3_cfg,
            stage="metrics",
        )
        is CompletionState.COMPLETE_STALE
    )


def test_simulation_publishes_authenticated_shards_manifest_and_completion(
    v3_cfg: AppConfig,
) -> None:
    v3_cfg.sim.row_dir = Path("rows")
    cell = v3_cfg.n_dir(2)
    cell.mkdir(parents=True)
    strategy_manifest = v3_cfg.strategy_manifest_root_path()
    pq.write_table(
        pa.table({"strategy_id": pa.array([1], type=pa.int32())}),
        strategy_manifest,
    )
    workload = cell / "simulation_workload_plan.json"
    workload.write_text(
        json.dumps({"required_shuffles": 1, "shuffles_per_batch": 1}),
        encoding="utf-8",
    )
    checkpoint = cell / "2p_checkpoint.pkl"
    checkpoint.write_bytes(pickle.dumps({"complete": True}))
    row_dir = v3_cfg.simulation_row_dir(2)
    assert row_dir is not None
    row_dir.mkdir()
    shard = row_dir / "rows_000000.parquet"
    pq.write_table(pa.table({"value": [1]}), shard)
    manifest = row_dir / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"path": shard.name, "shuffle_index": 0, "rows": 1}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="refusing to promote pre-existing simulation bytes"):
        write_simulation_done(
            v3_cfg,
            2,
            num_shuffles=1,
            shuffles_per_batch=1,
            n_strategies=1,
            outputs=[checkpoint, workload, strategy_manifest, row_dir],
        )
    assert not sidecar_path(strategy_manifest).exists()

    write_simulation_done(
        v3_cfg,
        2,
        num_shuffles=1,
        shuffles_per_batch=1,
        n_strategies=1,
        outputs=[checkpoint, workload, strategy_manifest, row_dir],
        allow_unsealed_v3_outputs=True,
    )

    assert simulation_is_complete(v3_cfg, 2)
    for artifact in (checkpoint, workload, strategy_manifest, shard, manifest):
        assert sidecar_path(artifact).is_file()
