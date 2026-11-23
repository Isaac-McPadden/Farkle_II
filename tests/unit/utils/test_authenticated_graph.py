from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.helpers.artifact_sidecars import (
    clean_test_code_identity,
    make_authenticated_v3_config,
    mutate_artifact_bytes,
    publish_v3_parquet,
)

import farkle.utils.authenticated_graph as authenticated_graph
from farkle.analysis.release_audit import (
    AuthenticatedReleaseAuditTarget,
    _audit_original_parents,
    _ProducerLaunch,
    audit_authenticated_release_graphs,
)
from farkle.config import ArtifactScope, ResourcesConfig
from farkle.orchestration.execution_history import (
    audit_published_launch,
    publish_invocation,
    retain_lifecycle,
)
from farkle.orchestration.run_contexts import (
    RootPairRunContext,
    SeedRunContext,
    write_run_context_atomic,
)
from farkle.orchestration.seed_utils import write_active_config
from farkle.utils.artifact_contract import sha256_file, sidecar_path
from farkle.utils.authenticated_contract import (
    ManifestEntry,
    reuse_stage_parquet_facts,
    validate_authenticated_artifact_unbound,
)
from farkle.utils.authenticated_graph import (
    AuthenticatedGraphSnapshot,
    SnapshotGeneration,
    SnapshotUseError,
    capture_authenticated_graph_snapshot,
)
from farkle.utils.authentication_telemetry import (
    AuthenticationTelemetry,
    use_authentication_telemetry,
)
from farkle.utils.completion_files import CompletionFileKind, CompletionNamespace
from farkle.utils.manifest import iter_manifest
from farkle.utils.partitioned_stage import (
    PartitionedStageIdentity,
    PartitionedUnit,
    run_partitioned_stage,
    validate_final_manifest,
)
from farkle.utils.release_identity import publish_native_manifest_v3, write_v3_stage_completion
from farkle.utils.stage_completion import CompletionState, resolve_stage_state


@dataclass(frozen=True)
class GraphFixture:
    context: SeedRunContext
    snapshot: AuthenticatedGraphSnapshot
    generation: SnapshotGeneration
    code: Any
    source: Path
    manifest: Path
    derived: Path
    completion: Path
    telemetry: AuthenticationTelemetry


def _json_mutate(path: Path, action: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    action(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_graph(
    tmp_path: Path,
    *,
    scope: str = "root",
    roots: tuple[int, ...] = (11,),
    run_id: str | None = None,
    manifest_path: Path | None = None,
) -> GraphFixture:
    cfg = make_authenticated_v3_config(
        tmp_path, name="task4b", root_seed=roots[0], player_counts=(2,)
    )
    code = clean_test_code_identity()
    context = SeedRunContext.from_config(cfg)
    if run_id is None:
        write_run_context_atomic(context, code_identity=code)
        write_active_config(cfg)
    else:
        publish_invocation(
            context,
            run_id=run_id,
            code_identity=code,
            completion_paths=[("metrics", cfg.stage_dir("metrics") / "metrics.done.json")],
            manifest_path=manifest_path or tmp_path / "manifest.jsonl",
        )

    source = cfg.scope_path("metrics", ArtifactScope.BY_K, "source.parquet", k=2)
    publish_v3_parquet(
        cfg,
        source,
        pa.table({"source_value": [1, 2]}),
        stage_key="metrics",
        producer="task4b_test",
        operation="task4b_source",
    )
    source_metadata = validate_authenticated_artifact_unbound(source, validate_provenance=False)
    assert source_metadata.artifact.arrow_schema is not None
    manifest = cfg.scope_path("metrics", ArtifactScope.BY_K, "fixture_manifest.jsonl", k=2)
    publish_native_manifest_v3(
        manifest,
        cfg=cfg,
        stage_key="metrics",
        entries=[
            ManifestEntry(
                coordinate=(0,),
                canonical_relative_path=source.name,
                data_sha256=source_metadata.artifact.content_sha256,
                sidecar_sha256=sha256_file(sidecar_path(source)),
                schema_fingerprint_sha256=(
                    source_metadata.artifact.arrow_schema.fingerprint_sha256
                ),
            )
        ],
        source_paths=[source],
        operation="task4b_manifest",
        player_counts=[2],
        native_records=[{"coordinate": [0], "path": source.name}],
    )
    derived = cfg.scope_path("metrics", ArtifactScope.BY_K, "derived.parquet", k=2)
    publish_v3_parquet(
        cfg,
        derived,
        pa.table({"derived_value": [3]}),
        stage_key="metrics",
        producer="task4b_test",
        operation="task4b_derived",
        sources=(source,),
        manifests=(manifest,),
    )
    completion = cfg.stage_dir("metrics") / "metrics.done.json"
    outputs = [derived]
    write_v3_stage_completion(
        completion,
        cfg=cfg,
        stage_key="metrics",
        inputs=[source, manifest],
        outputs=outputs,
        status="success",
    )
    state = resolve_stage_state(
        completion,
        inputs=[],
        outputs=outputs,
        cfg=cfg,
        stage="metrics",
    )
    assert state is CompletionState.COMPLETE_VALID
    generation = SnapshotGeneration()
    telemetry = AuthenticationTelemetry()
    with use_authentication_telemetry(telemetry):
        snapshot = capture_authenticated_graph_snapshot(
            cfg=cfg,
            scope=scope,
            roots=roots,
            graph_root=cfg.results_root if run_id and scope == "root" else cfg.analysis_dir,
            analysis_root=cfg.analysis_dir,
            run_context_path=context.run_context_path,
            active_config_path=context.active_config_path,
            stage_states={"metrics": CompletionState.COMPLETE_VALID.value},
            completion_paths=[("metrics", completion)],
            generation=generation,
            code_identity=code,
        )
    return GraphFixture(
        context=context,
        snapshot=snapshot,
        generation=generation,
        code=code,
        source=source,
        manifest=manifest,
        derived=derived,
        completion=completion,
        telemetry=telemetry,
    )


def _partition_identity(stage_name: str) -> PartitionedStageIdentity:
    return PartitionedStageIdentity(
        stage_name=stage_name,
        root_seed=11,
        input_identities=(("source", "a" * 64),),
        statistical_config_sha256="b" * 64,
        code_identity_sha256="c" * 64,
        schema_version=1,
        method_version=1,
    )


def _partition_resources() -> ResourcesConfig:
    return ResourcesConfig(
        scheduler_memory_budget_mb=256,
        parent_process_memory_mb=64,
        logical_cpu_budget=1,
        native_threads_per_worker=1,
        estimated_worker_memory_mb={"partitioned_stage": 32},
        stage_batch_bytes={"partitioned_stage": 4096},
    )


def _range_units() -> tuple[PartitionedUnit, ...]:
    return tuple(
        PartitionedUnit(
            (start, start + 50),
            f"replicates_{start:08d}_{start + 50:08d}.npy",
        )
        for start in range(0, 500, 50)
    )


def _write_range_unit(unit: PartitionedUnit, path: Path) -> None:
    path.write_bytes(json.dumps(list(unit.key)).encode("ascii"))


def test_pair_final_audit_accepts_production_shaped_partition_unit_trees(
    tmp_path: Path,
) -> None:
    fixture = _build_graph(tmp_path, scope="pair", roots=(11, 12))
    partition_results = []
    for directory, stage_name in (
        ("_bootstrap_top_n_ranges", "root_stability_top_n"),
        ("_joint_discrepancy_ranges", "root_stability_joint_discrepancy"),
    ):
        root = fixture.completion.parent / "cross_seed" / directory
        partition_results.append(
            (
                root,
                _partition_identity(stage_name),
                run_partitioned_stage(
                    root=root,
                    identity=_partition_identity(stage_name),
                    unit_source=lambda: iter(_range_units()),
                    writer=_write_range_unit,
                    resources=_partition_resources(),
                    requested_workers=1,
                ),
            )
        )

    namespace = CompletionNamespace.build(
        graph_root=fixture.snapshot.graph_root,
        analysis_root=fixture.snapshot.analysis_root,
        canonical_paths=(fixture.completion,),
    )
    stamps = sorted(fixture.snapshot.graph_root.rglob("*.unit.done.json"))
    assert len(stamps) == 20
    assert {namespace.classify(path) for path in stamps} == {CompletionFileKind.PARTITION_UNIT}
    assert _audit(fixture)["status"] == "passed"

    for root, identity, first in partition_results:
        resumed = run_partitioned_stage(
            root=root,
            identity=identity,
            unit_source=lambda: iter(_range_units()),
            writer=_write_range_unit,
            resources=_partition_resources(),
            requested_workers=1,
        )
        assert resumed.reused_units == 10
        assert resumed.completed_units == 0
        assert validate_final_manifest(
            first.manifest_path,
            root=root,
            identity=identity,
            unit_source=lambda: iter(_range_units()),
        ) == (first.manifest_sha256, 10)


def test_final_audit_rejects_missing_relocated_and_duplicated_completion(
    tmp_path: Path,
) -> None:
    for mutation in ("missing", "relocated", "duplicated"):
        fixture = _build_graph(tmp_path / mutation)
        original = fixture.completion
        relocated = original.parent.parent / "relocated" / original.name
        relocated.parent.mkdir(parents=True)
        if mutation == "missing":
            original.unlink()
        elif mutation == "relocated":
            original.replace(relocated)
        else:
            shutil.copyfile(original, relocated)

        result = _audit(fixture)
        assert result["status"] == "failed"
        assert any("completion inventory" in failure for failure in result["failures"])


def _audit(
    fixture: GraphFixture,
    *,
    current_code: Any | None = None,
    invocation: tuple[str, Path] | None = None,
) -> dict[str, Any]:
    with use_authentication_telemetry(fixture.telemetry):
        return audit_authenticated_release_graphs(
            [
                AuthenticatedReleaseAuditTarget(
                    cfg=fixture.context.config,
                    snapshot=fixture.snapshot,
                    generation=fixture.generation,
                )
            ],
            expected_code_identity=fixture.code,
            current_code_identity=current_code or fixture.code,
            invocation=invocation,
        )


def test_byte_deep_audit_locates_and_checks_retained_lifecycle(tmp_path: Path) -> None:
    run_id = "producer"
    fixture = _build_graph(tmp_path, run_id=run_id)
    paths = [("metrics", fixture.completion)]
    retain_lifecycle(fixture.snapshot, run_id, paths)
    invocation = (run_id, tmp_path / "manifest.jsonl")
    result = _audit(fixture, invocation=invocation)
    assert result["status"] == "passed", result["failures"]
    retained = result["run_contexts"]["root_11"]
    assert (
        Path(retained["retained_run_context"]).read_bytes()
        == fixture.context.run_context_path.read_bytes()
    )
    assert retained["lifecycle_sha256"] == fixture.snapshot.lifecycle_sha256
    inventory = Path(retained["lifecycle_inventory"])
    assert json.loads(inventory.read_bytes())["completion_stamps"] == [
        ["metrics", str(fixture.completion.resolve()), sha256_file(fixture.completion)]
    ]
    stamp = inventory.with_name("0.stamp")
    assert stamp.read_bytes() == fixture.completion.read_bytes()
    stamp.write_bytes(stamp.read_bytes() + b" ")
    rejected = _audit(fixture, invocation=invocation)
    assert rejected["status"] == "failed"
    assert any("retained lifecycle stamp differs" in msg for msg in rejected["failures"])
    shutil.copyfile(fixture.completion, stamp)
    receipt = Path(retained["invocation_record"])
    receipt.write_bytes(receipt.read_bytes() + b" ")
    rejected = _audit(fixture, invocation=invocation)
    assert any("differs from its manifest event" in msg for msg in rejected["failures"])


@pytest.mark.parametrize("swap", [False, True])
def test_final_pair_audit_binds_parents_in_configured_order(tmp_path: Path, swap: bool) -> None:
    run_id = "resume"
    seeds = (22, 11)
    targets = []
    roots = []
    parents = []
    code = clean_test_code_identity()
    for seed in seeds:
        f = _build_graph(
            tmp_path / str(seed),
            roots=(seed,),
            run_id=run_id,
            manifest_path=tmp_path / "manifest.jsonl",
        )
        retain_lifecycle(f.snapshot, run_id, [("metrics", f.completion)])
        roots.append(f.context)
        parents.append(f.snapshot.lifecycle_sha256)
        targets.append(AuthenticatedReleaseAuditTarget(f.context.config, f.snapshot, f.generation))

    pair = RootPairRunContext.from_root_contexts((roots[0], roots[1]), pair_root=tmp_path / "pair")
    cfg = pair.config
    publish_invocation(
        pair,
        run_id=run_id,
        manifest_path=tmp_path / "manifest.jsonl",
        code_identity=code,
        completion_paths=[],
        parent_lifecycle_roots=tuple(reversed(parents) if swap else parents),
    )
    stage = "root_stability"
    output = cfg.root_discrepancies_path()
    publish_v3_parquet(
        cfg, output, pa.table({"value": [1]}), stage_key=stage, producer=stage, operation="test"
    )
    stamp = cfg.stage_dir(stage) / "root_stability.done.json"
    write_v3_stage_completion(
        stamp, cfg=cfg, stage_key=stage, inputs=[], outputs=[output], status="success"
    )
    state = resolve_stage_state(stamp, inputs=[], outputs=[output], cfg=cfg, stage=stage)
    generation = SnapshotGeneration()
    snapshot = capture_authenticated_graph_snapshot(
        cfg=cfg,
        scope="pair",
        roots=tuple(sorted(seeds)),
        graph_root=pair.analysis_root,
        analysis_root=pair.analysis_root,
        run_context_path=pair.run_context_path,
        active_config_path=pair.active_config_path,
        stage_states={stage: state.value},
        completion_paths=[(stage, stamp)],
        generation=generation,
        code_identity=code,
    )
    retain_lifecycle(snapshot, run_id, [(stage, stamp)])
    targets.append(AuthenticatedReleaseAuditTarget(cfg, snapshot, generation))
    original_launches = []
    for target, event in zip(targets, iter_manifest(tmp_path / "manifest.jsonl"), strict=True):
        context, receipt = audit_published_launch(target.snapshot.run_context_path.parent, event)
        original_launches.append(_ProducerLaunch(context, receipt, event, code))
    originals = {
        (seed, run_id): launch for seed, launch in zip(seeds, original_launches[:2], strict=True)
    }
    if swap:
        with pytest.raises(ValueError, match="historical parent lifecycle"):
            _audit_original_parents(cfg, original_launches[2], originals)
    else:
        _audit_original_parents(cfg, original_launches[2], originals)
    result = audit_authenticated_release_graphs(
        targets,
        expected_code_identity=code,
        current_code_identity=code,
        invocation=(run_id, tmp_path / "manifest.jsonl"),
    )
    if swap:
        assert result["status"] == "failed"
        assert any("parent lifecycles differ" in msg for msg in result["failures"])
    else:
        assert result["status"] == "passed", result["failures"]
        assert [
            result["run_contexts"][f"root_{seed}"]["lifecycle_sha256"] for seed in seeds
        ] == parents


def _mutate_case(fixture: GraphFixture, mutation: str) -> None:
    artifact_sidecar = sidecar_path(fixture.derived)
    manifest_sidecar = sidecar_path(fixture.manifest)
    source_sidecar = sidecar_path(fixture.source)
    if mutation == "artifact_bytes":
        mutate_artifact_bytes(fixture.derived)
    elif mutation == "parquet_schema":
        pq.write_table(pa.table({"changed_schema": [1]}), fixture.derived)
    elif mutation == "artifact_sidecar_bytes":
        artifact_sidecar.write_bytes(artifact_sidecar.read_bytes() + b" ")
    elif mutation == "artifact_sidecar_field":
        _json_mutate(
            artifact_sidecar,
            lambda payload: payload["artifact"].__setitem__("logical_operation", "changed"),
        )
    elif mutation == "sidecar_contract_hash":
        _json_mutate(
            artifact_sidecar,
            lambda payload: payload.__setitem__("sidecar_contract_sha256", "0" * 64),
        )
    elif mutation == "manifest_bytes":
        fixture.manifest.write_bytes(fixture.manifest.read_bytes() + b"{}\n")
    elif mutation == "manifest_sidecar":
        manifest_sidecar.write_bytes(manifest_sidecar.read_bytes() + b" ")
    elif mutation == "manifest_summary":
        _json_mutate(
            manifest_sidecar,
            lambda payload: payload["summary"].__setitem__("entry_count", 2),
        )
    elif mutation == "completion_bytes":
        fixture.completion.write_bytes(fixture.completion.read_bytes() + b" ")
    elif mutation == "completion_order":
        _json_mutate(fixture.completion, lambda payload: payload["outputs"].reverse())
    elif mutation == "completion_omission":
        _json_mutate(fixture.completion, lambda payload: payload["outputs"].pop())
    elif mutation == "completion_duplication":
        _json_mutate(
            fixture.completion,
            lambda payload: payload["outputs"].append(payload["outputs"][0]),
        )
    elif mutation == "completion_inventory_addition":
        unexpected = fixture.completion.parent.parent / "unexpected" / "unexpected.done.json"
        unexpected.parent.mkdir(parents=True)
        shutil.copyfile(fixture.completion, unexpected)
    elif mutation == "run_context_bytes":
        fixture.context.run_context_path.write_bytes(
            fixture.context.run_context_path.read_bytes() + b" "
        )
    elif mutation == "active_config":
        fixture.context.active_config_path.write_text(
            fixture.context.active_config_path.read_text(encoding="utf-8") + "# changed\n",
            encoding="utf-8",
        )
    elif mutation == "statistical_config_sha":
        current_delta = fixture.context.config.screening.delta_across_k
        assert current_delta is not None
        fixture.context.config.screening.delta_across_k = current_delta + 0.001
    elif mutation == "run_lineage":
        fixture.context.config._run_lineage_sha256 = "2" * 64
    elif mutation == "upstream_source_artifact":
        mutate_artifact_bytes(fixture.source)
    elif mutation == "upstream_source_sidecar":
        source_sidecar.write_bytes(source_sidecar.read_bytes() + b" ")
    elif mutation == "manifest_root_binding":
        _json_mutate(
            artifact_sidecar,
            lambda payload: payload["manifest_roots"][0].__setitem__("manifest_sha256", "0" * 64),
        )
    elif mutation == "canonical_location":
        _json_mutate(
            artifact_sidecar,
            lambda payload: payload["artifact"]["location"].__setitem__(
                "relative_path", "changed.parquet"
            ),
        )
    elif mutation == "scope":
        _json_mutate(
            artifact_sidecar,
            lambda payload: payload["artifact"]["location"].__setitem__(
                "scope", ArtifactScope.DIAGNOSTICS.value
            ),
        )
    elif mutation == "player_count":
        _json_mutate(
            artifact_sidecar,
            lambda payload: payload["artifact"]["location"].__setitem__("player_count", 4),
        )
    elif mutation == "orphan_sidecar":
        shutil.copyfile(artifact_sidecar, artifact_sidecar.parent / "orphan.parquet.sidecar.json")
    elif mutation == "removed_sidecar":
        artifact_sidecar.unlink()
    elif mutation == "duplicate_canonical_location":
        duplicate = (
            artifact_sidecar.parent.parent.parent / "copy" / "by_k" / "2p" / fixture.derived.name
        )
        duplicate.parent.mkdir(parents=True)
        shutil.copyfile(fixture.derived, duplicate)
        shutil.copyfile(artifact_sidecar, sidecar_path(duplicate))
    elif mutation == "mixed_v2_v3":
        _json_mutate(
            artifact_sidecar,
            lambda payload: payload.__setitem__("artifact_contract_version", 2),
        )
    elif mutation == "mixed_global_identity":
        _json_mutate(
            artifact_sidecar,
            lambda payload: payload["versions"].__setitem__("rng_scheme_version", 1),
        )
    else:
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        "artifact_bytes",
        "parquet_schema",
        "artifact_sidecar_bytes",
        "artifact_sidecar_field",
        "sidecar_contract_hash",
        "manifest_bytes",
        "manifest_sidecar",
        "manifest_summary",
        "completion_bytes",
        "completion_order",
        "completion_omission",
        "completion_duplication",
        "completion_inventory_addition",
        "run_context_bytes",
        "active_config",
        "statistical_config_sha",
        "run_lineage",
        "upstream_source_artifact",
        "upstream_source_sidecar",
        "manifest_root_binding",
        "canonical_location",
        "scope",
        "player_count",
        "orphan_sidecar",
        "removed_sidecar",
        "duplicate_canonical_location",
        "mixed_v2_v3",
        "mixed_global_identity",
    ],
)
@reuse_stage_parquet_facts()
def test_final_byte_deep_audit_rejects_post_snapshot_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _build_graph(tmp_path)
    _mutate_case(fixture, mutation)

    result = _audit(fixture)

    assert result["status"] == "failed"
    assert result["failures"]


@pytest.mark.parametrize("change", ["missing", "changed_bytes", "changed_sidecar"])
def test_completion_rechecks_manifest_dependencies(tmp_path: Path, change: str) -> None:
    fixture = _build_graph(tmp_path)
    if change == "missing":
        fixture.manifest.unlink()
    else:
        path = fixture.manifest if change == "changed_bytes" else sidecar_path(fixture.manifest)
        path.write_bytes(path.read_bytes() + b" ")
    assert (
        resolve_stage_state(
            fixture.completion,
            inputs=[],
            outputs=[],
            cfg=fixture.context.config,
            stage="metrics",
        )
        is CompletionState.COMPLETE_STALE
    )


def test_completion_hashes_shared_manifest_dependency_once(tmp_path: Path) -> None:
    fixture = _build_graph(tmp_path)
    cfg = fixture.context.config
    other = fixture.derived.with_name("other.parquet")
    publish_v3_parquet(
        cfg,
        other,
        pa.table({"other_value": [4]}),
        stage_key="metrics",
        producer="task4b_test",
        operation="task4b_derived",
        sources=(fixture.source,),
        manifests=(fixture.manifest,),
    )
    outputs = [fixture.derived, other]
    write_v3_stage_completion(
        fixture.completion,
        cfg=cfg,
        stage_key="metrics",
        inputs=[],
        outputs=outputs,
        status="success",
    )

    telemetry = AuthenticationTelemetry()
    with use_authentication_telemetry(telemetry):
        state = resolve_stage_state(fixture.completion, [], outputs, cfg=cfg, stage="metrics")
    assert state is CompletionState.COMPLETE_VALID
    assert telemetry.manifest_opens == 1
    assert telemetry.manifest_bytes_hashed == fixture.manifest.stat().st_size


@pytest.mark.parametrize("duplicate", ["source", "manifest"])
def test_pair_completion_rejects_ambiguous_root_dependencies(
    tmp_path: Path, duplicate: str
) -> None:
    fixture = _build_graph(tmp_path)
    other_cfg = make_authenticated_v3_config(tmp_path, name="other", root_seed=22)
    pair = RootPairRunContext.from_root_contexts(
        (fixture.context, SeedRunContext.from_config(other_cfg)), pair_root=tmp_path
    )
    cfg = pair.config
    stage = "root_stability"
    output = cfg.root_discrepancies_path()
    publish_v3_parquet(
        cfg,
        output,
        pa.table({"value": [1]}),
        stage_key=stage,
        producer=stage,
        operation="test",
        sources=(fixture.source,),
        manifests=(fixture.manifest,),
    )
    stamp = cfg.stage_dir(stage) / "root_stability.done.json"
    write_v3_stage_completion(
        stamp, cfg=cfg, stage_key=stage, inputs=[], outputs=[output], status="success"
    )
    assert resolve_stage_state(stamp, [], [output], cfg=cfg, stage=stage) is (
        CompletionState.COMPLETE_VALID
    )
    original = fixture.source if duplicate == "source" else fixture.manifest
    new_path = other_cfg.results_root / original.relative_to(fixture.context.results_root)
    new_path.parent.mkdir(parents=True)
    shutil.copyfile(original, new_path)
    shutil.copyfile(sidecar_path(original), sidecar_path(new_path))
    assert resolve_stage_state(stamp, [], [output], cfg=cfg, stage=stage) is (
        CompletionState.COMPLETE_STALE
    )


def test_final_audit_rejects_changed_repository_code_identity(tmp_path: Path) -> None:
    fixture = _build_graph(tmp_path)
    changed = clean_test_code_identity(commit="b" * 40)

    result = _audit(fixture, current_code=changed)

    assert result["status"] == "failed"
    assert any("code identity changed" in failure for failure in result["failures"])


def test_snapshot_is_immutable_nonserializable_and_process_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_graph(tmp_path)
    with pytest.raises(TypeError, match="process-local"):
        fixture.snapshot.__reduce__()
    fixture.generation.validate(
        fixture.snapshot,
        expected_scope="root",
        expected_roots=(11,),
        expected_run_context_path=fixture.context.run_context_path,
    )

    monkeypatch.setattr(authenticated_graph.os, "getpid", lambda: fixture.snapshot.process_id + 1)
    with pytest.raises(SnapshotUseError, match="stale, foreign"):
        fixture.generation.validate(
            fixture.snapshot,
            expected_scope="root",
            expected_roots=(11,),
            expected_run_context_path=fixture.context.run_context_path,
        )


def test_snapshot_generation_and_scope_changes_fail_closed(tmp_path: Path) -> None:
    fixture = _build_graph(tmp_path)
    with pytest.raises(SnapshotUseError):
        fixture.generation.validate(
            fixture.snapshot,
            expected_scope="root",
            expected_roots=(12,),
            expected_run_context_path=fixture.context.run_context_path,
        )
    fixture.generation.invalidate("force rerun", telemetry=fixture.telemetry)
    with pytest.raises(SnapshotUseError):
        fixture.generation.validate(
            fixture.snapshot,
            expected_scope="root",
            expected_roots=(11,),
            expected_run_context_path=fixture.context.run_context_path,
        )
    assert fixture.telemetry.snapshot_invalidations == 1


@pytest.mark.parametrize(
    "boundary",
    [
        "run_context_authenticated",
        "active_config_authenticated",
        "completion:metrics",
        "inventory_complete",
        "snapshot_ready",
    ],
)
def test_interrupted_snapshot_construction_returns_no_snapshot(
    tmp_path: Path,
    boundary: str,
) -> None:
    fixture = _build_graph(tmp_path)

    def interrupt(name: str) -> None:
        if name == boundary:
            raise InterruptedError(name)

    with pytest.raises(InterruptedError, match=boundary):
        capture_authenticated_graph_snapshot(
            cfg=fixture.context.config,
            scope="root",
            roots=(11,),
            graph_root=fixture.context.config.analysis_dir,
            analysis_root=fixture.context.config.analysis_dir,
            run_context_path=fixture.context.run_context_path,
            active_config_path=fixture.context.active_config_path,
            stage_states={"metrics": CompletionState.COMPLETE_VALID.value},
            completion_paths=[("metrics", fixture.completion)],
            generation=SnapshotGeneration(),
            code_identity=fixture.code,
            boundary_hook=interrupt,
        )


@pytest.mark.parametrize(
    "state",
    [
        CompletionState.NOT_STARTED.value,
        CompletionState.PARTIAL_RESUMABLE.value,
        CompletionState.COMPLETE_STALE.value,
        CompletionState.BLOCKED_BY_CAP.value,
    ],
)
def test_snapshot_refuses_noncomplete_context(tmp_path: Path, state: str) -> None:
    fixture = _build_graph(tmp_path)
    with pytest.raises(SnapshotUseError, match="incomplete or stale"):
        capture_authenticated_graph_snapshot(
            cfg=fixture.context.config,
            scope="root",
            roots=(11,),
            graph_root=fixture.context.config.analysis_dir,
            analysis_root=fixture.context.config.analysis_dir,
            run_context_path=fixture.context.run_context_path,
            active_config_path=fixture.context.active_config_path,
            stage_states={"metrics": state},
            completion_paths=[("metrics", fixture.completion)],
            generation=SnapshotGeneration(),
            code_identity=fixture.code,
        )


def test_data_without_sidecar_and_sidecar_without_completion_cannot_snapshot(
    tmp_path: Path,
) -> None:
    fixture = _build_graph(tmp_path)
    sidecar_path(fixture.derived).unlink()
    with pytest.raises(SnapshotUseError, match="absent or stale"):
        capture_authenticated_graph_snapshot(
            cfg=fixture.context.config,
            scope="root",
            roots=(11,),
            graph_root=fixture.context.config.analysis_dir,
            analysis_root=fixture.context.config.analysis_dir,
            run_context_path=fixture.context.run_context_path,
            active_config_path=fixture.context.active_config_path,
            stage_states={"metrics": CompletionState.COMPLETE_VALID.value},
            completion_paths=[("metrics", fixture.completion)],
            generation=SnapshotGeneration(),
            code_identity=fixture.code,
        )
    with pytest.raises(SnapshotUseError, match="completion inventory"):
        capture_authenticated_graph_snapshot(
            cfg=fixture.context.config,
            scope="root",
            roots=(11,),
            graph_root=fixture.context.config.analysis_dir,
            analysis_root=fixture.context.config.analysis_dir,
            run_context_path=fixture.context.run_context_path,
            active_config_path=fixture.context.active_config_path,
            stage_states={"metrics": CompletionState.COMPLETE_VALID.value},
            completion_paths=[],
            generation=SnapshotGeneration(),
            code_identity=fixture.code,
        )


@reuse_stage_parquet_facts()
def test_snapshot_reuse_avoids_byte_hashing_before_final_audit(tmp_path: Path) -> None:
    fixture = _build_graph(tmp_path)
    before = fixture.telemetry.as_metadata()

    fixture.generation.validate(
        fixture.snapshot,
        expected_scope="root",
        expected_roots=(11,),
        expected_run_context_path=fixture.context.run_context_path,
        telemetry=fixture.telemetry,
    )

    reused = fixture.telemetry.as_metadata()
    assert reused["snapshot_hits"] == before["snapshot_hits"] + 1
    assert reused["sha256_calls"] == before["sha256_calls"]
    result = _audit(fixture)
    assert result["status"] == "passed"
    final = fixture.telemetry.as_metadata()
    assert final["graph_audit_invocations"] == 1
    assert final["graph_root_traversals"] == 1
    assert final["sha256_calls"] > reused["sha256_calls"]
    assert final["artifact_bytes_hashed"] - reused["artifact_bytes_hashed"] >= (
        fixture.source.stat().st_size + fixture.derived.stat().st_size
    )
    assert final["schema_validations"] - reused["schema_validations"] >= 2
