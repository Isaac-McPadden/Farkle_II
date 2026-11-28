"""Real checkout revisions, independent interpreters, and temporary results."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _run_fixture(folder: Path, invocation: str, producer_state: str) -> None:
    resume = invocation != "first"
    from farkle.analysis import ingest, release_audit
    from farkle.analysis.stage_registry import StageLayout
    from farkle.config import AppConfig, assign_config_sha
    from farkle.orchestration.execution_history import (
        audit_parent_lifecycle,
        audit_published_launch,
        publish_invocation,
        retain_lifecycle,
    )
    from farkle.orchestration.run_contexts import SeedRunContext
    from farkle.orchestration.seed_utils import seed_has_completion_markers
    from farkle.simulation import runner
    from farkle.utils.artifact_contract import sha256_file, sidecar_path
    from farkle.utils.authenticated_contract import (
        ArtifactMismatchError,
        CodeIdentityPolicy,
        load_authenticated_sidecar,
        make_stage_identity,
        resolve_code_identity,
    )
    from farkle.utils.authenticated_graph import (
        SnapshotGeneration,
        capture_authenticated_graph_snapshot,
    )
    from farkle.utils.scientific_invocation import ROOT, configured_producers, scientific_invocation
    from farkle.utils.scientific_reuse import compatible_producer

    cfg = AppConfig()
    cfg.io.results_dir_prefix = folder / "results"
    cfg.io.temp_dir = folder / ("scratch-b" if resume else "scratch-a")
    cfg.io.temp_dir.mkdir(exist_ok=True)
    cfg.sim.seed = 42
    cfg.sim.seed_list = [42]
    cfg.sim.n_players_list = [2]
    cfg.sim.n_jobs = 2 if resume else 1
    cfg.ingest.n_jobs = 1
    cfg.sim.row_dir = Path("rows")
    cfg.screening.resolution_delta = 0.9
    cfg.screening.projected_games_per_second = 25.0 if resume else 10.0
    cfg.batching.target_batches = 2
    cfg.batching.min_shuffles_per_batch = 1
    cfg.sim.score_thresholds = [300, 500]
    cfg.sim.dice_thresholds = [2]
    cfg.sim.smart_five_opts = cfg.sim.smart_one_opts = [False]
    cfg.sim.consider_score_opts = cfg.sim.consider_dice_opts = [True]
    cfg.sim.auto_hot_dice_opts = cfg.sim.run_up_score_opts = [False]
    cfg.screening.practical_delta_by_k = {2: 0.03}
    cfg.set_stage_layout(
        StageLayout(
            [p for p in cfg.stage_layout.placements if p.definition.key in {"ingest", "curate"}]
        )
    )
    cfg._code_identity = resolve_code_identity(ROOT, policy=CodeIdentityPolicy.RELEASE_CLEAN)
    assign_config_sha(cfg)
    roster, _, _ = runner._resolve_strategies(cfg, None)
    with scientific_invocation(configured_producers(cfg)):
        assert cfg._code_identity is not None
        context = SeedRunContext.from_config(cfg)
        invocation_manifest = folder / "pipeline.jsonl"
        stamps = [
            ("simulation_2p", runner.simulation_done_path(cfg, 2)),
            ("ingest", cfg.stage_dir("ingest") / "ingest.done.json"),
        ]
        publish_invocation(
            context,
            run_id=invocation,
            manifest_path=invocation_manifest,
            completion_paths=stamps,
            code_identity=cfg._code_identity,
        )
        artifact = cfg.strategy_manifest_root_path()
        done = runner.simulation_done_path(cfg, 2)
        if not resume:
            assert runner.run_single_n(cfg, 2, roster) > 0
            with patch.object(
                ingest, "write_stage_done", side_effect=InterruptedError("after units")
            ):
                try:
                    ingest.run(cfg)
                except InterruptedError:
                    return
            raise AssertionError("ingest did not reach its publication boundary")
        original = load_authenticated_sidecar(artifact)
        assert (
            original.stage_identity.immutable_design_identities["run_lineage_sha256"]
            != cfg._run_lineage_sha256
        )
        assert (
            cfg._code_identity is not None
            and original.stage_identity.code.commit != cfg._code_identity.commit
        )
        evidence = [
            done,
            artifact,
            sidecar_path(artifact),
            cfg.ingested_rows_raw(2),
            sidecar_path(cfg.ingested_rows_raw(2)),
        ]
        before = [(sha256_file(p), p.stat().st_mtime_ns) for p in evidence]
        code = cfg._code_identity
        assert code is not None

        def capture_target():
            generation = SnapshotGeneration()
            snapshot = capture_authenticated_graph_snapshot(
                cfg=cfg,
                scope="root",
                roots=(42,),
                graph_root=cfg.results_root,
                analysis_root=cfg.analysis_dir,
                run_context_path=context.run_context_path,
                active_config_path=context.active_config_path,
                stage_states={name: "complete_valid" for name, _ in stamps},
                completion_paths=stamps,
                code_identity=code,
                generation=generation,
            )
            retain_lifecycle(snapshot, invocation, stamps)
            return release_audit.AuthenticatedReleaseAuditTarget(cfg, snapshot, generation)

        def audit():
            return release_audit.audit_authenticated_release_graphs(
                [target],
                expected_code_identity=cfg._code_identity,
                current_code_identity=resolve_code_identity(
                    ROOT, policy=CodeIdentityPolicy.RELEASE_CLEAN
                ),
                invocation=(invocation, invocation_manifest),
            )

        if invocation == "changed":
            assert not compatible_producer(original.stage_identity, cfg._code_identity)
            assert not runner.simulation_is_complete(cfg, 2)
            row_dir = cfg.simulation_row_dir(2)
            assert row_dir is not None
            shards = list(row_dir.glob("rows_*.parquet"))
            assert shards
            raw_files = [file for shard in shards for file in (shard, sidecar_path(shard))]
            raw_before = [(sha256_file(p), p.stat().st_mtime_ns) for p in raw_files]
            with pytest.raises(ArtifactMismatchError, match="belongs to different code"):
                runner.run_single_n(cfg, 2, roster)
            assert raw_before == [(sha256_file(p), p.stat().st_mtime_ns) for p in raw_files]
            assert before == [(sha256_file(p), p.stat().st_mtime_ns) for p in evidence]
            # Even a snapshot supplied with successful stage states cannot qualify this graph.
            target = capture_target()
            assert audit()["revision_policy"]["status"] == "ineligible"
            return
        if invocation == "replay":
            assert seed_has_completion_markers(cfg)
            target = capture_target()
            replay = audit()["revision_policy"]
            assert replay["status"] == "qualified_mixed", replay
            assert replay["qualifying_commit"] == code.commit
            assert set(replay["producer_commits"]) == {
                record["code_identity"]["commit"] for record in replay["invocation_records"][:-1]
            }
            assert code.commit not in replay["producer_commits"]
            assert before == [(sha256_file(p), p.stat().st_mtime_ns) for p in evidence]
            return
        workload = cfg.n_dir(2) / "simulation_workload_plan.json"
        retained = [sha256_file(p) for p in (workload, sidecar_path(workload))]
        runner._write_workload_plan_simulation_output(
            cfg,
            workload,
            runner._plan_workload_from_config(cfg, n_strategies=len(roster), n_players=2),
            n_players=2,
            strategy_manifest=artifact,
        )
        assert retained == [sha256_file(p) for p in (workload, sidecar_path(workload))]
        assert compatible_producer(original.stage_identity, cfg._code_identity)
        assert seed_has_completion_markers(cfg)
        with pytest.raises(ValueError, match="roster differs"):
            runner.simulation_roster_manifest(cfg, list(reversed(roster)))
        stage = original.stage_identity
        for extra in ({}, {"reuse_v1/unknown/simulation/42": "0" * 64}):
            legacy = make_stage_identity(
                stage_key=stage.stage_key,
                stage_cache_key_version=stage.stage_cache_key_version,
                stage_config=stage.stage_config,
                versions=stage.versions,
                code=stage.code,
                method_contract=original.method_contract,
                upstream_identity_sha256=stage.upstream_identity_sha256,
                immutable_design_identities={
                    **{
                        k: v
                        for k, v in stage.immutable_design_identities.items()
                        if not k.startswith("reuse_v1/")
                    },
                    **extra,
                },
            )
            assert not compatible_producer(legacy, cfg._code_identity)
        with (
            patch.object(
                runner.tournament_mod,
                "run_tournament",
                side_effect=AssertionError("replayed games"),
            ),
            patch.object(
                ingest._IngestUnitWriter,
                "__call__",
                side_effect=AssertionError("replayed ingest unit"),
            ),
        ):
            assert runner.run_single_n(cfg, 2, roster) > 0
            ingest.run(cfg)
        assert before == [(sha256_file(p), p.stat().st_mtime_ns) for p in evidence]
        from farkle.utils.completion_files import CompletionFileKind, CompletionNamespace

        namespace = CompletionNamespace.build(
            graph_root=cfg.results_root, analysis_root=cfg.analysis_dir, canonical_paths=(done,)
        )
        unit_stamps = list(cfg.stage_dir("ingest").rglob("*.unit.done.json"))
        assert len(unit_stamps) == 1
        assert namespace.classify(unit_stamps[0]) is CompletionFileKind.PARTITION_UNIT
        index, errors = release_audit._build_audit_index(cfg.results_root.resolve())
        assert not errors
        assert not release_audit._validate_indexed_artifacts(index)
        assert str(artifact.resolve()) in release_audit._foreign_producers(
            index, cfg._code_identity
        )
        assert (
            load_authenticated_sidecar(cfg.ingested_rows_raw(2)).stage_identity.code
            == original.stage_identity.code
        )
        target = capture_target()
        result = audit()
        assert result["status"] == "passed", result["failures"]
        policy = result["revision_policy"]
        if producer_state == "hidden_edit":
            assert original.stage_identity.code.state == "clean"
            assert policy["status"] == "ineligible", policy
            assert "implementation" in str(policy["reasons"])
            return
        if original.stage_identity.code.state != "clean":
            assert policy["status"] == "ineligible", policy
            assert "clean producer" in str(policy["reasons"])
            return
        assert policy["status"] == "qualified_mixed", policy
        assert policy["producer_commits"] == sorted(
            [original.stage_identity.code.commit, cfg._code_identity.commit]
        )
        assert [item["code_identity"]["commit"] for item in policy["invocation_records"]] == [
            original.stage_identity.code.commit,
            code.commit,
        ]
        # The same retained root is the preimage a pair must prove, including its stamp order.
        from farkle.utils.manifest import iter_manifest

        event = list(iter_manifest(invocation_manifest))[-1]
        parent_context, receipt = audit_published_launch(cfg.results_root, event)
        parent = (
            cfg.results_root,
            invocation,
            receipt,
            parent_context,
            target.snapshot.lifecycle_sha256,
        )
        audit_parent_lifecycle(*parent)
        stamp = Path(event["invocation_record"]).parent / "lifecycle" / "0.stamp"
        original_stamp = stamp.read_bytes()
        stamp.write_bytes(original_stamp + b" ")
        with pytest.raises(ValueError, match="historical parent lifecycle"):
            audit_parent_lifecycle(*parent)
        stamp.write_bytes(original_stamp)
        history = cfg.results_root / "_provenance" / "invocations" / "first" / "invocation.json"
        archived = history.read_bytes()
        history.write_bytes(archived + b" ")
        refused = audit()
        assert refused["revision_policy"]["status"] == "ineligible"
        assert "digest differs" in str(refused["revision_policy"]["reasons"])
        # Even a hash-consistent historical receipt must describe this producer.
        events = invocation_manifest.read_bytes()
        invocation_manifest.write_bytes(events.split(b"\n", 1)[1])
        assert audit()["revision_policy"]["status"] == "ineligible"
        invocation_manifest.write_bytes(events)
        legacy = json.loads(archived)
        del legacy["scientific_launch_v1"]["producers"]["root/simulation"]
        history.write_text(json.dumps(legacy), encoding="utf-8")
        from hashlib import sha256

        invocation_manifest.write_bytes(
            events.replace(sha256(archived).hexdigest().encode(), sha256_file(history).encode())
        )
        missing = audit()["revision_policy"]
        assert missing["status"] == "ineligible" and "producer launch" in str(missing["reasons"])
        history.write_bytes(archived)
        invocation_manifest.write_bytes(events)
        original_bytes = artifact.read_bytes()
        artifact.write_bytes(original_bytes + b"corruption")
        damaged = audit()
        assert (
            damaged["status"] == "failed" and damaged["revision_policy"]["status"] == "ineligible"
        )
        artifact.write_bytes(original_bytes)


@pytest.mark.parametrize("producer_state", ["clean", "dirty", "hidden_edit"])
def test_resume_after_documentation_commit(tmp_path: Path, producer_state: str) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    source = Path(__file__).resolve().parents[3] / "src"
    shutil.copytree(source, checkout / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (checkout / ".gitignore").write_text("__pycache__/\n")

    def git(*args: str) -> None:
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Resume fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=checkout,
            check=True,
            capture_output=True,
        )

    git("init")
    git("add", ".")
    git("commit", "-m", "Scientific implementation")
    env = {
        **os.environ,
        "PYTHONPATH": str(checkout / "src"),
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    engine = checkout / "src/farkle/game/engine.py"
    if producer_state == "dirty":
        (checkout / "README.md").write_text("Uncommitted working-tree change.\n")
    if producer_state == "hidden_edit":
        git("update-index", "--assume-unchanged", "src/farkle/game/engine.py")
        with engine.open("a", encoding="utf-8") as handle:
            handle.write("\n# Present at both launches, absent from both commits.\n")
    invocations = ["first", "resume", "changed"]
    if producer_state == "clean":
        invocations.insert(2, "replay")
    for invocation in invocations:
        if invocation == "resume":
            (checkout / "README.md").write_text("A documentation-only revision.\n")
            git("add", "README.md")
            git("commit", "-m", "Document fixture")
        if invocation == "replay":
            (checkout / "README.md").write_text("Document the completed fixture.\n")
            git("add", "README.md")
            git("commit", "-m", "Document completed fixture")
        if invocation == "changed":
            git("update-index", "--no-assume-unchanged", "src/farkle/game/engine.py")
            with engine.open("a", encoding="utf-8") as handle:
                handle.write("\n# A new engine implementation revision.\n")
            git("add", str(engine.relative_to(checkout)))
            git("commit", "-m", "Change engine implementation")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(tmp_path),
                invocation,
                producer_state,
            ],
            cwd=checkout,
            env=env,
            check=True,
            timeout=240,
        )


if __name__ == "__main__":
    _run_fixture(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
