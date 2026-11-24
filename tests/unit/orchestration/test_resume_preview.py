from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import pyarrow as pa
import pytest
from tests.helpers.artifact_sidecars import publish_v3_parquet
from tests.unit.analysis.test_h2h_schedule import _write_frozen_family

from farkle import analysis
from farkle.analysis import h2h_authorization, h2h_schedule, ingest
from farkle.cli import launcher
from farkle.config import AppConfig, ArtifactScope
from farkle.orchestration import resume_preview, two_seed_pipeline
from farkle.orchestration.run_contexts import RootPairRunContext, write_run_context_atomic
from farkle.orchestration.seed_utils import write_active_config
from farkle.simulation import runner
from farkle.utils import evidence_reader, scientific_runtime
from farkle.utils.artifact_contract import sidecar_path
from farkle.utils.authenticated_contract import (
    CodeIdentity,
    CodeIdentityPolicy,
    load_authenticated_sidecar,
    resolve_code_identity,
)
from farkle.utils.scientific_invocation import scientific_invocation
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done

ROOTS = (42, 43)
SIMULATION = "root 42/simulation/2p"


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    cfg.sim.seed, cfg.sim.seed_list = ROOTS[0], list(ROOTS)
    cfg.sim.n_players_list = [2]
    cfg.sim.score_thresholds = [300, 500]
    cfg.sim.dice_thresholds = [2]
    cfg.sim.smart_five_opts = cfg.sim.smart_one_opts = [False]
    cfg.sim.consider_score_opts = cfg.sim.consider_dice_opts = [True]
    cfg.sim.auto_hot_dice_opts = cfg.sim.run_up_score_opts = [False]
    cfg.sim.row_dir = Path("rows")
    cfg.sim.n_jobs = 1
    cfg.sim.ckpt_every_sec = 0
    cfg.screening.resolution_delta = 0.9
    cfg.screening.practical_delta_by_k = {2: 0.03}
    cfg.batching.target_batches = 2
    cfg.batching.min_shuffles_per_batch = 30
    return cfg


@pytest.fixture
def small_build(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep real producer projections and byte authentication on these small
    # fixture publications while avoiding repeated installed-wheel measurements.
    monkeypatch.setattr(
        scientific_runtime,
        "numerical_environment",
        lambda names, **_: ("1" * 64, dict.fromkeys(names, "2" * 64)),
    )


def _pair(cfg: AppConfig) -> RootPairRunContext:
    pair, _, _ = two_seed_pipeline._prepare_pipeline(cfg, ROOTS)
    return pair


def _actions(report: resume_preview.ResumePreview) -> dict[str, resume_preview.Decision]:
    return {entry.scope: entry for entry in report.decisions}


def _snapshot(paths: Iterable[Path]) -> dict[str, tuple[int, ...]]:
    return {
        str(path): (
            stat.st_size,
            stat.st_atime_ns,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            getattr(stat, "st_birthtime_ns", 0),
        )
        for path in paths
        if (stat := path.stat())
    }


def _old_access_times(root: Path) -> list[Path]:
    paths = [root, *root.rglob("*")]
    for path in paths:
        os.utime(path, ns=(946684800000000000, path.stat().st_mtime_ns))
    return paths


def _unexpected(*args, **kwargs):
    pytest.fail(f"preview unexpectedly entered execution: {args!r}")


def test_public_preview_is_read_only_and_formats_agree(
    cfg: AppConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from farkle.cli import main as cli

    monkeypatch.setattr(launcher, "supervise_process", _unexpected)
    monkeypatch.setattr(cli, "configure_logging", _unexpected)
    monkeypatch.setattr(cli, "cap_authorization_policy", _unexpected)
    monkeypatch.setattr(two_seed_pipeline, "run_pipeline", _unexpected)
    monkeypatch.setattr(scientific_runtime, "numerical_environment", _unexpected)
    arguments = [
        "--set",
        f"io.results_dir_prefix={cfg.io.results_dir_prefix}",
        "two-seed-pipeline",
        "--seed-pair",
        "42",
        "43",
        "--resume-preview",
    ]
    assert launcher.main([*arguments, "--preview-format", "json"]) == 0
    machine = json.loads(capsys.readouterr().out)
    assert machine["schema_version"] == 1
    assert machine["roots"] == [42, 43]
    assert not any(tmp_path.iterdir())
    assert launcher.main(arguments) == 0
    human = capsys.readouterr().out
    for entry in machine["decisions"]:
        assert entry["scope"] in human
        assert entry["action"].replace("_", " ") in human
        assert entry["reason"] in human
        assert not entry["provisional"]
    assert not any(tmp_path.iterdir())


def test_namespace_conflict_is_a_definitive_block(cfg: AppConfig) -> None:
    pair = _pair(cfg)
    pair.pair_root.mkdir()
    pair.run_context_path.write_text(json.dumps({"public_config_sha256": "invalid"}))
    paths = _old_access_times(pair.pair_root)
    before = _snapshot(paths)
    report = resume_preview.preview_pipeline(cfg, seed_pair=ROOTS)
    decision = _actions(report)["pair 42/43"]
    assert decision.action == "blocked"
    assert "prevents exact resume" in decision.reason
    assert not decision.provisional
    assert all("/simulation/" not in item.scope for item in report.decisions)
    assert _snapshot(paths) == before


def test_large_manifest_is_deferred_without_opening(
    cfg: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = _pair(cfg)
    pair.pair_root.mkdir()
    ledger = pair.pair_root / "two_seed_pipeline_manifest.jsonl"
    with ledger.open("wb") as handle:
        handle.truncate(evidence_reader.MAX_PREVIEW_BYTES + 1)
    monkeypatch.setattr(evidence_reader, "_open_without_access_update", _unexpected)
    action = _actions(resume_preview.preview_pipeline(cfg, seed_pair=ROOTS))[
        "pair 42/43/pipeline manifest"
    ]
    assert action.action == "verification_required"
    assert "preview read limit" in action.reason


def test_force_defers_the_sealed_cell_check(cfg: AppConfig) -> None:
    root = _pair(cfg).root_contexts[0].config
    checkpoint = root.checkpoint_path(2)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"unsealed checkpoint")
    action = _actions(resume_preview.preview_pipeline(cfg, seed_pair=ROOTS, force=True))[SIMULATION]
    assert action.action == "verification_required"
    assert "look for sealed files" in action.reason
    assert checkpoint.exists()


@pytest.fixture
def tiny_workload(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = runner.plan_tournament_workload
    monkeypatch.setattr(
        runner,
        "plan_tournament_workload",
        lambda **kwargs: planner(**(kwargs | {"min_shuffles_per_batch": 1})),
    )


@pytest.fixture
def completed_simulation(
    cfg: AppConfig, small_build: None, tiny_workload: None
) -> RootPairRunContext:
    pair = _pair(cfg)
    for context in pair.root_contexts:
        root = context.config
        assert root._code_identity is not None
        write_run_context_atomic(context, code_identity=root._code_identity)
        write_active_config(root, context.results_root)
        assert runner.run_single_n(root, 2) > 0
    return pair


def test_completed_cells_and_resource_change(
    cfg: AppConfig, completed_simulation: RootPairRunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = completed_simulation
    for context in pair.root_contexts:
        with scientific_invocation([(context.config, "simulation")]):
            assert runner.simulation_is_complete(context.config, 2)
    cfg.sim.n_jobs = 2
    cfg.resources.logical_cpu_budget = 2
    cfg.io.temp_dir = pair.pair_root.parent / "unused-scratch"
    paths = _old_access_times(pair.pair_root)
    before = _snapshot(paths)

    io = []
    opener = evidence_reader._open_without_access_update

    def measured_open(path: Path):
        io.append((path, path.stat().st_size))
        return opener(path)

    monkeypatch.setattr(evidence_reader, "_open_without_access_update", measured_open)
    monkeypatch.setattr(scientific_runtime, "numerical_environment", _unexpected)
    monkeypatch.setattr(two_seed_pipeline, "_final_release_gate", _unexpected)
    started = perf_counter()
    decisions = _actions(resume_preview.preview_pipeline(cfg, seed_pair=ROOTS))
    elapsed = perf_counter() - started
    assert decisions["root 42"].action == "reusable"
    for seed in ROOTS:
        cell = decisions[f"root {seed}/simulation/2p"]
        assert cell.action == "verification_required"
        assert cell.provisional and "provisionally compatible" in cell.reason
    assert not cfg.io.temp_dir.exists()
    assert all(p.suffix not in {".parquet", ".pkl"} for p, _ in io)
    assert len({path for path, _ in io}) == len(io)
    assert _snapshot(paths) == before
    assert {pair.pair_root, *pair.pair_root.rglob("*")} == set(paths)
    print(
        f"complete cells: {elapsed:.3f}s, {len(io)} evidence opens, {sum(size for _, size in io)} bytes"
    )


@pytest.mark.parametrize("mutation", ["missing", "same_length", "partial", "manifest", "no_replay"])
def test_aggregate_recovery_classification(
    cfg: AppConfig,
    completed_simulation: RootPairRunContext,
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = completed_simulation.root_contexts[0].config
    runner.simulation_done_path(root, 2).unlink()
    checkpoint = root.checkpoint_path(2)
    row_dir = root.simulation_row_dir(2)
    assert row_dir is not None
    manifest = row_dir / "manifest.jsonl"
    if mutation in {"same_length", "no_replay"}:
        data = checkpoint.read_bytes()
        checkpoint.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    else:
        checkpoint.unlink()
    if mutation in {"partial", "no_replay"}:
        sidecar_path(manifest).unlink()
        if mutation == "partial":
            manifest.write_bytes(manifest.read_bytes().splitlines(keepends=True)[0])
        else:
            manifest.unlink()
    if mutation == "manifest":
        manifest.write_bytes(manifest.read_bytes() + b"corrupt")
    paths = _old_access_times(completed_simulation.pair_root)
    before = _snapshot(paths)
    with monkeypatch.context() as guard:
        guard.setattr(runner.tournament_mod, "_read_checkpoint", _unexpected)
        start = perf_counter()
        decisions = _actions(resume_preview.preview_pipeline(cfg, seed_pair=ROOTS))
    print(f"{mutation} checkpoint/manifest: {perf_counter() - start:.3f}s")
    assert _snapshot(paths) == before
    if mutation == "manifest":
        assert decisions[SIMULATION].action == "blocked"
        assert "sealed simulation manifest bytes differ" in decisions[SIMULATION].reason
    elif mutation == "no_replay":
        assert decisions[f"{SIMULATION}/checkpoint"].action == "blocked"
    else:
        assert decisions[f"{SIMULATION}/checkpoint"].action == "aggregate_reconstruction_required"
        if mutation == "partial":
            assert "1/2" in decisions[f"{SIMULATION}/checkpoint"].reason
            assert decisions[SIMULATION].action == "scientific_recomputation_required"
        else:
            assert "2/2" in decisions[f"{SIMULATION}/checkpoint"].reason
            assert SIMULATION not in decisions
            if mutation == "same_length":
                assert "checkpoint bytes differ" in decisions[f"{SIMULATION}/checkpoint"].reason
    if mutation == "missing":
        monkeypatch.setattr(runner.tournament_mod, "_play_one_shuffle", _unexpected)
        assert runner.run_single_n(root, 2) > 0
        assert runner.simulation_is_complete(root, 2)


def test_incomplete_ingest_reports_the_durable_partition(
    cfg: AppConfig, completed_simulation: RootPairRunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = completed_simulation.root_contexts[0].config

    def interrupted(*args: Any, **kwargs: Any) -> None:
        raise InterruptedError("before stage completion")

    with monkeypatch.context() as guard:
        guard.setattr(ingest, "write_stage_done", interrupted)
        with pytest.raises(InterruptedError, match="before stage completion"):
            ingest.run(root)
    paths = _old_access_times(completed_simulation.pair_root)
    before = _snapshot(paths)
    decisions = _actions(resume_preview.preview_pipeline(cfg, seed_pair=ROOTS))
    assert decisions["root 42/ingest"].action == "verification_required"
    partition = decisions["root 42/ingest/2p"]
    assert partition.action == "verification_required"
    assert "Durable unit metadata found" in partition.reason
    assert _snapshot(paths) == before


def test_frozen_h2h_above_cap_and_recorded_consent(
    cfg: AppConfig, small_build: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg.head2head.total_game_cap = 1
    cfg.screening.candidate_contribution_size = 2
    pair = _pair(cfg)
    pair_cfg = pair.config
    _write_frozen_family(pair_cfg, (1, 2))
    monkeypatch.setattr(h2h_schedule, "_minimum_block_games", lambda **_: 2)
    monkeypatch.setattr(h2h_schedule, "_worst_scenario_power", lambda **_: 0.9)
    h2h_schedule.plan_h2h_schedule(pair_cfg)
    power = pair_cfg.h2h_power_plan_path()
    plan = json.loads(power.read_bytes())
    scope = "pair 42/43/h2h authorization"
    token = h2h_authorization.authorization_token(plan)
    paths = _old_access_times(pair.pair_root)
    before = _snapshot(paths)
    monkeypatch.setattr("builtins.input", _unexpected)
    blocked = _actions(resume_preview.preview_pipeline(cfg, seed_pair=ROOTS))[scope]
    assert blocked.action == "blocked"
    assert token in blocked.reason
    supplied = _actions(resume_preview.preview_pipeline(cfg, seed_pair=ROOTS, h2h_override=token))
    assert supplied[scope].action == "verification_required"
    assert "Supplied authorization token matches" in supplied[scope].reason
    assert _snapshot(paths) == before
    with h2h_authorization.cap_authorization_policy(override=token):
        h2h_authorization.resolve_cap_authorization(pair_cfg, plan, request=True)
    approved = _actions(resume_preview.preview_pipeline(cfg, seed_pair=ROOTS))[scope]
    assert approved.action == "verification_required"
    assert "Retained authorization matches" in approved.reason


def _publish_stage(cfg: AppConfig, stage: str) -> tuple[Path, Path]:
    output = cfg.scope_path(stage, ArtifactScope.DIAGNOSTICS, "fixture.parquet")
    publish_v3_parquet(
        cfg,
        output,
        pa.table({"value": [1]}),
        stage_key=stage,
        producer=stage,
        operation="test_preview_fixture",
    )
    done = stage_done_path(cfg.stage_dir(stage), stage)
    write_stage_done(done, inputs=[], outputs=[output], cfg=cfg, stage=stage)
    return done, output


def test_complete_bounded_root_root_pair_plan(
    cfg: AppConfig, completed_simulation: RootPairRunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = completed_simulation
    build_root = analysis.build_root_stage_plan
    build_pair = analysis.build_root_pair_stage_plan
    for root in pair.root_contexts:
        _publish_stage(root.config, "game_stats")
    pair_item = next(item for item in build_pair(pair) if item.name == "trueskill")
    assert pair_item.completion_stamp is not None
    (output,) = pair_item.required_outputs
    publish_v3_parquet(
        pair.config,
        output,
        pa.table({"value": [1]}),
        stage_key="trueskill",
        producer="trueskill",
        operation="test_preview_fixture",
    )
    write_stage_done(
        pair_item.completion_stamp, inputs=[], outputs=[output], cfg=pair.config, stage="trueskill"
    )
    monkeypatch.setattr(
        analysis,
        "build_root_stage_plan",
        lambda inner, **kwargs: [
            item for item in build_root(inner, **kwargs) if item.name == "game_stats"
        ],
    )
    monkeypatch.setattr(
        analysis,
        "build_root_pair_stage_plan",
        lambda inner, **kwargs: [
            item for item in build_pair(inner, **kwargs) if item.name == "trueskill"
        ],
    )
    decisions = _actions(resume_preview.preview_pipeline(cfg, seed_pair=ROOTS))
    assert decisions["pair 42/43/trueskill"].action == "verification_required"
    for root in ROOTS:
        assert decisions[f"root {root}/game_stats"].action == "verification_required"
        assert decisions[f"root {root}/simulation/2p"].action == "verification_required"
    assert {item.action for item in decisions.values()} == {"reusable", "verification_required"}


def test_selective_config_change_matches_execution(cfg: AppConfig, small_build: None) -> None:
    root = _pair(cfg).root_contexts[0].config
    hgb, hgb_output = _publish_stage(root, "hgb")
    stats, stats_output = _publish_stage(root, "game_stats")
    for name, stamp, output in (("hgb", hgb, hgb_output), ("game_stats", stats, stats_output)):
        assert stage_is_up_to_date(stamp, inputs=[], outputs=[output], cfg=root, stage=name)
    cfg.hgb.max_depth = cfg.hgb.max_depth + 1
    actions = _actions(resume_preview.preview_pipeline(cfg, seed_pair=ROOTS))
    assert actions["root 42/hgb"].action == "scientific_recomputation_required"
    assert "scoped scientific configuration differs" in actions["root 42/hgb"].reason
    assert actions["root 42/game_stats"].action == "verification_required"
    changed = _pair(cfg).root_contexts[0].config
    assert not stage_is_up_to_date(hgb, inputs=[], outputs=[hgb_output], cfg=changed, stage="hgb")
    assert stage_is_up_to_date(
        stats, inputs=[], outputs=[stats_output], cfg=changed, stage="game_stats"
    )


def test_scientific_source_scope_across_git_commits(
    cfg: AppConfig,
    small_build: None,
    tiny_workload: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from farkle.analysis import release_audit
    from farkle.utils import scientific_invocation as invocations
    from farkle.utils.scientific_dependencies import Producer, dependencies_for
    from farkle.utils.scientific_reuse import compatible_producer

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    relevant = set().union(
        *(dependencies_for(Producer("root", stage)).source_files for stage in ("simulation", "hgb"))
    )
    for name in relevant:
        source = resume_preview.ROOT / name
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())

    def commit(message: str) -> CodeIdentity:
        for arguments in (("add", "."), ("commit", "-m", message)):
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=test@example.invalid",
                    "-c",
                    "commit.gpgsign=false",
                    *arguments,
                ],
                cwd=checkout,
                check=True,
                capture_output=True,
            )
        return resolve_code_identity(checkout, policy=CodeIdentityPolicy.RELEASE_CLEAN)

    subprocess.run(["git", "init"], cwd=checkout, check=True, capture_output=True)
    original = commit("Initial implementation")
    root = _pair(cfg).root_contexts[0].config
    root._code_identity = original
    with scientific_invocation([(root, "simulation")]):
        runner.run_single_n(root, 2)
    metadata = load_authenticated_sidecar(root.strategy_manifest_root_path()).stage_identity
    assert resume_preview.has_scoped_producer_evidence(metadata)
    (checkout / "README.md").write_text("Documentation only.\n", encoding="utf-8")
    docs = commit("Documentation")
    # Actual execution independently compares its current scoped digests.
    root._code_identity = docs
    with scientific_invocation([(root, "simulation")]):
        assert compatible_producer(metadata, docs)
        assert runner.simulation_is_complete(root, 2)
    monkeypatch.setattr(resume_preview, "ROOT", checkout)
    monkeypatch.setattr(invocations, "ROOT", checkout)
    assert release_audit._committed_files(docs.commit)
    monkeypatch.setattr(two_seed_pipeline, "resolve_code_identity", lambda *_args, **_kwargs: docs)
    started = perf_counter()
    action = _actions(resume_preview.preview_pipeline(cfg, seed_pair=ROOTS))[SIMULATION]
    print(f"clean documentation revision: {perf_counter() - started:.3f}s")
    assert action.action == "verification_required"
    assert "provisionally compatible" in action.reason
    hgb_source = checkout / "src/farkle/analysis/hgb_feat.py"
    hgb_source.write_bytes(hgb_source.read_bytes() + b"\n# Changed producer.\n")
    changed = commit("HGB source change")
    inspection = resume_preview._Inspection(changed, force=False)
    inspection.require_science(root, metadata)
    assert inspection._first_changed_source(Producer("root", "hgb"), original.commit) == (
        "src/farkle/analysis/hgb_feat.py"
    )
