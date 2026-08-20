"""Tests for canonical root and root-pair orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import (
    AnalysisConfig,
    AppConfig,
    BatchingConfig,
    IOConfig,
    ProfileConfig,
    ScreeningConfig,
    SimConfig,
    apply_dot_overrides,
    assign_config_sha,
    load_app_config,
)
from farkle.orchestration import two_seed_pipeline
from farkle.orchestration.run_contexts import (
    SEED_PAIR_ANALYSIS_DIRNAME,
    RootPairRunContext,
    SeedRunContext,
)
from farkle.utils.authenticated_contract import CodeIdentity, CodeIdentityPolicy


def _context(tmp_path: Path, root: int) -> SeedRunContext:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / f"root_{root}"),
        sim=SimConfig(seed=root, seed_list=[root], n_players_list=[2]),
    )
    assign_config_sha(cfg)
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    return SeedRunContext.from_config(cfg)


def test_run_per_root_analysis_stops_before_h2h(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _context(tmp_path, 9).config
    captured: dict[str, Any] = {}

    def fake_run(plan: list[Any], context: Any, raise_on_failure: bool) -> None:
        captured["plan"] = [item.name for item in plan]
        captured["label"] = context.run_label
        captured["raise"] = raise_on_failure

    monkeypatch.setattr(two_seed_pipeline.StageRunner, "run", staticmethod(fake_run))
    policy = two_seed_pipeline._derive_per_seed_job_budgets(cfg, seed_count=1)

    two_seed_pipeline._run_per_seed_analysis(
        cfg,
        seed=9,
        force=False,
        policy_bundle=policy,
    )

    assert captured["plan"][-1] == "screening"
    assert not {"candidate_freeze", "h2h_execute", "h2h_digest"}.intersection(captured["plan"])
    assert captured["label"] == "per_seed_pipeline_9"
    assert captured["raise"] is True


def test_root_config_replaces_pair_seed_list_with_owned_root(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
    )
    policy = two_seed_pipeline._derive_per_seed_job_budgets(cfg, seed_count=2)

    root_cfg = two_seed_pipeline._build_seed_cfg(
        cfg,
        seed_pair=(11, 22),
        seed=22,
        policy_bundle=policy,
    )

    assert root_cfg.sim.seed == 22
    assert root_cfg.sim.seed_list == [22]


def _install_root_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    failed_root: int | None = None,
    failure_classification: str | None = None,
    rejected_snapshot_root: int | None = None,
) -> None:
    class FakeGeneration:
        def __init__(self, *, rejected: bool = False) -> None:
            self.rejected = rejected

        def validate(self, *_args: object, **_kwargs: object) -> None:
            if self.rejected:
                raise RuntimeError("explicitly invalidated test snapshot")
            return None

    def fake_snapshot(scope: str, roots: tuple[int, ...], root: Path) -> Any:
        return SimpleNamespace(
            scope=scope,
            roots=roots,
            graph_root=root,
            lifecycle_sha256="e" * 64,
            stage_states=(("screening", "complete_valid"),),
        )

    def fake_run_one_seed(
        _cfg: AppConfig,
        *,
        seed: int,
        **_kwargs: object,
    ) -> two_seed_pipeline._SeedRunStatus:
        context = _context(tmp_path, seed)
        if seed == failed_root:
            return two_seed_pipeline._SeedRunStatus(
                seed=seed,
                context=context,
                simulation_ok=True,
                analysis_ok=False,
                analysis_error="root analysis failed",
                failure_classification=failure_classification,
            )
        snapshot = fake_snapshot("root", (seed,), context.results_root)
        return two_seed_pipeline._SeedRunStatus(
            seed=seed,
            context=context,
            simulation_ok=True,
            analysis_ok=True,
            lifecycle_sha256=snapshot.lifecycle_sha256,
            stage_states=dict(snapshot.stage_states),
            graph_snapshot=snapshot,
            snapshot_generation=FakeGeneration(rejected=seed == rejected_snapshot_root),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(two_seed_pipeline, "_run_one_seed", fake_run_one_seed)
    monkeypatch.setattr(
        two_seed_pipeline,
        "resolve_code_identity",
        lambda *_args, **_kwargs: CodeIdentity(
            commit="a" * 40,
            policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
            state="clean",
            dirty_fingerprint_sha256=None,
        ),
    )
    monkeypatch.setattr(two_seed_pipeline, "validate_manifest_contract", lambda _path: None)
    monkeypatch.setattr(two_seed_pipeline, "append_manifest_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(two_seed_pipeline, "write_active_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        two_seed_pipeline,
        "_build_pair_authenticated_snapshot",
        lambda context, *_args, **_kwargs: (
            fake_snapshot("pair", tuple(sorted(context.root_pair)), context.analysis_root),
            FakeGeneration(),
        ),
    )
    monkeypatch.setattr(
        two_seed_pipeline,
        "_current_plan_states",
        lambda _cfg, plan: {
            item.name: two_seed_pipeline.CompletionState.COMPLETE_VALID.value for item in plan
        },
    )
    monkeypatch.setattr(
        two_seed_pipeline,
        "_root_lifecycle_identity",
        lambda _context, _plan: ("e" * 64, {"simulation_2p": "complete_valid"}),
    )
    monkeypatch.setattr(
        two_seed_pipeline,
        "_final_release_gate",
        lambda *_args, **_kwargs: {
            "status": "passed",
            "release_eligible": True,
            "accepted_release_identity": [3, 2, 2, 2, 2, 2],
            "artifact_roots": [],
            "run_contexts": {},
            "failures": [],
        },
    )


def test_two_seed_pipeline_runs_pair_tail_once_at_pair_analysis_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
        profile=ProfileConfig(
            purpose="integration",
            reduced_resolution=True,
            production_eligible=False,
            release_eligible=False,
        ),
        batching=BatchingConfig(target_batches=20),
        screening=ScreeningConfig(practical_delta_by_k={2: 0.03}, delta_across_k=0.03),
    )
    _install_root_results(monkeypatch, tmp_path)
    calls: list[RootPairRunContext] = []
    health: dict[str, Any] = {}
    monkeypatch.setattr(
        two_seed_pipeline.analysis,
        "run_root_pair_analysis",
        lambda context, **_: calls.append(context),
    )
    monkeypatch.setattr(
        two_seed_pipeline,
        "_write_pipeline_health",
        lambda _path, payload: health.update(payload),
    )

    two_seed_pipeline.run_pipeline(cfg, seed_pair=(11, 22))

    assert len(calls) == 1
    context = calls[0]
    pair_root = two_seed_pipeline.seed_pair_root(cfg, (11, 22))
    assert context.root_pair == (11, 22)
    assert context.analysis_root == pair_root / SEED_PAIR_ANALYSIS_DIRNAME
    assert health["status"] == "complete_success"
    assert health["pair_workflow"]["analysis_root"] == str(context.analysis_root)
    policy = health["resource_telemetry"]["resource_policy"]
    assert set(policy) == {"requested", "resolved", "effective"}
    assert policy["requested"]["aggregate_memory_hard_limit_mb"] == 2304
    assert policy["resolved"]["logical_cpu_budget"] >= 1
    assert (
        policy["effective"]["aggregate_memory_hard_limit_mb"]
        == health["resource_telemetry"]["os_memory_boundary"]["effective_hard_limit_mb"]
    )
    worker_policy = health["resource_telemetry"]["worker_policy"]
    assert worker_policy["simulation"]["requested_n_jobs"] is None
    assert worker_policy["simulation"]["effective_n_jobs"] == 1
    timing = health["timing_summary"]
    assert timing["schema_version"] == 1
    assert timing["heartbeat_interval_seconds"] == 45.0
    phase_names = {phase["stage"] for phase in timing["pair"]["finalization_phases"]}
    assert {
        "pair_analysis",
        "pair_authenticated_graph_snapshot",
        "root_11_authenticated_graph_snapshot_reuse",
        "root_22_authenticated_graph_snapshot_reuse",
        "final_byte_deep_release_audit",
        "resource_final_check",
    }.issubset(phase_names)
    assert timing["pair"]["post_summary_phases"] == [
        "pipeline_health_publish",
        "run_end_manifest_publish",
    ]


def test_pipeline_health_is_not_rewritten_for_heartbeats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
        profile=ProfileConfig(
            purpose="integration",
            reduced_resolution=True,
            production_eligible=False,
            release_eligible=False,
        ),
        batching=BatchingConfig(target_batches=20),
        screening=ScreeningConfig(practical_delta_by_k={2: 0.03}, delta_across_k=0.03),
    )
    _install_root_results(monkeypatch, tmp_path)
    monkeypatch.setattr(
        two_seed_pipeline.analysis,
        "run_root_pair_analysis",
        lambda *_args, **_kwargs: None,
    )
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(
        two_seed_pipeline,
        "_write_pipeline_health",
        lambda _path, payload: writes.append(dict(payload)),
    )

    two_seed_pipeline.run_pipeline(cfg, seed_pair=(11, 22))

    assert len(writes) == 2
    assert writes[0]["status"] == "running"
    assert writes[0]["profile"]["purpose"] == "integration"
    assert writes[0]["profile"]["production_eligible"] is False
    assert writes[0]["profile"]["release_eligible"] is False
    assert "timing_summary" not in writes[0]
    assert writes[1]["status"] == "complete_success"
    assert writes[1]["profile"]["purpose"] == "integration"
    assert "timing_summary" in writes[1]


def test_final_release_gate_executes_one_top_level_audit_with_three_internal_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _context(tmp_path, 11)
    second = _context(tmp_path, 22)
    pair = RootPairRunContext.from_root_contexts(
        (first, second),
        pair_root=tmp_path / "pair",
    )
    code = CodeIdentity(
        commit="a" * 40,
        policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
        state="clean",
        dirty_fingerprint_sha256=None,
    )
    snapshots = {
        11: cast(
            two_seed_pipeline.AuthenticatedGraphSnapshot,
            SimpleNamespace(graph_root=first.results_root),
        ),
        22: cast(
            two_seed_pipeline.AuthenticatedGraphSnapshot,
            SimpleNamespace(graph_root=second.results_root),
        ),
    }
    generations = {
        11: cast(two_seed_pipeline.SnapshotGeneration, object()),
        22: cast(two_seed_pipeline.SnapshotGeneration, object()),
    }
    pair_snapshot = cast(
        two_seed_pipeline.AuthenticatedGraphSnapshot,
        SimpleNamespace(graph_root=pair.analysis_root),
    )
    calls: list[object] = []
    monkeypatch.setattr(two_seed_pipeline, "resolve_code_identity", lambda *_a, **_k: code)
    monkeypatch.setattr(
        two_seed_pipeline,
        "audit_authenticated_release_graphs",
        lambda targets, **_kwargs: calls.append(tuple(targets))
        or {
            "status": "passed",
            "failures": [],
            "run_contexts": {},
            "internal_roots": [{"label": "root_11"}, {"label": "root_22"}, {"label": "pair"}],
            "top_level_invocations": 1,
        },
    )

    result = two_seed_pipeline._final_release_gate(
        {
            11: two_seed_pipeline._SeedRunStatus(
                11,
                first,
                True,
                True,
                graph_snapshot=snapshots[11],
                snapshot_generation=generations[11],
            ),
            22: two_seed_pipeline._SeedRunStatus(
                22,
                second,
                True,
                True,
                graph_snapshot=snapshots[22],
                snapshot_generation=generations[22],
            ),
        },
        pair,
        code_identity=code,
        allow_oracle_code_identity=False,
        pair_snapshot=pair_snapshot,
        pair_generation=cast(two_seed_pipeline.SnapshotGeneration, object()),
        authentication_telemetry=two_seed_pipeline.AuthenticationTelemetry(),
    )

    assert result["status"] == "passed"
    assert result["release_eligible"] is True
    assert result["profile_release_eligible"] is True
    assert result["top_level_invocations"] == 1
    assert len(result["internal_roots"]) == 3
    assert len(calls) == 1


def test_clean_integration_profile_passes_audit_but_is_not_release_eligible(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _context(tmp_path, 11)
    second = _context(tmp_path, 22)
    pair = RootPairRunContext.from_root_contexts((first, second), pair_root=tmp_path / "pair")
    pair.config.profile = ProfileConfig(
        purpose="integration",
        reduced_resolution=True,
        production_eligible=False,
        release_eligible=False,
    )
    code = CodeIdentity(
        commit="a" * 40,
        policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
        state="clean",
        dirty_fingerprint_sha256=None,
    )
    monkeypatch.setattr(two_seed_pipeline, "resolve_code_identity", lambda *_a, **_k: code)
    monkeypatch.setattr(
        two_seed_pipeline,
        "audit_authenticated_release_graphs",
        lambda *_args, **_kwargs: {
            "status": "passed",
            "failures": [],
            "run_contexts": {},
            "internal_roots": [],
            "top_level_invocations": 1,
        },
    )
    root_snapshot = cast(
        two_seed_pipeline.AuthenticatedGraphSnapshot,
        SimpleNamespace(graph_root=first.results_root),
    )
    snapshot_generation = cast(two_seed_pipeline.SnapshotGeneration, object())

    result = two_seed_pipeline._final_release_gate(
        {
            11: two_seed_pipeline._SeedRunStatus(
                11,
                first,
                True,
                True,
                graph_snapshot=root_snapshot,
                snapshot_generation=snapshot_generation,
            ),
            22: two_seed_pipeline._SeedRunStatus(
                22,
                second,
                True,
                True,
                graph_snapshot=root_snapshot,
                snapshot_generation=snapshot_generation,
            ),
        },
        pair,
        code_identity=code,
        allow_oracle_code_identity=False,
        pair_snapshot=cast(
            two_seed_pipeline.AuthenticatedGraphSnapshot,
            SimpleNamespace(graph_root=pair.analysis_root),
        ),
        pair_generation=cast(two_seed_pipeline.SnapshotGeneration, object()),
        authentication_telemetry=two_seed_pipeline.AuthenticationTelemetry(),
    )

    assert result["status"] == "passed"
    assert result["profile_release_eligible"] is False
    assert result["release_eligible"] is False


def test_two_seed_pipeline_blocks_pair_tail_after_root_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
        screening=ScreeningConfig(practical_delta_by_k={2: 0.03}, delta_across_k=0.03),
    )
    _install_root_results(monkeypatch, tmp_path, failed_root=22)
    pair_calls: list[object] = []
    health: dict[str, Any] = {}
    monkeypatch.setattr(
        two_seed_pipeline.analysis,
        "run_root_pair_analysis",
        lambda context, **_: pair_calls.append(context),
    )
    monkeypatch.setattr(
        two_seed_pipeline,
        "_write_pipeline_health",
        lambda _path, payload: health.update(payload),
    )

    with pytest.raises(RuntimeError, match="root analysis failed"):
        two_seed_pipeline.run_pipeline(cfg, seed_pair=(11, 22))

    assert pair_calls == []
    assert health["status"] == "failed"
    assert health["pair_workflow"]["status"] == "failed"


def test_sequential_non_resource_failure_prevents_second_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
        screening=ScreeningConfig(practical_delta_by_k={2: 0.03}, delta_across_k=0.03),
    )
    _install_root_results(
        monkeypatch, tmp_path, failed_root=11, failure_classification="non_resource_failure"
    )
    health: dict[str, Any] = {}
    monkeypatch.setattr(
        two_seed_pipeline, "_write_pipeline_health", lambda _path, payload: health.update(payload)
    )

    with pytest.raises(RuntimeError, match="root analysis failed"):
        two_seed_pipeline.run_pipeline(cfg, seed_pair=(11, 22))

    assert health["root_workflows"]["22"]["analysis"] == "not_started"
    assert health["root_workflows"]["22"]["failure_classification"] == "not_started_fail_fast"


def test_sequential_resource_failure_prevents_second_root_and_pair_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
        screening=ScreeningConfig(practical_delta_by_k={2: 0.03}, delta_across_k=0.03),
    )
    _install_root_results(
        monkeypatch, tmp_path, failed_root=11, failure_classification="persistent_high_water"
    )
    pair_calls: list[object] = []
    health: dict[str, Any] = {}
    monkeypatch.setattr(
        two_seed_pipeline.analysis,
        "run_root_pair_analysis",
        lambda context, **_: pair_calls.append(context),
    )
    monkeypatch.setattr(
        two_seed_pipeline, "_write_pipeline_health", lambda _path, payload: health.update(payload)
    )

    with pytest.raises(two_seed_pipeline.ResourceFailureError) as raised:
        two_seed_pipeline.run_pipeline(cfg, seed_pair=(11, 22))

    assert raised.value.classification == "persistent_high_water"
    assert health["root_workflows"]["22"]["analysis"] == "not_started"
    assert pair_calls == []


def test_hard_supervisor_exit_cannot_leave_successful_health(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
        screening=ScreeningConfig(practical_delta_by_k={2: 0.03}, delta_across_k=0.03),
    )
    _install_root_results(monkeypatch, tmp_path)
    monkeypatch.setattr(
        two_seed_pipeline,
        "_run_one_seed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit(86)),
    )

    with pytest.raises(SystemExit, match="86"):
        two_seed_pipeline.run_pipeline(cfg, seed_pair=(11, 22))

    health_path = two_seed_pipeline.seed_pair_root(cfg, (11, 22)) / "pipeline_health.json"
    health = json.loads(health_path.read_text(encoding="utf-8"))
    assert health["status"] == "running"
    assert health["status"] != "complete_success"


def test_final_audit_interruption_cannot_publish_success_or_run_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
        screening=ScreeningConfig(practical_delta_by_k={2: 0.03}, delta_across_k=0.03),
    )
    _install_root_results(monkeypatch, tmp_path)
    monkeypatch.setattr(
        two_seed_pipeline.analysis,
        "run_root_pair_analysis",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        two_seed_pipeline,
        "_final_release_gate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            InterruptedError("final byte-deep audit interrupted")
        ),
    )
    health_writes: list[dict[str, Any]] = []
    manifest_events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        two_seed_pipeline,
        "_write_pipeline_health",
        lambda _path, payload: health_writes.append(dict(payload)),
    )
    monkeypatch.setattr(
        two_seed_pipeline,
        "append_manifest_event",
        lambda _path, event, **_kwargs: manifest_events.append(dict(event)),
    )

    with pytest.raises(InterruptedError, match="audit interrupted"):
        two_seed_pipeline.run_pipeline(cfg, seed_pair=(11, 22))

    assert [item["status"] for item in health_writes] == ["running"]
    assert not any(item.get("event") == two_seed_pipeline.EVENT_RUN_END for item in manifest_events)


def test_failed_root_precondition_executes_final_audit_zero_times(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
        screening=ScreeningConfig(practical_delta_by_k={2: 0.03}, delta_across_k=0.03),
    )
    _install_root_results(monkeypatch, tmp_path, failed_root=11)
    audit_calls: list[object] = []
    monkeypatch.setattr(
        two_seed_pipeline,
        "_final_release_gate",
        lambda *_args, **_kwargs: audit_calls.append(object()),
    )

    with pytest.raises(RuntimeError, match="root analysis failed"):
        two_seed_pipeline.run_pipeline(cfg, seed_pair=(11, 22))

    assert audit_calls == []


def test_pipeline_health_cannot_report_success_over_stale_pair_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
        screening=ScreeningConfig(practical_delta_by_k={2: 0.03}, delta_across_k=0.03),
    )
    _install_root_results(monkeypatch, tmp_path)
    health: dict[str, Any] = {}
    monkeypatch.setattr(
        two_seed_pipeline.analysis,
        "run_root_pair_analysis",
        lambda *_args, **_kwargs: None,
    )

    monkeypatch.setattr(
        two_seed_pipeline,
        "_build_pair_authenticated_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("pair workflow contains stale or incomplete canonical stages")
        ),
    )
    monkeypatch.setattr(
        two_seed_pipeline,
        "_write_pipeline_health",
        lambda _path, payload: health.update(payload),
    )

    with pytest.raises(RuntimeError, match="stale or incomplete"):
        two_seed_pipeline.run_pipeline(cfg, seed_pair=(11, 22))

    assert health["status"] == "failed"
    assert "stale or incomplete" in health["pair_workflow"]["error"]


def test_pipeline_health_rejects_explicitly_invalidated_root_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
        screening=ScreeningConfig(practical_delta_by_k={2: 0.03}, delta_across_k=0.03),
    )
    _install_root_results(monkeypatch, tmp_path, rejected_snapshot_root=22)
    health: dict[str, Any] = {}
    monkeypatch.setattr(
        two_seed_pipeline.analysis,
        "run_root_pair_analysis",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        two_seed_pipeline,
        "_write_pipeline_health",
        lambda _path, payload: health.update(payload),
    )

    with pytest.raises(RuntimeError, match="snapshot rejected"):
        two_seed_pipeline.run_pipeline(cfg, seed_pair=(11, 22))

    assert health["status"] == "failed"
    assert health["root_workflows"]["11"]["analysis"] == "complete"
    assert health["root_workflows"]["22"]["analysis"] == "failed"
    assert "explicitly invalidated" in health["root_workflows"]["22"]["error"]


def test_worker_budget_is_split_across_concurrent_roots(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    cfg.sim.n_jobs = 8
    cfg.analysis.n_jobs = 6
    cfg.orchestration.parallel_seeds = True

    policy = two_seed_pipeline._derive_per_seed_job_budgets(cfg, seed_count=2)

    assert policy.simulation.process_workers == 2
    assert policy.analysis.process_workers == 1
    assert policy.simulation.concurrent_roots == 2
    assert policy.analysis.concurrent_roots == 2


def test_worker_sections_own_independent_explicit_budgets() -> None:
    cfg = AppConfig(
        sim=SimConfig(n_jobs=12),
        analysis=AnalysisConfig(n_jobs=4),
    )

    policy = two_seed_pipeline._derive_per_seed_job_budgets(cfg, seed_count=2)

    assert policy.simulation.process_workers == 4
    assert policy.analysis.process_workers == 3
    assert policy.as_metadata()["simulation"]["requested_n_jobs"] == 12
    assert policy.as_metadata()["simulation"]["resolved_n_jobs"] == 12
    assert policy.as_metadata()["simulation"]["effective_n_jobs"] == 4
    assert policy.as_metadata()["analysis"]["requested_n_jobs"] == 4
    assert policy.as_metadata()["analysis"]["resolved_n_jobs"] == 4
    assert policy.as_metadata()["analysis"]["effective_n_jobs"] == 3
    assert policy.as_metadata()["head2head"]["effective_n_jobs"] == 4


def test_worker_defaults_and_explicit_auto_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AppConfig()
    defaults = two_seed_pipeline._derive_per_seed_job_budgets(cfg, seed_count=2)
    assert defaults.simulation.process_workers == 1
    assert defaults.analysis.process_workers == 1

    original = two_seed_pipeline.normalize_n_jobs
    monkeypatch.setattr(
        two_seed_pipeline,
        "normalize_n_jobs",
        lambda value, default=1: 16 if value == 0 else original(value, default=default),
    )
    cfg.sim.n_jobs = 0
    cfg.analysis.n_jobs = 0
    auto = two_seed_pipeline._derive_per_seed_job_budgets(cfg, seed_count=2)
    assert auto.resolved_n_jobs["simulation"] == 16
    assert auto.resolved_n_jobs["analysis"] == 16
    assert auto.simulation.process_workers == 4
    assert auto.analysis.process_workers == 3


def test_h2h_stage_uses_head2head_worker_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AppConfig(
        sim=SimConfig(seed=1, seed_list=[1]),
        analysis=AnalysisConfig(n_jobs=3),
    )
    cfg.head2head.n_jobs = 7
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "farkle.analysis.h2h_schedule.execute_h2h_schedule",
        lambda _cfg, *, n_jobs, oracle_game_profile: captured.update(n_jobs=n_jobs),
    )
    plan = two_seed_pipeline.analysis.build_single_root_h2h_tail_plan(cfg)
    next(item for item in plan if item.name == "h2h_execute").action(cfg)

    assert captured["n_jobs"] == 7


def test_yaml_then_cli_worker_override_precedence(tmp_path: Path) -> None:
    config_path = tmp_path / "workers.yaml"
    config_path.write_text("sim:\n  n_jobs: 7\nanalysis:\n  n_jobs: 3\n", encoding="utf-8")

    cfg = load_app_config(config_path)
    cfg = apply_dot_overrides(cfg, ["sim.n_jobs=9", "analysis.n_jobs=5"])
    policy = two_seed_pipeline._derive_per_seed_job_budgets(cfg, seed_count=2)

    assert policy.resolved_n_jobs["simulation"] == 9
    assert policy.resolved_n_jobs["analysis"] == 5
    assert policy.simulation.process_workers == 4
    assert policy.analysis.process_workers == 3


def test_file_capacity_projects_shards_blocks_sidecars_and_games(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = AppConfig(
        sim=SimConfig(n_players_list=[2], row_dir=Path("rows")),
        screening=ScreeningConfig(
            practical_delta_by_k={2: 0.03},
            candidate_contribution_size=75,
        ),
    )
    monkeypatch.setattr(
        two_seed_pipeline.runner,
        "_resolve_strategies",
        lambda _cfg, _strategies: ([], 10, True),
    )

    projection = two_seed_pipeline._project_file_capacity(cfg, root_count=2)

    assert projection["required_games_all_roots"] == 43_000
    assert projection["tournament_row_shards"] == 8_600
    assert projection["h2h_candidate_count_upper_envelope"] == 10
    assert projection["h2h_coordinate_blocks_upper_envelope"] == 180
    assert projection["projected_sidecars"] == 8_780
    assert projection["projected_total_files"] == 17_560
    assert projection["operational_capacity_only"] is True
    assert projection["warning_threshold"] is None
    assert projection["workload_by_k"] == [
        {
            "k": 2,
            "required_games_per_root": 21_500,
            "required_shuffles_per_root": 4_300,
            "row_shards_per_root": 4_300,
            "cap_exceeded": False,
            "cap_state": "not_started",
        }
    ]


def test_force_bypasses_authenticated_simulation_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
    )
    policy = two_seed_pipeline._derive_per_seed_job_budgets(cfg, seed_count=2)
    calls: list[bool] = []
    monkeypatch.setattr(two_seed_pipeline, "write_run_context_atomic", lambda *_a, **_k: "x")
    monkeypatch.setattr(two_seed_pipeline, "write_active_config", lambda *_a, **_k: None)
    monkeypatch.setattr(two_seed_pipeline, "seed_has_completion_markers", lambda _cfg: True)
    monkeypatch.setattr(
        two_seed_pipeline.runner,
        "run_tournament",
        lambda _cfg, *, force=False: calls.append(force),
    )
    monkeypatch.setattr(two_seed_pipeline, "_run_per_seed_analysis", lambda *_a, **_k: None)
    monkeypatch.setattr(
        two_seed_pipeline,
        "_root_lifecycle_identity",
        lambda *_a, **_k: ("f" * 64, {"simulation_2p": "complete_valid"}),
    )
    code = CodeIdentity(
        commit="a" * 40,
        policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
        state="clean",
        dirty_fingerprint_sha256=None,
    )

    two_seed_pipeline._run_one_seed(
        cfg,
        seed=11,
        seed_pair=(11, 22),
        manifest_path=tmp_path / "manifest.jsonl",
        run_id="run",
        force=False,
        policy_bundle=policy,
        code_identity=code,
    )
    two_seed_pipeline._run_one_seed(
        cfg,
        seed=11,
        seed_pair=(11, 22),
        manifest_path=tmp_path / "manifest.jsonl",
        run_id="run",
        force=True,
        policy_bundle=policy,
        code_identity=code,
    )

    assert calls == [True]
