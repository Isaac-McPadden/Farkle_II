"""Tests for root and root-pair run-context path wiring."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from tests.helpers.artifact_sidecars import clean_test_code_identity

from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import (
    AppConfig,
    IOConfig,
    ProfileConfig,
    SimConfig,
    assign_config_sha,
    compute_config_sha,
)
from farkle.orchestration import execution_history, run_contexts
from farkle.orchestration.run_contexts import (
    SEED_PAIR_ANALYSIS_DIRNAME,
    RootPairRunContext,
    RunContextConfig,
    SeedRunContext,
    load_run_context,
    write_run_context_atomic,
)
from farkle.utils.authenticated_contract import CodeIdentity, CodeIdentityPolicy


def _root_context(tmp_path: Path, root: int) -> SeedRunContext:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / f"root_{root}"),
        sim=SimConfig(seed=root, seed_list=[root], n_players_list=[2, 4]),
    )
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    return SeedRunContext.from_config(cfg)


def test_root_pair_context_uses_pair_analysis_root_and_both_roots(tmp_path: Path) -> None:
    first = _root_context(tmp_path, 11)
    second = _root_context(tmp_path, 22)
    pair_root = tmp_path / "pair"

    context = RootPairRunContext.from_root_contexts((first, second), pair_root=pair_root)

    assert context.root_pair == (11, 22)
    assert context.analysis_root == pair_root / SEED_PAIR_ANALYSIS_DIRNAME
    assert context.config.analysis_dir == context.analysis_root
    assert context.config.sim.seed_list == [11, 22]
    assert context.config.stage_layout.keys()[0] == "root_stability"
    assert not context.analysis_root.exists()


def test_invocation_history_preserves_legacy_pair_and_repeated_resumes(tmp_path: Path) -> None:
    from farkle.orchestration.seed_utils import write_active_config

    roots = (_root_context(tmp_path, 11), _root_context(tmp_path, 22))
    context = RootPairRunContext.from_root_contexts(roots, pair_root=tmp_path / "pair")
    code = clean_test_code_identity()
    parents = ("a" * 64, "b" * 64)
    workers = {"analysis": {"effective_n_jobs": 1}}
    write_run_context_atomic(
        context, code_identity=code, parent_lifecycle_roots=parents, worker_counts=workers
    )
    write_active_config(context.config, context.pair_root)
    original_context = context.run_context_path.read_bytes()
    original_config = context.active_config_path.read_bytes()
    stamp = context.pair_root / "partial.stamp"
    stamp.write_bytes(b"previous partial stage")

    def resume(run: str, lifecycle: tuple[str, ...]) -> Path:
        execution_history.publish_invocation(
            context,
            manifest_path=tmp_path / "manifest.jsonl",
            run_id=run,
            code_identity=code,
            parent_lifecycle_roots=lifecycle,
            completion_paths=[("partial", stamp)],
            worker_counts=workers,
        )
        return execution_history.invocation_dir(context.pair_root, run)

    first = resume("first", parents)
    second = resume("second", parents)
    context.config.resources.logical_cpu_budget = 2
    third = resume("third", parents)
    fourth = resume("fourth", tuple(reversed(parents)))
    for folder in (first / "previous", first, second, third / "previous"):
        assert (folder / "run_context.json").read_bytes() == original_context
        assert (folder / "active_config.yaml").read_bytes() == original_config
    assert (first / "previous" / "0.stamp").read_bytes() == b"previous partial stage"
    resource_change = load_run_context(
        third / "run_context.json", active_config_path=third / "active_config.yaml"
    )
    original = json.loads(original_context)
    assert resource_change["run_lineage_sha256"] == original["run_lineage_sha256"]
    assert resource_change["run_context_sha256"] != original["run_context_sha256"]
    latest = load_run_context(
        context.run_context_path, active_config_path=context.active_config_path
    )
    assert latest["parent_lifecycle_roots"] == list(reversed(parents))
    assert latest["run_lineage_sha256"] != original["run_lineage_sha256"]
    assert latest["code_identity"] == original["code_identity"]
    assert (fourth / "run_context.json").read_bytes() == context.run_context_path.read_bytes()


@pytest.mark.parametrize("conflict", [False, True])
def test_interrupted_invocation_reconciles_only_recorded_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    conflict: bool,
) -> None:
    context = _root_context(tmp_path, 11)
    root = context.results_root

    def resume(run: str) -> None:
        execution_history.publish_invocation(
            context,
            run_id=run,
            completion_paths=[],
            manifest_path=tmp_path / "manifest.jsonl",
            code_identity=clean_test_code_identity(),
            worker_counts={},
        )

    resume("original")
    original = context.run_context_path.read_bytes()
    context.config.resources.logical_cpu_budget = 3
    live_config = context.active_config_path
    atomic = execution_history.atomic_path

    def interrupt(path: str):
        if Path(path) == live_config:
            raise InterruptedError("between context and configuration")
        return atomic(path)

    with monkeypatch.context() as patch:
        patch.setattr(execution_history, "atomic_path", interrupt)
        with pytest.raises(InterruptedError):
            resume("interrupted")
    assert original != context.run_context_path.read_bytes()
    assert execution_history.resume_context_path(root).is_file()
    if conflict:
        live_config.write_bytes(b"unexplained modification")
        with pytest.raises(ValueError, match="conflicting evidence"):
            resume("retry")
        assert live_config.read_bytes() == b"unexplained modification"
        assert not execution_history.invocation_dir(root, "retry").exists()
    else:
        resume("retry")
        assert execution_history.resume_context_path(root) == context.run_context_path
        interrupted = execution_history.invocation_dir(root, "interrupted")
        assert (interrupted / "previous" / "run_context.json").read_bytes() == original
        assert (
            interrupted / "run_context.json"
        ).read_bytes() == context.run_context_path.read_bytes()
        load_run_context(context.run_context_path, active_config_path=live_config)


def test_root_pair_context_maps_first_root_inputs_without_changing_outputs(tmp_path: Path) -> None:
    first = _root_context(tmp_path, 7)
    second = _root_context(tmp_path, 8)
    context = RootPairRunContext.from_root_contexts(
        (first, second),
        pair_root=tmp_path / "pair",
    )

    combine_folder = first.config.stage_layout.require_folder("combine")
    assert context.config.root_input_stage_folder("combine") == combine_folder
    assert context.config.cross_seed_dir("root_stability").is_relative_to(context.analysis_root)


def test_root_pair_context_rejects_duplicate_roots(tmp_path: Path) -> None:
    first = _root_context(tmp_path, 7)

    with pytest.raises(ValueError, match="distinct roots"):
        RootPairRunContext.from_root_contexts((first, first), pair_root=tmp_path / "pair")


def test_run_context_config_preserves_all_typed_settings(tmp_path: Path) -> None:
    base = _root_context(tmp_path, 11).config
    base = replace(base, screening=replace(base.screening, resolution_delta=0.021))

    run_cfg = RunContextConfig.from_base(base, analysis_root=tmp_path / "pair" / "analysis")

    assert run_cfg.screening.resolution_delta == pytest.approx(0.021)
    assert run_cfg.analysis_dir == tmp_path / "pair" / "analysis"


def test_run_context_config_analysis_dir_falls_back_to_base(tmp_path: Path) -> None:
    base = _root_context(tmp_path, 11).config

    assert RunContextConfig.from_base(base).analysis_dir == base.analysis_dir


def test_authenticated_context_records_requested_resolved_and_effective_workers(
    tmp_path: Path,
) -> None:
    context = _root_context(tmp_path, 11)
    worker_counts = {
        "simulation": {
            "requested_n_jobs": 12,
            "resolved_n_jobs": 12,
            "effective_n_jobs": 6,
        },
        "ingest": {
            "requested_n_jobs": 3,
            "resolved_n_jobs": 3,
            "effective_n_jobs": 1,
        },
        "analysis": {
            "requested_n_jobs": 4,
            "resolved_n_jobs": 4,
            "effective_n_jobs": 2,
        },
        "head2head": {
            "requested_n_jobs": 0,
            "resolved_n_jobs": 16,
            "effective_n_jobs": 16,
        },
    }
    write_run_context_atomic(
        context,
        code_identity=CodeIdentity(
            commit="a" * 40,
            policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
            state="clean",
            dirty_fingerprint_sha256=None,
        ),
        cli_overrides=("analysis.n_jobs=4",),
        worker_counts=worker_counts,
    )

    payload = load_run_context(context.run_context_path)

    assert payload["run_context_contract_version"] == 4
    assert payload["execution_controls"]["resource_limiter_mode"] == "enabled"
    assert payload["execution_controls"]["worker_counts"] == worker_counts
    assert payload["execution_controls"]["resources"]["scheduler_memory_budget_mb"] == 768
    assert payload["execution_controls"]["resources"]["aggregate_memory_hard_limit_mb"] == 2304
    assert payload["execution_controls"]["resource_policy"]["requested"] == {
        "aggregate_memory_hard_limit_mb": 2304,
        "logical_cpu_budget": 0,
        "minimum_system_available_memory_mb": 1024,
        "native_threads_per_worker": 1,
        "parent_process_memory_mb": 192,
        "process_tree_warning_threshold_mb": 768,
        "scheduler_memory_budget_mb": 768,
    }
    assert payload["execution_controls"]["os_memory_boundary"]["backend"] in {
        "none",
        "windows_job",
        "cgroup_v2",
        "unenforced",
    }
    assert "effective_hard_limit_mb" in payload["execution_controls"]["os_memory_boundary"]
    assert payload["execution_controls"]["os_memory_boundary"]["requested_hard_limit_mb"] == 2304
    assert payload["cli_overrides"] == ["analysis.n_jobs=4"]
    assert payload["profile"]["purpose"] == "production"
    assert payload["profile"]["production_eligible"] is True
    assert "release_eligible" not in payload["profile"]
    assert payload["profile"]["workload_by_k"]["2"]["achieved_resolution"] <= 0.03


def test_limiter_mode_and_scratch_republish_runtime_provenance_without_hash_change(
    tmp_path: Path,
) -> None:
    context = _root_context(tmp_path, 11)
    code_identity = CodeIdentity(
        commit="a" * 40,
        policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
        state="clean",
        dirty_fingerprint_sha256=None,
    )
    statistical_hash = compute_config_sha(context.config)
    write_run_context_atomic(context, code_identity=code_identity)
    original = load_run_context(context.run_context_path)

    scratch = (tmp_path / "scratch").resolve()
    context.config.io.temp_dir = scratch
    context.config.set_resource_limiter_mode("disabled")
    write_run_context_atomic(context, code_identity=code_identity)
    changed = load_run_context(context.run_context_path)

    assert compute_config_sha(context.config) == statistical_hash
    assert changed["run_lineage_sha256"] == original["run_lineage_sha256"]
    assert changed["run_context_sha256"] != original["run_context_sha256"]
    assert changed["resolved_paths"]["temp_dir"] == str(scratch)
    assert changed["execution_controls"]["resource_limiter_mode"] == "disabled"
    assert changed["execution_controls"]["resource_policy"]["effective"] == {
        "aggregate_memory_hard_limit_mb": None,
        "logical_cpu_budget": None,
        "minimum_system_available_memory_mb": None,
        "native_threads_per_worker": None,
        "parent_process_memory_mb": None,
        "process_tree_warning_threshold_mb": None,
        "scheduler_memory_budget_mb": None,
    }
    assert list(scratch.iterdir()) == []


def test_run_context_binds_public_scratch_path(tmp_path: Path) -> None:
    from farkle.orchestration.seed_utils import write_active_config

    context = _root_context(tmp_path, 11)
    context.config.io.temp_dir = (tmp_path / "scratch-a").resolve()
    write_active_config(context.config)
    write_run_context_atomic(
        context,
        code_identity=CodeIdentity(
            commit="a" * 40,
            policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
            state="clean",
            dirty_fingerprint_sha256=None,
        ),
    )
    context.config.io.temp_dir = (tmp_path / "scratch-b").resolve()
    write_active_config(context.config)

    with pytest.raises(ValueError, match="scratch path"):
        load_run_context(
            context.run_context_path,
            active_config_path=context.active_config_path,
        )


def test_exact_resume_preserves_run_context_across_host_memory_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _root_context(tmp_path, 11)
    code_identity = CodeIdentity(
        commit="a" * 40,
        policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
        state="clean",
        dirty_fingerprint_sha256=None,
    )
    worker_counts = {"analysis": {"effective_n_jobs": 1}}
    samples = iter(
        (
            {"backend": "none", "effective_hard_limit_mb": None, "available_mb": 8000},
            {"backend": "none", "effective_hard_limit_mb": None, "available_mb": 7000},
        )
    )
    monkeypatch.setattr(run_contexts, "memory_boundary_provenance", lambda _cfg: next(samples))

    write_run_context_atomic(
        context,
        code_identity=code_identity,
        worker_counts=worker_counts,
    )
    original = context.run_context_path.read_bytes()
    write_run_context_atomic(
        context,
        code_identity=code_identity,
        worker_counts=worker_counts,
    )

    assert context.run_context_path.read_bytes() == original


def test_run_context_authenticates_execution_resource_controls(tmp_path: Path) -> None:
    context = _root_context(tmp_path, 11)
    assign_config_sha(context.config)
    write_run_context_atomic(
        context,
        code_identity=CodeIdentity(
            commit="a" * 40,
            policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
            state="clean",
            dirty_fingerprint_sha256=None,
        ),
    )
    original_payload = load_run_context(context.run_context_path)
    original = original_payload["run_context_sha256"]
    context.config.resources.logical_cpu_budget = 2
    from farkle.orchestration.seed_utils import write_active_config

    write_active_config(context.config)
    with pytest.raises(ValueError, match="execution resource controls"):
        load_run_context(
            context.run_context_path,
            active_config_path=context.active_config_path,
        )

    write_run_context_atomic(
        context,
        code_identity=CodeIdentity(
            commit="a" * 40,
            policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
            state="clean",
            dirty_fingerprint_sha256=None,
        ),
    )
    changed_payload = load_run_context(context.run_context_path)
    assert changed_payload["run_context_sha256"] != original
    assert changed_payload["public_config_sha256"] == original_payload["public_config_sha256"]
    assert changed_payload["resource_config_sha256"] != original_payload["resource_config_sha256"]


def test_purpose_only_change_preserves_compute_identity_but_invalidates_claim_context(
    tmp_path: Path,
) -> None:
    context = _root_context(tmp_path, 11)
    context.config.batching = replace(context.config.batching, target_batches=20)
    context.config.profile = ProfileConfig(purpose="integration")
    write_run_context_atomic(
        context,
        code_identity=CodeIdentity(
            commit="a" * 40,
            policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
            state="clean",
            dirty_fingerprint_sha256=None,
        ),
    )
    integration_payload = load_run_context(context.run_context_path)
    statistical_identity = compute_config_sha(context.config)

    assert integration_payload["profile"]["purpose"] == "integration"
    assert integration_payload["profile"]["reduced_resolution"] is False
    assert integration_payload["profile"]["production_eligible"] is False
    assert "release_eligible" not in integration_payload["profile"]

    context.config.profile = replace(context.config.profile, purpose="production")
    assert compute_config_sha(context.config) == statistical_identity
    assert integration_payload["public_config_sha256"] == statistical_identity
    from farkle.orchestration.profile_metadata import resolved_profile_metadata

    assert resolved_profile_metadata(context.config)["production_eligible"] is False
    from farkle.orchestration.seed_utils import write_active_config

    write_active_config(context.config)
    with pytest.raises(ValueError, match="run profile"):
        load_run_context(
            context.run_context_path,
            active_config_path=context.active_config_path,
        )

    context.config.screening.practical_delta_by_k = {2: 0.03, 4: 0.03}
    context.config.validate_statistical_contract()
