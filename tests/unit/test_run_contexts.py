"""Tests for root and root-pair run-context path wiring."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import (
    AppConfig,
    IOConfig,
    ProfileConfig,
    SimConfig,
    assign_config_sha,
    compute_config_sha,
)
from farkle.orchestration import run_contexts
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
    assert payload["profile"]["release_eligible"] is True
    assert payload["profile"]["workload_by_k"]["2"]["achieved_resolution"] <= 0.03


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
    context.config.profile = ProfileConfig(
        purpose="integration",
        reduced_resolution=True,
        production_eligible=False,
        release_eligible=False,
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
    integration_payload = load_run_context(context.run_context_path)
    statistical_identity = compute_config_sha(context.config)

    assert integration_payload["profile"]["purpose"] == "integration"
    assert integration_payload["profile"]["reduced_resolution"] is True
    assert integration_payload["profile"]["production_eligible"] is False
    assert integration_payload["profile"]["release_eligible"] is False

    context.config.profile = replace(context.config.profile, purpose="production")
    assert compute_config_sha(context.config) == statistical_identity
    assert integration_payload["public_config_sha256"] == statistical_identity
    assert context.config.profile.production_eligible is False
    assert context.config.profile.release_eligible is False
    from farkle.orchestration.seed_utils import write_active_config

    write_active_config(context.config)
    with pytest.raises(ValueError, match="run profile"):
        load_run_context(
            context.run_context_path,
            active_config_path=context.active_config_path,
        )

    with pytest.raises(ValueError, match="production profiles must be full-resolution"):
        context.config.validate_statistical_contract()
