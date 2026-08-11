from dataclasses import replace
from pathlib import Path

import pytest

from farkle.analysis.stage_registry import resolve_root_pair_stage_layout, resolve_stage_layout
from farkle.config import AppConfig, ResourcesConfig, compute_config_sha, load_app_config


def test_resource_defaults_use_distinct_explicit_memory_controls() -> None:
    resources = ResourcesConfig()
    assert resources.scheduler_memory_budget_mb == 768
    assert resources.process_tree_warning_threshold_mb == 768
    assert resources.aggregate_memory_hard_limit_mb == 2304
    assert resources.minimum_system_available_memory_mb == 1024
    assert resources.parent_process_memory_mb == 192
    assert resources.os_memory_limit_enabled is True
    assert resources.os_memory_limit_required is True
    assert resources.allow_unenforced_memory_fallback is False
    assert resources.native_threads_per_worker == 1
    assert all(value > 0 for value in resources.estimated_worker_memory_mb.values())
    assert all(value > 0 for value in resources.stage_batch_bytes.values())


def test_resource_controls_are_execution_provenance_not_statistical_freshness() -> None:
    baseline = AppConfig()
    changed = replace(
        baseline,
        resources=replace(
            baseline.resources,
            scheduler_memory_budget_mb=2048,
            process_tree_warning_threshold_mb=3072,
            aggregate_memory_hard_limit_mb=4096,
            minimum_system_available_memory_mb=2048,
            parent_process_memory_mb=256,
            logical_cpu_budget=2,
            native_threads_per_worker=2,
            stage_batch_bytes={**baseline.resources.stage_batch_bytes, "trueskill": 1024},
        ),
    )
    stage_keys = {
        "simulation",
        *resolve_stage_layout(baseline).keys(),
        *resolve_root_pair_stage_layout(baseline).keys(),
    }
    assert compute_config_sha(baseline) == compute_config_sha(changed)
    assert all(
        baseline.stage_config_sha(stage) == changed.stage_config_sha(stage) for stage in stage_keys
    )


def test_legacy_worker_counts_are_execution_only_for_all_hashes() -> None:
    baseline = AppConfig()
    changed = AppConfig()
    changed.sim.n_jobs = 7
    changed.ingest.n_jobs = 7
    changed.analysis.n_jobs = 7
    changed.head2head.n_jobs = 7
    stage_keys = {
        "simulation",
        *resolve_stage_layout(baseline).keys(),
        *resolve_root_pair_stage_layout(baseline).keys(),
    }
    assert compute_config_sha(baseline) == compute_config_sha(changed)
    assert all(
        baseline.stage_config_sha(stage) == changed.stage_config_sha(stage) for stage in stage_keys
    )


@pytest.mark.parametrize(
    "yaml_text, message",
    [
        (
            "resources:\n  scheduler_memory_budget_mb: 768\n"
            "  process_tree_warning_threshold_mb: 767\n",
            "parent_process_memory_mb",
        ),
        (
            "resources:\n  process_tree_warning_threshold_mb: 2304\n"
            "  aggregate_memory_hard_limit_mb: 2304\n",
            "aggregate_memory_hard_limit_mb",
        ),
        (
            "resources:\n  scheduler_memory_budget_mb: 192\n" "  parent_process_memory_mb: 192\n",
            "parent_process_memory_mb",
        ),
        (
            "resources:\n  minimum_system_available_memory_mb: 0\n",
            "minimum_system_available_memory_mb",
        ),
        ("resources:\n  native_threads_per_worker: 0\n", "native_threads_per_worker"),
        (
            "resources:\n  estimated_worker_memory_mb: {analysis: 0}\n",
            "estimated_worker_memory_mb",
        ),
        ("resources:\n  os_memory_limit_enabled: false\n", "required OS memory"),
        ("resources:\n  os_memory_limit_required: false\n", "development fallback"),
        (
            "resources:\n  os_memory_limit_required: true\n"
            "  allow_unenforced_memory_fallback: true\n",
            "strict OS memory",
        ),
    ],
)
def test_invalid_resource_envelopes_are_rejected(
    tmp_path: Path, yaml_text: str, message: str
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=message):
        load_app_config(path)


@pytest.mark.parametrize(
    "legacy_key,replacement",
    [
        ("target_memory_mb", "scheduler_memory_budget_mb"),
        ("memory_safety_factor", "aggregate_memory_hard_limit_mb"),
        ("parent_reserve_mb", "parent_process_memory_mb"),
        ("logical_cpu_workers", "logical_cpu_budget"),
    ],
)
def test_retired_resource_keys_fail_with_precise_replacement(
    tmp_path: Path, legacy_key: str, replacement: str
) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(f"resources:\n  {legacy_key}: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=replacement):
        load_app_config(path)
