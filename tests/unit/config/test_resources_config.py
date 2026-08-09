from dataclasses import replace
from pathlib import Path

import pytest

from farkle.config import AppConfig, ResourcesConfig, compute_config_sha, load_app_config


def test_resource_defaults_hold_one_gib_process_tree_envelope() -> None:
    resources = ResourcesConfig()
    assert resources.max_memory_mb == 1024
    assert resources.target_memory_mb == 768
    assert resources.target_memory_mb < resources.rss_abort_mb < resources.max_memory_mb
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
            logical_cpu_workers=2,
            native_threads_per_worker=2,
            stage_batch_bytes={**baseline.resources.stage_batch_bytes, "trueskill": 1024},
        ),
    )
    assert compute_config_sha(baseline) == compute_config_sha(changed)
    assert baseline.stage_config_sha("trueskill") == changed.stage_config_sha("trueskill")


@pytest.mark.parametrize(
    "yaml_text",
    [
        "resources:\n  max_memory_mb: 2048\n",
        "resources:\n  target_memory_mb: 960\n  rss_abort_mb: 950\n",
        "resources:\n  native_threads_per_worker: 0\n",
        "resources:\n  estimated_worker_memory_mb: {analysis: 0}\n",
        "resources:\n  os_memory_limit_enabled: false\n",
        "resources:\n  os_memory_limit_required: false\n",
        (
            "resources:\n  os_memory_limit_required: true\n"
            "  allow_unenforced_memory_fallback: true\n"
        ),
    ],
)
def test_invalid_resource_envelopes_are_rejected(tmp_path: Path, yaml_text: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        load_app_config(path)
