"""Targeted tests for aggregate operating-system memory enforcement."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from farkle.utils import os_memory

PROBE = Path(__file__).resolve().parents[2] / "helpers" / "os_memory_probe.py"


def _supervisor(limit_mb: int, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "farkle.utils.os_memory",
            "--limit-mb",
            str(limit_mb),
            "--",
            sys.executable,
            str(PROBE),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _resources(*, required: bool, fallback: bool) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_memory_budget_mb=64,
        process_tree_warning_threshold_mb=96,
        aggregate_memory_hard_limit_mb=128,
        minimum_system_available_memory_mb=1,
        parent_process_memory_mb=16,
        logical_cpu_budget=0,
        native_threads_per_worker=1,
        os_memory_limit_enabled=True,
        os_memory_limit_required=required,
        allow_unenforced_memory_fallback=fallback,
    )


def test_supervisor_seam_uses_explicit_hard_limit_without_recomputation() -> None:
    resources = _resources(required=True, fallback=False)
    resources.scheduler_memory_budget_mb = 17
    resources.process_tree_warning_threshold_mb = 31

    assert os_memory._hard_memory_limit_mb(resources) == 128
    provenance = os_memory.memory_boundary_provenance(resources)
    assert provenance["requested_hard_limit_mb"] == 128
    assert provenance["scheduler_memory_budget_mb"] == 17
    assert provenance["process_tree_warning_threshold_mb"] == 31


@pytest.mark.parametrize(
    ("platform", "backend_name"),
    [("win32", "_run_windows_job"), ("linux", "_run_cgroup_v2")],
)
def test_platform_supervisor_receives_explicit_hard_limit(
    monkeypatch: pytest.MonkeyPatch, platform: str, backend_name: str
) -> None:
    captured: dict[str, int] = {}

    def _backend(_command, resources, *, env):
        del env
        captured["hard_limit_mb"] = os_memory._hard_memory_limit_mb(resources)
        return 0

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(os_memory, backend_name, _backend)
    resources = _resources(required=True, fallback=False)
    resources.scheduler_memory_budget_mb = 17
    resources.process_tree_warning_threshold_mb = 31

    assert os_memory.supervise_process(["analysis"], resources) == 0
    assert captured == {"hard_limit_mb": 128}


@pytest.mark.skipif(sys.platform != "win32", reason="live Job Object test")
def test_aggregate_parent_child_and_grandchild_memory_is_constrained() -> None:
    completed = _supervisor(
        150,
        "tree",
        "--allocate-mb",
        "60",
        "--depth",
        "2",
        "--hold-seconds",
        "1",
    )

    assert completed.returncode == os_memory.MEMORY_LIMIT_EXIT_CODE
    assert "aggregate Job Object memory limit was reached" in completed.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="live Job Object test")
def test_two_concurrent_roots_share_one_aggregate_boundary() -> None:
    completed = _supervisor(
        150,
        "two-roots",
        "--parent-mb",
        "20",
        "--root-mb",
        "55",
        "--hold-seconds",
        "1",
    )

    assert completed.returncode == os_memory.MEMORY_LIMIT_EXIT_CODE


@pytest.mark.skipif(sys.platform != "win32", reason="live Job Object test")
def test_os_failure_preserves_completed_unit_but_not_final_publication(tmp_path: Path) -> None:
    completed = _supervisor(
        150,
        "publication",
        "--output-dir",
        str(tmp_path),
        "--allocate-mb",
        "60",
    )

    assert completed.returncode == os_memory.MEMORY_LIMIT_EXIT_CODE
    assert (tmp_path / "unit.complete").read_text(encoding="utf-8") == (
        "authenticated-complete-unit"
    )
    assert not (tmp_path / "pipeline.complete").exists()
    assert (tmp_path / "unit.partial.tmp").exists()
    assert not (tmp_path / "unit.partial").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="live Job Object test")
def test_outputs_are_identical_below_the_boundary() -> None:
    direct = subprocess.run(
        [sys.executable, str(PROBE), "identity", "--seed", "19"],
        check=True,
        capture_output=True,
        text=True,
    )
    protected = _supervisor(128, "identity", "--seed", "19")

    assert protected.returncode == 0
    assert protected.stdout.strip() == direct.stdout.strip()


@pytest.mark.skipif(sys.platform != "win32", reason="live Job Object test")
def test_windows_backend_records_job_configuration_and_effective_limit() -> None:
    completed = _supervisor(128, "status")

    assert completed.returncode == 0
    status = json.loads(completed.stdout)
    assert status["backend"] == "windows_job"
    assert status["enforced"] is True
    assert status["effective_hard_limit_mb"] <= 128
    assert status["detail"] == "aggregate JobMemoryLimit with kill-on-job-close"
    assert isinstance(status["enclosing_job"], bool)


@pytest.mark.skipif(sys.platform != "win32", reason="live Job Object test")
def test_partition_workers_inherit_the_aggregate_job_boundary() -> None:
    completed = _supervisor(512, "partition-workers", "--workers", "2")

    assert completed.returncode == 0
    statuses = json.loads(completed.stdout)
    assert len(statuses) == 2
    assert all(json.loads(status)["backend"] == "windows_job" for status in statuses)
    assert all(json.loads(status)["effective_hard_limit_mb"] <= 512 for status in statuses)


def test_strict_mode_fails_closed_before_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    launched = False

    def _unavailable(*_args, **_kwargs):
        raise os_memory.MemoryBoundaryError("synthetic backend denial")

    def _unexpected(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("analysis command must not launch")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os_memory, "_run_windows_job", _unavailable)
    monkeypatch.setattr(subprocess, "run", _unexpected)

    with pytest.raises(os_memory.MemoryBoundaryError, match="synthetic backend denial"):
        os_memory.supervise_process(["analysis"], _resources(required=True, fallback=False))
    assert not launched


def test_permissive_fallback_is_conspicuously_unenforced(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured_status: dict[str, object] = {}

    def _unavailable(*_args, **_kwargs):
        raise os_memory.MemoryBoundaryError("synthetic permission denial")

    def _run(*_args, **kwargs):
        captured_status.update(json.loads(kwargs["env"][os_memory.BOUNDARY_STATUS_ENV]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os_memory, "_run_windows_job", _unavailable)
    monkeypatch.setattr(subprocess, "run", _run)

    result = os_memory.supervise_process(
        ["analysis"],
        _resources(required=False, fallback=True),
    )

    assert result == 0
    assert captured_status["backend"] == "unenforced"
    assert captured_status["enforced"] is False
    assert captured_status["fallback_used"] is True
    assert "NOT ENFORCED" in capsys.readouterr().err


def test_enclosing_cgroup_limit_can_only_reduce_effective_limit(tmp_path: Path) -> None:
    mount = tmp_path / "cgroup"
    current = mount / "delegated" / "current"
    current.mkdir(parents=True)
    (mount / "memory.max").write_text("max", encoding="ascii")
    (mount / "delegated" / "memory.max").write_text(str(96 * 1024 * 1024), encoding="ascii")
    (current / "memory.max").write_text("max", encoding="ascii")

    assert os_memory._effective_cgroup_parent_limit(mount, current) == 96 * 1024 * 1024


def test_cgroup_permission_failure_is_reported_as_setup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    current = tmp_path / "current"
    current.mkdir()
    (current / "memory.max").write_text("max", encoding="ascii")
    monkeypatch.setattr(os_memory, "_cgroup_v2_location", lambda: (tmp_path, current))

    def _denied(*_args, **_kwargs):
        raise PermissionError("delegation denied")

    monkeypatch.setattr(Path, "mkdir", _denied)
    with pytest.raises(os_memory.MemoryBoundaryError, match="not delegated/writable"):
        os_memory._run_cgroup_v2(
            ["analysis"],
            _resources(required=True, fallback=False),
            env=None,
        )
