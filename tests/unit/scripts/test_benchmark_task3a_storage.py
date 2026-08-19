from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from scripts import benchmark_task3a_storage as benchmark


def _small_settings() -> benchmark.WorkloadSettings:
    return benchmark.WorkloadSettings(
        workers=1,
        rng_route_units=4,
        rng_records_per_unit=64,
        rng_partitions=2,
        rng_merge_fan_in=2,
        h2h_blocks=2,
        h2h_chunks_per_block=2,
        h2h_games_per_chunk=20,
        h2h_payload_records_per_chunk=4,
    )


def test_disposable_path_validation_requires_sync_separation_and_safe_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    local_parent = tmp_path / "local"
    local_parent.mkdir()
    monkeypatch.setenv("OneDrive", str(repository))
    monkeypatch.delenv("OneDriveConsumer", raising=False)
    monkeypatch.delenv("OneDriveCommercial", raising=False)

    synchronized = repository / "farkle-task3a-synchronized"
    local = local_parent / "farkle-task3a-local"
    assert benchmark.validate_disposable_roots(synchronized, local, repository_root=repository) == (
        synchronized,
        local,
    )

    with pytest.raises(ValueError, match="name must start"):
        benchmark.validate_disposable_roots(
            repository / "unsafe", local, repository_root=repository
        )
    with pytest.raises(ValueError, match="outside configured OneDrive"):
        benchmark.validate_disposable_roots(
            synchronized,
            repository / "farkle-task3a-not-local",
            repository_root=repository,
        )
    synchronized.mkdir()
    with pytest.raises(FileExistsError, match="must not already exist"):
        benchmark.validate_disposable_roots(synchronized, local, repository_root=repository)


def test_owned_cleanup_rejects_marker_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "farkle-task3a-owned"
    benchmark._create_owned_directory(root, "expected")
    marker = root / benchmark.MARKER_NAME
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["ownership_token"] = "wrong"
    marker.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ownership marker is invalid"):
        benchmark._remove_owned_directory(root, "expected")
    assert root.is_dir()


def test_bounded_workloads_are_byte_identical_and_report_exact_counts(tmp_path: Path) -> None:
    settings = _small_settings()
    roots = [tmp_path / "farkle-task3a-a", tmp_path / "farkle-task3a-b"]
    measurements = []
    for index, root in enumerate(roots):
        benchmark._create_owned_directory(root, str(index))
        rng = benchmark.run_rng_workload(root, settings, str(index))
        h2h = benchmark.run_h2h_workload(root, settings, str(index))
        measurements.append((rng, h2h))

    assert measurements[0][0]["correctness"] == measurements[1][0]["correctness"]
    assert measurements[0][1]["correctness"] == measurements[1][1]["correctness"]

    rng_metrics = measurements[0][0]["metrics"]
    assert rng_metrics["source_bytes_read"] == settings.rng_record_count * 32
    assert rng_metrics["route_bytes_read"] == settings.rng_record_count * 32 * 2
    assert rng_metrics["spill_runs_created"] == 8
    assert rng_metrics["merge_passes"] == 4
    assert rng_metrics["worker_failure_events"] == 0

    h2h_metrics = measurements[0][1]["metrics"]
    assert h2h_metrics["source_bytes_read"] > 0
    assert h2h_metrics["checkpoint_writes"] == 6
    assert h2h_metrics["sidecar_publications"] == 6
    assert h2h_metrics["authentication_calls"] == 6
    assert h2h_metrics["worker_failure_events"] == 0


def test_summary_uses_paired_ratios_and_order_positions() -> None:
    measured = []
    for repetition, order in enumerate(
        (("onedrive", "local"), ("local", "onedrive"), ("onedrive", "local")),
        start=1,
    ):
        for position, location in enumerate(order, start=1):
            wall = 12.0 if location == "onedrive" else 10.0
            measured.append(
                {
                    "location": location,
                    "repetition": repetition,
                    "position": position,
                    "rng": {"wall_seconds": wall, "total_cpu_seconds": wall / 2},
                    "h2h": {"wall_seconds": wall * 2, "total_cpu_seconds": wall},
                }
            )

    summary = benchmark.summarize(measured)
    rng = cast(dict[str, object], summary["rng"])
    h2h = cast(dict[str, object], summary["h2h"])
    assert cast(float, rng["onedrive_to_local_median_wall_ratio"]) == pytest.approx(1.2)
    assert cast(list[float], rng["paired_wall_ratios"]) == pytest.approx([1.2, 1.2, 1.2])
    assert rng["material_and_repeatable"] is True
    assert h2h["material_and_repeatable"] is True
