from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import benchmark_task3b_rng_coarsening as benchmark


def test_safe_root_requires_owned_absent_directory(tmp_path: Path) -> None:
    safe = tmp_path / "farkle-task3b-safe"
    assert benchmark._safe_root(safe) == safe.resolve()
    with pytest.raises(ValueError, match="name must start"):
        benchmark._safe_root(tmp_path / "unsafe")
    safe.mkdir()
    with pytest.raises(FileExistsError, match="must not already exist"):
        benchmark._safe_root(safe)


def test_owned_cleanup_rejects_marker_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "farkle-task3b-cleanup"
    settings: dict[str, object] = {"fixture": 1}
    benchmark._prepare_root(root, settings, resume=False)
    marker = root / benchmark.MARKER_NAME
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["settings"] = {"fixture": 2}
    marker.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="mismatched marker"):
        benchmark._remove_owned_root(root, settings)
    assert root.is_dir()


def test_small_benchmark_uses_actual_routes_and_proves_exact_equivalence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "farkle-task3b-small"
    payload = benchmark.run_benchmark(
        root,
        scales=(8,),
        range_sizes=(1, 4),
        repetitions=1,
        include_two_workers=False,
    )
    measurements = [item for item in payload["measurements"] if not item["scenario"]["warmup"]]
    baseline = next(item for item in measurements if item["scenario"]["row_groups_per_unit"] == 1)
    coarsened = next(item for item in measurements if item["scenario"]["row_groups_per_unit"] == 4)

    assert baseline["durable_route_files"] == 16
    assert coarsened["durable_route_files"] == 4
    assert baseline["durable_route_unit_stamps"] == 16
    assert coarsened["durable_route_unit_stamps"] == 4
    assert baseline["reducer_route_unit_opens"] == 128
    assert coarsened["reducer_route_unit_opens"] == 32
    assert baseline["selection_membership_loads"] == 8
    assert coarsened["selection_membership_loads"] == 2
    assert baseline["exact_equivalence_digest"] == coarsened["exact_equivalence_digest"]
    assert payload["summary"]["exact_equivalence_across_range_sizes"] == {"8": True}
    assert (root / "warmup" / "measurement.json").is_file()
    assert (root / "rg8-range1-w1-r1" / "measurement.json").is_file()
    resumed = benchmark.run_benchmark(
        root,
        scales=(8,),
        range_sizes=(1, 4),
        repetitions=1,
        include_two_workers=False,
        resume=True,
    )
    assert resumed["summary"] == payload["summary"]


def test_summary_uses_medians_and_checks_repetition_digests() -> None:
    measurements = []
    for repetition, wall in enumerate((3.0, 1.0, 2.0), start=1):
        measurements.append(
            {
                "scenario": {
                    "source_row_groups": 256,
                    "row_groups_per_unit": 32,
                    "workers": 1,
                    "repetition": repetition,
                    "order_position": 1,
                    "warmup": False,
                },
                "total_wall_seconds": wall,
                "total_cpu_seconds": wall / 2,
                "peak_process_tree_rss_bytes": 100 + repetition,
                "durable_route_files": 16,
                "durable_route_unit_stamps": 16,
                "reducer_route_unit_opens": 128,
                "selection_membership_loads": 8,
                "initial_spill_runs": 16,
                "exact_equivalence_digest": "a" * 64,
            }
        )
    summary = benchmark.summarize(measurements)
    group = summary["groups"][0]
    assert group["median_wall_seconds"] == 2.0
    assert group["maximum_peak_process_tree_rss_bytes"] == 103
    assert group["repetition_digests_identical"] is True
