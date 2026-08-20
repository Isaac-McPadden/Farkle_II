from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import cast

import pytest
from scripts import benchmark_task4a_h2h_execution as benchmark


def test_benchmark_rejects_unowned_existing_root(tmp_path: Path) -> None:
    root = tmp_path / "farkle-task4a-unowned"
    root.mkdir()

    with pytest.raises(RuntimeError, match="unowned"):
        benchmark.run_benchmark(
            root,
            benchmark.BenchmarkSettings(
                workers=1,
                repetitions=1,
                targets=(2,),
                include_exceptional=False,
            ),
        )


def test_quick_benchmark_uses_real_execute_path_and_resumes(tmp_path: Path) -> None:
    root = tmp_path / "farkle-task4a-quick"
    settings = benchmark.BenchmarkSettings(
        workers=1,
        repetitions=1,
        targets=(3,),
        include_exceptional=False,
        heartbeat_seconds=0.0,
    )

    first = benchmark.run_benchmark(root, settings)
    assert first["exact_policy_equivalence"] is True
    measurements = cast(list[dict[str, object]], first["measurements"])
    assert len(measurements) == 3
    for measurement in measurements:
        counts = cast(dict[str, int], measurement["counts"])
        telemetry = cast(dict[str, object], measurement["telemetry"])
        assert counts["blocks"] == 4
        assert counts["completed"] == 12
        assert counts["attempted"] == 12
        assert telemetry["scheduled_chunks"] == telemetry["completed_chunks"] == 4
        assert telemetry["checkpoint_writes"] == 4
        assert telemetry["pool_generations"] == 1
        assert telemetry["worker_initializer_loads"] == 1
        assert telemetry["sidecar_publications"] == (
            cast(int, telemetry["checkpoint_writes"])
            + cast(int, telemetry["execution_state_writes"])
            + 1
        )

    measurement_paths = sorted(root.glob("**/measurement.json"))
    mtimes = {path: path.stat().st_mtime_ns for path in measurement_paths}
    resumed = benchmark.run_benchmark(root, settings)
    assert resumed == first
    assert {path: path.stat().st_mtime_ns for path in measurement_paths} == mtimes


def test_summary_rejects_policy_output_difference() -> None:
    base: dict[str, object] = {
        "scenario": {"name": "normal", "target": 3, "profile_kind": "normal"},
        "repetition": 1,
        "policy": "baseline",
        "wall_seconds": 1.0,
        "counts": {"attempted": 12},
        "digests": {
            "logical_rows": "a",
            "aggregate_parquet": "b",
            "block_parquets": "c",
        },
        "telemetry": {
            "scheduled_chunks": 4,
            "checkpoint_writes": 4,
            "pool_generations": 1,
            "worker_initializer_loads": 1,
        },
    }
    changed = {
        **base,
        "policy": "selected",
        "digests": {**cast(dict[str, object], base["digests"]), "logical_rows": "different"},
    }

    with pytest.raises(AssertionError, match="policy outputs differ"):
        benchmark.summarize([base, changed])


@pytest.mark.skipif("spawn" not in mp.get_all_start_methods(), reason="spawn unavailable")
def test_real_h2h_path_is_identical_across_worker_counts(tmp_path: Path) -> None:
    scenario = benchmark.Scenario("worker-equivalence", 2)
    one_settings = benchmark.BenchmarkSettings(
        workers=1,
        repetitions=1,
        targets=(2,),
        include_exceptional=False,
        heartbeat_seconds=0.0,
    )
    two_settings = benchmark.BenchmarkSettings(
        workers=2,
        repetitions=1,
        targets=(2,),
        include_exceptional=False,
        heartbeat_seconds=0.0,
    )
    one = benchmark._measure_once(
        tmp_path / "one",
        scenario,
        "selected_cap_bounded_5000",
        5_000,
        1,
        one_settings,
    )
    two = benchmark._measure_once(
        tmp_path / "two",
        scenario,
        "selected_cap_bounded_5000",
        5_000,
        1,
        two_settings,
    )

    assert one["counts"] == two["counts"]
    assert one["digests"] == two["digests"]
    assert cast(dict[str, object], one["telemetry"])["pool_generations"] == 1
    assert cast(dict[str, object], two["telemetry"])["pool_generations"] == 1
    assert cast(dict[str, object], two["telemetry"])["worker_initializer_loads"] == 2
