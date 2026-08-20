from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from scripts import benchmark_task4b_authenticated_graph as benchmark


def test_task4b_benchmark_rejects_unowned_root(tmp_path: Path) -> None:
    root = tmp_path / "unowned"
    root.mkdir()

    with pytest.raises(RuntimeError, match="unowned"):
        benchmark.run_benchmark(root, sizes=(3,), repetitions=1, force=False)


def test_task4b_benchmark_uses_three_context_finalization_and_resumes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "farkle-task4b-test"

    first = benchmark.run_benchmark(root, sizes=(6,), repetitions=1, force=False)

    assert first["tamper_equivalence"]["both_fail_closed"] is True
    measurements = first["measurements"]
    assert len(measurements) == 2
    baseline = next(item for item in measurements if item["mode"] == "baseline")
    optimized = next(item for item in measurements if item["mode"] == "optimized")
    assert baseline["result"]["status"] == optimized["result"]["status"] == "passed"
    assert baseline["telemetry"]["graph_audit_invocations"] == 3
    assert optimized["telemetry"]["graph_audit_invocations"] == 1
    assert optimized["result"]["top_level_invocations"] == 1
    assert len(optimized["result"]["internal_roots"]) == 3
    assert first["summary"][0]["canonical_outputs_unchanged"] is True

    evidence = root / "task4b_authenticated_graph.json"
    mtime = evidence.stat().st_mtime_ns
    resumed: dict[str, Any] = benchmark.run_benchmark(
        root,
        sizes=(6,),
        repetitions=1,
        force=False,
    )
    assert resumed == first
    assert evidence.stat().st_mtime_ns == mtime
