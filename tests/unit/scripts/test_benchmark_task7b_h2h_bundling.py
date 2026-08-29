from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from scripts import benchmark_task7b_h2h_bundling as benchmark


def _quick_settings() -> benchmark.BenchmarkSettings:
    return benchmark.BenchmarkSettings(
        workers=1,
        repetitions=1,
        heartbeat_seconds=0.0,
        cases=(benchmark.BenchmarkCase("quick-12-block", 3, 2),),
    )


def test_owned_root_requires_exact_name_and_matching_settings(tmp_path: Path) -> None:
    settings = _quick_settings()
    with pytest.raises(ValueError, match="exact name"):
        benchmark.prepare_owned_root(tmp_path / "wrong", settings)

    root = tmp_path / benchmark.SAFE_ROOT_NAME
    benchmark.prepare_owned_root(root, settings)
    marker = root / benchmark.MARKER_NAME
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["settings"]["workers"] = 2
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="marker or settings mismatch"):
        benchmark.prepare_owned_root(root, settings)


def test_case_dimensions_use_executable_pair_relationship() -> None:
    case = benchmark.BenchmarkCase("accepted", 10, 4)
    assert case.expected_pairs == 45
    assert case.expected_blocks == 180
    assert benchmark.expected_checkpoint_count("baseline", 180) == 180
    assert benchmark.expected_checkpoint_count("candidate", 180) == 12
    assert benchmark.expected_checkpoint_count("candidate", 181) == 12
    with pytest.raises(ValueError, match="phase"):
        benchmark.expected_checkpoint_count("unknown", 180)


def test_quick_real_path_is_resumable_and_audited(tmp_path: Path) -> None:
    root = tmp_path / benchmark.SAFE_ROOT_NAME
    settings = _quick_settings()

    first = benchmark.run_phase(root, phase="baseline", settings=settings)
    assert first["exact_within_phase_equivalence"] is True
    rows = cast(list[dict[str, object]], first["measurements"])
    assert len(rows) == 1
    logical = cast(dict[str, object], rows[0]["logical_counts"])
    physical = cast(dict[str, object], rows[0]["physical_checkpoints"])
    assert logical["rows"] == 12
    assert logical["completed"] == 24
    assert rows[0]["release_audit_failures"] == []
    assert physical["count"] == 12

    measurement = next((root / "baseline").glob("**/measurement.json"))
    before = measurement.stat().st_mtime_ns
    resumed = benchmark.run_phase(root, phase="baseline", settings=settings)
    assert resumed == first
    assert measurement.stat().st_mtime_ns == before


def test_summary_rejects_logical_repetition_difference() -> None:
    base: dict[str, object] = {
        "case": {"name": "case"},
        "wall_seconds": 1.0,
        "telemetry": {"block_checkpoint_seconds": 0.5},
        "schedule": {"logical_arrow_sha256": "a"},
        "logical_counts": {"logical_arrow_sha256": "b"},
        "canonical_artifacts": {"inference": [{"sha256": "c"}]},
    }
    changed = {
        **base,
        "logical_counts": {"logical_arrow_sha256": "different"},
    }
    with pytest.raises(AssertionError, match="repetitions differ logically"):
        benchmark.summarize([base, changed])
