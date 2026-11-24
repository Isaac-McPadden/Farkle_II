from __future__ import annotations

import logging
import pickle
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pyarrow as pa
import pytest

from farkle.analysis.outcome_publication import _CoalescedWriter
from farkle.utils import parallel
from farkle.utils.telemetry import (
    SupervisorHeartbeatRecorder,
    observe_recovery,
    use_supervisor_recorder,
)


def test_interrupted_recovery_keeps_the_latest_counts_without_claiming_new_games(caplog):
    now = [1.0]
    logger = logging.getLogger("tests.recovery")
    resources = parallel.resource_telemetry_v2(
        {
            "process_tree_rss_bytes": 32 * 1024**2,
            "peak_process_tree_rss_bytes": 48 * 1024**2,
            "windows_job_committed_memory_peak_bytes": 110 * 1024**2,
            "rss_warning_active": True,
        }
    )
    recorder = SupervisorHeartbeatRecorder(
        logger,
        run="recovery",
        interval_seconds=45,
        clock=lambda: now[0],
        resource_sampler=lambda: resources,
        autostart=False,
    )
    with caplog.at_level(logging.INFO, logger=logger.name):
        scope = recorder.begin_scope(
            "recovery",
            run="test",
            stage="simulation",
            phase="simulation",
            progress={"planned_games": 80, "completed_games": 0},
        )
        with use_supervisor_recorder(recorder, scope):
            phase = observe_recovery("reconstruction", "Replay shards", total_files=3)
            assert phase is not None
            now[0] += 2
            phase.advance(files=1, rows=12, byte_count=2048)
            assert recorder.emit_if_due(force=True)
            phase.advance(rows=5)
        interrupted = scope.finish(status="interrupted")
        counts = cast(dict[str, dict[str, Any]], interrupted.as_metadata()["recovery_phases"])
        assert counts["simulation_recovery_reconstruction"]["rows_replayed"] == 17
    heartbeat = next(
        record
        for record in caplog.records
        if getattr(record, "telemetry_kind", None) == "heartbeat"
    )
    assert heartbeat.process_tree_rss_bytes == heartbeat.process_tree_resident_memory_bytes
    assert heartbeat.resource_telemetry_version == 2
    assert "files_checked=1/3" in heartbeat.message
    assert "games=0" not in heartbeat.message
    assert "process_tree_resident_memory=32.0 MiB" in heartbeat.message
    assert "windows_job_peak_committed_memory=110.0 MiB" in heartbeat.message
    assert "scheduler_backpressure=active" in heartbeat.message
    resumed = recorder.begin_scope("recovery", run="test", stage="simulation", phase="simulation")
    with use_supervisor_recorder(recorder, resumed):
        phase = observe_recovery("authentication", "Authenticate resumed work", total_files=3)
        assert phase is not None
        phase.advance(files=3, byte_count=4096)
        phase.finish()
    assert resumed.recovery_summary()["simulation_recovery_authentication"]["rows_replayed"] == 0
    resumed.finish(status="success")
    recorder.close()


def test_footer_budget_classification_survives_worker_serialization(tmp_path: Path) -> None:
    table = pa.table({"a": [1], "b": [2]})
    output = tmp_path / "summary.parquet"
    writer = _CoalescedWriter(
        lambda _: pytest.fail("over-budget writer ran"),
        table.schema,
        1536,
        path=output,
        budget_divisor=12,
    )
    writer.write_batch(table)
    with pytest.raises(parallel.ResourceBudgetError) as caught:
        writer.flush()
    wire_error = pickle.loads(pickle.dumps(caught.value))
    assert parallel.classify_resource_exception(wire_error) == "budget_admission_exceeded"
    assert wire_error.details["requirement"] == 2048
    assert wire_error.details["allowance"] == 1536
    assert wire_error.details["estimated"] is True
    assert wire_error.details["path"] == str(output)
    assert wire_error.details["setting"] == "resources.stage_batch_bytes.analysis"
    assert "1,024 bytes per column" in str(wire_error)
    assert "Reducing worker count leaves this per-writer allowance unchanged" in str(wire_error)
    assert not parallel.resource_failure_allows_retry(wire_error.classification)
    allocation = pa.ArrowMemoryError("64 MiB allocation rejected")
    failure = parallel.describe_resource_failure(
        allocation, "allocator_memory_error", operation="Decode Parquet"
    )
    assert "ArrowMemoryError allocation failed" in str(failure)
    assert parallel.resource_failure_allows_retry(failure.classification)


def test_recovery_observation_coalesces_progress_updates() -> None:
    recorder = SupervisorHeartbeatRecorder(
        logging.getLogger("tests.recovery"), run="bench", interval_seconds=0, autostart=False
    )
    scope = recorder.begin_scope("bench", run="test", stage="simulation", phase="simulation")
    with use_supervisor_recorder(recorder, scope):
        phase = observe_recovery("authentication", "Sample completed shard checks")
        assert phase is not None
        count = 20_000
        started = perf_counter()
        for _ in range(count):
            phase.advance(files=1, byte_count=8192)
        elapsed = perf_counter() - started
        phase.finish()
    snapshot = scope.recovery_summary()["simulation_recovery_authentication"]
    assert snapshot["files_checked"] == count
    assert snapshot["bytes_processed"] == count * 8192
    scope.finish(status="success")
    recorder.close()
    print(
        f"{count:,} recovery observations in {elapsed:.4f}s ({elapsed * 1e6 / count:.2f} us each)"
    )
