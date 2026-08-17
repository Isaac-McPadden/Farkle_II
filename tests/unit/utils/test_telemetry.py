from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

import pytest

from farkle.utils.telemetry import SupervisorHeartbeatRecorder


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.origin = datetime(2026, 1, 1, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.value

    def utc_now(self) -> datetime:
        return self.origin + timedelta(seconds=self.value)

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _resources() -> dict[str, object]:
    return {
        "process_tree_rss_bytes": 100,
        "peak_process_tree_rss_bytes": 125,
        "native_threads": 2,
        "peak_native_threads": 3,
        "aggregate_memory_bytes": 150,
        "peak_aggregate_memory_bytes": 175,
        "aggregate_memory_hard_limit_bytes": 1_000,
        "aggregate_memory_source": "test",
        "host_available_memory_bytes": 2_000,
        "warning_crossings": 0,
        "backpressure_seconds": 0.0,
        "near_hard_boundary": False,
        "monitoring_error": None,
    }


def test_fake_clock_heartbeat_schedule_coalesces_scopes_and_missed_deadlines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    logger = logging.getLogger("tests.telemetry.schedule")
    recorder = SupervisorHeartbeatRecorder(
        logger,
        run="pair",
        interval_seconds=45.0,
        resource_sampler=_resources,
        clock=clock.monotonic,
        utc_clock=clock.utc_now,
        autostart=False,
    )
    first = recorder.begin_scope(
        "root_48",
        run="root_48",
        stage="rng_diagnostics",
        phase="action",
    )
    second = recorder.begin_scope(
        "root_49",
        run="root_49",
        stage="simulation",
        phase="action",
    )

    with caplog.at_level(logging.INFO, logger=logger.name):
        clock.advance(44.9)
        assert recorder.emit_if_due() is False
        first.update(phase="completion_authentication", state="authenticating")
        clock.advance(0.1)
        assert recorder.emit_if_due() is True
        clock.advance(180.0)
        assert recorder.emit_if_due() is True
        assert recorder.emit_if_due() is False

    heartbeats = [
        record for record in caplog.records if getattr(record, "telemetry_kind", None) == "heartbeat"
    ]
    assert len(heartbeats) == 2
    assert all(record.__dict__["active_scope_count"] == 2 for record in heartbeats)
    assert heartbeats[0].__dict__["scope"] == "aggregate"
    active = {
        item["scope"]: item for item in heartbeats[0].__dict__["active_scopes"]
    }
    assert active["root_48"]["phase"] == "completion_authentication"
    assert heartbeats[0].__dict__["owner_pid"] == os.getpid()

    first_summary = first.finish(status="success")
    second.finish(status="success")
    assert first_summary.heartbeat_count == 2
    assert recorder.active_scope_count == 0
    assert recorder.close() == 0.0
    assert recorder.close() == 0.0


def test_owner_pid_and_bounded_scope_capacity_never_emit_from_non_owner(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    logger = logging.getLogger("tests.telemetry.owner")
    recorder = SupervisorHeartbeatRecorder(
        logger,
        run="test",
        interval_seconds=1.0,
        resource_sampler=_resources,
        clock=clock.monotonic,
        utc_clock=clock.utc_now,
        max_active_scopes=2,
        autostart=False,
    )
    recorder.begin_scope("one", run="test", stage="one", phase="action")
    recorder.begin_scope("two", run="test", stage="two", phase="action")
    dropped = recorder.begin_scope("three", run="test", stage="three", phase="action")
    assert recorder.active_scope_count == 2
    assert recorder.dropped_updates == 1
    assert dropped.finish(status="success").telemetry_error is not None

    non_owner = SupervisorHeartbeatRecorder(
        logger,
        run="child",
        interval_seconds=1.0,
        resource_sampler=_resources,
        clock=clock.monotonic,
        utc_clock=clock.utc_now,
        owner_pid=os.getpid() + 10_000,
        autostart=False,
    )
    non_owner.begin_scope("child", run="child", stage="child", phase="action")
    clock.advance(2.0)
    with caplog.at_level(logging.INFO, logger=logger.name):
        assert non_owner.emit_if_due() is False
    assert not [
        record for record in caplog.records if getattr(record, "telemetry_kind", None) == "heartbeat"
    ]


def test_resource_sampler_failure_is_reported_once_and_never_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    logger = logging.getLogger("tests.telemetry.failure")

    def fail() -> dict[str, object]:
        raise RuntimeError("sample failed")

    recorder = SupervisorHeartbeatRecorder(
        logger,
        run="test",
        interval_seconds=1.0,
        resource_sampler=fail,
        clock=clock.monotonic,
        utc_clock=clock.utc_now,
        autostart=False,
    )
    with caplog.at_level(logging.WARNING, logger=logger.name):
        scope = recorder.begin_scope("scope", run="test", stage="stage", phase="action")
        assert recorder.resource_snapshot()["monitoring_error"] == "RuntimeError: sample failed"
        assert recorder.resource_snapshot()["monitoring_error"] == "RuntimeError: sample failed"
    errors = [
        record
        for record in caplog.records
        if getattr(record, "telemetry_kind", None) == "telemetry_error"
    ]
    assert len(errors) == 1
    assert recorder.telemetry_errors == ("RuntimeError: sample failed",)
    assert scope.finish(status="success").telemetry_error == "RuntimeError: sample failed"


def test_context_manager_stops_supervisor_thread_on_exception() -> None:
    logger = logging.getLogger("tests.telemetry.cleanup")
    recorder = SupervisorHeartbeatRecorder(
        logger,
        run="test",
        interval_seconds=60.0,
        resource_sampler=_resources,
    )
    with pytest.raises(RuntimeError, match="boom"), recorder:
        recorder.begin_scope("scope", run="test", stage="stage", phase="action")
        raise RuntimeError("boom")
    assert recorder.close() == 0.0
