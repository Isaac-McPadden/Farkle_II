from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
from datetime import UTC, datetime, timedelta

import pytest

import farkle.utils.telemetry as telemetry_module
from farkle.utils.parallel import process_map
from farkle.utils.telemetry import (
    SupervisorHeartbeatRecorder,
    WorkerProgressEndpoint,
    install_worker_progress_endpoint,
    report_worker_progress,
)


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


def _spawn_progress_worker(value: int) -> tuple[int, bool]:
    return os.getpid(), report_worker_progress(
        "bounded_unit",
        counters={"units_observed": 1, "value_sum": value},
    )


class _BrokenQueue:
    def put_nowait(self, _value: object) -> None:
        raise OSError("closed")


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
        record
        for record in caplog.records
        if getattr(record, "telemetry_kind", None) == "heartbeat"
    ]
    assert len(heartbeats) == 2
    assert all(record.__dict__["active_scope_count"] == 2 for record in heartbeats)
    assert heartbeats[0].__dict__["scope"] == "aggregate"
    active = {item["scope"]: item for item in heartbeats[0].__dict__["active_scopes"]}
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
        record
        for record in caplog.records
        if getattr(record, "telemetry_kind", None) == "heartbeat"
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


def test_worker_endpoint_is_nonblocking_when_full_closed_or_broken() -> None:
    bounded: queue.Queue[object] = queue.Queue(maxsize=1)
    endpoint = WorkerProgressEndpoint(
        queue=bounded,
        scope="scope",
        owner_pid=os.getpid() + 1,
    )
    assert endpoint.emit("first", counters={"units": 1}) is True
    assert endpoint.emit("full", counters={"units": 1}) is False
    broken = WorkerProgressEndpoint(
        queue=_BrokenQueue(),
        scope="scope",
        owner_pid=os.getpid() + 1,
    )
    assert broken.emit("broken", counters={"units": 1}) is False


def test_duplicate_retried_and_out_of_order_worker_events_do_not_inflate_progress(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("tests.telemetry.reconciliation")
    snapshots: list[dict[str, object]] = []
    recorder = SupervisorHeartbeatRecorder(
        logger,
        run="root_52",
        interval_seconds=1.0,
        resource_sampler=_resources,
        heartbeat_sink=lambda payload: snapshots.append(dict(payload)),
        autostart=False,
    )
    scope = recorder.begin_scope(
        "simulation",
        run="root_52",
        stage="simulation",
        phase="simulation",
        progress={
            "root_seed": 52,
            "current_player_count": 2,
            "planned_games": 100,
            "completed_games": 40,
            "planned_shuffles": 10,
            "completed_shuffles": 4,
            "planned_chunks": 5,
            "completed_chunks": 2,
        },
    )
    channel_queue: queue.Queue[object] = queue.Queue()
    recorder._worker_channels.append(  # noqa: SLF001 - focused channel reconciliation test
        telemetry_module._WorkerProgressChannel(queue=channel_queue, scope=scope.scope)
    )
    messages = (
        ("shuffle:7", 101, 1, 10),
        ("shuffle:7", 202, 1, 10),  # retried duplicate on another worker
        ("shuffle:9", 101, 3, 20),
        ("shuffle:8", 101, 2, 5),  # delayed and out of order, but unique
    )
    for event_id, worker_pid, sequence, games in messages:
        channel_queue.put_nowait(
            {
                "phase": "simulation_shuffle_complete",
                "worker_pid": worker_pid,
                "sequence": sequence,
                "event_id": event_id,
                "counters": {
                    "worker_completed_shuffles": 1,
                    "worker_completed_games": games,
                },
            }
        )

    with caplog.at_level(logging.INFO, logger=logger.name):
        assert recorder.emit_if_due(force=True)

    record = next(
        item for item in caplog.records if getattr(item, "telemetry_kind", None) == "heartbeat"
    )
    progress = record.__dict__["active_scopes"][0]["progress"]
    assert progress["worker_messages_received"] == 4
    assert progress["worker_unique_messages"] == 3
    assert progress["worker_duplicate_messages"] == 1
    assert progress["worker_completed_shuffles"] == 3
    assert progress["worker_completed_games"] == 35
    assert progress["completed_games"] == 40
    assert "games=40/100 (40.0%)" in record.getMessage()
    assert "shuffles=4/10" in record.getMessage()
    assert "chunks=2/5" in record.getMessage()
    assert "seed=?" not in record.getMessage()
    assert snapshots[0]["active_scopes"] == record.__dict__["active_scopes"]

    scope.finish(status="success")
    recorder.close()


@pytest.mark.skipif("spawn" not in mp.get_all_start_methods(), reason="spawn unavailable")
def test_spawn_workers_feed_one_parent_recorder_through_bounded_channel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("tests.telemetry.spawn_channel")
    recorder = SupervisorHeartbeatRecorder(
        logger,
        run="spawn",
        interval_seconds=0.001,
        resource_sampler=_resources,
        autostart=False,
    )
    scope = recorder.begin_scope(
        "spawn_scope",
        run="spawn",
        stage="rng_diagnostics",
        phase="rng_reduce",
    )
    context = mp.get_context("spawn")
    endpoint = recorder.create_worker_progress_endpoint(
        scope.scope,
        mp_context=context,
        capacity=4,
    )
    assert endpoint is not None
    results: list[tuple[int, bool]] = []
    with caplog.at_level(logging.INFO, logger=logger.name):
        for result in process_map(
            _spawn_progress_worker,
            range(12),
            n_jobs=2,
            window=4,
            mp_context=context,
            initializer=install_worker_progress_endpoint,
            initargs=(endpoint,),
        ):
            results.append(result)
            recorder.emit_if_due(force=True)
    recorder.close_worker_progress_endpoint(endpoint)
    scope.finish(status="success")
    recorder.close()

    worker_pids = {pid for pid, _sent in results}
    assert worker_pids and os.getpid() not in worker_pids
    assert any(sent for _pid, sent in results)
    heartbeats = [
        record
        for record in caplog.records
        if getattr(record, "telemetry_kind", None) == "heartbeat"
    ]
    assert heartbeats
    assert {record.process for record in heartbeats} == {os.getpid()}
    progress = heartbeats[-1].__dict__["active_scopes"][0]["progress"]
    assert 1 <= progress["worker_messages_received"] <= 12
    assert 1 <= progress["units_observed"] <= 12
