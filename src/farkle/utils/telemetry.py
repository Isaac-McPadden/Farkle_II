"""Parent-owned heartbeat and phase-timing telemetry.

The recorder in this module is deliberately operational.  It emits only to the
normal log stream and its state is not part of configuration, freshness, or
artifact identity.  Process workers must not own a recorder or emit heartbeat
records; later stage-specific progress can feed the parent through a bounded
channel without changing this ownership rule.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from queue import Empty, Full
from typing import Any

import psutil

from farkle.utils.parallel import (
    aggregate_memory_sample,
    process_tree_native_thread_count,
    process_tree_rss_bytes,
)

HEARTBEAT_INTERVAL_SECONDS = 45.0
MAX_ACTIVE_TELEMETRY_SCOPES = 128
MAX_COMPLETED_PROGRESS_SUMMARIES = 128
MAX_PROGRESS_FIELDS = 64
WORKER_PROGRESS_QUEUE_CAPACITY = 256

MonotonicClock = Callable[[], float]
UtcClock = Callable[[], datetime]
Waiter = Callable[[threading.Event, float], bool]
ResourceSampler = Callable[[], Mapping[str, object]]

_CURRENT_SUPERVISOR_RECORDER: ContextVar[SupervisorHeartbeatRecorder | None] = ContextVar(
    "farkle_current_supervisor_recorder",
    default=None,
)
_CURRENT_SUPERVISOR_SCOPE: ContextVar[SupervisorScope | None] = ContextVar(
    "farkle_current_supervisor_scope",
    default=None,
)
_WORKER_PROGRESS_ENDPOINT: WorkerProgressEndpoint | None = None


def _wait_for_event(event: threading.Event, timeout: float) -> bool:
    return event.wait(timeout)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def sample_process_resource_state() -> dict[str, object]:
    """Return a read-only best-effort resource snapshot for standalone runs."""

    rss = process_tree_rss_bytes()
    native_threads = process_tree_native_thread_count()
    aggregate = aggregate_memory_sample()
    aggregate_current = None
    aggregate_peak = None
    aggregate_limit = None
    aggregate_source = None
    if aggregate is not None:
        aggregate_current = aggregate.current_bytes
        aggregate_peak = aggregate.peak_bytes
        aggregate_limit = aggregate.hard_limit_bytes
        aggregate_source = aggregate.source
    return {
        "process_tree_rss_bytes": int(rss),
        "peak_process_tree_rss_bytes": int(rss),
        "native_threads": int(native_threads),
        "peak_native_threads": int(native_threads),
        "aggregate_memory_bytes": int(aggregate_current or 0),
        "aggregate_memory_current_bytes": aggregate_current,
        "peak_aggregate_memory_bytes": int(aggregate_peak or aggregate_current or 0),
        "windows_job_committed_memory_peak_bytes": (
            int(aggregate_peak or 0) if aggregate_source == "windows_job" else None
        ),
        "aggregate_memory_hard_limit_bytes": int(aggregate_limit or 0),
        "aggregate_memory_source": aggregate_source,
        "host_available_memory_bytes": int(psutil.virtual_memory().available),
        "warning_crossings": 0,
        "rss_warning_active": False,
        "rss_warning_threshold_bytes": None,
        "backpressure_seconds": 0.0,
        "near_hard_boundary": False,
        "monitoring_error": None,
    }


@dataclass(frozen=True, slots=True)
class ScopeTimingSummary:
    """Serializable timing and resource summary for one supervisor scope."""

    scope: str
    run: str
    stage: str
    status: str
    started_utc: str
    ended_utc: str
    elapsed_seconds: float
    heartbeat_count: int
    resource_start: Mapping[str, object]
    resource_end: Mapping[str, object]
    telemetry_error: str | None = None

    def as_metadata(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(slots=True)
class _ActiveScope:
    scope: str
    run: str
    stage: str
    phase: str
    state: str
    started_at: float
    started_utc: datetime
    phase_changed_at: float
    heartbeat_count: int
    resource_start: Mapping[str, object]
    progress: dict[str, object]


@dataclass(frozen=True, slots=True)
class WorkerProgressEndpoint:
    """Pickleable, nonblocking worker endpoint for lossy operational updates."""

    queue: Any
    scope: str
    owner_pid: int

    def emit(
        self,
        phase: str,
        *,
        counters: Mapping[str, int | float] | None = None,
        state: str = "working",
    ) -> bool:
        """Try one bounded update; failure or queue pressure never affects work."""

        if os.getpid() == self.owner_pid:
            return False
        message = {
            "scope": self.scope,
            "phase": str(phase),
            "state": str(state),
            "worker_pid": os.getpid(),
            "counters": dict(counters or {}),
        }
        try:
            self.queue.put_nowait(message)
        except (Full, OSError, EOFError, ValueError, AssertionError):
            return False
        except BaseException:  # telemetry must not interrupt valid worker work
            return False
        return True


@dataclass(slots=True)
class _WorkerProgressChannel:
    queue: Any
    scope: str
    messages_received: int = 0


def install_worker_progress_endpoint(endpoint: WorkerProgressEndpoint | None) -> None:
    """Install a process-local nonlogging endpoint from a pool initializer."""

    global _WORKER_PROGRESS_ENDPOINT
    _WORKER_PROGRESS_ENDPOINT = endpoint
    if endpoint is not None:
        # A full/broken telemetry pipe must never delay worker shutdown.
        with suppress(Exception):
            endpoint.queue.cancel_join_thread()


def report_worker_progress(
    phase: str,
    *,
    counters: Mapping[str, int | float] | None = None,
    state: str = "working",
) -> bool:
    """Submit one optional worker update without logging or blocking."""

    endpoint = _WORKER_PROGRESS_ENDPOINT
    return bool(endpoint and endpoint.emit(phase, counters=counters, state=state))


@contextmanager
def use_supervisor_recorder(
    recorder: SupervisorHeartbeatRecorder | None,
    scope: SupervisorScope | None = None,
) -> Any:
    """Expose the parent recorder to nested stage code for this call context."""

    token = _CURRENT_SUPERVISOR_RECORDER.set(recorder)
    scope_token = _CURRENT_SUPERVISOR_SCOPE.set(scope)
    try:
        yield recorder
    finally:
        _CURRENT_SUPERVISOR_SCOPE.reset(scope_token)
        _CURRENT_SUPERVISOR_RECORDER.reset(token)


def current_supervisor_recorder() -> SupervisorHeartbeatRecorder | None:
    """Return the recorder bound by the parent StageRunner, if any."""

    recorder = _CURRENT_SUPERVISOR_RECORDER.get()
    if recorder is None or os.getpid() != recorder.owner_pid:
        return None
    return recorder


def current_supervisor_scope() -> SupervisorScope | None:
    """Return the StageRunner scope bound in the parent call context, if any."""

    if current_supervisor_recorder() is None:
        return None
    return _CURRENT_SUPERVISOR_SCOPE.get()


class SupervisorScope:
    """Handle for updating and finishing one recorder-owned scope."""

    def __init__(
        self,
        recorder: SupervisorHeartbeatRecorder,
        scope: str,
        *,
        registered: bool,
    ) -> None:
        self._recorder = recorder
        self.scope = scope
        self._registered = registered
        self._summary: ScopeTimingSummary | None = None

    def update(
        self,
        *,
        phase: str,
        state: str = "working",
        progress: Mapping[str, object] | None = None,
    ) -> None:
        """Coalesce the latest scope state in the parent recorder."""

        if self._registered and self._summary is None:
            self._recorder.update_scope(
                self.scope,
                phase=phase,
                state=state,
                progress=progress,
            )

    def finish(self, *, status: str) -> ScopeTimingSummary:
        """Finish the scope idempotently and return its timing summary."""

        if self._summary is None:
            self._summary = self._recorder.finish_scope(
                self.scope,
                status=status,
                registered=self._registered,
            )
        return self._summary


class SupervisorHeartbeatRecorder:
    """Emit one aggregate parent-process heartbeat at each bounded interval."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        run: str,
        interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
        resource_sampler: ResourceSampler | None = None,
        clock: MonotonicClock = time.monotonic,
        utc_clock: UtcClock = _utc_now,
        waiter: Waiter = _wait_for_event,
        owner_pid: int | None = None,
        max_active_scopes: int = MAX_ACTIVE_TELEMETRY_SCOPES,
        autostart: bool = True,
    ) -> None:
        self._logger = logger
        self.run = run
        self.interval_seconds = max(0.0, float(interval_seconds))
        self._resource_sampler = resource_sampler or sample_process_resource_state
        self._clock = clock
        self._utc_clock = utc_clock
        self._waiter = waiter
        self.owner_pid = os.getpid() if owner_pid is None else int(owner_pid)
        self._max_active_scopes = max(1, int(max_active_scopes))
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = float(self._clock())
        self._next_deadline = (
            self._started_at + self.interval_seconds
            if self.interval_seconds > 0
            else math.inf
        )
        self._active: dict[str, _ActiveScope] = {}
        self._closed = False
        self._heartbeat_count = 0
        self._dropped_updates = 0
        self._errors: list[str] = []
        self._error_keys: set[str] = set()
        self._worker_channels: list[_WorkerProgressChannel] = []
        self._completed_progress: dict[str, dict[str, object]] = {}
        if autostart:
            self.start()

    def __enter__(self) -> SupervisorHeartbeatRecorder:
        self.start()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    @property
    def heartbeat_count(self) -> int:
        with self._lock:
            return self._heartbeat_count

    @property
    def active_scope_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def dropped_updates(self) -> int:
        with self._lock:
            return self._dropped_updates

    @property
    def telemetry_errors(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._errors)

    def start(self) -> None:
        """Start the single supervisor thread idempotently in the owner process."""

        if os.getpid() != self.owner_pid or self.interval_seconds <= 0:
            return
        with self._lock:
            if self._closed or (self._thread is not None and self._thread.is_alive()):
                return
            self._thread = threading.Thread(
                target=self._run,
                name="farkle-supervisor-heartbeat",
                daemon=True,
            )
            self._thread.start()

    def begin_scope(
        self,
        scope: str,
        *,
        run: str,
        stage: str,
        phase: str,
        state: str = "working",
    ) -> SupervisorScope:
        """Register or replace one bounded, coalesced active scope."""

        now = float(self._clock())
        resource_start = self.resource_snapshot()
        registered = os.getpid() == self.owner_pid
        with self._lock:
            if registered and scope not in self._active and len(self._active) >= self._max_active_scopes:
                registered = False
                self._dropped_updates += 1
                self._record_error_locked(
                    "scope_capacity",
                    f"telemetry active-scope capacity {self._max_active_scopes} exceeded",
                )
            if registered:
                self._active[scope] = _ActiveScope(
                    scope=scope,
                    run=run,
                    stage=stage,
                    phase=phase,
                    state=state,
                    started_at=now,
                    started_utc=self._utc_clock(),
                    phase_changed_at=now,
                    heartbeat_count=0,
                    resource_start=resource_start,
                    progress={},
                )
        return SupervisorScope(self, scope, registered=registered)

    def update_scope(
        self,
        scope: str,
        *,
        phase: str,
        state: str = "working",
        progress: Mapping[str, object] | None = None,
    ) -> bool:
        """Replace the latest state for a scope without accumulating events."""

        if os.getpid() != self.owner_pid:
            return False
        now = float(self._clock())
        with self._lock:
            active = self._active.get(scope)
            if active is None:
                self._dropped_updates += 1
                return False
            if active.phase != phase or active.state != state:
                active.phase = phase
                active.state = state
                active.phase_changed_at = now
            if progress is not None:
                merged = dict(active.progress)
                merged.update(dict(progress))
                active.progress = dict(list(merged.items())[:MAX_PROGRESS_FIELDS])
        return True

    def create_worker_progress_endpoint(
        self,
        scope: str,
        *,
        mp_context: Any,
        capacity: int = WORKER_PROGRESS_QUEUE_CAPACITY,
    ) -> WorkerProgressEndpoint | None:
        """Attach one bounded worker queue drained by this recorder's thread."""

        if os.getpid() != self.owner_pid:
            return None
        try:
            context = mp.get_context() if mp_context is None else mp_context
            queue = context.Queue(maxsize=max(1, int(capacity)))
        except BaseException as exc:  # telemetry setup cannot fail computation
            with self._lock:
                self._record_error_locked(
                    "worker_channel_create",
                    f"{type(exc).__name__}: {exc}",
                )
            return None
        with self._lock:
            self._worker_channels.append(_WorkerProgressChannel(queue=queue, scope=scope))
        return WorkerProgressEndpoint(queue=queue, scope=scope, owner_pid=self.owner_pid)

    def close_worker_progress_endpoint(
        self,
        endpoint: WorkerProgressEndpoint | None,
    ) -> None:
        """Drain and close one worker channel after its executor is quiescent."""

        if endpoint is None or os.getpid() != self.owner_pid:
            return
        self._drain_worker_progress()
        channel: _WorkerProgressChannel | None = None
        with self._lock:
            for candidate in self._worker_channels:
                if candidate.queue is endpoint.queue:
                    channel = candidate
                    self._worker_channels.remove(candidate)
                    break
        if channel is not None:
            with suppress(Exception):
                channel.queue.close()
            with suppress(Exception):
                channel.queue.join_thread()

    def record_completion_summary(
        self,
        scope: str,
        *,
        stage: str,
        summary: Mapping[str, object],
    ) -> None:
        """Retain and log one authoritative, bounded operational summary."""

        if os.getpid() != self.owner_pid:
            return
        payload = dict(list(summary.items())[:MAX_PROGRESS_FIELDS])
        with self._lock:
            if scope not in self._completed_progress and len(self._completed_progress) >= (
                MAX_COMPLETED_PROGRESS_SUMMARIES
            ):
                oldest = next(iter(self._completed_progress))
                self._completed_progress.pop(oldest, None)
                self._dropped_updates += 1
            self._completed_progress[scope] = payload
        with suppress(Exception):
            self._logger.info(
                "Pipeline progress summary",
                extra={
                    "telemetry_kind": "completion_summary",
                    "run": self.run,
                    "scope": scope,
                    "stage": stage,
                    "owner_pid": self.owner_pid,
                    "progress": payload,
                },
            )

    def finish_scope(
        self,
        scope: str,
        *,
        status: str,
        registered: bool = True,
    ) -> ScopeTimingSummary:
        """Remove a scope and return its immutable timing/resource summary."""

        ended_at = float(self._clock())
        ended_utc = self._utc_clock()
        resource_end = self.resource_snapshot()
        active: _ActiveScope | None = None
        if registered and os.getpid() == self.owner_pid:
            with self._lock:
                active = self._active.pop(scope, None)
        if active is None:
            return ScopeTimingSummary(
                scope=scope,
                run=self.run,
                stage=scope,
                status=status,
                started_utc=_utc_text(ended_utc),
                ended_utc=_utc_text(ended_utc),
                elapsed_seconds=0.0,
                heartbeat_count=0,
                resource_start={},
                resource_end=resource_end,
                telemetry_error="scope was not registered by the owner process",
            )
        return ScopeTimingSummary(
            scope=scope,
            run=active.run,
            stage=active.stage,
            status=status,
            started_utc=_utc_text(active.started_utc),
            ended_utc=_utc_text(ended_utc),
            elapsed_seconds=max(0.0, ended_at - active.started_at),
            heartbeat_count=active.heartbeat_count,
            resource_start=dict(active.resource_start),
            resource_end=resource_end,
            telemetry_error=self.telemetry_errors[-1] if self.telemetry_errors else None,
        )

    def resource_snapshot(self) -> dict[str, object]:
        """Sample resources without allowing telemetry failure to affect work."""

        try:
            return dict(self._resource_sampler())
        except Exception as exc:  # noqa: BLE001 - telemetry is non-authoritative
            message = f"{type(exc).__name__}: {exc}"
            should_warn = False
            with self._lock:
                should_warn = self._record_error_locked("resource_sampler", message)
            if should_warn and os.getpid() == self.owner_pid:
                with suppress(Exception):
                    self._logger.warning(
                        "Supervisor telemetry resource sampling failed",
                        extra={
                            "telemetry_kind": "telemetry_error",
                            "run": self.run,
                            "owner_pid": self.owner_pid,
                            "error": message,
                        },
                    )
            return {"monitoring_error": message}

    def emit_if_due(self, *, force: bool = False) -> bool:
        """Emit at most one aggregate heartbeat and advance missed deadlines."""

        if os.getpid() != self.owner_pid or self.interval_seconds <= 0:
            return False
        self._drain_worker_progress()
        now = float(self._clock())
        with self._lock:
            if not force and now < self._next_deadline:
                return False
            while self._next_deadline <= now:
                self._next_deadline += self.interval_seconds
            active = [self._active[key] for key in sorted(self._active)]
            if not active:
                return False
            for item in active:
                item.heartbeat_count += 1
            self._heartbeat_count += 1
            heartbeat_count = self._heartbeat_count
        resources = self.resource_snapshot()
        primary = active[0]
        active_payload = [
            {
                "scope": item.scope,
                "run": item.run,
                "stage": item.stage,
                "phase": item.phase,
                "state": item.state,
                "elapsed_seconds": max(0.0, now - item.started_at),
                "seconds_since_phase_change": max(0.0, now - item.phase_changed_at),
                "progress": dict(item.progress),
            }
            for item in active
        ]
        extra: dict[str, Any] = {
            "telemetry_kind": "heartbeat",
            "run": primary.run if len(active) == 1 else self.run,
            "scope": primary.scope if len(active) == 1 else "aggregate",
            "stage": primary.stage if len(active) == 1 else "multiple",
            "phase": primary.phase if len(active) == 1 else "multiple",
            "state": primary.state if len(active) == 1 else "working",
            "elapsed_seconds": max(0.0, now - self._started_at),
            "active_scope_count": len(active),
            "active_scopes": active_payload,
            "owner_pid": self.owner_pid,
            "heartbeat_count": heartbeat_count,
            **resources,
        }
        try:
            self._logger.info("Pipeline heartbeat", extra=extra)
        except Exception as exc:  # noqa: BLE001 - logging cannot fail computation
            with self._lock:
                self._record_error_locked(
                    "heartbeat_logging",
                    f"{type(exc).__name__}: {exc}",
                )
            return False
        return True

    def summary(self) -> dict[str, object]:
        """Return bounded recorder state for the one final health publication."""

        with self._lock:
            return {
                "heartbeat_count": self._heartbeat_count,
                "dropped_updates": self._dropped_updates,
                "telemetry_errors": list(self._errors),
                "active_scope_count": len(self._active),
                "resource_summary": self.resource_snapshot(),
                "completed_progress": dict(self._completed_progress),
            }

    def close(self) -> float:
        """Stop and join the supervisor thread idempotently."""

        started = float(self._clock())
        with self._lock:
            if self._closed:
                return 0.0
            self._closed = True
            thread = self._thread
        self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, min(self.interval_seconds, 5.0)))
            if thread.is_alive():
                with self._lock:
                    self._record_error_locked(
                        "heartbeat_shutdown",
                        "supervisor heartbeat thread did not stop within the join timeout",
                    )
        self._drain_worker_progress()
        with self._lock:
            channels = list(self._worker_channels)
            self._worker_channels.clear()
            orphaned_scopes = len(self._active)
            self._active.clear()
            if orphaned_scopes:
                self._record_error_locked(
                    "orphaned_scopes",
                    f"telemetry closed with {orphaned_scopes} unfinished scope(s)",
                )
        for channel in channels:
            with suppress(Exception):
                channel.queue.close()
            with suppress(Exception):
                channel.queue.join_thread()
        return max(0.0, float(self._clock()) - started)

    def _run(self) -> None:
        while not self._stop.is_set():
            now = float(self._clock())
            timeout = max(0.0, self._next_deadline - now)
            if self._waiter(self._stop, timeout):
                return
            self.emit_if_due()

    def _record_error_locked(self, key: str, message: str) -> bool:
        if key in self._error_keys:
            return False
        self._error_keys.add(key)
        self._errors.append(message)
        return True

    def _drain_worker_progress(self) -> None:
        """Coalesce bounded worker messages into active parent scope snapshots."""

        if os.getpid() != self.owner_pid:
            return
        with self._lock:
            channels = list(self._worker_channels)
        for channel in channels:
            while True:
                try:
                    message = channel.queue.get_nowait()
                except Empty:
                    break
                except (OSError, EOFError, ValueError, AssertionError):
                    break
                except BaseException:
                    break
                if not isinstance(message, Mapping):
                    continue
                counters = message.get("counters")
                with self._lock:
                    active = self._active.get(channel.scope)
                    if active is None:
                        continue
                    channel.messages_received += 1
                    progress = dict(active.progress)
                    progress["worker_messages_received"] = channel.messages_received
                    progress["worker_last_phase"] = str(message.get("phase", "unknown"))
                    progress["worker_last_pid"] = int(message.get("worker_pid", 0))
                    if isinstance(counters, Mapping):
                        for key, value in list(counters.items())[:MAX_PROGRESS_FIELDS]:
                            if isinstance(value, bool) or not isinstance(value, (int, float)):
                                continue
                            current = progress.get(str(key), 0)
                            if isinstance(current, (int, float)) and not isinstance(current, bool):
                                progress[str(key)] = current + value
                    active.progress = dict(list(progress.items())[:MAX_PROGRESS_FIELDS])


__all__ = [
    "HEARTBEAT_INTERVAL_SECONDS",
    "MAX_ACTIVE_TELEMETRY_SCOPES",
    "MAX_COMPLETED_PROGRESS_SUMMARIES",
    "WORKER_PROGRESS_QUEUE_CAPACITY",
    "ScopeTimingSummary",
    "SupervisorHeartbeatRecorder",
    "SupervisorScope",
    "WorkerProgressEndpoint",
    "current_supervisor_recorder",
    "current_supervisor_scope",
    "install_worker_progress_endpoint",
    "report_worker_progress",
    "sample_process_resource_state",
    "use_supervisor_recorder",
]
