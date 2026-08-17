"""Reusable stage runner with manifest logging and operational timing."""

from __future__ import annotations

import dataclasses
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from farkle.config import AppConfig
from farkle.utils.manifest import (
    EVENT_RUN_END,
    EVENT_RUN_START,
    EVENT_STAGE_END,
    EVENT_STAGE_START,
    append_manifest_event,
    make_run_id,
    validate_manifest_contract,
)
from farkle.utils.stage_completion import CompletionState, read_stage_done, resolve_stage_state
from farkle.utils.telemetry import (
    HEARTBEAT_INTERVAL_SECONDS,
    SupervisorHeartbeatRecorder,
    use_supervisor_recorder,
)

LOGGER = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _elapsed(clock: Callable[[], float], started: float) -> float:
    return max(0.0, float(clock()) - started)


@dataclasses.dataclass(frozen=True)
class StagePlanItem:
    """One stage entry for the runner."""

    name: str
    action: Callable[[AppConfig], None]
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    required_outputs: Sequence[Path] = dataclasses.field(default_factory=tuple)
    completion_stamp: Path | None = None
    freshness_key: Mapping[str, Any] | None = None


class StageValidationError(RuntimeError):
    """Raised when a stage completes without required artifacts."""

    def __init__(self, stage: str, missing_outputs: Sequence[Path]):
        self.stage = stage
        self.missing_outputs = tuple(missing_outputs)
        missing_text = ", ".join(str(path) for path in self.missing_outputs)
        super().__init__(f"Stage {stage!r} missing required outputs: {missing_text}")


class StageCompletionError(RuntimeError):
    """Raised when a stage's declared completion contract is not successful."""

    def __init__(self, stage: str, stamp: Path, status: str, reason: object = None):
        self.stage = stage
        self.stamp = stamp
        self.status = status
        self.reason = reason
        detail = f": {reason}" if reason else ""
        super().__init__(f"Stage {stage!r} completion stamp {stamp} has status {status!r}{detail}")


@dataclasses.dataclass(frozen=True)
class StageRunContext:
    """Configuration for stage execution, manifest logging, and telemetry."""

    config: AppConfig
    manifest_path: Path
    run_label: str
    run_id: str | None = None
    run_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    run_end_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    run_start_event: str = EVENT_RUN_START
    run_end_event: str = EVENT_RUN_END
    stage_start_event: str = EVENT_STAGE_START
    stage_end_event: str = EVENT_STAGE_END
    continue_on_error: bool = False
    use_progress: bool = False
    progress_desc: str = "pipeline"
    logger: logging.Logger = LOGGER
    telemetry: SupervisorHeartbeatRecorder | None = None
    heartbeat_interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS
    resource_sampler: Callable[[], Mapping[str, object]] | None = None
    monotonic_clock: Callable[[], float] = dataclasses.field(
        default=time.monotonic,
        repr=False,
        compare=False,
    )
    utc_clock: Callable[[], datetime] = dataclasses.field(
        default=_utc_now,
        repr=False,
        compare=False,
    )


@dataclasses.dataclass(frozen=True)
class StageTimingSummary:
    """Operational timing for one StageRunner item."""

    stage: str
    status: str
    started_utc: str
    ended_utc: str
    elapsed_seconds: float
    stage_start_publication_seconds: float
    action_seconds: float
    output_validation_seconds: float
    completion_read_seconds: float
    authentication_seconds: float
    stage_end_publication_seconds: float
    cleanup_seconds: float
    heartbeat_count: int
    resource_start: Mapping[str, object]
    resource_end: Mapping[str, object]
    telemetry_error: str | None = None

    def as_metadata(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class StageRunResult:
    """Summary of stage execution."""

    failed_steps: Sequence[str]
    first_failure: Exception | None
    stage_timings: Sequence[StageTimingSummary] = dataclasses.field(default_factory=tuple)
    telemetry_cleanup_seconds: float = 0.0


class StageRunner:
    """Execute stage plans while recording results to a manifest."""

    @staticmethod
    def run(
        plan: Sequence[StagePlanItem],
        context: StageRunContext,
        *,
        raise_on_failure: bool = True,
    ) -> StageRunResult:
        """Execute a stage plan while recording manifest events and validation."""

        manifest_path = context.manifest_path
        config_sha = getattr(context.config, "config_sha", None)
        run_id = context.run_id or make_run_id(context.run_label)
        recorder = context.telemetry
        owns_recorder = recorder is None
        if recorder is None:
            recorder = SupervisorHeartbeatRecorder(
                context.logger,
                run=context.run_label,
                interval_seconds=context.heartbeat_interval_seconds,
                resource_sampler=context.resource_sampler,
                clock=context.monotonic_clock,
                utc_clock=context.utc_clock,
            )
        telemetry_cleanup_seconds = 0.0
        stage_timings: list[StageTimingSummary] = []

        try:
            validate_manifest_contract(manifest_path)
            append_manifest_event(
                manifest_path,
                {
                    "event": context.run_start_event,
                    "run": context.run_label,
                    "stage_count": len(plan),
                    **context.run_metadata,
                },
                run_id=run_id,
                config_sha=config_sha,
            )

            failed_steps: list[str] = []
            first_failure: Exception | None = None
            degraded_steps: list[str] = []
            stage_health_states: dict[str, str] = {}
            iterator: Iterable[StagePlanItem] = plan
            if context.use_progress and len(plan) > 1:
                from tqdm import tqdm

                iterator = tqdm(plan, desc=context.progress_desc)

            for index, item in enumerate(iterator):
                stage_started_at = float(context.monotonic_clock())
                stage_started_utc = context.utc_clock()
                durations = {
                    "stage_start_publication": 0.0,
                    "action": 0.0,
                    "output_validation": 0.0,
                    "completion_read": 0.0,
                    "authentication": 0.0,
                    "stage_end_publication": 0.0,
                }
                status = "aborted"
                should_break = False
                scope = recorder.begin_scope(
                    f"{run_id}:{index}:{item.name}",
                    run=context.run_label,
                    stage=item.name,
                    phase="stage_start_publication",
                    state="publishing",
                )
                context.logger.info(
                    "Stage start: %s/%s",
                    context.run_label,
                    item.name,
                    extra={"run": context.run_label, "step": item.name},
                )
                try:
                    phase_started = float(context.monotonic_clock())
                    try:
                        append_manifest_event(
                            manifest_path,
                            {
                                "event": context.stage_start_event,
                                "run": context.run_label,
                                "stage": item.name,
                                **item.metadata,
                            },
                            run_id=run_id,
                            config_sha=config_sha,
                        )
                    finally:
                        durations["stage_start_publication"] += _elapsed(
                            context.monotonic_clock,
                            phase_started,
                        )

                    try:
                        scope.update(phase="action", state="working")
                        phase_started = float(context.monotonic_clock())
                        try:
                            with use_supervisor_recorder(recorder, scope):
                                item.action(context.config)
                        finally:
                            durations["action"] += _elapsed(
                                context.monotonic_clock,
                                phase_started,
                            )

                        scope.update(phase="output_validation", state="authenticating")
                        phase_started = float(context.monotonic_clock())
                        try:
                            missing_outputs = [
                                path for path in item.required_outputs if not path.exists()
                            ]
                        finally:
                            durations["output_validation"] += _elapsed(
                                context.monotonic_clock,
                                phase_started,
                            )
                        if missing_outputs:
                            raise StageValidationError(item.name, missing_outputs)

                        scope.update(phase="completion_read", state="authenticating")
                        phase_started = float(context.monotonic_clock())
                        try:
                            stage_done = (
                                read_stage_done(item.completion_stamp)
                                if item.completion_stamp is not None
                                else {"status": "success"}
                            )
                        finally:
                            durations["completion_read"] += _elapsed(
                                context.monotonic_clock,
                                phase_started,
                            )
                        stage_status = str(stage_done.get("status", "success"))

                        if item.completion_stamp is not None:
                            scope.update(
                                phase="completion_authentication",
                                state="authenticating",
                            )
                            phase_started = float(context.monotonic_clock())
                            try:
                                completion_state = resolve_stage_state(
                                    item.completion_stamp,
                                    inputs=[],
                                    outputs=item.required_outputs,
                                    cfg=context.config,
                                    stage=item.name,
                                    freshness_key=item.freshness_key,
                                )
                            finally:
                                durations["authentication"] += _elapsed(
                                    context.monotonic_clock,
                                    phase_started,
                                )
                            if completion_state is not CompletionState.COMPLETE_VALID:
                                failure_status = (
                                    completion_state.value
                                    if stage_status == "success"
                                    else stage_status
                                )
                                raise StageCompletionError(
                                    item.name,
                                    item.completion_stamp,
                                    failure_status,
                                    stage_done.get("reason"),
                                )

                        stage_health = "healthy"
                        stage_health_states[item.name] = stage_health
                        status = stage_status
                        stage_end_payload: dict[str, object] = {
                            "event": context.stage_end_event,
                            "run": context.run_label,
                            "stage": item.name,
                            "ok": stage_health == "healthy",
                            "status": stage_status,
                            "health": stage_health,
                            "reason": stage_done.get("reason"),
                            "blocking_dependency": stage_done.get("blocking_dependency"),
                            "upstream_stage": stage_done.get("upstream_stage"),
                        }
                    except Exception as exc:  # noqa: BLE001
                        status = "failed"
                        failed_steps.append(item.name)
                        first_failure = first_failure or exc
                        context.logger.exception(
                            "Stage failed: %s/%s",
                            context.run_label,
                            item.name,
                            extra={
                                "run": context.run_label,
                                "step": item.name,
                                "error": exc,
                            },
                        )
                        stage_end_payload = {
                            "event": context.stage_end_event,
                            "run": context.run_label,
                            "stage": item.name,
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            **(
                                {
                                    "missing_outputs": [
                                        str(path) for path in exc.missing_outputs
                                    ],
                                }
                                if isinstance(exc, StageValidationError)
                                else {}
                            ),
                            **(
                                {
                                    "completion_stamp": str(exc.stamp),
                                    "completion_status": exc.status,
                                }
                                if isinstance(exc, StageCompletionError)
                                else {}
                            ),
                        }
                        should_break = not context.continue_on_error

                    scope.update(phase="stage_end_publication", state="publishing")
                    phase_started = float(context.monotonic_clock())
                    try:
                        append_manifest_event(
                            manifest_path,
                            stage_end_payload,
                            run_id=run_id,
                            config_sha=config_sha,
                        )
                    finally:
                        durations["stage_end_publication"] += _elapsed(
                            context.monotonic_clock,
                            phase_started,
                        )
                finally:
                    cleanup_started = float(context.monotonic_clock())
                    scope_summary = scope.finish(status=status)
                    cleanup_seconds = _elapsed(context.monotonic_clock, cleanup_started)
                    stage_ended_at = float(context.monotonic_clock())
                    stage_timings.append(
                        StageTimingSummary(
                            stage=item.name,
                            status=status,
                            started_utc=_utc_text(stage_started_utc),
                            ended_utc=_utc_text(context.utc_clock()),
                            elapsed_seconds=max(0.0, stage_ended_at - stage_started_at),
                            stage_start_publication_seconds=durations[
                                "stage_start_publication"
                            ],
                            action_seconds=durations["action"],
                            output_validation_seconds=durations["output_validation"],
                            completion_read_seconds=durations["completion_read"],
                            authentication_seconds=durations["authentication"],
                            stage_end_publication_seconds=durations[
                                "stage_end_publication"
                            ],
                            cleanup_seconds=cleanup_seconds,
                            heartbeat_count=scope_summary.heartbeat_count,
                            resource_start=scope_summary.resource_start,
                            resource_end=scope_summary.resource_end,
                            telemetry_error=scope_summary.telemetry_error,
                        )
                    )
                if should_break:
                    break

            run_end_payload = {
                "event": context.run_end_event,
                "run": context.run_label,
                "ok": not failed_steps and not degraded_steps,
                "health": "healthy" if (not failed_steps and not degraded_steps) else "degraded",
                **context.run_end_metadata,
            }
            if failed_steps:
                run_end_payload["failed_steps"] = failed_steps
            if degraded_steps:
                run_end_payload["degraded_steps"] = degraded_steps
                run_end_payload["stage_health"] = stage_health_states
            append_manifest_event(
                manifest_path,
                run_end_payload,
                run_id=run_id,
                config_sha=config_sha,
            )

            if owns_recorder:
                telemetry_cleanup_seconds = recorder.close()
                owns_recorder = False
            result = StageRunResult(
                failed_steps=failed_steps,
                first_failure=first_failure,
                stage_timings=tuple(stage_timings),
                telemetry_cleanup_seconds=telemetry_cleanup_seconds,
            )
            if failed_steps:
                context.logger.error(
                    "Stage run completed with failures: %s",
                    failed_steps,
                    extra={"run": context.run_label, "failed_steps": failed_steps},
                )
                if raise_on_failure and first_failure is not None:
                    raise first_failure
            else:
                context.logger.info("Stage run complete", extra={"run": context.run_label})
            return result
        finally:
            if owns_recorder:
                recorder.close()


__all__ = [
    "StageCompletionError",
    "StagePlanItem",
    "StageRunContext",
    "StageRunResult",
    "StageRunner",
    "StageTimingSummary",
    "StageValidationError",
]
