# src/farkle/analysis/stage_runner.py
"""Reusable stage runner with manifest logging."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from farkle.analysis.stage_state import read_stage_done, stage_done_path
from farkle.config import AppConfig
from farkle.utils.manifest import append_manifest_line

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class StagePlanItem:
    """One stage entry for the runner."""

    name: str
    action: Callable[[AppConfig], None]
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    required_outputs: Sequence[Path] = dataclasses.field(default_factory=tuple)


class StageValidationError(RuntimeError):
    """Raised when a stage completes without required artifacts."""

    def __init__(self, stage: str, missing_outputs: Sequence[Path]):
        self.stage = stage
        self.missing_outputs = tuple(missing_outputs)
        missing_text = ", ".join(str(path) for path in self.missing_outputs)
        super().__init__(f"Stage {stage!r} missing required outputs: {missing_text}")


@dataclasses.dataclass(frozen=True)
class StageRunContext:
    """Configuration for stage execution and manifest logging."""

    config: AppConfig
    manifest_path: Path
    run_label: str
    run_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    run_end_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    run_start_event: str = "run_start"
    run_end_event: str = "run_end"
    stage_start_event: str = "stage_start"
    stage_end_event: str = "stage_end"
    continue_on_error: bool = False
    use_progress: bool = False
    progress_desc: str = "pipeline"
    logger: logging.Logger = LOGGER


@dataclasses.dataclass(frozen=True)
class StageRunResult:
    """Summary of stage execution."""

    failed_steps: Sequence[str]
    first_failure: Exception | None


class StageRunner:
    """Execute stage plans while recording results to a manifest."""

    @staticmethod
    def run(
        plan: Sequence[StagePlanItem],
        context: StageRunContext,
        *,
        raise_on_failure: bool = True,
    ) -> StageRunResult:
        manifest_path = context.manifest_path
        run_payload = {
            "event": context.run_start_event,
            "run": context.run_label,
            "stage_count": len(plan),
            **context.run_metadata,
        }
        append_manifest_line(manifest_path, run_payload)

        failed_steps: list[str] = []
        first_failure: Exception | None = None
        degraded_steps: list[str] = []
        stage_health_states: dict[str, str] = {}
        iterator: Iterable[StagePlanItem] = plan
        if context.use_progress and len(plan) > 1:
            from tqdm import tqdm

            iterator = tqdm(plan, desc=context.progress_desc)
        for item in iterator:
            context.logger.info(
                "Stage start: %s/%s",
                context.run_label,
                item.name,
                extra={"run": context.run_label, "step": item.name},
            )
            append_manifest_line(
                manifest_path,
                {
                    "event": context.stage_start_event,
                    "run": context.run_label,
                    "stage": item.name,
                    **item.metadata,
                },
            )
            try:
                item.action(context.config)
                missing_outputs = [path for path in item.required_outputs if not path.exists()]
                if missing_outputs:
                    raise StageValidationError(item.name, missing_outputs)
                try:
                    done_path = stage_done_path(context.config.stage_dir(item.name), item.name)
                except Exception:  # noqa: BLE001
                    done_path = None
                stage_done = read_stage_done(done_path) if done_path is not None else {"status": "success"}
                stage_status = str(stage_done.get("status", "success"))
                stage_health = "healthy"
                if stage_status == "skipped":
                    stage_health = "degraded"
                    degraded_steps.append(item.name)
                elif stage_status == "failed":
                    stage_health = "unhealthy"
                    degraded_steps.append(item.name)
                stage_health_states[item.name] = stage_health
                append_manifest_line(
                    manifest_path,
                    {
                        "event": context.stage_end_event,
                        "run": context.run_label,
                        "stage": item.name,
                        "ok": stage_health == "healthy",
                        "status": stage_status,
                        "health": stage_health,
                        "reason": stage_done.get("reason"),
                        "blocking_dependency": stage_done.get("blocking_dependency"),
                        "upstream_stage": stage_done.get("upstream_stage"),
                    },
                )
            except Exception as exc:  # noqa: BLE001
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
                append_manifest_line(
                    manifest_path,
                    {
                        "event": context.stage_end_event,
                        "run": context.run_label,
                        "stage": item.name,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        **(
                            {
                                "missing_outputs": [str(path) for path in exc.missing_outputs],
                            }
                            if isinstance(exc, StageValidationError)
                            else {}
                        ),
                    },
                )
                if not context.continue_on_error:
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
        append_manifest_line(manifest_path, run_end_payload)

        if failed_steps:
            context.logger.error(
                "Stage run completed with failures: %s",
                failed_steps,
                extra={"run": context.run_label, "failed_steps": failed_steps},
            )
            if raise_on_failure and first_failure is not None:
                raise first_failure
        else:
            context.logger.info(
                "Stage run complete", extra={"run": context.run_label}
            )

        return StageRunResult(failed_steps=failed_steps, first_failure=first_failure)
