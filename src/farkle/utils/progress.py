# src/farkle/utils/progress.py
"""Shared helpers for time-based progress logging."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

__all__ = ["ProgressLogConfig", "ScheduledProgressLogger"]


@dataclass(frozen=True, slots=True)
class ProgressLogConfig:
    """Schedule controlling how often progress is logged."""

    frequent_interval_sec: float = 30.0
    info_phase_sec: float = 180.0
    ongoing_interval_sec: float = 600.0

    def normalized(self) -> "ProgressLogConfig":
        """Return a non-negative copy suitable for runtime scheduling."""

        return ProgressLogConfig(
            frequent_interval_sec=max(0.0, float(self.frequent_interval_sec)),
            info_phase_sec=max(0.0, float(self.info_phase_sec)),
            ongoing_interval_sec=max(0.0, float(self.ongoing_interval_sec)),
        )


def _advance_deadline(current: float, interval: float, now: float) -> float:
    """Advance a log deadline forward until it lies after the current time.

    Args:
        current: Current scheduled deadline.
        interval: Interval between successive deadlines.
        now: Current monotonic timestamp.

    Returns:
        Next deadline after ``now``, or ``inf`` when the interval is disabled.
    """
    if interval <= 0:
        return math.inf
    deadline = current
    while deadline <= now:
        deadline += interval
    return deadline


def _format_duration(seconds: float | None) -> str:
    """Format a duration in seconds for progress-log display.

    Args:
        seconds: Duration in seconds, or ``None`` when unknown.

    Returns:
        Human-readable duration string.
    """
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _format_count(value: float | int) -> str:
    """Format a count-like value for progress-log display.

    Args:
        value: Integer or float count value.

    Returns:
        Human-readable count string with separators when appropriate.
    """
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _format_rate(value: float, unit: str) -> str:
    """Format a per-second rate for progress-log display.

    Args:
        value: Per-second rate value.
        unit: Unit label associated with the rate.

    Returns:
        Human-readable rate string.
    """
    if not math.isfinite(value) or value <= 0:
        return f"rate unknown {unit}/s"
    if value >= 100:
        return f"rate {value:,.0f} {unit}/s"
    if value >= 10:
        return f"rate {value:,.1f} {unit}/s"
    return f"rate {value:,.2f} {unit}/s"


class ScheduledProgressLogger:
    """Emit progress logs on a fast-then-slow schedule."""

    def __init__(
        self,
        logger: logging.Logger,
        *,
        label: str,
        schedule: ProgressLogConfig,
        unit: str = "items",
        total: float | int | None = None,
        start_time: float | None = None,
    ) -> None:
        self._logger = logger
        self._label = label
        self._schedule = schedule.normalized()
        self._unit = unit
        self._total = total
        self._start = time.monotonic() if start_time is None else float(start_time)
        self._next_frequent = (
            self._start + self._schedule.frequent_interval_sec
            if self._schedule.frequent_interval_sec > 0
            else math.inf
        )
        self._next_ongoing_info = (
            self._start + self._schedule.ongoing_interval_sec
            if self._schedule.ongoing_interval_sec > 0
            else math.inf
        )

    def maybe_log(
        self,
        completed: float | int,
        *,
        total: float | int | None = None,
        unit: str | None = None,
        detail: str | None = None,
        extra: dict[str, object] | None = None,
        force_info: bool = False,
    ) -> bool:
        """Log progress when the configured schedule says it is due."""

        now = time.monotonic()
        due_frequent = now >= self._next_frequent
        due_ongoing = now >= self._next_ongoing_info
        if not (force_info or due_frequent or due_ongoing):
            return False

        elapsed = max(0.0, now - self._start)
        in_info_phase = elapsed <= self._schedule.info_phase_sec
        level = logging.INFO if (force_info or due_ongoing or in_info_phase) else logging.DEBUG
        resolved_total = self._total if total is None else total
        resolved_unit = self._unit if unit is None else unit
        message = self._build_message(
            completed=completed,
            total=resolved_total,
            unit=resolved_unit,
            elapsed=elapsed,
            detail=detail,
        )
        self._logger.log(level, message, extra=extra)

        if due_frequent:
            self._next_frequent = _advance_deadline(
                self._next_frequent,
                self._schedule.frequent_interval_sec,
                now,
            )
        if due_ongoing:
            self._next_ongoing_info = _advance_deadline(
                self._next_ongoing_info,
                self._schedule.ongoing_interval_sec,
                now,
            )
        return True

    def _build_message(
        self,
        *,
        completed: float | int,
        total: float | int | None,
        unit: str,
        elapsed: float,
        detail: str | None,
    ) -> str:
        """Assemble the formatted progress-log message body.

        Args:
            completed: Completed work units.
            total: Optional total work units.
            unit: Unit label for the progress values.
            elapsed: Elapsed wall-clock time in seconds.
            detail: Optional detail suffix to append.

        Returns:
            Formatted progress-log message string.
        """
        parts = [f"{self._label} progress:"]
        if total is not None and total > 0:
            progress_fraction = max(0.0, min(1.0, float(completed) / float(total)))
            rate = float(completed) / elapsed if elapsed > 0 else math.nan
            remaining = max(0.0, float(total) - float(completed))
            eta = (remaining / rate) if rate > 0 and math.isfinite(rate) else None
            parts.append(
                f"{_format_count(completed)}/{_format_count(total)} {unit} "
                f"({progress_fraction * 100.0:.1f}%)"
            )
            parts.append(f"elapsed {_format_duration(elapsed)}")
            parts.append(f"eta {_format_duration(eta)}")
            parts.append(_format_rate(rate, unit))
        else:
            rate = float(completed) / elapsed if elapsed > 0 else math.nan
            parts.append(f"{_format_count(completed)} {unit}")
            parts.append(f"elapsed {_format_duration(elapsed)}")
            parts.append(_format_rate(rate, unit))
        if detail:
            parts.append(detail)
        return ", ".join(parts)
