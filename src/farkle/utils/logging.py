# src/farkle/utils/logging.py
"""Configure console/file root logging with a shared format and clean handler replacement."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

_STANDARD_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_SUMMARY_FIELDS = frozenset(
    {
        "total_units",
        "completed_units",
        "reused_units",
        "requested_workers",
        "effective_workers",
        "attempt_count",
        "retry_downshifted",
        "completed_games",
        "planned_games",
        "completed_chunks",
        "planned_chunks",
        "completed_blocks",
        "total_blocks",
        "selected_groups",
        "candidate_groups",
        "completion_status",
        "root_seed",
        "current_player_count",
        "safety_limit_games",
        "nonviable_blocks",
        "reconciled_from",
    }
)


class _StructuredFormatter(logging.Formatter):
    """Render attached operational fields before the standard exception traceback."""

    def __init__(self, fmt: str, *, datefmt: str, verbose_telemetry: bool = False) -> None:
        super().__init__(fmt, datefmt=datefmt)
        self._verbose_telemetry = verbose_telemetry

    def formatMessage(self, record: logging.LogRecord) -> str:
        message = super().formatMessage(record)
        fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_FIELDS
        }
        if not self._verbose_telemetry and record.levelno == logging.INFO:
            kind = fields.get("telemetry_kind")
            if kind == "heartbeat":
                return message  # The heartbeat message already renders current progress.
            if kind == "completion_summary":
                fields["progress"] = {
                    key: value
                    for key, value in fields["progress"].items()
                    if key in _SUMMARY_FIELDS
                }
            elif kind == "telemetry_cleanup":
                completed = fields.pop("completed_progress", {})
                fields["completed_scope_count"] = len(completed)
                fields.pop("resource_summary", None)
            elif kind == "timing_summary":
                fields["finalization_phases"] = [
                    {
                        key: phase[key]
                        for key in ("scope", "status", "elapsed_seconds")
                        if key in phase
                    }
                    for phase in fields["finalization_phases"]
                ]
        if fields:
            message += " | " + json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str)
        return message


def configure_logging(*, level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Configure root logging once.

    Parameters
    ----------
    level:
        Logging level as string (e.g., "INFO") or numeric (e.g., logging.INFO).
    log_file:
        Optional file to tee logs to. Parent dirs are created and UTF-8 is used.
    """
    if isinstance(level, str):
        level = cast(int, getattr(logging, level.upper(), logging.INFO))

    handlers: list[logging.Handler] = [logging.StreamHandler()]  # default: stderr
    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p, encoding="utf-8"))

    formatter = _StructuredFormatter(
        "%(asctime)s [%(process)d] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        verbose_telemetry=level <= logging.DEBUG,
    )
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,  # ensure a clean config when re-running in notebooks/CLIs
    )


# Back-compat convenience wrappers (used in some tests)
def setup_info_logging(log_file: Path | None = None) -> None:
    """Configure INFO-level logging, optionally teeing to ``log_file``."""
    configure_logging(level="INFO", log_file=log_file)


def setup_warning_logging(log_file: Path | None = None) -> None:
    """Configure WARNING-level logging, optionally teeing to ``log_file``."""
    configure_logging(level="WARNING", log_file=log_file)


__all__ = ["configure_logging", "setup_info_logging", "setup_warning_logging"]
