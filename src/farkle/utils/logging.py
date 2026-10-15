# src/farkle/utils/logging.py
"""Configure console/file root logging with a shared format and clean handler replacement."""

from __future__ import annotations

import json
import logging
from pathlib import Path

_STANDARD_RECORD_FIELDS = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class _StructuredFormatter(logging.Formatter):
    """Render attached operational fields before the standard exception traceback."""

    def formatMessage(self, record: logging.LogRecord) -> str:
        message = super().formatMessage(record)
        fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_RECORD_FIELDS
        }
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
        level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]  # default: stderr
    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p, encoding="utf-8"))

    formatter = _StructuredFormatter(
        "%(asctime)s [%(process)d] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
