"""Logging helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List


def _setup_logging(level: str, log_file: Path | None = None) -> None:
    """Configure the root logger exactly once."""

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(process)d] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def setup_info_logging(log_file: Path | None = None) -> None:
    """INFO-level console/file logging."""

    _setup_logging("INFO", log_file)


def setup_warning_logging(log_file: Path | None = None) -> None:
    """WARNING-level console/file logging (quieter default)."""

    _setup_logging("WARNING", log_file)


__all__ = ["setup_info_logging", "setup_warning_logging"]

