"""Simple logging configuration helpers."""
from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(*, level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Configure root logging.

    Parameters
    ----------
    level:
        Logging level as string or numeric constant. Defaults to ``"INFO"``.
    log_file:
        Optional path to a log file. When provided, messages are written to the
        file in addition to stderr.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(str(log_file)))
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(asctime)s [%(process)d] %(levelname)s %(name)s: %(message)s",
    )
