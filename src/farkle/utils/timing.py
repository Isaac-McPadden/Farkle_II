"""Timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator


@contextmanager
def time_block(description: str, log: Callable[[str], None] | None = None) -> Iterator[float]:
    """Context manager that measures execution time.

    Parameters
    ----------
    description:
        Label for the timed block which will be included in the log message.
    log:
        Optional callable to receive a formatted message.  Defaults to
        :func:`print` when ``None``.
    """

    start = time.perf_counter()
    yield start
    elapsed = time.perf_counter() - start
    if log is None:
        log = print
    log(f"{description}: {elapsed:.6f} s")


__all__ = ["time_block"]
