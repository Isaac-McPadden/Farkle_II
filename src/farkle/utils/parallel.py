"""Parallel execution helpers used by simulations.

This module centralises small utilities for spawning deterministic
random seeds and running work in a :class:`ProcessPoolExecutor`.
The helpers are intentionally lightweight so unit tests can patch or
stub them easily.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, Iterator, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def spawn_worker_seeds(seed: int, n: int) -> list[int]:
    """Return ``n`` 32-bit seeds derived from ``seed``.

    The function relies on :func:`numpy.random.default_rng` to generate a
    reproducible sequence of unsigned 32-bit integers that can be used as
    independent seeds for worker processes.
    """

    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**32 - 1, size=n, dtype=np.uint32).tolist()


def process_map(
    fn: Callable[[T], T] | Callable[[T], object],
    items: Sequence[T],
    *,
    n_jobs: int | None = None,
    initializer: Callable[..., object] | None = None,
    initargs: Sequence[object] | None = None,
) -> Iterator[object]:
    """Yield ``fn(item)`` for each *item* using a process pool.

    Parameters
    ----------
    fn:
        Callable executed in the worker processes.
    items:
        Work items passed one by one to ``fn``.
    n_jobs:
        Maximum number of worker processes. ``None`` lets
        :class:`ProcessPoolExecutor` decide based on the CPU count.
    initializer, initargs:
        Forwarded to :class:`ProcessPoolExecutor` to allow initialisation of
        per-process state.
    """

    if initargs is None:
        initargs = ()
    with ProcessPoolExecutor(
        max_workers=n_jobs, initializer=initializer, initargs=tuple(initargs)
    ) as pool:
        futures = [pool.submit(fn, item) for item in items]
        for fut in as_completed(futures):
            yield fut.result()
