"""Parallel execution helpers used by simulations.

Small, testable utilities for seeding workers and mapping work with a
ProcessPoolExecutor. Keep simulation-specific logic outside utils.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, Iterator, Sequence, TypeVar
import numpy as np

T = TypeVar("T")
U = TypeVar("U")

def spawn_worker_seeds(master_seed: int, n: int) -> list[int]:
    """Derive ``n`` independent 32-bit seeds from ``master_seed``."""
    rng = np.random.default_rng(master_seed)
    return rng.integers(0, 2**32 - 1, size=n, dtype=np.uint32).tolist()

def process_map(
    fn: Callable[[T], U],
    items: Sequence[T] | Iterable[T],
    *,
    n_jobs: int | None = None,
    initializer: Callable[..., object] | None = None,
    initargs: Sequence[object] | None = None,
) -> Iterator[U]:
    """Yield ``fn(item)`` for each item using a process pool."""
    if initargs is None:
        initargs = ()
    with ProcessPoolExecutor(
        max_workers=n_jobs, initializer=initializer, initargs=tuple(initargs)
    ) as pool:
        futs = [pool.submit(fn, it) for it in items]
        for fut in as_completed(futs):
            yield fut.result()

__all__ = ["spawn_worker_seeds", "process_map"]
