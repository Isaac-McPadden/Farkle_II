# src/farkle/utils/parallel.py
"""Parallel execution helpers used by simulations.

Small, testable utilities for seeding workers and mapping work with a
ProcessPoolExecutor. Keep simulation-specific logic outside utils.
"""

from __future__ import annotations

import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_map(fn, items, *, n_jobs=None, initializer=None, initargs=None, window=0):
    """Map ``fn`` across ``items`` with optional multiprocessing support."""
    if initargs is None:
        initargs = ()
    if n_jobs in (None, 0, 1):
        # Single-process path: still run initializer so modules relying on
        # per-process globals (e.g., run_tournament._STATE) are set up.
        if initializer is not None:
            initializer(*tuple(initargs))
        for it in items:
            yield fn(it)
        return
    if window <= 0:
        window = (n_jobs or 1) * 4

    with ProcessPoolExecutor(
        max_workers=n_jobs, initializer=initializer, initargs=tuple(initargs)
    ) as pool:
        it = iter(items)
        futs = []
        # prefill the window
        for _ in range(window):
            try:
                futs.append(pool.submit(fn, next(it)))
            except StopIteration:
                break
        while futs:
            done = next(as_completed(futs))
            futs.remove(done)
            yield done.result()
            with contextlib.suppress(StopIteration):
                futs.append(pool.submit(fn, next(it)))


__all__ = ["process_map"]
