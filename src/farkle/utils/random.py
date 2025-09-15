"""Random number generator helpers."""

from __future__ import annotations

import numpy as np

# Max unsigned 32-bit integer for random seed generation.  Using this value
# keeps seeds compatible with languages like C/C++ that expect ``uint32``.
MAX_UINT32 = 2**32 - 1


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` seeded with *seed*."""

    return np.random.default_rng(seed)


def spawn_seeds(n: int, *, seed: int | None = None) -> np.ndarray:
    """Return ``n`` 32-bit seeds derived from ``seed``.

    The function relies on :func:`numpy.random.default_rng` to generate a
    reproducible sequence of unsigned 32-bit integers that can be used as
    independent seeds for worker processes.
    """

    rng = make_rng(seed)
    return rng.integers(0, MAX_UINT32, size=n, dtype=np.uint32)


__all__ = ["MAX_UINT32", "make_rng", "spawn_seeds"]

