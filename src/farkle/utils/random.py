# src/farkle/utils/random.py
"""Random number generator helpers."""

from __future__ import annotations

import importlib
import random

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


def seed_everything(seed: int) -> None:
    """Best-effort seeding across common RNG systems."""

    random.seed(seed)
    np.random.seed(seed)
    _ = np.random.default_rng(seed)

    try:  # optional dependency; ignore if unavailable
        torch = importlib.import_module("torch")

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CUDA not in CI
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional / CUDA absence
        pass

    try:  # optional dependency; ignore if unavailable
        tf = importlib.import_module("tensorflow")

        try:
            tf.random.set_seed(seed)
        except Exception:
            pass
        try:
            keras_utils = getattr(tf.keras, "utils", None)
            if keras_utils and hasattr(keras_utils, "set_random_seed"):
                keras_utils.set_random_seed(seed)
        except Exception:
            pass
    except Exception:  # pragma: no cover - tensorflow optional
        pass


__all__ = ["MAX_UINT32", "make_rng", "seed_everything", "spawn_seeds"]
