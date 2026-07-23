# src/farkle/utils/__init__.py
"""Utility subpackage for Farkle.

This package collects small helpers that are shared by the
simulation and analysis pipelines.  Functions are organised into
focused modules such as :mod:`files`, :mod:`parallel` and
:mod:`logging` so that the core game logic remains free of side
effects like file I/O or multiprocessing.

The most commonly used helpers are re-exported here for convenience.
"""

from __future__ import annotations

from .analysis_shared import to_int, to_stat_value
from .logging import configure_logging, setup_info_logging, setup_warning_logging
from .random import (
    MAX_UINT32,
    PURPOSE_NAMESPACES,
    RNG_SCHEME_VERSION,
    RandomPurpose,
    coordinate_entropy,
    coordinate_rng,
    coordinate_seed,
    make_rng,
    spawn_seeds,
)

__all__ = [
    "to_int",
    "to_stat_value",
    "configure_logging",
    "setup_info_logging",
    "setup_warning_logging",
    "MAX_UINT32",
    "PURPOSE_NAMESPACES",
    "RNG_SCHEME_VERSION",
    "RandomPurpose",
    "coordinate_entropy",
    "coordinate_rng",
    "coordinate_seed",
    "make_rng",
    "spawn_seeds",
]
