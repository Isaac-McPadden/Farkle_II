"""Utility subpackage for Farkle.

This package collects small helpers that are shared by the
simulation and analysis pipelines.  Functions are organised into
focused modules such as :mod:`files`, :mod:`parallel` and
:mod:`logging` so that the core game logic remains free of side
effects like file I/O or multiprocessing.

The most commonly used helpers are re-exported here for convenience.
"""

from __future__ import annotations

from .logging import setup_info_logging, setup_warning_logging
from .random import MAX_UINT32, make_rng, spawn_seeds

__all__ = [
    "setup_info_logging",
    "setup_warning_logging",
    "MAX_UINT32",
    "make_rng",
    "spawn_seeds",
]


def __getattr__(name: str):  # pragma: no cover - simple dynamic loader
    if name == "simulate_many_games_stream":
        from .parallel import simulate_many_games_stream as fn

        return fn
    raise AttributeError(name)
