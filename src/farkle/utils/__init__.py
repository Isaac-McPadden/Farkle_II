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

from .logging import configure_logging, setup_info_logging, setup_warning_logging
from .mdd import (
    VarianceComponents,
    compute_mdd_for_tiers,
    estimate_tau2_seed,
    estimate_tau2_sxk,
    prepare_cell_means,
    tiering_ingredients_from_df,
)
from .random import MAX_UINT32, make_rng, spawn_seeds

__all__ = [
    "configure_logging",
    "setup_info_logging",
    "setup_warning_logging",
    "MAX_UINT32",
    "make_rng",
    "spawn_seeds",
    "VarianceComponents",
    "prepare_cell_means",
    "estimate_tau2_seed",
    "estimate_tau2_sxk",
    "compute_mdd_for_tiers",
    "tiering_ingredients_from_df",
]
