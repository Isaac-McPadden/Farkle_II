# src/farkle/utils/stats.py
"""Utility helpers for statistical analysis and scheduling.

This module exposes a small collection of functions that support the
simulation framework:

* ``games_for_power`` computes the number of games needed for statistical power.
* ``build_tiers`` groups strategies into overlapping confidence tiers.
* ``benjamini_hochberg``/``bh_correct`` perform false discovery rate control.
* ``bonferroni_pairs`` creates deterministic head-to-head schedules with RNG
  seeds.

The constant :data:`~farkle.utils.random.MAX_UINT32` is re-exported for
convenience.
"""

from __future__ import annotations

import itertools
import warnings
from math import ceil, sqrt
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm

from .random import MAX_UINT32


def games_for_power(
    n_strategies: int,
    delta: float = 0.03,
    base_p: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
    method: str = "bh",  # "bh" or "bonferroni"
    full_pairwise: bool = True,  # baseline-vs-all or full pairwise
    *,
    pairwise: bool | None = None,  # deprecated alias
) -> int:
    """Calculate the number of games needed for each strategy.

    Parameters
    ----------
    n_strategies : int
        Total number of strategies included in the experiment.
    delta : float, default 0.03
        Smallest detectable difference in win probability between two
        strategies.
    base_p : float, default 0.5
        Baseline probability of winning against which ``delta`` is
        measured.
    alpha : float, default 0.05
        Desired family wise error rate.
    power : float, default 0.8
        Target statistical power for each comparison.
    method : {{'bh', 'bonferroni'}}, default ``'bh'``
        Multiple comparison correction to apply.
    full_pairwise :
        ``True`` → compare every pair of strategies (k = *n*·(*n*-1)/2).
        ``False`` → compare each strategy only to a single baseline (*n*-1 tests).
    pairwise :
        **Deprecated** alias for ``full_pairwise``.  Will be removed in a future
        version.

    Returns
    -------
    int
        Number of games required per strategy (rounded up to the next
        integer).

    Raises
    ------
    ValueError
        If ``method`` is not ``'bh'`` or ``'bonferroni'``.

    Examples
    --------
    >>> games_for_power(n_strategies=3, delta=0.2, method='bh')
    111
    """
    # per-test alpha*

    # ------------------ handle deprecated alias ---------------------------
    if pairwise is not None:
        warnings.warn(
            "`pairwise` is deprecated; use `full_pairwise` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        full_pairwise = pairwise

    # ------------------ argument validation -------------------------------
    if not 0 < base_p < 1:
        raise ValueError("base_p must be in (0, 1)")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if base_p + delta >= 1:
        raise ValueError("base_p + delta must be < 1")
    if method == "bonferroni" and n_strategies <= 1:
        raise ValueError("bonferroni adjustment requires more than one strategy")
    if n_strategies <= 1:
        raise ValueError("n_strategies must be > 1")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")
    if not (0 < power < 1):
        raise ValueError("power must be between 0 and 1")

    # ------------------ per-test alpha* -----------------------------------
    if method == "bonferroni":
        k = n_strategies * (n_strategies - 1) // 2 if full_pairwise else n_strategies - 1
        alpha_star = alpha / k
    elif method == "bh":
        h_m = sum(1 / i for i in range(1, n_strategies + 1))  # harmonic number
        alpha_star = alpha / h_m
    else:
        raise ValueError("method must be 'bh' or 'bonferroni'")

    # ------------------ sample-size formula -------------------------------
    z_alpha = norm.ppf(1 - alpha_star / 2)
    z_beta = norm.ppf(power)

    p1, p2 = base_p, base_p + delta
    p_bar = (p1 + p2) / 2
    numerator = z_alpha * sqrt(2 * p_bar * (1 - p_bar)) + z_beta * sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )
    n = (numerator / delta) ** 2

    return ceil(n)  # always round *up* to the next whole game


def build_tiers(
    means: Dict[str, float],
    stdevs: Dict[str, float],
    z: float = 2.326,
) -> Dict[str, int]:
    """
    Group strategies into **overlapping confidence tiers**.

    Parameters
    ----------
    means :
        Mapping ``strategy → estimated mean performance``.
    stdevs :
        Mapping ``strategy → sample σ`` (must have the same keys as *means*).
    z :
        One-sided *z*-score for the desired confidence level.
        Default **2.326** ≈ 99 % one-sided (α ≈ 0.01).

    Returns
    -------
    Dict[str, int]
        ``strategy → tier`` starting at 1 (1 = top tier).

    Notes
    -----
    Strategies are first sorted by mean.  A new tier begins when the
    *upper* confidence bound of the current strategy falls below the *lowest*
    lower-bound seen so far.

    Examples
    --------
    >>> build_tiers({'A': 100.0, 'B': 99.0}, {'A': 0.5, 'B': 0.5})
    {'A': 1, 'B': 1}
    """

    if set(means) != set(stdevs):  # extra safety check
        raise ValueError("means and stdevs must have identical strategy keys")

    sorted_items = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    tier_map: Dict[str, int] = {}
    if not sorted_items:  # fast-exit for empty
        return tier_map

    current_tier = 1
    current_lower = means[sorted_items[0][0]] - z * stdevs[sorted_items[0][0]]
    tier_map[sorted_items[0][0]] = current_tier

    for name, _ in sorted_items[1:]:
        lower = means[name] - z * stdevs[name]
        upper = means[name] + z * stdevs[name]
        if upper < current_lower:  # strict separation
            current_tier += 1
            current_lower = lower
        else:
            current_lower = min(current_lower, lower)
        tier_map[name] = current_tier
    return tier_map


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """Apply the Benjamini–Hochberg FDR procedure.

    Parameters
    ----------
    pvals:
        Array of p-values to correct.
    alpha:
        Desired false discovery rate. Defaults to ``0.02``.

    Returns
    -------
    np.ndarray
        Boolean array indicating which hypotheses pass the FDR threshold.

    Edge Cases
    ----------
    If ``pvals`` is empty, an empty boolean array is returned.

    Examples
    --------
    >>> benjamini_hochberg(np.array([0.01, 0.02, 0.2]), alpha=0.05).tolist()
    [True, True, False]
    """

    pvals_array = np.asarray(pvals)
    sorted_indices = np.argsort(pvals_array)
    ranks = np.arange(1, len(pvals_array) + 1)
    critical_values = alpha * ranks / len(pvals_array)
    passed_mask = pvals_array[sorted_indices] <= critical_values
    if not passed_mask.any():
        return np.full_like(pvals_array, False, dtype=bool)

    threshold = pvals_array[sorted_indices][passed_mask].max()
    return pvals_array <= threshold


def bh_correct(pvals: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """Backward-compatible alias for :func:`benjamini_hochberg`."""
    return benjamini_hochberg(pvals, alpha)


def bonferroni_pairs(strategies: List[str], games_needed: int, seed: int) -> pd.DataFrame:
    """
    Create a deterministic schedule of head-to-head games for every strategy
    pair, assigning a unique RNG seed to each game.

    Parameters
    ----------
    strategies :
        List of strategy names.
    games_needed :
        **Non-negative** number of games to schedule per pair.
    seed :
        Seed for the NumPy generator so the schedule is reproducible.

    Returns
    -------
    pandas.DataFrame
        Columns ``"a"``, ``"b"``, ``"seed"`` – one row per game.
    """
    if games_needed < 0:  # explicit sanity check
        raise ValueError("games_needed must be non-negative")

    random_generator = np.random.default_rng(seed)
    schedule_rows = []
    for strat_a, strat_b in itertools.combinations(strategies, 2):
        random_seeds = random_generator.integers(0, MAX_UINT32, size=games_needed)
        for game_seed in random_seeds:
            schedule_rows.append({"a": strat_a, "b": strat_b, "seed": int(game_seed)})
    return pd.DataFrame(schedule_rows)
