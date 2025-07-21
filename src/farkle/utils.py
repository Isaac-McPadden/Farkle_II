from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd

# Max unsigned 32-bit integer for random seed generation
MAX_UINT32 = 2**32 - 1


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

    if set(means) != set(stdevs):                         # extra safety check
        raise ValueError("means and stdevs must have identical strategy keys")

    sorted_items = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    tier_map: Dict[str, int] = {}
    if not sorted_items:                                  # fast-exit for empty
        return tier_map

    current_tier = 1
    current_lower = means[sorted_items[0][0]] - z * stdevs[sorted_items[0][0]]
    tier_map[sorted_items[0][0]] = current_tier

    for name, _ in sorted_items[1:]:
        lower = means[name] - z * stdevs[name]
        upper = means[name] + z * stdevs[name]
        if upper < current_lower:                         # strict separation
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


def bonferroni_pairs(strats: List[str], games_needed: int, seed: int) -> pd.DataFrame:
    """
    Create a deterministic schedule of head-to-head games for every strategy
    pair, assigning a unique RNG seed to each game.

    Parameters
    ----------
    strats :
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
    if games_needed < 0:
        raise ValueError("games_needed must be non-negative")

    random_generator = np.random.default_rng(seed)
    schedule_rows = []
    for strat_a, strat_b in itertools.combinations(strats, 2):
        random_seeds = random_generator.integers(0, MAX_UINT32, size=games_needed)
        for game_seed in random_seeds:
            schedule_rows.append({"a": strat_a, "b": strat_b, "seed": int(game_seed)})
    return pd.DataFrame(schedule_rows)
