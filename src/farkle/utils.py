from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd


def build_tiers(
    means: Dict[str, float],
    stdevs: Dict[str, float],
    z: float = 2.326,
) -> Dict[str, int]:
    """Group strategies into overlapping confidence tiers.

    Parameters
    ----------
    means:
        Mapping from strategy name to estimated mean performance.
    stdevs:
        Mapping from strategy name to the corresponding standard deviation.
    z:
        Z-score for the desired confidence level, defaulting to 2.326 (approx.
        99% one-sided).

    Returns
    -------
    Dict[str, int]
        Mapping from strategy name to a tier index starting at 1.

    Edge Cases
    ----------
    If ``means`` is empty, the returned dictionary is empty.

    Examples
    --------
    >>> build_tiers({'A': 100.0, 'B': 99.0}, {'A': 0.5, 'B': 0.5})
    {'A': 1, 'B': 1}
    """

    sorted_items = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    tier_map: Dict[str, int] = {}
    if not sorted_items:
        return tier_map
    current_tier = 1
    current_lower = means[sorted_items[0][0]] - z * stdevs[sorted_items[0][0]]
    tier_map[sorted_items[0][0]] = current_tier
    for name, _ in sorted_items[1:]:
        lower_bound = means[name] - z * stdevs[name]
        upper_bound = means[name] + z * stdevs[name]
        if upper_bound < current_lower:
            current_tier += 1
            current_lower = lower_bound
        else:
            current_lower = min(current_lower, lower_bound)
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
    threshold = pvals_array[sorted_indices][passed_mask].max() if passed_mask.any() else 0.0
    return pvals_array <= threshold


def bh_correct(pvals: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """Backward-compatible alias for :func:`benjamini_hochberg`."""
    return benjamini_hochberg(pvals, alpha)


def bonferroni_pairs(strats: List[str], games_needed: int, seed: int) -> pd.DataFrame:
    """Generate a schedule for Bonferroni-corrected head-to-head games.

    Parameters
    ----------
    strats:
        List of strategy names.
    games_needed:
        Number of games to schedule per strategy pair.
    seed:
        Seed for the random number generator to ensure determinism.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``"a"``, ``"b"`` and ``"seed"`` describing the schedule.

    Edge Cases
    ----------
    If fewer than two strategies are provided, an empty :class:`pandas.DataFrame`
    is returned.

    Examples
    --------
    >>> df = bonferroni_pairs(["S1", "S2"], games_needed=1, seed=0)
    >>> df.shape[0]
    1
    """

    random_generator = np.random.default_rng(seed)
    schedule_rows = []
    for strat_a, strat_b in itertools.combinations(strats, 2):
        random_seeds = random_generator.integers(0, 2**32 - 1, size=games_needed)
        for game_seed in random_seeds:
            schedule_rows.append({"a": strat_a, "b": strat_b, "seed": int(game_seed)})
    return pd.DataFrame(schedule_rows)
