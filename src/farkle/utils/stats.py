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
from math import ceil, sqrt
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm

from farkle.utils.random import MAX_UINT32


def _num_hypotheses(n: int, full_pairwise: bool) -> int:
    return n * (n - 1) // 2 if full_pairwise else n - 1


def _per_test_level(method: str, m: int, control: float, use_BY: bool) -> float:
    """
    control = alpha (Bonferroni/FWER) or q (BH/FDR).
    Returns the planning per-test level α* (or q* surrogate).
    """
    if not (0 < control < 1):
        raise ValueError("control must be in (0,1)")
    if method == "bonferroni":
        return control / m
    if use_BY:
        H_m = sum(1.0 / i for i in range(1, m + 1))
        return control / (m * H_m)       # BY surrogate (more conservative)
    return control / m             # BH planning surrogate (power-friendlier)


def games_for_power(
    *,
    n_strategies: int = 7140,
    k_players: int = 2,
    method: str = "bh",          # "bh" or "bonferroni"
    power: float = 0.8,
    control: float = 0.1,           # q for BH, alpha for Bonferroni
    detectable_lift: float = 0.03,
    baseline_rate: float = 0.5,
    tail: str = "two_sided",
    full_pairwise: bool = True,     # m = n(n-1)/2 if True else n-1
    use_BY: bool = False,           # only meaningful for BH, makes BH more conservative
    min_games_floor: int | None = None,
    max_games_cap: int | None = None,
) -> int:
    """
    Returns the required number of GAMES PER STRATEGY (rounded up) for a
    k-player round-robin style experiment, using BH false discovery rate (FDR) 
    or Bonferroni family-wise error rate (FWER).

    Notes:
    H0 is strategies have identical winrates.  H1 is strategies have different winrates.
      - Converts per-pair co-appearance requirement to k-player games by dividing by (k-1).
      - For BH, 'control' is FDR q; for Bonferroni, 'control' is FWER alpha.
      - 'tail' controls whether α*/2 or α* is used inside the normal quantile.
    Calculate the number of games needed for each strategy.

    Parameters
    ----------
    n_strategies : int
        Total number of strategies included in the experiment.
    k_players : int
        Game-match size, e.g. 5 players
    method : str
        "bh" or "bonferroni"
    power : float, default 0.8
        Target statistical power for each comparison.
    control : float
        For BH, 'control' is FDR q; for Bonferroni, 'control' is FWER alpha.
    detectable_lift : float, default 0.03
        Smallest detectable difference in win probability between two
        strategies.
    baseline_rate : float, default 0.5
        Baseline probability of winning against which ``delta`` is
        measured. Game count is maximized at 0.5.
    tail : str
        "two_sided" or "one_sided".  "two_sided" adds more games but detects better 
        and worse while "one_sided" can only detect better 
    full_pairwise :
        ``True`` → compare every pair of strategies (k = *n*·(*n*-1)/2).
        ``False`` → compare each strategy only to a single baseline (*n*-1 tests).
    use_BY : bool
        Accounts for dependencies between data where BH assumes independent outcomes.
        Method is more conservative and results in a higher game count.
        Defaults to False
    min_games_floor : int
       Manual override available if needed.
       Forces each strategy to play at least ``min_games_floor`` games.
    max_games_cap : int
        Manual override available if needed.
        Forces each strategy to play at most ``max_games_cap`` games.
        
    Returns
    -------
    int
        Number of games required per strategy (rounded up to the next
        integer).
    """
    # ---- validation ----
    if n_strategies <= 1: 
        raise ValueError("n_strategies must be > 1")
    if k_players   <  2:  
        raise ValueError("k_players must be >= 2")
    if not (0 < power < 1):            
        raise ValueError("power must be in (0,1)")
    if not (0 < baseline_rate < 1):    
        raise ValueError("baseline_rate must be in (0,1)")
    if not (0 < detectable_lift < 1):  
        raise ValueError("detectable_lift must be in (0,1)")
    if baseline_rate + detectable_lift >= 1:
        raise ValueError("baseline_rate + detectable_lift must be < 1")
    if tail not in {"one_sided", "two_sided"}:
        raise ValueError("tail must be 'one_sided' or 'two_sided'")

    # ---- hypotheses & per-test level ----
    m = _num_hypotheses(n_strategies, full_pairwise)
    if m < 1:
        raise ValueError("No hypotheses implied; check n_strategies/full_pairwise")

    alpha_star = _per_test_level(method=method, m=m, control=control, use_BY=(use_BY if method=="bh" else False))

    # ---- z-quantiles with tailing ----
    z_alpha = norm.ppf(1 - (alpha_star / 2.0 if tail == "two_sided" else alpha_star))
    z_beta  = norm.ppf(power)

    # ---- two-proportion per-pair (per arm) ----
    p1, p2 = baseline_rate, baseline_rate + detectable_lift
    pbar = 0.5 * (p1 + p2)
    num  = z_alpha * sqrt(2 * pbar * (1 - pbar)) + z_beta * sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    n_arm = (num / detectable_lift) ** 2   # required co-appearances per pair, per arm

    # ---- convert to k-player games per strategy ----
    games_per_strategy = ceil(n_arm * (n_strategies - 1) / (k_players - 1))

    # ---- floor/cap ----
    if min_games_floor is not None:
        games_per_strategy = max(games_per_strategy, int(min_games_floor))
    if max_games_cap   is not None:
        games_per_strategy = min(games_per_strategy, int(max_games_cap))

    return int(games_per_strategy)


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
