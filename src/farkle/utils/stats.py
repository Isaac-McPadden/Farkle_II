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
import logging
from math import ceil, sqrt
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm

from farkle.utils.random import MAX_UINT32


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Return the Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    k :
        Number of observed successes. Must satisfy ``0 <= k <= n``.
    n :
        Total number of Bernoulli trials. Must be positive.
    alpha :
        Two-sided significance level. ``alpha=0.05`` yields a 95 % interval.

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` bounds clipped to ``[0, 1]`` with ``lower <= upper``.

    Raises
    ------
    ValueError
        If ``n <= 0``, ``k`` lies outside ``[0, n]``, or ``alpha`` is not in ``(0, 1)``.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n (inclusive)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    if n == 0:  # pragma: no cover - defensive (guard above already rejects)
        return 0.0, 1.0

    proportion = k / n
    z = norm.ppf(1.0 - alpha / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = proportion + z2 / (2.0 * n)
    margin = z * sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * n)) / n)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    lower = float(max(0.0, min(1.0, lower)))
    upper = float(max(0.0, min(1.0, upper)))
    if lower > upper:  # possible if numeric noise occurs
        lower = upper
    return lower, upper


def _num_hypotheses(n: int, full_pairwise: bool) -> int:
    """Count hypotheses under full pairwise or top-vs-rest testing."""
    return n * (n - 1) // 2 if full_pairwise else n - 1


def _per_test_level(
    method: str,
    m: int,
    control: float,  # q for BH, alpha for Bonferroni
    use_BY: bool,
    bh_target_rank: int | None = None,  # e.g., K (top-K pairs you want power for)
    bh_target_frac: float | None = None,  # e.g., 0.01 for top 1% of tests
) -> float:
    """
    Returns a planning per-test level α*:
      - Bonferroni: α*/test = alpha / m
      - BH:         α* ≈ (i*/m) * q, where i* is target rank (or fraction*m)
      - BY-BH:      α* ≈ (i*/m) * (q / H_m)
    """
    if not (0 < control < 1):
        raise ValueError("control must be in (0,1)")
    if m <= 0:
        raise ValueError("m must be positive")

    if method == "bonferroni":
        return control / m

    # --- BH (optionally BY-corrected) ---
    q = control
    c_m = sum(1.0 / i for i in range(1, m + 1)) if use_BY else 1.0

    # Choose a target rank i*:
    if bh_target_rank is not None:
        i_star = max(1, min(m, int(ceil(bh_target_rank))))
    elif bh_target_frac is not None:
        # ceil prevents small m from undershooting the requested fraction
        target = bh_target_frac * m
        i_star = max(1, min(m, int(ceil(target))))
    else:
        # Reasonable default if caller doesn't specify:
        #   - full_pairwise: power around the top 1% discoveries
        #   - baseline (m ~ n-1): power at the first discovery
        # You can override this default from the caller based on full_pairwise.
        i_star = max(1, int(round(0.01 * m)))

    return (i_star / m) * (q / c_m)


def games_for_power(
    *,
    n_strategies: int = 7140,
    k_players: int = 2,
    method: str = "bh",  # "bh" or "bonferroni"
    power: float = 0.8,
    control: float = 0.1,  # q for BH, alpha for Bonferroni
    detectable_lift: float = 0.03,  # absolute lift
    baseline_rate: float | None = None,
    tail: str = "two_sided",
    full_pairwise: bool = False,  # only used for endpoint="pairwise"
    use_BY: bool = False,  # BH-only; conservative
    min_games_floor: int | None = None,
    max_games_cap: int | None = None,
    bh_target_rank: int | None = None,
    bh_target_frac: float | None = None,
    endpoint: str = "top1",  # "pairwise" (A vs B) or "top1" (wins whole game)
) -> int:
    """
    Returns the required number of GAMES PER STRATEGY (rounded up) for a
    k-player round-robin style experiment, using BH false discovery rate (FDR)
    or Bonferroni family-wise error rate (FWER).

    Notes:
      - H0 is strategies have identical winrates.  H1 is strategies have different winrates.
      - Converts per-pair co-appearance requirement to k-player games by dividing by (k-1).
      - For BH, 'control' is FDR q; for Bonferroni, 'control' is FWER alpha.
      - 'tail' controls whether α*/2 or α* is used inside the normal quantile.
      - endpoint="pairwise" plans per-pair co-appearances via two-sample proportion sizing (baseline ~0.5),
        then converts to games by dividing by (k_players - 1).
      - endpoint="top1" plans directly in games per strategy via one-sample proportion sizing against
        p0 = 1/k_players (or given baseline_rate). No /(k-1) conversion.

    Function calculates the number of games needed for each strategy.

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
    bh_target_rank : int, optional
        Target order statistic ``i*`` for BH planning (e.g., power for top-K tests).
    bh_target_frac : float, optional
        Fractional version of the target rank (e.g., 0.03 focuses on top 3%).
        Method is more conservative and results in a higher game count.
        Defaults to False
    min_games_floor : int
       Manual override available if needed.
       Forces each strategy to play at least ``min_games_floor`` games.
    max_games_cap : int
        Manual override available if needed.
        Forces each strategy to play at most ``max_games_cap`` games.
    endpoint : str
        "pairwise" or "top1".  "pairwise" considers outcomes based on all players in
        a match.  "top1" only cares about the winner.  "pairwise" is extremely granular
        and better used after screening with "top1".
    Returns
    -------
    int
        Number of games required per strategy (rounded up to the next
        integer).
    """
    LOGGER = logging.getLogger(__name__)

    # ---- validation ----
    if n_strategies <= 1:
        raise ValueError("n_strategies must be > 1")
    if k_players < 2:
        raise ValueError("k_players must be >= 2")
    if not (0 < power < 1):
        raise ValueError("power must be in (0,1)")
    if isinstance(baseline_rate, float) and not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be in (0,1)")
    if not (0 < detectable_lift < 1):
        raise ValueError("detectable_lift must be in (0,1)")
    if isinstance(baseline_rate, float) and baseline_rate + detectable_lift >= 1:
        raise ValueError("baseline_rate + detectable_lift must be < 1")
    if tail not in {"one_sided", "two_sided"}:
        raise ValueError("tail must be 'one_sided' or 'two_sided'")

    # Endpoint-specific baseline default
    if endpoint == "pairwise":
        # default to 0.5 if not provided
        p0 = 0.5 if baseline_rate is None else baseline_rate
    else:  # endpoint == "top1"
        # default to 1/k if not provided
        p0 = (1.0 / k_players) if baseline_rate is None else baseline_rate
    # in games_for_power(), after computing p0 for top1
    if endpoint == "top1" and baseline_rate is not None:
        ideal = 1.0 / k_players
        if abs(baseline_rate - ideal) > 1e-6:
            LOGGER.warning(
                "top1 baseline_rate=%.6f differs from 1/k (%.6f) for k=%d; "
                "sample size may be miscalibrated. Set baseline_rate: null to use 1/k.",
                baseline_rate,
                ideal,
                k_players,
            )

    if not (0 < p0 < 1):
        raise ValueError("baseline_rate (effective p0) must be in (0,1)")
    if p0 + detectable_lift >= 1:
        raise ValueError("baseline_rate + detectable_lift must be < 1")

    # -------------------- number of hypotheses (m) ----------------------
    if endpoint == "pairwise":
        # full_pairwise => all pairs; else => each vs single baseline
        m = (n_strategies * (n_strategies - 1)) // 2 if full_pairwise else (n_strategies - 1)
    else:
        # one test per strategy: "is this strategy's top-1 win rate > p0?"
        m = n_strategies

    if m < 1:
        raise ValueError("No hypotheses implied; check inputs")

    # --- Resolve BH target defaults/overrides (after m is known) ---
    if method == "bh":
        if (bh_target_rank is not None) and (bh_target_frac is not None):
            LOGGER.info(
                "Both bh_target_rank (%s) and bh_target_frac (%.6g) supplied; "
                "ignoring rank and using fraction.",
                str(bh_target_rank),
                bh_target_frac,
            )
            bh_target_rank = None

        if (bh_target_rank is None) and (bh_target_frac is None):
            bh_target_frac = 0.01
            # i* uses ceil to avoid undershooting at small m; clamp to [1, m]
            i_star = max(1, min(m, int(ceil(bh_target_frac * m))))
            LOGGER.info(
                "No BH target supplied; defaulting to bh_target_frac=%.4f -> i*=%d of m=%d",
                bh_target_frac,
                i_star,
                m,
            )
        elif bh_target_frac is not None:
            i_star = max(1, min(m, int(ceil(bh_target_frac * m))))
            LOGGER.info(
                "BH using fraction: bh_target_frac=%.6g -> i*=%d of m=%d", bh_target_frac, i_star, m
            )
        else:
            # rank provided
            assert bh_target_rank is not None
            i_star = max(1, min(m, int(ceil(bh_target_rank))))
            LOGGER.info("BH using rank: bh_target_rank=%d of m=%d", i_star, m)
    # -------------------- per-test planning level α* --------------------
    alpha_star = _per_test_level(
        method=method,
        m=m,
        control=control,
        use_BY=(use_BY if method == "bh" else False),
        bh_target_rank=(bh_target_rank if method == "bh" else None),
        bh_target_frac=(bh_target_frac if method == "bh" else None),
    )
    alpha_for_z = (alpha_star / 2.0) if tail == "two_sided" else alpha_star

    z_alpha = norm.ppf(1.0 - alpha_for_z)
    z_beta = norm.ppf(power)

    # -------------------- endpoint-specific sizing ----------------------
    if endpoint == "pairwise":
        # Two-sample proportions (equal allocation), per-pair per-arm co-appearances
        p1, p2 = p0, p0 + detectable_lift
        pbar = 0.5 * (p1 + p2)
        numerator = z_alpha * sqrt(2.0 * pbar * (1.0 - pbar)) + z_beta * sqrt(
            p1 * (1.0 - p1) + p2 * (1.0 - p2)
        )
        n_arm_per_pair = (numerator / detectable_lift) ** 2

        # Convert co-appearances → games per strategy via /(k-1)
        games_per_strategy = ceil(n_arm_per_pair * (n_strategies - 1) / (k_players - 1))

    else:  # endpoint == "top1"
        # One-sample proportion vs known p0; each game yields one Bernoulli for strategy i
        p1 = p0 + detectable_lift
        numerator = z_alpha * sqrt(p0 * (1.0 - p0)) + z_beta * sqrt(p1 * (1.0 - p1))
        n_games_per_strategy = (numerator / (p1 - p0)) ** 2
        games_per_strategy = ceil(n_games_per_strategy)

    # -------------------- floors / caps --------------------
    if min_games_floor is not None:
        games_per_strategy = max(games_per_strategy, int(min_games_floor))
    if max_games_cap is not None:
        games_per_strategy = min(games_per_strategy, int(max_games_cap))

    LOGGER.info(
        "stats.py: endpoint=%s method=%s full_pairwise=%s | n=%d k=%d m=%d | "
        "control=%.6g tail=%s BY=%s | p0=%.6g p1=%.6g delta=%.6g | "
        "alpha*=%.6g alpha_for_z=%.6g z_alpha=%.4f z_beta=%.4f -> games/strategy=%d",
        endpoint,
        method,
        full_pairwise if endpoint == "pairwise" else False,
        n_strategies,
        k_players,
        m,
        control,
        tail,
        bool(use_BY) if method == "bh" else False,
        p0,
        (p0 + detectable_lift),
        detectable_lift,
        alpha_star,
        alpha_for_z,
        z_alpha,
        z_beta,
        games_per_strategy,
    )
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
