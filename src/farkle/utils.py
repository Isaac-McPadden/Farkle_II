from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd


def build_tiers(mu: Dict[str, float], sigma: Dict[str, float], z: float = 2.326) -> Dict[str, int]:
    """Group strategies into overlapping confidence tiers."""
    sorted_items = sorted(mu.items(), key=lambda kv: kv[1], reverse=True)
    tier_map: Dict[str, int] = {}
    if not sorted_items:
        return tier_map
    current_tier = 1
    current_lower = mu[sorted_items[0][0]] - z * sigma[sorted_items[0][0]]
    tier_map[sorted_items[0][0]] = current_tier
    for name, _ in sorted_items[1:]:
        lower_bound = mu[name] - z * sigma[name]
        upper_bound = mu[name] + z * sigma[name]
        if upper_bound < current_lower:
            current_tier += 1
            current_lower = lower_bound
        else:
            current_lower = min(current_lower, lower_bound)
        tier_map[name] = current_tier
    return tier_map


def bh_correct(pvals: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """Return a boolean mask for Benjamini–Hochberg FDR control."""
    pvals_array = np.asarray(pvals)
    sorted_indices = np.argsort(pvals_array)
    ranks = np.arange(1, len(pvals_array) + 1)
    critical_values = alpha * ranks / len(pvals_array)
    passed_mask = pvals_array[sorted_indices] <= critical_values
    threshold = pvals_array[sorted_indices][passed_mask].max() if passed_mask.any() else 0.0
    return pvals_array <= threshold


def bonferroni_pairs(strats: List[str], games_needed: int, seed: int) -> pd.DataFrame:
    """Generate a schedule for Bonferroni-corrected head-to-head games."""
    random_generator = np.random.default_rng(seed)
    schedule_rows = []
    for strat_a, strat_b in itertools.combinations(strats, 2):
        random_seeds = random_generator.integers(0, 2**32 - 1, size=games_needed)
        for game_seed in random_seeds:
            schedule_rows.append({"a": strat_a, "b": strat_b, "seed": int(game_seed)})
    return pd.DataFrame(schedule_rows)
