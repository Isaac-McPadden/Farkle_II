from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np
import pandas as pd


def build_tiers(mu: Dict[str, float], sigma: Dict[str, float], z: float = 2.326) -> Dict[str, int]:
    """Group strategies into overlapping confidence tiers."""
    items = sorted(mu.items(), key=lambda kv: kv[1], reverse=True)
    tiers: Dict[str, int] = {}
    if not items:
        return tiers
    tier = 1
    lower = mu[items[0][0]] - z * sigma[items[0][0]]
    tiers[items[0][0]] = tier
    for name, _ in items[1:]:
        l = mu[name] - z * sigma[name]
        u = mu[name] + z * sigma[name]
        if u < lower:
            tier += 1
            lower = l
        else:
            lower = min(lower, l)
        tiers[name] = tier
    return tiers


def bh_correct(pvals: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """Return a boolean mask for Benjamini–Hochberg FDR control."""
    p = np.asarray(pvals)
    order = np.argsort(p)
    ranks = np.arange(1, len(p) + 1)
    crit = alpha * ranks / len(p)
    passed = p[order] <= crit
    thresh = p[order][passed].max() if passed.any() else 0.0
    return p <= thresh


def bonferroni_pairs(strats: List[str], games_needed: int, seed: int) -> pd.DataFrame:
    """Generate a schedule for Bonferroni-corrected head-to-head games."""
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in itertools.combinations(strats, 2):
        seeds = rng.integers(0, 2**32 - 1, size=games_needed)
        for sd in seeds:
            rows.append({"a": a, "b": b, "seed": int(sd)})
    return pd.DataFrame(rows)