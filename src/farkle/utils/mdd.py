# src/farkle/utils/mdd.py
"""
Variance decomposition and minimum detectable difference helpers.

These utilities estimate binomial, seed-level, and strategy-by-player-count
variance components from per-cell win-rate data and produce a defensible MDD
threshold on the probability scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class VarianceComponents:
    """Container describing estimated variance components."""

    tau2_seed: float
    tau2_sxk: Optional[float] = None
    binom_by_k: Optional[pd.Series] = None
    R: int = 0
    K: int = 0


def _ensure_winrate(
    df: pd.DataFrame,
    wins_col: Optional[str],
    games_col: Optional[str],
    winrate_col: Optional[str],
    use_jeffreys: bool = True,
) -> pd.Series:
    """
    Return per-row win-rates.

    If ``winrate_col`` is present it is trusted; otherwise wins/games are used,
    optionally with a Jeffreys prior.
    """
    if winrate_col is not None and winrate_col in df.columns:
        return df[winrate_col].astype(float)

    if wins_col is None or games_col is None:
        raise ValueError("Provide winrate_col, or both wins_col and games_col.")

    w = df[wins_col].astype(float)
    n = df[games_col].astype(float)
    if use_jeffreys:
        return (w + 0.5) / (n + 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = w / n
    return out.fillna(0.0)


def prepare_cell_means(
    df: pd.DataFrame,
    *,
    strategy_col: str = "strategy",
    k_col: str = "player_count",
    seed_col: str = "seed",
    wins_col: Optional[str] = "wins",
    games_col: Optional[str] = "games",
    winrate_col: Optional[str] = None,
    use_jeffreys: bool = True,
) -> pd.DataFrame:
    """
    Collapse raw data to per-(strategy, k, seed) win-rates and games.
    """
    df = df.copy()
    df["__winrate__"] = _ensure_winrate(
        df,
        wins_col=wins_col,
        games_col=games_col,
        winrate_col=winrate_col,
        use_jeffreys=use_jeffreys,
    )

    if games_col is not None and games_col in df.columns:
        df["__games__"] = df[games_col].astype(float)
    else:
        df["__games__"] = 1.0

    gb = df.groupby([strategy_col, k_col, seed_col], as_index=False)
    out = gb.agg(winrate=("__winrate__", "mean"), games=("__games__", "sum"))

    out = out.rename(columns={strategy_col: "strategy", k_col: "k", seed_col: "seed"})
    return out[["strategy", "k", "seed", "winrate", "games"]]


def estimate_tau2_seed(
    cell: pd.DataFrame,
    *,
    robust: bool = True,
) -> VarianceComponents:
    """
    Estimate across-seed variance component on the probability scale.
    """
    sk = cell.groupby(["strategy", "k"], as_index=False).agg(
        mean_p=("winrate", "mean"),
        var_across_seed=("winrate", "var"),
        mean_games=("games", "mean"),
        R=("seed", "nunique"),
    )
    sk["var_across_seed"] = sk["var_across_seed"].fillna(0.0)

    sk["v_binom"] = sk["mean_p"] * (1.0 - sk["mean_p"]) / sk["mean_games"].clip(lower=1.0)
    sk["tau2_seed_sk"] = (sk["var_across_seed"] - sk["v_binom"]).clip(lower=0.0)

    tau2_seed = sk["tau2_seed_sk"].median() if robust else sk["tau2_seed_sk"].mean()
    binom_by_k = sk.groupby("k")["v_binom"].median()
    binom_by_k.index = pd.Index(binom_by_k.index.astype(int), name="k")

    R = int(cell["seed"].nunique())
    K = int(cell["k"].nunique())

    return VarianceComponents(
        tau2_seed=float(tau2_seed),
        tau2_sxk=None,
        binom_by_k=binom_by_k,
        R=R,
        K=K,
    )


def estimate_tau2_sxk(
    cell: pd.DataFrame,
    tau2_seed: float,
    *,
    weights_by_k: Optional[Dict[int, float]] = None,
    robust: bool = True,
) -> float:
    """
    Estimate strategy-by-player-count heterogeneity (tau^2_{s×k}).
    """
    sk = cell.groupby(["strategy", "k"], as_index=False).agg(
        mean_p=("winrate", "mean"),
        mean_games=("games", "mean"),
        R=("seed", "nunique"),
    )

    s = sk.groupby("strategy").agg(var_across_k=("mean_p", "var")).reset_index()
    s["var_across_k"] = s["var_across_k"].fillna(0.0)

    if weights_by_k is None:
        weights_by_k = {int(k): 1.0 for k in sk["k"].unique()}
    wsum = float(sum(weights_by_k.values()))
    weights = {int(k): v / wsum for k, v in weights_by_k.items()}

    # Per-(s,k) noise feeding into the weighted mean across k:
    sk["v_binom_seeded"] = (
        sk["mean_p"] * (1 - sk["mean_p"]) / sk["mean_games"].clip(lower=1.0)
    ) / sk["R"]
    sk["v_seed_only"] = tau2_seed / sk["R"]
    sk["_w"] = sk["k"].map(weights).astype(float)
    sk["noise_term"] = (sk["_w"] ** 2) * (sk["v_binom_seeded"] + sk["v_seed_only"])

    # Use squared weights for variance propagation
    noise_by_s = sk.groupby("strategy", as_index=False)["noise_term"].sum()
    noise_by_s = noise_by_s.rename(columns={"noise_term": "noise_into_k"})
    noise_by_s["noise_into_k"] = noise_by_s["noise_into_k"].astype(float)

    s = s.merge(noise_by_s, on="strategy", how="left")
    s["tau2_sxk_s"] = (s["var_across_k"] - s["noise_into_k"]).clip(lower=0.0)

    tau2_sxk = s["tau2_sxk_s"].median() if robust else s["tau2_sxk_s"].mean()
    return float(tau2_sxk)


def compute_mdd_for_tiers(
    *,
    tau2_seed: float,
    tau2_sxk: float = 0.0,
    binom_by_k: pd.Series,
    weights_by_k: Optional[Dict[int, float]] = None,
    R: int,
    z_star: float = 2.0,
) -> float:
    """
    Compute the pairwise minimum detectable difference on the probability scale.
    """
    if binom_by_k is None or len(binom_by_k) == 0:
        raise ValueError("binom_by_k must contain at least one entry")

    # Default to equal weights across all k present
    if weights_by_k is None:
        weights_by_k = {int(k): 1.0 for k in binom_by_k.index}

    # Normalize weights
    wsum = float(sum(weights_by_k.values()))
    weights = {int(k): (v / wsum) for k, v in weights_by_k.items()}
    if not weights:
        raise ValueError("weights_by_k produced no entries")

    # Ensure binom_by_k has an Int64Index so .loc[int(k)] is safe
    if not pd.api.types.is_integer_dtype(binom_by_k.index):
        binom_by_k = pd.Series(
            binom_by_k.values,
            index=pd.Index(
                [int(k) for k in binom_by_k.index], dtype="int64", name=binom_by_k.index.name
            ),
            name=binom_by_k.name,
        )

    # Validate that all keys in weights exist in binom_by_k
    missing = [k for k in weights if k not in binom_by_k.index]
    if missing:
        raise ValueError(
            f"weights_by_k contains k values not present in binom_by_k: {sorted(missing)}. "
            f"Known k values: {sorted(map(int, binom_by_k.index.tolist()))}"
        )

    # Squared-weights variance propagation
    sum_w2 = sum(wk**2 for wk in weights.values())
    v_binom = sum((wk**2) * (float(binom_by_k.loc[k]) / R) for k, wk in weights.items())
    v_seed = tau2_seed / R
    v_sxk = tau2_sxk * sum_w2

    var_theta = v_binom + v_seed + v_sxk
    se_diff = np.sqrt(2.0 * var_theta)
    mdd = z_star * se_diff
    return float(mdd)


def tiering_ingredients_from_df(
    df: pd.DataFrame,
    *,
    strategy_col: str = "strategy",
    k_col: str = "player_count",
    seed_col: str = "seed",
    wins_col: Optional[str] = "wins",
    games_col: Optional[str] = "games",
    winrate_col: Optional[str] = None,
    use_jeffreys: bool = True,
    weights_by_k: Optional[Dict[int, float]] = None,
    z_star: float = 2.0,
) -> Dict[str, object]:
    """
    High-level helper returning variance components, tau2_sxk, and MDD.
    """
    cell = prepare_cell_means(
        df,
        strategy_col=strategy_col,
        k_col=k_col,
        seed_col=seed_col,
        wins_col=wins_col,
        games_col=games_col,
        winrate_col=winrate_col,
        use_jeffreys=use_jeffreys,
    )

    comps = estimate_tau2_seed(cell, robust=True)

    tau2_sxk = estimate_tau2_sxk(
        cell,
        comps.tau2_seed,
        weights_by_k=weights_by_k,
        robust=True,
    )

    # Ensure binom_by_k exists
    if comps.binom_by_k is None:
        raise ValueError("estimate_tau2_seed did not return binom_by_k")

    # Populate the dataclass so callers can rely on it
    comps.tau2_sxk = tau2_sxk

    mdd = compute_mdd_for_tiers(
        tau2_seed=comps.tau2_seed,
        tau2_sxk=tau2_sxk,
        binom_by_k=comps.binom_by_k,
        weights_by_k=weights_by_k,
        R=comps.R,
        z_star=z_star,
    )

    return {
        "cell": cell,
        "components": comps,
        "tau2_sxk": tau2_sxk,
        "mdd": mdd,
    }


__all__ = [
    "VarianceComponents",
    "prepare_cell_means",
    "estimate_tau2_seed",
    "estimate_tau2_sxk",
    "compute_mdd_for_tiers",
    "tiering_ingredients_from_df",
]
