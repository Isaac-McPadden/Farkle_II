"""Low-level helpers for variance decomposition and MDD estimation.

These functions operate on per-cell win-rate data and are kept in ``utils``
because they are pure tabular/statistical transforms. Analysis-facing orchestration
that assembles frequentist inputs from artifacts lives under ``farkle.analysis``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    wins_col: str | None,
    games_col: str | None,
    winrate_col: str | None,
    use_jeffreys: bool = True,
) -> pd.Series:
    """Return per-row win-rates from a dataframe."""

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
    wins_col: str | None = "wins",
    games_col: str | None = "games",
    winrate_col: str | None = None,
    use_jeffreys: bool = True,
) -> pd.DataFrame:
    """Collapse raw data to per-``(strategy, k, seed)`` win-rates and games.

    Notes
    -----
    This helper averages the per-row win-rate values within each
    ``(strategy, k, seed)`` group and sums the per-row game counts. That is
    appropriate when each input row is already a fully aggregated cell, or when
    the grouped rows should contribute equally on the win-rate scale.

    It does *not* recompute the grouped win-rate as ``sum(wins) / sum(games)``.
    If future callers pass multiple rows per cell with materially different
    game counts, the resulting ``winrate`` column will be an arithmetic mean of
    row-level rates rather than a game-weighted pooled rate.
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
    """Estimate across-seed variance component on the probability scale."""

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
    weights_by_k: dict[int, float] | None = None,
    robust: bool = True,
) -> float:
    """Estimate strategy-by-player-count heterogeneity."""

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

    sk["v_binom_seeded"] = (
        sk["mean_p"] * (1 - sk["mean_p"]) / sk["mean_games"].clip(lower=1.0)
    ) / sk["R"]
    sk["v_seed_only"] = tau2_seed / sk["R"]
    sk["_w"] = sk["k"].map(weights).astype(float)
    sk["noise_term"] = (sk["_w"] ** 2) * (sk["v_binom_seeded"] + sk["v_seed_only"])

    noise_by_s = sk.groupby("strategy", as_index=False).agg(noise_into_k=("noise_term", "sum"))
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
    weights_by_k: dict[int, float] | None = None,
    R: int,
    z_star: float = 2.0,
) -> float:
    """Compute the pairwise minimum detectable difference on the probability scale."""

    if binom_by_k is None or len(binom_by_k) == 0:
        raise ValueError("binom_by_k must contain at least one entry")

    if weights_by_k is None:
        weights_by_k = {int(k): 1.0 for k in binom_by_k.index}

    wsum = float(sum(weights_by_k.values()))
    weights = {int(k): (v / wsum) for k, v in weights_by_k.items()}
    if not weights:
        raise ValueError("weights_by_k produced no entries")

    if not pd.api.types.is_integer_dtype(binom_by_k.index):
        binom_by_k = pd.Series(
            binom_by_k.values,
            index=pd.Index(
                [int(k) for k in binom_by_k.index],
                dtype="int64",
                name=binom_by_k.index.name,
            ),
            name=binom_by_k.name,
        )

    missing = [k for k in weights if k not in binom_by_k.index]
    if missing:
        raise ValueError(
            f"weights_by_k contains k values not present in binom_by_k: {sorted(missing)}. "
            f"Known k values: {sorted(map(int, binom_by_k.index.tolist()))}"
        )

    sum_w2 = sum(wk**2 for wk in weights.values())
    v_binom = sum((wk**2) * (float(binom_by_k.loc[k]) / R) for k, wk in weights.items())
    v_seed = tau2_seed / R
    v_sxk = tau2_sxk * sum_w2

    var_theta = v_binom + v_seed + v_sxk
    se_diff = np.sqrt(2.0 * var_theta)
    mdd = z_star * se_diff
    return float(mdd)


__all__ = [
    "VarianceComponents",
    "_ensure_winrate",
    "prepare_cell_means",
    "estimate_tau2_seed",
    "estimate_tau2_sxk",
    "compute_mdd_for_tiers",
]
