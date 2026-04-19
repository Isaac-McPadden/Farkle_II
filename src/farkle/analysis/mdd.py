"""Analysis-facing assembly for frequentist ranking MDD inputs.

This module owns the frequentist-ranking orchestration that turns raw isolated
metrics into per-cell summaries, variance components, and the final minimum
detectable difference used for human-readable frequentist tier bands.
"""

from __future__ import annotations

import pandas as pd

from farkle.utils.mdd_helpers import (
    VarianceComponents,
    compute_mdd_for_tiers,
    estimate_tau2_seed,
    estimate_tau2_sxk,
    prepare_cell_means,
)


def frequentist_ingredients_from_df(
    df: pd.DataFrame,
    *,
    strategy_col: str = "strategy",
    k_col: str = "player_count",
    seed_col: str = "seed",
    wins_col: str | None = "wins",
    games_col: str | None = "games",
    winrate_col: str | None = None,
    use_jeffreys: bool = True,
    weights_by_k: dict[int, float] | None = None,
    z_star: float = 2.0,
) -> dict[str, object]:
    """Return frequentist variance components, ``tau2_sxk``, and ``mdd``."""

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

    if comps.binom_by_k is None:
        raise ValueError("estimate_tau2_seed did not return binom_by_k")

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


__all__ = ["VarianceComponents", "frequentist_ingredients_from_df"]
