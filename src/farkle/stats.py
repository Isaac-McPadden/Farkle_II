# src/farkle/stats.py
from __future__ import annotations

from math import ceil, sqrt
import warnings

from scipy.stats import norm


def games_for_power(
    n_strategies: int,
    delta: float = 0.03,
    base_p: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
    method: str = "bh",          # "bh" or "bonferroni"
    full_pairwise: bool = True,  # baseline-vs-all or full pairwise
    *,
    pairwise: bool | None = None,  # deprecated alias
) -> int:
    """
    Return the games-per-strategy required to detect a win-rate difference
    ``delta`` with the requested power after multiple-test adjustment.

    Parameters
    ----------
    full_pairwise :
        ``True`` → compare every pair of strategies (k = *n*·(*n*-1)/2).
        ``False`` → compare each strategy only to a single baseline (*n*-1 tests).
    pairwise :
        **Deprecated** alias for ``full_pairwise``.  Will be removed in a future
        version.
    """

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

    # ------------------ per-test alpha* -----------------------------------
    if method == "bonferroni":
        k = (
            n_strategies * (n_strategies - 1) // 2
            if full_pairwise
            else n_strategies - 1
        )
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
