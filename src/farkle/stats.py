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
    """Calculate the number of games needed for each strategy.

    Parameters
    ----------
    n_strategies : int
        Total number of strategies included in the experiment.
    delta : float, default 0.03
        Smallest detectable difference in win probability between two
        strategies.
    base_p : float, default 0.5
        Baseline probability of winning against which ``delta`` is
        measured.
    alpha : float, default 0.05
        Desired family wise error rate.
    power : float, default 0.8
        Target statistical power for each comparison.
    method : {{'bh', 'bonferroni'}}, default ``'bh'``
        Multiple comparison correction to apply.
    full_pairwise :
        ``True`` → compare every pair of strategies (k = *n*·(*n*-1)/2).
        ``False`` → compare each strategy only to a single baseline (*n*-1 tests).
    pairwise :
        **Deprecated** alias for ``full_pairwise``.  Will be removed in a future
        version.

    Returns
    -------
    int
        Number of games required per strategy (rounded up to the next
        integer).

    Raises
    ------
    ValueError
        If ``method`` is not ``'bh'`` or ``'bonferroni'``.

    Examples
    --------
    >>> games_for_power(n_strategies=3, delta=0.2, method='bh')
    111
    """
    
    # per-test alpha*


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
