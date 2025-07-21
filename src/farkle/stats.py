# src/farkle/stats.py

from math import sqrt

from scipy.stats import norm


def games_for_power(
    n_strategies: int,
    delta: float = 0.03,
    base_p: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
    method: str = "bh",  # "bh" or "bonferroni"
    pairwise: bool = True # baseline vs all or full pairwise
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
    pairwise : bool, default ``True``
        If ``True`` perform a full pairwise analysis where each strategy
        is compared with every other.  If ``False`` only compare each
        strategy against a single baseline.

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
    if method == "bonferroni":
        k = n_strategies * (n_strategies-1) // 2 if pairwise else n_strategies-1
        alpha_star = alpha / k
    elif method == "bh":
        h_m = sum(1/i for i in range(1, n_strategies+1))  # harmonic number
        alpha_star = alpha / h_m
    else:
        raise ValueError("method must be 'bh' or 'bonferroni'")

    z_alpha = norm.ppf(1 - alpha_star/2)
    z_beta = norm.ppf(power)

    p1, p2 = base_p, base_p + delta
    pbar = (p1 + p2)/2
    numerator = z_alpha*sqrt(2*pbar*(1-pbar)) + z_beta*sqrt(p1*(1-p1)+p2*(1-p2))
    n = (numerator / delta)**2
    return int(n) + 1
