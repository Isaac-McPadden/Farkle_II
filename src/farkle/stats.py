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
    """Return games per strategy for desired power."""

    if not 0 < base_p < 1:
        raise ValueError("base_p must be in (0,1)")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    if base_p + delta >= 1:
        raise ValueError("base_p + delta must be < 1")

    if method == "bonferroni" and n_strategies <= 1:
        raise ValueError("bonferroni adjustment requires more than one strategy")
    
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
