# src/farkle/stats.py

from math import ceil, sqrt

from scipy.stats import norm


def games_for_power(
    n_strategies: int,
    delta: float = 0.03,
    base_p: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
    method: str = "bh",  # "bh" or "bonferroni"
    pairwise: bool = True,  # baseline vs all or full pairwise
) -> int:
    """Return games per strategy for desired power."""

    # per-test alpha*
    if method == "bonferroni":
        n_tests = n_strategies * (n_strategies - 1) // 2 if pairwise else n_strategies - 1
        alpha_star = alpha / n_tests
    elif method == "bh":
        h_m = sum(1 / i for i in range(1, n_strategies + 1))  # harmonic number
        alpha_star = alpha / h_m
    else:
        raise ValueError("method must be 'bh' or 'bonferroni'")

    z_alpha = norm.ppf(1 - alpha_star / 2)
    z_beta = norm.ppf(power)

    p1, p2 = base_p, base_p + delta
    pooled_p = (p1 + p2) / 2
    numerator = z_alpha * sqrt(2 * pooled_p * (1 - pooled_p)) + z_beta * sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )
    n = (numerator / delta) ** 2
    return ceil(n)
