# src/farkle/stats.py
from math import ceil
from scipy.stats import norm

def games_for_power(M: int,        # #strategies being compared
                    alpha=.05,     # type-I error (FWER controlled by Bonferroni)
                    power=.9,      # 1 – type-II
                    p1=.25, p2=.30 # smallest win-rate diff worth detecting
                    ) -> int:
    """Return games per strategy for a two-proportion test."""
    z_alpha = norm.ppf(1 - alpha/(2*M))          # Bonferroni
    z_beta  = norm.ppf(power)
    p_bar   = 0.5*(p1+p2)
    n = (z_alpha* (2*p_bar*(1-p_bar))**0.5 + z_beta*((p1*(1-p1)+p2*(1-p2)))**0.5)**2 / (p2-p1)**2
    return ceil(n)
