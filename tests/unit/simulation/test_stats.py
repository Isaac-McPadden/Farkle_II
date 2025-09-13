from math import ceil, sqrt
from typing import Any

import pytest
from scipy.stats import norm

from farkle.simulation.stats import games_for_power


@pytest.mark.parametrize(
    "params",
    [
        {"n_strategies": 1},
        {"delta": 0},
        {"delta": 1},
        {"base_p": 0},
        {"base_p": 1},
        {"alpha": 0},
        {"alpha": 1},
        {"power": 0},
        {"power": 1},
        {"method": "foo"},
    ],
)
def test_games_for_power_invalid(params):
    base: dict[str, Any] = {"n_strategies": 2}
    base.update(params)
    with pytest.raises(ValueError):
        games_for_power(**base)


def test_bh_vs_bonferroni():
    n_bh = games_for_power(n_strategies=3, method="bh")
    n_bonf = games_for_power(n_strategies=3, method="bonferroni")
    assert n_bh < n_bonf


def test_games_for_power_rounding():
    n = 2
    delta = 0.03
    base_p = 0.5
    alpha = 0.05
    power = 0.8
    h_m = sum(1 / i for i in range(1, n + 1))
    alpha_star = alpha / h_m
    z_alpha = norm.ppf(1 - alpha_star / 2)
    z_beta = norm.ppf(power)
    p1, p2 = base_p, base_p + delta
    pbar = (p1 + p2) / 2
    numerator = z_alpha * sqrt(2 * pbar * (1 - pbar)) + z_beta * sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    expected = ceil((numerator / delta) ** 2)
    assert games_for_power(n_strategies=n, delta=delta) == expected


def test_bonferroni_requires_multiple_strategies():
    with pytest.raises(ValueError):
        games_for_power(n_strategies=1, method="bonferroni")


def test_base_p_out_of_range():
    for base_p in [0, 1, -0.1, 1.1]:
        with pytest.raises(ValueError):
            games_for_power(n_strategies=2, base_p=base_p)


def test_delta_out_of_range():
    for delta in [0, 1, -0.2, 2.0]:
        with pytest.raises(ValueError):
            games_for_power(n_strategies=2, delta=delta)


def test_base_p_plus_delta_too_large():
    with pytest.raises(ValueError):
        games_for_power(n_strategies=2, base_p=0.7, delta=0.4)
