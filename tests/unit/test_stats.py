import pytest

from farkle.stats import games_for_power


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
