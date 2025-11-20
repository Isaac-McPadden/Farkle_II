from math import ceil, sqrt
from typing import Any

from typing import Any

import pytest

from farkle.utils.stats import games_for_power


@pytest.mark.parametrize(
    "params",
    [
        {"n_strategies": 1},
        {"detectable_lift": 0},
        {"detectable_lift": 1},
        {"baseline_rate": 0},
        {"baseline_rate": 1},
        {"power": 0},
        {"power": 1},
        {"control": -0.1},
        {"k_players": 1},
        {"tail": "invalid"},
    ],
)
def test_games_for_power_invalid(params):
    base: dict[str, Any] = {"n_strategies": 2}
    base.update(params)
    with pytest.raises(ValueError):
        games_for_power(**base)


<<<<<<< ours
<<<<<<< ours
def test_games_for_power_invalid_method_defaults_to_bh():
    base: dict[str, Any] = {"n_strategies": 2}
    bh_games = games_for_power(**base)
    assert games_for_power(method="foo", **base) == bh_games


=======
>>>>>>> theirs
@pytest.mark.xfail(
    reason=(
        "Updated power sizing yields equal counts for bh vs bonferroni; "
        "see https://github.com/Isaac-McPadden/Farkle_II/issues/203"
    ),
    strict=False,
)
<<<<<<< ours
=======
>>>>>>> theirs
=======
>>>>>>> theirs
def test_bh_vs_bonferroni():
    n_bh = games_for_power(n_strategies=3, method="bh")
    n_bonf = games_for_power(n_strategies=3, method="bonferroni")
    assert n_bh < n_bonf


def test_games_for_power_monotonicity():
    small_delta = games_for_power(n_strategies=4, detectable_lift=0.01)
    large_delta = games_for_power(n_strategies=4, detectable_lift=0.05)
    assert large_delta < small_delta


def test_bonferroni_requires_multiple_strategies():
    with pytest.raises(ValueError):
        games_for_power(n_strategies=1, method="bonferroni")


def test_baseline_rate_out_of_range():
    for base_p in [0, 1, -0.1, 1.1]:
        with pytest.raises(ValueError):
            games_for_power(n_strategies=2, baseline_rate=base_p)


def test_detectable_lift_out_of_range():
    for delta in [0, 1, -0.2, 2.0]:
        with pytest.raises(ValueError):
            games_for_power(n_strategies=2, detectable_lift=delta)


def test_baseline_plus_delta_too_large():
    with pytest.raises(ValueError):
        games_for_power(n_strategies=2, baseline_rate=0.7, detectable_lift=0.4)
