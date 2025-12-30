from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from farkle.utils.stats import bh_correct, games_for_power, wilson_ci


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


def test_games_for_power_invalid_method_defaults_to_bh():
    base: dict[str, Any] = {"n_strategies": 2}
    bh_games = games_for_power(**base)
    assert games_for_power(method="foo", **base) == bh_games


def test_bh_vs_bonferroni():
    n_bh_1 = games_for_power(n_strategies=3, method="bh")
    n_bonf_1 = games_for_power(n_strategies=3, method="bonferroni")
    n_bh_2 = games_for_power(n_strategies=500, method="bh")
    n_bonf_2 = games_for_power(n_strategies=500, method="bonferroni")
    assert n_bh_1 == n_bonf_1
    assert n_bh_2 < n_bonf_2


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


@pytest.mark.parametrize(
    "k,n,expected",
    [
        (0, 10, (0.0, 0.2775327998628892)),
        (5, 10, (0.236593090512564, 0.763406909487436)),
        (10, 10, (0.7224672001371106, 1.0)),
    ],
)
def test_wilson_ci_values(k: int, n: int, expected: tuple[float, float]):
    lower, upper = wilson_ci(k, n, alpha=0.05)
    assert lower == pytest.approx(expected[0], rel=1e-3)
    assert upper == pytest.approx(expected[1], rel=1e-3)


@pytest.mark.parametrize(
    "params",
    [
        {"k": -1, "n": 10},
        {"k": 11, "n": 10},
        {"k": 0, "n": 0},
        {"k": 0, "n": 10, "alpha": 0},
        {"k": 0, "n": 10, "alpha": 1},
    ],
)
def test_wilson_ci_invalid(params: dict[str, Any]):
    with pytest.raises(ValueError):
        wilson_ci(**params)


def test_games_for_power_with_logging_defaults(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    games = games_for_power(
        n_strategies=10,
        method="bh",
        baseline_rate=0.05,
        k_players=4,
        endpoint="top1",
        bh_target_rank=5,
        bh_target_frac=0.5,
    )
    assert games > 0
    assert any("ignoring rank" in message for message in caplog.messages)


def test_games_for_power_floor_and_cap() -> None:
    games = games_for_power(n_strategies=6, min_games_floor=1000)
    assert games >= 1000
    capped = games_for_power(n_strategies=6, min_games_floor=5, max_games_cap=10)
    assert capped == 10


def test_bh_correct_handles_strings_castable_to_float() -> None:
    pvals = np.array(["0.01", "0.02", "0.5"], dtype=object)
    mask = bh_correct(pvals.astype(float), alpha=0.05)
    assert mask.tolist() == [True, True, False]
