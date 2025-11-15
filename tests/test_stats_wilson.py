"""Unit tests for :func:`farkle.utils.stats.wilson_ci`."""

from __future__ import annotations

import pytest

from farkle.utils.stats import wilson_ci


def test_wilson_ci_balanced_sample() -> None:
    lower, upper = wilson_ci(5, 10)

    assert lower == pytest.approx(0.2365930905, rel=1e-9)
    assert upper == pytest.approx(0.7634069095, rel=1e-9)


def test_wilson_ci_handles_extreme_counts() -> None:
    low_success = wilson_ci(0, 10)
    high_success = wilson_ci(10, 10)

    assert low_success[0] == pytest.approx(0.0)
    assert 0.0 <= low_success[1] <= 1.0
    assert 0.0 <= high_success[0] <= 1.0
    assert high_success[1] == pytest.approx(1.0)
    assert low_success[0] <= low_success[1]
    assert high_success[0] <= high_success[1]


def test_wilson_ci_alpha_variation() -> None:
    lower, upper = wilson_ci(30, 50, alpha=0.10)

    assert lower == pytest.approx(0.4837527059, rel=1e-9)
    assert upper == pytest.approx(0.7059806569, rel=1e-9)


@pytest.mark.parametrize(
    "k,n,alpha",
    [
        (-1, 10, 0.05),
        (11, 10, 0.05),
        (1, 0, 0.05),
        (1, 10, -0.1),
        (1, 10, 1.0),
    ],
)
def test_wilson_ci_invalid_inputs(k: int, n: int, alpha: float) -> None:
    with pytest.raises(ValueError):
        wilson_ci(k, n, alpha)
