"""Unit tests for :func:`farkle.utils.stats.wilson_ci`."""

from __future__ import annotations

# tests/test_stats_wilson.py
"""Unit tests for Wilson confidence interval calculations."""

import pytest

from farkle.utils.stats import wilson_ci


def test_wilson_ci_balanced_sample() -> None:
    """Verify Wilson CI for a balanced number of successes and failures.

    Returns:
        None
    """

    lower, upper = wilson_ci(5, 10)

    assert lower == pytest.approx(0.2365930905, rel=1e-9)
    assert upper == pytest.approx(0.7634069095, rel=1e-9)


def test_wilson_ci_handles_extreme_counts() -> None:
    """Ensure Wilson CI remains bounded for extreme success counts.

    Returns:
        None
    """

    low_success = wilson_ci(0, 10)
    high_success = wilson_ci(10, 10)

    assert low_success[0] == pytest.approx(0.0)
    assert 0.0 <= low_success[1] <= 1.0
    assert 0.0 <= high_success[0] <= 1.0
    assert high_success[1] == pytest.approx(1.0)
    assert low_success[0] <= low_success[1]
    assert high_success[0] <= high_success[1]


def test_wilson_ci_alpha_variation() -> None:
    """Check that changing alpha adjusts interval bounds accordingly.

    Returns:
        None
    """

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
    """Confirm invalid inputs raise ValueError.

    Args:
        k: Number of observed successes.
        n: Total observations.
        alpha: Significance level provided to Wilson CI.

    Returns:
        None
    """

    with pytest.raises(ValueError):
        wilson_ci(k, n, alpha)
