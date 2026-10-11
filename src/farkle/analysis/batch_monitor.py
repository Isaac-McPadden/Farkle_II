"""Task 5B's bounded diagnostic arithmetic; production wiring belongs to 5C.

These are descriptive investigation thresholds, not tests with calibrated error
control. Callers must authenticate complete source support before using them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np

METHOD_ID: Final = "batch_lag_half_drift_descriptive_v1"
METRICS: Final = {
    "attempted_win_rate": ("wins", "attempted"),
    "attempted_safety_rate": ("safety", "attempted"),
    "completed_win_rate": ("wins", "completed"),
    "mean_deficit_completed": ("deficit_sum", "completed"),
    "mean_losing_margin": ("deficit_sum", "losses_completed"),
    "mean_winning_margin": ("victory_margin_sum", "wins"),
}


@dataclass(frozen=True)
class MonitorPolicy:
    minimum_batches: int = 30
    supported_batch_counts: tuple[int, ...] = (30, 100)
    lag_effect_floor: float = 0.5
    lag_support_multiplier: float = 4.0
    half_drift_threshold: float = 1.5

    def __post_init__(self) -> None:
        # This method ID binds an exact policy; changing it needs new capability.
        if (
            self.minimum_batches != 30
            or self.supported_batch_counts != (30, 100)
            or self.lag_effect_floor != 0.5
            or self.lag_support_multiplier != 4.0
            or self.half_drift_threshold != 1.5
        ):
            raise ValueError("policy differs from the frozen method; new capability is required")


@dataclass(frozen=True)
class BatchDiagnostic:
    check_id: str
    support: int
    statistic: float | None
    effect_threshold: float
    result: str
    reason: str
    method_id: str = METHOD_ID

    def payload(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_POLICY: Final = MonitorPolicy()


def validate_batch_counts(
    batch_ids: Sequence[int] | np.ndarray,
    counts: Mapping[str, np.ndarray],
    *,
    expected_batches: int,
) -> None:
    """Check one already-certified root/k/strategy all-seat count vector.

    This is defense in depth, not a substitute for upstream full game-coordinate,
    roster, actual-seat, or authenticated artifact checks.
    """
    ids = np.asarray(batch_ids)
    if (
        isinstance(expected_batches, bool)
        or not isinstance(expected_batches, (int, np.integer))
        or expected_batches < 1
        or ids.dtype.kind not in "iu"
        or ids.shape != (expected_batches,)
        or not np.array_equal(ids, np.arange(expected_batches))
    ):
        raise ValueError("batch coverage/order differs from the frozen expected inventory")
    for name in sorted({name for pair in METRICS.values() for name in pair}):
        if name not in counts:
            raise ValueError(f"missing required batch count {name}")
        values = np.asarray(counts[name])
        if values.shape != ids.shape or values.dtype.kind not in "iu" or np.any(values < 0):
            raise ValueError(f"invalid nonnegative integer batch counts: {name}")
    a, c, s = (np.asarray(counts[key]) for key in ("attempted", "completed", "safety"))
    w, losses = (np.asarray(counts[key]) for key in ("wins", "losses_completed"))
    # Subtractions after bounds checks avoid overflowing integer sums.
    if np.any(c > a) or np.any(w > c) or np.any(a - c != s) or np.any(c - w != losses):
        raise ValueError("batch completion or win/loss conservation failure")
    if np.any((losses == 0) & (counts["deficit_sum"] != 0)) or np.any(
        (w == 0) & (counts["victory_margin_sum"] != 0)
    ):
        raise ValueError("nonzero margin sum without its conditional support")


def lag_one_correlation(values: np.ndarray) -> float | None:
    """Pearson association of adjacent observations, with separate lag means."""
    x, y = values[:-1], values[1:]
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0 or not math.isfinite(denominator):
        return None
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def diagnose_series(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    policy: MonitorPolicy = DEFAULT_POLICY,
) -> tuple[BatchDiagnostic, BatchDiagnostic]:
    """Evaluate a complete ordered vector; never omit zero-denominator batches.

    The caller owns source certification and capability applicability. This pure
    arithmetic function alone grants neither of them.
    """
    num, den = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    if num.ndim != 1 or den.shape != num.shape or len(num) == 0:
        raise ValueError("expected nonempty equal one-dimensional batch vectors")
    if not np.isfinite(num).all() or not np.isfinite(den).all() or np.any(den < 0):
        raise ValueError("invalid finite batch sufficient statistics")
    n = len(num)
    lag_threshold = max(
        policy.lag_effect_floor, policy.lag_support_multiplier / math.sqrt(max(1, n - 1))
    )

    def rows(reason: str) -> tuple[BatchDiagnostic, BatchDiagnostic]:
        return (
            BatchDiagnostic("lag_one", n, None, lag_threshold, "insufficient_evidence", reason),
            BatchDiagnostic(
                "half_drift", n, None, policy.half_drift_threshold, "insufficient_evidence", reason
            ),
        )

    if n < policy.minimum_batches:
        return rows("fewer_than_30_batches")
    if n not in policy.supported_batch_counts:
        return rows("batch_count_outside_capability_recipe")
    if np.any(den == 0):
        return rows("zero_denominator_batch_not_dropped")
    values = num / den
    if not np.isfinite(values).all():
        raise ValueError("nonfinite batch ratios")
    sd = float(values.std(ddof=1))
    if not math.isfinite(sd):
        raise ValueError("nonfinite batch variation")
    if sd == 0:
        return rows("constant_series")
    midpoint = n // 2
    drift = float((values[midpoint:].mean() - values[:midpoint].mean()) / sd)
    lag = lag_one_correlation(values)
    output = []
    for check, value, threshold in (
        ("lag_one", lag, lag_threshold),
        ("half_drift", drift, policy.half_drift_threshold),
    ):
        if value is None or not math.isfinite(value):
            output.append(
                BatchDiagnostic(
                    check, n, None, threshold, "insufficient_evidence", "undefined_statistic"
                )
            )
        else:
            warning = abs(value) > threshold
            output.append(
                BatchDiagnostic(
                    check,
                    n,
                    value,
                    threshold,
                    "investigate" if warning else "no_concern_detected",
                    (
                        "descriptive_effect_threshold_exceeded"
                        if warning
                        else "within_declared_descriptive_threshold"
                    ),
                )
            )
    return output[0], output[1]


def diagnose_counts(
    batch_ids: Sequence[int] | np.ndarray,
    counts: Mapping[str, np.ndarray],
    *,
    expected_batches: int,
) -> dict[str, tuple[BatchDiagnostic, BatchDiagnostic]]:
    validate_batch_counts(batch_ids, counts, expected_batches=expected_batches)
    return {
        metric: diagnose_series(counts[num], counts[den]) for metric, (num, den) in METRICS.items()
    }
