"""Deterministic workload planning for broad tournament screening."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from scipy.stats import norm

from farkle.utils.writer import atomic_path

WORKLOAD_PLAN_VERSION = 1
CAP_CONFIG_KEY = "screening.max_shuffles_per_root_k"


@dataclass(frozen=True, slots=True)
class TournamentWorkloadPlan:
    """Resolved work and precision contract for one root/player-count cell."""

    root_seed: int
    k: int
    strategy_count: int
    confidence: float
    resolution_delta: float
    required_shuffles_unrounded: int
    required_shuffles: int
    batch_count: int
    shuffles_per_batch: int
    batch_construction: str
    games_per_shuffle: int
    required_games: int
    achieved_resolution: float
    shuffle_cap: int | None
    cap_exceeded: bool
    achieved_resolution_at_cap: float | None
    projected_games_per_second: float | None = None
    projected_runtime_seconds: float | None = None
    plan_version: int = WORKLOAD_PLAN_VERSION

    @property
    def status(self) -> str:
        """Return the stage lifecycle state implied by the operational cap."""

        return "blocked_by_cap" if self.cap_exceeded else "not_started"

    def with_games_per_second(self, games_per_second: float) -> TournamentWorkloadPlan:
        """Return a copy with a validated runtime projection."""

        if not math.isfinite(games_per_second) or games_per_second <= 0.0:
            raise ValueError("games_per_second must be finite and positive")
        return replace(
            self,
            projected_games_per_second=float(games_per_second),
            projected_runtime_seconds=self.required_games / float(games_per_second),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {**asdict(self), "status": self.status, "cap_config_key": CAP_CONFIG_KEY}


class WorkloadCapExceeded(RuntimeError):
    """Raised before scheduling when a screening safety cap is insufficient."""

    def __init__(self, plan: TournamentWorkloadPlan) -> None:
        self.plan = plan
        super().__init__(
            f"Required {plan.required_shuffles} shuffles for root={plan.root_seed}, "
            f"k={plan.k}, but {CAP_CONFIG_KEY}={plan.shuffle_cap}. "
            f"Raise {CAP_CONFIG_KEY} to at least {plan.required_shuffles} and resume."
        )


def worst_case_wilson_width(n: int, *, confidence: float = 0.95) -> float:
    """Return the maximum full Wilson interval width for a binomial sample size."""

    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    z2 = z * z

    def _width(successes: int) -> float:
        estimate = successes / n
        denominator = 1.0 + z2 / n
        radius = z * math.sqrt(estimate * (1.0 - estimate) / n + z2 / (4.0 * n * n))
        return 2.0 * radius / denominator

    return max(_width(n // 2), _width((n + 1) // 2))


def minimum_shuffles_for_resolution(
    resolution_delta: float,
    *,
    confidence: float = 0.95,
) -> int:
    """Find the smallest sample size meeting the maximum Wilson-width target."""

    if not 0.0 < resolution_delta < 1.0:
        raise ValueError("resolution_delta must be between 0 and 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    lower = 0
    upper = 1
    while worst_case_wilson_width(upper, confidence=confidence) > resolution_delta:
        lower = upper
        upper *= 2
    while lower + 1 < upper:
        midpoint = (lower + upper) // 2
        if worst_case_wilson_width(midpoint, confidence=confidence) <= resolution_delta:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def plan_tournament_workload(
    *,
    root_seed: int,
    k: int,
    strategy_count: int,
    resolution_delta: float,
    confidence: float = 0.95,
    batch_count: int = 100,
    min_shuffles_per_batch: int = 30,
    shuffle_cap: int | None = None,
    projected_games_per_second: float | None = None,
) -> TournamentWorkloadPlan:
    """Resolve precision, batches, game count, cap state, and runtime for one cell."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("k must be an integer of at least 2")
    if (
        isinstance(strategy_count, bool)
        or not isinstance(strategy_count, int)
        or strategy_count < k
        or strategy_count % k
    ):
        raise ValueError("strategy_count must be a positive multiple of k")
    if isinstance(batch_count, bool) or not isinstance(batch_count, int) or batch_count < 2:
        raise ValueError("batch_count must be an integer of at least 2")
    if (
        isinstance(min_shuffles_per_batch, bool)
        or not isinstance(min_shuffles_per_batch, int)
        or min_shuffles_per_batch < 1
    ):
        raise ValueError("min_shuffles_per_batch must be a positive integer")
    if shuffle_cap is not None and (
        isinstance(shuffle_cap, bool) or not isinstance(shuffle_cap, int) or shuffle_cap < 1
    ):
        raise ValueError("shuffle_cap must be positive when configured")

    unrounded = minimum_shuffles_for_resolution(
        resolution_delta,
        confidence=confidence,
    )
    shuffles_per_batch = max(min_shuffles_per_batch, math.ceil(unrounded / batch_count))
    required_shuffles = batch_count * shuffles_per_batch
    games_per_shuffle = strategy_count // k
    cap_exceeded = shuffle_cap is not None and required_shuffles > shuffle_cap
    plan = TournamentWorkloadPlan(
        root_seed=int(root_seed),
        k=k,
        strategy_count=strategy_count,
        confidence=float(confidence),
        resolution_delta=float(resolution_delta),
        required_shuffles_unrounded=unrounded,
        required_shuffles=required_shuffles,
        batch_count=batch_count,
        shuffles_per_batch=shuffles_per_batch,
        batch_construction="equal_contiguous",
        games_per_shuffle=games_per_shuffle,
        required_games=required_shuffles * games_per_shuffle,
        achieved_resolution=worst_case_wilson_width(
            required_shuffles,
            confidence=confidence,
        ),
        shuffle_cap=shuffle_cap,
        cap_exceeded=cap_exceeded,
        achieved_resolution_at_cap=(
            worst_case_wilson_width(shuffle_cap, confidence=confidence)
            if cap_exceeded and shuffle_cap is not None
            else None
        ),
    )
    if projected_games_per_second is not None:
        plan = plan.with_games_per_second(projected_games_per_second)
    return plan


def write_workload_plan(path: Path, plan: TournamentWorkloadPlan) -> None:
    """Write a workload plan atomically using deterministic canonical JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as temporary:
        Path(temporary).write_text(
            json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__ = [
    "CAP_CONFIG_KEY",
    "WORKLOAD_PLAN_VERSION",
    "TournamentWorkloadPlan",
    "WorkloadCapExceeded",
    "minimum_shuffles_for_resolution",
    "plan_tournament_workload",
    "worst_case_wilson_width",
    "write_workload_plan",
]
