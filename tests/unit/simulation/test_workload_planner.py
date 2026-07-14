from __future__ import annotations

import json
from pathlib import Path

import pytest

from farkle.simulation.workload_planner import (
    CAP_CONFIG_KEY,
    WorkloadCapExceeded,
    minimum_shuffles_for_resolution,
    plan_tournament_workload,
    worst_case_wilson_width,
    write_workload_plan,
)


def test_default_resolution_finds_smallest_n_then_rounds_to_equal_batches() -> None:
    plan = plan_tournament_workload(
        root_seed=17,
        k=4,
        strategy_count=200,
        resolution_delta=0.03,
    )

    assert plan.required_shuffles_unrounded == 4265
    assert worst_case_wilson_width(4264) > 0.03
    assert worst_case_wilson_width(4265) <= 0.03
    assert plan.required_shuffles == 4300
    assert plan.batch_count == 100
    assert plan.shuffles_per_batch == 43
    assert plan.required_games == 215_000
    assert plan.achieved_resolution <= 0.03
    assert plan.batch_construction == "equal_contiguous"


def test_minimum_batch_size_controls_low_precision_workload() -> None:
    plan = plan_tournament_workload(
        root_seed=1,
        k=2,
        strategy_count=20,
        resolution_delta=0.9,
    )

    assert minimum_shuffles_for_resolution(0.9) == 1
    assert plan.required_shuffles == 3_000
    assert plan.shuffles_per_batch == 30


def test_cap_reports_achieved_resolution_and_actionable_key() -> None:
    plan = plan_tournament_workload(
        root_seed=9,
        k=2,
        strategy_count=10,
        resolution_delta=0.03,
        shuffle_cap=4_000,
    )

    assert plan.cap_exceeded is True
    assert plan.status == "blocked_by_cap"
    assert plan.achieved_resolution_at_cap == pytest.approx(worst_case_wilson_width(4_000))
    with pytest.raises(WorkloadCapExceeded, match=CAP_CONFIG_KEY):
        raise WorkloadCapExceeded(plan)


def test_runtime_projection_and_atomic_plan_output(tmp_path: Path) -> None:
    plan = plan_tournament_workload(
        root_seed=3,
        k=2,
        strategy_count=10,
        resolution_delta=0.03,
        projected_games_per_second=250.0,
    )
    output = tmp_path / "plan.json"

    write_workload_plan(output, plan)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["projected_runtime_seconds"] == pytest.approx(plan.required_games / 250.0)
    assert payload["cap_config_key"] == CAP_CONFIG_KEY
    assert payload["status"] == "not_started"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 1},
        {"strategy_count": 11},
        {"batch_count": 1},
        {"min_shuffles_per_batch": 0},
        {"shuffle_cap": 0},
        {"projected_games_per_second": 0.0},
    ],
)
def test_invalid_workload_inputs_fail(kwargs: dict[str, int | float]) -> None:
    base: dict[str, int | float | None] = {
        "root_seed": 1,
        "k": 2,
        "strategy_count": 10,
        "resolution_delta": 0.03,
        "batch_count": 100,
        "min_shuffles_per_batch": 30,
        "shuffle_cap": None,
        "projected_games_per_second": None,
    }
    base.update(kwargs)

    with pytest.raises(ValueError):
        plan_tournament_workload(**base)  # type: ignore[arg-type]
