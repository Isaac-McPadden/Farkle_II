"""Reproduce Task 4's detection-power illustration with the current exact planner.

No games are simulated and no production artifacts are changed. The example
family has one observed nonzero contrast and null observations for all other
pairs; it illustrates decision semantics, not empirical Farkle performance.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from farkle.analysis.h2h_inference import (
    _holm_adjust,
    pair_conclusions,
    score_difference_interval,
    two_proportion_score_test,
)
from farkle.analysis.h2h_schedule import (
    POWER_METHOD_ID,
    _minimum_block_games,
    _scenario_probabilities,
    implemented_score_test_power,
)
from farkle.config import AppConfig, load_app_config
from farkle.protocols import H2H_SEAT1_ADVANTAGE_SCENARIOS
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.writer import atomic_path


def numerical_example(cfg: AppConfig, candidate_count: int) -> dict:
    h = cfg.head2head
    roots = tuple(cfg.sim.seed_list or [cfg.sim.seed])
    pair_count = candidate_count * (candidate_count - 1) // 2
    alpha = h.family_alpha / pair_count
    scenarios = tuple(H2H_SEAT1_ADVANTAGE_SCENARIOS)
    block = _minimum_block_games(
        root_count=len(roots),
        effect=h.practical_delta,
        scenarios=scenarios,
        alpha_per_pair=alpha,
        target_power=h.target_power,
    )
    n_order = block * len(roots)
    # Deterministic integer observations close to the planning alternative,
    # without a seat effect; counts refer to seat-one wins in opposite orders.
    count1 = round((0.5 + h.practical_delta) * n_order)
    count2 = n_order - count1
    score = two_proportion_score_test(count1, n_order, count2, n_order)
    bounds = score_difference_interval(count1, n_order, count2, n_order, alpha=alpha)
    p_values = np.ones(pair_count)
    p_values[0] = score.p_value
    adjusted, _ = _holm_adjust(p_values)
    decisions = pair_conclusions(
        eligible=True,
        holm_reject=bool(adjusted[0] <= h.family_alpha),
        effect=score.difference / 2,
        low=bounds[0] / 2,
        high=bounds[1] / 2,
        practical_delta=h.practical_delta,
        delta_equivalence=h.delta_equivalence,
    )
    powers = [
        {
            "seat1_advantage": scenario,
            "rejection_power": implemented_score_test_power(
                n_order, *_scenario_probabilities(h.practical_delta, scenario), alpha
            ),
        }
        for scenario in scenarios
    ]
    return {
        "contract": "first_release_task0_v1",
        "power_method": POWER_METHOD_ID,
        "candidate_count": candidate_count,
        "pair_count": pair_count,
        "root_seeds": list(roots),
        "family_alpha": h.family_alpha,
        "alpha_per_pair": alpha,
        "planned_effect": h.practical_delta,
        "target_detection_power": h.target_power,
        "equivalence_margin": h.delta_equivalence,
        "n_completed_per_root_order_block": block,
        "n_completed_per_order": n_order,
        "n_completed_per_pair": 2 * n_order,
        "total_planned_completed_games": pair_count * 2 * n_order,
        "scenario_powers": powers,
        "worst_scenario_rejection_power": min(p["rejection_power"] for p in powers),
        "example": {
            "seat1_wins_order_ab": count1,
            "seat1_wins_order_ba": count2,
            "d_ab": score.difference / 2,
            "simultaneous_d_low": bounds[0] / 2,
            "simultaneous_d_high": bounds[1] / 2,
            "p_value": score.p_value,
            "holm_adjusted_p": float(adjusted[0]),
            "other_pair_p_values": 1.0,
            **decisions,
        },
        "interpretation": (
            "Planned power is rejection power for zero advantage under the stated alternatives, "
            "conditional on completed-game support. It is not power to prove practical dominance "
            "or equivalence. Separate flags describe what the synthetic observed interval supports. "
            "Holm superiority and simultaneous-interval equivalence have separate multiplicity "
            "statements; no single joint family-wise guarantee is asserted."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/farkle_mega_config.yaml"))
    parser.add_argument("--candidates", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.candidates < 2:
        parser.error("at least two candidates are required")
    if args.output.exists() and not args.force:
        print(f"Existing example retained: {args.output}")
        return
    started = time.perf_counter()
    result = numerical_example(load_app_config(args.config), args.candidates)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["config_sha256"] = sha256_file(args.config)
    result["source_sha256"] = {
        str(path): sha256_file(path)
        for path in (
            Path(__file__),
            Path("src/farkle/analysis/h2h_schedule.py"),
            Path("src/farkle/analysis/h2h_inference.py"),
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(args.output)) as staged:
        Path(staged).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
