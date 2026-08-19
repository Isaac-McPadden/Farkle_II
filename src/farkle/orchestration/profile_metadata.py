"""Resolved run-profile metadata shared by contexts, health, and reports."""

from __future__ import annotations

import math
from dataclasses import asdict
from functools import lru_cache
from typing import Any

from farkle.config import AppConfig
from farkle.simulation import runner

PROFILE_METADATA_CONTRACT_VERSION = 1


@lru_cache(maxsize=64)
def _project_h2h_at_candidate_count(
    *,
    root_count: int,
    candidate_count: int,
    family_alpha: float,
    target_power: float,
    practical_delta: float,
    scenarios: tuple[float, ...],
    max_attempt_multiplier: float,
) -> dict[str, int]:
    """Use the existing exact first-crossing method for one bounded family size."""

    from farkle.analysis.h2h_schedule import _minimum_block_games

    pair_count = candidate_count * (candidate_count - 1) // 2
    block_games = _minimum_block_games(
        root_count=root_count,
        effect=practical_delta,
        scenarios=scenarios,
        alpha_per_pair=family_alpha / pair_count,
        target_power=target_power,
    )
    block_count = pair_count * root_count * 2
    return {
        "candidate_count": candidate_count,
        "unordered_pair_count": pair_count,
        "root_count": root_count,
        "seat_orientation_count": 2,
        "n_completed_required_per_root_order_block": block_games,
        "total_block_count": block_count,
        "planned_completed_games": block_games * block_count,
        "maximum_total_attempts": (math.ceil(max_attempt_multiplier * block_games) * block_count),
    }


def resolved_profile_metadata(
    cfg: AppConfig,
    *,
    final_candidate_count: int | None = None,
    pair_count: int | None = None,
    planned_h2h_games: int | None = None,
    h2h_games_per_root_order_block: int | None = None,
) -> dict[str, Any]:
    """Resolve configuration and workload facts without reading result artifacts."""

    _strategies, strategy_count, _custom = runner._resolve_strategies(cfg, None)
    roots = tuple(int(root) for root in (cfg.sim.seed_list or [cfg.sim.seed]))
    player_counts = sorted({int(value) for value in cfg.sim.n_players_list})
    workload_by_k: dict[str, dict[str, int | float]] = {}
    for k in player_counts:
        plan = runner._plan_workload_from_config(cfg, strategy_count, k)
        workload_by_k[str(k)] = {
            "configured_resolution": plan.resolution_delta,
            "achieved_resolution": plan.achieved_resolution,
            "required_shuffles_unrounded": plan.required_shuffles_unrounded,
            "required_shuffles": plan.required_shuffles,
            "batch_count": plan.batch_count,
            "shuffles_per_batch": plan.shuffles_per_batch,
            "games_per_shuffle": plan.games_per_shuffle,
            "required_games_per_root": plan.required_games,
        }

    protected_count = len(set(cfg.screening.controls) | set(cfg.screening.mandatory_diagnostics))
    maximum_candidates = min(
        strategy_count,
        2 * cfg.screening.candidate_contribution_size + protected_count,
    )
    if cfg.head2head.candidate_cap is not None:
        maximum_candidates = min(maximum_candidates, cfg.head2head.candidate_cap)
    maximum_pairs = maximum_candidates * (maximum_candidates - 1) // 2
    projected_h2h: dict[str, int | str] = {
        "candidate_count": maximum_candidates,
        "unordered_pair_count": maximum_pairs,
        "root_count": len(roots),
        "seat_orientation_count": 2,
        "projection_role": "configured_candidate_upper_envelope",
    }
    if cfg.head2head.candidate_cap is not None:
        projected_h2h.update(
            _project_h2h_at_candidate_count(
                root_count=len(roots),
                candidate_count=maximum_candidates,
                family_alpha=float(cfg.head2head.family_alpha),
                target_power=float(cfg.head2head.target_power),
                practical_delta=float(cfg.head2head.practical_delta),
                scenarios=tuple(float(value) for value in cfg.head2head.seat1_advantage_scenarios),
                max_attempt_multiplier=float(cfg.head2head.max_attempt_multiplier),
            )
        )
    else:
        projected_h2h["allocation_status"] = "pending_frozen_candidate_count"

    actual_h2h: dict[str, int] | None = None
    if final_candidate_count is not None or pair_count is not None or planned_h2h_games is not None:
        if None in (final_candidate_count, pair_count, planned_h2h_games):
            raise ValueError("final H2H metadata must provide candidate, pair, and game counts")
        assert final_candidate_count is not None
        assert pair_count is not None
        assert planned_h2h_games is not None
        expected_pairs = int(final_candidate_count) * (int(final_candidate_count) - 1) // 2
        if int(pair_count) != expected_pairs:
            raise ValueError("final H2H pair count does not match final candidate count")
        actual_h2h = {
            "final_candidate_count": int(final_candidate_count),
            "unordered_pair_count": int(pair_count),
            "planned_completed_games": int(planned_h2h_games),
        }
        if h2h_games_per_root_order_block is not None:
            actual_h2h["n_completed_required_per_root_order_block"] = int(
                h2h_games_per_root_order_block
            )

    claim_scope = (
        "production_release_evidence"
        if cfg.profile.production_eligible and cfg.profile.release_eligible
        else "non_production_non_release_development_evidence"
    )
    return {
        "profile_metadata_contract_version": PROFILE_METADATA_CONTRACT_VERSION,
        **asdict(cfg.profile),
        "claim_scope": claim_scope,
        "strategy_count": strategy_count,
        "roots": list(roots),
        "player_counts": player_counts,
        "configured_resolution": cfg.screening.resolution_delta,
        "workload_by_k": workload_by_k,
        "bootstrap_replicates": cfg.screening.bootstrap_replicates,
        "rng_partitions": cfg.analysis.rng_diagnostic_partitions,
        "candidate_contribution_size_by_method": {
            "win_rate": cfg.screening.candidate_contribution_size,
            "trueskill": cfg.screening.candidate_contribution_size,
        },
        "frozen_candidate_cap": cfg.head2head.candidate_cap,
        "maximum_candidate_count": maximum_candidates,
        "maximum_pair_count": maximum_pairs,
        "projected_h2h": projected_h2h,
        "final_h2h": actual_h2h,
    }


__all__ = [
    "PROFILE_METADATA_CONTRACT_VERSION",
    "resolved_profile_metadata",
]
