"""Run the bounded maximum-k, two-concurrent-root Step 5.5 canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from farkle.config import assign_config_sha, load_app_config
from farkle.orchestration.seed_utils import seed_pair_root
from farkle.orchestration.two_seed_pipeline import run_pipeline
from farkle.simulation.game_profile import GameProfile
from farkle.utils.os_memory import current_memory_boundary
from farkle.utils.writer import atomic_path

ROOTS = (11, 22)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Dedicated directory for the bounded canary configuration and artifacts",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _write_canary_config(workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    payload = {
        "io": {
            "results_dir_prefix": str((workspace / "results").resolve()),
            "analysis_subdir": "analysis",
        },
        "sim": {
            "n_players_list": [2, 12],
            "seed": ROOTS[0],
            "seed_list": list(ROOTS),
            "n_jobs": 2,
            "expanded_metrics": True,
            "row_dir": "rows",
            "metric_chunk_dir": "metric_chunks",
            "desired_sec_per_chunk": 1,
            "ckpt_every_sec": 1,
            "score_thresholds": [500],
            "dice_thresholds": [1, 2],
            "smart_five_opts": [False],
            "smart_one_opts": [False],
            "consider_score_opts": [True, False],
            "consider_dice_opts": [True],
            "auto_hot_dice_opts": [False, True],
            "run_up_score_opts": [False],
            "include_stop_at": False,
            "include_stop_at_heuristic": False,
        },
        "analysis": {
            "disable_rng_diagnostics": True,
            "n_jobs": 2,
            "log_level": "INFO",
            "rare_event_target_score": 100,
            "game_stats_margin_thresholds": [500],
        },
        "ingest": {"row_group_size": 64, "batch_rows": 64, "n_jobs": 2},
        "combine": {"max_players": 12},
        "trueskill": {"beta": 1.0, "tau": 0.0, "draw_probability": 0.0},
        "head2head": {
            "n_jobs": 1,
            "family_alpha": 0.5,
            "target_power": 0.1,
            "practical_delta": 0.2,
            "sensitivity_deltas": [0.2, 0.04],
            "seat1_advantage_scenarios": [0.0, 0.03, 0.06],
            "delta_equivalence": None,
            "candidate_cap": 3,
            "candidate_cap_policy": "balanced-tail",
            "min_candidate_completion_rate": 0.99,
            "max_attempt_multiplier": 2.0,
            "total_game_cap": 24,
            "allow_single_root": True,
        },
        "screening": {
            "resolution_delta": 0.9,
            "interval_confidence": 0.95,
            "practical_delta_by_k": {2: 0.2, 12: 0.2},
            "delta_across_k": 0.2,
            "bootstrap_replicates": 1,
            "candidate_contribution_size": 1,
            "controls": [0, 1, 3],
            "mandatory_diagnostics": [],
        },
        "batching": {"target_batches": 2, "min_shuffles_per_batch": 1},
        "robustness": {
            "report_pareto": True,
            "report_maximin": True,
            "delta_seed_stability": 0.2,
            "joint_discrepancy_alpha": 0.05,
            "matched_count_fractions": [1.0],
        },
        "artifact_contract": {
            "artifact_contract_version": 3,
            "estimand_version": 2,
            "schema_version": 2,
            "conditioning_version": 2,
        },
        "k_aggregation": {"method": "equal-k", "k_weights": None},
        "hgb": {
            "max_depth": 1,
            "n_estimators": 1,
            "heldout_folds": 2,
            "permutation_repeats": 1,
            "future_proposal_limit": 1,
        },
        "orchestration": {"parallel_seeds": True},
        "resources": {
            "scheduler_memory_budget_mb": 768,
            "process_tree_warning_threshold_mb": 768,
            "aggregate_memory_hard_limit_mb": 2304,
            "minimum_system_available_memory_mb": 1024,
            "parent_process_memory_mb": 192,
            "logical_cpu_budget": 4,
            "native_threads_per_worker": 1,
            "max_in_flight_per_worker": 1,
            "rss_sample_interval_seconds": 0.1,
            "os_memory_limit_enabled": True,
            "os_memory_limit_required": True,
            "allow_unenforced_memory_fallback": False,
        },
    }
    path = workspace / "step_5_5_canary.yaml"
    with atomic_path(str(path)) as temporary:
        Path(temporary).write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


def main() -> int:
    args = _parser().parse_args()
    boundary = current_memory_boundary()
    if boundary is None or not bool(boundary.get("enforced")):
        raise RuntimeError(
            "the Step 5.5 canary must be started through the strict OS memory supervisor"
        )
    config_path = _write_canary_config(args.workspace.resolve())
    cfg = load_app_config(config_path, seed_list_len=2)
    assign_config_sha(cfg)
    run_pipeline(
        cfg,
        seed_pair=ROOTS,
        force=bool(args.force),
        oracle_game_profile=GameProfile(default_target_score=100, default_max_rounds=200),
    )
    health_path = seed_pair_root(cfg, ROOTS) / "pipeline_health.json"
    health = json.loads(health_path.read_text(encoding="utf-8"))
    print(json.dumps(health["resource_telemetry"], indent=2, sort_keys=True))
    return 0 if health.get("status") == "complete_success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
