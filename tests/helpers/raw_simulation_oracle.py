"""Tiny real-simulation fixture for the Task 14 raw oracle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from farkle import analysis
from farkle.analysis.stage_runner import StageRunContext, StageRunner
from farkle.config import AppConfig, assign_config_sha, load_app_config
from farkle.orchestration.run_contexts import (
    RootPairRunContext,
    SeedRunContext,
    configure_run_lineage,
    load_run_context,
    write_run_context_atomic,
)
from farkle.orchestration.seed_utils import (
    seed_pair_root,
    seed_pair_seed_root,
    write_active_config,
)
from farkle.orchestration.two_seed_pipeline import (
    _build_seed_cfg,
    _derive_per_seed_job_budgets,
    _root_lifecycle_identity,
)
from farkle.simulation import runner
from farkle.simulation.game_profile import (
    GameProfile,
    H2HMaxRoundsOverride,
    TournamentMaxRoundsOverride,
)
from farkle.utils.artifact_contract import sha256_file, sidecar_path
from farkle.utils.authenticated_contract import (
    CodeIdentity,
    CodeIdentityPolicy,
    resolve_code_identity,
)

ORACLE_ROOTS = (11, 22)
ORACLE_PLAYER_COUNTS = (2, 4)


@dataclass(frozen=True)
class FrozenFamilySnapshot:
    """Pre-H2H byte identities for the immutable candidate family."""

    family_hash: str
    file_sha256: tuple[tuple[str, str], ...]
    plan_file_sha256: tuple[tuple[str, str], ...]


def oracle_game_profile() -> GameProfile:
    """Return the authenticated limit-only profile from the blueprint."""

    return GameProfile(
        default_target_score=100,
        default_max_rounds=200,
        tournament_max_rounds_overrides=(
            TournamentMaxRoundsOverride(
                root_seed=11,
                k=2,
                shuffle_index=0,
                game_index=0,
                max_rounds=0,
            ),
        ),
        h2h_max_rounds_overrides=(
            H2HMaxRoundsOverride(11, 0, 0, 0, 0),
            H2HMaxRoundsOverride(11, 1, 0, 0, 0),
            H2HMaxRoundsOverride(11, 1, 0, 1, 0),
        ),
    )


def write_tiny_oracle_config(tmp_path: Path) -> Path:
    """Write and return the tiny two-root YAML under ``tmp_path``."""

    results_prefix = (tmp_path / "oracle_results").resolve()
    payload = {
        "io": {
            "results_dir_prefix": str(results_prefix),
            "analysis_subdir": "analysis",
        },
        "sim": {
            "n_players_list": list(ORACLE_PLAYER_COUNTS),
            "seed": 11,
            "seed_list": list(ORACLE_ROOTS),
            "n_jobs": 1,
            "expanded_metrics": True,
            "row_dir": "rows",
            "metric_chunk_dir": "metric_chunks",
            "desired_sec_per_chunk": 1,
            "ckpt_every_sec": 1,
            "score_thresholds": [500],
            "dice_thresholds": [2],
            "smart_five_opts": [False],
            "smart_one_opts": [False],
            "consider_score_opts": [True],
            "consider_dice_opts": [True],
            "auto_hot_dice_opts": [False, True],
            "run_up_score_opts": [False],
            "include_stop_at": False,
            "include_stop_at_heuristic": False,
        },
        "analysis": {
            "disable_rng_diagnostics": True,
            "n_jobs": 1,
            "log_level": "INFO",
            "rare_event_target_score": 100,
            "game_stats_margin_thresholds": [500],
        },
        "ingest": {"row_group_size": 64, "batch_rows": 64, "n_jobs": 1},
        "combine": {"max_players": 4},
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
            "practical_delta_by_k": {2: 0.2, 4: 0.2},
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
            "artifact_contract_version": 2,
            "estimand_version": 1,
            "schema_version": 1,
        },
        "k_aggregation": {"method": "equal-k", "k_weights": None},
        "hgb": {
            "max_depth": 1,
            "n_estimators": 1,
            "heldout_folds": 2,
            "permutation_repeats": 1,
            "future_proposal_limit": 1,
        },
        "orchestration": {"parallel_seeds": False},
    }
    path = tmp_path / "tiny_oracle.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


def load_tiny_oracle_config(tmp_path: Path) -> tuple[Path, AppConfig]:
    """Construct and load the fixture through the production config loader."""

    path = write_tiny_oracle_config(tmp_path)
    cfg = load_app_config(path, seed_list_len=2)
    assign_config_sha(cfg)
    return path, cfg


def load_completed_pipeline_contexts(
    cfg: AppConfig,
) -> tuple[tuple[SeedRunContext, SeedRunContext], RootPairRunContext]:
    """Reconstruct authenticated contexts after the public orchestrator returns."""

    policy_bundle = _derive_per_seed_job_budgets(cfg, seed_count=2)
    contexts: list[SeedRunContext] = []
    for seed in ORACLE_ROOTS:
        root_cfg = _build_seed_cfg(
            cfg,
            seed_pair=ORACLE_ROOTS,
            seed=seed,
            policy_bundle=policy_bundle,
        )
        context = SeedRunContext.from_config(root_cfg)
        persisted = load_run_context(
            context.run_context_path,
            active_config_path=context.active_config_path,
        )
        code_identity = CodeIdentity(**persisted["code_identity"])
        lineage = configure_run_lineage(
            context,
            code_identity=code_identity,
            game_profile_sha256=persisted["lineage_extensions"]["game_profile_sha256"],
        )
        assert lineage == persisted["run_lineage_sha256"]
        contexts.append(context)

    root_contexts = (contexts[0], contexts[1])
    pair_context = RootPairRunContext.from_root_contexts(
        root_contexts,
        pair_root=seed_pair_root(cfg, ORACLE_ROOTS),
    )
    pair_persisted = load_run_context(
        pair_context.run_context_path,
        active_config_path=pair_context.active_config_path,
    )
    pair_code_identity = CodeIdentity(**pair_persisted["code_identity"])
    pair_lineage = configure_run_lineage(
        pair_context,
        code_identity=pair_code_identity,
        parent_lifecycle_roots=tuple(pair_persisted["parent_lifecycle_roots"]),
        game_profile_sha256=pair_persisted["lineage_extensions"]["game_profile_sha256"],
    )
    assert pair_lineage == pair_persisted["run_lineage_sha256"]
    assert all(
        context.results_root == seed_pair_seed_root(cfg, ORACLE_ROOTS, context.seed)
        for context in root_contexts
    )
    return root_contexts, pair_context


def snapshot_frozen_family(cfg: AppConfig) -> FrozenFamilySnapshot:
    """Capture immutable family and H2H-plan identities from a completed run."""

    family_paths = (
        cfg.h2h_candidate_family_path(),
        sidecar_path(cfg.h2h_candidate_family_path()),
        cfg.h2h_candidate_family_manifest_path(),
        sidecar_path(cfg.h2h_candidate_family_manifest_path()),
    )
    plan_paths = (
        cfg.h2h_power_plan_path(),
        sidecar_path(cfg.h2h_power_plan_path()),
        cfg.h2h_block_manifest_path(),
        sidecar_path(cfg.h2h_block_manifest_path()),
    )
    manifest = json.loads(cfg.h2h_candidate_family_manifest_path().read_text(encoding="utf-8"))
    return FrozenFamilySnapshot(
        family_hash=str(manifest["family_hash"]),
        file_sha256=tuple((str(path), sha256_file(path)) for path in family_paths),
        plan_file_sha256=tuple((str(path), sha256_file(path)) for path in plan_paths),
    )


def run_raw_simulation_roots(
    cfg: AppConfig,
    profile: GameProfile,
) -> tuple[SeedRunContext, SeedRunContext]:
    """Run exactly the two raw root simulations without any analysis stage."""

    from farkle.orchestration.seed_utils import seed_pair_root

    code_identity = resolve_code_identity(
        Path(__file__).resolve().parents[2],
        policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY,
    )
    policy_bundle = _derive_per_seed_job_budgets(cfg, seed_count=2)
    pair_root = seed_pair_root(cfg, ORACLE_ROOTS)
    contexts: list[SeedRunContext] = []
    for seed in ORACLE_ROOTS:
        root_cfg = _build_seed_cfg(
            cfg,
            seed_pair=ORACLE_ROOTS,
            seed=seed,
            policy_bundle=policy_bundle,
        )
        context = SeedRunContext.from_config(root_cfg)
        write_run_context_atomic(
            context,
            code_identity=code_identity,
            game_profile_sha256=profile.sha256,
        )
        write_active_config(root_cfg)
        runner.run_tournament(root_cfg, oracle_game_profile=profile)
        contexts.append(context)
    assert all(pair_root in context.results_root.parents for context in contexts)
    return (contexts[0], contexts[1])


def run_analysis_through_candidate_freeze(
    cfg: AppConfig,
    contexts: tuple[SeedRunContext, SeedRunContext],
    profile: GameProfile,
) -> RootPairRunContext:
    """Run canonical root analysis and the pair workflow through family freeze."""

    code_identity = resolve_code_identity(
        Path(__file__).resolve().parents[2],
        policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY,
    )
    parent_lifecycle_roots: list[str] = []
    for context in contexts:
        analysis.run_root_analysis(context.config)
        root_plan = analysis.build_root_stage_plan(context.config)
        lifecycle_sha256, states = _root_lifecycle_identity(context, root_plan)
        assert lifecycle_sha256 is not None
        assert set(states.values()) == {"complete_valid"}
        parent_lifecycle_roots.append(lifecycle_sha256)

    pair_root = seed_pair_root(cfg, ORACLE_ROOTS)
    pair_context = RootPairRunContext.from_root_contexts(
        contexts,
        pair_root=pair_root,
    )
    write_run_context_atomic(
        pair_context,
        code_identity=code_identity,
        parent_lifecycle_roots=tuple(parent_lifecycle_roots),
        game_profile_sha256=profile.sha256,
    )
    write_active_config(pair_context.config, dest_dir=pair_root)
    candidate_plan = analysis.build_root_pair_stage_plan(pair_context)[:3]
    StageRunner.run(
        candidate_plan,
        StageRunContext(
            config=pair_context.config,
            manifest_path=pair_context.analysis_root / pair_context.config.manifest_name,
            run_label="raw_oracle_through_candidate_freeze",
            run_metadata={"execution_scope": "root_pair"},
            run_end_metadata={"execution_scope": "root_pair"},
            continue_on_error=False,
        ),
        raise_on_failure=True,
    )
    return pair_context


def run_pair_h2h_through_agreement(
    pair_context: RootPairRunContext,
    profile: GameProfile,
) -> FrozenFamilySnapshot:
    """Run real H2H planning/execution and pair analysis, stopping before reporting."""

    cfg = pair_context.config
    pair_plan = analysis.build_root_pair_stage_plan(
        pair_context,
        oracle_game_profile=profile,
    )
    StageRunner.run(
        pair_plan[3:4],
        StageRunContext(
            config=cfg,
            manifest_path=pair_context.analysis_root / cfg.manifest_name,
            run_label="raw_oracle_h2h_plan",
            run_metadata={"execution_scope": "root_pair"},
            run_end_metadata={"execution_scope": "root_pair"},
            continue_on_error=False,
        ),
        raise_on_failure=True,
    )
    snapshot = snapshot_frozen_family(cfg)
    StageRunner.run(
        pair_plan[4:8],
        StageRunContext(
            config=cfg,
            manifest_path=pair_context.analysis_root / cfg.manifest_name,
            run_label="raw_oracle_h2h_execute_through_agreement",
            run_metadata={"execution_scope": "root_pair"},
            run_end_metadata={"execution_scope": "root_pair"},
            continue_on_error=False,
        ),
        raise_on_failure=True,
    )
    return snapshot


__all__ = [
    "FrozenFamilySnapshot",
    "ORACLE_PLAYER_COUNTS",
    "ORACLE_ROOTS",
    "load_completed_pipeline_contexts",
    "load_tiny_oracle_config",
    "oracle_game_profile",
    "run_analysis_through_candidate_freeze",
    "run_pair_h2h_through_agreement",
    "run_raw_simulation_roots",
    "snapshot_frozen_family",
    "write_tiny_oracle_config",
]
