"""Two-root simulation followed by one canonical root-pair analysis workflow."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

from farkle import analysis
from farkle.analysis.release_audit import audit_sidecar_completeness
from farkle.analysis.stage_runner import StageRunContext, StageRunner
from farkle.config import AppConfig, assign_config_sha
from farkle.orchestration.run_contexts import (
    SEED_PAIR_ANALYSIS_DIRNAME,
    RootPairRunContext,
    SeedRunContext,
    load_run_context,
    write_run_context_atomic,
)
from farkle.orchestration.seed_utils import (
    prepare_seed_config,
    seed_has_completion_markers,
    seed_pair_root,
    seed_pair_seed_root,
    write_active_config,
)
from farkle.simulation import runner
from farkle.simulation.game_profile import GameProfile
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.authenticated_contract import (
    CodeIdentity,
    CodeIdentityPolicy,
    resolve_code_identity,
)
from farkle.utils.manifest import (
    EVENT_RUN_END,
    EVENT_RUN_START,
    append_manifest_event,
    make_run_id,
    validate_manifest_contract,
)
from farkle.utils.parallel import (
    StageParallelPolicy,
    apply_native_thread_limits,
    normalize_n_jobs,
    resolve_stage_parallel_policy,
)
from farkle.utils.stage_completion import CompletionState, freshness_sha256, resolve_stage_state
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SeedRunStatus:
    """Simulation and root-analysis outcome for one root."""

    seed: int
    context: SeedRunContext
    simulation_ok: bool
    analysis_ok: bool
    simulation_error: str | None = None
    analysis_error: str | None = None
    lifecycle_sha256: str | None = None
    stage_states: dict[str, str] | None = None


@dataclass(frozen=True)
class _PerSeedPolicyBundle:
    """Resolved process-worker policies for one root workflow."""

    simulation: StageParallelPolicy
    ingest: StageParallelPolicy
    analysis: StageParallelPolicy
    head2head: StageParallelPolicy
    requested_n_jobs: dict[str, int | None]
    resolved_n_jobs: dict[str, int]

    def as_metadata(self) -> dict[str, dict[str, int | None]]:
        return {
            name: {
                "requested_n_jobs": self.requested_n_jobs[name],
                "resolved_n_jobs": self.resolved_n_jobs[name],
                "effective_n_jobs": policy.process_workers,
                "total_cores": policy.total_cores,
                "process_workers": policy.process_workers,
                "python_threads": policy.python_threads,
                "arrow_threads": policy.arrow_threads,
                "native_threads_per_process": policy.native_threads_per_process,
            }
            for name, policy in (
                ("simulation", self.simulation),
                ("ingest", self.ingest),
                ("analysis", self.analysis),
                ("head2head", self.head2head),
            )
        }


def _per_seed_worker_budget(total_workers: int, seed_count: int) -> int:
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    return max(1, total_workers // seed_count)


def _derive_per_seed_job_budgets(cfg: AppConfig, seed_count: int) -> _PerSeedPolicyBundle:
    concurrency = seed_count if cfg.orchestration.parallel_seeds else 1
    requested: dict[str, int | None] = {
        "simulation": cfg.sim.n_jobs,
        "ingest": cfg.ingest.n_jobs,
        "analysis": cfg.analysis.n_jobs,
        "head2head": cfg.head2head.n_jobs,
    }
    resolved = {name: normalize_n_jobs(value, default=1) for name, value in requested.items()}
    effective = {
        name: (value if name == "head2head" else _per_seed_worker_budget(value, concurrency))
        for name, value in resolved.items()
    }
    bundle = _PerSeedPolicyBundle(
        simulation=resolve_stage_parallel_policy(
            "simulation",
            cfg.sim,
            n_jobs_override=effective["simulation"],
        ),
        ingest=resolve_stage_parallel_policy(
            "ingest",
            cfg.ingest,
            n_jobs_override=effective["ingest"],
        ),
        analysis=resolve_stage_parallel_policy(
            "analysis",
            cfg.analysis,
            n_jobs_override=effective["analysis"],
        ),
        head2head=resolve_stage_parallel_policy(
            "head2head",
            cfg.head2head,
            n_jobs_override=effective["head2head"],
        ),
        requested_n_jobs=requested,
        resolved_n_jobs=resolved,
    )
    LOGGER.info(
        "Resolved root process-worker policies",
        extra={
            "stage": "orchestration",
            "root_count": seed_count,
            "parallel_roots": cfg.orchestration.parallel_seeds,
            "resolved_policy": bundle.as_metadata(),
        },
    )
    return bundle


def _project_file_capacity(cfg: AppConfig, *, root_count: int) -> dict[str, object]:
    """Project the high-cardinality file classes before pipeline execution.

    This is operational capacity information.  Tournament precision planning
    determines the shuffle count, but this projection is not a statistical
    sample-size calculation and does not authorize or block work.
    """

    _strategies, strategy_count, _custom = runner._resolve_strategies(cfg, None)
    workload_by_k: list[dict[str, object]] = []
    tournament_shards = 0
    required_games = 0
    for k in sorted({int(value) for value in cfg.sim.n_players_list}):
        plan = runner._plan_workload_from_config(cfg, strategy_count, k)
        row_shards_per_root = plan.required_shuffles if cfg.simulation_row_dir(k) else 0
        tournament_shards += row_shards_per_root * root_count
        required_games += plan.required_games * root_count
        workload_by_k.append(
            {
                "k": k,
                "required_games_per_root": plan.required_games,
                "required_shuffles_per_root": plan.required_shuffles,
                "row_shards_per_root": row_shards_per_root,
                "cap_exceeded": plan.cap_exceeded,
                "cap_state": plan.status,
            }
        )

    protected_count = len(set(cfg.screening.controls) | set(cfg.screening.mandatory_diagnostics))
    candidate_upper = min(
        strategy_count,
        2 * cfg.screening.candidate_contribution_size + protected_count,
    )
    if cfg.head2head.candidate_cap is not None:
        candidate_upper = min(candidate_upper, cfg.head2head.candidate_cap)
    h2h_blocks = candidate_upper * (candidate_upper - 1) // 2 * root_count * 2
    projected_data_files = tournament_shards + h2h_blocks
    projected_sidecars = projected_data_files
    return {
        "estimate_kind": "declared_high_cardinality_upper_envelope",
        "operational_capacity_only": True,
        "statistical_sample_size_calculation": False,
        "root_count": root_count,
        "strategy_count": strategy_count,
        "workload_by_k": workload_by_k,
        "required_games_all_roots": required_games,
        "tournament_row_shards": tournament_shards,
        "h2h_candidate_count_upper_envelope": candidate_upper,
        "h2h_coordinate_blocks_upper_envelope": h2h_blocks,
        "projected_sidecars": projected_sidecars,
        "projected_total_files": projected_data_files + projected_sidecars,
        "projected_total_scope": (
            "tournament row shards, H2H coordinate blocks, and their adjacent sidecars; "
            "bounded fixed workflow artifacts, manifests, logs, and completion stamps excluded"
        ),
        "warning_threshold": None,
        "warning": False,
    }


def _build_seed_cfg(
    cfg: AppConfig,
    *,
    seed_pair: tuple[int, int],
    seed: int,
    policy_bundle: _PerSeedPolicyBundle,
) -> AppConfig:
    root_cfg = prepare_seed_config(
        cfg,
        seed=seed,
        base_results_dir=seed_pair_seed_root(cfg, seed_pair, seed),
    )
    root_cfg.sim.n_jobs = policy_bundle.simulation.process_workers
    root_cfg.ingest.n_jobs = policy_bundle.ingest.process_workers
    root_cfg.analysis.n_jobs = policy_bundle.analysis.process_workers
    assign_config_sha(root_cfg)
    return root_cfg


def _current_plan_states(cfg: AppConfig, plan: Sequence[Any]) -> dict[str, str]:
    states: dict[str, str] = {}
    for item in plan:
        if item.completion_stamp is None:
            states[item.name] = CompletionState.COMPLETE_STALE.value
            continue
        states[item.name] = resolve_stage_state(
            item.completion_stamp,
            inputs=[],
            outputs=item.required_outputs,
            cfg=cfg,
            stage=item.name,
            freshness_key=item.freshness_key,
        ).value
    return states


def _root_lifecycle_identity(
    context: SeedRunContext, plan: Sequence[Any]
) -> tuple[str | None, dict[str, str]]:
    states = {
        f"simulation_{k}p": (
            CompletionState.COMPLETE_VALID.value
            if runner.simulation_is_complete(context.config, int(k))
            else CompletionState.COMPLETE_STALE.value
        )
        for k in context.config.sim.n_players_list
    }
    states.update(_current_plan_states(context.config, plan))
    if any(value != CompletionState.COMPLETE_VALID.value for value in states.values()):
        return None, states
    stamps = [
        runner.simulation_done_path(context.config, int(k))
        for k in context.config.sim.n_players_list
    ]
    stamps.extend(item.completion_stamp for item in plan if item.completion_stamp is not None)
    identity = freshness_sha256(
        {
            "run_lineage_sha256": context.config._run_lineage_sha256,
            "completion_stamps": [sha256_file(path) for path in stamps],
        }
    )
    return identity, states


def _run_per_seed_analysis(
    cfg: AppConfig,
    *,
    seed: int,
    force: bool,
    policy_bundle: _PerSeedPolicyBundle,
) -> None:
    """Execute a root workflow that ends after screening and diagnostics."""

    apply_native_thread_limits(policy_bundle.analysis)
    manifest_path = cfg.analysis_dir / cfg.manifest_name
    plan = analysis.build_root_stage_plan(cfg, force=force)
    StageRunner.run(
        plan,
        StageRunContext(
            config=cfg,
            manifest_path=manifest_path,
            run_label=f"per_seed_pipeline_{seed}",
            run_metadata={
                "seed": seed,
                "execution_scope": "root",
                "results_dir": str(cfg.results_root),
                "analysis_dir": str(cfg.analysis_dir),
                "resolved_policy": policy_bundle.as_metadata(),
            },
            run_end_metadata={"execution_scope": "root"},
            continue_on_error=False,
            logger=LOGGER,
        ),
        raise_on_failure=True,
    )


def _run_one_seed(
    cfg: AppConfig,
    *,
    seed: int,
    seed_pair: tuple[int, int],
    manifest_path: Path,
    run_id: str,
    force: bool,
    policy_bundle: _PerSeedPolicyBundle,
    code_identity: CodeIdentity,
    cli_overrides: tuple[str, ...] = (),
    oracle_game_profile: GameProfile | None = None,
) -> _SeedRunStatus:
    root_cfg = _build_seed_cfg(
        cfg,
        seed_pair=seed_pair,
        seed=seed,
        policy_bundle=policy_bundle,
    )
    context = SeedRunContext.from_config(root_cfg)
    write_run_context_atomic(
        context,
        code_identity=code_identity,
        cli_overrides=cli_overrides,
        worker_counts=policy_bundle.as_metadata(),
        game_profile_sha256=(
            oracle_game_profile.sha256 if oracle_game_profile is not None else None
        ),
    )
    write_active_config(root_cfg)
    apply_native_thread_limits(policy_bundle.simulation)
    try:
        if not force and seed_has_completion_markers(root_cfg):
            simulation_event = "root_simulation_skipped_complete"
        else:
            if oracle_game_profile is None:
                runner.run_tournament(root_cfg, force=force)
            else:
                runner.run_tournament(
                    root_cfg,
                    force=force,
                    oracle_game_profile=oracle_game_profile,
                )
            simulation_event = "root_simulation_complete"
        append_manifest_event(
            manifest_path,
            {"event": simulation_event, "root_seed": seed},
            run_id=run_id,
            config_sha=root_cfg.config_sha,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        return _SeedRunStatus(
            seed=seed,
            context=context,
            simulation_ok=False,
            analysis_ok=False,
            simulation_error=error,
        )
    try:
        _run_per_seed_analysis(
            root_cfg,
            seed=seed,
            force=force,
            policy_bundle=policy_bundle,
        )
    except Exception as exc:  # noqa: BLE001
        return _SeedRunStatus(
            seed=seed,
            context=context,
            simulation_ok=True,
            analysis_ok=False,
            analysis_error=f"{type(exc).__name__}: {exc}",
        )
    plan = analysis.build_root_stage_plan(root_cfg, force=False)
    lifecycle_sha, stage_states = _root_lifecycle_identity(context, plan)
    if lifecycle_sha is None:
        return _SeedRunStatus(
            seed=seed,
            context=context,
            simulation_ok=True,
            analysis_ok=False,
            analysis_error="root workflow contains stale or incomplete canonical stages",
            stage_states=stage_states,
        )
    return _SeedRunStatus(
        seed=seed,
        context=context,
        simulation_ok=True,
        analysis_ok=True,
        lifecycle_sha256=lifecycle_sha,
        stage_states=stage_states,
    )


def _write_pipeline_health(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as temporary:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _final_release_gate(
    root_results: dict[int, _SeedRunStatus],
    pair_context: RootPairRunContext,
    *,
    code_identity: CodeIdentity,
    allow_oracle_code_identity: bool,
) -> dict[str, Any]:
    """Authenticate run contexts and every canonical descendant before success."""

    failures: list[str] = []
    contexts: list[SeedRunContext | RootPairRunContext] = [
        root_results[seed].context for seed in sorted(root_results)
    ]
    contexts.append(pair_context)
    run_contexts: dict[str, dict[str, str]] = {}
    for context in contexts:
        label = "pair" if isinstance(context, RootPairRunContext) else f"root_{int(context.seed)}"
        try:
            persisted = load_run_context(
                context.run_context_path,
                active_config_path=context.active_config_path,
            )
            run_contexts[label] = {
                "path": str(context.run_context_path),
                "sha256": sha256_file(context.run_context_path),
                "identity_sha256": str(persisted["run_context_sha256"]),
            }
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label} run context: {type(exc).__name__}: {exc}")
        for failure in audit_sidecar_completeness(context.analysis_root):
            failures.append(f"{label} authenticated graph: {failure}")
    release_eligible = (
        code_identity.policy == CodeIdentityPolicy.RELEASE_CLEAN.value
        and code_identity.state == "clean"
    )
    if not release_eligible and not allow_oracle_code_identity:
        failures.append("release approval requires release-clean code identity")
    return {
        "status": "passed" if not failures else "failed",
        "release_eligible": release_eligible,
        "accepted_release_identity": [3, 2, 2, 2, 2, 2],
        "artifact_roots": [str(context.analysis_root) for context in contexts],
        "run_contexts": run_contexts,
        "code_identity": {
            "commit": code_identity.commit,
            "policy": code_identity.policy,
            "state": code_identity.state,
            "dirty_fingerprint_sha256": code_identity.dirty_fingerprint_sha256,
        },
        "failures": sorted(failures),
    }


def run_pipeline(
    cfg: AppConfig,
    *,
    seed_pair: tuple[int, int],
    force: bool = False,
    cli_overrides: tuple[str, ...] = (),
    oracle_game_profile: GameProfile | None = None,
) -> None:
    """Run both roots, then combination, H2H, agreement, and reporting once."""

    if len(set(seed_pair)) != 2:
        raise ValueError(f"two-seed-pipeline requires two distinct roots, found {seed_pair}")
    if oracle_game_profile is None:
        cfg.validate_statistical_contract(require_two_roots=True)
    else:
        contract = cfg.artifact_contract
        release_identity = (
            contract.artifact_contract_version,
            cfg.rng.scheme_version,
            runner.OUTCOME_SCHEMA_VERSION,
            contract.schema_version,
            contract.estimand_version,
            contract.conditioning_version,
        )
        if release_identity != (3, 2, 2, 2, 2, 2):
            raise ValueError(
                "the test-only oracle seam still requires release identity " "3/2/2/2/2/2"
            )
    if cfg.config_sha is None:
        assign_config_sha(cfg)
    code_identity = resolve_code_identity(
        Path(__file__).resolve().parents[3],
        policy=(
            CodeIdentityPolicy.DEVELOPMENT_DIRTY
            if oracle_game_profile is not None
            else CodeIdentityPolicy.RELEASE_CLEAN
        ),
    )
    pair_root = seed_pair_root(cfg, seed_pair)
    manifest_path = pair_root / "two_seed_pipeline_manifest.jsonl"
    health_path = pair_root / "pipeline_health.json"
    run_id = make_run_id(f"two_seed_pipeline_{seed_pair[0]}_{seed_pair[1]}")
    validate_manifest_contract(manifest_path)
    policy_bundle = _derive_per_seed_job_budgets(cfg, len(seed_pair))
    file_capacity = _project_file_capacity(cfg, root_count=len(seed_pair))
    LOGGER.info(
        "Projected pipeline file-count capacity",
        extra={"stage": "orchestration_preflight", **file_capacity},
    )
    append_manifest_event(
        manifest_path,
        {
            "event": EVENT_RUN_START,
            "seed_pair": list(seed_pair),
            "results_dir": str(pair_root),
            "pair_analysis_dir": str(pair_root / SEED_PAIR_ANALYSIS_DIRNAME),
            "resolved_policy": policy_bundle.as_metadata(),
            "file_count_capacity": file_capacity,
        },
        run_id=run_id,
        config_sha=cfg.config_sha,
    )
    if cfg.orchestration.parallel_seeds:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                seed: executor.submit(
                    _run_one_seed,
                    cfg,
                    seed=seed,
                    seed_pair=seed_pair,
                    manifest_path=manifest_path,
                    run_id=run_id,
                    force=force,
                    policy_bundle=policy_bundle,
                    code_identity=code_identity,
                    cli_overrides=cli_overrides,
                    oracle_game_profile=oracle_game_profile,
                )
                for seed in seed_pair
            }
            root_results = {seed: futures[seed].result() for seed in seed_pair}
    else:
        root_results = {
            seed: _run_one_seed(
                cfg,
                seed=seed,
                seed_pair=seed_pair,
                manifest_path=manifest_path,
                run_id=run_id,
                force=force,
                policy_bundle=policy_bundle,
                code_identity=code_identity,
                cli_overrides=cli_overrides,
                oracle_game_profile=oracle_game_profile,
            )
            for seed in seed_pair
        }
    root_health = {
        str(seed): {
            "simulation": "complete" if result.simulation_ok else "failed",
            "analysis": "complete" if result.analysis_ok else "failed",
            "error": result.simulation_error or result.analysis_error,
            "lifecycle_sha256": result.lifecycle_sha256,
            "stage_states": result.stage_states,
            "public_config_sha256": result.context.config.config_sha,
        }
        for seed, result in root_results.items()
    }
    root_failures = [
        f"root {seed}: {status['error']}"
        for seed, status in root_health.items()
        if status["analysis"] != "complete"
    ]
    pair_context: RootPairRunContext | None = None
    pair_error: str | None = None
    if not root_failures:
        root_contexts = cast(
            tuple[SeedRunContext, SeedRunContext],
            tuple(root_results[seed].context for seed in seed_pair),
        )
        pair_context = RootPairRunContext.from_root_contexts(
            root_contexts,
            pair_root=pair_root,
        )
        parent_lifecycle_roots = tuple(
            cast(str, root_results[seed].lifecycle_sha256) for seed in seed_pair
        )
        write_run_context_atomic(
            pair_context,
            code_identity=code_identity,
            parent_lifecycle_roots=parent_lifecycle_roots,
            cli_overrides=cli_overrides,
            worker_counts=policy_bundle.as_metadata(),
            game_profile_sha256=(
                oracle_game_profile.sha256 if oracle_game_profile is not None else None
            ),
        )
        write_active_config(pair_context.config, dest_dir=pair_root)
        apply_native_thread_limits(policy_bundle.analysis)
        try:
            analysis.run_root_pair_analysis(
                pair_context,
                force=force,
                manifest_path=manifest_path,
                oracle_game_profile=oracle_game_profile,
            )
        except Exception as exc:  # noqa: BLE001
            pair_error = f"{type(exc).__name__}: {exc}"
    pair_stage_states: dict[str, str] = {}
    if pair_context is not None and pair_error is None:
        pair_stage_states = _current_plan_states(
            pair_context.config,
            analysis.build_root_pair_stage_plan(pair_context, force=False),
        )
        if any(
            value != CompletionState.COMPLETE_VALID.value for value in pair_stage_states.values()
        ):
            pair_error = "pair workflow contains stale or incomplete canonical stages"
    for seed, result in root_results.items():
        if not result.simulation_ok or not result.analysis_ok:
            continue
        current_lifecycle, current_states = _root_lifecycle_identity(
            result.context,
            analysis.build_root_stage_plan(result.context.config, force=False),
        )
        root_health[str(seed)]["lifecycle_sha256"] = current_lifecycle
        root_health[str(seed)]["stage_states"] = current_states
        if current_lifecycle is None:
            root_health[str(seed)]["analysis"] = "failed"
            root_health[str(seed)][
                "error"
            ] = "root workflow became stale before final health publication"
    root_failures = [
        f"root {seed}: {status['error']}"
        for seed, status in root_health.items()
        if status["analysis"] != "complete"
    ]
    release_audit: dict[str, Any] = {
        "status": "not_run",
        "release_eligible": False,
        "accepted_release_identity": [3, 2, 2, 2, 2, 2],
        "artifact_roots": [],
        "run_contexts": {},
        "failures": ["pair workflow did not reach the final release gate"],
    }
    if not root_failures and pair_context is not None and pair_error is None:
        release_audit = _final_release_gate(
            root_results,
            pair_context,
            code_identity=code_identity,
            allow_oracle_code_identity=oracle_game_profile is not None,
        )
        if release_audit["status"] != "passed":
            pair_error = "final release audit failed: " + str(release_audit["failures"][0])
    overall_status = "complete_success" if not root_failures and pair_error is None else "failed"
    health = {
        "seed_pair": list(seed_pair),
        "status": overall_status,
        "config_sha": cfg.config_sha,
        "pair_public_config_sha256": (
            pair_context.config.config_sha if pair_context is not None else None
        ),
        "root_workflows": root_health,
        "release_audit": release_audit,
        "pair_workflow": {
            "status": "complete" if pair_context is not None and pair_error is None else "failed",
            "analysis_root": str(pair_context.analysis_root) if pair_context else None,
            "error": pair_error or (root_failures[0] if root_failures else None),
            "stage_states": pair_stage_states,
            "run_lineage_sha256": (
                pair_context.config._run_lineage_sha256 if pair_context is not None else None
            ),
        },
    }
    _write_pipeline_health(health_path, health)
    append_manifest_event(
        manifest_path,
        {
            "event": EVENT_RUN_END,
            "status": overall_status,
            "health_artifact": str(health_path),
        },
        run_id=run_id,
        config_sha=cfg.config_sha,
    )
    if root_failures or pair_error is not None:
        raise RuntimeError(pair_error or root_failures[0])


__all__ = ["run_pipeline"]
