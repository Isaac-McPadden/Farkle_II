"""Two-root simulation followed by one canonical root-pair analysis workflow."""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

from farkle import analysis
from farkle.analysis.stage_runner import StageRunContext, StageRunner
from farkle.config import AppConfig, apply_dot_overrides, assign_config_sha, load_app_config
from farkle.orchestration.run_contexts import RootPairRunContext, SeedRunContext
from farkle.orchestration.seed_utils import (
    prepare_seed_config,
    resolve_seed_pair_args,
    seed_has_completion_markers,
    seed_pair_root,
    seed_pair_seed_root,
    write_active_config,
)
from farkle.simulation import runner
from farkle.utils.logging import setup_info_logging
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


@dataclass(frozen=True)
class _PerSeedPolicyBundle:
    """Resolved process-worker policies for one root workflow."""

    simulation: StageParallelPolicy
    ingest: StageParallelPolicy
    analysis: StageParallelPolicy

    def as_metadata(self) -> dict[str, dict[str, int]]:
        return {
            name: {
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
            )
        }


def _per_seed_worker_budget(total_workers: int, seed_count: int) -> int:
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    return max(1, total_workers // seed_count)


def _derive_per_seed_job_budgets(cfg: AppConfig, seed_count: int) -> _PerSeedPolicyBundle:
    total_workers = normalize_n_jobs(cfg.sim.n_jobs, default=1)
    concurrency = seed_count if cfg.orchestration.parallel_seeds else 1
    per_root_workers = _per_seed_worker_budget(total_workers, concurrency)
    bundle = _PerSeedPolicyBundle(
        simulation=resolve_stage_parallel_policy(
            "simulation",
            cfg.sim,
            n_jobs_override=per_root_workers,
        ),
        ingest=resolve_stage_parallel_policy(
            "ingest",
            cfg.ingest,
            n_jobs_override=min(per_root_workers, normalize_n_jobs(cfg.ingest.n_jobs)),
        ),
        analysis=resolve_stage_parallel_policy(
            "analysis",
            cfg.analysis,
            n_jobs_override=per_root_workers,
        ),
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
    return root_cfg


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
) -> _SeedRunStatus:
    root_cfg = _build_seed_cfg(
        cfg,
        seed_pair=seed_pair,
        seed=seed,
        policy_bundle=policy_bundle,
    )
    context = SeedRunContext.from_config(root_cfg)
    write_active_config(root_cfg)
    apply_native_thread_limits(policy_bundle.simulation)
    try:
        if not force and seed_has_completion_markers(root_cfg):
            simulation_event = "root_simulation_skipped_complete"
        else:
            runner.run_tournament(root_cfg, force=force)
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
    return _SeedRunStatus(
        seed=seed,
        context=context,
        simulation_ok=True,
        analysis_ok=True,
    )


def _write_pipeline_health(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as temporary:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def run_pipeline(
    cfg: AppConfig,
    *,
    seed_pair: tuple[int, int],
    force: bool = False,
) -> None:
    """Run both roots, then combination, H2H, agreement, and reporting once."""

    if len(set(seed_pair)) != 2:
        raise ValueError(f"two-seed-pipeline requires two distinct roots, found {seed_pair}")
    if cfg.config_sha is None:
        assign_config_sha(cfg)
    pair_root = seed_pair_root(cfg, seed_pair)
    manifest_path = pair_root / "two_seed_pipeline_manifest.jsonl"
    health_path = pair_root / "pipeline_health.json"
    run_id = make_run_id(f"two_seed_pipeline_{seed_pair[0]}_{seed_pair[1]}")
    validate_manifest_contract(manifest_path)
    policy_bundle = _derive_per_seed_job_budgets(cfg, len(seed_pair))
    append_manifest_event(
        manifest_path,
        {
            "event": EVENT_RUN_START,
            "seed_pair": list(seed_pair),
            "results_dir": str(pair_root),
            "pair_analysis_dir": str(pair_root / cfg.io.analysis_subdir),
            "resolved_policy": policy_bundle.as_metadata(),
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
            )
            for seed in seed_pair
        }
    root_health = {
        str(seed): {
            "simulation": "complete" if result.simulation_ok else "failed",
            "analysis": "complete" if result.analysis_ok else "failed",
            "error": result.simulation_error or result.analysis_error,
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
        write_active_config(pair_context.config, dest_dir=pair_root)
        apply_native_thread_limits(policy_bundle.analysis)
        try:
            analysis.run_root_pair_analysis(
                pair_context,
                force=force,
                manifest_path=manifest_path,
            )
        except Exception as exc:  # noqa: BLE001
            pair_error = f"{type(exc).__name__}: {exc}"
    overall_status = "complete_success" if not root_failures and pair_error is None else "failed"
    health = {
        "seed_pair": list(seed_pair),
        "status": overall_status,
        "config_sha": cfg.config_sha,
        "root_workflows": root_health,
        "pair_workflow": {
            "status": "complete" if pair_context is not None and pair_error is None else "failed",
            "analysis_root": str(pair_context.analysis_root) if pair_context else None,
            "error": pair_error or (root_failures[0] if root_failures else None),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="farkle two-seed-pipeline")
    parser.add_argument("--config", type=Path, default=Path("configs/fast_config.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--seed-a", type=int)
    parser.add_argument("--seed-b", type=int)
    parser.add_argument("--seed-pair", type=int, nargs=2, metavar=("A", "B"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--parallel-seeds", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_info_logging()
    cfg = load_app_config(Path(args.config), seed_list_len=2)
    cfg = apply_dot_overrides(cfg, list(args.overrides or []))
    if args.parallel_seeds:
        cfg.orchestration.parallel_seeds = True
    override_pair = resolve_seed_pair_args(args, parser)
    if override_pair is not None:
        cfg.sim.seed_list = list(override_pair)
    roots = cfg.sim.populate_seed_list(2)
    assign_config_sha(cfg)
    run_pipeline(cfg, seed_pair=(roots[0], roots[1]), force=args.force)
    return 0


__all__ = ["main", "run_pipeline"]
