"""Canonical root and root-pair statistical workflow orchestration."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from farkle.analysis.stage_runner import StagePlanItem, StageRunContext, StageRunner
from farkle.config import AppConfig
from farkle.orchestration.run_contexts import RootPairRunContext, SeedRunContext
from farkle.utils.stage_completion import stage_done_path

LOGGER = logging.getLogger(__name__)


@dataclass
class StageLogger:
    """Standardized logging helper for analysis stages."""

    stage: str
    logger: logging.Logger = LOGGER

    def start(self, **extra: object) -> None:
        self.logger.info("Analysis stage start", extra={"stage": self.stage, **extra})

    def missing_dependency(self, dependency: str, *, error: str | None = None) -> None:
        payload = {"stage": self.stage, "missing_module": dependency, "status": "SKIPPED"}
        if error:
            payload["missing"] = error
        self.logger.info("Analysis module unavailable", extra=payload)

    def missing_input(self, reason: str, **extra: object) -> None:
        self.logger.info(
            "Analysis stage skipped: %s",
            self.stage,
            extra={"stage": self.stage, "reason": reason, "status": "SKIPPED", **extra},
        )


def stage_logger(stage: str, *, logger: logging.Logger | None = None) -> StageLogger:
    """Construct a stage logger."""

    return StageLogger(stage=stage, logger=logger or LOGGER)


def _optional_import(module: str, *, stage_log: StageLogger | None = None) -> ModuleType | None:
    """Import an optional closeout module and report only a missing module."""

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        (stage_log or stage_logger("analysis")).missing_dependency(module, error=str(exc))
        return None


def _manifest_metadata(cfg: AppConfig, *, execution_scope: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "results_dir": str(cfg.results_root),
        "analysis_dir": str(cfg.analysis_dir),
        "execution_scope": execution_scope,
    }
    if cfg.config_sha:
        payload["config_sha"] = cfg.config_sha
    return payload


def build_root_stage_plan(
    cfg: AppConfig,
    *,
    force: bool = False,
    rng_lags: Sequence[int] | None = None,
    run_rng_diagnostics: bool | None = None,
) -> list[StagePlanItem]:
    """Build the root-local workflow, which ends after descriptive screening."""

    from farkle.analysis import (
        combine,
        curate,
        game_stats,
        hgb_feat,
        ingest,
        metrics,
        screening,
        trueskill,
    )
    from farkle.analysis import rng_diagnostics as rng_module

    use_rng = (
        not cfg.analysis.disable_rng_diagnostics
        if run_rng_diagnostics is None
        else bool(run_rng_diagnostics)
    )
    plan = [
        StagePlanItem("ingest", ingest.run),
        StagePlanItem("curate", curate.run),
        StagePlanItem("combine", combine.run),
        StagePlanItem("metrics", metrics.run),
        StagePlanItem("game_stats", lambda inner: game_stats.run(inner, force=force)),
    ]
    if use_rng:
        plan.append(
            StagePlanItem(
                "rng_diagnostics",
                lambda inner: rng_module.run(inner, lags=rng_lags, force=force),
            )
        )
    plan.extend(
        [
            StagePlanItem("trueskill", trueskill.run),
            StagePlanItem("hgb", hgb_feat.run),
            StagePlanItem("screening", lambda inner: screening.run(inner, force=force)),
        ]
    )
    return plan


def _h2h_tail_plan(
    cfg: AppConfig,
    *,
    force: bool,
    execution_scope: str,
) -> list[StagePlanItem]:
    """Build the common single-root or root-pair H2H tail."""

    from farkle.analysis.candidate_family import freeze_h2h_candidate_family
    from farkle.analysis.dominance import build_dominance_outputs
    from farkle.analysis.h2h_inference import run_h2h_inference
    from farkle.analysis.h2h_schedule import execute_h2h_schedule, plan_h2h_schedule
    from farkle.analysis.structure_agreement import run as run_structure_agreement
    from farkle.analysis.structure_reporting import run as run_structure_reporting

    def _candidate_freeze(inner: AppConfig) -> None:
        freeze_h2h_candidate_family(inner, force=force)

    def _power(inner: AppConfig) -> None:
        plan_h2h_schedule(inner, force=force)

    def _execute(inner: AppConfig) -> None:
        execute_h2h_schedule(inner, n_jobs=inner.analysis.n_jobs)

    def _inference(inner: AppConfig) -> None:
        run_h2h_inference(inner, force=force)

    def _digest(inner: AppConfig) -> None:
        build_dominance_outputs(inner, force=force)

    def _agreement(inner: AppConfig) -> None:
        run_structure_agreement(inner, force=force, execution_scope=execution_scope)

    def _reporting(inner: AppConfig) -> None:
        run_structure_reporting(inner, force=force, execution_scope=execution_scope)

    return [
        StagePlanItem(
            "candidate_freeze",
            _candidate_freeze,
            metadata={"execution_scope": execution_scope},
            required_outputs=(
                cfg.h2h_candidate_family_path(),
                cfg.h2h_candidate_family_manifest_path(),
            ),
            completion_stamp=stage_done_path(cfg.stage_dir("candidate_freeze"), "candidate_freeze"),
        ),
        StagePlanItem(
            "h2h_power",
            _power,
            metadata={"execution_scope": execution_scope},
            required_outputs=(cfg.h2h_power_plan_path(),),
            completion_stamp=stage_done_path(cfg.stage_dir("h2h_power"), "h2h_power"),
        ),
        StagePlanItem(
            "h2h_execute",
            _execute,
            metadata={"execution_scope": execution_scope},
            required_outputs=(
                cfg.h2h_execution_state_path(),
                cfg.h2h_order_counts_path(),
            ),
            completion_stamp=stage_done_path(cfg.stage_dir("h2h_execute"), "h2h_execute"),
        ),
        StagePlanItem(
            "h2h_inference",
            _inference,
            metadata={"execution_scope": execution_scope},
            required_outputs=(
                cfg.h2h_combined_order_counts_path(),
                cfg.h2h_pairwise_inference_path(),
                cfg.h2h_root_pairwise_diagnostics_path(),
                cfg.h2h_root_agreement_path(),
            ),
            completion_stamp=stage_done_path(cfg.stage_dir("h2h_inference"), "h2h_inference"),
        ),
        StagePlanItem(
            "h2h_digest",
            _digest,
            metadata={"execution_scope": execution_scope},
            required_outputs=(
                cfg.h2h_dominance_edges_path(),
                cfg.h2h_cycle_groups_path(),
                cfg.h2h_dominance_fronts_path(),
                cfg.h2h_dominance_summary_path(),
            ),
            completion_stamp=stage_done_path(cfg.stage_dir("h2h_digest"), "h2h_digest"),
        ),
        StagePlanItem(
            "agreement",
            _agreement,
            metadata={"execution_scope": execution_scope},
            required_outputs=(
                cfg.structure_agreement_pairs_path(),
                cfg.structure_agreement_summary_path(),
            ),
            completion_stamp=stage_done_path(cfg.stage_dir("agreement"), "structure_agreement"),
        ),
        StagePlanItem(
            "reporting",
            _reporting,
            metadata={"execution_scope": execution_scope},
            required_outputs=(
                cfg.structure_report_json_path(),
                cfg.structure_report_markdown_path(),
                cfg.structure_report_plot_path(),
                cfg.migration_report_path(),
            ),
            completion_stamp=stage_done_path(cfg.stage_dir("reporting"), "structure_reporting"),
        ),
    ]


def build_single_root_h2h_tail_plan(
    cfg: AppConfig,
    *,
    force: bool = False,
) -> list[StagePlanItem]:
    """Build an explicitly labelled H2H tail for a standalone root run."""

    roots = tuple(int(root) for root in (cfg.sim.seed_list or [cfg.sim.seed]))
    if roots != (int(cfg.sim.seed),):
        raise ValueError(f"single-root H2H requires one configured root, found {roots}")
    return _h2h_tail_plan(cfg, force=force, execution_scope="single_root")


def build_root_pair_stage_plan(
    context: RootPairRunContext,
    *,
    force: bool = False,
) -> list[StagePlanItem]:
    """Build the one-time pair workflow from root combination through reporting."""

    from farkle.analysis import trueskill
    from farkle.analysis.root_stability import RootBatchCell, build_two_root_stability

    cfg = context.config
    required_k = tuple(sorted({int(k) for k in cfg.sim.n_players_list}))
    cells = [
        RootBatchCell(
            root_seed=root_context.seed,
            k=k,
            path=root_context.config.metrics_all_player_batch_path(k),
        )
        for root_context in context.root_contexts
        for k in required_k
    ]

    def _root_stability(inner: AppConfig) -> None:
        build_two_root_stability(inner, cells, force=force)

    def _pair_trueskill(inner: AppConfig) -> None:
        trueskill.run_root_pair(inner, context.root_contexts, force=force)

    root_stability_outputs = (
        *(cfg.root_combined_performance_by_k_path(k) for k in required_k),
        cfg.root_combined_performance_across_k_path(),
        cfg.root_discrepancies_path(),
        cfg.root_joint_discrepancy_path(),
        cfg.root_rank_stability_path(),
        cfg.root_top_n_stability_path(),
        cfg.root_bootstrap_top_n_inclusion_path(),
        cfg.root_control_movement_path(),
        cfg.root_shortlist_changes_path(),
        cfg.root_matched_count_convergence_path(),
        cfg.root_half_drift_path(),
    )

    return [
        StagePlanItem(
            "root_stability",
            _root_stability,
            metadata={"root_pair": list(context.root_pair)},
            required_outputs=root_stability_outputs,
            completion_stamp=stage_done_path(cfg.stage_dir("root_stability"), "root_stability"),
        ),
        StagePlanItem(
            "trueskill",
            _pair_trueskill,
            metadata={"root_pair": list(context.root_pair), "role": "candidate_contribution"},
            required_outputs=(cfg.trueskill_candidate_contribution_path(),),
            completion_stamp=stage_done_path(
                cfg.stage_dir("trueskill"), "trueskill_percentile_contribution"
            ),
        ),
        *_h2h_tail_plan(cfg, force=force, execution_scope="root_pair"),
    ]


def _run_plan(
    cfg: AppConfig,
    plan: Sequence[StagePlanItem],
    *,
    run_label: str,
    execution_scope: str,
    manifest_path: Path | None = None,
) -> None:
    path = manifest_path or (cfg.analysis_dir / cfg.manifest_name)
    metadata = _manifest_metadata(cfg, execution_scope=execution_scope)
    StageRunner.run(
        plan,
        StageRunContext(
            config=cfg,
            manifest_path=path,
            run_label=run_label,
            run_metadata=metadata,
            run_end_metadata=metadata,
            continue_on_error=False,
            logger=LOGGER,
        ),
        raise_on_failure=True,
    )


def run_root_analysis(
    cfg: AppConfig,
    *,
    force: bool = False,
    manifest_path: Path | None = None,
    rng_lags: Sequence[int] | None = None,
    run_rng_diagnostics: bool | None = None,
) -> None:
    """Run one root through screening and diagnostics, then stop."""

    _run_plan(
        cfg,
        build_root_stage_plan(
            cfg,
            force=force,
            rng_lags=rng_lags,
            run_rng_diagnostics=run_rng_diagnostics,
        ),
        run_label=f"root_workflow_{cfg.sim.seed}",
        execution_scope="root",
        manifest_path=manifest_path,
    )


def run_single_root_analysis(
    cfg: AppConfig,
    *,
    force: bool = False,
    manifest_path: Path | None = None,
    rng_lags: Sequence[int] | None = None,
    run_rng_diagnostics: bool | None = None,
) -> None:
    """Run a standalone root and append its explicitly labelled H2H tail."""

    run_root_analysis(
        cfg,
        force=force,
        manifest_path=manifest_path,
        rng_lags=rng_lags,
        run_rng_diagnostics=run_rng_diagnostics,
    )
    _run_plan(
        cfg,
        build_single_root_h2h_tail_plan(cfg, force=force),
        run_label=f"single_root_h2h_{cfg.sim.seed}",
        execution_scope="single_root",
        manifest_path=manifest_path,
    )


def run_root_pair_analysis(
    context: RootPairRunContext,
    *,
    force: bool = False,
    manifest_path: Path | None = None,
) -> None:
    """Run the root-pair workflow exactly once at the pair analysis root."""

    _run_plan(
        context.config,
        build_root_pair_stage_plan(context, force=force),
        run_label=f"root_pair_workflow_{context.root_pair[0]}_{context.root_pair[1]}",
        execution_scope="root_pair",
        manifest_path=manifest_path,
    )


def run_all(
    cfg: AppConfig,
    *,
    run_rng_diagnostics: bool | None = None,
    rng_lags: Sequence[int] | None = None,
    allow_missing_upstream: bool = False,
) -> None:
    """Run the public single-root analytics workflow and labelled H2H tail."""

    if allow_missing_upstream:
        raise ValueError("canonical workflows do not permit missing upstream artifacts")
    roots = tuple(int(root) for root in (cfg.sim.seed_list or [cfg.sim.seed]))
    if len(roots) != 1:
        raise ValueError("two-root analysis must use the two-seed-pipeline command")
    run_single_root_analysis(
        cfg,
        rng_lags=rng_lags,
        run_rng_diagnostics=run_rng_diagnostics,
    )


__all__ = [
    "RootPairRunContext",
    "SeedRunContext",
    "StageLogger",
    "build_root_pair_stage_plan",
    "build_root_stage_plan",
    "build_single_root_h2h_tail_plan",
    "run_all",
    "run_root_analysis",
    "run_root_pair_analysis",
    "run_single_root_analysis",
    "stage_logger",
]
