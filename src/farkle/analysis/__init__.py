# src/farkle/analysis/__init__.py
"""Lightweight orchestrator for downstream statistical analyses."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

from farkle.analysis.stage_registry import resolve_interseed_stage_layout
from farkle.analysis.stage_runner import StagePlanItem, StageRunContext, StageRunner
from farkle.config import AppConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class StageLogger:
    """Standardized logging helper for analytics stages."""

    stage: str
    logger: logging.Logger = LOGGER

    def start(self, **extra: object) -> None:
        """Log that a stage started running."""

        self.logger.info(
            "Analytics stage start",
            extra={"stage": self.stage, **extra},
        )

    def missing_dependency(self, dependency: str, *, error: str | None = None) -> None:
        """Log that a dependency is missing for a stage."""

        payload = {"stage": self.stage, "missing_module": dependency, "status": "SKIPPED"}
        if error:
            payload["missing"] = error
        self.logger.info(
            "Analytics module skipped due to missing dependency",
            extra=payload,
        )

    def missing_input(self, reason: str, **extra: object) -> None:
        """Log that a stage skipped due to missing or invalid inputs."""

        self.logger.info(
            "Analytics: skipping %s",
            self.stage,
            extra={"stage": self.stage, "reason": reason, "status": "SKIPPED", **extra},
        )


def stage_logger(stage: str, *, logger: logging.Logger | None = None) -> StageLogger:
    """Construct a :class:`StageLogger` for the given stage name."""

    return StageLogger(stage=stage, logger=logger or LOGGER)


def _optional_import(module: str, *, stage_log: StageLogger | None = None) -> ModuleType | None:
    """Attempt to import an analytics module while tolerating missing deps.

    Args:
        module: Fully qualified module path to import.
        stage_log: Helper used to report missing dependencies.

    Returns:
        Imported module object, or ``None`` when the dependency is absent.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in tests
        (stage_log or stage_logger("analysis")).missing_dependency(module, error=str(exc))
        return None


def run_seed_summaries(cfg: AppConfig, *, force: bool = False) -> None:
    """Wrapper around :mod:`farkle.analysis.seed_summaries`."""
    from farkle.analysis import seed_summaries

    seed_summaries.run(cfg, force=force)


def run_coverage_by_k(cfg: AppConfig, *, force: bool = False) -> None:
    """Wrapper around :mod:`farkle.analysis.coverage_by_k`."""
    from farkle.analysis import coverage_by_k

    coverage_by_k.run(cfg, force=force)


def run_meta(cfg: AppConfig, *, force: bool = False) -> None:
    """Wrapper around :mod:`farkle.analysis.meta`."""
    from farkle.analysis import meta

    meta.run(cfg, force=force)


def run_variance(cfg: AppConfig, *, force: bool = False) -> None:
    """Wrapper around :mod:`farkle.analysis.variance`."""
    from farkle.analysis import variance

    variance.run(cfg, force=force)


def run_h2h_tier_trends(cfg: AppConfig, *, force: bool = False) -> None:
    """Wrapper around :mod:`farkle.analysis.h2h_tier_trends`."""
    from farkle.analysis import h2h_tier_trends

    h2h_tier_trends.run(cfg, force=force)


def run_seed_symmetry(cfg: AppConfig, *, force: bool = False) -> None:
    """Wrapper around :mod:`farkle.analysis.seed_symmetry`."""
    from farkle.analysis import seed_symmetry

    seed_symmetry.run(cfg, force=force)


def run_interseed_analysis(
    cfg: AppConfig,
    *,
    force: bool = False,
    manifest_path: Path | None = None,
    rng_lags: Sequence[int] | None = None,
    run_rng_diagnostics: bool | None = None,
) -> None:
    """Run interseed analytics in order (rng → variance → meta → trueskill → agreement → summary)."""
    previous_layout = cfg._stage_layout
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))
    resolved_rng_diagnostics = (
        run_rng_diagnostics
        if run_rng_diagnostics is not None
        else not cfg.analysis.disable_rng_diagnostics
    )

    def _require_interseed_inputs(
        stage: str, runner: Callable[[AppConfig], None]
    ) -> Callable[[AppConfig], None]:
        def _wrapped(inner_cfg: AppConfig) -> None:
            ready, reason = inner_cfg.interseed_ready()
            if not ready:
                stage_logger(stage, logger=LOGGER).missing_input(reason)
                return
            runner(inner_cfg)

        return _wrapped

    def _rng_diagnostics(cfg: AppConfig) -> None:
        if not resolved_rng_diagnostics:
            reason = (
                "disabled by CLI flag"
                if run_rng_diagnostics is False
                else "disabled by config"
            )
            stage_logger("rng_diagnostics", logger=LOGGER).missing_input(reason)
            return
        from farkle.analysis import rng_diagnostics

        rng_diagnostics.run(cfg, lags=rng_lags, force=force)

    def _variance(cfg: AppConfig) -> None:
        run_variance(cfg, force=force)

    def _interseed_game_stats(cfg: AppConfig) -> None:
        stage_log = stage_logger("interseed_game_stats", logger=LOGGER)
        stats_mod = _optional_import(
            "farkle.analysis.game_stats_interseed",
            stage_log=stage_log,
        )
        if stats_mod is None:
            return
        stats_mod.run(cfg, force=force)

    def _meta(cfg: AppConfig) -> None:
        run_meta(cfg, force=force)

    def _trueskill(cfg: AppConfig) -> None:
        stage_log = stage_logger("trueskill", logger=LOGGER)
        ts_mod = _optional_import("farkle.analysis.trueskill", stage_log=stage_log)
        if ts_mod is None:
            return
        ts_mod.run(cfg)

    def _agreement(cfg: AppConfig) -> None:
        ratings_pooled_path = cfg.trueskill_path("ratings_k_weighted.parquet")
        if not ratings_pooled_path.exists() or ratings_pooled_path.stat().st_size <= 0:
            stage_logger("agreement", logger=LOGGER).missing_input(
                "missing required TrueSkill pooled ratings input",
                dependency="trueskill.ratings_k_weighted.parquet",
                path=str(ratings_pooled_path),
            )
            return
        stage_log = stage_logger("agreement", logger=LOGGER)
        agreement_mod = _optional_import("farkle.analysis.agreement", stage_log=stage_log)
        if agreement_mod is None:
            return
        agreement_mod.run(cfg)

    def _interseed_summary(cfg: AppConfig) -> None:
        stage_log = stage_logger("interseed", logger=LOGGER)
        interseed_mod = _optional_import(
            "farkle.analysis.interseed_analysis",
            stage_log=stage_log,
        )
        if interseed_mod is not None:
            interseed_mod.run(
                cfg,
                force=force,
                run_stages=False,
                run_rng_diagnostics=run_rng_diagnostics,
            )
            return

    plan = [
        StagePlanItem(
            "rng_diagnostics",
            _require_interseed_inputs("rng_diagnostics", _rng_diagnostics),
        ),
        StagePlanItem("variance", _require_interseed_inputs("variance", _variance)),
        StagePlanItem(
            "interseed_game_stats",
            _require_interseed_inputs("interseed_game_stats", _interseed_game_stats),
        ),
        StagePlanItem("meta", _require_interseed_inputs("meta", _meta)),
        StagePlanItem("trueskill", _require_interseed_inputs("trueskill", _trueskill)),
        StagePlanItem("agreement", _require_interseed_inputs("agreement", _agreement)),
        StagePlanItem(
            "interseed",
            _require_interseed_inputs("interseed", _interseed_summary),
        ),
    ]
    manifest_path = manifest_path or (cfg.analysis_dir / cfg.manifest_name)
    context = StageRunContext(
        config=cfg,
        manifest_path=manifest_path,
        run_label="interseed_analysis",
        run_metadata=_run_manifest_metadata(cfg),
        run_end_metadata=_run_manifest_metadata(cfg),
        continue_on_error=False,
        logger=LOGGER,
    )
    try:
        StageRunner.run(plan, context, raise_on_failure=True)
    finally:
        cfg._stage_layout = previous_layout


def _run_manifest_metadata(cfg: AppConfig) -> dict[str, Any]:
    payload = {
        "results_dir": str(cfg.results_root),
        "analysis_dir": str(cfg.analysis_dir),
    }
    config_sha = getattr(cfg, "config_sha", None)
    if config_sha:
        payload["config_sha"] = config_sha
    return payload


def run_single_seed_analysis(
    cfg: AppConfig,
    *,
    force: bool = False,
    manifest_path: Path | None = None,
) -> None:
    """Run per-seed analytics in order (seed summaries → coverage_by_k → trueskill → tiering → head2head → seed_symmetry → post_h2h → hgb)."""
    def _seed_summaries(cfg: AppConfig) -> None:
        run_seed_summaries(cfg, force=force)

    def _trueskill(cfg: AppConfig) -> None:
        stage_log = stage_logger("trueskill", logger=LOGGER)
        ts_mod = _optional_import("farkle.analysis.trueskill", stage_log=stage_log)
        if ts_mod is None:
            return
        ts_mod.run(cfg)

    def _tiering(cfg: AppConfig) -> None:
        stage_log = stage_logger("tiering", logger=LOGGER)
        freq_mod = _optional_import("farkle.analysis.tiering_report", stage_log=stage_log)
        if freq_mod is None:
            return
        freq_mod.run(cfg)

    def _head2head(cfg: AppConfig) -> None:
        stage_log = stage_logger("head2head", logger=LOGGER)
        h2h_mod = _optional_import("farkle.analysis.head2head", stage_log=stage_log)
        if h2h_mod is None:
            return
        h2h_mod.run(cfg)

    def _post_h2h(cfg: AppConfig) -> None:
        stage_log = stage_logger("post_h2h", logger=LOGGER)
        post_h2h_mod = _optional_import("farkle.analysis.h2h_analysis", stage_log=stage_log)
        if post_h2h_mod is None:
            return
        post_h2h_mod.run_post_h2h(cfg)

    def _seed_symmetry(cfg: AppConfig) -> None:
        run_seed_symmetry(cfg, force=force)

    def _hgb(cfg: AppConfig) -> None:
        stage_log = stage_logger("hgb", logger=LOGGER)
        hgb_mod = _optional_import("farkle.analysis.hgb_feat", stage_log=stage_log)
        if hgb_mod is None:
            return
        hgb_mod.run(cfg)

    plan = [
        StagePlanItem("seed_summaries", _seed_summaries),
        StagePlanItem("coverage_by_k", lambda cfg: run_coverage_by_k(cfg, force=force)),
        StagePlanItem("trueskill", _trueskill),
        StagePlanItem("tiering", _tiering),
        StagePlanItem(
            "head2head",
            _head2head,
            required_outputs=(
                cfg.head2head_stage_dir / "bonferroni_pairwise.parquet",
                cfg.head2head_stage_dir / "bonferroni_pairwise_ordered.parquet",
                cfg.head2head_stage_dir / "bonferroni_selfplay_symmetry.parquet",
                cfg.head2head_stage_dir / "bonferroni_head2head.done.json",
            ),
        ),
        StagePlanItem("seed_symmetry", _seed_symmetry),
        StagePlanItem("post_h2h", _post_h2h),
        StagePlanItem("hgb", _hgb),
    ]
    manifest_path = manifest_path or (cfg.analysis_dir / cfg.manifest_name)
    context = StageRunContext(
        config=cfg,
        manifest_path=manifest_path,
        run_label="single_seed_analysis",
        run_metadata=_run_manifest_metadata(cfg),
        run_end_metadata=_run_manifest_metadata(cfg),
        continue_on_error=False,
        logger=LOGGER,
    )
    StageRunner.run(plan, context, raise_on_failure=True)


def run_all(
    cfg: AppConfig,
    *,
    run_rng_diagnostics: bool | None = None,
    rng_lags: Sequence[int] | None = None,
) -> None:
    """Run single-seed analytics first, then interseed analytics if enabled."""
    LOGGER.info("Analytics: starting all modules", extra={"stage": "analysis"})
    run_single_seed_analysis(cfg)
    run_interseed_analysis(
        cfg,
        run_rng_diagnostics=run_rng_diagnostics,
        rng_lags=rng_lags,
    )
    run_h2h_tier_trends(cfg)
    LOGGER.info("Analytics: all modules finished", extra={"stage": "analysis"})
