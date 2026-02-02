# src/farkle/analysis/__init__.py
"""Lightweight orchestrator for downstream statistical analyses."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from farkle.config import AppConfig
from farkle.analysis.stage_runner import StagePlanItem, StageRunContext, StageRunner

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

        payload = {"stage": self.stage, "missing_module": dependency}
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
            extra={"stage": self.stage, "reason": reason, **extra},
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


def run_meta(cfg: AppConfig, *, force: bool = False) -> None:
    """Wrapper around :mod:`farkle.analysis.meta`."""
    from farkle.analysis import meta

    meta.run(cfg, force=force)


def run_variance(cfg: AppConfig, *, force: bool = False) -> None:
    """Wrapper around :mod:`farkle.analysis.variance`."""
    from farkle.analysis import variance

    variance.run(cfg, force=force)


def run_interseed_analysis(
    cfg: AppConfig,
    *,
    force: bool = False,
    manifest_path: Path | None = None,
) -> None:
    """Run interseed analytics in order (variance → meta → trueskill → agreement → summary)."""
    if not cfg.analysis.run_interseed:
        _skip_message("variance", "run_interseed=False")
        _skip_message("meta", "run_interseed=False")
        _skip_message("trueskill", "run_interseed=False")
        _skip_message("agreement", "run_interseed=False")
        _skip_message("interseed", "run_interseed=False")
        return

    def _variance(cfg: AppConfig) -> None:
        run_variance(cfg, force=force)

    def _meta(cfg: AppConfig) -> None:
        run_meta(cfg, force=force)

    def _trueskill(cfg: AppConfig) -> None:
        ts_mod = _optional_import("farkle.analysis.trueskill")
        if cfg.analysis.run_trueskill and not cfg.analysis.disable_trueskill and ts_mod is not None:
            ts_mod.run(cfg)
            return
        LOGGER.info(
            "Analytics: skipping trueskill",
            extra={
                "stage": "analysis",
                "reason": (
                    "run_trueskill=False"
                    if not cfg.analysis.run_trueskill
                    else "disabled"
                    if cfg.analysis.disable_trueskill
                    else "unavailable"
                ),
            },
        )

    def _agreement(cfg: AppConfig) -> None:
        agreement_mod = _optional_import("farkle.analysis.agreement")
        if (
            cfg.analysis.run_agreement
            and not cfg.analysis.disable_agreement
            and agreement_mod is not None
        ):
            agreement_mod.run(cfg)
            return
        LOGGER.info(
            "Analytics: skipping agreement",
            extra={
                "stage": "analysis",
                "reason": (
                    "run_agreement=False"
                    if not cfg.analysis.run_agreement
                    else "disabled"
                    if cfg.analysis.disable_agreement
                    else "unavailable"
                ),
            },
        )

    def _interseed_summary(cfg: AppConfig) -> None:
        interseed_mod = _optional_import("farkle.analysis.interseed_analysis")
        if interseed_mod is not None:
            interseed_mod.run(cfg, force=force, run_stages=False)
            return
        LOGGER.info(
            "Analytics: skipping interseed summary",
            extra={"stage": "analysis", "reason": "unavailable"},
        )

    plan = [
        StagePlanItem("variance", _variance),
        StagePlanItem("meta", _meta),
        StagePlanItem("trueskill", _trueskill),
        StagePlanItem("agreement", _agreement),
        StagePlanItem("interseed", _interseed_summary),
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
    StageRunner.run(plan, context, raise_on_failure=True)


def _skip_message(step: str, reason: str) -> None:
    """Log a standardized skip message for optional analysis stages.

    Args:
        step: Name of the analytics step being skipped.
        reason: Human-readable explanation for the skip condition.
    """
    LOGGER.info(
        "Analytics: skipping %s",
        step,
        extra={"stage": "analysis", "step": step, "reason": reason},
    )


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
    """Run per-seed analytics in order (seed summaries → tiering → head2head → post_h2h → hgb)."""
    def _seed_summaries(cfg: AppConfig) -> None:
        run_seed_summaries(cfg, force=force)

    def _tiering(cfg: AppConfig) -> None:
        freq_mod = _optional_import("farkle.analysis.tiering_report")
        if (
            getattr(cfg.analysis, "run_frequentist", False)
            and not cfg.analysis.disable_tiering
            and freq_mod is not None
        ):
            freq_mod.run(cfg)
            return
        LOGGER.info(
            "Analytics: skipping tiering report",
            extra={
                "stage": "analysis",
                "reason": (
                    "run_frequentist=False"
                    if not getattr(cfg.analysis, "run_frequentist", False)
                    else "disabled"
                    if cfg.analysis.disable_tiering
                    else "unavailable"
                ),
            },
        )

    def _trueskill(cfg: AppConfig) -> None:
        ts_mod = _optional_import("farkle.analysis.trueskill")
        if cfg.analysis.run_trueskill and not cfg.analysis.disable_trueskill and ts_mod is not None:
            ts_mod.run(cfg)
            return
        LOGGER.info(
            "Analytics: skipping trueskill",
            extra={
                "stage": "analysis",
                "reason": (
                    "run_trueskill=False"
                    if not cfg.analysis.run_trueskill
                    else "disabled"
                    if cfg.analysis.disable_trueskill
                    else "unavailable"
                ),
            },
        )

    def _head2head(cfg: AppConfig) -> None:
        h2h_mod = _optional_import("farkle.analysis.head2head")
        if (
            cfg.analysis.run_head2head
            and not cfg.analysis.disable_head2head
            and h2h_mod is not None
        ):
            h2h_mod.run(cfg)
            return
        LOGGER.info(
            "Analytics: skipping head-to-head",
            extra={
                "stage": "analysis",
                "reason": (
                    "run_head2head=False"
                    if not cfg.analysis.run_head2head
                    else "disabled"
                    if cfg.analysis.disable_head2head
                    else "unavailable"
                ),
            },
        )

    def _post_h2h(cfg: AppConfig) -> None:
        post_h2h_mod = _optional_import("farkle.analysis.h2h_analysis")
        if cfg.analysis.run_post_h2h_analysis and post_h2h_mod is not None:
            post_h2h_mod.run_post_h2h(cfg)
            return
        LOGGER.info(
            "Analytics: skipping post head-to-head analysis",
            extra={
                "stage": "analysis",
                "reason": (
                    "run_post_h2h_analysis=False"
                    if not cfg.analysis.run_post_h2h_analysis
                    else "unavailable"
                ),
            },
        )

    def _hgb(cfg: AppConfig) -> None:
        hgb_mod = _optional_import("farkle.analysis.hgb_feat")
        if cfg.analysis.run_hgb and not cfg.analysis.disable_hgb and hgb_mod is not None:
            hgb_mod.run(cfg)
            return
        LOGGER.info(
            "Analytics: skipping hist gradient boosting",
            extra={
                "stage": "analysis",
                "reason": (
                    "run_hgb=False"
                    if not cfg.analysis.run_hgb
                    else "disabled"
                    if cfg.analysis.disable_hgb
                    else "unavailable"
                ),
            },
        )

    plan = [
        StagePlanItem("seed_summaries", _seed_summaries),
        StagePlanItem("tiering", _tiering),
        StagePlanItem("trueskill", _trueskill),
        StagePlanItem("head2head", _head2head),
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


def run_all(cfg: AppConfig) -> None:
    """Run single-seed analytics first, then interseed analytics if enabled."""
    LOGGER.info("Analytics: starting all modules", extra={"stage": "analysis"})
    run_single_seed_analysis(cfg)
    run_interseed_analysis(cfg)
    LOGGER.info("Analytics: all modules finished", extra={"stage": "analysis"})
