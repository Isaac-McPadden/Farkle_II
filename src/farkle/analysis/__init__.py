# src/farkle/analysis/__init__.py
"""Lightweight orchestrator for downstream statistical analyses."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

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
    def _require_interseed_inputs(stage: str, runner: Callable[[AppConfig], None]) -> Callable[[AppConfig], None]:
        def _wrapped(inner_cfg: AppConfig) -> None:
            ready, reason = inner_cfg.interseed_ready()
            if not ready:
                stage_logger(stage, logger=LOGGER).missing_input(reason)
                return
            runner(inner_cfg)

        return _wrapped

    def _variance(cfg: AppConfig) -> None:
        run_variance(cfg, force=force)

    def _meta(cfg: AppConfig) -> None:
        run_meta(cfg, force=force)

    def _trueskill(cfg: AppConfig) -> None:
        stage_log = stage_logger("trueskill", logger=LOGGER)
        ts_mod = _optional_import("farkle.analysis.trueskill", stage_log=stage_log)
        if ts_mod is None:
            return
        ts_mod.run(cfg)

    def _agreement(cfg: AppConfig) -> None:
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
            interseed_mod.run(cfg, force=force, run_stages=False)
            return

    plan = [
        StagePlanItem("variance", _require_interseed_inputs("variance", _variance)),
        StagePlanItem("meta", _require_interseed_inputs("meta", _meta)),
        StagePlanItem("trueskill", _require_interseed_inputs("trueskill", _trueskill)),
        StagePlanItem("agreement", _require_interseed_inputs("agreement", _agreement)),
        StagePlanItem("interseed", _require_interseed_inputs("interseed", _interseed_summary)),
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
        stage_log = stage_logger("tiering", logger=LOGGER)
        freq_mod = _optional_import("farkle.analysis.tiering_report", stage_log=stage_log)
        if freq_mod is None:
            return
        freq_mod.run(cfg)

    def _trueskill(cfg: AppConfig) -> None:
        stage_log = stage_logger("trueskill", logger=LOGGER)
        ts_mod = _optional_import("farkle.analysis.trueskill", stage_log=stage_log)
        if ts_mod is None:
            return
        ts_mod.run(cfg)

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

    def _hgb(cfg: AppConfig) -> None:
        stage_log = stage_logger("hgb", logger=LOGGER)
        hgb_mod = _optional_import("farkle.analysis.hgb_feat", stage_log=stage_log)
        if hgb_mod is None:
            return
        hgb_mod.run(cfg)

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
