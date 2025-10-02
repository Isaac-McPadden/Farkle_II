"""Lightweight orchestrator for downstream statistical analyses."""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

from farkle.analysis.analysis_config import PipelineCfg
from farkle.config import AppConfig

LOGGER = logging.getLogger(__name__)


def _optional_import(module: str) -> ModuleType | None:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in tests
        LOGGER.info(
            "Analytics module skipped due to missing dependency",
            extra={"stage": "analysis", "module": module, "missing": str(exc)},
        )
        return None


def _pipeline_cfg(cfg: AppConfig | PipelineCfg) -> AppConfig | PipelineCfg:
    if hasattr(cfg, "results_dir") and hasattr(cfg, "analysis_dir") or isinstance(cfg, PipelineCfg):
        return cfg
    elif hasattr(cfg, "analysis") and isinstance(cfg.analysis, PipelineCfg):
        return cfg.analysis
    else:
        return cfg


def run_all(cfg: AppConfig | PipelineCfg) -> None:
    """Run every analytics pass in sequence."""
    cfg = _pipeline_cfg(cfg)
    LOGGER.info("Analytics: starting all modules", extra={"stage": "analysis"})
    ts_mod = _optional_import("farkle.analysis.trueskill")
    if cfg.run_trueskill and ts_mod is not None:
        ts_mod.run(cfg)
    else:
        LOGGER.info(
            "Analytics: skipping trueskill",
            extra={
                "stage": "analysis",
                "reason": "run_trueskill=False" if not cfg.run_trueskill else "unavailable",
            },
        )

    h2h_mod = _optional_import("farkle.analysis.head2head")
    if cfg.run_head2head and h2h_mod is not None:
        h2h_mod.run(cfg)
    else:
        LOGGER.info(
            "Analytics: skipping head-to-head",
            extra={
                "stage": "analysis",
                "reason": "run_head2head=False" if not cfg.run_head2head else "unavailable",
            },
        )

    hgb_mod = _optional_import("farkle.analysis.hgb_feat")
    if cfg.run_hgb and hgb_mod is not None:
        hgb_mod.run(cfg)
    else:
        LOGGER.info(
            "Analytics: skipping hist gradient boosting",
            extra={
                "stage": "analysis",
                "reason": "run_hgb=False" if not cfg.run_hgb else "unavailable",
            },
        )
    LOGGER.info("Analytics: all modules finished", extra={"stage": "analysis"})
