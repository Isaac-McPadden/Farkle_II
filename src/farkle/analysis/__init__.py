"""Lightweight orchestrator for downstream statistical analyses."""

from __future__ import annotations

import logging

from farkle.analysis import head2head as _h2h
from farkle.analysis import hgb_feat as _hgb
from farkle.analysis import trueskill as _ts
from farkle.analysis.analysis_config import PipelineCfg
from farkle.app_config import AppConfig

LOGGER = logging.getLogger(__name__)


def _pipeline_cfg(cfg: AppConfig | PipelineCfg) -> PipelineCfg:
    return cfg.analysis if isinstance(cfg, AppConfig) else cfg


def run_all(cfg: AppConfig | PipelineCfg) -> None:
    """Run every analytics pass in sequence."""
    cfg = _pipeline_cfg(cfg)
    LOGGER.info("Analytics: starting all modules", extra={"stage": "analysis"})
    if cfg.run_trueskill:
        _ts.run(cfg)
    else:
        LOGGER.info(
            "Analytics: skipping trueskill",
            extra={"stage": "analysis", "reason": "run_trueskill=False"},
        )

    if cfg.run_head2head:
        _h2h.run(cfg)
    else:
        LOGGER.info(
            "Analytics: skipping head-to-head",
            extra={"stage": "analysis", "reason": "run_head2head=False"},
        )

    if cfg.run_hgb:
        _hgb.run(cfg)
    else:
        LOGGER.info(
            "Analytics: skipping hist gradient boosting",
            extra={"stage": "analysis", "reason": "run_hgb=False"},
        )
    LOGGER.info("Analytics: all modules finished", extra={"stage": "analysis"})
