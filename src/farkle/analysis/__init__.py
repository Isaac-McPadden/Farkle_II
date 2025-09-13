"""Lightweight orchestrator for downstream statistical analyses."""

from __future__ import annotations

import logging

from farkle.analysis.analysis_config import PipelineCfg
from farkle.analysis import head2head as _h2h
from farkle.analysis import hgb_feat as _hgb
from farkle.analysis import trueskill as _ts
from farkle.app_config import AppConfig

log = logging.getLogger(__name__)


def _pipeline_cfg(cfg: AppConfig | PipelineCfg) -> PipelineCfg:
    return cfg.analysis if isinstance(cfg, AppConfig) else cfg


def run_all(cfg: AppConfig | PipelineCfg) -> None:
    """Run every analytics pass in sequence."""
    cfg = _pipeline_cfg(cfg)
    log.info("Analytics: starting all modules")
    if cfg.run_trueskill:
        _ts.run(cfg)
    else:
        log.info("Analytics: skipping trueskill (run_trueskill=False)")

    if cfg.run_head2head:
        _h2h.run(cfg)
    else:
        log.info("Analytics: skipping head-to-head (run_head2head=False)")

    if cfg.run_hgb:
        _hgb.run(cfg)
    else:
        log.info("Analytics: skipping hist gradient boosting (run_hgb=False)")
    log.info("Analytics: all modules finished")
