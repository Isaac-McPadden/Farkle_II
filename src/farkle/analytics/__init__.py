"""Lightweight orchestrator for downstream statistical analyses."""

from __future__ import annotations

import logging

from farkle.analysis_config import PipelineCfg
from farkle.analytics import head2head as _h2h
from farkle.analytics import rf_feat as _rf
from farkle.analytics import trueskill as _ts

log = logging.getLogger(__name__)


def run_all(cfg: PipelineCfg) -> None:
    """Run every analytics pass in sequence."""
    log.info("Analytics: starting all modules")
    if cfg.run_trueskill:
        _ts.run(cfg)
    else:
        log.info("Analytics: skipping trueskill (run_trueskill=False)")

    if cfg.run_head2head:
        _h2h.run(cfg)
    else:
        log.info("Analytics: skipping head-to-head (run_head2head=False)")

    if cfg.run_rf:
        _rf.run(cfg)
    else:
        log.info("Analytics: skipping random forest (run_rf=False)")
    log.info("Analytics: all modules finished")
