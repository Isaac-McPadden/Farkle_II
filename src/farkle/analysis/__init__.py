"""Lightweight orchestrator for downstream statistical analyses."""

from __future__ import annotations

import logging

from farkle.analysis.analysis_config import PipelineCfg
from farkle.analysis import head2head as _h2h
from farkle.analysis import hgb_feat as _hgb
from farkle.analysis import trueskill as _ts

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

    if cfg.run_hgb:
        _hgb.run(cfg)
    else:
        log.info("Analytics: skipping hist gradient boosting (run_hgb=False)")
    log.info("Analytics: all modules finished")
