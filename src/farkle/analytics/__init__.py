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
    _ts.run(cfg)
    _h2h.run(cfg)
    _rf.run(cfg)
    log.info("Analytics: all modules finished")
