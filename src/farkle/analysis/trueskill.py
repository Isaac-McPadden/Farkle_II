from __future__ import annotations

import logging

from farkle.analysis import run_trueskill
from farkle.analysis.analysis_config import PipelineCfg
from farkle.app_config import AppConfig

log = logging.getLogger(__name__)


def _pipeline_cfg(cfg: AppConfig | PipelineCfg) -> PipelineCfg:
    return cfg.analysis if isinstance(cfg, AppConfig) else cfg


def run(cfg: AppConfig | PipelineCfg) -> None:
    """Thin wrapper around the legacy script so the new pipeline stays small."""
    cfg = _pipeline_cfg(cfg)
    out = cfg.analysis_dir / "tiers.json"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        log.info("TrueSkill: results up-to-date - skipped")
        return

    log.info("TrueSkill: running in-process")
    run_trueskill.run_trueskill(root=cfg.analysis_dir)
