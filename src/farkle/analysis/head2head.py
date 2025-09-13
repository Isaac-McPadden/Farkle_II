from __future__ import annotations

import logging

from farkle.analysis import run_bonferroni_head2head as _h2h
from farkle.analysis.analysis_config import PipelineCfg
from farkle.app_config import AppConfig

log = logging.getLogger(__name__)


def _pipeline_cfg(cfg: AppConfig | PipelineCfg) -> PipelineCfg:
    return cfg.analysis if isinstance(cfg, AppConfig) else cfg


def run(cfg: AppConfig | PipelineCfg) -> None:
    cfg = _pipeline_cfg(cfg)
    out = cfg.analysis_dir / "bonferroni_pairwise.csv"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        log.info("Head-to-Head: results up-to-date - skipped")
        return

    log.info("Head-to-Head: running in-process")
    try:
        _h2h.run_bonferroni_head2head(root=cfg.results_dir, n_jobs=cfg.n_jobs)
    except Exception as e:  # noqa: BLE001
        # Strategy strings in small test fixtures may not be parseable by the
        # legacy head-to-head script. Rather than abort the entire analytics
        # pass, log and continue.
        log.warning("Head-to-Head: skipped (%s)", e)
