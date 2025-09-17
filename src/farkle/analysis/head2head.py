from __future__ import annotations

import logging

from farkle.analysis import run_bonferroni_head2head as _h2h
from farkle.analysis.analysis_config import PipelineCfg
from farkle.app_config import AppConfig

LOGGER = logging.getLogger(__name__)


def _pipeline_cfg(cfg: AppConfig | PipelineCfg) -> PipelineCfg:
    return cfg.analysis if isinstance(cfg, AppConfig) else cfg


def run(cfg: AppConfig | PipelineCfg) -> None:
    cfg = _pipeline_cfg(cfg)
    out = cfg.analysis_dir / "bonferroni_pairwise.parquet"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        LOGGER.info(
            "Head-to-head results up-to-date",
            extra={"stage": "head2head", "path": str(out)},
        )
        return

    LOGGER.info(
        "Head-to-head analysis running",
        extra={
            "stage": "head2head",
            "results_dir": str(cfg.results_dir),
            "n_jobs": cfg.n_jobs,
        },
    )
    try:
        _h2h.run_bonferroni_head2head(root=cfg.results_dir, n_jobs=cfg.n_jobs)
    except Exception as e:  # noqa: BLE001
        # Strategy strings in small test fixtures may not be parseable by the
        # legacy head-to-head script. Rather than abort the entire analytics
        # pass, log and continue.
        LOGGER.warning(
            "Head-to-head skipped",
            extra={"stage": "head2head", "error": str(e)},
        )
