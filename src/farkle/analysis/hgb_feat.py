from __future__ import annotations

import logging
from pathlib import Path

from farkle.analysis import run_hgb as _hgb
from farkle.analysis.analysis_config import PipelineCfg
from farkle.app_config import AppConfig
from farkle.utils.writer import atomic_path

log = logging.getLogger(__name__)


def _pipeline_cfg(cfg: AppConfig | PipelineCfg) -> PipelineCfg:
    return cfg.analysis if isinstance(cfg, AppConfig) else cfg


def run(cfg: AppConfig | PipelineCfg) -> None:
    cfg = _pipeline_cfg(cfg)
    out = cfg.analysis_dir / "hgb_importance.json"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        log.info("Hist-Gradient-Boosting: results up-to-date - skipped")
        return

    log.info("Hist-Gradient-Boosting: running in-process")
    ratings_src = cfg.results_dir / "ratings_pooled.pkl"
    ratings_dst = cfg.analysis_dir / "ratings_pooled.pkl"
    if ratings_src.exists() and not ratings_dst.exists():
        ratings_dst.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(str(ratings_dst)) as tmp_path:
            Path(tmp_path).write_bytes(ratings_src.read_bytes())

    _hgb.run_hgb(root=cfg.analysis_dir, output_path=out)
