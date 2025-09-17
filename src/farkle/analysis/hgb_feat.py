from __future__ import annotations

import logging

from farkle.analysis import run_hgb as _hgb
from farkle.analysis.analysis_config import PipelineCfg
from farkle.app_config import AppConfig

LOGGER = logging.getLogger(__name__)


def _pipeline_cfg(cfg: AppConfig | PipelineCfg) -> PipelineCfg:
    return cfg.analysis if isinstance(cfg, AppConfig) else cfg


def run(cfg: AppConfig | PipelineCfg) -> None:
    cfg = _pipeline_cfg(cfg)
    out = cfg.analysis_dir / "hgb_importance.json"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        LOGGER.info(
            "HGB feature importance up-to-date",
            extra={"stage": "hgb", "path": str(out)},
        )
        return

    LOGGER.info(
        "HGB feature importance running",
        extra={
            "stage": "hgb",
            "analysis_dir": str(cfg.analysis_dir),
        },
    )
    _hgb.run_hgb(root=cfg.analysis_dir, output_path=out)
    LOGGER.info(
        "HGB feature importance complete",
        extra={"stage": "hgb", "path": str(out)},
    )
