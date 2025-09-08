from __future__ import annotations

import logging

from farkle import run_hgb as _hgb
from farkle.analysis_config import PipelineCfg

log = logging.getLogger(__name__)


def run(cfg: PipelineCfg) -> None:
    out = cfg.analysis_dir / "hgb_importance.json"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        log.info("Hist-Gradient-Boosting: results up-to-date - skipped")
        return

    log.info("Hist-Gradient-Boosting: running in-process")

    _hgb.main(["--root", str(cfg.analysis_dir), "--output", str(out)])
