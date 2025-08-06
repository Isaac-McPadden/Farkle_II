from __future__ import annotations

import logging

from farkle import run_bonferroni_head2head as _h2h
from farkle.analysis_config import PipelineCfg

log = logging.getLogger(__name__)


def run(cfg: PipelineCfg) -> None:
    out = cfg.analysis_dir / "bonferroni_pairwise.csv"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        log.info("Head-to-Head: results up-to-date - skipped")
        return

    log.info("Head-to-Head: running in-process")
    _h2h.main(["--root", str(cfg.analysis_dir)])
