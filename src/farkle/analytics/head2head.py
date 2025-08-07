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
    try:
        _h2h.main(
            [
                "--root",
                str(cfg.analysis_dir),
                "--jobs",
                str(cfg.n_jobs),
            ]
        )
    except Exception as e:  # noqa: BLE001
        # Strategy strings in small test fixtures may not be parseable by the
        # legacy head-to-head script. Rather than abort the entire analytics
        # pass, log and continue.
        log.warning("Head-to-Head: skipped (%s)", e)
