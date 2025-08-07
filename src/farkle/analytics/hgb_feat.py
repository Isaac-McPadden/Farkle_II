from __future__ import annotations

import logging
import shutil

from farkle import run_hgb as _hgb
from farkle.analysis_config import PipelineCfg

log = logging.getLogger(__name__)


def run(cfg: PipelineCfg) -> None:
    out = cfg.results_dir / "hgb_importance.json"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        log.info("Hist-Gradient-Boosting: results up-to-date - skipped")
        return

    log.info("Hist-Gradient-Boosting: running in-process")

    # ``run_hgb`` expects ratings and metrics under the same root. The metrics
    # live in ``analysis_dir`` while ``ratings_pooled.pkl`` is produced in the
    # results root, so copy it into place before invoking the legacy script.
    ratings_src = cfg.results_dir / "ratings_pooled.pkl"
    ratings_dst = cfg.analysis_dir / "ratings_pooled.pkl"
    if ratings_src.exists():
        shutil.copy2(ratings_src, ratings_dst)

    _hgb.main(["--root", str(cfg.analysis_dir), "--output", str(out)])
