from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from farkle.analysis_config import PipelineCfg

log = logging.getLogger(__name__)
SCRIPT = Path(__file__).resolve().parents[1] / "run_trueskill.py"


def run(cfg: PipelineCfg) -> None:
    """Thin wrapper around the legacy script so the new pipeline stays small."""
    out = cfg.analysis_dir / "trueskill.parquet"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        log.info("TrueSkill: results up-to-date - skipped")
        return

    cmd = [sys.executable, str(SCRIPT), "--root", str(cfg.root)]
    log.info("TrueSkill: calling %s", " ".join(cmd))
    subprocess.check_call(cmd)
