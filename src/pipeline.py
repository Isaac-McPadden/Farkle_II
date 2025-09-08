"""Front door helpers for running the analytics pipeline.

This module used to only re-export :func:`farkle.pipeline.main` for
backwards compatibility.  It now exposes small convenience functions for
running analytics passes directly on an experiment directory.  Each pass
creates a ``.done.json`` stamp next to its primary output and skips work
when inputs are unchanged.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

__all__ = [
    "main",
    "analyze_all",
    "analyze_trueskill",
    "analyze_h2h",
    "analyze_hgb",
    "fingerprint",
    "write_done",
    "is_up_to_date",
]


def main(argv: object | None = None) -> int:
    """Thin wrapper importing :func:`farkle.pipeline.main` lazily."""

    from farkle.pipeline import main as _main

    return _main(argv)


# ---------------------------------------------------------------------------
# Helper utilities for done-file tracking
# ---------------------------------------------------------------------------

def fingerprint(paths: list[Path]) -> list[dict]:
    """Return ``[{path, mtime, sha256}]`` fingerprint for *paths*.

    The ``sha256`` key is omitted for directories or unreadable paths.
    """

    out: list[dict] = []
    for p in paths:
        info: dict[str, object] = {"path": str(p), "mtime": p.stat().st_mtime}
        try:
            if p.is_file():
                info["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        except Exception:  # pragma: no cover - best-effort
            pass
        out.append(info)
    return out


def write_done(
    done_path: Path, inputs: list[Path], outputs: list[Path], tool: str
) -> None:
    """Write a ``.done.json`` stamp for ``tool``."""

    stamp = {
        "inputs": fingerprint(inputs),
        "outputs": [{"path": str(p)} for p in outputs],
        "tool": tool,
        "version": 1,
        "created_at": datetime.utcnow().isoformat(),
    }
    done_path.write_text(json.dumps(stamp, indent=2, sort_keys=True))


def is_up_to_date(done_path: Path, inputs: list[Path], outputs: list[Path]) -> bool:
    """Return ``True`` if all *inputs* are older than ``done_path`` and outputs exist."""

    if not done_path.exists():
        return False
    done_mtime = done_path.stat().st_mtime
    for inp in inputs:
        if not inp.exists() or inp.stat().st_mtime > done_mtime:
            return False
    for out in outputs:
        if not out.exists():
            return False
    return True


# ---------------------------------------------------------------------------
# Individual analytics stages
# ---------------------------------------------------------------------------

def _done_path(out: Path) -> Path:
    return out.with_name(out.name + ".done.json")


def analyze_trueskill(exp_dir: Path) -> None:
    """Compute TrueSkill tiers for *exp_dir* simulations."""

    exp_dir = Path(exp_dir)
    analysis_dir = exp_dir / "analysis"
    out = analysis_dir / "tiers.json"
    done = _done_path(out)
    inputs = [p for p in exp_dir.iterdir() if p.name != "analysis"]
    if is_up_to_date(done, inputs, [out]):
        print("SKIP trueskill (up to date)")
        return

    analysis_dir.mkdir(parents=True, exist_ok=True)
    from farkle import run_trueskill as _rt

    _rt.main(["--dataroot", str(exp_dir), "--root", str(analysis_dir)])
    write_done(done, inputs, [out], "farkle.analytics.trueskill")
    print("trueskill")


def analyze_h2h(exp_dir: Path) -> None:
    """Run Bonferroni head-to-head analysis."""

    exp_dir = Path(exp_dir)
    analysis_dir = exp_dir / "analysis"
    out = analysis_dir / "bonferroni_pairwise.csv"
    done = _done_path(out)
    tiers = analysis_dir / "tiers.json"
    inputs = [tiers]
    if is_up_to_date(done, inputs, [out]):
        print("SKIP h2h (up to date)")
        return

    analysis_dir.mkdir(parents=True, exist_ok=True)
    from farkle import run_bonferroni_head2head as _h2h

    _h2h.run_bonferroni_head2head(root=exp_dir, n_jobs=1)
    write_done(done, inputs, [out], "farkle.analytics.head2head")
    print("h2h")


def analyze_hgb(exp_dir: Path) -> None:
    """Run hist gradient boosting feature importance analysis."""

    exp_dir = Path(exp_dir)
    analysis_dir = exp_dir / "analysis"
    out = analysis_dir / "hgb_importance.json"
    done = _done_path(out)
    metrics = analysis_dir / "metrics.parquet"
    ratings = analysis_dir / "ratings_pooled.parquet"
    inputs = [metrics, ratings]
    if is_up_to_date(done, inputs, [out]):
        print("SKIP hgb (up to date)")
        return

    analysis_dir.mkdir(parents=True, exist_ok=True)
    from farkle import run_hgb as _hgb

    _hgb.run_hgb(root=analysis_dir, output_path=out)
    write_done(done, inputs, [out], "farkle.analytics.hgb")
    print("hgb")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def analyze_all(exp_dir: Path) -> None:
    """Run all analytics passes in order."""

    analyze_trueskill(exp_dir)
    analyze_h2h(exp_dir)
    analyze_hgb(exp_dir)


