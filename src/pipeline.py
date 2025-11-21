# src/pipeline.py
"""Front door helpers for running the analytics pipeline.

This module used to only re-export :func:`farkle.pipeline.main` for
backwards compatibility.  It now exposes small convenience functions for
running analytics passes directly on an experiment directory.  Each pass
creates a ``.done.json`` stamp next to its primary output and skips work
when inputs are unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

from farkle.config import AppConfig, IOConfig
from farkle.utils.writer import atomic_path

__all__ = [
    "main",
    "analyze_all",
    "analyze_trueskill",
    "analyze_h2h",
    "analyze_hgb",
    "analyze_agreement",
    "fingerprint",
    "write_done",
    "is_up_to_date",
]


def main(argv: object | None = None) -> int:
    """Thin wrapper importing :func:`farkle.pipeline.main` lazily."""

    from farkle.analysis.pipeline import main as _main

    return _main(argv)  # pyright: ignore[reportArgumentType]


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


def write_done(done_path: Path, inputs: list[Path], outputs: list[Path], tool: str) -> None:
    """Write a ``.done.json`` stamp for ``tool``."""

    stamp = {
        "inputs": fingerprint(inputs),
        "outputs": [{"path": str(p)} for p in outputs],
        "tool": tool,
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    done_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(done_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(stamp, indent=2, sort_keys=True))


def is_up_to_date(done_path: Path, inputs: list[Path], outputs: list[Path]) -> bool:
    """Return ``True`` if all *inputs* are older than ``done_path`` and outputs exist."""

    if not done_path.exists():
        return False
    done_mtime = done_path.stat().st_mtime
    for inp in inputs:
        if not inp.exists() or inp.stat().st_mtime > done_mtime:
            return False
    return all(out.exists() for out in outputs)


# ---------------------------------------------------------------------------
# Individual analytics stages
# ---------------------------------------------------------------------------


def _done_path(out: Path) -> Path:
    """Return the companion ``.done.json`` path for a given output file."""
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
    from farkle.analysis import run_trueskill as _rt

    cfg = AppConfig(io=IOConfig(results_dir=exp_dir))
    _rt.run_trueskill_all_seeds(cfg)
    write_done(done, inputs, [out], "farkle.analytics.trueskill")
    print("trueskill")


def analyze_h2h(exp_dir: Path) -> None:
    """Run Bonferroni head-to-head analysis."""

    exp_dir = Path(exp_dir)
    analysis_dir = exp_dir / "analysis"
    out = analysis_dir / "bonferroni_pairwise.parquet"
    done = _done_path(out)
    tiers = analysis_dir / "tiers.json"
    inputs = [tiers]
    if is_up_to_date(done, inputs, [out]):
        print("SKIP h2h (up to date)")
        return

    analysis_dir.mkdir(parents=True, exist_ok=True)
    from farkle.analysis import run_bonferroni_head2head as _h2h

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
    from farkle.analysis import run_hgb as _hgb

    _hgb.run_hgb(root=analysis_dir, output_path=out)
    write_done(done, inputs, [out], "farkle.analytics.hgb")
    print("hgb")


def analyze_agreement(exp_dir: Path) -> None:
    """Compute cross-method agreement metrics."""

    exp_dir = Path(exp_dir)
    analysis_dir = exp_dir / "analysis"
    ratings = analysis_dir / "ratings_pooled.parquet"
    if not ratings.exists():
        raise FileNotFoundError(
            "agreement analysis requires ratings_pooled.parquet; run trueskill first"
        )

    players = _detect_player_counts(analysis_dir)
    outputs = [analysis_dir / f"agreement_{p}p.json" for p in players]
    done = _done_path(outputs[0])
    inputs = [ratings]
    for candidate in (
        analysis_dir / "frequentist_scores.parquet",
        analysis_dir / "bonferroni_decisions.parquet",
    ):
        if candidate.exists():
            inputs.append(candidate)

    if is_up_to_date(done, inputs, outputs):
        print("SKIP agreement (up to date)")
        return

    analysis_dir.mkdir(parents=True, exist_ok=True)
    from farkle.analysis import agreement as _agreement

    cfg = AppConfig(io=IOConfig(results_dir=exp_dir))
    cfg.sim.n_players_list = players
    _agreement.run(cfg)
    write_done(done, inputs, outputs, "farkle.analytics.agreement")
    print("agreement")


def _detect_player_counts(analysis_dir: Path) -> list[int]:
    """Infer available player counts from existing metrics outputs."""
    metrics = analysis_dir / "metrics.parquet"
    if metrics.exists():
        pandas_spec = importlib.util.find_spec("pandas")
        if pandas_spec is not None:
            import pandas as pd

            try:
                df = pd.read_parquet(metrics, columns=["n_players"])
                values = sorted({int(v) for v in df["n_players"].dropna().unique()})
                if values:
                    return values
            except Exception:  # noqa: BLE001 - fall back to default
                pass
    return [5]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def analyze_all(exp_dir: Path) -> None:
    """Run all analytics passes in order."""

    analyze_trueskill(exp_dir)
    analyze_h2h(exp_dir)
    analyze_hgb(exp_dir)
    analyze_agreement(exp_dir)
