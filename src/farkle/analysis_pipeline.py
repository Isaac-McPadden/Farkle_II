# src/farkle/analysis_pipeline.py
"""High‑level wrapper that kicks off all second‑stage analyses (TrueSkill,
Bonferroni head‑to‑head, random‑forest feature importance).

Key points
==========
*   **No recursion on Windows** – when symlinks are unavailable we now
    ``copytree`` while *ignoring the ``analysis/`` directory* to avoid the
    exponential nesting the user just observed.
*   Per‑strategy metrics are aggregated once per *n_players* block and as
    a combined table.  Each stage‑2 script runs inside
    ``<seed>/analysis`` so that its hard‑coded ``data/`` look‑ups resolve.
*   Module is both importable *and* CLI‑invokable via
    ``python -m farkle.analysis_pipeline <results_seed_dir>``.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List

import pandas as pd

from . import run_bonferroni_head2head, run_rf, run_trueskill
from .run_trueskill import _read_loose_parquets, _read_row_shards, _read_winners_csv

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_link(src: Path, dst: Path) -> None:
    """Make *dst* point at *src*.

    1.  Try a real directory symlink (preferred).
    2.  If that fails on Windows (`WinError 1314`), make a **shallow
        copy** while *ignoring the ``analysis`` directory* so we don’t end
        up with ``…/analysis/data/results/analysis/data/results/…``.
    """
    try:
        dst.symlink_to(src, target_is_directory=True)  # type: ignore[arg-type]
        return
    except (OSError, NotImplementedError) as exc:
        log.warning("Symlink failed (%s). Falling back to copytree …", exc)

    if dst.exists():
        shutil.rmtree(dst)

    # Skip the analysis directory to prevent infinite nesting on re‑runs.
    ignore_analysis = shutil.ignore_patterns("analysis")
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore_analysis)


def _load_results_df(block: Path) -> pd.DataFrame:
    """Return a DataFrame with **all** game rows for one *n_players* sub‑folder."""

    row_dirs: List[Path] = [p for p in block.glob("*_rows") if p.is_dir()]
    if row_dirs:
        frames = [_read_row_shards(d) for d in row_dirs]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    parquet_df = _read_loose_parquets(block)
    if parquet_df is not None:
        return parquet_df

    winners = block / "winners.csv"
    if winners.exists():
        return _read_winners_csv(block)

    return pd.DataFrame()


def _strategy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per‑strategy win‑rate + average rounds/score."""
    if df.empty:
        return pd.DataFrame(columns=["strategy", "wins", "avg_rounds", "avg_score"])

    winner_col = "winner_strategy" if "winner_strategy" in df.columns else "winner" if "winner" in df.columns else None
    if winner_col is None:
        return pd.DataFrame(columns=["strategy", "wins", "avg_rounds", "avg_score"])

    grouped = df.groupby(winner_col)
    out = pd.DataFrame(
        {
            "strategy": grouped.size().index,
            "wins": grouped.size().values,
            "avg_rounds": grouped["n_rounds"].mean() if "n_rounds" in df.columns else float("nan"),
            "avg_score": (
                grouped["winning_score"].mean() if "winning_score" in df.columns else float("nan")
            ),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Public entry‑point
# ---------------------------------------------------------------------------


def run_analysis_pipeline(
    seed_folder: str | Path,
) -> Path:  # noqa: C901 – acceptable for orchestrator
    """Run the full analysis chain for the given *result seed* directory.

    Parameters
    ----------
    seed_folder : str | Path
        Directory that contains per‑table‑size sub‑folders such as
        ``2_players/``, ``3_players/`` … Each must in turn contain either
        ``winners.csv``, loose Parquet files, or row‑shard directories
        produced by :pymod:`farkle.run_tournament`.

    Returns
    -------
    Path
        The freshly populated ``analysis/`` directory.
    """
    base = Path(seed_folder).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)

    analysis_dir = base / "analysis"
    data_dir = analysis_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Wire raw results → analysis/data/results (symlink or safe copy)
    # ------------------------------------------------------------------
    results_link = data_dir / "results"
    if not results_link.exists():
        _safe_link(base, results_link)

    # ------------------------------------------------------------------
    # Derive per‑strategy metrics for each *n_players* sub‑folder
    # ------------------------------------------------------------------
    metrics_frames: list[pd.DataFrame] = []

    for block in sorted(base.glob("*_players")):
        n_players = int(block.name.split("_")[0])
        df = _load_results_df(block)
        metrics = _strategy_metrics(df).assign(n_players=n_players)
        if not metrics.empty:
            metrics.to_parquet(analysis_dir / f"{n_players}p_metrics.parquet", index=False)
            metrics_frames.append(metrics)

    metrics_frames = [m for m in metrics_frames if not m.empty]
    if metrics_frames:
        pd.concat(metrics_frames, ignore_index=True).to_parquet(data_dir / "metrics.parquet", index=False)
    else:
        log.warning("No per‑strategy metrics generated – check input folders.")

    # ------------------------------------------------------------------
    # Second‑stage stats (run inside analysis/ so relative paths match)
    # ------------------------------------------------------------------
    cwd = Path.cwd()
    os.chdir(analysis_dir)
    try:
        run_trueskill.run_trueskill(dataroot=data_dir)
        run_bonferroni_head2head.run_bonferroni_head2head(dataroot=data_dir)
        run_rf.run_rf(dataroot=data_dir)
    finally:
        os.chdir(cwd)

    log.info("✓ Analysis pipeline finished → %s", analysis_dir)
    return analysis_dir


# ---------------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m farkle.analysis_pipeline <results_seed_dir>")
        sys.exit(1)
    run_analysis_pipeline(sys.argv[1])
