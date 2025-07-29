# src/farkle/analysis_pipeline.py
"""High‑level convenience wrapper that stitches together the three
analysis sub‑modules (TrueSkill, Bonferroni head‑to‑head, Gradient‑Boost
feature importance).

The script is intentionally I/O‑bound only.  All heavy computation still
lives in :pymod:`farkle.run_trueskill`, :pymod:`farkle.run_bonferroni_head2head`
and :pymod:`farkle.run_rf`.  This wrapper merely:

1.  Locates a *result seed folder* (eg. ``data/results_seed_0``).
2.  Builds a dedicated ``analysis/`` sub‑directory and wires the raw
    outputs into it via a **symlink** (or a read‑only copy when symlinks
    are unavailable – see :pyfunc:`_safe_link`).
3.  Derives per‑strategy metrics (wins/avg rounds/avg score) for each
    table‑size block and writes them to Parquet.
4.  Executes the three stage‑2 analyses *in the correct order* while
    temporarily ``chdir``‑ing to the analysis directory so that their
    hard‑coded ``data/`` look‑ups succeed.

The module is designed for *importability* (everything wrapped in
functions) but can also be used as a CLI::

    python -m farkle.analysis_pipeline path/to/results_seed_7

>>> run_analysis_pipeline(Path("data/results_seed_0"))  # doctest: +ELLIPSIS
PosixPath('data/results_seed_0/analysis')
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
    """Create *dst* → *src* directory symlink or fall back to copytree.

    On Windows, non‑admin users (or files on FAT/ExFAT) cannot create
    symlinks.  In that case we make a **shallow** copy – individual files
    are still memory‑mapped by Pandas/PyArrow so the extra disk usage is
    negligible for parquet‑heavy folders.
    """
    try:
        dst.symlink_to(src, target_is_directory=True)  # type: ignore[arg-type]
        return
    except (OSError, NotImplementedError) as exc:
        log.warning("Symlink failed (%s). Falling back to copytree …", exc)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _load_results_df(block: Path) -> pd.DataFrame:
    """Return a DataFrame with **all** game rows for one *n_players* block."""

    # Prefer row‑shard directories → compact parquet → legacy CSV order.
    row_dirs: List[Path] = [p for p in block.glob("*_rows") if p.is_dir()]
    if row_dirs:
        frames = [_read_row_shards(d) for d in row_dirs]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    parquet_df = _read_loose_parquets(block)
    if parquet_df is not None:
        return parquet_df

    if (block / "winners.csv").exists():
        return _read_winners_csv(block)

    return pd.DataFrame()


def _strategy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute wins / avg rounds / avg score per *winner* strategy."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "wins",
                "avg_rounds",
                "avg_score",
            ]
        )

    if "winner_strategy" in df.columns:
        winner_col = "winner_strategy"
    elif "winner" in df.columns:
        winner_col = "winner"
    else:
        return pd.DataFrame(
            columns=[
                "strategy",
                "wins",
                "avg_rounds",
                "avg_score",
            ]
        )

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

    # --- wire raw results into analysis/data/results -------------------
    results_link = data_dir / "results"
    if not results_link.exists():
        _safe_link(base, results_link)

    # --- derive per‑strategy metrics for every *n_players* block -------
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
        combined = pd.concat(metrics_frames, ignore_index=True)
        combined.to_parquet(data_dir / "metrics.parquet", index=False)
    else:
        log.warning("No per‑strategy metrics generated – check input folders.")

    # --- stage‑2 analyses (need to operate inside analysis/ dir) -------
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
