# src/farkle/analysis/seed_summaries.py
"""Build per-seed, per-player summaries with Wilson confidence intervals.

The summaries are derived strictly from the resolved metrics parquet written by
the :mod:`farkle.analysis.metrics` stage.  Each summary file contains the union
of strategies observed for a given ``(seed, players)`` pair; no interpolation is
performed for missing combinations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa

from farkle.config import AppConfig
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.stats import wilson_ci

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"strategy", "n_players", "games", "wins"}
BASE_COLUMNS = [
    "strategy_id",
    "players",
    "seed",
    "games",
    "wins",
    "win_rate",
    "ci_lo",
    "ci_hi",
]
SUMMARY_TEMPLATE = "strategy_summary_{players}p_seed{seed}.parquet"
MEAN_NAME_OVERRIDES = {
    "n_rounds": "turns",
    "farkles": "farkles",
    "rolls": "rolls",
    "score": "score",
    "highest_turn": "highest_turn",
    "smart_five_uses": "smart_five_uses",
    "n_smart_five_dice": "smart_five_dice",
    "smart_one_uses": "smart_one_uses",
    "n_smart_one_dice": "smart_one_dice",
    "hot_dice": "hot_dice",
}


def run(cfg: AppConfig, *, force: bool = False) -> None:
    """Materialize per-seed strategy summaries with confidence intervals."""

    metrics_frame, metrics_path = _load_metrics_frame(cfg)
    if metrics_frame.empty:
        LOGGER.warning(
            "Seed summaries skipped: metrics parquet is empty",
            extra={"stage": "seed_summaries", "metrics_path": str(metrics_path)},
        )
        return

    seeds = sorted(metrics_frame["seed"].unique())
    player_counts = sorted(metrics_frame["players"].unique())
    for seed in seeds:
        for players in player_counts:
            subset = metrics_frame[
                (metrics_frame["seed"] == seed) & (metrics_frame["players"] == players)
            ]
            if subset.empty:
                continue
            summary = _build_summary(subset, players=int(players), seed=int(seed))
            if summary.empty:
                continue
            output_path = cfg.analysis_dir / SUMMARY_TEMPLATE.format(players=players, seed=seed)
            if not force and _existing_summary_matches(output_path, summary):
                LOGGER.info(
                    "Seed summary already up-to-date",
                    extra={
                        "stage": "seed_summaries",
                        "players": players,
                        "seed": seed,
                        "path": str(output_path),
                    },
                )
            else:
                _write_summary(summary, output_path)
                LOGGER.info(
                    "Seed summary written",
                    extra={
                        "stage": "seed_summaries",
                        "players": players,
                        "seed": seed,
                        "rows": len(summary),
                        "path": str(output_path),
                    },
                )

            _sync_meta_summary(cfg, summary, output_path)


def _load_metrics_frame(cfg: AppConfig) -> tuple[pd.DataFrame, Path]:
    """Load the combined metrics parquet and standardize core columns."""
    metrics_path = cfg.analysis_dir / cfg.metrics_name
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    df = pd.read_parquet(metrics_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"metrics parquet missing required columns: {sorted(missing)}")

    df = df.copy()
    if "seed" not in df.columns:
        df["seed"] = int(cfg.sim.seed)
    if df["seed"].isna().any():
        raise ValueError("metrics parquet contains null seed values")

    df.rename(columns={"strategy": "strategy_id", "n_players": "players"}, inplace=True)
    df["strategy_id"] = df["strategy_id"].astype(str)
    df["players"] = df["players"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["games"] = df["games"].fillna(0).astype(int)
    df["wins"] = df["wins"].fillna(0).astype(int)
    if (df["games"] < 0).any():
        raise ValueError("metrics parquet contains negative game counts")
    df = df.sort_values(["seed", "players", "strategy_id"], kind="mergesort").reset_index(drop=True)
    return df, metrics_path


def _build_summary(frame: pd.DataFrame, *, players: int, seed: int) -> pd.DataFrame:
    """Aggregate metrics for a seed/player subset into summary rows."""
    subset = frame.copy()
    if subset.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)

    records: list[dict[str, float | int]] = []
    strategies = sorted(subset["strategy_id"].unique())
    mean_columns = [c for c in subset.columns if c.startswith("mean_")]
    for strategy_id in strategies:
        chunk = subset[subset["strategy_id"] == strategy_id]
        games = int(chunk["games"].sum())
        wins = int(chunk["wins"].sum())

        if games <= 0:
            win_rate = 0.0
            ci_lo, ci_hi = 0.0, 1.0
        else:
            win_rate = wins / games
            ci_lo, ci_hi = wilson_ci(wins, games)

        row: dict[str, float | int] = {
            "strategy_id": strategy_id,
            "players": players,
            "seed": seed,
            "games": games,
            "wins": wins,
            "win_rate": win_rate,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }

        for mean_col in mean_columns:
            if mean_col not in chunk.columns:
                continue
            value = _weighted_mean(chunk, mean_col)
            out_col = _mean_output_name(mean_col)
            row[out_col] = value

        records.append(row)

    summary = pd.DataFrame(records)
    if summary.empty:
        return summary
    summary = _normalize_summary(summary)
    extra_cols = [c for c in summary.columns if c not in BASE_COLUMNS]
    ordered = BASE_COLUMNS + sorted(extra_cols)
    return summary[ordered]


def _mean_output_name(column: str) -> str:
    """Convert a ``mean_<metric>`` column into a user-facing label."""
    base = column.removeprefix("mean_")
    label = MEAN_NAME_OVERRIDES.get(base, base)
    return f"{label}_mean"


def _weighted_mean(frame: pd.DataFrame, column: str) -> float:
    """Compute a weighted mean using ``games`` as the weight column."""
    weights = frame["games"].astype(float)
    values = frame[column].astype(float)
    mask = (weights > 0) & values.notna()
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce deterministic types and ordering for summary comparison."""
    normalized = df.copy()
    normalized["strategy_id"] = normalized["strategy_id"].astype(str)
    for col in ("players", "seed", "games", "wins"):
        normalized[col] = normalized[col].astype(np.int64)
    for col in ("win_rate", "ci_lo", "ci_hi"):
        normalized[col] = normalized[col].astype(float)
    extra_cols = [c for c in normalized.columns if c not in BASE_COLUMNS]
    for col in extra_cols:
        normalized[col] = normalized[col].astype(float)
    return normalized


def _existing_summary_matches(path: Path, new_df: pd.DataFrame) -> bool:
    """Check if an existing summary parquet matches ``new_df`` exactly."""
    if not path.exists():
        return False
    try:
        existing = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return False
    existing_norm = _normalize_summary(existing)
    new_norm = _normalize_summary(new_df)
    if list(existing_norm.columns) != list(new_norm.columns):
        return False
    try:
        pdt.assert_frame_equal(existing_norm, new_norm, check_dtype=True, check_exact=True)
        return True
    except AssertionError:
        return False


def _write_summary(df: pd.DataFrame, path: Path) -> None:
    """Write a summary frame to parquet using atomic semantics."""

    table = pa.Table.from_pandas(df, preserve_index=False)
    write_parquet_atomic(table, path)


def _sync_meta_summary(cfg: AppConfig, summary: pd.DataFrame, analysis_path: Path) -> None:
    """Copy the latest summary into the shared meta directory when configured."""

    meta_dir = cfg.meta_analysis_dir
    meta_path = meta_dir / analysis_path.name
    if meta_path == analysis_path:
        return
    if _existing_summary_matches(meta_path, summary):
        return

    _write_summary(summary, meta_path)
    LOGGER.info(
        "Seed summary synced to meta directory",
        extra={
            "stage": "seed_summaries",
            "players": summary["players"].iloc[0] if not summary.empty else None,
            "seed": summary["seed"].iloc[0] if not summary.empty else None,
            "path": str(meta_path),
        },
    )


__all__ = ["run"]
