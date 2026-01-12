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

from farkle.analysis import stage_logger
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
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


def _summary_path(cfg: AppConfig, *, players: int, seed: int) -> Path:
    filename = SUMMARY_TEMPLATE.format(players=players, seed=seed)
    stage_dir = cfg.seed_summaries_dir(players)
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir / filename


def run(cfg: AppConfig, *, force: bool = False) -> None:
    """Materialize per-seed strategy summaries with confidence intervals."""

    stage_log = stage_logger("seed_summaries", logger=LOGGER)
    stage_log.start()

    metrics_path = cfg.metrics_input_path()
    if not metrics_path.exists():
        stage_log.missing_input("metrics parquet missing", path=str(metrics_path))
        return

    metrics_frame, metrics_path = _load_metrics_frame(cfg)
    if metrics_frame.empty:
        stage_log.missing_input("metrics parquet empty", metrics_path=str(metrics_path))
        return

    seeds = sorted(metrics_frame["seed"].unique())
    player_counts = sorted(metrics_frame["players"].unique())
    done = stage_done_path(cfg.seed_summaries_stage_dir, "seed_summaries")

    expected_outputs: list[Path] = []
    for seed in seeds:
        for players in player_counts:
            subset = metrics_frame[
                (metrics_frame["seed"] == seed) & (metrics_frame["players"] == players)
            ]
            if subset.empty:
                continue
            expected_outputs.append(_summary_path(cfg, players=int(players), seed=int(seed)))

    if not force and expected_outputs and stage_is_up_to_date(
        done, inputs=[metrics_path], outputs=expected_outputs, config_sha=cfg.config_sha
    ):
        LOGGER.info(
            "Seed summaries up-to-date",
            extra={"stage": "seed_summaries", "stamp": str(done)},
        )
        return

    outputs: list[Path] = []
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
            output_path = _summary_path(cfg, players=int(players), seed=int(seed))
            outputs.append(output_path)
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

    if outputs:
        write_stage_done(done, inputs=[metrics_path], outputs=outputs, config_sha=cfg.config_sha)


def _load_metrics_frame(cfg: AppConfig) -> tuple[pd.DataFrame, Path]:
    """Load the combined metrics parquet and standardize core columns."""
    metrics_path = cfg.metrics_input_path()
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

    mean_columns = [c for c in subset.columns if c.startswith("mean_")]
    totals = (
        subset.groupby("strategy_id", sort=True)
        .agg({"games": "sum", "wins": "sum"})
        .reset_index()
    )
    if mean_columns:
        weighted = _weighted_means_by_strategy(subset, mean_columns).reset_index()
        summary = totals.merge(weighted, on="strategy_id", how="left")
    else:
        summary = totals
    summary["players"] = players
    summary["seed"] = seed

    games = summary["games"].to_numpy()
    wins = summary["wins"].to_numpy()
    summary["win_rate"] = np.where(games > 0, wins / games, 0.0)
    ci_bounds = [
        wilson_ci(int(wins_i), int(games_i)) if games_i > 0 else (0.0, 1.0)
        for wins_i, games_i in zip(wins, games)
    ]
    summary["ci_lo"] = [bounds[0] for bounds in ci_bounds]
    summary["ci_hi"] = [bounds[1] for bounds in ci_bounds]
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


def _weighted_means_by_strategy(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute weighted mean columns grouped by ``strategy_id``."""
    def weighted_means(group: pd.DataFrame) -> pd.Series:
        weights = group["games"].astype(float)
        valid_weights = weights > 0
        results: dict[str, float] = {}
        for column in columns:
            values = group[column].astype(float)
            mask = valid_weights & values.notna()
            if mask.any():
                results[_mean_output_name(column)] = float(
                    np.average(values[mask], weights=weights[mask])
                )
            else:
                results[_mean_output_name(column)] = float("nan")
        return pd.Series(results)

    return frame.groupby("strategy_id", sort=True).apply(weighted_means)


def _normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce deterministic types and ordering for summary comparison."""
    normalized = df.copy()
    normalized["strategy_id"] = normalized["strategy_id"].astype(str)
    for col in ("players", "seed", "games", "wins"):
        normalized[col] = _cast_int32_if_safe(normalized[col])
    for col in ("win_rate", "ci_lo", "ci_hi"):
        normalized[col] = normalized[col].astype(float)
    extra_cols = [c for c in normalized.columns if c not in BASE_COLUMNS]
    for col in extra_cols:
        normalized[col] = normalized[col].astype(float)
    return normalized


def _cast_int32_if_safe(series: pd.Series) -> pd.Series:
    """Cast series to int32 when values fit, otherwise keep int64."""
    values = pd.to_numeric(series, errors="coerce")
    non_null = values.dropna()
    if non_null.empty:
        return values.astype(np.int64)
    int32_min = np.iinfo(np.int32).min
    int32_max = np.iinfo(np.int32).max
    if non_null.min() < int32_min or non_null.max() > int32_max:
        return values.astype(np.int64)
    return values.astype(np.int32)


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
