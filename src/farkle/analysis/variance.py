"""Cross-seed variance estimates for strategy win rates.

This module ingests the combined ``metrics.parquet`` produced by the metrics
stage along with the per-seed strategy summaries.  For every
``(strategy_id, n_players)`` pair it computes the variance of the ``win_rate``
across seeds, the corresponding standard error, and a simple signal-to-noise
heuristic (distance from a fair coin scaled by the cross-seed standard error).

The outputs are written to ``analysis/variance.parquet`` and a compact summary
aggregated by ``n_players`` is written to ``analysis/variance_summary.parquet``.
The module also derives seed-level variance components for win rate, total
score, and game length and writes them to ``analysis/variance_components.parquet``.
All outputs share a done-stamp that captures input/output freshness so that the
module can be skipped when rerun unless ``force`` is requested.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa

from farkle.config import AppConfig
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)

SUMMARY_PATTERN = re.compile(r"strategy_summary_(\d+)p_seed(\d+)\.parquet$")
VARIANCE_OUTPUT = "variance.parquet"
SUMMARY_OUTPUT = "variance_summary.parquet"
COMPONENTS_OUTPUT = "variance_components.parquet"
STAMP_NAME = "variance.done.json"
MIN_SEEDS = 2

COMPONENT_COLUMN_MAP = {
    "win_rate": "win_rate",
    "total_score": "score_mean",
    "game_length": "turns_mean",
}


def run(cfg: AppConfig, *, force: bool = False) -> None:
    """Compute cross-seed win-rate variance and write parquet outputs.

    Args:
        cfg: Application configuration used to resolve paths.
        force: Recompute outputs even when the done-stamp is up-to-date.
    """

    analysis_dir = cfg.analysis_dir
    metrics_path = analysis_dir / cfg.metrics_name
    variance_path = analysis_dir / VARIANCE_OUTPUT
    summary_path = analysis_dir / SUMMARY_OUTPUT
    components_path = analysis_dir / COMPONENTS_OUTPUT
    stamp_path = analysis_dir / STAMP_NAME

    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    seed_summary_paths = _discover_seed_summaries(analysis_dir)
    if not seed_summary_paths:
        LOGGER.info(
            "Variance skipped: no per-seed summaries found",
            extra={"stage": "variance", "analysis_dir": str(analysis_dir)},
        )
        return

    inputs = [metrics_path, *seed_summary_paths]
    outputs = [variance_path, summary_path, components_path]
    if not force and _is_up_to_date(stamp_path, inputs=inputs, outputs=outputs):
        LOGGER.info(
            "Variance outputs up-to-date",
            extra={
                "stage": "variance",
                "variance_path": str(variance_path),
                "summary_path": str(summary_path),
                "components_path": str(components_path),
                "stamp": str(stamp_path),
            },
        )
        return

    LOGGER.info(
        "Computing cross-seed variance",
        extra={
            "stage": "variance",
            "analysis_dir": str(analysis_dir),
            "metrics_path": str(metrics_path),
            "variance_path": str(variance_path),
            "summary_path": str(summary_path),
            "components_path": str(components_path),
            "force": force,
        },
    )

    metrics_frame = _load_metrics(metrics_path)
    seed_frame = _load_seed_summaries(seed_summary_paths)
    if seed_frame.empty:
        LOGGER.info(
            "Variance skipped: seed summaries empty",
            extra={"stage": "variance", "analysis_dir": str(analysis_dir)},
        )
        return

    variance_frame = _compute_variance(seed_frame)
    components_frame = _compute_variance_components(seed_frame)
    detailed = _merge_metrics(metrics_frame, variance_frame)
    if detailed.empty:
        LOGGER.info(
            "Variance skipped: no overlapping strategies",
            extra={"stage": "variance", "analysis_dir": str(analysis_dir)},
        )
        return

    summary = _summarize_variance(detailed)

    variance_table = pa.Table.from_pandas(detailed, preserve_index=False)
    summary_table = pa.Table.from_pandas(summary, preserve_index=False)
    components_table = pa.Table.from_pandas(
        components_frame, preserve_index=False
    )
    write_parquet_atomic(variance_table, variance_path, codec=cfg.parquet_codec)
    write_parquet_atomic(summary_table, summary_path, codec=cfg.parquet_codec)
    write_parquet_atomic(
        components_table, components_path, codec=cfg.parquet_codec
    )
    _write_stamp(stamp_path, inputs=inputs, outputs=outputs)

    LOGGER.info(
        "Variance outputs written",
        extra={
            "stage": "variance",
            "rows": len(detailed),
            "variance_path": str(variance_path),
            "summary_path": str(summary_path),
            "components_path": str(components_path),
        },
    )


def _discover_seed_summaries(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if SUMMARY_PATTERN.search(p.name))


def _load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame(columns=["strategy_id", "players", "win_rate"])

    df = df.copy()
    if "strategy" in df.columns and "strategy_id" not in df.columns:
        df.rename(columns={"strategy": "strategy_id"}, inplace=True)
    if "n_players" in df.columns and "players" not in df.columns:
        df.rename(columns={"n_players": "players"}, inplace=True)

    required = {"strategy_id", "players", "win_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metrics parquet missing required columns: {sorted(missing)}")

    out = df[list(required)]
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["players"] = out["players"].astype(int)
    out["win_rate"] = pd.to_numeric(out["win_rate"], errors="coerce")
    return out


def _load_seed_summaries(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df = df.copy()
        df["strategy_id"] = df["strategy_id"].astype(str)
        df["players"] = df["players"].astype(int)
        df["seed"] = df["seed"].astype(int)
        for column in {"win_rate", *COMPONENT_COLUMN_MAP.values()}:
            if column not in df.columns:
                df[column] = float("nan")
            df[column] = pd.to_numeric(df[column], errors="coerce")
        frames.append(df)

    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)
    columns = ["strategy_id", "players", "seed", "win_rate", *COMPONENT_COLUMN_MAP.values()]
    return pd.DataFrame(columns=columns)


def _compute_variance(seed_frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    grouped = seed_frame.groupby(["strategy_id", "players"], sort=True)
    for (strategy_id, players), group in grouped:
        rates = pd.to_numeric(group["win_rate"], errors="coerce").dropna()
        count = int(rates.size)
        if count == 0:
            continue

        variance_scalar = rates.var(ddof=1)
        # pandas returns a Scalar; after confirming it is not NA we safely cast to float
        if count > 1:
            variance = (
                float(cast(float, variance_scalar))
                if not pd.isna(variance_scalar)
                else float("nan")
            )
        else:
            variance = 0.0
        variance = max(variance, 0.0) if not math.isnan(variance) else variance
        std = float(math.sqrt(variance)) if variance == variance else float("nan")
        se = float(std / math.sqrt(count)) if count > 0 and std == std else float("nan")

        records.append(
            {
                "strategy_id": strategy_id,
                "players": int(players),
                "n_seeds": count,
                "mean_seed_win_rate": float(rates.mean()),
                "variance_win_rate": variance,
                "std_win_rate": std,
                "se_win_rate": se,
            }
        )

    cols = [
        "strategy_id",
        "players",
        "n_seeds",
        "mean_seed_win_rate",
        "variance_win_rate",
        "std_win_rate",
        "se_win_rate",
    ]
    return pd.DataFrame(records, columns=cols)


def _compute_variance_components(
    seed_frame: pd.DataFrame, *, min_seeds: int = MIN_SEEDS
) -> pd.DataFrame:
    columns = [
        "strategy_id",
        "players",
        "component",
        "n_seeds",
        "mean",
        "variance",
        "std_dev",
        "se_mean",
        "ci_lower",
        "ci_upper",
    ]

    if seed_frame.empty:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, float | int | str]] = []
    grouped = seed_frame.groupby(["strategy_id", "players"], sort=True)
    for (strategy_id, players), group in grouped:
        seed_count = group["seed"].nunique()
        if seed_count < min_seeds:
            LOGGER.info(
                "Skipping variance components: insufficient seeds",
                extra={
                    "stage": "variance",
                    "strategy_id": strategy_id,
                    "players": int(players),
                    "seeds": int(seed_count),
                    "min_seeds": int(min_seeds),
                },
            )
            continue

        for component, column in COMPONENT_COLUMN_MAP.items():
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            observations = int(values.size)
            if observations < min_seeds:
                LOGGER.info(
                    "Skipping component metric: insufficient observations",
                    extra={
                        "stage": "variance",
                        "strategy_id": strategy_id,
                        "players": int(players),
                        "component": component,
                        "observations": observations,
                        "min_seeds": int(min_seeds),
                    },
                )
                continue

            variance_scalar = values.var(ddof=1)
            # pandas returns a Scalar; after confirming it is not NA we safely cast to float
            variance = (
                float(cast(float, variance_scalar))
                if observations > 1 and not pd.isna(variance_scalar)
                else float("nan")
            )
            variance = max(variance, 0.0) if not math.isnan(variance) else variance
            std_dev = float(math.sqrt(variance)) if variance == variance else float("nan")
            se_mean = (
                float(std_dev / math.sqrt(observations))
                if observations > 0 and std_dev == std_dev
                else float("nan")
            )
            ci_bounds = (
                (values.mean() - 1.96 * se_mean, values.mean() + 1.96 * se_mean)
                if observations > 1 and se_mean == se_mean
                else (float("nan"), float("nan"))
            )

            records.append(
                {
                    "strategy_id": strategy_id,
                    "players": int(players),
                    "component": component,
                    "n_seeds": observations,
                    "mean": float(values.mean()),
                    "variance": variance,
                    "std_dev": std_dev,
                    "se_mean": se_mean,
                    "ci_lower": ci_bounds[0],
                    "ci_upper": ci_bounds[1],
                }
            )

    return pd.DataFrame(records, columns=columns)


def _merge_metrics(metrics_frame: pd.DataFrame, variance_frame: pd.DataFrame) -> pd.DataFrame:
    if metrics_frame.empty and variance_frame.empty:
        return pd.DataFrame(
            columns=[
                "strategy_id",
                "players",
                "win_rate",
                "mean_seed_win_rate",
                "variance_win_rate",
                "std_win_rate",
                "se_win_rate",
                "signal_to_noise",
                "n_seeds",
            ]
        )

    merged = metrics_frame.merge(
        variance_frame,
        on=["strategy_id", "players"],
        how="inner" if not metrics_frame.empty else "right",
    )

    merged["win_rate"] = pd.to_numeric(merged["win_rate"], errors="coerce")
    mean_seed_win_rate: pd.Series
    if "mean_seed_win_rate" in merged:
        mean_seed_win_rate = merged["mean_seed_win_rate"]
    else:
        mean_seed_win_rate = pd.Series(np.nan, index=merged.index, dtype="float64")

    merged["win_rate_mean"] = merged["win_rate"].combine_first(mean_seed_win_rate)

    def _signal_to_noise(row: pd.Series) -> float:
        se = row.get("se_win_rate", float("nan"))
        if se and se > 0:
            return abs(float(row.get("win_rate_mean", float("nan"))) - 0.5) / se
        return float("nan")

    merged["signal_to_noise"] = merged.apply(_signal_to_noise, axis=1)
    desired_order = [
        "strategy_id",
        "players",
        "win_rate",
        "mean_seed_win_rate",
        "variance_win_rate",
        "std_win_rate",
        "se_win_rate",
        "signal_to_noise",
        "n_seeds",
    ]
    remaining = [c for c in merged.columns if c not in desired_order]
    ordered = merged[desired_order + remaining]
    ordered.sort_values(["players", "strategy_id"], inplace=True)
    ordered.reset_index(drop=True, inplace=True)
    return ordered


def _summarize_variance(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["n_players", "mean_variance", "median_variance"])

    grouped = frame.dropna(subset=["variance_win_rate"]).groupby("players")
    summary = grouped["variance_win_rate"].agg(mean_variance="mean", median_variance="median")
    summary = summary.reset_index().rename(columns={"players": "n_players"})
    summary["n_players"] = summary["n_players"].astype(int)
    return summary


def _stamp(path: Path) -> dict[str, float | int]:
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _write_stamp(stamp_path: Path, *, inputs: Iterable[Path], outputs: Iterable[Path]) -> None:
    payload = {
        "inputs": {str(p): _stamp(p) for p in inputs if p.exists()},
        "outputs": {str(p): _stamp(p) for p in outputs if p.exists()},
    }
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(stamp_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2))


def _is_up_to_date(stamp_path: Path, *, inputs: Iterable[Path], outputs: Iterable[Path]) -> bool:
    if not (stamp_path.exists() and all(p.exists() for p in outputs)):
        return False
    try:
        meta = json.loads(stamp_path.read_text())
    except Exception:  # noqa: BLE001
        return False

    in_meta = meta.get("inputs", {})
    out_meta = meta.get("outputs", {})

    def _matches(paths: Iterable[Path], recorded: dict[str, dict[str, float | int]]) -> bool:
        for p in paths:
            data = recorded.get(str(p))
            if data is None:
                return False
            stat = p.stat()
            if data.get("mtime") != stat.st_mtime or data.get("size") != stat.st_size:
                return False
        return True

    return _matches(inputs, in_meta) and _matches(outputs, out_meta)


__all__ = ["run"]
