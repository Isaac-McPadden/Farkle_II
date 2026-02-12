# src/farkle/analysis/h2h_tier_trends.py
"""Join head-to-head S tiers with pooled win-rate trends across player counts."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from farkle.analysis import stage_logger
from farkle.analysis.meta import MIN_VARIANCE, Z_975
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig
from farkle.utils.artifacts import write_parquet_atomic

LOGGER = logging.getLogger(__name__)

OUTPUT_PARQUET = "s_tier_trends.parquet"
META_TEMPLATE = "strategy_summary_{players}p_meta.parquet"


def run(cfg: AppConfig, *, force: bool = False) -> None:
    """Generate per-strategy win-rate trends for S-tier head-to-head picks."""
    stage_log = stage_logger("h2h_tier_trends", logger=LOGGER)
    stage_log.start()

    s_tiers_path = _resolve_s_tiers_path(cfg)
    if s_tiers_path is None or not s_tiers_path.exists():
        stage_log.missing_input("h2h_s_tiers.json missing", path=str(s_tiers_path))
        return

    s_tiers = _load_s_tiers(s_tiers_path)
    if not s_tiers:
        stage_log.missing_input("h2h_s_tiers.json empty", path=str(s_tiers_path))
        return

    meta_paths = _collect_meta_paths(cfg)
    if not meta_paths:
        stage_log.missing_input("meta pooled summaries missing")
        return

    stage_dir = cfg.stage_dir("h2h_tier_trends")
    output_path = stage_dir / OUTPUT_PARQUET
    done_path = stage_done_path(stage_dir, "h2h_tier_trends")
    inputs = [s_tiers_path, *meta_paths]
    outputs = [output_path]

    if not force and stage_is_up_to_date(
        done_path, inputs=inputs, outputs=outputs, config_sha=cfg.config_sha
    ):
        LOGGER.info(
            "H2H tier trend outputs up-to-date",
            extra={"stage": "h2h_tier_trends", "path": str(done_path)},
        )
        return

    meta_frame = _load_meta_frames(meta_paths)
    if meta_frame.empty:
        stage_log.missing_input("meta pooled summaries empty")
        return

    s_tier_df = pd.DataFrame(
        [{"strategy_id": str(strategy), "s_tier": str(label)} for strategy, label in s_tiers.items()]
    )
    joined = meta_frame.merge(s_tier_df, on="strategy_id", how="inner")
    if joined.empty:
        stage_log.missing_input("no overlapping strategies between S tiers and meta")
        return

    pooled_stats = _pooled_across_k(joined)
    joined = joined.merge(pooled_stats, on="strategy_id", how="left")

    joined["baseline_rate"] = 1.0 / joined["players"].astype(float)
    joined["delta_vs_baseline"] = joined["win_rate"] - joined["baseline_rate"]
    joined["delta_vs_pooled"] = joined["win_rate"] - joined["pooled_win_rate"]

    ordered = [
        "strategy_id",
        "s_tier",
        "players",
        "win_rate",
        "se",
        "ci_lo",
        "ci_hi",
        "baseline_rate",
        "delta_vs_baseline",
        "pooled_win_rate",
        "pooled_se",
        "pooled_ci_lo",
        "pooled_ci_hi",
        "delta_vs_pooled",
        "k_count",
        "Q",
        "I2",
    ]
    remaining = [col for col in joined.columns if col not in ordered]
    output = (
        joined[ordered + remaining]
        .sort_values(["s_tier", "strategy_id", "players"], kind="mergesort")
        .reset_index(drop=True)
    )

    table = pa.Table.from_pandas(output, preserve_index=False)
    write_parquet_atomic(table, output_path, codec=cfg.parquet_codec)
    write_stage_done(done_path, inputs=inputs, outputs=outputs, config_sha=cfg.config_sha)


def _resolve_s_tiers_path(cfg: AppConfig) -> Path | None:
    candidates: list[Path] = []
    for stage in ("post_h2h", "head2head"):
        stage_dir = cfg.stage_dir_if_active(stage)
        if stage_dir is not None:
            candidates.append(stage_dir / "h2h_s_tiers.json")
    candidates.append(cfg.analysis_dir / "h2h_s_tiers.json")
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if candidates else None


def _load_s_tiers(path: Path) -> dict[str, str]:
    try:
        payload_any: Any = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload_any, dict):
        return {}

    validated: dict[str, str] = {}
    for strategy, label in payload_any.items():
        if strategy == "_meta":
            continue
        if not isinstance(label, str):
            continue
        validated[str(strategy)] = label
    return cast(dict[str, str], validated)


def _collect_meta_paths(cfg: AppConfig) -> list[Path]:
    players_list: list[int] = []
    for entry in cfg.sim.n_players_list:
        try:
            players_list.append(int(entry))
        except (TypeError, ValueError):
            continue
    players_list = sorted({players for players in players_list if players > 0})

    paths: list[Path] = []
    for players in players_list:
        path = cfg.meta_input_path(players, META_TEMPLATE.format(players=players))
        if path.exists():
            paths.append(path)

    if paths:
        return paths

    meta_stage_dir = cfg.stage_dir_if_active("meta")
    if meta_stage_dir is None:
        return []
    return sorted(meta_stage_dir.glob("*p/strategy_summary_*p_meta.parquet"))


def _load_meta_frames(meta_paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in meta_paths:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df = df.copy()
        if "strategy_id" not in df.columns:
            continue
        df["strategy_id"] = df["strategy_id"].astype(str)
        if "players" in df.columns:
            df["players"] = df["players"].astype(int)
        else:
            df["players"] = _players_from_path(path)
        for col in ("win_rate", "se", "ci_lo", "ci_hi"):
            if col in df.columns:
                df[col] = df[col].astype(float)
        frames.append(df[["strategy_id", "players", "win_rate", "se", "ci_lo", "ci_hi"]])
    if not frames:
        return pd.DataFrame(columns=["strategy_id", "players", "win_rate", "se", "ci_lo", "ci_hi"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.dropna(subset=["players", "win_rate"]).reset_index(drop=True)


def _players_from_path(path: Path) -> int:
    for part in path.parts:
        if part.endswith("p") and part[:-1].isdigit():
            return int(part[:-1])
    return 0


def _pooled_across_k(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for strategy_id, group in frame.groupby("strategy_id", sort=False):
        win_rates = group["win_rate"].astype(float).to_numpy()
        variances = _clean_variances((group["se"].astype(float) ** 2).to_numpy())
        weights = 1.0 / variances
        sum_w = float(weights.sum())
        if sum_w <= 0.0:
            pooled_rate = float("nan")
            pooled_se = float("nan")
        else:
            pooled_rate = float((weights * win_rates).sum() / sum_w)
            pooled_se = float(math.sqrt(1.0 / sum_w))

        pooled_ci_lo = max(0.0, pooled_rate - Z_975 * pooled_se)
        pooled_ci_hi = min(1.0, pooled_rate + Z_975 * pooled_se)

        Q = float((weights * ((win_rates - pooled_rate) ** 2)).sum())
        df = max(len(win_rates) - 1, 0)
        I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0.0 and df > 0 else 0.0

        rows.append(
            {
                "strategy_id": str(strategy_id),
                "pooled_win_rate": pooled_rate,
                "pooled_se": pooled_se,
                "pooled_ci_lo": pooled_ci_lo,
                "pooled_ci_hi": pooled_ci_hi,
                "k_count": int(len(win_rates)),
                "Q": Q,
                "I2": I2,
            }
        )
    return pd.DataFrame(rows)


def _clean_variances(variances: Iterable[float]) -> np.ndarray:
    series = pd.Series(variances, dtype="float64")
    series = series.fillna(MIN_VARIANCE)
    series = series.clip(lower=MIN_VARIANCE)
    return series.to_numpy()
