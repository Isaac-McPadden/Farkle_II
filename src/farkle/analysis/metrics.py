# src/farkle/analysis/metrics.py
"""Aggregate curated data into per-strategy metrics and outputs.

Computes win rates and seat advantages from combined parquet shards, validates
input schemas, and emits CSV/Parquet artifacts for downstream reporting.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from farkle.analysis.checks import check_pre_metrics
from farkle.analysis.isolated_metrics import build_isolated_metrics
from farkle.analysis.seat_stats import (
    SeatMetricConfig,
    compute_seat_advantage,
    compute_seat_metrics,
    compute_symmetry_checks,
)
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig
from farkle.utils.artifacts import write_csv_atomic, write_parquet_atomic
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


def run(cfg: AppConfig) -> None:
    """Compute per-strategy metrics and seat-advantage tables."""

    analysis_dir = cfg.analysis_dir
    metrics_dir = cfg.metrics_pooled_dir
    data_file = cfg.curated_parquet
    symmetry_input = cfg.ingested_rows_curated(2)
    if not symmetry_input.exists():
        symmetry_input = data_file
    out_metrics = cfg.metrics_output_path()
    out_seats = cfg.metrics_output_path("seat_advantage.csv")
    out_seats_parquet = cfg.metrics_output_path("seat_advantage.parquet")
    out_seat_metrics = cfg.metrics_output_path("seat_metrics.parquet")
    out_seat_metrics_csv = cfg.metrics_output_path("seat_metrics.csv")
    out_symmetry = cfg.metrics_output_path("symmetry_checks.parquet")
    out_symmetry_csv = cfg.metrics_output_path("symmetry_checks.csv")
    stamp = cfg.metrics_output_path("metrics.done.json")

    done = stage_done_path(cfg.metrics_stage_dir, "metrics")
    done_isolated = stage_done_path(cfg.metrics_stage_dir, "metrics_isolated")
    done_core = stage_done_path(cfg.metrics_stage_dir, "metrics_core")
    done_seat_advantage = stage_done_path(cfg.metrics_stage_dir, "metrics_seat_advantage")
    done_seat_metrics = stage_done_path(cfg.metrics_stage_dir, "metrics_seat_metrics")
    done_symmetry = stage_done_path(cfg.metrics_stage_dir, "metrics_symmetry")
    stamp_isolated = cfg.metrics_output_path("metrics.isolated.stamp.json")
    stamp_core = cfg.metrics_output_path("metrics.core.stamp.json")
    stamp_seat_advantage = cfg.metrics_output_path("metrics.seat_advantage.stamp.json")
    stamp_seat_metrics = cfg.metrics_output_path("metrics.seat_metrics.stamp.json")
    stamp_symmetry = cfg.metrics_output_path("metrics.symmetry.stamp.json")
    player_counts = sorted({int(n) for n in cfg.sim.n_players_list})
    raw_metric_inputs = [cfg.results_dir / f"{n}_players" / f"{n}p_metrics.parquet" for n in player_counts]
    available_raw_inputs = [path for path in raw_metric_inputs if path.exists()]
    iso_targets = []
    for n in player_counts:
        raw_path = cfg.results_dir / f"{n}_players" / f"{n}p_metrics.parquet"
        if not raw_path.exists():
            continue
        preferred = cfg.metrics_isolated_path(n)
        legacy = cfg.legacy_metrics_isolated_path(n)
        if preferred.exists():
            iso_targets.append(preferred)
        elif legacy.exists():
            iso_targets.append(legacy)
        else:
            iso_targets.append(preferred)
    outputs = [
        out_metrics,
        out_seats,
        out_seats_parquet,
        out_seat_metrics,
        out_seat_metrics_csv,
        out_symmetry,
        out_symmetry_csv,
        *iso_targets,
    ]
    stage_inputs = [data_file, *raw_metric_inputs]
    if symmetry_input != data_file and symmetry_input.exists():
        stage_inputs.append(symmetry_input)
    if stage_is_up_to_date(
        done,
        inputs=stage_inputs,
        outputs=outputs,
        config_sha=getattr(cfg, "config_sha", None),
    ):
        LOGGER.info(
            "Metrics stage up-to-date",
            extra={"stage": "metrics", "path": str(done)},
        )
        return

    if not data_file.exists():
        raise FileNotFoundError(
            f"metrics: missing combined parquet {data_file} – run combine step first"
        )

    LOGGER.info(
        "Metrics stage start",
        extra={
            "stage": "metrics",
            "data_file": str(data_file),
            "analysis_dir": str(analysis_dir),
            "metrics_dir": str(metrics_dir),
        },
    )

    check_pre_metrics(data_file, winner_col="winner_seat")

    if stage_is_up_to_date(
        done_isolated,
        inputs=[data_file, *available_raw_inputs],
        outputs=iso_targets,
        config_sha=getattr(cfg, "config_sha", None),
    ):
        iso_paths = [path for path in iso_targets if path.exists()]
        raw_inputs = raw_metric_inputs
    else:
        iso_paths, raw_inputs = _ensure_isolated_metrics(cfg, player_counts)
        _write_stamp(
            stamp_isolated,
            inputs=[data_file, *available_raw_inputs],
            outputs=iso_paths,
        )
        write_stage_done(
            done_isolated,
            inputs=[data_file, *available_raw_inputs],
            outputs=iso_paths,
            config_sha=getattr(cfg, "config_sha", None),
        )

    outputs = [
        out_metrics,
        out_seats,
        out_seats_parquet,
        out_seat_metrics,
        out_seat_metrics_csv,
        out_symmetry,
        out_symmetry_csv,
        *iso_paths,
    ]

    if stage_is_up_to_date(
        done_core,
        inputs=iso_paths,
        outputs=[out_metrics],
        config_sha=getattr(cfg, "config_sha", None),
    ):
        metrics_df = pd.read_parquet(out_metrics)
    else:
        metrics_df = _collect_metrics_frames(iso_paths)
        if metrics_df.empty:
            raise RuntimeError("metrics: no isolated metric files generated")

        metrics_df = _add_win_rate_uncertainty(metrics_df)
        metrics_df = _downcast_metric_counters(metrics_df)

        metrics_table = pa.Table.from_pandas(metrics_df, preserve_index=False)
        write_parquet_atomic(metrics_table, out_metrics)
        _write_stamp(stamp_core, inputs=iso_paths, outputs=[out_metrics])
        write_stage_done(
            done_core,
            inputs=iso_paths,
            outputs=[out_metrics],
            config_sha=getattr(cfg, "config_sha", None),
        )

    seat_cfg = SeatMetricConfig(seat_range=cfg.metrics_seat_range)
    if stage_is_up_to_date(
        done_seat_advantage,
        inputs=[data_file],
        outputs=[out_seats, out_seats_parquet],
        config_sha=getattr(cfg, "config_sha", None),
    ):
        seat_df = pd.read_csv(out_seats)
    else:
        seat_df = compute_seat_advantage(cfg, data_file, seat_cfg)
        write_csv_atomic(seat_df, out_seats)
        seat_table = pa.Table.from_pandas(seat_df, preserve_index=False)
        write_parquet_atomic(seat_table, out_seats_parquet)
        _write_stamp(
            stamp_seat_advantage,
            inputs=[data_file],
            outputs=[out_seats, out_seats_parquet],
        )
        write_stage_done(
            done_seat_advantage,
            inputs=[data_file],
            outputs=[out_seats, out_seats_parquet],
            config_sha=getattr(cfg, "config_sha", None),
        )

    if stage_is_up_to_date(
        done_seat_metrics,
        inputs=[data_file],
        outputs=[out_seat_metrics, out_seat_metrics_csv],
        config_sha=getattr(cfg, "config_sha", None),
    ):
        seat_metrics_df = pd.read_parquet(out_seat_metrics)
    else:
        seat_metrics_df = compute_seat_metrics(data_file, seat_cfg)
        seat_metrics_table = pa.Table.from_pandas(seat_metrics_df, preserve_index=False)
        write_parquet_atomic(seat_metrics_table, out_seat_metrics)
        write_csv_atomic(seat_metrics_df, out_seat_metrics_csv)
        _write_stamp(
            stamp_seat_metrics,
            inputs=[data_file],
            outputs=[out_seat_metrics, out_seat_metrics_csv],
        )
        write_stage_done(
            done_seat_metrics,
            inputs=[data_file],
            outputs=[out_seat_metrics, out_seat_metrics_csv],
            config_sha=getattr(cfg, "config_sha", None),
        )

    if stage_is_up_to_date(
        done_symmetry,
        inputs=[symmetry_input],
        outputs=[out_symmetry, out_symmetry_csv],
        config_sha=getattr(cfg, "config_sha", None),
    ):
        symmetry_df = pd.read_parquet(out_symmetry)
    else:
        symmetry_df = compute_symmetry_checks(symmetry_input, seat_cfg)
        symmetry_table = pa.Table.from_pandas(symmetry_df, preserve_index=False)
        write_parquet_atomic(symmetry_table, out_symmetry)
        write_csv_atomic(symmetry_df, out_symmetry_csv)
        _write_stamp(
            stamp_symmetry,
            inputs=[symmetry_input],
            outputs=[out_symmetry, out_symmetry_csv],
        )
        write_stage_done(
            done_symmetry,
            inputs=[symmetry_input],
            outputs=[out_symmetry, out_symmetry_csv],
            config_sha=getattr(cfg, "config_sha", None),
        )

    if not metrics_df.empty:
        leader = metrics_df.sort_values(["wins", "win_rate"], ascending=False).iloc[0]
        LOGGER.info(
            "Metrics leaderboard computed",
            extra={
                "stage": "metrics",
                "top_strategy": leader["strategy"],
                "wins": int(leader["wins"]),
                "games": int(leader["games"]),
            },
        )

    _write_stamp(
        stamp,
        inputs=stage_inputs,
        outputs=[
            out_metrics,
            out_seats,
            out_seats_parquet,
            out_seat_metrics,
            out_seat_metrics_csv,
            out_symmetry,
            out_symmetry_csv,
            *iso_paths,
        ],
    )

    LOGGER.info(
        "Metrics stage complete",
        extra={
            "stage": "metrics",
            "rows": len(metrics_df),
            "seat_rows": len(seat_df),
            "metrics_path": str(out_metrics),
            "seat_path": str(out_seats),
            "seat_parquet": str(out_seats_parquet),
            "seat_metrics": str(out_seat_metrics),
            "symmetry_checks": str(out_symmetry),
        },
    )
    write_stage_done(
        done,
        inputs=stage_inputs,
        outputs=outputs,
        config_sha=getattr(cfg, "config_sha", None),
    )


def _ensure_isolated_metrics(
    cfg: AppConfig, player_counts: Sequence[int]
) -> tuple[list[Path], list[Path]]:
    """Generate normalized per-player-count metrics where available.

    Args:
        cfg: Application configuration containing metrics locations.
        player_counts: Player counts to process.

    Returns:
        Tuple of normalized parquet paths discovered and the corresponding raw
        inputs checked on disk.
    """
    iso_paths: list[Path] = []
    raw_inputs: list[Path] = []
    for n in player_counts:
        raw_path = cfg.results_dir / f"{n}_players" / f"{n}p_metrics.parquet"
        preferred = cfg.metrics_isolated_path(n)
        legacy = cfg.legacy_metrics_isolated_path(n)
        raw_inputs.append(raw_path)
        if not raw_path.exists():
            LOGGER.warning(
                "Expanded metrics missing",
                extra={"stage": "metrics", "player_count": n, "path": str(raw_path)},
            )
            continue
        try:
            iso_paths.append(build_isolated_metrics(cfg, n))
            continue
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Failed to normalize metrics parquet",
                extra={
                    "stage": "metrics",
                    "player_count": n,
                    "path": str(raw_path),
                    "error": str(exc),
                },
            )
        if preferred.exists():
            iso_paths.append(preferred)
        elif legacy.exists():
            LOGGER.info(
                "Using legacy isolated metrics path",
                extra={
                    "stage": "metrics",
                    "player_count": n,
                    "path": str(legacy),
                },
            )
            iso_paths.append(legacy)
    return iso_paths, raw_inputs


def _collect_metrics_frames(paths: Iterable[Path]) -> pd.DataFrame:
    """Load multiple metrics parquets into a single DataFrame."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(
            columns=[
                "strategy",
                "n_players",
                "games",
                "wins",
                "win_rate",
                "win_prob",
                "expected_score",
            ]
        )
    df = pd.concat(frames, ignore_index=True)
    if "win_prob" not in df.columns:
        df["win_prob"] = df["win_rate"]
    base_cols = [
        "strategy",
        "n_players",
        "games",
        "wins",
        "win_rate",
        "win_prob",
        "expected_score",
    ]
    remainder = [c for c in df.columns if c not in base_cols]
    return df[base_cols + remainder]


def _add_win_rate_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """Attach standard errors and normal-approximation CIs for win rates."""

    if df.empty:
        return df

    out = df.copy()
    games = pd.to_numeric(out["games"], errors="coerce")
    win_rate = pd.to_numeric(out["win_rate"], errors="coerce")
    positive_games = games > 0

    se = pd.Series(0.0, index=out.index, dtype="float64")
    safe_games = games.where(positive_games)
    win_prob = win_rate.loc[positive_games]
    se.loc[positive_games] = ((win_prob * (1.0 - win_prob)) / safe_games.loc[positive_games]).pow(0.5)
    out["se_win_rate"] = se

    z = 1.96
    ci_lo = (win_rate - z * se).clip(lower=0.0, upper=1.0)
    ci_hi = (win_rate + z * se).clip(lower=0.0, upper=1.0)
    out["win_rate_ci_lo"] = ci_lo.where(positive_games, win_rate)
    out["win_rate_ci_hi"] = ci_hi.where(positive_games, win_rate)

    desired_order = [
        "strategy",
        "n_players",
        "games",
        "wins",
        "win_rate",
        "se_win_rate",
        "win_rate_ci_lo",
        "win_rate_ci_hi",
    ]
    remaining = [c for c in out.columns if c not in desired_order]
    return out[desired_order + remaining]


def _downcast_metric_counters(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast integer counters when safe to reduce parquet size."""
    if df.empty:
        return df
    out = df.copy()
    int_cols = {"games", "wins", "n_players", "seed"}
    int32_min = np.iinfo(np.int32).min
    int32_max = np.iinfo(np.int32).max

    for col in out.columns:
        if col in int_cols or col.endswith("_count") or col.startswith("n_"):
            if col not in out.columns:
                continue
            series = pd.to_numeric(out[col], errors="coerce")
            non_null = series.dropna()
            if non_null.empty:
                continue
            if not np.all(np.isclose(non_null, np.floor(non_null))):
                continue
            if non_null.min() < int32_min or non_null.max() > int32_max:
                continue
            out[col] = pd.to_numeric(series, downcast="integer")
    return out


def _compute_seat_advantage(cfg: AppConfig, combined: Path) -> pd.DataFrame:
    """Backwards-compatible wrapper for seat-advantage calculations."""

    seat_cfg = SeatMetricConfig(seat_range=cfg.metrics_seat_range)
    return compute_seat_advantage(cfg, combined, seat_cfg)


def _stamp(path: Path) -> dict[str, float | int]:
    """Capture filesystem metadata for cache stamps."""
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _write_stamp(stamp_path: Path, *, inputs: Iterable[Path], outputs: Iterable[Path]) -> None:
    """Persist a JSON stamp summarizing inputs and outputs for auditing."""
    payload = {
        "inputs": {str(p): _stamp(p) for p in inputs if p.exists()},
        "outputs": {str(p): _stamp(p) for p in outputs if p.exists()},
    }
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(stamp_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2))
