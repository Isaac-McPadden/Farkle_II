# src/farkle/analysis/seed_symmetry.py
"""Seed-level symmetry diagnostics derived from head-to-head self-play artifacts."""

from __future__ import annotations

import logging

import pandas as pd
import pyarrow as pa

from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig
from farkle.utils.artifacts import write_csv_atomic, write_parquet_atomic

LOGGER = logging.getLogger(__name__)


_REQUIRED_COLUMNS = {
    "players",
    "seed",
    "strategy",
    "games",
    "seat1_win_rate",
    "seat2_win_rate",
    "seat_win_rate_diff",
    "mean_farkles_seat1",
    "mean_farkles_seat2",
    "mean_score_seat1",
    "mean_score_seat2",
}


def run(
    cfg: AppConfig,
    *,
    force: bool = False,
    allow_missing_upstream: bool = False,
) -> None:
    """Compute deterministic seed-symmetry diagnostics from head-to-head self-play."""

    source = cfg.head2head_path("bonferroni_selfplay_symmetry.parquet")
    stage_dir = cfg.seed_symmetry_stage_dir
    done = stage_done_path(stage_dir, "seed_symmetry")
    out_seed = stage_dir / "seed_symmetry_checks.parquet"
    out_seed_csv = stage_dir / "seed_symmetry_checks.csv"
    out_summary = stage_dir / "seed_symmetry_summary.parquet"
    out_summary_csv = stage_dir / "seed_symmetry_summary.csv"
    outputs = [out_seed, out_seed_csv, out_summary, out_summary_csv]

    if not source.exists():
        reason = f"missing input: {source}"
        if not allow_missing_upstream:
            write_stage_done(
                done,
                inputs=[],
                outputs=[],
                config_sha=getattr(cfg, "config_sha", None),
                status="failed",
                reason=reason,
                blocking_dependency=str(source),
                upstream_stage="bonferroni_head2head",
            )
            raise FileNotFoundError(f"seed_symmetry requires upstream artifact: {source}")

        LOGGER.warning(
            "Seed-symmetry skipped: missing head2head self-play artifact",
            extra={"stage": "seed_symmetry", "path": str(source)},
        )
        write_stage_done(
            done,
            inputs=[],
            outputs=[],
            config_sha=getattr(cfg, "config_sha", None),
            status="skipped",
            reason=reason,
            blocking_dependency=str(source),
            upstream_stage="bonferroni_head2head",
        )
        return

    if not force and stage_is_up_to_date(
        done,
        inputs=[source],
        outputs=outputs,
        config_sha=getattr(cfg, "config_sha", None),
    ):
        LOGGER.info("Seed-symmetry stage up-to-date", extra={"stage": "seed_symmetry"})
        return

    source_df = pd.read_parquet(source)
    missing = sorted(_REQUIRED_COLUMNS - set(source_df.columns))
    if missing:
        raise ValueError(
            f"seed_symmetry: required columns missing from {source}: {missing}"
        )

    per_seed = _build_seed_level(source_df)
    summary = _build_summary(per_seed)

    write_parquet_atomic(pa.Table.from_pandas(per_seed, preserve_index=False), out_seed)
    write_csv_atomic(per_seed, out_seed_csv)
    write_parquet_atomic(pa.Table.from_pandas(summary, preserve_index=False), out_summary)
    write_csv_atomic(summary, out_summary_csv)

    write_stage_done(
        done,
        inputs=[source],
        outputs=outputs,
        config_sha=getattr(cfg, "config_sha", None),
    )


def _build_seed_level(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["players"] = pd.to_numeric(out["players"], errors="raise").astype(int)
    out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)
    out["games"] = pd.to_numeric(out["games"], errors="raise").astype(int)

    numeric_cols = [
        "seat1_win_rate",
        "seat2_win_rate",
        "seat_win_rate_diff",
        "mean_farkles_seat1",
        "mean_farkles_seat2",
        "mean_score_seat1",
        "mean_score_seat2",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    tol = 1e-3
    out["seat_win_rate_diff_abs"] = out["seat_win_rate_diff"].abs()
    out["win_rate_flagged"] = out["seat_win_rate_diff_abs"] > tol
    out["farkles_diff"] = out["mean_farkles_seat1"] - out["mean_farkles_seat2"]
    out["score_diff"] = out["mean_score_seat1"] - out["mean_score_seat2"]

    cols = [
        "players",
        "seed",
        "strategy",
        "games",
        "seat1_win_rate",
        "seat2_win_rate",
        "seat_win_rate_diff",
        "seat_win_rate_diff_abs",
        "win_rate_flagged",
        "mean_farkles_seat1",
        "mean_farkles_seat2",
        "farkles_diff",
        "mean_score_seat1",
        "mean_score_seat2",
        "score_diff",
    ]
    out = out[cols].sort_values(["players", "seed", "strategy"], kind="mergesort")
    out.reset_index(drop=True, inplace=True)
    return out


def _build_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_seed.groupby(["players", "strategy"], observed=True, sort=False)
        .agg(
            seeds=("seed", "nunique"),
            total_games=("games", "sum"),
            mean_seat_win_rate_diff=("seat_win_rate_diff", "mean"),
            std_seat_win_rate_diff=("seat_win_rate_diff", "std"),
            max_abs_seat_win_rate_diff=("seat_win_rate_diff_abs", "max"),
            mean_abs_seat_win_rate_diff=("seat_win_rate_diff_abs", "mean"),
            flagged_seed_count=("win_rate_flagged", "sum"),
        )
        .reset_index()
    )
    grouped["flagged_seed_count"] = pd.to_numeric(grouped["flagged_seed_count"], errors="coerce").fillna(0).astype(int)
    grouped["flagged_any"] = grouped["flagged_seed_count"] > 0
    grouped = grouped.sort_values(["players", "strategy"], kind="mergesort")
    grouped.reset_index(drop=True, inplace=True)
    return grouped
