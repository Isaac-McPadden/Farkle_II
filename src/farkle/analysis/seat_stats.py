"""Seat-level statistics derived from combined curated rows."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

from farkle.config import AppConfig
from farkle.utils.progress import ProgressLogConfig, ScheduledProgressLogger
from farkle.utils.schema_helpers import n_players_from_schema
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeatMetricConfig:
    """Lightweight container describing seat-level aggregation needs."""

    seat_range: tuple[int, int]
    symmetry_tolerance: float = 1e-3


def _empty_seat_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "strategy",
            "seat",
            "n_players",
            "games",
            "wins",
            "win_rate",
            "mean_score",
            "mean_farkles",
            "mean_rounds",
        ]
    )


def _write_seat_metrics_progress(
    path: Path | None,
    *,
    combined: Path,
    batch_count: int,
    row_count: int,
    group_count: int,
    seat_ids: list[int],
    complete: bool,
) -> None:
    if path is None:
        return
    payload = {
        "stage": "metrics_seat_metrics",
        "combined": str(combined),
        "processed_batches": int(batch_count),
        "processed_rows": int(row_count),
        "groups": int(group_count),
        "seat_ids": [int(seat) for seat in seat_ids],
        "complete": bool(complete),
    }
    with atomic_path(str(path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def compute_seat_metrics(
    combined: Path,
    seat_config: SeatMetricConfig,
    *,
    include_players: set[int] | None = None,
    progress_path: Path | None = None,
    progress_every_batches: int = 50,
    progress_logging: ProgressLogConfig | None = None,
) -> pd.DataFrame:
    """Aggregate per-(strategy, seat, n_players) metrics from curated rows.

    Args:
        combined: Path to the consolidated curated parquet.
        seat_config: Configuration controlling seat range and QA tolerances.

    Returns:
        DataFrame with per-seat win rates and descriptive statistics for
        scores, farkles, and rounds.
    """

    ds_in = ds.dataset(combined, format="parquet", partitioning="hive")
    schema = ds_in.schema
    n_players_fallback = n_players_from_schema(schema)

    seats = list(range(seat_config.seat_range[0], seat_config.seat_range[1] + 1))
    base_columns = [
        col for col in ["winner_seat", "n_players", "seat_ranks", "n_rounds"] if col in schema.names
    ]
    seat_ids = [seat for seat in seats if f"P{seat}_strategy" in schema.names]
    if not seat_ids:
        return _empty_seat_metrics_frame()

    batch_size = 100_000
    seat_columns: list[str] = []
    for seat in seat_ids:
        seat_columns.extend(
            [
                f"P{seat}_strategy",
                f"P{seat}_score",
                f"P{seat}_farkles",
                f"P{seat}_rounds",
            ]
        )

    columns = base_columns + [col for col in seat_columns if col in schema.names]
    filter_expr = None
    if include_players and "n_players" in schema.names:
        selected = sorted(int(v) for v in include_players)
        expr = None
        for val in selected:
            branch = ds.field("n_players") == int(val)
            expr = branch if expr is None else (expr | branch)
        filter_expr = expr
    scanner = ds_in.scanner(columns=columns, batch_size=batch_size, filter=filter_expr)
    total_rows = int(ds_in.count_rows(filter=filter_expr))

    running: pd.DataFrame | None = None
    processed_batches = 0
    processed_rows = 0
    started_at = time.perf_counter()
    progress_logger = (
        ScheduledProgressLogger(
            LOGGER,
            label="Seat metrics",
            schedule=progress_logging,
            unit="rows",
            total=total_rows,
        )
        if progress_every_batches > 0 and progress_logging is not None and total_rows > 0
        else None
    )

    for batch in scanner.to_batches():
        processed_batches += 1
        processed_rows += int(batch.num_rows)

        if "n_players" not in batch.schema.names and "seat_ranks" in batch.schema.names:
            seat_ranks_idx = batch.schema.get_field_index("seat_ranks")
            seat_ranks = batch.column(seat_ranks_idx)
            n_players_array = pc.list_value_length(seat_ranks)
            batch = batch.append_column("n_players", n_players_array)

        df = batch.to_pandas()
        if "n_players" in df.columns:
            numeric_players = pd.to_numeric(df["n_players"], errors="coerce").fillna(n_players_fallback)
            df["n_players"] = numeric_players.astype(np.int16)
        elif "seat_ranks" in df.columns:
            player_counts = pd.to_numeric(df["seat_ranks"].map(len), errors="coerce").fillna(
                n_players_fallback
            )
            df["n_players"] = player_counts.astype(np.int16)
        else:
            df["n_players"] = n_players_fallback

        for seat in seat_ids:
            strat_col = f"P{seat}_strategy"
            if strat_col not in df.columns:
                continue
            strategies = df[strat_col]
            valid = strategies.notna()
            if not bool(valid.any()):
                continue
            rows = df.loc[valid]

            score_col = f"P{seat}_score"
            farkles_col = f"P{seat}_farkles"
            rounds_col = f"P{seat}_rounds"

            score_series = rows[score_col] if score_col in rows.columns else pd.Series(np.nan, index=rows.index)
            farkles_series = (
                rows[farkles_col]
                if farkles_col in rows.columns
                else pd.Series(np.nan, index=rows.index)
            )
            if rounds_col in df.columns:
                rounds_series = rows[rounds_col]
            else:
                rounds_series = (
                    rows["n_rounds"] if "n_rounds" in rows.columns else pd.Series(np.nan, index=rows.index)
                )
            winner_series = (
                rows["winner_seat"] if "winner_seat" in rows.columns else pd.Series("", index=rows.index)
            )

            seat_frame = pd.DataFrame(
                {
                    "strategy": rows[strat_col],
                    "seat": seat,
                    "n_players": rows["n_players"],
                    "is_win": winner_series == f"P{seat}",
                    "score": pd.to_numeric(score_series, errors="coerce"),
                    "farkles": pd.to_numeric(farkles_series, errors="coerce"),
                    "rounds": pd.to_numeric(rounds_series, errors="coerce"),
                }
            )
            seat_frame["strategy"] = seat_frame["strategy"].astype("category")

            grouped = seat_frame.groupby(["strategy", "seat", "n_players"], observed=True, sort=False).agg(
                games=("strategy", "size"),
                wins=("is_win", "sum"),
                score_sum=("score", "sum"),
                score_count=("score", "count"),
                farkles_sum=("farkles", "sum"),
                farkles_count=("farkles", "count"),
                rounds_sum=("rounds", "sum"),
                rounds_count=("rounds", "count"),
            )
            running = grouped if running is None else running.add(grouped, fill_value=0.0)

        if progress_logger is not None:
            group_count = int(running.shape[0]) if running is not None else 0
            progress_logger.maybe_log(
                processed_rows,
                detail=f"{processed_batches:,} batches, {group_count:,} groups",
                extra={
                    "stage": "metrics",
                    "batches": processed_batches,
                    "rows": processed_rows,
                    "groups": group_count,
                },
            )
            _write_seat_metrics_progress(
                progress_path,
                combined=combined,
                batch_count=processed_batches,
                row_count=processed_rows,
                group_count=group_count,
                seat_ids=seat_ids,
                complete=False,
            )

    if running is None:
        return _empty_seat_metrics_frame()

    totals = running.reset_index()
    for col in ("games", "wins", "score_count", "farkles_count", "rounds_count"):
        totals[col] = pd.to_numeric(totals[col], errors="coerce").fillna(0).round().astype(np.int64)
    for col in ("score_sum", "farkles_sum", "rounds_sum"):
        totals[col] = pd.to_numeric(totals[col], errors="coerce").fillna(0.0).astype(float)

    score_count = totals["score_count"].replace(0, np.nan)
    farkles_count = totals["farkles_count"].replace(0, np.nan)
    rounds_count = totals["rounds_count"].replace(0, np.nan)

    totals["win_rate"] = np.where(totals["games"] > 0, totals["wins"] / totals["games"], 0.0)
    totals["mean_score"] = totals["score_sum"] / score_count
    totals["mean_farkles"] = totals["farkles_sum"] / farkles_count
    totals["mean_rounds"] = totals["rounds_sum"] / rounds_count

    final_group_count = int(running.shape[0])
    _write_seat_metrics_progress(
        progress_path,
        combined=combined,
        batch_count=processed_batches,
        row_count=processed_rows,
        group_count=final_group_count,
        seat_ids=seat_ids,
        complete=True,
    )
    LOGGER.info(
        "Seat metrics complete",
        extra={
            "stage": "metrics",
            "batches": processed_batches,
            "rows": processed_rows,
            "groups": final_group_count,
            "elapsed_sec": float(round(time.perf_counter() - started_at, 1)),
        },
    )

    return totals[
        [
            "strategy",
            "seat",
            "n_players",
            "games",
            "wins",
            "win_rate",
            "mean_score",
            "mean_farkles",
            "mean_rounds",
        ]
    ]


def compute_seat_advantage(
    cfg: AppConfig,
    combined: Path,
    seat_config: SeatMetricConfig,
    *,
    include_players: set[int] | None = None,
) -> pd.DataFrame:
    """Aggregate win rates by seat position with advantage deltas."""

    def _rows_for_n(n: int) -> int:
        manifest = cfg.manifest_for(n)
        if not manifest.exists():
            return 0
        try:
            meta = json.loads(manifest.read_text())
            return int(meta.get("row_count", 0))
        except Exception:  # noqa: BLE001
            return 0

    start, stop = seat_config.seat_range
    seats = list(range(start, stop + 1))

    selected_players = set(include_players or set(range(1, 13)))
    denom = {
        i: sum(_rows_for_n(n) for n in range(i, 13) if n in selected_players)
        for i in seats
    }
    ds_all = ds.dataset(combined, format="parquet", partitioning="hive")
    players_filter = None
    if include_players and "n_players" in ds_all.schema.names:
        selected = sorted(int(v) for v in include_players)
        for val in selected:
            branch = ds.field("n_players") == int(val)
            players_filter = branch if players_filter is None else (players_filter | branch)
    seat_wins = {}
    for i in seats:
        win_filter = ds.field("winner_seat") == f"P{i}"
        combined_filter = win_filter if players_filter is None else (win_filter & players_filter)
        seat_wins[i] = int(ds_all.count_rows(filter=combined_filter))

    seat_rows: list[dict[str, float]] = []
    for i in seats:
        games = denom.get(i, 0)
        wins = seat_wins.get(i, 0)
        rate = (wins / games) if games else 0.0
        seat_rows.append({"seat": i, "wins": wins, "games_with_seat": games, "win_rate": rate})

    df = pd.DataFrame(seat_rows, columns=["seat", "wins", "games_with_seat", "win_rate"])
    if df.empty:
        return df

    df.sort_values("seat", inplace=True)
    df.reset_index(drop=True, inplace=True)

    seat1_rate = df.loc[df["seat"] == min(seats), "win_rate"]
    seat1_rate_val = float(seat1_rate.iloc[0]) if not seat1_rate.empty else float("nan")
    df["win_rate_delta_prev"] = df["win_rate"].diff().fillna(0.0)
    df["win_rate_delta_seat1"] = seat1_rate_val - df["win_rate"]
    return df


def compute_symmetry_checks(curated_rows: Path, seat_config: SeatMetricConfig) -> pd.DataFrame:
    """Compare P1 vs P2 stats for symmetric two-player matchups."""

    ds_in = ds.dataset(curated_rows)
    required = {"P1_strategy", "P2_strategy", "P1_farkles", "P2_farkles", "P1_rounds", "P2_rounds"}
    if not required.issubset(ds_in.schema.names):
        available_columns = sorted(ds_in.schema.names)
        sample_limit = 10
        LOGGER.warning(
            "Skipping symmetry diagnostics; required columns missing",
            extra={
                "missing_columns": sorted(required - set(ds_in.schema.names)),
                "curated_path": str(curated_rows),
                "available_columns_sample": available_columns[:sample_limit],
            },
        )
        return pd.DataFrame(
            columns=[
                "strategy",
                "n_players",
                "observations",
                "mean_p1_farkles",
                "mean_p2_farkles",
                "farkle_diff",
                "mean_p1_rounds",
                "mean_p2_rounds",
                "rounds_diff",
                "farkle_flagged",
                "rounds_flagged",
            ]
        )

    columns = list(required | {"seat_ranks", "n_players"})
    column_list = [c for c in columns if c in ds_in.schema.names]
    filter_expr = ds.field("n_players") == 2 if "n_players" in ds_in.schema.names else None
    scanner = ds_in.scanner(
        columns=column_list,
        batch_size=100_000,
        use_threads=True,
        filter=filter_expr,
    )

    running: pd.DataFrame | None = None
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        categories = [col for col in ("P1_strategy", "P2_strategy") if col in batch.schema.names]
        df = batch.to_pandas(categories=categories)
        if df.empty:
            continue

        if "n_players" in df.columns:
            df["n_players"] = pd.to_numeric(df["n_players"], errors="coerce").fillna(2).astype(int)
        elif "seat_ranks" in df.columns:
            df["n_players"] = df["seat_ranks"].apply(
                lambda ranks: len(ranks) if isinstance(ranks, (list, tuple, np.ndarray)) else 2
            )
        else:
            df["n_players"] = 2

        symmetric_mask = (df["n_players"] == 2) & (df["P1_strategy"] == df["P2_strategy"])
        symmetric = df.loc[symmetric_mask]
        if symmetric.empty:
            continue

        symmetric = symmetric.copy()
        symmetric["P1_farkles"] = pd.to_numeric(symmetric["P1_farkles"], errors="coerce")
        symmetric["P2_farkles"] = pd.to_numeric(symmetric["P2_farkles"], errors="coerce")
        symmetric["P1_rounds"] = pd.to_numeric(symmetric["P1_rounds"], errors="coerce")
        symmetric["P2_rounds"] = pd.to_numeric(symmetric["P2_rounds"], errors="coerce")

        grouped = symmetric.groupby(["P1_strategy", "n_players"], observed=True, sort=False).agg(
            observations=("P1_strategy", "size"),
            p1_farkles_sum=("P1_farkles", "sum"),
            p1_farkles_count=("P1_farkles", "count"),
            p2_farkles_sum=("P2_farkles", "sum"),
            p2_farkles_count=("P2_farkles", "count"),
            p1_rounds_sum=("P1_rounds", "sum"),
            p1_rounds_count=("P1_rounds", "count"),
            p2_rounds_sum=("P2_rounds", "sum"),
            p2_rounds_count=("P2_rounds", "count"),
        )
        running = grouped if running is None else running.add(grouped, fill_value=0.0)

    if running is None:
        return pd.DataFrame(
            columns=[
                "strategy",
                "n_players",
                "observations",
                "mean_p1_farkles",
                "mean_p2_farkles",
                "farkle_diff",
                "mean_p1_rounds",
                "mean_p2_rounds",
                "rounds_diff",
                "farkle_flagged",
                "rounds_flagged",
            ]
        )

    totals = running.reset_index()
    count_cols = [
        "observations",
        "p1_farkles_count",
        "p2_farkles_count",
        "p1_rounds_count",
        "p2_rounds_count",
    ]
    for col in count_cols:
        totals[col] = pd.to_numeric(totals[col], errors="coerce").fillna(0).round().astype(np.int64)
    sum_cols = [
        "p1_farkles_sum",
        "p2_farkles_sum",
        "p1_rounds_sum",
        "p2_rounds_sum",
    ]
    for col in sum_cols:
        totals[col] = pd.to_numeric(totals[col], errors="coerce").fillna(0.0).astype(float)

    totals.rename(columns={"P1_strategy": "strategy"}, inplace=True)
    totals["n_players"] = pd.to_numeric(totals["n_players"], errors="coerce").fillna(2).astype(int)
    totals["mean_p1_farkles"] = totals["p1_farkles_sum"] / totals["p1_farkles_count"].replace(0, np.nan)
    totals["mean_p2_farkles"] = totals["p2_farkles_sum"] / totals["p2_farkles_count"].replace(0, np.nan)
    totals["mean_p1_rounds"] = totals["p1_rounds_sum"] / totals["p1_rounds_count"].replace(0, np.nan)
    totals["mean_p2_rounds"] = totals["p2_rounds_sum"] / totals["p2_rounds_count"].replace(0, np.nan)
    totals["farkle_diff"] = totals["mean_p1_farkles"] - totals["mean_p2_farkles"]
    totals["rounds_diff"] = totals["mean_p1_rounds"] - totals["mean_p2_rounds"]

    tol = seat_config.symmetry_tolerance
    totals["farkle_flagged"] = totals["farkle_diff"].abs() > tol
    totals["rounds_flagged"] = totals["rounds_diff"].abs() > tol
    totals.sort_values(["strategy", "n_players"], inplace=True, kind="mergesort")
    totals.reset_index(drop=True, inplace=True)
    return totals[
        [
            "strategy",
            "n_players",
            "observations",
            "mean_p1_farkles",
            "mean_p2_farkles",
            "farkle_diff",
            "mean_p1_rounds",
            "mean_p2_rounds",
            "rounds_diff",
            "farkle_flagged",
            "rounds_flagged",
        ]
    ]
