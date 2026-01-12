"""Seat-level statistics derived from combined curated rows."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from farkle.config import AppConfig
from farkle.utils.schema_helpers import n_players_from_schema

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeatMetricConfig:
    """Lightweight container describing seat-level aggregation needs."""

    seat_range: tuple[int, int]
    symmetry_tolerance: float = 1e-3


def compute_seat_metrics(combined: Path, seat_config: SeatMetricConfig) -> pd.DataFrame:
    """Aggregate per-(strategy, seat, n_players) metrics from curated rows.

    Args:
        combined: Path to the consolidated curated parquet.
        seat_config: Configuration controlling seat range and QA tolerances.

    Returns:
        DataFrame with per-seat win rates and descriptive statistics for
        scores, farkles, and rounds.
    """

    ds_in = ds.dataset(combined)
    schema = ds_in.schema
    n_players_fallback = n_players_from_schema(schema)

    seats = list(range(seat_config.seat_range[0], seat_config.seat_range[1] + 1))
    base_columns = [col for col in ["winner_seat", "seat_ranks", "n_rounds"] if col in schema.names]
    seat_ids = [seat for seat in seats if f"P{seat}_strategy" in schema.names]
    if not seat_ids:
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
    scanner = ds_in.scanner(columns=columns, batch_size=batch_size)

    aggregate_frames: list[pd.DataFrame] = []

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        if "seat_ranks" in df.columns:
            df["n_players"] = df["seat_ranks"].apply(
                lambda ranks: len(ranks)
                if isinstance(ranks, (list, tuple, np.ndarray))
                else n_players_fallback
            )
        else:
            df["n_players"] = n_players_fallback

        seat_frames: list[pd.DataFrame] = []
        for seat in seat_ids:
            strat_col = f"P{seat}_strategy"
            if strat_col not in df.columns:
                continue

            score_col = f"P{seat}_score"
            farkles_col = f"P{seat}_farkles"
            rounds_col = f"P{seat}_rounds"

            score_series = df[score_col] if score_col in df.columns else pd.Series(np.nan, index=df.index)
            farkles_series = (
                df[farkles_col] if farkles_col in df.columns else pd.Series(np.nan, index=df.index)
            )
            if rounds_col in df.columns:
                rounds_series = df[rounds_col]
            else:
                rounds_series = df["n_rounds"] if "n_rounds" in df.columns else pd.Series(np.nan, index=df.index)

            seat_frames.append(
                pd.DataFrame(
                    {
                        "strategy": df[strat_col],
                        "seat": seat,
                        "n_players": df["n_players"],
                        "is_win": df.get("winner_seat") == f"P{seat}",
                        "score": pd.to_numeric(score_series, errors="coerce"),
                        "farkles": pd.to_numeric(farkles_series, errors="coerce"),
                        "rounds": pd.to_numeric(rounds_series, errors="coerce"),
                    }
                )
            )

        if not seat_frames:
            continue

        records = pd.concat(seat_frames, ignore_index=True)
        records = records.dropna(subset=["strategy"])
        if records.empty:
            continue
        records["strategy"] = records["strategy"].astype("category")

        grouped = (
            records.groupby(["strategy", "seat", "n_players"], observed=True, sort=False)
            .agg(
                games=("strategy", "size"),
                wins=("is_win", "sum"),
                score_sum=("score", "sum"),
                score_count=("score", "count"),
                farkles_sum=("farkles", "sum"),
                farkles_count=("farkles", "count"),
                rounds_sum=("rounds", "sum"),
                rounds_count=("rounds", "count"),
            )
            .reset_index()
        )
        aggregate_frames.append(grouped)

    if not aggregate_frames:
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

    aggregated = pd.concat(aggregate_frames, ignore_index=True)
    totals = (
        aggregated.groupby(["strategy", "seat", "n_players"], observed=True, sort=False)
        .agg(
            games=("games", "sum"),
            wins=("wins", "sum"),
            score_sum=("score_sum", "sum"),
            score_count=("score_count", "sum"),
            farkles_sum=("farkles_sum", "sum"),
            farkles_count=("farkles_count", "sum"),
            rounds_sum=("rounds_sum", "sum"),
            rounds_count=("rounds_count", "sum"),
        )
        .reset_index()
    )

    score_count = totals["score_count"].replace(0, np.nan)
    farkles_count = totals["farkles_count"].replace(0, np.nan)
    rounds_count = totals["rounds_count"].replace(0, np.nan)

    totals["win_rate"] = np.where(totals["games"] > 0, totals["wins"] / totals["games"], 0.0)
    totals["mean_score"] = totals["score_sum"] / score_count
    totals["mean_farkles"] = totals["farkles_sum"] / farkles_count
    totals["mean_rounds"] = totals["rounds_sum"] / rounds_count

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


def compute_seat_advantage(cfg: AppConfig, combined: Path, seat_config: SeatMetricConfig) -> pd.DataFrame:
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

    denom = {i: sum(_rows_for_n(n) for n in range(i, 13)) for i in seats}
    ds_all = ds.dataset(combined, format="parquet")
    seat_wins = {
        i: int(ds_all.count_rows(filter=(ds.field("winner_seat") == f"P{i}"))) for i in seats
    }

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
        LOGGER.warning(
            "Skipping symmetry diagnostics; required columns missing",
            extra={"missing_columns": sorted(required - set(ds_in.schema.names))},
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
    filter_expr = None
    if "n_players" in ds_in.schema.names:
        filter_expr = ds.field("n_players") == 2
    if filter_expr is not None:
        table = ds_in.to_table(columns=column_list, filter=filter_expr)
    else:
        table = ds_in.to_table(columns=column_list)
    df = table.to_pandas(categories=["P1_strategy", "P2_strategy"])

    df["P1_strategy"] = df["P1_strategy"].astype("category")
    df["P2_strategy"] = df["P2_strategy"].astype("category")
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

    grouped = symmetric.groupby(["P1_strategy", "n_players"], observed=True, sort=False)
    rows: list[pd.Series] = []
    tol = seat_config.symmetry_tolerance

    for (strategy, players), block in grouped:
        p1_farkles = pd.to_numeric(block["P1_farkles"], errors="coerce")
        p2_farkles = pd.to_numeric(block["P2_farkles"], errors="coerce")
        p1_rounds = pd.to_numeric(block["P1_rounds"], errors="coerce")
        p2_rounds = pd.to_numeric(block["P2_rounds"], errors="coerce")

        row = {
            "strategy": strategy,
            "n_players": int(players),
            "observations": int(block.shape[0]),
            "mean_p1_farkles": float(p1_farkles.mean()),
            "mean_p2_farkles": float(p2_farkles.mean()),
            "farkle_diff": float(p1_farkles.mean() - p2_farkles.mean()),
            "mean_p1_rounds": float(p1_rounds.mean()),
            "mean_p2_rounds": float(p2_rounds.mean()),
            "rounds_diff": float(p1_rounds.mean() - p2_rounds.mean()),
        }
        row["farkle_flagged"] = abs(row["farkle_diff"]) > tol
        row["rounds_flagged"] = abs(row["rounds_diff"]) > tol
        rows.append(pd.Series(row))

    return pd.DataFrame(rows)
