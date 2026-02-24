"""Lightweight benchmark comparing legacy vs refactored per-seat processing.

Usage:
    python scripts/benchmark_game_stats_batch_refactor.py --rows 200000 --seats 6
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from farkle.analysis import game_stats


def _build_synthetic(path: Path, *, rows: int, seats: int, seed: int) -> list[tuple[int, Path]]:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {"n_rounds": rng.integers(2, 30, size=rows, dtype=np.int16)}
    for seat in range(1, seats + 1):
        data[f"P{seat}_strategy"] = rng.integers(1, 20, size=rows, dtype=np.int16)
        data[f"P{seat}_score"] = rng.integers(0, 10000, size=rows, dtype=np.int32)
    pd.DataFrame(data).to_parquet(path)
    return [(seats, path)]


def _legacy_per_strategy_stats(per_n_inputs: list[tuple[int, Path]]) -> pd.DataFrame:
    long_frames: list[pd.DataFrame] = []
    for n_players, path in per_n_inputs:
        ds_in = game_stats.ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        for col in strategy_cols:
            scanner = ds_in.scanner(columns=["n_rounds", col], batch_size=65_536)
            for batch in scanner.to_batches():
                df = batch.to_pandas(categories=[col])
                melted = df.melt(id_vars=["n_rounds"], value_vars=[col], value_name="strategy")
                melted = melted.dropna(subset=["strategy"])
                if melted.empty:
                    continue
                melted["n_players"] = n_players
                long_frames.append(melted[["strategy", "n_players", "n_rounds"]])
    long_df = pd.concat(long_frames, ignore_index=True)
    long_df["n_rounds"] = pd.to_numeric(long_df["n_rounds"], errors="coerce")
    long_df = long_df.dropna(subset=["n_rounds", "strategy"])
    grouped = long_df.groupby(["strategy", "n_players"], observed=True, sort=False)["n_rounds"]
    stats = grouped.agg(
        observations="count",
        mean_rounds="mean",
        median_rounds="median",
        std_rounds=lambda s: s.std(ddof=0),
        p10_rounds=lambda s: s.quantile(0.1),
        p50_rounds=lambda s: s.quantile(0.5),
        p90_rounds=lambda s: s.quantile(0.9),
    )
    stats = stats.reset_index()
    return stats


def _legacy_per_strategy_margin_stats(per_n_inputs: list[tuple[int, Path]]) -> pd.DataFrame:
    long_frames: list[pd.DataFrame] = []
    for n_players, path in per_n_inputs:
        ds_in = game_stats.ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        score_cols = [name for name in ds_in.schema.names if name.startswith("P") and name.endswith("_score")]
        for col in strategy_cols:
            scanner = ds_in.scanner(columns=[*score_cols, col], batch_size=65_536)
            for batch in scanner.to_batches():
                df = batch.to_pandas(categories=[col])
                margins = game_stats._compute_margin_columns(df, score_cols)
                melted = df.assign(
                    margin_runner_up=margins["margin_runner_up"], score_spread=margins["score_spread"]
                ).melt(
                    id_vars=["margin_runner_up", "score_spread"],
                    value_vars=[col],
                    value_name="strategy",
                )
                melted = melted.dropna(subset=["strategy"])
                if melted.empty:
                    continue
                melted["n_players"] = n_players
                long_frames.append(
                    melted[["strategy", "n_players", "margin_runner_up", "score_spread"]]
                )
    long_df = pd.concat(long_frames, ignore_index=True)
    long_df["margin_runner_up"] = pd.to_numeric(long_df["margin_runner_up"], errors="coerce")
    long_df["score_spread"] = pd.to_numeric(long_df["score_spread"], errors="coerce")
    long_df = long_df.dropna(subset=["margin_runner_up", "strategy"])
    grouped = long_df.groupby(["strategy", "n_players"], observed=True, sort=False)
    runner_stats = grouped["margin_runner_up"].agg(
        observations="count",
        mean_margin_runner_up="mean",
        median_margin_runner_up="median",
        std_margin_runner_up=lambda s: s.std(ddof=0),
    )
    spread_stats = grouped["score_spread"].agg(
        mean_score_spread="mean",
        median_score_spread="median",
        std_score_spread=lambda s: s.std(ddof=0),
    )
    return runner_stats.join([spread_stats]).reset_index()


def _time_peak_mb(fn, *args, **kwargs):
    tracemalloc.start()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / (1024 * 1024), len(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--seats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tmp", type=Path, default=Path(".bench") / "game_stats_medium.parquet")
    args = parser.parse_args()

    args.tmp.parent.mkdir(parents=True, exist_ok=True)
    per_n_inputs = _build_synthetic(args.tmp, rows=args.rows, seats=args.seats, seed=args.seed)

    results = [
        ("legacy_rounds", *_time_peak_mb(_legacy_per_strategy_stats, per_n_inputs)),
        ("refactored_rounds", *_time_peak_mb(game_stats._per_strategy_stats, per_n_inputs)),
        ("legacy_margin", *_time_peak_mb(_legacy_per_strategy_margin_stats, per_n_inputs)),
        (
            "refactored_margin",
            *_time_peak_mb(game_stats._per_strategy_margin_stats, per_n_inputs, thresholds=(500, 1000)),
        ),
    ]

    for name, elapsed, peak_mb, rows_out in results:
        print(f"{name}: rows={rows_out} elapsed_s={elapsed:.3f} peak_mb={peak_mb:.1f}")


if __name__ == "__main__":
    main()
