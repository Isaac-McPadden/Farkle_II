# src/farkle/metrics.py
from __future__ import annotations

import csv
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from farkle.analysis_config import PipelineCfg

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
_WIN_COLS = ["winner", "winner_seat", "winning_score", "n_rounds"]


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (center - margin) / denom, (center + margin) / denom


def _write_parquet(tmp: Path, final: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    tbl = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(tbl, tmp, compression="zstd")
    tmp.replace(final)
    log.info("✓ metrics → %s  (%d rows)", final.name, tbl.num_rows)


def _write_csv(tmp: Path, final: Path, rows: list[dict[str, Any]]) -> None:
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(final)
    log.info("✓ seat_advantage → %s", final.name)


def _update_batch_counters(
    arr_win: Any,
    arr_wseat: Any,
    arr_score: Any,
    arr_nrounds: Any,
    wins_by_strategy: Counter[str],
    rounds_by_strategy: Counter[str],
    score_by_strategy: Counter[str],
    wins_by_seat: Counter[str],
) -> None:
    """Vectorised in-memory update using NumPy; no pandas required."""
    # Convert once – arrow → NumPy (zero-copy for numeric columns)
    win = np.asarray(arr_win)
    wseat = np.asarray(arr_wseat)
    score = np.asarray(arr_score,  dtype=np.int64)
    nrounds = np.asarray(arr_nrounds, dtype=np.int64)

    # 1) wins by strategy  ────────────────────────────────────────────
    uniq, counts = np.unique(win, return_counts=True)
    wins_by_strategy.update(dict(zip(uniq.tolist(), counts.tolist(), strict=True)))

    # 2) wins by seat (P1, P2 …)  ────────────────────────────────────
    uniq_s, counts_s = np.unique(wseat, return_counts=True)
    wins_by_seat.update(dict(zip(uniq_s.tolist(), counts_s.tolist(), strict=True)))

    # 3) rounds & scores per winning strategy  ───────────────────────
    # build an index map strategy→idx
    strat_idx = {s: i for i, s in enumerate(uniq)}
    # aggregate via bincount on aligned index array
    idx = np.vectorize(strat_idx.get, otypes=[np.int64])(win)
    rounds_sum = np.bincount(idx, weights=nrounds, minlength=len(uniq))
    score_sum  = np.bincount(idx, weights=score,   minlength=len(uniq))

    rounds_by_strategy.update(dict(zip(uniq.tolist(), rounds_sum.tolist(), strict=True)))
    score_by_strategy.update(dict(zip(uniq.tolist(),  score_sum.tolist(), strict=True)))


# ──────────────────────────────────────────────────────────────────────────────
def run(cfg: PipelineCfg) -> None:
    """Compute win statistics and seat advantage tables.

    Outputs
    -------
    <analysis_dir>/metrics.parquet         (per-strategy & overall KPIs)
    <analysis_dir>/seat_advantage.csv      (P1..P6 win-rates with CI)

    Notes
    -----
    ``mean_score`` and ``mean_rounds`` are conditional averages over games
    the strategy actually won, capturing a typical winning performance. The
    ``expected_score`` divides total points by ``total_games`` to give the
    per-game score expectation with losses counted as zero.
    """
    analysis_dir = cfg.analysis_dir
    data_file = cfg.curated_parquet
    out_metrics = analysis_dir / cfg.metrics_name
    out_seats = analysis_dir / "seat_advantage.csv"

    if all(
        p.exists() and p.stat().st_mtime >= data_file.stat().st_mtime
        for p in (out_metrics, out_seats)
    ):
        log.info("Metrics: outputs up-to-date - skipped")
        return

    # Running aggregates -------------------------------------------------------
    wins_by_strategy: Counter[str] = Counter()
    rounds_by_strategy: Counter[str] = Counter()
    score_by_strategy: Counter[str] = Counter()

    wins_by_seat: Counter[str] = Counter()
    total_games = 0

    reader = ds.dataset(data_file, format="parquet")
    strategy_cols = [
        name
        for name in reader.schema.names
        if name.endswith("_strategy") and name != "winner_strategy"
    ]
    all_strategies: set[str] = set()

    for batch in reader.to_batches(columns=_WIN_COLS + strategy_cols):
        try:
            arr_win = batch["winner"].to_numpy(zero_copy_only=True)
        except pa.ArrowInvalid:  # fallback for types requiring a copy
            arr_win = batch["winner"].to_numpy()

        try:
            arr_wseat = batch["winner_seat"].to_numpy(zero_copy_only=True)
        except pa.ArrowInvalid:
            arr_wseat = batch["winner_seat"].to_numpy()

        arr_score = batch["winning_score"].to_numpy(zero_copy_only=True)
        arr_nrounds = batch["n_rounds"].to_numpy(zero_copy_only=True)

        all_strategies.update(arr_win)
        for col in strategy_cols:
            all_strategies.update(
                s for s in batch[col].to_pylist() if s is not None
            )

        _update_batch_counters(
            arr_win,
            arr_wseat,
            arr_score,
            arr_nrounds,
            wins_by_strategy,
            rounds_by_strategy,
            score_by_strategy,
            wins_by_seat,
        )
        total_games += len(arr_win)

    # Build rows ---------------------------------------------------------------
    metrics_rows: list[dict[str, Any]] = []
    for strat in sorted(all_strategies):
        n = wins_by_strategy[strat]
        metrics_rows.append(
            {
                "strategy": strat,
                "games": total_games,  # all strategies saw every game
                "wins": n,
                "win_rate": n / total_games,
                # Expected value of points per game, counting zero for losses
                "expected_score": score_by_strategy[strat] / total_games,
                # Typical winning score – conditional mean over only games won
                "mean_score": score_by_strategy[strat] / n,
                # Typical number of rounds in a win (losing rounds are unknown)
                "mean_rounds": rounds_by_strategy[strat] / n,
            }
        )

    seat_rows: list[dict[str, Any]] = []
    for seat in sorted(wins_by_seat):
        k = wins_by_seat[seat]
        lo, hi = _wilson_ci(k, total_games)
        seat_rows.append(
            {
                "seat": seat,
                "wins": k,
                "games": total_games,
                "win_rate": k / total_games,
                "ci_lower": lo,
                "ci_upper": hi,
            }
        )

    # Schemas ------------------------------------------------------------------
    metrics_schema = pa.schema(
        [
            ("strategy", pa.string()),
            ("games", pa.int32()),
            ("wins", pa.int32()),
            ("win_rate", pa.float32()),
            ("expected_score", pa.float32()),
            ("mean_score", pa.float32()),
            ("mean_rounds", pa.float32()),
        ]
    )

    # Atomic writes ------------------------------------------------------------
    tmp_metrics = out_metrics.with_suffix(".parquet.in-progress")
    tmp_seats = out_seats.with_suffix(".csv.in-progress")

    _write_parquet(tmp_metrics, out_metrics, metrics_rows, metrics_schema)
    _write_csv(tmp_seats, out_seats, seat_rows)
