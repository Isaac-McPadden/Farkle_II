# src/farkle/metrics.py
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd

from farkle.analysis.analysis_config import PipelineCfg
from farkle.analysis.checks import check_pre_metrics
from farkle.app_config import AppConfig
from farkle.utils.writer import atomic_path
from farkle.utils.artifacts import write_parquet_atomic, write_csv_atomic

LOGGER = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
_WIN_COLS = ["winner_strategy", "winner_seat", "winning_score", "n_rounds"]

def _write_parquet(final: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    tbl = pa.Table.from_pylist(rows, schema=schema)
    write_parquet_atomic(tbl, final)
    LOGGER.info(
        "Metrics parquet written",
        extra={
            "stage": "metrics",
            "path": final.name,
            "rows": tbl.num_rows,
        },
    )


def _update_batch_counters(
    arr_win_strategy: Any,
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
    win = np.asarray(arr_win_strategy)
    wseat = np.asarray(arr_wseat)
    score = np.asarray(arr_score, dtype=np.int64)
    nrounds = np.asarray(arr_nrounds, dtype=np.int64)

    # 1) wins by strategy  ────────────────────────────────────────────
    uniq, counts = np.unique(win, return_counts=True)
    wins_by_strategy.update(dict(zip(uniq.tolist(), counts.tolist(), strict=True)))

    # 2) wins by seat (P1, P2 …)  ────────────────────────────────────
    uniq_seat, counts_s = np.unique(wseat, return_counts=True)
    wins_by_seat.update(dict(zip(uniq_seat.tolist(), counts_s.tolist(), strict=True)))

    # 3) rounds & scores per winning strategy  ───────────────────────
    # build an index map strategy→idx
    strat_idx = {s: i for i, s in enumerate(uniq)}
    # aggregate via bincount on aligned index array
    idx = np.vectorize(strat_idx.get, otypes=[np.int64])(win)
    rounds_sum = np.bincount(idx, weights=nrounds, minlength=len(uniq))
    score_sum = np.bincount(idx, weights=score, minlength=len(uniq))

    rounds_by_strategy.update(dict(zip(uniq.tolist(), rounds_sum.tolist(), strict=True)))
    score_by_strategy.update(dict(zip(uniq.tolist(), score_sum.tolist(), strict=True)))


# ──────────────────────────────────────────────────────────────────────────────
def _pipeline_cfg(cfg: AppConfig | PipelineCfg) -> PipelineCfg:
    return cfg.analysis if isinstance(cfg, AppConfig) else cfg


def run(cfg: AppConfig | PipelineCfg) -> None:
    cfg = _pipeline_cfg(cfg)
    """Compute win statistics and seat advantage tables.

    Outputs
    -------
    <analysis_dir>/metrics.parquet         (per-strategy & overall KPIs)
    <analysis_dir>/seat_advantage.csv      (seat-specific win rates, CSV view)
    <analysis_dir>/seat_advantage.parquet  (seat-specific win rates, Parquet mirror)

    Notes
    -----
    ``mean_score`` and ``mean_rounds`` are conditional averages over games
    the strategy actually won, capturing a typical winning performance. The
    ``expected_score`` divides total points by ``total_games`` to give the
    per-game score expectation with losses counted as zero.
    """
    analysis_dir = cfg.analysis_dir
    data_file = cfg.curated_parquet
    winner_col = "winner_seat"
    out_metrics = analysis_dir / cfg.metrics_name
    out_seats = analysis_dir / "seat_advantage.csv"
    out_seats_parquet = analysis_dir / "seat_advantage.parquet"
    stamp = analysis_dir / "metrics.done.json"

    if not data_file.exists():
        raise FileNotFoundError(
            f"metrics: missing combined parquet {data_file} – run combine step first"
        )

    def _stamp(path: Path) -> dict[str, float | int]:
        st = path.stat()
        return {"mtime": st.st_mtime, "size": st.st_size}

    LOGGER.info(
        "Metrics stage start",
        extra={
            "stage": "metrics",
            "data_file": str(data_file),
            "analysis_dir": str(analysis_dir),
            "batch_rows": cfg.batch_rows,
        },
    )

    if stamp.exists():
        try:
            meta = json.loads(stamp.read_text())
            inputs = meta.get("inputs", {})
            outputs = meta.get("outputs", {})
            if (
                inputs.get(str(data_file)) == _stamp(data_file)
                and all(
                    Path(p).exists() and outputs.get(p) == _stamp(Path(p))
                    for p in (str(out_metrics), str(out_seats), str(out_seats_parquet))
                )
            ):
                LOGGER.info(
                    "Metrics: outputs up-to-date",
                    extra={"stage": "metrics", "path": str(analysis_dir)},
                )
                return
        except Exception:
            pass

    check_pre_metrics(data_file, winner_col=winner_col)

    # Running aggregates -------------------------------------------------------
    wins_by_strategy: Counter[str] = Counter()
    rounds_by_strategy: Counter[str] = Counter()
    score_by_strategy: Counter[str] = Counter()
    appearances_by_strategy: Counter[str] = Counter()

    wins_by_seat: Counter[str] = Counter()

    reader = ds.dataset(data_file, format="parquet")
    strategy_cols = [
        name
        for name in reader.schema.names
        if name.endswith("_strategy") and name != "winner_strategy"
    ]
    all_strategies: set[str] = set()

    for batch in reader.to_batches(
            columns=_WIN_COLS + strategy_cols,
            batch_size=cfg.batch_rows,
        ):
        # Arrow's ``to_numpy`` has strict zero-copy semantics for string data
        # which can raise ``ArrowInvalid`` even for small, null-free arrays. The
        # win/seat columns are tiny, so converting via ``to_pylist`` is simpler
        # and avoids these edge cases.
        arr_wstrat = np.asarray(batch["winner_strategy"].to_pylist())
        arr_wseat = np.asarray(batch["winner_seat"].to_pylist())

        arr_score = batch["winning_score"].to_numpy(zero_copy_only=True)
        arr_nrounds = batch["n_rounds"].to_numpy(zero_copy_only=True)

        all_strategies.update(arr_wstrat)
        for col in strategy_cols:
            col_vals = [s for s in batch[col].to_pylist() if s is not None]
            all_strategies.update(col_vals)
            # appearances denominator for win_rate/expected_score
            if col_vals:
                uniq, counts = np.unique(np.asarray(col_vals), return_counts=True)
                appearances_by_strategy.update(
                    dict(zip(uniq.tolist(), counts.tolist(), strict=True))
                )

        _update_batch_counters(
            arr_wstrat,
            arr_wseat,
            arr_score,
            arr_nrounds,
            wins_by_strategy,
            rounds_by_strategy,
            score_by_strategy,
            wins_by_seat,
        )

    # Build rows ---------------------------------------------------------------
    metrics_rows: list[dict[str, Any]] = []
    for strat in sorted(all_strategies):
        n_wins = wins_by_strategy[strat]
        n_games = appearances_by_strategy[strat]
        metrics_rows.append(
            {
                "strategy": strat,
                "games": n_games,
                "wins": n_wins,
                "win_rate": (n_wins / n_games) if n_games else 0.0,
                "expected_score": (score_by_strategy[strat] / n_games) if n_games else 0.0,
                "mean_score": None if n_wins == 0 else (score_by_strategy[strat] / n_wins),
                "mean_rounds": None if n_wins == 0 else (rounds_by_strategy[strat] / n_wins),
            }
        )

    # Seat denominators from manifests (rows where seat existed)
    def _rows_for_n(n: int) -> int:
        mpath = cfg.manifest_for(n)
        if not mpath.exists():
            return 0
        try:
            meta = json.loads(mpath.read_text())
            return int(meta.get("row_count", 0))
        except Exception:
            return 0

    # denom[seat] = sum of rows across all N where that seat exists (i.e., N >= seat)
    denom = {i: sum(_rows_for_n(n) for n in range(i, 13)) for i in range(1, 13)}

    # wins per seat (from combined parquet)
    ds_all = ds.dataset(data_file, format="parquet")
    seat_wins: dict[int, int] = {
        i: int(ds_all.count_rows(filter=(ds.field(winner_col) == f"P{i}")))
        for i in range(1, 13)
    }

    # Write corrected seat_advantage.csv
    seat_rows: list[dict[str, Any]] = []
    for i in range(1, 13):
        games = denom[i]
        wins = seat_wins.get(i, 0)
        rate = (wins / games) if games else 0.0
        seat_rows.append(
            {
                "seat": i,
                "wins": wins,
                "games_with_seat": games,
                "win_rate": rate,
            }
        )

    seat_df = pd.DataFrame(
        seat_rows,
        columns=["seat", "wins", "games_with_seat", "win_rate"],
    )
    write_csv_atomic(seat_df, out_seats)
    seat_table = pa.Table.from_pandas(
        seat_df,
        preserve_index=False,
        schema=pa.schema(
            [
                ("seat", pa.int32()),
                ("wins", pa.int64()),
                ("games_with_seat", pa.int64()),
                ("win_rate", pa.float64()),
            ]
        ),
    )
    write_parquet_atomic(seat_table, out_seats_parquet)

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
    _write_parquet(out_metrics, metrics_rows, metrics_schema)

    if metrics_rows:
        leader = max(metrics_rows, key=lambda row: row["wins"])
        LOGGER.info(
            "Metrics leaderboard computed",
            extra={
                "stage": "metrics",
                "top_strategy": leader["strategy"],
                "wins": leader["wins"],
                "games": leader["games"],
            },
        )

    stamp.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(stamp)) as tmp_path:
        Path(tmp_path).write_text(
            json.dumps(
                {
                    "inputs": {str(data_file): _stamp(data_file)},
                    "outputs": {
                        str(out_metrics): _stamp(out_metrics),
                        str(out_seats): _stamp(out_seats),
                        str(out_seats_parquet): _stamp(out_seats_parquet),
                    },
                },
                indent=2,
            )
        )

    LOGGER.info(
        "Metrics stage complete",
        extra={
            "stage": "metrics",
            "rows": len(metrics_rows),
            "seat_rows": len(seat_wins),
            "metrics_path": str(out_metrics),
            "seat_path": str(out_seats),
            "seat_parquet": str(out_seats_parquet),
        },
    )
