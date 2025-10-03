# src/farkle/analysis/metrics.py
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from farkle.analysis.checks import check_pre_metrics
from farkle.config import AppConfig
from farkle.utils.artifacts import write_csv_atomic, write_parquet_atomic
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)

_WIN_COLS = ["winner_strategy", "winner_seat", "winning_score", "n_rounds"]


def _write_parquet(final: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    write_parquet_atomic(table, final)
    LOGGER.info(
        "Metrics parquet written",
        extra={"stage": "metrics", "path": final.name, "rows": table.num_rows},
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
    win = np.asarray(arr_win_strategy)
    wseat = np.asarray(arr_wseat)
    score = np.asarray(arr_score, dtype=np.int64)
    nrounds = np.asarray(arr_nrounds, dtype=np.int64)

    uniq, counts = np.unique(win, return_counts=True)
    wins_by_strategy.update(dict(zip(uniq.tolist(), counts.tolist(), strict=True)))

    uniq_seat, seat_counts = np.unique(wseat, return_counts=True)
    wins_by_seat.update(dict(zip(uniq_seat.tolist(), seat_counts.tolist(), strict=True)))

    strat_index = {strategy: idx for idx, strategy in enumerate(uniq)}
    index_array = np.vectorize(strat_index.get, otypes=[np.int64])(win)
    rounds_sum = np.bincount(index_array, weights=nrounds, minlength=len(uniq))
    score_sum = np.bincount(index_array, weights=score, minlength=len(uniq))

    rounds_by_strategy.update(dict(zip(uniq.tolist(), rounds_sum.tolist(), strict=True)))
    score_by_strategy.update(dict(zip(uniq.tolist(), score_sum.tolist(), strict=True)))


def run(cfg: AppConfig) -> None:
    """Compute win statistics and seat-advantage tables from curated data."""

    analysis_dir = cfg.analysis_dir
    data_file = cfg.curated_parquet
    winner_col = "winner_seat"
    out_metrics = analysis_dir / cfg.analysis.metrics_filename
    out_seats = analysis_dir / "seat_advantage.csv"
    out_seats_parquet = analysis_dir / "seat_advantage.parquet"
    stamp = analysis_dir / (cfg.analysis.metrics_filename + cfg.analysis.done_suffix)

    if not data_file.exists():
        raise FileNotFoundError(
            f"metrics: missing combined parquet {data_file} – run combine step first",
        )

    def _stamp(path: Path) -> dict[str, float | int]:
        stats = path.stat()
        return {"mtime": stats.st_mtime, "size": stats.st_size}

    LOGGER.info(
        "Metrics stage start",
        extra={
            "stage": "metrics",
            "data_file": str(data_file),
            "analysis_dir": str(analysis_dir),
            "batch_rows": cfg.analysis.batch_rows,
        },
    )

    if stamp.exists():
        try:
            meta = json.loads(stamp.read_text())
            inputs = meta.get("inputs", {})
            outputs = meta.get("outputs", {})
            if inputs.get(str(data_file)) == _stamp(data_file) and all(
                Path(path).exists() and outputs.get(path) == _stamp(Path(path))
                for path in (str(out_metrics), str(out_seats), str(out_seats_parquet))
            ):
                LOGGER.info(
                    "Metrics: outputs up-to-date",
                    extra={"stage": "metrics", "path": str(analysis_dir)},
                )
                return
        except Exception:
            pass

    check_pre_metrics(data_file, winner_col=winner_col)

    wins_by_strategy: Counter[str] = Counter()
    rounds_by_strategy: Counter[str] = Counter()
    score_by_strategy: Counter[str] = Counter()
    appearances_by_strategy: Counter[str] = Counter()
    wins_by_seat: Counter[str] = Counter()

    reader = ds.dataset(data_file, format="parquet")
    strategy_cols = [
        name for name in reader.schema.names if name.endswith("_strategy") and name != "winner_strategy"
    ]
    all_strategies: set[str] = set()

    for batch in reader.to_batches(columns=_WIN_COLS + strategy_cols, batch_size=cfg.analysis.batch_rows):
        batch_dict = {name: batch.column(name).to_numpy() for name in _WIN_COLS}
        _update_batch_counters(
            batch_dict["winner_strategy"],
            batch_dict["winner_seat"],
            batch_dict["winning_score"],
            batch_dict["n_rounds"],
            wins_by_strategy,
            rounds_by_strategy,
            score_by_strategy,
            wins_by_seat,
        )

        for column in strategy_cols:
            col_values = batch.column(column).to_numpy(zero_copy_only=False)
            for value in col_values:
                if value is not None:
                    appearances_by_strategy[str(value)] += 1
                    all_strategies.add(str(value))

    metrics_rows: list[dict[str, Any]] = []
    for strategy in sorted(all_strategies):
        wins = wins_by_strategy[strategy]
        games = appearances_by_strategy[strategy]
        metrics_rows.append(
            {
                "strategy": strategy,
                "games": games,
                "wins": wins,
                "win_rate": (wins / games) if games else 0.0,
                "expected_score": (score_by_strategy[strategy] / games) if games else 0.0,
                "mean_score": (score_by_strategy[strategy] / wins) if wins else None,
                "mean_rounds": (rounds_by_strategy[strategy] / wins) if wins else None,
            }
        )

    def _rows_for_n(n: int) -> int:
        manifest_path = cfg.manifest_for(n)
        if not manifest_path.exists():
            return 0
        try:
            meta = json.loads(manifest_path.read_text())
            return int(meta.get("row_count", 0))
        except Exception:
            return 0

    denom = {seat: sum(_rows_for_n(n) for n in range(seat, cfg.combine.max_players + 1)) for seat in range(1, cfg.combine.max_players + 1)}

    ds_all = ds.dataset(data_file, format="parquet")
    seat_wins: dict[int, int] = {
        seat: int(ds_all.count_rows(filter=(ds.field(winner_col) == f"P{seat}")))
        for seat in range(1, cfg.combine.max_players + 1)
    }

    seat_rows = []
    for seat in range(1, cfg.combine.max_players + 1):
        games = denom.get(seat, 0)
        wins = seat_wins.get(seat, 0)
        seat_rows.append(
            {
                "seat": seat,
                "wins": wins,
                "games_with_seat": games,
                "win_rate": (wins / games) if games else 0.0,
            }
        )

    seat_df = pd.DataFrame(seat_rows, columns=["seat", "wins", "games_with_seat", "win_rate"])
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
