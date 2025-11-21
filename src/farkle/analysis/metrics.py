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

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from farkle.analysis.checks import check_pre_metrics
from farkle.analysis.isolated_metrics import build_isolated_metrics
from farkle.config import AppConfig
from farkle.utils.artifacts import write_csv_atomic, write_parquet_atomic
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


def run(cfg: AppConfig) -> None:
    """Compute per-strategy metrics and seat-advantage tables."""

    analysis_dir = cfg.analysis_dir
    data_file = cfg.curated_parquet
    out_metrics = analysis_dir / cfg.metrics_name
    out_seats = analysis_dir / "seat_advantage.csv"
    out_seats_parquet = analysis_dir / "seat_advantage.parquet"
    stamp = analysis_dir / "metrics.done.json"

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
        },
    )

    check_pre_metrics(data_file, winner_col="winner_seat")

    player_counts = sorted({int(n) for n in cfg.sim.n_players_list})
    iso_paths, raw_inputs = _ensure_isolated_metrics(cfg, player_counts)
    metrics_df = _collect_metrics_frames(iso_paths)
    if metrics_df.empty:
        raise RuntimeError("metrics: no isolated metric files generated")

    metrics_table = pa.Table.from_pandas(metrics_df, preserve_index=False)
    write_parquet_atomic(metrics_table, out_metrics)

    seat_df = _compute_seat_advantage(cfg, data_file)
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
        inputs=[data_file, *raw_inputs],
        outputs=[out_metrics, out_seats, out_seats_parquet, *iso_paths],
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
        },
    )


def _ensure_isolated_metrics(cfg: AppConfig, player_counts: Sequence[int]) -> tuple[list[Path], list[Path]]:
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
        raw_inputs.append(raw_path)
        if not raw_path.exists():
            LOGGER.warning(
                "Expanded metrics missing",
                extra={"stage": "metrics", "player_count": n, "path": str(raw_path)},
            )
            continue
        try:
            iso_paths.append(build_isolated_metrics(cfg, n))
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
                "expected_score",
            ]
        )
    df = pd.concat(frames, ignore_index=True)
    base_cols = ["strategy", "n_players", "games", "wins", "win_rate", "expected_score"]
    remainder = [c for c in df.columns if c not in base_cols]
    return df[base_cols + remainder]


def _compute_seat_advantage(cfg: AppConfig, combined: Path) -> pd.DataFrame:
    """Aggregate win rates by seat position across all player counts."""
    def _rows_for_n(n: int) -> int:
        """Return the number of rows recorded for a player-count manifest."""
        manifest = cfg.manifest_for(n)
        if not manifest.exists():
            return 0
        try:
            meta = json.loads(manifest.read_text())
            return int(meta.get("row_count", 0))
        except Exception:  # noqa: BLE001
            return 0

    denom = {i: sum(_rows_for_n(n) for n in range(i, 13)) for i in range(1, 13)}
    ds_all = ds.dataset(combined, format="parquet")
    seat_wins = {
        i: int(ds_all.count_rows(filter=(ds.field("winner_seat") == f"P{i}")))
        for i in range(1, 13)
    }

    seat_rows: list[dict[str, float]] = []
    for i in range(1, 13):
        games = denom[i]
        wins = seat_wins.get(i, 0)
        rate = (wins / games) if games else 0.0
        seat_rows.append(
            {"seat": i, "wins": wins, "games_with_seat": games, "win_rate": rate}
        )
    return pd.DataFrame(seat_rows, columns=["seat", "wins", "games_with_seat", "win_rate"])


def _stamp(path: Path) -> dict[str, float | int]:
    """Capture filesystem metadata for cache stamps."""
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _write_stamp(stamp_path: Path, *, inputs: Iterable[Path], outputs: Iterable[Path]) -> None:
    """Persist a JSON stamp summarizing inputs and outputs for auditing."""
    payload = {
        "inputs": {
            str(p): _stamp(p)
            for p in inputs
            if p.exists()
        },
        "outputs": {
            str(p): _stamp(p)
            for p in outputs
            if p.exists()
        },
    }
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(stamp_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2))
