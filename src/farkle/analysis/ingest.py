# src/farkle/analysis/ingest.py
from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.schema import expected_schema_for
from farkle.config import AppConfig
from farkle.utils.streaming_loop import run_streaming_shard

LOGGER = logging.getLogger(__name__)


def _iter_shards(block: Path, cols: tuple[str, ...]):
    """Yield ``(DataFrame, source_path)`` pairs containing requested columns."""

    def _read_subset(path: Path, wanted: Iterable[str]) -> pd.DataFrame:
        if path.suffix == ".csv":
            df = pd.read_csv(path, dtype_backend="pyarrow")
        else:
            df = pd.read_parquet(path)
        wanted_list = list(wanted)
        present = [column for column in wanted_list if column in df.columns]
        if LOGGER.isEnabledFor(logging.DEBUG) and len(present) < len(wanted_list):
            missing = sorted(set(wanted_list) - set(present))
            LOGGER.debug(
                "Shard missing requested columns",
                extra={"stage": "ingest", "path": path.name, "missing": missing},
            )
        return df[present]

    row_files = sorted(block.glob("*p_rows.parquet"))
    row_file = row_files[0] if row_files else None
    if row_file is not None:
        parquet = pq.ParquetFile(row_file)
        wanted = list(cols)
        present = [column for column in wanted if column in parquet.schema_arrow.names]
        missing = sorted(set(wanted) - set(present))
        if LOGGER.isEnabledFor(logging.DEBUG) and missing:
            LOGGER.debug(
                "Row file missing requested columns",
                extra={"stage": "ingest", "path": row_file.name, "missing": missing},
            )
        for index in range(parquet.num_row_groups):
            table = parquet.read_row_group(index, columns=present)
            yield table.to_pandas(), row_file
        return

    row_dirs = [path for path in block.glob("*_rows") if path.is_dir()]
    if row_dirs:
        for shard_path in sorted(row_dirs[0].glob("*.parquet")):
            yield _read_subset(shard_path, cols), shard_path
        return

    for parquet_path in block.glob("*.parquet"):
        yield _read_subset(parquet_path, cols), parquet_path
    csv = block / "winners.csv"
    if csv.exists():
        yield _read_subset(csv, cols), csv


_SEAT_RE = re.compile(r"^P(\d+)_strategy$")


def _fix_winner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "winner" in df.columns:
        if "winner_seat" in df.columns:
            df.drop(columns=["winner"], inplace=True)
        else:
            df.rename(columns={"winner": "winner_seat"}, inplace=True)

    seat_cols = sorted(
        [column for column in df.columns if _SEAT_RE.match(column)],
        key=lambda column: int(_SEAT_RE.match(column).group(1)),  # type: ignore[arg-type]
    )

    if "winner_strategy" not in df.columns and seat_cols:
        seat_idx = (
            df["winner_seat"].str.extract(r"P(?P<num>\d+)", expand=True)["num"].astype("Int64")
        )
        strategies = df[seat_cols].to_numpy(dtype=object)
        out = np.empty(len(df), dtype=object)
        rows = np.arange(len(df))
        mask = seat_idx.notna().to_numpy()
        out[mask] = strategies[rows[mask], (seat_idx[mask] - 1).astype(int)]
        out[~mask] = None
        df["winner_strategy"] = out

    if "seat_ranks" not in df.columns and seat_cols:
        rank_cols = [col.replace("_strategy", "_rank") for col in seat_cols]
        if all(col in df.columns for col in rank_cols):
            ranks = df[rank_cols].to_numpy(dtype=float)
            fill = ranks.shape[1] + 1
            np.nan_to_num(ranks, copy=False, nan=fill)
            seats = np.array([col.split("_", 1)[0] for col in seat_cols], dtype=object)
            order = np.argsort(ranks, axis=1)
            df["seat_ranks"] = [list(seats[idx]) for idx in order]
        elif "winner_seat" in df.columns:
            df["seat_ranks"] = df["winner_seat"].apply(lambda seat: [seat])

    return df


def _n_from_block(name: str) -> int:
    match = re.match(r"^(\d+)_players$", name)
    return int(match.group(1)) if match else 0


def _process_block(block: Path, cfg: AppConfig) -> int:
    n = _n_from_block(block.name)
    LOGGER.info(
        "Ingest block discovered",
        extra={"stage": "ingest", "block": block.name, "path": str(block)},
    )

    raw_out = cfg.ingested_rows_raw(n)
    src_mtime = 0.0
    row_files = sorted(block.glob("*p_rows.parquet"))
    if row_files:
        src_mtime = row_files[0].stat().st_mtime
    else:
        row_dirs = [path for path in block.glob("*_rows") if path.is_dir()]
        if row_dirs:
            shards = list(row_dirs[0].glob("*.parquet"))
            if shards:
                src_mtime = max(shard.stat().st_mtime for shard in shards)
    if raw_out.exists() and src_mtime and raw_out.stat().st_mtime >= src_mtime:
        LOGGER.info(
            "Ingest block up-to-date",
            extra={"stage": "ingest", "n_players": n, "path": str(raw_out)},
        )
        return 0

    canon = expected_schema_for(n)
    seat_cols = [column for column in canon.names if column.startswith("P")]
    wanted = ("winner", "n_rounds", "winning_score", *seat_cols)

    total = 0

    def _iter_tables():
        nonlocal total
        for shard_df, shard_path in _iter_shards(block, tuple(wanted)):
            if shard_df.empty:
                LOGGER.debug(
                    "Empty shard skipped",
                    extra={"stage": "ingest", "path": shard_path.name},
                )
                continue
            LOGGER.debug(
                "Shard processed",
                extra={"stage": "ingest", "path": shard_path.name, "rows": len(shard_df)},
            )

            shard_df = _fix_winner(shard_df)
            canon_names = canon.names
            extras = sorted(
                column
                for column in shard_df.columns
                if column not in canon_names and not column.startswith("P")
            )
            if extras:
                LOGGER.error(
                    "Schema mismatch",
                    extra={"stage": "ingest", "path": str(shard_path), "unexpected_columns": extras},
                )
                raise RuntimeError("Schema mismatch")
            for name in canon_names:
                if name not in shard_df.columns:
                    shard_df[name] = pd.NA
            shard_df = shard_df[canon_names]
            table = pa.Table.from_pandas(shard_df, schema=canon, preserve_index=False)
            total += len(shard_df)
            yield table

    batches = _iter_tables()
    first = next(batches, None)
    if first is None:
        if raw_out.exists():
            raw_out.unlink()
        manifest_candidate = raw_out.with_suffix(".manifest.jsonl")
        if manifest_candidate.exists():
            manifest_candidate.unlink()
        LOGGER.info(
            "Ingest block produced zero rows",
            extra={"stage": "ingest", "n_players": n, "path": str(block)},
        )
        return 0

    def _all_batches():
        yield first
        yield from batches

    manifest_path = raw_out.with_suffix(".manifest.jsonl")
    run_streaming_shard(
        out_path=str(raw_out),
        manifest_path=str(manifest_path),
        schema=canon,
        batch_iter=_all_batches(),
        row_group_size=cfg.ingest.row_group_size,
        compression=cfg.ingest.parquet_codec,
        manifest_extra={
            "path": raw_out.name,
            "n_players": n,
            "source_block": block.name,
        },
    )
    LOGGER.info(
        "Ingest block complete",
        extra={
            "stage": "ingest",
            "n_players": n,
            "rows": total,
            "path": str(raw_out),
            "manifest": str(manifest_path),
        },
    )
    return total


def run(cfg: AppConfig) -> None:
    LOGGER.info(
        "Ingest started",
        extra={
            "stage": "ingest",
            "root": str(cfg.results_dir),
            "data_dir": str(cfg.data_dir),
            "n_jobs": cfg.ingest.n_jobs,
        },
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    blocks = sorted(
        (path for path in cfg.results_dir.iterdir() if path.is_dir() and path.name.endswith("_players")),
        key=lambda path: (_n_from_block(path.name), path.name),
    )

    total_rows = 0
    if cfg.ingest.n_jobs <= 1:
        for block in blocks:
            total_rows += _process_block(block, cfg)
    else:
        with ProcessPoolExecutor(max_workers=cfg.ingest.n_jobs) as pool:
            futures = [pool.submit(_process_block, block, cfg) for block in blocks]
            for future in futures:
                total_rows += future.result()

    LOGGER.info(
        "Ingest finished",
        extra={"stage": "ingest", "blocks": len(blocks), "rows": total_rows},
    )
