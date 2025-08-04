# src/farkle/ingest.py
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis_config import PipelineCfg

log = logging.getLogger(__name__)

def _iter_shards(block: Path, cols: tuple[str, ...]):
    """Yield one (DataFrame, source_path) per shard, limited to *cols*."""
    # Newer runs emit a single `<Np_rows>.parquet` file directly under the
    # block directory. Prefer that if present.
    row_file = next(block.glob("*p_rows.parquet"), None)
    if row_file is not None:
        yield pd.read_parquet(row_file, columns=list(cols)), row_file
        return

    # Legacy layout with `<Np_rows>` directory containing shards
    row_dirs = [p for p in block.glob("*_rows") if p.is_dir()]
    if row_dirs:
        for shard_path in row_dirs[0].glob("*.parquet"):
            yield pd.read_parquet(shard_path, columns=list(cols)), shard_path
    else:  # compact parquet or CSV
        for pqt in block.glob("*.parquet"):
            yield pd.read_parquet(pqt, columns=list(cols)), pqt
        csv = block / "winners.csv"
        if csv.exists():
            yield pd.read_csv(csv, usecols=list(cols), dtype_backend="pyarrow"), csv



# Regex once, reuse
_SEAT_RE = re.compile(r"^P(\d+)_strategy$")

def _fix_winner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich *df* with:
    • winner_strategy – the strategy string for the winning seat
    • winner_seat     – seat label (P1 … PN)
    • seat_ranks      – tuple of seat labels ordered by rank (winner first)

    Leaves all original columns intact so caller can `drop()` later.
    """
    if "winner_strategy" in df.columns:
        # Already processed — nothing to do.
        return df

    df = df.copy(deep=False)                     # cheap shallow copy
    seat_cols = sorted(
        [c for c in df.columns if _SEAT_RE.match(c)],
        key=lambda c: int(_SEAT_RE.match(c).group(1)),  # natural seat order  # type: ignore
    )

    # --- Step 1: find seat label (P#) that won --------------------------
    df["winner_seat"] = df["winner"]            # winner column still holds P#

    # --- Step 2: promote strategy string -------------------------------
    seat_idx = df["winner_seat"].str.extract(r"P(\d+)").astype("int64")[0] - 1
    df["winner_strategy"] = df[seat_cols].to_numpy()[
        np.arange(len(df)), seat_idx.to_numpy()
    ]

    # --- Step 3: capture finishing order -------------------------------
    def _row_to_seat_ranks(row):
        pairs = [
            (seat[:-9], row[seat.replace("_strategy", "_rank")]) for seat in seat_cols
        ]
        return tuple(seat for seat, rk in sorted(pairs, key=lambda p: p[1]))

    rank_cols = [c.replace("_strategy", "_rank") for c in seat_cols]
    if all(col in df.columns for col in rank_cols):
        df["seat_ranks"] = df.apply(_row_to_seat_ranks, axis=1)

    return df


def run(cfg: PipelineCfg) -> None:
    log.info("Ingest started: root=%s", cfg.root)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.curated_parquet
    if out_path.exists():
        out_path.unlink()  # remove old file; idempotent

    # Prepare Arrow writer once
    writer = None
    total_rows = 0

    try:
        for block in sorted(cfg.root.glob(cfg.results_glob)):
            log.info("Reading block %s", block.name)
            expected_cols = set(cfg.ingest_cols)
            for shard_df, shard_path in _iter_shards(block, cfg.ingest_cols):
                if shard_df.empty:
                    log.debug("Shard %s is empty — skipped", shard_path.name)
                    continue

                if set(shard_df.columns) != expected_cols:
                    log.error("Schema mismatch in %s", shard_path)
                    raise RuntimeError("Shard DataFrame columns do not match expected columns")

                shard_df = shard_df.reindex(columns=cfg.ingest_cols)
                log.debug("Shard %s → %d rows", shard_path.name, len(shard_df))

                shard_df = _fix_winner(shard_df)
                canonical_cols = list(cfg.ingest_cols) + [
                    "winner_seat",
                    "winner_strategy",
                ]
                if "seat_ranks" in shard_df.columns:
                    canonical_cols.append("seat_ranks")
                shard_df = shard_df.reindex(columns=canonical_cols)

                total_rows += len(shard_df)

                # Lazily open writer on first chunk
                table = pa.Table.from_pandas(shard_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        out_path,
                        table.schema,
                        compression=cfg.parquet_codec,
                    )

                writer.write_table(table, row_group_size=cfg.row_group_size)
    finally:
        if writer:
            writer.close()
    log.info("Ingest finished — %d rows written to %s", total_rows, out_path)
