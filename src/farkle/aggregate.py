# src/farkle/aggregate.py
from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis_config import PipelineCfg, expected_schema_for

log = logging.getLogger("aggregate")

def _pad_to_schema(tbl: pa.Table, target: pa.Schema) -> pa.Table:
    cols = []
    for f in target:
        if f.name in tbl.column_names:
            cols.append(tbl[f.name].cast(f.type))
        else:
            cols.append(pa.nulls(len(tbl), f.type))
    return pa.table(cols, names=target.names)

def run(cfg: PipelineCfg) -> None:
    """Concatenate all per-N parquets into a 12-seat superset with null padding.

    Streaming implementation: copy row-groups into a single writer to bound RAM.
    """
    files: list[Path] = sorted((cfg.data_dir).glob("*p/*_ingested_rows.parquet"))
    if not files:
        log.info("aggregate: no per-N files found under %s", cfg.data_dir)
        return

    target = expected_schema_for(12)  # superset up to P12_*
    out_dir = cfg.data_dir / "all_n_players_combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "all_ingested_rows.parquet"
    tmp = out.with_suffix(out.suffix + ".in-progress")
    if tmp.exists():
        tmp.unlink()

    total = 0
    writer = pq.ParquetWriter(tmp, target, compression=cfg.parquet_codec)
    try:
        for p in files:
            pf = pq.ParquetFile(p)
            for i in range(pf.num_row_groups):
                t = pf.read_row_group(i)          # small chunk in RAM
                if t.schema.names != target.names:
                    t = _pad_to_schema(t, target) # pad/reorder lazily
                writer.write_table(t)
                total += t.num_rows
    finally:
        writer.close()
    tmp.replace(out)
    log.info("aggregate: wrote %s (%d rows)", out, total)
