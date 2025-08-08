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
    """Concatenate all per-N parquets into a 12-seat superset with null padding."""
    files: list[Path] = sorted((cfg.data_dir).glob("*p/*_ingested_rows.parquet"))
    if not files:
        log.info("aggregate: no per-N files found under %s", cfg.data_dir)
        return

    target = expected_schema_for(12)  # superset up to P12_*
    parts: list[pa.Table] = []
    total = 0
    for p in files:
        t = pq.read_table(p)
        t = _pad_to_schema(t, target)
        parts.append(t)
        total += t.num_rows

    out_dir = cfg.data_dir / "all_n_players_combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "all_ingested_rows.parquet"
    tmp = out.with_suffix(out.suffix + ".in-progress")

    all_tbl = pa.concat_tables(parts, promote=True)
    pq.write_table(all_tbl, tmp, compression=cfg.parquet_codec, use_dictionary=True)
    tmp.replace(out)
    log.info("aggregate: wrote %s (%d rows)", out, total)
