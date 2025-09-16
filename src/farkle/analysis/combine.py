# src/farkle/combine.py
from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.analysis_config import PipelineCfg, expected_schema_for
from farkle.analysis.checks import check_post_combine
from farkle.app_config import AppConfig
from farkle.utils.streaming_loop import run_streaming_shard

log = logging.getLogger(__name__)

def _pad_to_schema(tbl: pa.Table, target: pa.Schema) -> pa.Table:
    cols = []
    for f in target:
        if f.name in tbl.column_names:
            cols.append(tbl[f.name].cast(f.type))
        else:
            cols.append(pa.nulls(len(tbl), f.type))
    return pa.table(cols, names=target.names)

def _pipeline_cfg(cfg: AppConfig | PipelineCfg) -> PipelineCfg:
    return cfg.analysis if isinstance(cfg, AppConfig) else cfg


def run(cfg: AppConfig | PipelineCfg) -> None:
    cfg = _pipeline_cfg(cfg)
    """Concatenate all per-N parquets into a 12-seat superset with null padding.

    Streaming implementation: copy row-groups into a single writer to bound RAM.
    """
    files: list[Path] = sorted((cfg.data_dir).glob("*p/*_ingested_rows.parquet"))
    if not files:
        log.info("combine: no per-N files found under %s", cfg.data_dir)
        return

    target = expected_schema_for(12)  # superset up to P12_*
    out_dir = cfg.data_dir / "all_n_players_combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "all_ingested_rows.parquet"
    
    # Up-to-date guard: if output is newer than all inputs, skip
    if out.exists():
        newest = max((p.stat().st_mtime for p in files), default=0.0)
        if out.stat().st_mtime >= newest:
            log.info("combine: outputs up-to-date - skipped")
            return

    total = 0

    def _iter_row_groups():
        nonlocal total
        for p in files:
            pf = pq.ParquetFile(p)
            for i in range(pf.num_row_groups):
                t = pf.read_row_group(i)  # small chunk in RAM
                if t.num_rows == 0:
                    continue
                if t.schema.names != target.names:
                    t = _pad_to_schema(t, target)  # pad/reorder lazily
                total += t.num_rows
                yield t

    batches = _iter_row_groups()
    first = next(batches, None)
    if first is None:
        if out.exists():
            out.unlink()
        manifest_candidate = out.with_suffix(".manifest.jsonl")
        if manifest_candidate.exists():
            manifest_candidate.unlink()
        log.info("combine: inputs produced zero rows - skipped")
        return

    def _all_batches():
        yield first
        yield from batches

    manifest_path = out.with_suffix(".manifest.jsonl")
    run_streaming_shard(
        out_path=str(out),
        manifest_path=str(manifest_path),
        schema=target,
        batch_iter=_all_batches(),
        row_group_size=cfg.row_group_size,
        compression=cfg.parquet_codec,
        manifest_extra={
            "path": out.name,
            "source_files": len(files),
        },
    )

    # Sanity check: file opens and row-count matches
    pf_out = pq.ParquetFile(out)
    if pf_out.metadata.num_rows != total:
        raise RuntimeError(
            f"combine: row-count mismatch {pf_out.metadata.num_rows} != {total}"
        )
    if pq.read_schema(out).names != target.names:
        raise RuntimeError("combine: output schema mismatch")

    log.info("combine: wrote %s (%d rows)", out, total)
    check_post_combine(files, out)
