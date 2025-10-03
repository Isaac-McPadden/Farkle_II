# src/farkle/analysis/combine.py
from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.checks import check_post_combine
from farkle.analysis.schema import expected_schema_for
from farkle.config import AppConfig
from farkle.utils.streaming_loop import run_streaming_shard

LOGGER = logging.getLogger(__name__)


def _pad_to_schema(tbl: pa.Table, target: pa.Schema) -> pa.Table:
    columns = []
    for field in target:
        if field.name in tbl.column_names:
            columns.append(tbl[field.name].cast(field.type))
        else:
            columns.append(pa.nulls(len(tbl), field.type))
    return pa.table(columns, names=target.names)


def run(cfg: AppConfig) -> None:
    """Concatenate all per-N parquet files into a superset with null padding."""

    files: list[Path] = sorted(cfg.data_dir.glob("*p/*_ingested_rows.parquet"))
    if not files:
        LOGGER.info(
            "Combine: no inputs discovered",
            extra={"stage": "combine", "path": str(cfg.data_dir)},
        )
        return

    target = expected_schema_for(cfg.combine.max_players)
    out_dir = cfg.data_dir / "all_n_players_combined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / cfg.analysis.combined_filename

    if out.exists():
        newest = max((path.stat().st_mtime for path in files), default=0.0)
        if out.stat().st_mtime >= newest:
            LOGGER.info(
                "Combine: output up-to-date",
                extra={"stage": "combine", "path": str(out)},
            )
            return

    total = 0

    def _iter_row_groups():
        nonlocal total
        for path in files:
            parquet = pq.ParquetFile(path)
            for index in range(parquet.num_row_groups):
                table = parquet.read_row_group(index)
                if table.num_rows == 0:
                    continue
                if table.schema.names != target.names:
                    table = _pad_to_schema(table, target)
                total += table.num_rows
                yield table

    batches = _iter_row_groups()
    first = next(batches, None)
    if first is None:
        if out.exists():
            out.unlink()
        manifest_candidate = out.with_suffix(".manifest.jsonl")
        if manifest_candidate.exists():
            manifest_candidate.unlink()
        LOGGER.info(
            "Combine: inputs produced zero rows",
            extra={"stage": "combine", "path": str(cfg.data_dir)},
        )
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
        row_group_size=cfg.ingest.row_group_size,
        compression=cfg.ingest.parquet_codec,
        manifest_extra={
            "path": out.name,
            "source_files": len(files),
        },
    )

    parquet_out = pq.ParquetFile(out)
    if parquet_out.metadata.num_rows != total:
        raise RuntimeError(
            f"combine: row-count mismatch {parquet_out.metadata.num_rows} != {total}",
        )
    if pq.read_schema(out).names != target.names:
        raise RuntimeError("combine: output schema mismatch")

    LOGGER.info(
        "Combine: parquet written",
        extra={
            "stage": "combine",
            "path": str(out),
            "rows": total,
            "manifest": str(manifest_path),
        },
    )
    check_post_combine(files, out, max_players=cfg.combine.max_players)
