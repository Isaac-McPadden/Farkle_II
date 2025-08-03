# src/farkle/curate.py
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis_config import PipelineCfg

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
def _write_manifest(manifest_path: Path, *, rows: int, schema: pa.Schema, cfg: PipelineCfg) -> None:
    """Dump a simple JSON manifest next to the curated parquet."""
    payload: dict[str, Any] = {
        "rows": rows,
        "schema": [str(f) for f in schema],
        "codec": cfg.parquet_codec,
        "row_group_size": cfg.row_group_size,
        "git_sha": cfg.git_sha,
        "created": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "pid": str(os.getpid()),
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    log.info("✓ manifest → %s", manifest_path)


def _already_curated(out_file: Path, manifest: Path) -> bool:
    """Return True if both parquet & manifest exist and appear consistent."""
    if not (out_file.exists() and manifest.exists()):
        return False
    try:
        meta = json.loads(manifest.read_text())
    except Exception:  # corrupt JSON?  redo
        return False
    try:
        parquet_rows = pq.read_metadata(out_file).num_rows
    except Exception:
        return False
    return parquet_rows == meta.get("rows")


# ──────────────────────────────────────────────────────────────────────────────
def run(cfg: PipelineCfg) -> None:
    """Finalize the curated parquet written by ingest.run(cfg).
    
    Side-effects
    ------------
    •  Writes <analysis_dir>/data/game_rows.parquet        (if absent or stale)
    •  Writes <analysis_dir>/manifest.json
    •  Logs progress with the same logger used by ingest.py
    """
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    src_tmp = cfg.curated_parquet.with_suffix(".in-progress")
    dst_file = cfg.curated_parquet
    manifest = cfg.analysis_dir / cfg.manifest_name

    if _already_curated(dst_file, manifest):
        log.info("Curate: %s already up-to-date – skipped", dst_file.name)
        return

    #  Stream-copy the row-groups exactly as ingest produced them
    src_rows = 0
    source_pq = cfg.curated_parquet  # ingest’s output
    if not source_pq.exists():
        raise FileNotFoundError(source_pq)

    md = pq.read_metadata(source_pq)
    schema = md.schema.to_arrow_schema()
    with pq.ParquetFile(source_pq) as reader, pq.ParquetWriter(
        src_tmp, schema, compression=cfg.parquet_codec
    ) as writer:
        for i in range(reader.num_row_groups):
            tbl = reader.read_row_group(i)
            writer.write_table(tbl, row_group_size=cfg.row_group_size)
            src_rows += tbl.num_rows
            if i % 10 == 0:
                log.debug("curate: %d/%d row-groups copied", i + 1, reader.num_row_groups)

    # Atomic publish
    src_tmp.replace(dst_file)

    #  Verify & manifest
    log.info("Curate: wrote %s (%d rows, %d row-groups)", dst_file.name, src_rows, md.num_row_groups)
    _write_manifest(manifest, rows=src_rows, schema=schema, cfg=cfg)
