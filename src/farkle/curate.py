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

from farkle.analysis_config import (
    PipelineCfg,
    expected_schema_for,
    n_players_from_schema,
)

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
def _schema_hash(n_players: int) -> str:
    schema = expected_schema_for(n_players)
    try:
        return pa.ipc.serialize(schema).checksum.hex()
    except AttributeError:  # pyarrow >=20 removed ipc.serialize
        import hashlib

        return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _write_manifest(manifest_path: Path, *, rows: int, schema: pa.Schema, cfg: PipelineCfg) -> None:
    """Dump a simple JSON manifest next to the curated parquet."""
    n_players = n_players_from_schema(schema)
    schema_hash = _schema_hash(n_players)
    schema_list = [str(f) for f in schema]
    payload: dict[str, Any] = {
        "rows": rows,
        "schema": schema_list,
        "schema_hash": schema_hash,
        "codec": cfg.parquet_codec,
        "row_group_size": cfg.row_group_size,
        "git_sha": cfg.git_sha,
        "created": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "pid": str(os.getpid()),
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    log.info("✓ manifest → %s", manifest_path)


def _already_curated(out_file: Path, manifest: Path) -> bool:
    """Return True if both parquet & manifest exist and appear consistent.
    Prevents redoing analyses. Expected behavior is to return a silent False.
    """
    if not (out_file.exists() and manifest.exists()):
        return False
    try:
        meta = json.loads(manifest.read_text())
    except Exception:  # corrupt JSON?  redo
        return False
    expected_rows = meta.get("rows")
    expected_hash = meta.get("schema_hash")
    if expected_rows is None or expected_hash is None:
        return False
    try:
        md = pq.read_metadata(out_file)
        parquet_rows = md.num_rows
        schema = md.schema.to_arrow_schema()
    except Exception:
        return False
    if parquet_rows != expected_rows:
        return False
    n_players = n_players_from_schema(schema)
    actual_hash = _schema_hash(n_players)
    if actual_hash != expected_hash:
        log.info(
            "Curate: %s schema mismatch (expected %s, found %s)",
            out_file.name,
            expected_hash,
            actual_hash,
        )
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
def run(cfg: PipelineCfg) -> None:
    """Finalize the raw parquet written by :func:`farkle.ingest.run`.

    Side-effects
    ------------
    •  Reads ``game_rows.raw.parquet`` produced by ingest
    •  Writes ``<analysis_dir>/data/game_rows.parquet`` and a JSON manifest
    •  Logs progress with the same logger used by :mod:`ingest`
    """
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    raw_file = cfg.curated_parquet.with_suffix(".raw.parquet")
    dst_file = cfg.curated_parquet
    manifest = cfg.analysis_dir / cfg.manifest_name

    if _already_curated(dst_file, manifest):
        log.info("Curate: %s already up-to-date - skipped", dst_file.name)
        return

    if not raw_file.exists():
        raise FileNotFoundError(raw_file)

    md = pq.read_metadata(raw_file)
    schema = md.schema.to_arrow_schema()
    _write_manifest(manifest, rows=md.num_rows, schema=schema, cfg=cfg)

    # Atomic publish: rename raw file to curated destination
    raw_file.replace(dst_file)

    log.info(
        "Curate: wrote %s (%d rows, %d row-groups)",
        dst_file.name,
        md.num_rows,
        md.num_row_groups,
    )
