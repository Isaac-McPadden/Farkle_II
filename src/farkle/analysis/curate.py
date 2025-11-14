# src/farkle/analysis/curate.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.config import AppConfig
from farkle.utils.schema_helpers import (
    expected_schema_for,
    n_players_from_schema,
)
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
def _schema_hash(n_players: int) -> str:
    schema = expected_schema_for(n_players)

    # ---- get raw bytes ---------------------------------------------------
    pa_serialize = getattr(pa.ipc, "serialize", None)
    if pa_serialize is not None:  # PyArrow ≤ 19
        buf_bytes = pa_serialize(schema).to_buffer().to_pybytes()
    else:  # PyArrow ≥ 20
        buf_bytes = schema.serialize().to_pybytes()

    # ---- hash ------------------------------------------------------------
    return hashlib.sha256(buf_bytes).hexdigest()


def _write_manifest(manifest_path: Path, *, rows: int, schema: pa.Schema, cfg: AppConfig) -> None:
    """Dump a JSON manifest next to the curated parquet."""
    n_players = n_players_from_schema(schema)
    schema_hash = _schema_hash(n_players)
    payload: dict[str, Any] = {
        "row_count": rows,
        "schema_hash": schema_hash,
        "compression": cfg.parquet_codec,
        "config_sha": getattr(cfg, "config_sha", None),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(manifest_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2))
    LOGGER.info(
        "Curate manifest written",
        extra={
            "stage": "curate",
            "path": str(manifest_path),
            "rows": rows,
            "n_players": n_players,
        },
    )


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
    expected_rows = meta.get("row_count")
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
        LOGGER.info(
            "Curate schema mismatch detected",
            extra={
                "stage": "curate",
                "path": out_file.name,
                "expected_hash": expected_hash,
                "actual_hash": actual_hash,
            },
        )
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
def run(cfg: AppConfig) -> None:
    """Curate raw parquet files produced by :func:`farkle.ingest.run`."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    # Ensure existing curated files always have a manifest
    for curated in sorted(cfg.data_dir.glob("*p/*_ingested_rows.parquet")):
        n = int(curated.parent.name.removesuffix("p"))
        manifest = cfg.manifest_for(n)
        if not manifest.exists():
            raise FileNotFoundError(f"missing manifest for {curated}")

    finalized_files = 0
    finalized_rows = 0

    # New layout: analysis/data/*p/*_ingested_rows.raw.parquet
    raw_files = sorted((cfg.data_dir).glob("*p/*_ingested_rows.raw.parquet"))
    if raw_files:
        for raw_file in raw_files:
            n = int(raw_file.parent.name.removesuffix("p"))
            dst_file = cfg.ingested_rows_curated(n)
            manifest = cfg.manifest_for(n)

            md = pq.read_metadata(raw_file)
            schema = md.schema.to_arrow_schema()
            _write_manifest(manifest, rows=md.num_rows, schema=schema, cfg=cfg)

            canonical = raw_file.parent / f"{n}p_ingested_rows.parquet"
            raw_file.replace(canonical)

            if dst_file != canonical:
                if dst_file.exists():
                    dst_file.unlink()
                try:
                    os.link(canonical, dst_file)
                except OSError:
                    shutil.copy2(canonical, dst_file)

            LOGGER.info(
                "Curate: parquet finalized",
                extra={
                    "stage": "curate",
                    "path": canonical.name,
                    "rows": md.num_rows,
                    "row_groups": md.num_row_groups,
                },
            )
            finalized_files += 1
            finalized_rows += md.num_rows
        LOGGER.info(
            "Curate finished",
            extra={
                "stage": "curate",
                "files": finalized_files,
                "rows": finalized_rows,
            },
        )
        return

    # Legacy single-file layout
    raw_file = cfg.curated_parquet.with_suffix(".raw.parquet")
    dst_file = cfg.curated_parquet
    manifest = cfg.analysis_dir / "manifest.jsonl"

    if _already_curated(dst_file, manifest):
        LOGGER.info(
            "Curate: output up-to-date",
            extra={"stage": "curate", "path": dst_file.name},
        )
        LOGGER.info(
            "Curate finished",
            extra={"stage": "curate", "files": finalized_files, "rows": finalized_rows},
        )
        return

    if not raw_file.exists():
        raise FileNotFoundError(raw_file)

    md = pq.read_metadata(raw_file)
    schema = md.schema.to_arrow_schema()
    _write_manifest(manifest, rows=md.num_rows, schema=schema, cfg=cfg)

    raw_file.replace(dst_file)
    LOGGER.info(
        "Curate: parquet finalized",
        extra={
            "stage": "curate",
            "path": dst_file.name,
            "rows": md.num_rows,
            "row_groups": md.num_row_groups,
        },
    )
    finalized_files += 1
    finalized_rows += md.num_rows
    LOGGER.info(
        "Curate finished",
        extra={"stage": "curate", "files": finalized_files, "rows": finalized_rows},
    )
