"""Finalize canonical raw by-k rows and publish schema manifests."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.config import AppConfig
from farkle.utils.schema_helpers import expected_schema_for, n_players_from_schema
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


def _schema_hash(n_players: int) -> str:
    schema = expected_schema_for(n_players)
    serializer = getattr(pa.ipc, "serialize", None)
    data = (
        serializer(schema).to_buffer().to_pybytes()
        if serializer is not None
        else schema.serialize().to_pybytes()
    )
    return hashlib.sha256(data).hexdigest()


def _write_manifest(
    manifest_path: Path,
    *,
    rows: int,
    schema: pa.Schema,
    cfg: AppConfig,
) -> None:
    n_players = n_players_from_schema(schema)
    payload: dict[str, Any] = {
        "row_count": rows,
        "schema_hash": _schema_hash(n_players),
        "compression": cfg.parquet_codec,
        "config_sha": cfg.config_sha,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(manifest_path)) as staged:
        Path(staged).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def run(cfg: AppConfig) -> None:
    """Finalize every configured by-k raw artifact without scope fallback."""

    player_counts = sorted({int(k) for k in cfg.sim.n_players_list})
    raw_files = [cfg.ingested_rows_raw(k) for k in player_counts]
    curated_files = [cfg.ingested_rows_curated(k) for k in player_counts]
    manifests = [cfg.manifest_for(k) for k in player_counts]
    done = stage_done_path(cfg.curate_stage_dir, "curate")
    if stage_is_up_to_date(
        done,
        inputs=raw_files,
        outputs=[*curated_files, *manifests],
        cfg=cfg,
        stage="curate",
    ):
        LOGGER.info("Curate up-to-date", extra={"stage": "curate", "path": str(done)})
        return

    missing = [path for path in raw_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "curate: incomplete canonical raw by-k support: "
            + ", ".join(str(path) for path in missing)
        )

    total_rows = 0
    for raw, output, manifest in zip(raw_files, curated_files, manifests, strict=True):
        metadata = pq.read_metadata(raw)
        schema = metadata.schema.to_arrow_schema()
        expected = expected_schema_for(n_players_from_schema(schema))
        if schema.names != expected.names:
            raise ValueError(f"curate: incompatible schema for {raw}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(str(output)) as staged:
            shutil.copy2(raw, staged)
        _write_manifest(manifest, rows=metadata.num_rows, schema=schema, cfg=cfg)
        total_rows += metadata.num_rows

    write_stage_done(
        done,
        inputs=raw_files,
        outputs=[*curated_files, *manifests],
        cfg=cfg,
        stage="curate",
    )
    LOGGER.info(
        "Curate complete",
        extra={"stage": "curate", "files": len(curated_files), "rows": total_rows},
    )


__all__ = ["run"]
