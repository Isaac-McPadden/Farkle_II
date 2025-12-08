# src/farkle/analysis/combine.py
"""Combine curated shards into a unified metrics parquet file.

This stage reads per-strategy shards emitted during ingestion, pads them to a
common schema, and concatenates the results into a single dataset for later
analysis steps.
"""
from __future__ import annotations

import logging
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.checks import check_post_combine
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig
from farkle.utils.schema_helpers import expected_schema_for
from farkle.utils.streaming_loop import run_streaming_shard

LOGGER = logging.getLogger(__name__)


def _pad_to_schema(tbl: pa.Table, target: pa.Schema) -> pa.Table:
    """Align a table to a target schema by casting or adding null columns.

    Args:
        tbl: Input Arrow table to adjust.
        target: Desired schema ordering and types.

    Returns:
        Table whose columns match ``target`` in order and dtype.
    """
    cols = []
    for f in target:
        if f.name in tbl.column_names:
            cols.append(tbl[f.name].cast(f.type))
        else:
            cols.append(pa.nulls(len(tbl), f.type))
    return pa.table(cols, names=target.names)


def _migrate_combined_output(cfg: AppConfig) -> Path:
    """Ensure any legacy combined outputs are relocated beside the new pooled path."""

    preferred_dir = cfg.combine_pooled_dir(cfg.combine_max_players)
    preferred_out = preferred_dir / "all_ingested_rows.parquet"
    legacy_candidates = [
        cfg.data_dir / "all_n_players_combined" / "all_ingested_rows.parquet",
        cfg.analysis_dir / "all_n_players_combined" / "all_ingested_rows.parquet",
    ]
    for legacy in legacy_candidates:
        if preferred_out.exists() or not legacy.exists() or legacy == preferred_out:
            continue
        preferred_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Combine: migrating legacy pooled parquet",
            extra={"stage": "combine", "source": str(legacy), "dest": str(preferred_out)},
        )
        shutil.move(str(legacy), preferred_out)
        legacy_manifest = legacy.with_suffix(".manifest.jsonl")
        new_manifest = preferred_out.with_suffix(".manifest.jsonl")
        if legacy_manifest.exists():
            shutil.move(str(legacy_manifest), new_manifest)
    return preferred_out


def run(cfg: AppConfig) -> None:
    """Concatenate all per-N parquets into a 12-seat superset with null padding.

    Streaming implementation: copy row-groups into a single writer to bound RAM.
    """
    preferred = sorted(cfg.data_dir.glob(f"*p/{cfg.curated_rows_name}"))
    legacy = sorted(cfg.data_dir.glob("*p/*_ingested_rows.parquet"))
    files: list[Path] = preferred or legacy
    if not files:
        LOGGER.info(
            "Combine: no inputs discovered",
            extra={"stage": "combine", "path": str(cfg.data_dir)},
        )
        return

    target = expected_schema_for(12)  # superset up to P12_*
    out = _migrate_combined_output(cfg)
    manifest_path = cfg.combined_manifest_path()

    done = stage_done_path(cfg.combine_stage_dir, "combine")
    if stage_is_up_to_date(
        done,
        inputs=files,
        outputs=[out, manifest_path],
        config_sha=getattr(cfg, "config_sha", None),
    ):
        LOGGER.info(
            "Combine: output up-to-date",
            extra={"stage": "combine", "path": str(out)},
        )
        return

    # Up-to-date guard: if output is newer than all inputs, skip
    if out.exists():
        newest = max((p.stat().st_mtime for p in files), default=0.0)
        if out.stat().st_mtime >= newest:
            LOGGER.info(
                "Combine: output up-to-date",
                extra={"stage": "combine", "path": str(out)},
            )
            write_stage_done(
                done,
                inputs=files,
                outputs=[out, manifest_path],
                config_sha=getattr(cfg, "config_sha", None),
            )
            return

    total = 0

    def _iter_row_groups():
        """Yield padded row groups from all input parquet shards."""
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
        LOGGER.info(
            "Combine: inputs produced zero rows",
            extra={"stage": "combine", "path": str(cfg.data_dir)},
        )
        return

    def _all_batches():
        """Yield the first batch followed by all remaining batches."""
        yield first
        yield from batches

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
        raise RuntimeError(f"combine: row-count mismatch {pf_out.metadata.num_rows} != {total}")
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
    check_post_combine(files, out)
    write_stage_done(
        done,
        inputs=files,
        outputs=[out, manifest_path],
        config_sha=getattr(cfg, "config_sha", None),
    )
