# src/farkle/analysis/combine.py
"""Combine curated shards into partitioned and compatibility outputs."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from farkle.analysis.checks import check_post_combine
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig
from farkle.utils.schema_helpers import expected_schema_for
from farkle.utils.streaming_loop import run_streaming_shard

LOGGER = logging.getLogger(__name__)


def _pad_to_schema(tbl: pa.Table, target: pa.Schema) -> pa.Table:
    """Pad or cast a table so it matches the target schema exactly.

    Args:
        tbl: Source table read from a curated shard.
        target: Target schema for the combined output.

    Returns:
        Table whose columns and types match ``target``.
    """
    cols = []
    for f in target:
        if f.name in tbl.column_names:
            cols.append(tbl[f.name].cast(f.type))
        else:
            cols.append(pa.nulls(len(tbl), f.type))
    return pa.table(cols, names=target.names)


def _migrate_combined_output(cfg: AppConfig) -> Path:
    """Move legacy combined outputs into the current pooled location when needed.

    Args:
        cfg: Application config used to resolve preferred and legacy paths.

    Returns:
        Preferred combined parquet path after any legacy migration.
    """
    preferred_dir = cfg.combine_pooled_dir()
    preferred_out = preferred_dir / "all_ingested_rows.parquet"
    legacy_candidates = [
        cfg.combine_stage_dir / f"{cfg.combine_max_players}p" / "pooled" / "all_ingested_rows.parquet",
        cfg.combine_stage_dir / "all_n_players_combined" / "all_ingested_rows.parquet",
        cfg.analysis_dir / "all_n_players_combined" / "all_ingested_rows.parquet",
        cfg.analysis_dir / "data" / "all_n_players_combined" / "all_ingested_rows.parquet",
    ]
    for legacy in legacy_candidates:
        if preferred_out.exists() or not legacy.exists() or legacy == preferred_out:
            continue
        preferred_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), preferred_out)
        legacy_manifest = legacy.with_suffix(".manifest.jsonl")
        new_manifest = preferred_out.with_suffix(".manifest.jsonl")
        if legacy_manifest.exists():
            shutil.move(str(legacy_manifest), new_manifest)
    return preferred_out


def _partition_done_path(cfg: AppConfig, n_players: int) -> Path:
    return cfg.combine_stage_dir / f"combine_partition_{int(n_players)}p.done.json"


def _partition_paths(cfg: AppConfig, n_players: int) -> tuple[Path, Path]:
    """Resolve the partition parquet and manifest paths for one player count.

    Args:
        cfg: Application config used to resolve partition directories.
        n_players: Player count represented by the partition.

    Returns:
        Tuple of ``(partition_parquet_path, partition_manifest_path)``.
    """
    prefix = f"{int(n_players)}p"
    partition_dir = cfg.combine_partitioned_dir
    manifest_dir = cfg.combine_stage_dir / "partition_manifests"
    return partition_dir / f"{prefix}_part-00000.parquet", manifest_dir / f"{prefix}_partition.manifest.jsonl"


def _reset_output_manifest(manifest_path: Path) -> None:
    """Clear a rewrite-owned manifest before appending fresh shard metadata."""

    if manifest_path.exists():
        manifest_path.unlink()


def _write_partitioned_dataset(cfg: AppConfig, files: list[Path], target: pa.Schema) -> tuple[list[Path], list[Path]]:
    """Write per-player-count partitioned outputs from curated shard parquet files.

    Args:
        cfg: Application config used to resolve output directories and stamps.
        files: Curated shard parquet files grouped by player count.
        target: Target schema for normalized partition outputs.

    Returns:
        Tuple of written partition parquet paths and their manifest paths.
    """
    outputs: list[Path] = []
    manifests: list[Path] = []
    cfg.combine_partitioned_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        try:
            n_players = int(src.parent.name.removesuffix("p"))
        except ValueError:
            continue
        out_file, manifest_path = _partition_paths(cfg, n_players)
        done = _partition_done_path(cfg, n_players)
        if stage_is_up_to_date(
            done,
            inputs=[src],
            outputs=[out_file, manifest_path],
            cfg=cfg,
            stage="combine",
        ):
            outputs.append(out_file)
            manifests.append(manifest_path)
            continue

        pf = pq.ParquetFile(src)
        if pf.metadata.num_rows == 0:
            if out_file.exists():
                out_file.unlink()
            if manifest_path.exists():
                manifest_path.unlink()
            continue

        def _iter_row_groups(pf_obj: pq.ParquetFile = pf):
            """Yield normalized row groups from one parquet file.

            Args:
                pf_obj: Open parquet file whose row groups should be streamed.

            Yields:
                Row-group tables matching the target schema.
            """
            for idx in range(pf_obj.num_row_groups):
                table = pf_obj.read_row_group(idx)
                if table.num_rows == 0:
                    continue
                if table.schema.names != target.names:
                    table = _pad_to_schema(table, target)
                yield table

        _reset_output_manifest(manifest_path)
        run_streaming_shard(
            out_path=str(out_file),
            manifest_path=str(manifest_path),
            schema=target,
            batch_iter=_iter_row_groups(),
            row_group_size=cfg.row_group_size,
            compression=cfg.parquet_codec,
            manifest_extra={"path": str(out_file), "n_players": int(n_players), "source_file": str(src)},
        )
        outputs.append(out_file)
        manifests.append(manifest_path)
        write_stage_done(
            done,
            inputs=[src],
            outputs=[out_file, manifest_path],
            cfg=cfg,
            stage="combine",
        )
    return outputs, manifests


def _write_monolithic_compatibility_from_partitions(cfg: AppConfig, out: Path, manifest_path: Path) -> int:
    """Write the compatibility parquet by streaming rows from partitioned outputs.

    Args:
        cfg: Application config used to resolve partition directories and schema.
        out: Destination parquet path for the monolithic compatibility output.
        manifest_path: Manifest path paired with ``out``.

    Returns:
        Total number of rows written to the compatibility output.
    """
    dataset = ds.dataset(cfg.combine_partitioned_dir, format="parquet", partitioning="hive")
    scanner = dataset.scanner(batch_size=cfg.row_group_size, use_threads=True)
    total = 0

    def _iter_tables():
        """Yield partition batches as tables, dropping partition columns.

        Yields:
            Compatibility tables ready for monolithic streaming output.
        """
        nonlocal total
        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue
            total += int(batch.num_rows)
            table = pa.Table.from_batches([batch])
            if "n_players" in table.column_names:
                table = table.drop(["n_players"])
            yield table

    _reset_output_manifest(manifest_path)
    run_streaming_shard(
        out_path=str(out),
        manifest_path=str(manifest_path),
        schema=expected_schema_for(12),
        batch_iter=_iter_tables(),
        row_group_size=cfg.row_group_size,
        compression=cfg.parquet_codec,
        manifest_extra={"path": out.name, "source": "partitioned_compatibility"},
    )
    return total


def run(cfg: AppConfig) -> None:
    """Combine curated per-player shards into partitioned and compatibility outputs.

    Args:
        cfg: Loaded application config providing input directories and output paths.

    Returns:
        ``None``. The function writes partitioned parquet artifacts, a monolithic
        compatibility parquet file, and corresponding stage metadata.
    """
    preferred = sorted(cfg.data_dir.glob(f"*p/{cfg.curated_rows_name}"))
    legacy = sorted(cfg.data_dir.glob("*p/*_ingested_rows.parquet"))
    files: list[Path] = preferred or legacy
    if not files:
        LOGGER.info("Combine: no inputs discovered", extra={"stage": "combine", "path": str(cfg.data_dir)})
        return

    target = expected_schema_for(12)
    out = _migrate_combined_output(cfg)
    manifest_path = cfg.combined_manifest_path()
    done = stage_done_path(cfg.combine_stage_dir, "combine")

    if stage_is_up_to_date(
        done,
        inputs=files,
        outputs=[cfg.combine_partitioned_dir, out, manifest_path],
        cfg=cfg,
        stage="combine",
    ):
        LOGGER.info("Combine: output up-to-date", extra={"stage": "combine", "path": str(out)})
        return

    partition_outputs, partition_manifests = _write_partitioned_dataset(cfg, files, target)
    if not partition_outputs:
        if out.exists():
            out.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
        LOGGER.info("Combine: inputs produced zero rows", extra={"stage": "combine", "path": str(cfg.data_dir)})
        return

    monolithic_total = _write_monolithic_compatibility_from_partitions(cfg, out, manifest_path)
    pf_out = pq.ParquetFile(out)
    if pf_out.metadata.num_rows != monolithic_total:
        raise RuntimeError(f"combine: row-count mismatch {pf_out.metadata.num_rows} != {monolithic_total}")
    if pq.read_schema(out).names != expected_schema_for(12).names:
        raise RuntimeError("combine: output schema mismatch")

    LOGGER.info(
        "Combine: parquet written",
        extra={
            "stage": "combine",
            "path": str(out),
            "rows": monolithic_total,
            "manifest": str(manifest_path),
            "partitioned_root": str(cfg.combine_partitioned_dir),
            "partitions": len(partition_outputs),
        },
    )
    check_post_combine(files, out)
    write_stage_done(
        done,
        inputs=files,
        outputs=[cfg.combine_partitioned_dir, *partition_outputs, *partition_manifests, out, manifest_path],
        cfg=cfg,
        stage="combine",
    )
