# src/farkle/analysis/combine.py
"""Combine curated shards into partitioned and compatibility outputs."""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from farkle.analysis.checks import check_post_combine
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import (
    make_artifact_sidecar,
    sidecar_path,
    write_artifact_with_sidecar_atomic,
)
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


def _iter_parquet_tables(
    paths: Iterable[Path],
    *,
    target: pa.Schema,
    normalize: bool,
) -> Iterator[pa.Table]:
    """Yield non-empty parquet row groups in path order with optional alignment."""

    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(row_group_index)
            if table.num_rows == 0:
                continue
            if normalize and not table.schema.equals(target, check_metadata=False):
                table = _pad_to_schema(table, target)
            yield table


def _assert_row_stream_identity(
    source_tables: Iterable[pa.Table],
    output_path: Path,
    *,
    target: pa.Schema,
    label: str,
) -> int:
    """Prove row-order and value identity without materializing either dataset."""

    source_iter = iter(source_tables)
    output_iter = _iter_parquet_tables([output_path], target=target, normalize=False)
    source_table = next(source_iter, None)
    output_table = next(output_iter, None)
    source_offset = 0
    output_offset = 0
    compared_rows = 0

    while source_table is not None and output_table is not None:
        source_remaining = source_table.num_rows - source_offset
        output_remaining = output_table.num_rows - output_offset
        take = min(source_remaining, output_remaining)
        source_slice = source_table.slice(source_offset, take)
        output_slice = output_table.slice(output_offset, take)
        if not source_slice.equals(output_slice, check_metadata=False):
            raise RuntimeError(f"{label}: concatenation changed source row values or order")
        compared_rows += take
        source_offset += take
        output_offset += take
        if source_offset == source_table.num_rows:
            source_table = next(source_iter, None)
            source_offset = 0
        if output_offset == output_table.num_rows:
            output_table = next(output_iter, None)
            output_offset = 0

    if source_table is not None or output_table is not None:
        raise RuntimeError(f"{label}: concatenation changed the total row count")
    return compared_rows


def _concat_ks_output(cfg: AppConfig) -> Path:
    """Return the canonical row-concatenation output path.

    Args:
        cfg: Application config used to resolve preferred and legacy paths.

    Returns:
        Canonical concatenated parquet path. Legacy locations are ignored.
    """
    return cfg.concat_ks_dir("combine") / "all_ingested_rows.parquet"


def _legacy_concat_candidates(cfg: AppConfig) -> tuple[Path, ...]:
    """Return retired concatenation locations without using them as inputs."""

    filename = "all_ingested_rows.parquet"
    return (
        cfg.combine_stage_dir / f"{cfg.combine_max_players}p" / "combined" / filename,
        cfg.combine_stage_dir / "combined" / filename,
        cfg.combine_stage_dir / filename,
        cfg.analysis_dir / filename,
    )


def _write_migration_report(cfg: AppConfig, canonical_output: Path) -> Path:
    """Inventory legacy concatenations that were deliberately ignored."""

    report_path = cfg.diagnostics_dir("combine") / "migration_report.json"
    ignored = [path for path in _legacy_concat_candidates(cfg) if path.exists()]
    payload = {
        "migration_report_version": 1,
        "stage": "combine",
        "ignored_legacy_artifacts": [
            {
                "path": str(path),
                "replacement": str(canonical_output),
                "reason": "legacy scope is not a valid concat_ks input",
            }
            for path in ignored
        ],
    }
    sidecar = make_artifact_sidecar(
        cfg,
        report_path,
        producer="combine",
        scope=ArtifactScope.DIAGNOSTICS,
        source_scope=ArtifactScope.CONCAT_KS,
        operation="legacy_artifact_inventory",
        source_artifacts=ignored,
        conditioning="not_applicable",
        seed_scope="single_root",
    )

    def _write_data(staged_path: Path) -> None:
        staged_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    write_artifact_with_sidecar_atomic(report_path, sidecar, _write_data)
    return report_path


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
    return (
        partition_dir / f"{prefix}_part-00000.parquet",
        manifest_dir / f"{prefix}_partition.manifest.jsonl",
    )


def _reset_output_manifest(manifest_path: Path) -> None:
    """Clear a rewrite-owned manifest before appending fresh shard metadata."""

    if manifest_path.exists():
        manifest_path.unlink()


def _write_partitioned_dataset(
    cfg: AppConfig, files: list[Path], target: pa.Schema
) -> tuple[list[Path], list[Path]]:
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
            sidecar_artifacts=[out_file],
        ):
            outputs.append(out_file)
            manifests.append(manifest_path)
            continue

        pf = pq.ParquetFile(src)
        if pf.metadata.num_rows == 0:
            if out_file.exists():
                out_file.unlink()
            sidecar_path(out_file).unlink(missing_ok=True)
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
        sidecar = make_artifact_sidecar(
            cfg,
            out_file,
            producer="combine",
            scope=ArtifactScope.BY_K,
            source_scope=ArtifactScope.BY_K,
            operation="combine",
            source_artifacts=[src],
            consistency_columns=target.names,
            player_counts=[n_players],
            required_player_counts=[n_players],
            missing_cell_policy="fail",
        )
        run_streaming_shard(
            out_path=str(out_file),
            manifest_path=str(manifest_path),
            schema=target,
            batch_iter=_iter_row_groups(),
            row_group_size=cfg.row_group_size,
            compression=cfg.parquet_codec,
            manifest_extra={
                "path": str(out_file),
                "n_players": int(n_players),
                "source_file": str(src),
            },
            sidecar=sidecar,
        )
        _assert_row_stream_identity(
            _iter_parquet_tables([src], target=target, normalize=True),
            out_file,
            target=target,
            label=f"combine by_k/{n_players}p",
        )
        outputs.append(out_file)
        manifests.append(manifest_path)
        write_stage_done(
            done,
            inputs=[src],
            outputs=[out_file, manifest_path],
            cfg=cfg,
            stage="combine",
            sidecar_artifacts=[out_file],
        )
    return outputs, manifests


def _write_monolithic_compatibility_from_partitions(
    cfg: AppConfig,
    out: Path,
    manifest_path: Path,
    partition_outputs: list[Path],
) -> int:
    """Write the compatibility parquet by streaming rows from partitioned outputs.

    Args:
        cfg: Application config used to resolve partition directories and schema.
        out: Destination parquet path for the monolithic compatibility output.
        manifest_path: Manifest path paired with ``out``.

    Returns:
        Total number of rows written to the compatibility output.
    """
    dataset = ds.dataset([str(path) for path in partition_outputs], format="parquet")
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
    player_counts = sorted(int(path.name.split("p_", maxsplit=1)[0]) for path in partition_outputs)
    sidecar = make_artifact_sidecar(
        cfg,
        out,
        producer="combine",
        scope=ArtifactScope.CONCAT_KS,
        source_scope=ArtifactScope.BY_K,
        operation="concat",
        source_artifacts=partition_outputs,
        consistency_columns=expected_schema_for(12).names,
        player_counts=player_counts,
        required_player_counts=player_counts,
        missing_cell_policy="not_applicable",
    )
    run_streaming_shard(
        out_path=str(out),
        manifest_path=str(manifest_path),
        schema=expected_schema_for(12),
        batch_iter=_iter_tables(),
        row_group_size=cfg.row_group_size,
        compression=cfg.parquet_codec,
        manifest_extra={"path": out.name, "source": "partitioned_compatibility"},
        sidecar=sidecar,
    )
    verified_rows = _assert_row_stream_identity(
        _iter_parquet_tables(partition_outputs, target=expected_schema_for(12), normalize=False),
        out,
        target=expected_schema_for(12),
        label="combine concat_ks",
    )
    if verified_rows != total:
        raise RuntimeError(
            f"combine concat_ks: verified row count {verified_rows} does not match {total}"
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
    out = _concat_ks_output(cfg)
    migration_report = _write_migration_report(cfg, out)
    files = sorted((cfg.data_dir / "by_k").glob(f"*p/{cfg.curated_rows_name}"))
    if not files:
        LOGGER.info(
            "Combine: no inputs discovered", extra={"stage": "combine", "path": str(cfg.data_dir)}
        )
        return

    target = expected_schema_for(12)
    manifest_path = cfg.combined_manifest_path()
    done = stage_done_path(cfg.combine_stage_dir, "combine")

    if stage_is_up_to_date(
        done,
        inputs=files,
        outputs=[cfg.combine_partitioned_dir, out, manifest_path, migration_report],
        cfg=cfg,
        stage="combine",
        sidecar_artifacts=[out, migration_report],
    ):
        LOGGER.info("Combine: output up-to-date", extra={"stage": "combine", "path": str(out)})
        return

    partition_outputs, partition_manifests = _write_partitioned_dataset(cfg, files, target)
    if not partition_outputs:
        if out.exists():
            out.unlink()
        sidecar_path(out).unlink(missing_ok=True)
        if manifest_path.exists():
            manifest_path.unlink()
        LOGGER.info(
            "Combine: inputs produced zero rows",
            extra={"stage": "combine", "path": str(cfg.data_dir)},
        )
        return

    monolithic_total = _write_monolithic_compatibility_from_partitions(
        cfg, out, manifest_path, partition_outputs
    )
    pf_out = pq.ParquetFile(out)
    if pf_out.metadata.num_rows != monolithic_total:
        raise RuntimeError(
            f"combine: row-count mismatch {pf_out.metadata.num_rows} != {monolithic_total}"
        )
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
        outputs=[
            cfg.combine_partitioned_dir,
            *partition_outputs,
            *partition_manifests,
            out,
            manifest_path,
            migration_report,
        ],
        cfg=cfg,
        stage="combine",
        sidecar_artifacts=[*partition_outputs, out, migration_report],
    )
