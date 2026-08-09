"""Normalize curated rows into canonical, resumable by-k dataset partitions."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from farkle.config import AppConfig, ArtifactScope
from farkle.utils.arrow_batches import iter_parquet_tables_by_bytes
from farkle.utils.artifact_contract import (
    ArtifactSidecar,
    make_artifact_sidecar,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
    write_artifact_with_sidecar_atomic,
)
from farkle.utils.partitioned_stage import (
    PartitionedStageIdentity,
    PartitionedUnit,
    resolved_code_identity_sha256,
    run_partitioned_stage,
    validate_final_manifest,
)
from farkle.utils.schema_helpers import expected_schema_for
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.types import Compression

LOGGER = logging.getLogger(__name__)
_COMBINE_SCHEMA_VERSION = 2
_COMBINE_METHOD_VERSION = 2


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _pad_to_schema(table: pa.Table, target: pa.Schema) -> pa.Table:
    """Pad or cast one bounded table so its schema matches the target exactly."""

    columns = []
    for field in target:
        if field.name in table.column_names:
            columns.append(table[field.name].cast(field.type))
        else:
            columns.append(pa.nulls(len(table), field.type))
    return pa.Table.from_arrays(columns, schema=target)


def _partition_paths(cfg: AppConfig, n_players: int) -> tuple[Path, Path]:
    """Return the canonical partition and shared dataset manifest paths."""

    return cfg.combined_rows_by_k(n_players), cfg.combined_manifest_path()


def _source_identity(path: Path) -> tuple[tuple[str, str], ...]:
    metadata = validate_artifact_sidecar(
        path,
        expected={"scope": ArtifactScope.BY_K.value, "operation": "curate_game_rows"},
    )
    return (
        ("curated_artifact", metadata.artifact_sha256),
        ("curated_sidecar", sha256_file(sidecar_path(path))),
    )


def _required_sources(cfg: AppConfig) -> tuple[tuple[int, Path], ...]:
    player_counts = tuple(sorted({int(k) for k in cfg.sim.n_players_list}))
    if not player_counts:
        raise ValueError("combine: sim.n_players_list must not be empty")
    sources = tuple((k, cfg.ingested_rows_curated(k)) for k in player_counts)
    missing = [path for _k, path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "combine: incomplete canonical curated by-k support: "
            + ", ".join(str(path) for path in missing)
        )
    return sources


def _units(
    cfg: AppConfig,
    sources: Sequence[tuple[int, Path]],
) -> tuple[PartitionedUnit, ...]:
    stage_root = cfg.combine_stage_dir
    units: list[PartitionedUnit] = []
    for k, source in sources:
        relative = cfg.combined_rows_by_k(k).relative_to(stage_root).as_posix()
        units.append(
            PartitionedUnit(
                (k,),
                relative,
                input_identities=_source_identity(source),
            )
        )
    return tuple(units)


def _identity(cfg: AppConfig, player_counts: Sequence[int]) -> PartitionedStageIdentity:
    target = expected_schema_for(cfg.combine_max_players)
    inventory = _canonical_sha256(
        {
            "ordered_player_counts": list(player_counts),
            "target_schema_sha256": _schema_sha256(target),
        }
    )
    return PartitionedStageIdentity(
        stage_name="combine",
        root_seed=int(cfg.sim.seed),
        input_identities=(("partition_inventory", inventory),),
        statistical_config_sha256=cfg.stage_config_sha("combine"),
        code_identity_sha256=resolved_code_identity_sha256(cfg),
        schema_version=_COMBINE_SCHEMA_VERSION,
        method_version=_COMBINE_METHOD_VERSION,
    )


@dataclass(frozen=True, slots=True)
class _PartitionWriter:
    sources: tuple[tuple[int, str], ...]
    target: pa.Schema
    batch_bytes: int
    batch_rows: int
    compression: Compression

    def __call__(self, unit: PartitionedUnit, output: Path) -> None:
        k = int(unit.key[0])
        source_map = dict(self.sources)
        source = Path(source_map[k])
        source_schema = pq.read_schema(source)
        with pq.ParquetWriter(output, self.target, compression=self.compression) as writer:
            for _row_group, _batch_index, table in iter_parquet_tables_by_bytes(
                source,
                columns=source_schema.names,
                max_batch_bytes=self.batch_bytes,
                max_batch_rows=self.batch_rows,
                use_threads=False,
            ):
                normalized = (
                    table
                    if table.schema.equals(self.target, check_metadata=False)
                    else _pad_to_schema(table, self.target)
                )
                writer.write_table(normalized, row_group_size=self.batch_rows)


@dataclass(frozen=True, slots=True)
class _PartitionSidecarFactory:
    cfg: AppConfig
    sources: tuple[tuple[int, str], ...]
    target_names: tuple[str, ...]

    def __call__(self, unit: PartitionedUnit, output: Path) -> ArtifactSidecar:
        k = int(unit.key[0])
        source = Path(dict(self.sources)[k])
        return make_artifact_sidecar(
            self.cfg,
            output,
            producer="combine",
            scope=ArtifactScope.BY_K,
            source_scope=ArtifactScope.BY_K,
            operation="concatenate_rows_within_k",
            weighted_quantity="canonical_game_rows",
            support_count_role="curated_games",
            replication_unit="game",
            conditioning="unconditional",
            source_artifacts=[source],
            consistency_columns=self.target_names,
            grouping_keys=["root_seed", "k", "shuffle_index", "game_index"],
            player_counts=[k],
            required_player_counts=[k],
            missing_cell_policy="fail",
            seed_scope="single_root",
        )


@dataclass(frozen=True, slots=True)
class _PartitionValidator:
    sources: tuple[tuple[int, str], ...]
    target_schema: pa.Schema

    def __call__(self, unit: PartitionedUnit, output: Path) -> dict[str, Any] | bool:
        k = int(unit.key[0])
        source = Path(dict(self.sources)[k])
        try:
            source_sidecar = validate_artifact_sidecar(
                source,
                expected={
                    "scope": ArtifactScope.BY_K.value,
                    "operation": "curate_game_rows",
                },
            )
            output_sidecar = validate_artifact_sidecar(
                output,
                expected={
                    "scope": ArtifactScope.BY_K.value,
                    "operation": "concatenate_rows_within_k",
                    "player_counts": [k],
                },
            )
            source_metadata = pq.read_metadata(source)
            output_metadata = pq.read_metadata(output)
            output_schema = output_metadata.schema.to_arrow_schema()
        except Exception:  # noqa: BLE001 - validation failure makes a unit non-reusable
            return False
        if output_metadata.num_rows != source_metadata.num_rows or not output_schema.equals(
            self.target_schema, check_metadata=False
        ):
            return False
        target_schema_sha256 = _schema_sha256(self.target_schema)
        return {
            "logical_partition": k,
            "row_count": int(output_metadata.num_rows),
            "schema_sha256": target_schema_sha256,
            "source_artifact_sha256": source_sidecar.artifact_sha256,
            "source_sidecar_sha256": sha256_file(sidecar_path(source)),
            "output_artifact_sha256": output_sidecar.artifact_sha256,
        }


def _manifest_sidecar(
    cfg: AppConfig,
    manifest: Path,
    sources: Sequence[tuple[int, Path]],
) -> ArtifactSidecar:
    player_counts = [k for k, _path in sources]
    return make_artifact_sidecar(
        cfg,
        manifest,
        producer="combine",
        scope=ArtifactScope.CONCAT_KS,
        source_scope=ArtifactScope.BY_K,
        operation="concatenate",
        weighted_quantity="canonical_game_rows",
        support_count_role="partition_row_counts",
        replication_unit="game",
        conditioning="unconditional",
        source_artifacts=[path for _k, path in sources],
        consistency_columns=expected_schema_for(cfg.combine_max_players).names,
        grouping_keys=["k", "root_seed", "shuffle_index", "game_index"],
        player_counts=player_counts,
        required_player_counts=player_counts,
        missing_cell_policy="fail",
        seed_scope="single_root",
    )


def _validated_layout(
    cfg: AppConfig,
) -> tuple[tuple[tuple[int, Path], ...], tuple[PartitionedUnit, ...], PartitionedStageIdentity]:
    sources = _required_sources(cfg)
    units = _units(cfg, sources)
    identity = _identity(cfg, [k for k, _path in sources])
    validator = _PartitionValidator(
        tuple((k, str(path)) for k, path in sources),
        expected_schema_for(cfg.combine_max_players),
    )
    validated = validate_final_manifest(
        cfg.combined_manifest_path(),
        root=cfg.combine_stage_dir,
        identity=identity,
        unit_source=lambda: iter(units),
        output_prefix=".",
        validator=validator,
        require_sidecar=True,
    )
    if validated is None:
        raise RuntimeError("combine: dataset manifest or a required partition is invalid")
    return sources, units, identity


def combined_partition_paths(cfg: AppConfig) -> tuple[Path, ...]:
    """Return validated canonical partitions in deterministic k order."""

    _sources, units, _identity_value = _validated_layout(cfg)
    return tuple(cfg.combine_stage_dir / unit.relative_output for unit in units)


def concat_ks_dataset(cfg: AppConfig) -> ds.Dataset:
    """Return the logical ``concat_ks`` table without creating a monolithic file."""

    return ds.dataset([str(path) for path in combined_partition_paths(cfg)], format="parquet")


def scan_concat_ks(
    cfg: AppConfig,
    *,
    columns: Sequence[str] | None = None,
    max_batch_bytes: int | None = None,
    max_batch_rows: int | None = None,
) -> Iterator[pa.RecordBatch]:
    """Scan partitions and rows deterministically with explicit byte and row ceilings."""

    batch_bytes = int(
        max_batch_bytes
        if max_batch_bytes is not None
        else cfg.resources.stage_batch_bytes["combine"]
    )
    batch_rows = int(max_batch_rows if max_batch_rows is not None else cfg.row_group_size)
    for path in combined_partition_paths(cfg):
        selected = pq.read_schema(path).names if columns is None else list(columns)
        for _row_group, _batch_index, table in iter_parquet_tables_by_bytes(
            path,
            columns=selected,
            max_batch_bytes=batch_bytes,
            max_batch_rows=batch_rows,
            use_threads=False,
        ):
            yield from table.to_batches()


def verify_concat_ks(cfg: AppConfig, *, deep: bool = False) -> dict[str, Any]:
    """Validate routine manifest evidence, optionally rereading all logical rows."""

    paths = combined_partition_paths(cfg)
    rows = sum(int(pq.read_metadata(path).num_rows) for path in paths)
    scanned_rows = 0
    if deep:
        for batch in scan_concat_ks(cfg):
            scanned_rows += int(batch.num_rows)
        if scanned_rows != rows:
            raise RuntimeError(f"combine: deep row-count mismatch {scanned_rows} != {rows}")
        for path in paths:
            metadata = validate_artifact_sidecar(path)
            if sha256_file(path) != metadata.artifact_sha256:
                raise RuntimeError(f"combine: deep byte hash mismatch for {path}")
    return {
        "partitions": len(paths),
        "rows": rows,
        "deep_verified": deep,
        "deep_scanned_rows": scanned_rows,
    }


def materialize_concat_ks(cfg: AppConfig, destination: Path) -> Path:
    """Materialize a non-canonical compatibility Parquet for release packaging."""

    destination = Path(destination)
    canonical = cfg.curated_parquet
    if destination.resolve() == canonical.resolve():
        raise ValueError("compatibility materialization cannot occupy the retired canonical path")
    try:
        destination.resolve().relative_to(cfg.concat_ks_dir("combine").resolve())
    except ValueError as exc:
        raise ValueError(
            "authenticated compatibility materialization must remain inside combine/concat_ks"
        ) from exc
    verify_concat_ks(cfg, deep=True)
    manifest = cfg.combined_manifest_path()
    sidecar = make_artifact_sidecar(
        cfg,
        destination,
        producer="combine",
        scope=ArtifactScope.CONCAT_KS,
        source_scope=ArtifactScope.CONCAT_KS,
        operation="materialize_concat_ks_compatibility",
        weighted_quantity="canonical_game_rows",
        support_count_role="partition_row_counts",
        replication_unit="game",
        conditioning="unconditional",
        source_artifacts=[manifest],
        consistency_columns=expected_schema_for(cfg.combine_max_players).names,
        player_counts=sorted({int(k) for k in cfg.sim.n_players_list}),
        required_player_counts=sorted({int(k) for k in cfg.sim.n_players_list}),
        missing_cell_policy="fail",
        seed_scope="single_root",
    )

    def _write(staged: Path) -> None:
        schema = expected_schema_for(cfg.combine_max_players)
        with pq.ParquetWriter(staged, schema, compression=cfg.parquet_codec) as writer:
            for batch in scan_concat_ks(cfg):
                writer.write_batch(batch, row_group_size=cfg.row_group_size)

    write_artifact_with_sidecar_atomic(destination, sidecar, _write)
    validate_artifact_sidecar(
        destination,
        expected={"operation": "materialize_concat_ks_compatibility"},
    )
    return destination


def _retire_legacy_outputs(cfg: AppConfig, player_counts: Sequence[int]) -> None:
    legacy = [cfg.curated_parquet, sidecar_path(cfg.curated_parquet)]
    for k in player_counts:
        by_k = cfg.by_k_dir("combine", k)
        legacy.extend(
            [
                by_k / f"{k}p_partition.manifest.jsonl",
                sidecar_path(by_k / f"{k}p_partition.manifest.jsonl"),
                cfg.combine_stage_dir / f"combine_partition_{k}p.done.json",
            ]
        )
    for path in legacy:
        path.unlink(missing_ok=True)


def _remove_orphan_partitions(cfg: AppConfig, player_counts: Sequence[int]) -> None:
    required = {int(k) for k in player_counts}
    root = cfg.combine_partitioned_dir
    if not root.is_dir():
        return
    for partition in sorted(root.glob("*p/*p_part-*.parquet"), key=lambda path: path.as_posix()):
        label = partition.parent.name.removesuffix("p")
        if not label.isdigit() or int(label) in required:
            continue
        for path in (
            partition,
            sidecar_path(partition),
            partition.with_name(f"{partition.name}.unit.done.json"),
        ):
            path.unlink(missing_ok=True)
        with suppress(OSError):
            partition.parent.rmdir()


def run(cfg: AppConfig, *, force: bool = False) -> None:
    """Publish or precisely resume the canonical partitioned ``concat_ks`` dataset."""

    sources = _required_sources(cfg)
    player_counts = tuple(k for k, _path in sources)
    source_paths = [path for _k, path in sources]
    units = _units(cfg, sources)
    outputs = [cfg.combine_stage_dir / unit.relative_output for unit in units]
    manifest = cfg.combined_manifest_path()
    done = stage_done_path(cfg.combine_stage_dir, "combine")
    if not force and stage_is_up_to_date(
        done,
        inputs=source_paths,
        outputs=[*outputs, manifest],
        cfg=cfg,
        stage="combine",
        sidecar_artifacts=[*outputs, manifest],
    ):
        _validated_layout(cfg)
        LOGGER.info("Combine: output up-to-date", extra={"stage": "combine", "path": str(manifest)})
        return

    target = expected_schema_for(cfg.combine_max_players)
    source_strings = tuple((k, str(path)) for k, path in sources)
    result = run_partitioned_stage(
        root=cfg.combine_stage_dir,
        identity=_identity(cfg, player_counts),
        unit_source=lambda: iter(units),
        writer=_PartitionWriter(
            source_strings,
            target,
            int(cfg.resources.stage_batch_bytes["combine"]),
            int(cfg.row_group_size),
            cfg.parquet_codec,
        ),
        resources=cfg.resources,
        requested_workers=cfg.analysis.n_jobs,
        mp_start_method=cfg.analysis.mp_start_method,
        force=force,
        output_prefix=".",
        sidecar_factory=_PartitionSidecarFactory(
            cfg,
            source_strings,
            tuple(target.names),
        ),
        validator=_PartitionValidator(source_strings, target),
        manifest_path=manifest,
        manifest_sidecar=_manifest_sidecar(cfg, manifest, sources),
    )
    if result.required_units != len(units):
        raise RuntimeError("combine: final manifest does not cover every configured partition")
    summary = verify_concat_ks(cfg)
    _retire_legacy_outputs(cfg, player_counts)
    _remove_orphan_partitions(cfg, player_counts)
    write_stage_done(
        done,
        inputs=source_paths,
        outputs=[*outputs, manifest],
        cfg=cfg,
        stage="combine",
        sidecar_artifacts=[*outputs, manifest],
    )
    LOGGER.info(
        "Combine: partitioned dataset written",
        extra={
            "stage": "combine",
            "path": str(manifest),
            "rows": summary["rows"],
            "partitions": summary["partitions"],
            "reused_partitions": result.reused_units,
            "written_partitions": result.completed_units,
            "peak_sampled_rss_mb": result.peak_sampled_rss_mb,
            "workers": result.policy.process_workers,
        },
    )


__all__ = [
    "combined_partition_paths",
    "concat_ks_dataset",
    "materialize_concat_ks",
    "run",
    "scan_concat_ks",
    "verify_concat_ks",
]
