"""Normalize curated rows into canonical, resumable by-k dataset partitions."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
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
from farkle.utils.authenticated_contract import load_authenticated_sidecar
from farkle.utils.partitioned_stage import (
    PartitionedStageIdentity,
    PartitionedUnit,
    resolved_code_identity_sha256,
    run_partitioned_stage,
    validate_final_manifest,
)
from farkle.utils.release_identity import CapturedV3Inputs, capture_v3_inputs
from farkle.utils.schema_helpers import expected_schema_for
from farkle.utils.source_snapshot import (
    AuthenticatedParquetSnapshot,
    parquet_snapshot_from_captured_inputs,
    raise_classified_resource_failure,
)
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


@dataclass(frozen=True, slots=True)
class _CombineSourceSnapshot:
    """One fully authenticated curated input resolved by the parent."""

    k: int
    parquet: AuthenticatedParquetSnapshot
    captured_inputs: CapturedV3Inputs
    output_sidecar: ArtifactSidecar


def _required_source_paths(cfg: AppConfig) -> tuple[tuple[int, Path], ...]:
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


def _partition_sidecar_template(
    cfg: AppConfig,
    *,
    k: int,
    source: Path,
    target_names: Sequence[str],
) -> ArtifactSidecar:
    return make_artifact_sidecar(
        cfg,
        cfg.combined_rows_by_k(k),
        producer="combine",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation="concatenate_rows_within_k",
        weighted_quantity="canonical_game_rows",
        support_count_role="curated_games",
        replication_unit="game",
        conditioning="unconditional",
        source_artifacts=[source],
        consistency_columns=target_names,
        grouping_keys=["root_seed", "k", "shuffle_index", "game_index"],
        player_counts=[k],
        required_player_counts=[k],
        missing_cell_policy="fail",
        seed_scope="single_root",
    )


def _resolve_sources(cfg: AppConfig) -> tuple[_CombineSourceSnapshot, ...]:
    """Authenticate every curated source once and freeze its worker snapshot."""

    target_names = tuple(expected_schema_for(cfg.combine_max_players).names)
    snapshots: list[_CombineSourceSnapshot] = []
    for k, path in _required_source_paths(cfg):
        template = _partition_sidecar_template(
            cfg,
            k=k,
            source=path,
            target_names=target_names,
        )
        try:
            captured = capture_v3_inputs(template)
            parquet = parquet_snapshot_from_captured_inputs(
                captured,
                expected_path=path,
                expected_schema=expected_schema_for(k),
            )
        except BaseException as exc:
            raise_classified_resource_failure(exc)
            raise
        snapshots.append(
            _CombineSourceSnapshot(
                k,
                parquet,
                captured,
                replace(template, _captured_v3_inputs=captured),
            )
        )
    return tuple(snapshots)


def _units(
    cfg: AppConfig,
    sources: Sequence[_CombineSourceSnapshot],
) -> tuple[PartitionedUnit, ...]:
    stage_root = cfg.combine_stage_dir
    units: list[PartitionedUnit] = []
    for snapshot in sources:
        k = snapshot.k
        relative = cfg.combined_rows_by_k(k).relative_to(stage_root).as_posix()
        units.append(
            PartitionedUnit(
                (k,),
                relative,
                input_identities=snapshot.parquet.input_identities,
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
    sources: tuple[_CombineSourceSnapshot, ...]
    target: pa.Schema
    batch_bytes: int
    batch_rows: int
    compression: Compression

    def __call__(self, unit: PartitionedUnit, output: Path) -> None:
        k = int(unit.key[0])
        snapshot = {item.k: item for item in self.sources}[k]
        source = snapshot.parquet.artifact_path
        source_columns = expected_schema_for(k).names
        with pq.ParquetWriter(output, self.target, compression=self.compression) as writer:
            for _row_group, _batch_index, table in iter_parquet_tables_by_bytes(
                source,
                columns=source_columns,
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
    sources: tuple[_CombineSourceSnapshot, ...]

    def __call__(self, unit: PartitionedUnit, output: Path) -> ArtifactSidecar:
        k = int(unit.key[0])
        snapshot = {item.k: item for item in self.sources}[k]
        if output.name != snapshot.output_sidecar.artifact_name:
            raise ValueError("combine partition output does not match its captured sidecar")
        return snapshot.output_sidecar


@dataclass(frozen=True, slots=True)
class _PartitionValidator:
    sources: tuple[_CombineSourceSnapshot, ...]
    target_schema: pa.Schema

    def __call__(self, unit: PartitionedUnit, output: Path) -> dict[str, Any] | bool:
        k = int(unit.key[0])
        snapshot = {item.k: item for item in self.sources}[k]
        try:
            output_sidecar = validate_artifact_sidecar(
                output,
                expected={
                    "scope": ArtifactScope.BY_K.value,
                    "operation": "concatenate_rows_within_k",
                    "player_counts": [k],
                },
            )
            output_metadata = pq.read_metadata(output)
            output_schema = output_metadata.schema.to_arrow_schema()
            authenticated = load_authenticated_sidecar(output)
        except Exception as exc:  # noqa: BLE001 - invalid evidence is not reusable
            raise_classified_resource_failure(exc)
            return False
        if (
            authenticated.source_artifacts != (snapshot.parquet.source,)
            or output_metadata.num_rows != snapshot.parquet.row_count
            or not output_schema.equals(self.target_schema, check_metadata=False)
        ):
            return False
        target_schema_sha256 = _schema_sha256(self.target_schema)
        return {
            "logical_partition": k,
            "row_count": int(output_metadata.num_rows),
            "schema_sha256": target_schema_sha256,
            "source_artifact_sha256": snapshot.parquet.source.artifact.content_sha256,
            "source_sidecar_sha256": snapshot.parquet.source.sidecar_sha256,
            "output_artifact_sha256": output_sidecar.artifact_sha256,
        }


def _manifest_sidecar(
    cfg: AppConfig,
    manifest: Path,
    sources: Sequence[_CombineSourceSnapshot],
) -> ArtifactSidecar:
    player_counts = [source.k for source in sources]
    template = make_artifact_sidecar(
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
        source_artifacts=[source.parquet.artifact_path for source in sources],
        consistency_columns=expected_schema_for(cfg.combine_max_players).names,
        grouping_keys=["k", "root_seed", "shuffle_index", "game_index"],
        player_counts=player_counts,
        required_player_counts=player_counts,
        missing_cell_policy="fail",
        seed_scope="single_root",
    )
    captured = CapturedV3Inputs(
        sources=tuple(
            sorted(
                (source.parquet.source for source in sources),
                key=lambda item: item.logical_role,
            )
        ),
        manifests=(),
        source_paths=tuple(
            sorted(
                ((source.parquet.source.logical_role, source.parquet.path) for source in sources),
                key=lambda item: item[0],
            )
        ),
        manifest_paths=(),
        controls=(),
    )
    return replace(template, _captured_v3_inputs=captured)


def _validated_layout(
    cfg: AppConfig,
) -> tuple[
    tuple[_CombineSourceSnapshot, ...],
    tuple[PartitionedUnit, ...],
    PartitionedStageIdentity,
]:
    sources = _resolve_sources(cfg)
    units, identity = _validate_resolved_layout(cfg, sources)
    return sources, units, identity


def _validate_resolved_layout(
    cfg: AppConfig,
    sources: tuple[_CombineSourceSnapshot, ...],
) -> tuple[tuple[PartitionedUnit, ...], PartitionedStageIdentity]:
    """Validate published partitions against one parent-resolved source set."""

    units = _units(cfg, sources)
    identity = _identity(cfg, [source.k for source in sources])
    validator = _PartitionValidator(
        sources,
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
    return units, identity


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

    sources = _resolve_sources(cfg)
    player_counts = tuple(source.k for source in sources)
    source_paths = [source.parquet.artifact_path for source in sources]
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
        _validate_resolved_layout(cfg, sources)
        LOGGER.info("Combine: output up-to-date", extra={"stage": "combine", "path": str(manifest)})
        return

    target = expected_schema_for(cfg.combine_max_players)
    result = run_partitioned_stage(
        root=cfg.combine_stage_dir,
        identity=_identity(cfg, player_counts),
        unit_source=lambda: iter(units),
        writer=_PartitionWriter(
            sources,
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
            sources,
        ),
        validator=_PartitionValidator(sources, target),
        manifest_path=manifest,
        manifest_sidecar=_manifest_sidecar(cfg, manifest, sources),
    )
    if result.required_units != len(units):
        raise RuntimeError("combine: final manifest does not cover every configured partition")
    summary = {
        "partitions": len(sources),
        "rows": sum(source.parquet.row_count for source in sources),
    }
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
