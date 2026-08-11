"""Immutable parent-resolved source identities for partition workers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.utils.authenticated_contract import ArrowSchemaIdentity, SourceArtifactIdentity
from farkle.utils.parallel import ResourceFailureError, classify_resource_exception
from farkle.utils.release_identity import CapturedV3Inputs


@dataclass(frozen=True, slots=True)
class AuthenticatedParquetSnapshot:
    """Exact identity and bounded structural metadata for one Parquet source."""

    path: str
    source: SourceArtifactIdentity
    row_count: int

    @property
    def artifact_path(self) -> Path:
        return Path(self.path)

    @property
    def schema_identity(self) -> ArrowSchemaIdentity:
        schema = self.source.artifact.arrow_schema
        if schema is None:
            raise ValueError(f"authenticated source is not Parquet: {self.path}")
        return schema

    @property
    def input_identities(self) -> tuple[tuple[str, str], ...]:
        return (
            ("source_artifact", self.source.artifact.content_sha256),
            ("source_sidecar", self.source.sidecar_sha256),
        )


def raise_classified_resource_failure(exc: BaseException) -> None:
    """Preserve allocator failures as execution failures, not data corruption."""

    classification = classify_resource_exception(exc)
    if classification is not None:
        if isinstance(exc, ResourceFailureError):
            raise exc
        raise ResourceFailureError(classification, str(exc)) from exc


def parquet_snapshot_from_captured_inputs(
    captured: CapturedV3Inputs,
    *,
    expected_path: Path,
    expected_schema: pa.Schema,
) -> AuthenticatedParquetSnapshot:
    """Build a source snapshot after one full parent-owned v3 authentication."""

    if len(captured.sources) != 1 or captured.manifests or captured.controls:
        raise ValueError("Parquet source snapshots require exactly one ordinary source")
    source = captured.sources[0]
    paths = dict(captured.source_paths)
    path = Path(paths.get(source.logical_role, ""))
    if path != expected_path:
        raise ValueError("captured source path does not match the canonical input")
    schema = source.artifact.arrow_schema
    if schema is None:
        raise ValueError(f"authenticated source is not Parquet: {path}")
    try:
        metadata = pq.read_metadata(path)
        actual_schema = metadata.schema.to_arrow_schema()
    except BaseException as exc:
        raise_classified_resource_failure(exc)
        raise
    from farkle.utils.authenticated_contract import arrow_schema_identity

    if (
        arrow_schema_identity(actual_schema, schema_version=schema.schema_version) != schema
        or actual_schema.names != expected_schema.names
        or any(
            actual.type != expected.type
            for actual, expected in zip(actual_schema, expected_schema, strict=True)
        )
    ):
        raise ValueError(f"authenticated source schema is not canonical: {path}")
    row_count = int(metadata.num_rows)
    return AuthenticatedParquetSnapshot(str(path), source, row_count)


__all__ = [
    "AuthenticatedParquetSnapshot",
    "parquet_snapshot_from_captured_inputs",
    "raise_classified_resource_failure",
]
