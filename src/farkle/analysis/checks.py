# src/farkle/analysis/checks.py
"""Parquet validation helpers for the analysis pipeline outputs.

These routines assert expected schemas, positive counters, and manifest
consistency before later stages consume combined or metrics datasets.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from farkle.utils.artifact_contract import validate_artifact_sidecar
from farkle.utils.schema_helpers import expected_schema_for

LOGGER = logging.getLogger(__name__)


def _scan_negative_columns(path: Path, columns: list[str]) -> set[str]:
    """Return negative columns from one streaming scan of the requested fields."""

    if not columns:
        return set()
    dataset = ds.dataset(
        path,
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
    )
    unresolved = set(columns)
    negative: set[str] = set()
    for batch in dataset.scanner(columns=columns).to_batches():
        for name in tuple(unresolved):
            column = batch.column(batch.schema.get_field_index(name))
            zero = pa.scalar(0, type=column.type)
            has_negative = pc.any(pc.less(column, zero)).as_py()
            if has_negative is True:
                negative.add(name)
                unresolved.remove(name)
        if not unresolved:
            break
    return negative


def _negative_columns_from_metadata(
    path: Path,
    parquet: pq.ParquetFile,
    columns: list[str],
) -> set[str]:
    """Use row-group statistics, scanning all fallback columns at most once."""

    physical_columns = {
        parquet.schema.column(index).path: index for index in range(len(parquet.schema))
    }
    negative: set[str] = set()
    fallback: list[str] = []
    metadata = parquet.metadata
    for name in columns:
        column_index = physical_columns.get(name)
        if column_index is None:
            fallback.append(name)
            continue
        for row_group_index in range(parquet.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            statistics = row_group.column(column_index).statistics
            if statistics is not None and statistics.has_min_max:
                if statistics.min is not None and int(statistics.min) < 0:
                    negative.add(name)
                    break
                continue
            if (
                statistics is not None
                and statistics.null_count is not None
                and int(statistics.null_count) == row_group.num_rows
            ):
                continue
            fallback.append(name)
            break
    if fallback:
        negative.update(_scan_negative_columns(path, fallback))
    return negative


def check_pre_metrics(combined_parquet: Path, winner_col: str = "winner") -> None:
    """Assert winner column exists, no negative counters, row counts match manifests.

    Parameters
    ----------
    combined_parquet:
        Path to the combined parquet produced by :mod:`combine`.
    winner_col:
        Name of the column holding the winner label.
    """
    if not combined_parquet.exists():
        raise RuntimeError(f"check_pre_metrics: missing {combined_parquet}")
    if combined_parquet.parent.name != "concat_ks":
        raise RuntimeError(
            "check_pre_metrics: scope mismatch; expected a concat_ks artifact, "
            f"got {combined_parquet}"
        )

    if combined_parquet.is_dir():
        raise RuntimeError(
            "check_pre_metrics: expected the canonical concat_ks parquet, not a directory"
        )
    validate_artifact_sidecar(
        combined_parquet,
        expected={"scope": "concat_ks", "operation": "concatenate"},
    )

    parquet = pq.ParquetFile(combined_parquet)
    schema = parquet.schema_arrow
    if winner_col not in schema.names:
        raise RuntimeError(
            f"check_pre_metrics: missing '{winner_col}' column in {combined_parquet}"
        )

    checked_columns = [
        field.name
        for field in schema
        if pa.types.is_signed_integer(field.type) and field.name != "loss_margin"
    ]
    negative = _negative_columns_from_metadata(combined_parquet, parquet, checked_columns)
    neg_cols = [name for name in checked_columns if name in negative]
    if neg_cols:
        raise RuntimeError(f"check_pre_metrics: negative values present in {', '.join(neg_cols)}")

    manifest_path = combined_parquet.with_suffix(".manifest.jsonl")

    def _rows_from_manifest(path: Path) -> int:
        """Read the one row-count record from a canonical NDJSON manifest.

        Args:
            path: Manifest path to inspect.

        Returns:
            Total row count reported by the manifest.
        """
        try:
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(records) != 1 or not isinstance(records[0], dict):
                raise ValueError("expected exactly one manifest record")
            return int(records[0]["rows"])
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"check_pre_metrics: failed to parse {path}: {e}") from e

    if not manifest_path.exists():
        raise RuntimeError(f"check_pre_metrics: missing canonical manifest {manifest_path}")
    manifest_rows = _rows_from_manifest(manifest_path)

    combined_rows = parquet.metadata.num_rows
    if combined_rows != manifest_rows:
        raise RuntimeError(
            "check_pre_metrics: row-count mismatch " f"{combined_rows} != {manifest_rows}"
        )

    LOGGER.info(
        "check_pre_metrics passed",
        extra={"stage": "checks", "path": str(combined_parquet)},
    )


def check_post_combine(
    curated_files: list[Path],
    combined_parquet: Path,
    max_players: int = 12,
) -> None:
    """Assert sum(rows per N) == combined rows; schema has P1..P12 templates."""
    if not combined_parquet.exists():
        raise RuntimeError(f"check_post_combine: missing {combined_parquet}")

    try:
        combined_pf = pq.ParquetFile(combined_parquet)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"check_post_combine: unable to read {combined_parquet}: {e}") from e
    combined_rows = combined_pf.metadata.num_rows

    total_rows = 0
    for f in curated_files:
        try:
            total_rows += pq.ParquetFile(f).metadata.num_rows
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"check_post_combine: unable to read {f}: {e}") from e
    if combined_rows != total_rows:
        raise RuntimeError(
            "check_post_combine: row-count mismatch " f"{combined_rows} != {total_rows}"
        )

    expected = expected_schema_for(max_players).names
    actual = pq.read_schema(combined_parquet).names
    if actual != expected:
        raise RuntimeError("check_post_combine: output schema mismatch")

    LOGGER.info(
        "check_post_combine passed",
        extra={"stage": "checks", "path": str(combined_parquet)},
    )
