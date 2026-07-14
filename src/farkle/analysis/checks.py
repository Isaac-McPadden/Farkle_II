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
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from farkle.utils.artifact_contract import validate_artifact_sidecar
from farkle.utils.schema_helpers import expected_schema_for

LOGGER = logging.getLogger(__name__)


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

    dataset = ds.dataset(
        combined_parquet,
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
    )
    schema = dataset.schema
    if winner_col not in schema.names:
        raise RuntimeError(
            f"check_pre_metrics: missing '{winner_col}' column in {combined_parquet}"
        )

    neg_cols: list[str] = []
    for field in schema:
        if (
            pa.types.is_signed_integer(field.type)
            and field.name != "loss_margin"
            and dataset.count_rows(filter=ds.field(field.name) < 0) > 0
        ):
            neg_cols.append(field.name)
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

    combined_rows = dataset.count_rows()
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
