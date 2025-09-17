from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .analysis_config import expected_schema_for

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

    schema = pq.read_schema(combined_parquet)
    if winner_col not in schema.names:
        raise RuntimeError(
            f"check_pre_metrics: missing '{winner_col}' column in {combined_parquet}"
        )

    dataset = ds.dataset(combined_parquet, format="parquet")
    neg_cols: list[str] = []
    for field in schema:
        if (
            pa.types.is_signed_integer(field.type)
            and field.name != "loss_margin"
            and dataset.count_rows(filter=ds.field(field.name) < 0) > 0
        ):
                neg_cols.append(field.name)
    if neg_cols:
        raise RuntimeError(
            f"check_pre_metrics: negative values present in {', '.join(neg_cols)}"
        )

    data_dir = (
        combined_parquet.parent.parent
        if combined_parquet.parent.name == "all_n_players_combined"
        else combined_parquet.parent
    )
    manifests = sorted(data_dir.glob("*p/manifest_*p.json"))
    if not manifests:
        raise RuntimeError(
            f"check_pre_metrics: no manifest files found under {data_dir}"
        )
    manifest_rows = 0
    for m in manifests:
        try:
            meta = json.loads(m.read_text())
            manifest_rows += int(meta.get("row_count", 0))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"check_pre_metrics: failed to parse {m}: {e}") from e

    combined_rows = dataset.count_rows()
    if combined_rows != manifest_rows:
        raise RuntimeError(
            "check_pre_metrics: row-count mismatch "
            f"{combined_rows} != {manifest_rows}"
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
        raise RuntimeError(
            f"check_post_combine: unable to read {combined_parquet}: {e}"
        ) from e
    combined_rows = combined_pf.metadata.num_rows

    total_rows = 0
    for f in curated_files:
        try:
            total_rows += pq.ParquetFile(f).metadata.num_rows
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"check_post_combine: unable to read {f}: {e}") from e
    if combined_rows != total_rows:
        raise RuntimeError(
            "check_post_combine: row-count mismatch "
            f"{combined_rows} != {total_rows}"
        )

    expected = expected_schema_for(max_players).names
    actual = pq.read_schema(combined_parquet).names
    if actual != expected:
        raise RuntimeError("check_post_combine: output schema mismatch")

    LOGGER.info(
        "check_post_combine passed",
        extra={"stage": "checks", "path": str(combined_parquet)},
    )
