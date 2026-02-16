# src/farkle/analysis/checks.py
"""Parquet validation helpers for the analysis pipeline outputs.

These routines assert expected schemas, positive counters, and manifest
consistency before later stages consume combined or metrics datasets.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from farkle.utils.manifest import iter_manifest
from farkle.utils.schema_helpers import expected_schema_for

LOGGER = logging.getLogger(__name__)


_ARTIFACT_FAMILY_MATRIX: dict[str, dict[str, tuple[str, ...]]] = {
    "combine": {
        "pooled_concat": ("all_ingested_rows.parquet",),
    },
    "metrics": {
        "per_k": ("{k}p_isolated_metrics.parquet",),
        "pooled_concat": ("metrics.parquet",),
    },
    "game_stats": {
        "per_k": ("game_length.parquet", "margin_stats.parquet"),
        "pooled_concat": ("game_length.parquet", "margin_stats.parquet"),
        "pooled_weighted": ("game_length_k_weighted.parquet", "margin_k_weighted.parquet"),
    },
}


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
        raise RuntimeError(f"check_pre_metrics: negative values present in {', '.join(neg_cols)}")

    if combined_parquet.parent.name == "all_n_players_combined":
        data_dir = combined_parquet.parent.parent
    elif combined_parquet.parent.name == "pooled":
        analysis_root = combined_parquet.parent.parent.parent
        candidate = next(
            (p for p in analysis_root.iterdir() if p.name.endswith("_curate") and p.is_dir()),
            None,
        )
        data_dir = candidate if candidate is not None else analysis_root
    else:
        data_dir = combined_parquet.parent
    manifest_rows = 0
    seen_manifest = False

    def _rows_from_manifest(manifest_path: Path) -> int:
        try:
            # Try single-JSON first for backward compatibility.
            meta = json.loads(manifest_path.read_text())
            value = meta.get("row_count")
            if value is None:
                value = meta.get("rows")
            if value is None:
                return 0
            return int(value)
        except json.JSONDecodeError:
            pass
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"check_pre_metrics: failed to parse {manifest_path}: {e}") from e

        rows = 0
        try:
            for record in iter_manifest(manifest_path):
                if not isinstance(record, dict):
                    continue
                value = record.get("row_count")
                if value is None:
                    value = record.get("rows")
                if value is None:
                    continue
                rows += int(value)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"check_pre_metrics: failed to parse {manifest_path}: {e}") from e
        return rows

    for seat_dir in sorted(p for p in data_dir.glob("*p") if p.is_dir()):
        manifest_candidates = [
            seat_dir / "manifest.jsonl",
            *seat_dir.glob("manifest_*p.json"),
        ]
        manifest_path = next((m for m in manifest_candidates if m.exists()), None)
        if manifest_path is None:
            continue
        manifest_rows += _rows_from_manifest(manifest_path)
        seen_manifest = True

    if not seen_manifest:
        manifest_path = combined_parquet.with_suffix(".manifest.jsonl")
        if manifest_path.exists():
            manifest_rows = _rows_from_manifest(manifest_path)
            seen_manifest = True
    if not seen_manifest:
        raise RuntimeError(f"check_pre_metrics: no manifest files found under {data_dir}")

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


def check_stage_artifact_families(
    analysis_dir: Path,
    stage_dirs: Mapping[str, Path],
    k_values: Sequence[int],
    matrix: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> None:
    """Validate per-stage artifact families and directory conventions.

    Parameters
    ----------
    analysis_dir:
        Root analysis directory used in error messages.
    stage_dirs:
        Mapping of stage key to resolved stage directory.
    k_values:
        Player-count values expected for per-k artifacts.
    matrix:
        Optional override for the stage artifact-family matrix.
    """

    contracts = matrix or _ARTIFACT_FAMILY_MATRIX
    failures: list[str] = []
    stage_dirs_norm = {key: Path(path) for key, path in stage_dirs.items()}

    for stage, families in contracts.items():
        stage_dir = stage_dirs_norm.get(stage)
        if stage_dir is None or not stage_dir.exists():
            continue

        for pattern in families.get("per_k", ()):
            for k in k_values:
                expected = stage_dir / f"{k}p" / pattern.format(k=k)
                if not expected.exists():
                    failures.append(f"{stage}: missing per-k artifact {expected}")

                drift = stage_dir / pattern.format(k=k)
                if drift.exists():
                    failures.append(
                        f"{stage}: layout drift; expected {expected} but found {drift}"
                    )

        for family in ("pooled_concat", "pooled_weighted"):
            for filename in families.get(family, ()):
                expected = stage_dir / "pooled" / filename
                if not expected.exists():
                    failures.append(f"{stage}: missing {family} artifact {expected}")

                drift = stage_dir / filename
                if drift.exists():
                    failures.append(
                        f"{stage}: layout drift; expected {expected} but found {drift}"
                    )

    if failures:
        details = "\n - ".join(["", *failures])
        raise RuntimeError(
            f"check_stage_artifact_families failed under {analysis_dir}:{details}"
        )

    LOGGER.info(
        "check_stage_artifact_families passed",
        extra={"stage": "checks", "path": str(analysis_dir)},
    )
