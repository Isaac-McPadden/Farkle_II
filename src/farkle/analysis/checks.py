# src/farkle/analysis/checks.py
"""Parquet validation helpers for the analysis pipeline outputs.

These routines assert expected schemas, positive counters, and manifest
consistency before later stages consume combined or metrics datasets.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Mapping

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from farkle.utils.manifest import iter_manifest
from farkle.utils.schema_helpers import expected_schema_for

LOGGER = logging.getLogger(__name__)


_PER_K_DIR_RE = re.compile(r"^\d+p$")


def _resolve_stage_dir(analysis_root: Path, stage_key: str) -> Path | None:
    suffix = f"_{stage_key}"
    matches = [p for p in analysis_root.iterdir() if p.is_dir() and p.name.endswith(suffix)]
    if not matches:
        return None
    return sorted(matches, key=lambda p: p.name)[-1]


def check_stage_artifact_families(
    analysis_root: Path,
    *,
    expected_matrix: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None,
) -> None:
    """Validate stage output families (per-k, pooled concat, pooled aggregate).

    The default contract intentionally focuses on artifact families that should
    remain stable for CI checks:

    - ``game_stats`` must emit per-k game/margin files.
    - ``game_stats`` must emit pooled weighted aggregates.
    """

    matrix = expected_matrix or {
        "game_stats": {
            "per_k": ("game_length.parquet", "margin_stats.parquet"),
            "pooled_concat": (),
            "pooled_aggregate": (
                "game_length_k_weighted.parquet",
                "margin_k_weighted.parquet",
            ),
        }
    }

    violations: list[str] = []
    for stage_key, families in matrix.items():
        stage_dir = _resolve_stage_dir(analysis_root, stage_key)
        if stage_dir is None:
            continue

        stage_subdirs = [p for p in stage_dir.iterdir() if p.is_dir()]
        allowed_dir_names = {"pooled"}
        allowed_dir_names.update(p.name for p in stage_subdirs if _PER_K_DIR_RE.fullmatch(p.name))

        drift_dirs = sorted(
            p.name
            for p in stage_subdirs
            if p.name not in allowed_dir_names
        )
        if drift_dirs:
            violations.append(
                f"{stage_key}: unexpected directory layout entries: {', '.join(drift_dirs)}"
            )

        per_k_expected = tuple(families.get("per_k", ()))
        per_k_dirs = sorted(p for p in stage_subdirs if _PER_K_DIR_RE.fullmatch(p.name))
        if per_k_expected and not per_k_dirs:
            violations.append(f"{stage_key}: missing per-k directories")
        for per_k_dir in per_k_dirs:
            for filename in per_k_expected:
                path = per_k_dir / filename
                if not path.exists():
                    violations.append(f"{stage_key}: missing {path.relative_to(stage_dir)}")

        pooled_dir = stage_dir / "pooled"
        pooled_concat_expected = tuple(families.get("pooled_concat", ()))
        pooled_agg_expected = tuple(families.get("pooled_aggregate", ()))
        pooled_expected = pooled_concat_expected + pooled_agg_expected
        if pooled_expected and not pooled_dir.exists():
            violations.append(f"{stage_key}: missing pooled directory")
        for filename in pooled_expected:
            path = pooled_dir / filename
            if not path.exists():
                violations.append(f"{stage_key}: missing pooled/{filename}")

    if violations:
        raise RuntimeError("artifact family contract violated:\n- " + "\n- ".join(violations))


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
