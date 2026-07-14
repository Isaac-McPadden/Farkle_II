# src/farkle/utils/artifacts.py
"""
Atomic artifact helpers for writing Parquet and CSV outputs. Uses temporary
files via :func:`atomic_path` to ensure durable writes before replacing the
target path.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .artifact_contract import (
    ArtifactSidecar,
    validate_artifact_sidecar,
    write_artifact_with_sidecar_atomic,
)
from .types import Compression, normalize_compression
from .writer import atomic_path


def write_parquet_atomic(
    table: pa.Table, path: Union[Path, str], *, codec: Compression = "snappy"
) -> None:
    """Write *table* to *path* atomically using Parquet compression."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(final_path)) as tmp_path:
        pq.write_table(table, tmp_path, compression=normalize_compression(codec))


def write_csv_atomic(df: pd.DataFrame, path: Union[Path, str]) -> None:
    """Write *df* to *path* atomically as UTF-8 CSV without index."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        atomic_path(str(final_path)) as tmp_path,
        Path(tmp_path).open("w", encoding="utf-8", newline="") as handle,
    ):
        df.to_csv(handle, index=False)


def write_parquet_artifact_atomic(
    table: pa.Table,
    path: Union[Path, str],
    *,
    sidecar: ArtifactSidecar,
    codec: Compression = "snappy",
) -> ArtifactSidecar:
    """Write a Parquet artifact and its hash-bound sidecar as one publication."""

    compression = normalize_compression(codec)

    def _write(staged_path: Path) -> None:
        pq.write_table(table, staged_path, compression=compression)

    return write_artifact_with_sidecar_atomic(Path(path), sidecar, _write)


def write_csv_artifact_atomic(
    df: pd.DataFrame,
    path: Union[Path, str],
    *,
    sidecar: ArtifactSidecar,
) -> ArtifactSidecar:
    """Write a CSV artifact and its hash-bound sidecar as one publication."""

    def _write(staged_path: Path) -> None:
        with staged_path.open("w", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=False)

    return write_artifact_with_sidecar_atomic(Path(path), sidecar, _write)


def write_json_artifact_atomic(
    payload: Any,
    path: Union[Path, str],
    *,
    sidecar: ArtifactSidecar,
) -> ArtifactSidecar:
    """Write canonical JSON and its hash-bound sidecar as one publication."""

    def _write(staged_path: Path) -> None:
        staged_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return write_artifact_with_sidecar_atomic(Path(path), sidecar, _write)


def read_parquet_artifact(
    path: Union[Path, str], *, expected_sidecar: Mapping[str, Any] | None = None
) -> pa.Table:
    """Validate a Parquet artifact's sidecar before reading its table."""

    validate_artifact_sidecar(path, expected=expected_sidecar)
    return pq.read_table(path)


def read_csv_artifact(
    path: Union[Path, str], *, expected_sidecar: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Validate a CSV artifact's sidecar before reading its frame."""

    validate_artifact_sidecar(path, expected=expected_sidecar)
    return pd.read_csv(path)


def read_json_artifact(
    path: Union[Path, str], *, expected_sidecar: Mapping[str, Any] | None = None
) -> Any:
    """Validate a JSON artifact's sidecar before parsing its payload."""

    validate_artifact_sidecar(path, expected=expected_sidecar)
    return json.loads(Path(path).read_text(encoding="utf-8"))
