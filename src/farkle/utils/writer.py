# src/farkle/utils/writer.py
"""
Atomic Parquet writing helpers. Exposes :func:`atomic_path` for safe file
replacement and :class:`ParquetShardWriter` for streaming Parquet shards with
row tracking.
"""
from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .types import Compression, normalize_compression


@contextmanager
def atomic_path(final_path: str):
    """Write to a temp file in the same directory, then atomic replace."""
    dir_ = os.path.dirname(os.path.abspath(final_path)) or "."
    fd, tmp = tempfile.mkstemp(prefix="._tmp_", dir=dir_)
    os.close(fd)
    try:
        yield tmp
        os.replace(tmp, final_path)  # atomic on same filesystem
    finally:
        with suppress(FileNotFoundError):
            os.remove(tmp)


@dataclass
class ParquetShardWriter:
    """Context manager for streaming parquet writes with atomic finalization."""

    out_path: str
    schema: pa.Schema | None = None
    compression: Compression = "snappy"
    row_group_size: int = 200_000

    _writer: Optional[pq.ParquetWriter] = None
    _tmp_path: str = ""
    _rows_written: int = 0

    def __post_init__(self) -> None:
        self.compression = normalize_compression(self.compression)
        dir_ = os.path.dirname(os.path.abspath(self.out_path)) or "."
        fd, tmp = tempfile.mkstemp(prefix="._tmp_", dir=dir_)
        os.close(fd)
        self._tmp_path = tmp

    def __enter__(self) -> "ParquetShardWriter":
        return self

    @property
    def rows_written(self) -> int:
        """Number of rows written so far."""
        return self._rows_written

    def _ensure_writer(self, tbl: pa.Table) -> None:
        """Create the underlying writer lazily based on the first batch."""
        if self._writer is None:
            assert self._tmp_path, "Temporary path must be set before writing."
            schema = self.schema or tbl.schema
            self._writer = pq.ParquetWriter(
                self._tmp_path,
                schema,
                compression=self.compression,
                use_dictionary=True,
            )
            self.schema = schema

    def write_batch(self, tbl: pa.Table) -> None:
        """Write a single table batch to the parquet file."""
        self._ensure_writer(tbl)
        assert self._writer is not None
        self._writer.write_table(tbl, row_group_size=self.row_group_size)
        self._rows_written += tbl.num_rows

    def write_batches(self, tables: Iterable[pa.Table]) -> None:
        """Write multiple batches sequentially."""
        for tbl in tables:
            self.write_batch(tbl)

    def close(self, success: bool = True) -> None:
        """Close the writer and atomically move the temp file into place."""
        if not self._tmp_path:
            return
        if self._writer is not None:
            self._writer.close()
        if success:
            os.replace(self._tmp_path, self.out_path)
        else:
            with suppress(FileNotFoundError):
                os.remove(self._tmp_path)
        self._tmp_path = ""
        self._writer = None

    def __exit__(self, exc_type, exc, tb):
        """Ensure resources are closed, cleaning up on exception."""
        self.close(exc_type is None)
