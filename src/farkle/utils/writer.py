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
    out_path: str
    schema: pa.Schema | None = None
    compression: str = "snappy"
    row_group_size: int = 200_000

    _writer: Optional[pq.ParquetWriter] = None
    _tmp_path: Optional[str] = None
    _rows_written: int = 0

    def __post_init__(self) -> None:
        dir_ = os.path.dirname(os.path.abspath(self.out_path)) or "."
        fd, tmp = tempfile.mkstemp(prefix="._tmp_", dir=dir_)
        os.close(fd)
        self._tmp_path = tmp

    def __enter__(self) -> "ParquetShardWriter":
        return self

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def _ensure_writer(self, tbl: pa.Table) -> None:
        if self._writer is None:
            schema = self.schema or tbl.schema
            self._writer = pq.ParquetWriter(
                self._tmp_path,
                schema,
                compression=self.compression,
                use_dictionary=True,
            )
            self.schema = schema

    def write_batch(self, tbl: pa.Table) -> None:
        self._ensure_writer(tbl)
        assert self._writer is not None
        self._writer.write_table(tbl, row_group_size=self.row_group_size)
        self._rows_written += tbl.num_rows

    def write_batches(self, tables: Iterable[pa.Table]) -> None:
        for tbl in tables:
            self.write_batch(tbl)

    def close(self, success: bool = True) -> None:
        if not self._tmp_path:
            return
        if self._writer is not None:
            self._writer.close()
        if success:
            os.replace(self._tmp_path, self.out_path)
        else:
            with suppress(FileNotFoundError):
                os.remove(self._tmp_path)
        self._tmp_path = None
        self._writer = None

    def __exit__(self, exc_type, exc, tb):
        self.close(exc_type is None)
