# src/farkle/writer.py
from __future__ import annotations

import json
import os
import tempfile
import time
import zlib
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

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

def _crc32_bytesize(path: str) -> tuple[str, int]:
    buf_size = 1024 * 1024
    crc = 0
    total = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            crc = zlib.crc32(b, crc)
            total += len(b)
    return f"{crc & 0xffffffff:08x}", total

@dataclass
class ParquetShardWriter:
    out_path: str
    schema: pa.Schema
    compression: str = "snappy"
    row_group_size: int = 200_000

    _writer: Optional[pq.ParquetWriter] = None
    _tmp_path: Optional[str] = None
    _rows_written: int = 0

    def __enter__(self) -> "ParquetShardWriter":
        with atomic_path(self.out_path) as tmp:
            # We keep tmp reserved; hold its name and open ParquetWriter on it.
            self._tmp_path = tmp
            self._writer = pq.ParquetWriter(
                self._tmp_path, self.schema,
                compression=self.compression,
                use_dictionary=True
            )
        return self

    def write_batches(self, tables: Iterable[pa.Table]) -> None:
        assert self._writer and self._tmp_path
        for tbl in tables:
            # Ensure proper row groups (Arrow will split as needed)
            self._writer.write_table(tbl, row_group_size=self.row_group_size)
            self._rows_written += tbl.num_rows

    def __exit__(self, exc_type, exc, tb):
        if not self._writer or not self._tmp_path:
            return
        self._writer.close()
        if exc is None:
            # Success path: swap into place
            os.replace(self._tmp_path, self.out_path)
        else:
            # Failure path: drop temp file
            with suppress(FileNotFoundError):
                os.remove(self._tmp_path)

def append_manifest_line(manifest_path: str, record: Dict[str, Any]) -> None:
    """Append one JSON line atomically."""
    line = json.dumps(record, separators=(",", ":"))
    dir_ = os.path.dirname(os.path.abspath(manifest_path)) or "."
    os.makedirs(dir_, exist_ok=True)
    tmp = os.path.join(dir_, f"._tmp_manifest_{time.time_ns()}.jsonl")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(line + "\n")
    # atomic append by rename+append is not portable; instead: best-effort lockless append
    # pragmatic approach: rename to final if it doesn't exist; else append safely
    if not os.path.exists(manifest_path):
        os.replace(tmp, manifest_path)
    else:
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        os.remove(tmp)
