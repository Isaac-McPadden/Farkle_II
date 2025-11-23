# src/farkle/utils/streaming_loop.py
"""
Helpers for streaming Parquet shard creation. Provides producer/consumer
thread utilities, a bounded queue wrapper, and the runner that writes batches
while recording manifest metadata.
"""
from __future__ import annotations

import os
import queue
from typing import Any, Callable, Dict, Iterable

import pyarrow as pa

from .manifest import append_manifest_line
from .writer import ParquetShardWriter


def run_streaming_shard(
    *,
    out_path: str,
    manifest_path: str,
    schema: pa.Schema,
    batch_iter: Iterable[pa.Table],
    row_group_size: int = 200_000,
    compression: str = "snappy",
    manifest_extra: Dict[str, Any] | None = None,
):
    """Stream batches to parquet and append a manifest entry on success."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with ParquetShardWriter(
        out_path=out_path, schema=schema, compression=compression, row_group_size=row_group_size
    ) as w:
        w.write_batches(batch_iter)
    rows = getattr(w, "rows_written", None)
    # On success, append a manifest line
    manifest_dir = os.path.dirname(manifest_path)
    manifest_dir = os.path.abspath(manifest_dir) if manifest_dir else os.path.abspath(os.curdir)

    try:
        rel_path = os.path.relpath(out_path)
    except ValueError:
        try:
            rel_path = os.path.relpath(out_path, start=manifest_dir)
        except ValueError:
            rel_path = os.path.abspath(out_path)

    append_manifest_line(
        manifest_path,
        {
            "path": rel_path,
            "rows": rows,
            **(manifest_extra or {}),
        },
    )


def producer_thread(
    push: Callable[[pa.Table], None],
    mk_batches: Callable[[], Iterable[pa.Table]],
):
    """Produce tables from ``mk_batches`` and hand them to ``push``."""
    for tbl in mk_batches():
        push(tbl)


def writer_thread(
    pop: Callable[[], pa.Table],
    *,
    out_path: str,
    manifest_path: str,
    schema: pa.Schema,
    row_group_size: int,
    compression: str,
    manifest_extra: Dict[str, Any] | None,
):
    """Consume tables from ``pop`` and write them via :func:`run_streaming_shard`."""

    def batches():
        """Yield tables until a ``None`` sentinel is received."""
        while True:
            tbl = pop()
            if tbl is None:  # poison pill
                break
            yield tbl

    run_streaming_shard(
        out_path=out_path,
        manifest_path=manifest_path,
        schema=schema,
        batch_iter=batches(),
        row_group_size=row_group_size,
        compression=compression,
        manifest_extra=manifest_extra,
    )


class BoundedQueue:
    """Thin wrapper around :class:`queue.Queue` with explicit push/pop methods."""

    def __init__(self, maxsize: int):
        self.q: queue.Queue[int | None] = queue.Queue(maxsize=maxsize)

    def push(self, tbl: pa.Table):
        """Enqueue a table, blocking if the queue is full."""
        self.q.put(tbl)

    def pop(self):
        """Dequeue a table, blocking until available."""
        return self.q.get()

    def close(self):
        """Signal consumers to stop by enqueuing a sentinel ``None``."""
        self.q.put(None)
