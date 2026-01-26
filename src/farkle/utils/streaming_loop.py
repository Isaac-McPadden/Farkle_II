# src/farkle/utils/streaming_loop.py
"""
Helpers for streaming Parquet shard creation. Provides producer/consumer
thread utilities, a bounded queue wrapper, and the runner that writes batches
while recording manifest metadata.
"""
from __future__ import annotations

import logging
import os
import queue
import time
from typing import Any, Callable, Dict, Iterable

import pyarrow as pa

from .manifest import append_manifest_line
from .types import Compression
from .writer import ParquetShardWriter

LOGGER = logging.getLogger(__name__)


def run_streaming_shard(
    *,
    out_path: str,
    manifest_path: str,
    schema: pa.Schema,
    batch_iter: Iterable[pa.Table],
    row_group_size: int = 200_000,
    compression: Compression = "snappy",
    manifest_extra: Dict[str, Any] | None = None,
):
    """Stream batches to parquet and append a manifest entry on success."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with ParquetShardWriter(
        out_path=out_path, schema=schema, compression=compression, row_group_size=row_group_size
    ) as w:
        w.write_batches(batch_iter)
    rows = getattr(w, "rows_written", None)
    if not _output_ready(out_path, manifest_path, manifest_extra):
        return
    # On success, append a manifest line
    manifest_dir = os.path.dirname(manifest_path)
    manifest_dir = os.path.abspath(manifest_dir) if manifest_dir else os.path.abspath(os.curdir)

    try:
        rel_path = os.path.relpath(out_path, start=manifest_dir)
    except ValueError:
        rel_path = os.path.abspath(out_path)

    rel_norm = os.path.normpath(rel_path)
    rel_is_outside = os.path.isabs(rel_path) or rel_norm == os.pardir or rel_norm.startswith(
        os.pardir + os.sep
    )
    resolved_path = rel_path if os.path.isabs(rel_path) else os.path.join(manifest_dir, rel_path)
    resolved_exists = os.path.exists(resolved_path)
    if rel_is_outside or not resolved_exists:
        LOGGER.error(
            "Manifest path %s is invalid relative to %s (exists=%s).",
            rel_path,
            manifest_dir,
            resolved_exists,
        )
        if os.path.exists(out_path):
            rel_path = os.path.abspath(out_path)
        else:
            LOGGER.error("Skipping manifest append because output path %s is missing.", out_path)
            return

    append_manifest_line(
        manifest_path,
        {
            "path": rel_path,
            "rows": rows,
            **(manifest_extra or {}),
        },
    )


def _output_ready(
    out_path: str, manifest_path: str, manifest_extra: Dict[str, Any] | None
) -> bool:
    """Return True when the output path exists and is non-empty."""
    if os.path.exists(out_path):
        return _check_output_size(out_path, manifest_path, manifest_extra)

    time.sleep(0.05)
    if os.path.exists(out_path):
        return _check_output_size(out_path, manifest_path, manifest_extra)

    LOGGER.error(
        "Missing output file after shard write (out_path=%s, manifest_path=%s, manifest_extra=%s).",
        out_path,
        manifest_path,
        manifest_extra,
    )
    return False


def _check_output_size(
    out_path: str, manifest_path: str, manifest_extra: Dict[str, Any] | None
) -> bool:
    try:
        out_size = os.path.getsize(out_path)
    except OSError:
        out_size = 0
    if out_size <= 0:
        LOGGER.error(
            "Empty output file after shard write (out_path=%s, manifest_path=%s, manifest_extra=%s).",
            out_path,
            manifest_path,
            manifest_extra,
        )
        return False
    return True


def producer_thread(
    push: Callable[[pa.Table], None],
    mk_batches: Callable[[], Iterable[pa.Table]],
):
    """Produce tables from ``mk_batches`` and hand them to ``push``."""
    for tbl in mk_batches():
        push(tbl)


def writer_thread(
    pop: Callable[[], pa.Table | None],
    *,
    out_path: str,
    manifest_path: str,
    schema: pa.Schema,
    row_group_size: int,
    compression: Compression,
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
        self.q: queue.Queue[pa.Table | None] = queue.Queue(maxsize=maxsize)

    def push(self, tbl: pa.Table) -> None:
        """Enqueue a table, blocking if the queue is full."""
        self.q.put(tbl)

    def pop(self) -> pa.Table | None:
        """Dequeue a table, blocking until available."""
        return self.q.get()

    def close(self):
        """Signal consumers to stop by enqueuing a sentinel ``None``."""
        self.q.put(None)
