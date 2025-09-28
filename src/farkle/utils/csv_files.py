"""File related helper functions.

Only very small, generic utilities live here so that modules which
need to perform disk I/O do not have to duplicate boilerplate.  The
helpers are intentionally lightweight and avoid pulling in heavy
dependencies so they are safe to import in worker processes.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from pathlib import Path
from typing import Mapping, Sequence

# Batching and queue sizes for file I/O.  These values strike a balance
# between throughput and memory usage when many games are streamed to a
# CSV file.
BUFFER_SIZE = 10_000
QUEUE_SIZE = 2_000


def _writer_worker(queue: mp.Queue, out_csv: str, header: Sequence[str]) -> None:
    """Consume *queue* and append rows to ``out_csv``.

    The worker writes ``header`` once when the file is created.  Rows are
    buffered in memory until :data:`BUFFER_SIZE` is reached to reduce the
    number of disk writes.  A ``None`` item on the queue terminates the
    worker.
    """

    first = not Path(out_csv).exists() or Path(out_csv).stat().st_size == 0
    with open(out_csv, "a", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=header)
        if first:
            writer.writeheader()
        buffer: list[Mapping[str, object]] = []
        while True:
            row = queue.get()
            if row is None:
                break
            buffer.append(row)
            if len(buffer) >= BUFFER_SIZE:
                writer.writerows(buffer)
                file_handle.flush()
                buffer.clear()
        if buffer:
            writer.writerows(buffer)


__all__ = ["BUFFER_SIZE", "QUEUE_SIZE", "_writer_worker"]
