# src/farkle/utils/manifest.py
"""Append-only NDJSON manifest utilities.

This module provides a tiny, crash-resilient manifest for long runs:
one JSON object per line (NDJSON). Each shard writer appends exactly
one line when it finishes successfully.

Why this approach:
- Append-safe: we open with O_APPEND and write each record as a single
  os.write() call (atomic append on the same filesystem).
- Crash tolerant: if your process dies mid-run, previously written
  lines remain valid; you can resume by scanning the file.
- Tool-friendly: easy to grep/jq or ingest with DuckDB/Polars/Arrow.

Typical usage (after a shard is atomically moved into place):
    append_manifest_line(
        manifest_path,
        {
            "path": rel_parquet_path,
            "rows": rows_written,
            "players": players,
            "block": block_id,
            "pid": os.getpid(),
            # optional: anything else useful
        },
    )
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

if os.name == "nt":
    import msvcrt
else:
    import fcntl

__all__ = ["append_manifest_line", "append_manifest_many", "iter_manifest"]


def _ensure_parent_dir(path: os.PathLike[str] | str) -> None:
    """Create parent directories for ``path`` if they do not exist."""
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def _json_line(record: Mapping[str, Any], *, add_timestamp: bool = True) -> bytes:
    """Serialize *record* as a single NDJSON line (UTF-8 bytes)."""
    if add_timestamp and "ts" not in record:
        # RFC 3339/ISO 8601 UTC with 'Z' suffix for easy machine parsing
        record = {**record, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    text = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def _open_append_fd(path: os.PathLike[str] | str) -> int:
    """Open *path* for append as a raw file descriptor (binary)."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    # On Windows, ensure binary mode to avoid newline transformation
    if os.name == "nt":
        flags |= os.O_BINARY  # type: ignore[attr-defined]
    return os.open(os.fspath(path), flags, 0o644)


def _lock_fd(fd: int) -> None:
    """Acquire an exclusive lock for *fd* (blocks until available)."""
    if os.name == "nt":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_EX)


def _unlock_fd(fd: int) -> None:
    """Release an exclusive lock for *fd*."""
    if os.name == "nt":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _write_all(fd: int, data: bytes) -> None:
    """Write all bytes to *fd*, handling partial writes."""
    view = memoryview(data)
    total = 0
    length = len(view)
    while total < length:
        written = os.write(fd, view[total:])
        total += written


def append_manifest_line(
    manifest_path: os.PathLike[str] | str,
    record: Mapping[str, Any],
    *,
    add_timestamp: bool = True,
    ensure_dir: bool = True,
) -> None:
    """Append one JSON object to *manifest_path* as NDJSON (atomic append).

    Parameters
    ----------
    manifest_path
        File to append to (created if missing).
    record
        Mapping to serialize as one JSON object on a single line.
    add_timestamp
        When True (default), add ``"ts"`` with UTC ISO8601 if absent.
    ensure_dir
        When True (default), create parent directories if needed.
    """
    if ensure_dir:
        _ensure_parent_dir(manifest_path)
    data = _json_line(record, add_timestamp=add_timestamp)
    fd = _open_append_fd(manifest_path)
    try:
        _lock_fd(fd)
        try:
            _write_all(fd, data)
            os.fsync(fd)
        finally:
            _unlock_fd(fd)
    finally:
        os.close(fd)


def append_manifest_many(
    manifest_path: os.PathLike[str] | str,
    records: Iterable[Mapping[str, Any]],
    *,
    add_timestamp: bool = True,
    ensure_dir: bool = True,
) -> None:
    """Append multiple records efficiently in one write (when possible)."""
    if ensure_dir:
        _ensure_parent_dir(manifest_path)
    # Build once to minimize fd churn; still a single append write.
    blob = b"".join(_json_line(r, add_timestamp=add_timestamp) for r in records)
    if not blob:
        return
    fd = _open_append_fd(manifest_path)
    try:
        _lock_fd(fd)
        try:
            _write_all(fd, blob)
            os.fsync(fd)
        finally:
            _unlock_fd(fd)
    finally:
        os.close(fd)


def iter_manifest(manifest_path: os.PathLike[str] | str) -> Iterator[dict[str, Any]]:
    """Yield records from an NDJSON manifest. Skips blank lines."""
    p = Path(manifest_path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
