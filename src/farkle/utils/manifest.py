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
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


_WINDOWS_LOCK_OFFSET = 0
_WINDOWS_LOCK_BYTES = 1
_WINDOWS_LOCK_RETRY_S = 0.05
MANIFEST_SCHEMA_VERSION = 2
EVENT_RUN_START = "run_start"
EVENT_RUN_END = "run_end"
EVENT_STAGE_START = "stage_start"
EVENT_STAGE_END = "stage_end"
LEGACY_MANIFEST_EVENTS = frozenset({"stage-end", "step_start", "step_end"})

__all__ = [
    "EVENT_RUN_END",
    "EVENT_RUN_START",
    "EVENT_STAGE_END",
    "EVENT_STAGE_START",
    "LEGACY_MANIFEST_EVENTS",
    "MANIFEST_SCHEMA_VERSION",
    "append_manifest_event",
    "append_manifest_line",
    "append_manifest_many",
    "ensure_manifest_v2",
    "iter_manifest",
    "make_run_id",
]


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



if sys.platform == "win32":

    def _lock_fd(fd: int) -> None:
        """Acquire an exclusive lock for *fd* (blocks until available)."""
        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, _WINDOWS_LOCK_BYTES)  # type: ignore[attr-defined]
                return
            except OSError:
                time.sleep(_WINDOWS_LOCK_RETRY_S)


    def _unlock_fd(fd: int) -> None:
        """Release an exclusive lock for *fd*."""
        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        with suppress(OSError):
            msvcrt.locking(fd, msvcrt.LK_UNLCK, _WINDOWS_LOCK_BYTES)  # type: ignore[attr-defined]

else:

    def _lock_fd(fd: int) -> None:
        """Acquire an exclusive lock for *fd* (blocks until available)."""
        fcntl.flock(fd, fcntl.LOCK_EX)  # type: ignore[attr-defined]


    def _unlock_fd(fd: int) -> None:
        """Release an exclusive lock for *fd*."""
        fcntl.flock(fd, fcntl.LOCK_UN)  # type: ignore[attr-defined]


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


def make_run_id(run_label: str) -> str:
    """Return a filesystem- and log-friendly run identifier."""

    normalized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in run_label)
    return f"{normalized}_{time.time_ns()}"


def _manifest_lines_are_v2(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            saw_record = False
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                saw_record = True
                try:
                    record = json.loads(line)
                except Exception:
                    return False
                if not isinstance(record, dict):
                    return False
                event = record.get("event")
                if event in LEGACY_MANIFEST_EVENTS:
                    return False
                if record.get("schema_version") != MANIFEST_SCHEMA_VERSION:
                    return False
            return True if saw_record else True
    except Exception:
        return False


def ensure_manifest_v2(manifest_path: os.PathLike[str] | str) -> Path | None:
    """Rotate a legacy manifest aside before appending v2 event records."""

    path = Path(manifest_path)
    if not path.exists() or _manifest_lines_are_v2(path):
        return None

    rotated = path.with_name(f"{path.stem}.pre_v2{path.suffix}")
    rotated.parent.mkdir(parents=True, exist_ok=True)
    if rotated.exists():
        rotated.unlink()
    path.replace(rotated)
    return rotated


def append_manifest_event(
    manifest_path: os.PathLike[str] | str,
    record: Mapping[str, Any],
    *,
    run_id: str,
    config_sha: str | None,
    add_timestamp: bool = True,
) -> None:
    """Append a v2 pipeline/orchestration manifest event."""

    event = str(record.get("event", "")).strip()
    if not event:
        raise ValueError("manifest event record is missing 'event'")
    if event in LEGACY_MANIFEST_EVENTS:
        raise ValueError(f"legacy manifest event {event!r} is not allowed in v2 manifests")

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "event": event,
        "config_sha": config_sha,
        **{key: value for key, value in record.items() if key != "event"},
    }
    append_manifest_line(
        manifest_path,
        payload,
        add_timestamp=add_timestamp,
    )
