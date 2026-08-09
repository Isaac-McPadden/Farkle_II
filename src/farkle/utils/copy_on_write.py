"""Atomic-writer helpers for copy-on-write clones with bounded-copy fallback."""

from __future__ import annotations

import errno
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CopyProvenance:
    """Execution provenance for one byte-preserving file representation."""

    representation: str
    backend: str
    fallback_reason: str | None
    copy_buffer_bytes: int


def _linux_reflink(source: Path, destination: Path) -> None:
    import fcntl  # Linux-only import

    ficlone = 0x40049409
    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        ioctl: Any = fcntl.ioctl  # type: ignore[attr-defined]
        ioctl(destination_handle.fileno(), ficlone, source_handle.fileno())


def _windows_reflink(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    completed = subprocess.run(
        ["fsutil", "file", "clone", str(destination), str(source)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0 or not destination.is_file():
        detail = (completed.stderr or completed.stdout or "fsutil clone failed").strip()
        raise OSError(errno.EOPNOTSUPP, detail[:512])


def try_reflink(source: Path, destination: Path) -> str:
    """Clone *source* into *destination* or raise a deterministic ``OSError``."""

    if sys.platform.startswith("linux"):
        _linux_reflink(source, destination)
        return "linux_ficlone"
    if os.name == "nt":
        _windows_reflink(source, destination)
        return "windows_fsutil_clone"
    raise OSError(errno.EOPNOTSUPP, f"reflink backend unavailable on {sys.platform}")


def _fallback_reason(exc: OSError) -> str:
    error_name = errno.errorcode.get(exc.errno or 0, "OSERROR")
    return f"{error_name}:{type(exc).__name__}"


def clone_or_copy_bounded(
    source: Path,
    destination: Path,
    *,
    copy_buffer_bytes: int,
) -> CopyProvenance:
    """Prefer an independent copy-on-write clone, then copy with a bounded buffer."""

    if copy_buffer_bytes < 1:
        raise ValueError("copy buffer ceiling must be positive")
    try:
        backend = try_reflink(source, destination)
    except OSError as exc:
        destination.unlink(missing_ok=True)
        with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
            shutil.copyfileobj(
                source_handle,
                destination_handle,
                length=copy_buffer_bytes,
            )
        shutil.copystat(source, destination, follow_symlinks=True)
        return CopyProvenance(
            representation="physical_copy",
            backend="bounded_copyfileobj",
            fallback_reason=_fallback_reason(exc),
            copy_buffer_bytes=copy_buffer_bytes,
        )
    shutil.copystat(source, destination, follow_symlinks=True)
    return CopyProvenance(
        representation="reflink",
        backend=backend,
        fallback_reason=None,
        copy_buffer_bytes=copy_buffer_bytes,
    )


__all__ = ["CopyProvenance", "clone_or_copy_bounded", "try_reflink"]
