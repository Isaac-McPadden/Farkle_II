"""Bound metadata reads and preserve access times during a resume preview."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from pathlib import Path
from typing import Any, BinaryIO

_PREVIEW: ContextVar[int | None] = ContextVar("preview_evidence_reads", default=None)
MAX_PREVIEW_BYTES = 1024 * 1024
_PREVIEW_TOTAL_BYTES = 8 * MAX_PREVIEW_BYTES


class EvidenceReadDeferred(RuntimeError):
    """The evidence cannot be read safely within the metadata preview."""


@contextmanager
def preview_evidence_reads() -> Iterator[None]:
    if _PREVIEW.get() is not None:
        yield
        return
    token = _PREVIEW.set(_PREVIEW_TOTAL_BYTES)
    try:
        yield
    finally:
        _PREVIEW.reset(token)


@cache
def _windows_file_api() -> Any:
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.SetFileTime.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
    )
    kernel.SetFileTime.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def _open_windows(path: Path) -> int:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = _windows_file_api()
    # OPEN_EXISTING, shared read/write/delete. FILE_WRITE_ATTRIBUTES permits
    # setting the per-handle no-access-time-update flag without setting a time.
    handle = kernel.CreateFileW(str(path.resolve()), 0x80000100, 7, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        suppress_access_time = wintypes.FILETIME(0xFFFFFFFF, 0xFFFFFFFF)
        if not kernel.SetFileTime(handle, None, ctypes.byref(suppress_access_time), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise


def _open_without_access_update(path: Path) -> BinaryIO:
    try:
        if os.name == "nt":
            fd = _open_windows(path)
        elif hasattr(os, "O_NOATIME"):
            fd = os.open(path, os.O_RDONLY | os.O_NOATIME)
        else:
            raise EvidenceReadDeferred(f"access-time-preserving reads are unavailable for {path}")
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EPERM, errno.EINVAL, errno.ENOTSUP}:
            raise EvidenceReadDeferred(
                f"cannot open without updating access time at {path}"
            ) from exc
        raise
    return os.fdopen(fd, "rb")


def read_evidence_bytes(path: Path) -> bytes:
    """Use the ordinary reader for execution and a bounded, no-atime reader for preview."""
    budget = _PREVIEW.get()
    if budget is None:
        return path.read_bytes()
    limit = min(MAX_PREVIEW_BYTES, budget - 1)
    properties = path.stat()
    if not stat.S_ISREG(properties.st_mode):
        raise EvidenceReadDeferred(f"expected a regular evidence file at {path}")
    if limit < 0 or properties.st_size > limit:
        raise EvidenceReadDeferred(f"metadata exceeds the preview read limit at {path}")
    with _open_without_access_update(path) as stream:
        data = stream.read(limit + 1)
    _PREVIEW.set(budget - len(data))
    if len(data) > limit:
        raise EvidenceReadDeferred(f"metadata exceeds the preview read limit at {path}")
    return data
