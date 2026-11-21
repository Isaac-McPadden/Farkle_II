"""Process-local protection for the files measured by a scientific invocation."""

from __future__ import annotations

import ctypes
import hashlib
import os
import sys
from contextlib import ExitStack
from pathlib import Path


class ScientificDrift(RuntimeError):
    """The launch evidence no longer describes this computation."""


class MeasuredFiles:
    """Protect measured bytes and observe changes to import directory names.

    Acquire protection before reading any bytes. Directory identities also matter:
    renaming an ancestor can redirect an import without modifying the child's inode.
    """

    libc: ctypes.CDLL
    fd: int

    def __init__(self) -> None:
        self.stack = ExitStack()
        self.pid = os.getpid()
        self.digests: dict[Path, str] = {}
        self.directories: set[Path] = set()
        self.watched: dict[int, Path] = {}
        self.failure = ""
        self.sources: dict[tuple[Path, frozenset[str]], dict[str, str]] = {}
        self.namespaces: dict[Path, int] = {}
        self.leases: dict[int, Path] = {}
        self.readonly_volumes: set[str] = set()
        if os.name == "nt":
            from ctypes import wintypes

            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.CreateFileW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.c_void_p,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.HANDLE,
            ]
            self.kernel.CreateFileW.restype = wintypes.HANDLE
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.kernel.FindFirstChangeNotificationW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            self.kernel.FindFirstChangeNotificationW.restype = wintypes.HANDLE
            self.kernel.FindCloseChangeNotification.argtypes = [wintypes.HANDLE]
            self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        elif sys.platform.startswith("linux"):
            self.libc = ctypes.CDLL(None, use_errno=True)
            self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if self.fd < 0:
                raise OSError(ctypes.get_errno(), "Cannot monitor scientific dependencies")
            self.stack.callback(os.close, self.fd)
            self.linux_mounts = self._linux_mounts()
        else:
            raise ScientificDrift(f"No qualified file stability mechanism for {sys.platform}")

    def _protect(self, path: Path, *, directory: bool) -> None:
        if os.name == "nt":
            # FILE_READ_ATTRIBUTES on directories, GENERIC_READ on files. Allow
            # normal directory activity, but never deletion/renaming of the directory.
            handle = self.kernel.CreateFileW(
                str(path),
                0 if directory else 0x80000000,
                3 if directory else 1,
                None,
                3,
                0x02000000 if directory else 0,
                None,
            )
            if handle == ctypes.c_void_p(-1).value:
                raise ScientificDrift(
                    f"Cannot protect scientific dependency {path}: "
                    f"{ctypes.WinError(ctypes.get_last_error())}"
                )
            self.stack.callback(self.kernel.CloseHandle, handle)
        else:
            # IN_MOVE_SELF, IN_DELETE_SELF, plus IN_MODIFY / IN_CLOSE_WRITE on files.
            mask = 0xC00 if directory else 0xC0A
            wd = self.libc.inotify_add_watch(self.fd, os.fsencode(path), mask | 0x20000000)
            if wd < 0:
                raise ScientificDrift(f"Cannot monitor {path}: {os.strerror(ctypes.get_errno())}")
            self.watched[wd] = path
            if not directory:
                self._lease(path)

    @staticmethod
    def _linux_mounts() -> dict[str, tuple[str, bool]]:
        volumes = {}
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            mount, options = line.split(" - ", 1)
            filesystem, _, flags = options.split()
            volumes[mount.split()[2]] = filesystem, "ro" in flags.split(",")
        return volumes

    def _lease(self, path: Path) -> None:
        # A directory watch alone misses writes through mmap. Refuse a mutable
        # dependency when the kernel cannot grant a lease before measurement.
        if sys.platform != "linux":
            raise ScientificDrift("Linux read lease requested on another platform")
        import fcntl
        import signal

        device = path.stat().st_dev
        volume = f"{os.major(device)}:{os.minor(device)}"
        filesystem, readonly = self.linux_mounts.get(volume, ("unknown", False))
        if filesystem not in {
            "ext2",
            "ext3",
            "ext4",
            "xfs",
            "btrfs",
            "tmpfs",
            "f2fs",
            "erofs",
            "squashfs",
            "iso9660",
        }:
            raise ScientificDrift(f"Unqualified scientific filesystem {filesystem} for {path}")
        if readonly:
            self.readonly_volumes.add(volume)
            return
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        self.stack.callback(os.close, fd)
        try:
            # SIGURG is ignored by default. Check F_GETLEASE synchronously at
            # boundaries, including breaks already forced by the kernel timeout.
            fcntl.fcntl(fd, fcntl.F_SETSIG, signal.SIGURG)
            fcntl.fcntl(fd, fcntl.F_SETLEASE, fcntl.F_RDLCK)
        except OSError as exc:
            raise ScientificDrift(
                f"Cannot protect {path} with a Linux read lease. Use a read-only "
                f"scientific checkout and numerical installation: {exc}"
            ) from exc
        self.leases[fd] = path

    def watch_names(self, directory: Path) -> None:
        if directory in self.namespaces:
            return
        if os.name == "nt":
            handle = self.kernel.FindFirstChangeNotificationW(str(directory), False, 3)
            if handle == ctypes.c_void_p(-1).value:
                raise ScientificDrift(f"Cannot monitor import directory {directory}")
            self.stack.callback(self.kernel.FindCloseChangeNotification, handle)
        else:
            handle = self.libc.inotify_add_watch(self.fd, os.fsencode(directory), 0x200003C0)
            if handle < 0:
                raise ScientificDrift(f"Cannot monitor import directory {directory}")
            self.watched[handle] = directory
        self.namespaces[directory] = handle

    def check(self) -> None:
        if os.name == "nt" and not self.failure:
            for path, handle in self.namespaces.items():
                if self.kernel.WaitForSingleObject(handle, 0) != 258:
                    self.failure = f"Scientific import directory changed: {path}"
                    break
        if os.name != "nt" and not self.failure:
            import struct

            try:
                event = os.read(self.fd, 65536)
            except BlockingIOError:
                return
            if event:
                wd, mask, _, _ = struct.unpack_from("iIII", event)
                self.failure = f"Scientific dependency changed: {self.watched.get(wd, 'watch queue')} ({mask:#x})"
        if self.failure:
            raise ScientificDrift(self.failure)

    def boundary(self) -> None:
        self.check()
        if sys.platform == "linux":
            import fcntl

            for fd, path in self.leases.items():
                if fcntl.fcntl(fd, fcntl.F_GETLEASE) != fcntl.F_RDLCK:
                    self.failure = f"Scientific file lease broken: {path}"
                    raise ScientificDrift(self.failure)
            mounts = self._linux_mounts() if self.readonly_volumes else {}
            for volume in self.readonly_volumes:
                if mounts.get(volume) != self.linux_mounts[volume]:
                    self.failure = f"Scientific read-only filesystem changed: {volume}"
                    raise ScientificDrift(self.failure)

    def digest(self, path: Path) -> str:
        if self.failure:
            raise ScientificDrift(self.failure)
        if path in self.digests:
            return self.digests[path]
        # An alias with a replaceable link is not covered by a handle on its target.
        path = path.absolute()
        resolved = path.resolve(strict=True)
        if path != resolved:
            raise ScientificDrift(f"Unprotected dependency path alias: {path}")
        for parent in reversed(path.parents):
            if parent not in self.directories:
                self._protect(parent, directory=True)
                self.directories.add(parent)
        self._protect(path, directory=False)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.digests[path] = digest
        return digest

    def close(self) -> None:
        self.stack.close()
        self.failure = self.failure or "Scientific measurement scope is closed"
