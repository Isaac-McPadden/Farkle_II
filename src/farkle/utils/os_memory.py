"""Operating-system aggregate memory boundaries for pipeline process trees.

The existing :mod:`farkle.utils.parallel` RSS guard remains the cooperative
early-warning mechanism.  This module supplies the final, aggregate backstop:
a Windows Job Object or a delegated cgroup-v2 child cgroup.  A lightweight
supervisor establishes the boundary before importing the analysis CLI, so all
worker processes and native allocations are descendants of one protected root.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

BOUNDARY_STATUS_ENV = "FARKLE_OS_MEMORY_BOUNDARY"
MEMORY_LIMIT_EXIT_CODE = 86
SETUP_FAILURE_EXIT_CODE = 78
_MIB = 1024 * 1024
_WINDOWS_JOB_MEMORY = 0x00000200
_WINDOWS_KILL_ON_JOB_CLOSE = 0x00002000
_WINDOWS_CREATE_SUSPENDED = 0x00000004
_WINDOWS_THREAD_SUSPEND_RESUME = 0x0002
_WINDOWS_TH32CS_SNAPTHREAD = 0x00000004
_NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


class MemoryBoundaryError(RuntimeError):
    """Raised when a required aggregate OS memory boundary cannot be established."""


@dataclass(frozen=True)
class MemoryBoundaryStatus:
    """Serializable execution provenance for the selected OS memory boundary."""

    contract_version: int
    backend: str
    platform: str
    enforced: bool
    requested_hard_limit_mb: int
    effective_hard_limit_mb: int | None
    enclosing_hard_limit_mb: int | None
    strict_required: bool
    fallback_used: bool
    detail: str
    boundary_path: str | None = None
    enclosing_job: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible provenance."""

        return asdict(self)


def current_memory_boundary() -> dict[str, Any] | None:
    """Return inherited boundary provenance, if this process has a supervisor."""

    encoded = os.environ.get(BOUNDARY_STATUS_ENV)
    if not encoded:
        return None
    try:
        payload = json.loads(encoded)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def memory_boundary_provenance(resources: Any | None = None) -> dict[str, Any]:
    """Return active provenance or an explicit not-established record."""

    active = current_memory_boundary()
    if active is not None:
        return active
    requested = int(getattr(resources, "max_memory_mb", 0)) if resources is not None else 0
    required = bool(getattr(resources, "os_memory_limit_required", False))
    return MemoryBoundaryStatus(
        contract_version=1,
        backend="none",
        platform=_platform_label(),
        enforced=False,
        requested_hard_limit_mb=requested,
        effective_hard_limit_mb=None,
        enclosing_hard_limit_mb=None,
        strict_required=required,
        fallback_used=False,
        detail="no launcher boundary provenance is present in this process",
    ).as_dict()


def _platform_label() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform.startswith("linux"):
        try:
            version = Path("/proc/version").read_text(encoding="utf-8").lower()
        except OSError:
            version = ""
        if "microsoft" in version or os.environ.get("WSL_INTEROP"):
            return "wsl"
        if Path("/.dockerenv").exists():
            return "linux-container"
        return "linux"
    return sys.platform


def _status_environment(
    base: Mapping[str, str] | None, status: MemoryBoundaryStatus
) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env[BOUNDARY_STATUS_ENV] = json.dumps(status.as_dict(), sort_keys=True, separators=(",", ":"))
    return env


def _preimport_environment(
    base: Mapping[str, str] | None,
    resources: Any,
) -> dict[str, str]:
    """Apply the existing native-thread budget before numerical libraries import."""

    env = dict(os.environ if base is None else base)
    native_threads = str(max(1, int(getattr(resources, "native_threads_per_worker", 1))))
    for variable in _NATIVE_THREAD_ENV_VARS:
        env[variable] = native_threads
    env["PYARROW_NUM_THREADS"] = native_threads
    return env


def _warn_unenforced(status: MemoryBoundaryStatus) -> None:
    message = (
        "OS MEMORY CEILING IS NOT ENFORCED; development fallback is active: " f"{status.detail}"
    )
    LOGGER.warning(message, extra={"stage": "os_memory_preflight", **status.as_dict()})
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def _run_unenforced(
    command: Sequence[str],
    resources: Any,
    *,
    env: Mapping[str, str] | None,
    detail: str,
) -> int:
    status = MemoryBoundaryStatus(
        contract_version=1,
        backend="unenforced",
        platform=_platform_label(),
        enforced=False,
        requested_hard_limit_mb=int(resources.max_memory_mb),
        effective_hard_limit_mb=None,
        enclosing_hard_limit_mb=None,
        strict_required=bool(resources.os_memory_limit_required),
        fallback_used=True,
        detail=detail,
    )
    _warn_unenforced(status)
    return subprocess.run(command, env=_status_environment(env, status), check=False).returncode


def supervise_process(
    command: Sequence[str],
    resources: Any,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run ``command`` under the configured aggregate process-tree limit."""

    if not command:
        raise ValueError("protected command cannot be empty")
    env = _preimport_environment(env, resources)
    enabled = bool(resources.os_memory_limit_enabled)
    required = bool(resources.os_memory_limit_required)
    fallback = bool(resources.allow_unenforced_memory_fallback)
    if required and (not enabled or fallback):
        raise MemoryBoundaryError("invalid strict OS memory enforcement configuration")
    if not required and not fallback:
        raise MemoryBoundaryError("non-strict execution requires explicit fallback permission")
    if not enabled:
        return _run_unenforced(
            command,
            resources,
            env=env,
            detail="OS memory enforcement was explicitly disabled",
        )

    try:
        if sys.platform == "win32":
            return _run_windows_job(command, resources, env=env)
        if sys.platform.startswith("linux"):
            return _run_cgroup_v2(command, resources, env=env)
        raise MemoryBoundaryError(f"no aggregate memory backend for {sys.platform!r}")
    except MemoryBoundaryError as exc:
        if required:
            raise
        if not fallback:  # pragma: no cover - validated above
            raise
        return _run_unenforced(command, resources, env=env, detail=str(exc))


def supervise_module_if_needed(
    module: str,
    argv: Sequence[str],
    resources: Any,
) -> int | None:
    """Relaunch a direct module entry point unless a boundary is already inherited."""

    if current_memory_boundary() is not None:
        return None
    return supervise_process(
        [sys.executable, "-m", module, *argv],
        resources,
        env=os.environ,
    )


if sys.platform == "win32":
    from ctypes import wintypes

    class _JobBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JobExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _ThreadEntry32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        ]


def _windows_api() -> Any:
    if sys.platform != "win32":  # pragma: no cover - platform gate
        raise MemoryBoundaryError("Windows Job Objects are unavailable")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.IsProcessInJob.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.BOOL),
    ]
    kernel32.IsProcessInJob.restype = wintypes.BOOL
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32)]
    kernel32.Thread32First.restype = wintypes.BOOL
    kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32)]
    kernel32.Thread32Next.restype = wintypes.BOOL
    kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenThread.restype = wintypes.HANDLE
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    return kernel32


def _query_windows_job_info(kernel32: Any, handle: Any) -> Any:
    info = _JobExtendedLimitInformation()
    returned = wintypes.DWORD()
    if not kernel32.QueryInformationJobObject(
        handle,
        9,
        ctypes.byref(info),
        ctypes.sizeof(info),
        ctypes.byref(returned),
    ):
        error = ctypes.get_last_error()
        raise MemoryBoundaryError(f"QueryInformationJobObject failed with WinError {error}")
    return info


def _resume_windows_process(kernel32: Any, pid: int) -> None:
    """Resume every initial thread in a process created with CREATE_SUSPENDED."""

    snapshot = kernel32.CreateToolhelp32Snapshot(_WINDOWS_TH32CS_SNAPTHREAD, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        error = ctypes.get_last_error()
        raise MemoryBoundaryError(f"CreateToolhelp32Snapshot failed with WinError {error}")
    resumed = 0
    try:
        entry = _ThreadEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        has_entry = bool(kernel32.Thread32First(snapshot, ctypes.byref(entry)))
        while has_entry:
            if int(entry.th32OwnerProcessID) == pid:
                thread = kernel32.OpenThread(
                    _WINDOWS_THREAD_SUSPEND_RESUME,
                    False,
                    entry.th32ThreadID,
                )
                if thread:
                    try:
                        if kernel32.ResumeThread(thread) != 0xFFFFFFFF:
                            resumed += 1
                    finally:
                        kernel32.CloseHandle(thread)
            has_entry = bool(kernel32.Thread32Next(snapshot, ctypes.byref(entry)))
    finally:
        kernel32.CloseHandle(snapshot)
    if resumed == 0:
        raise MemoryBoundaryError("the suspended analysis process had no resumable thread")


def _run_windows_job(
    command: Sequence[str], resources: Any, *, env: Mapping[str, str] | None
) -> int:
    kernel32 = _windows_api()
    process = kernel32.GetCurrentProcess()
    enclosing = wintypes.BOOL()
    if not kernel32.IsProcessInJob(process, None, ctypes.byref(enclosing)):
        error = ctypes.get_last_error()
        raise MemoryBoundaryError(f"IsProcessInJob failed with WinError {error}")

    enclosing_limit_bytes: int | None = None
    if enclosing.value:
        outer = _query_windows_job_info(kernel32, None)
        if outer.BasicLimitInformation.LimitFlags & _WINDOWS_JOB_MEMORY:
            enclosing_limit_bytes = int(outer.JobMemoryLimit)

    requested_bytes = int(resources.max_memory_mb) * _MIB
    effective_bytes = min(
        requested_bytes,
        enclosing_limit_bytes if enclosing_limit_bytes is not None else requested_bytes,
    )
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        error = ctypes.get_last_error()
        raise MemoryBoundaryError(f"CreateJobObjectW failed with WinError {error}")

    child: subprocess.Popen[Any] | None = None
    try:
        info = _JobExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = _WINDOWS_JOB_MEMORY | _WINDOWS_KILL_ON_JOB_CLOSE
        info.JobMemoryLimit = effective_bytes
        if not kernel32.SetInformationJobObject(
            job,
            9,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            error = ctypes.get_last_error()
            raise MemoryBoundaryError(
                f"SetInformationJobObject(job memory) failed with WinError {error}"
            )
        status = MemoryBoundaryStatus(
            contract_version=1,
            backend="windows_job",
            platform="windows",
            enforced=True,
            requested_hard_limit_mb=int(resources.max_memory_mb),
            effective_hard_limit_mb=effective_bytes // _MIB,
            enclosing_hard_limit_mb=(
                enclosing_limit_bytes // _MIB if enclosing_limit_bytes is not None else None
            ),
            strict_required=bool(resources.os_memory_limit_required),
            fallback_used=False,
            detail="aggregate JobMemoryLimit with kill-on-job-close",
            enclosing_job=bool(enclosing.value),
        )
        child = subprocess.Popen(
            command,
            env=_status_environment(env, status),
            creationflags=_WINDOWS_CREATE_SUSPENDED,
        )
        child_handle = wintypes.HANDLE(child._handle)  # type: ignore[attr-defined]
        if not kernel32.AssignProcessToJobObject(job, child_handle):
            error = ctypes.get_last_error()
            nested = " inside an enclosing Job Object" if enclosing.value else ""
            raise MemoryBoundaryError(
                f"AssignProcessToJobObject failed{nested} with WinError {error}"
            )
        _resume_windows_process(kernel32, int(child.pid))
        child_returncode = int(child.wait())
        final_info = _query_windows_job_info(kernel32, job)
        peak = int(final_info.PeakJobMemoryUsed)
        known_memory_codes = {0xC0000017, -1073741801}
        if child_returncode in known_memory_codes or (
            child_returncode != 0 and peak >= int(effective_bytes * 0.98)
        ):
            print(
                "Farkle aggregate Job Object memory limit was reached "
                f"(peak={peak / _MIB:.1f} MiB, limit={effective_bytes / _MIB:.1f} MiB).",
                file=sys.stderr,
                flush=True,
            )
            return MEMORY_LIMIT_EXIT_CODE
        return child_returncode
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        kernel32.CloseHandle(job)


def _unescape_mount_field(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _cgroup_v2_location() -> tuple[Path, Path]:
    try:
        membership = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        mount_lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MemoryBoundaryError(f"cannot inspect cgroup-v2 state: {exc}") from exc
    relative = next(
        (line.split("::", 1)[1] for line in membership if line.startswith("0::")),
        None,
    )
    if relative is None:
        raise MemoryBoundaryError("unified cgroup-v2 membership was not detected")
    for line in mount_lines:
        before, separator, after = line.partition(" - ")
        if not separator or not after.startswith("cgroup2 "):
            continue
        fields = before.split()
        if len(fields) < 5:
            continue
        mount_root = Path(_unescape_mount_field(fields[3]))
        mount_point = Path(_unescape_mount_field(fields[4]))
        relative_path = Path(relative)
        try:
            suffix = relative_path.relative_to(mount_root)
        except ValueError:
            continue
        current = (mount_point / suffix).resolve()
        if not current.is_relative_to(mount_point.resolve()):
            raise MemoryBoundaryError("resolved cgroup path escaped its cgroup-v2 mount")
        return mount_point.resolve(), current
    raise MemoryBoundaryError("a cgroup-v2 filesystem mount was not detected")


def _read_cgroup_limit(path: Path) -> int | None:
    try:
        value = (path / "memory.max").read_text(encoding="ascii").strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise MemoryBoundaryError(f"invalid {path / 'memory.max'} value {value!r}") from exc


def _effective_cgroup_parent_limit(mount: Path, current: Path) -> int | None:
    limits: list[int] = []
    cursor = current
    while True:
        limit = _read_cgroup_limit(cursor)
        if limit is not None:
            limits.append(limit)
        if cursor == mount:
            break
        if not cursor.is_relative_to(mount):
            raise MemoryBoundaryError("current cgroup is outside the unified mount")
        cursor = cursor.parent
    return min(limits) if limits else None


def _read_memory_events(path: Path) -> dict[str, int]:
    try:
        lines = (path / "memory.events").read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise MemoryBoundaryError(f"cannot read cgroup memory events: {exc}") from exc
    events: dict[str, int] = {}
    for line in lines:
        key, value = line.split(maxsplit=1)
        events[key] = int(value)
    return events


def _run_cgroup_v2(command: Sequence[str], resources: Any, *, env: Mapping[str, str] | None) -> int:
    mount, current = _cgroup_v2_location()
    parent_limit = _effective_cgroup_parent_limit(mount, current)
    requested_bytes = int(resources.max_memory_mb) * _MIB
    effective_bytes = min(
        requested_bytes,
        parent_limit if parent_limit is not None else requested_bytes,
    )
    boundary = current / f"farkle-{os.getpid()}-{time.time_ns()}"
    try:
        boundary.mkdir(mode=0o700)
        (boundary / "memory.max").write_text(str(effective_bytes), encoding="ascii")
        oom_group = boundary / "memory.oom.group"
        if oom_group.exists():
            oom_group.write_text("1", encoding="ascii")
        before = _read_memory_events(boundary)
        (boundary / "cgroup.procs").write_text(str(os.getpid()), encoding="ascii")
    except OSError as exc:
        with contextlib.suppress(OSError):
            boundary.rmdir()
        raise MemoryBoundaryError(
            "cgroup v2 is present but the current hierarchy is not delegated/writable: " f"{exc}"
        ) from exc

    status = MemoryBoundaryStatus(
        contract_version=1,
        backend="cgroup_v2",
        platform=_platform_label(),
        enforced=True,
        requested_hard_limit_mb=int(resources.max_memory_mb),
        effective_hard_limit_mb=effective_bytes // _MIB,
        enclosing_hard_limit_mb=(parent_limit // _MIB if parent_limit is not None else None),
        strict_required=bool(resources.os_memory_limit_required),
        fallback_used=False,
        detail="aggregate cgroup-v2 memory.max with descendant inheritance",
        boundary_path=str(boundary),
    )
    child_returncode = SETUP_FAILURE_EXIT_CODE
    memory_failure = False
    after: dict[str, int] = {}
    try:
        child_returncode = subprocess.run(
            command,
            env=_status_environment(env, status),
            check=False,
        ).returncode
        after = _read_memory_events(boundary)
        memory_failure = any(
            after.get(key, 0) > before.get(key, 0) for key in ("oom", "oom_kill", "oom_group_kill")
        )
    finally:
        try:
            (current / "cgroup.procs").write_text(str(os.getpid()), encoding="ascii")
            remaining = (boundary / "cgroup.procs").read_text(encoding="ascii").split()
            if remaining:
                kill_path = boundary / "cgroup.kill"
                if kill_path.exists():
                    kill_path.write_text("1", encoding="ascii")
            if not memory_failure:
                boundary.rmdir()
        except OSError as exc:
            print(
                f"WARNING: cgroup cleanup retained {boundary} for diagnosis: {exc}",
                file=sys.stderr,
                flush=True,
            )
    if memory_failure:
        print(
            "Farkle aggregate cgroup-v2 memory limit was reached; "
            f"events={json.dumps(after, sort_keys=True)}, cgroup={boundary}",
            file=sys.stderr,
            flush=True,
        )
        return MEMORY_LIMIT_EXIT_CODE
    if child_returncode == -int(getattr(signal, "SIGKILL", 9)):
        return MEMORY_LIMIT_EXIT_CODE
    return int(child_returncode)


def _supervisor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a command under a Farkle memory boundary")
    parser.add_argument("--limit-mb", required=True, type=int)
    parser.add_argument("--permissive", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Internal subprocess/canary launcher used by targeted boundary tests."""

    args = _supervisor_parser().parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("a command is required after --")
    resources = SimpleNamespace(
        max_memory_mb=int(args.limit_mb),
        os_memory_limit_enabled=True,
        os_memory_limit_required=not args.permissive,
        allow_unenforced_memory_fallback=bool(args.permissive),
    )
    try:
        return supervise_process(command, resources)
    except MemoryBoundaryError as exc:
        print(f"Farkle OS memory boundary setup failed closed: {exc}", file=sys.stderr)
        return SETUP_FAILURE_EXIT_CODE


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())


__all__ = [
    "BOUNDARY_STATUS_ENV",
    "MEMORY_LIMIT_EXIT_CODE",
    "SETUP_FAILURE_EXIT_CODE",
    "MemoryBoundaryError",
    "MemoryBoundaryStatus",
    "current_memory_boundary",
    "memory_boundary_provenance",
    "supervise_module_if_needed",
    "supervise_process",
]
