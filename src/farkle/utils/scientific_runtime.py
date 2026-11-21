"""Observe the installed numerical build and the effective native execution settings."""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import struct
import sys
import sysconfig
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib import import_module, metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pyarrow as pa
from packaging.requirements import Requirement
from threadpoolctl import threadpool_info, threadpool_limits

from farkle.utils.authenticated_contract import identity_sha256


@dataclass
class _BuildObservation:
    installed: dict[str, str] = field(default_factory=dict)
    host: dict[frozenset[str], str] = field(default_factory=dict)


_OBSERVATION: ContextVar[_BuildObservation | None] = ContextVar(
    "scientific_build_observation", default=None
)


@contextmanager
def numerical_observation() -> Iterator[None]:
    """Share producer-entry measurements only within this action, never across separate calls."""
    if _OBSERVATION.get() is not None:
        yield
        return
    token = _OBSERVATION.set(_BuildObservation())
    try:
        yield
    finally:
        _OBSERVATION.reset(token)


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _installed_build(name: str, seen: dict[str, str]) -> str:
    """Observe the installed build for this interpreter invocation, including required packages."""
    if name in seen:
        return seen[name]
    dist = metadata.distribution(name)
    files = dist.files
    if not files:
        raise ValueError(f"Missing installed file inventory for numerical dependency {name}")
    hashes = {}
    for file in files:
        # Console wrappers, test suites and generated interpreter caches are outside the numerical build.
        if ".." in file.parts or "__pycache__" in file.parts or "tests" in file.parts:
            continue
        if file.suffix == ".pyc" or any(part.endswith(".dist-info") for part in file.parts):
            continue
        path = Path(str(dist.locate_file(file)))
        # Installed wheel bytes describe the computation only when Python resolves
        # its top-level modules to that wheel (including single-file packages).
        module = None
        if len(file.parts) == 2 and file.name == "__init__.py":
            module = file.parts[0]
        elif len(file.parts) == 1 and file.suffix == ".py":
            module = file.stem
        if module and module.isidentifier():
            spec = find_spec(module)
            if not spec or not spec.origin or Path(spec.origin).resolve() != path.resolve():
                raise ValueError(f"Numerical module {module} is outside the installed {name} build")
        if file.suffix == ".pth" and file.name.startswith("__editable__"):
            raise ValueError(f"Editable numerical dependency {name} has no fixed wheel inventory")
        hashes[file.as_posix()] = _file_digest(path)
    if not hashes:
        raise ValueError(f"Cannot identify installed implementation for {name}")
    required = {}
    for entry in dist.requires or ():
        req = Requirement(entry)
        if req.marker is None or req.marker.evaluate({"extra": ""}):
            required[req.name] = _installed_build(req.name, seen)
    result = identity_sha256(
        {"distribution": name, "version": dist.version, "files": hashes, "requires": required}
    )
    seen[name] = result
    return result


def _windows_imports(roots: list[Path]) -> set[Path]:
    """Follow the loader's import tables, including API-set -> host DLL resolution."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel.GetModuleHandleW.restype = wintypes.HMODULE
    kernel.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    directory = ctypes.WinDLL("dbghelp").ImageDirectoryEntryToData
    directory.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ubyte,
        ctypes.c_ushort,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    directory.restype = ctypes.c_void_p
    pending = [str(path) for path in roots]
    files: set[Path] = set()
    visited = set()
    while pending:
        name = pending.pop()
        handle = kernel.GetModuleHandleW(name)
        if not handle:
            raise ValueError(f"Cannot resolve loaded native dependency {name}")
        if handle in visited:
            continue
        visited.add(handle)
        path = ctypes.create_unicode_buffer(32768)
        if not 0 < kernel.GetModuleFileNameW(handle, path, len(path)) < len(path):
            raise ValueError(f"Cannot identify loaded native image {name}")
        files.add(Path(path.value))
        size = ctypes.c_ulong()
        # IMAGE_DIRECTORY_ENTRY_IMPORT, mapped as an image. Each validated PE
        # import descriptor is five DWORDs; its fourth is the DLL name RVA.
        address = directory(handle, 1, 1, ctypes.byref(size))
        if address:
            for _, _, _, rva, _ in struct.iter_unpack(
                "<5I", ctypes.string_at(address, size.value - size.value % 20)
            ):
                if not rva:
                    break
                pending.append(ctypes.string_at(handle + rva).decode("ascii"))
    return files


def _host_build(names: frozenset[str]) -> str:
    """Fingerprint the interpreter, numerical extensions and their host runtime."""
    modules = [
        "numpy._core._multiarray_umath",
        "scipy.linalg._fblas",
        "pyarrow.lib",
        "pandas._libs.algos",
        "numba._helperlib",
    ]
    if "scikit-learn" in names:
        modules.append("sklearn.utils._openmp_helpers")
    roots = [Path(import_module(module).__file__ or "") for module in modules]
    llvm = import_module("llvmlite.binding")
    mapped = [Path(image.path) for image in psutil.Process().memory_maps()]
    roots.extend(
        path
        for path in mapped
        if "llvmlite" in path.name.lower() and path.suffix in {".dll", ".so", ".dylib"}
    )
    roots.extend(
        Path(library["filepath"])
        for library in threadpool_info()
        if "sklearn" not in library["filepath"].lower() or "scikit-learn" in names
    )
    if os.name == "nt":
        images = _windows_imports(roots)
    else:
        # Overbind additional mapped runtime libraries on other hosts.
        images = {
            p
            for p in mapped
            if p.is_absolute() and (p.suffix in {".so", ".dylib"} or ".so." in p.name)
        }
    if not images:
        raise ValueError("Cannot identify loaded interpreter and native host images")
    return identity_sha256(
        {
            "interpreter": _file_digest(Path(sys.executable)),
            "abi": sysconfig.get_config_var("SOABI"),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "host_images": sorted((p.name.lower(), _file_digest(p)) for p in images),
            "llvm_cpu": str(llvm.get_host_cpu_name()),
            "llvm_features": llvm.get_host_cpu_features().flatten(),
        }
    )


def numerical_environment(
    names: frozenset[str], *, native_threads: int | None = None, observe_worker: bool = False
) -> tuple[str, dict[str, str]]:
    """Return build evidence and native controls for the scoped numerical methods.

    A managed worker has an explicit thread contract. An ambient producer binds
    the loaded libraries' actual thread counts. Data/sidecar verification is independent.
    """
    # Forked children inherit context variables, but must establish their own evidence.
    observation = (None if observe_worker else _OBSERVATION.get()) or _BuildObservation()
    if names not in observation.host:
        observation.host[names] = _host_build(names)
    host = observation.host[names]

    # Probe the actual controller state under the same limiter as a managed
    # producer. At worker initialization compare its live state without applying a probe.
    if native_threads is None or observe_worker:
        native_libraries = threadpool_info()
    else:
        with threadpool_limits(limits=native_threads):
            native_libraries = threadpool_info()
    libraries = []
    for library in native_libraries:
        path = Path(library["filepath"])
        # A parent that previously ran an optional sklearn stage can have its
        # private OpenMP runtime loaded while producing unrelated outputs.
        if "sklearn" in path.as_posix().lower() and "scikit-learn" not in names:
            continue
        info = {k: v for k, v in library.items() if k != "filepath"}
        info["image"] = path.name
        if native_threads is not None and info["num_threads"] != native_threads:
            raise ValueError(
                f"Native library {path.name} did not establish {native_threads} threads"
            )
        libraries.append(info)
    np_impl = import_module("numpy._core._multiarray_umath")
    arrow = pa.runtime_info()
    optional = ["numexpr", "bottleneck"]
    if "trueskill" in names:
        optional.append("mpmath")
    optional_builds = {
        package: (
            _installed_build(package, observation.installed)
            if find_spec(package) is not None
            else None
        )
        for package in optional
    }
    libc = ctypes.CDLL("ucrtbase" if os.name == "nt" else None)
    controls: dict[str, Any] = {
        "host": host,
        "floating_point_rounding": libc.fegetround(),
        "optional_builds": optional_builds,
        "numpy_cpu_features": np_impl.__cpu_features__,
        "numpy_error_policy": np.geterr(),
        "arrow_simd": [arrow.simd_level, arrow.detected_simd_level],
        "native_libraries": sorted(libraries, key=lambda p: str(p["image"])),
        "dispatch_settings": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("NPY_", "NUMBA_", "MKL_CBWR", "ARROW_USER_SIMD_LEVEL"))
        },
    }
    return identity_sha256(controls), {
        name: _installed_build(name, observation.installed) for name in sorted(names)
    }
