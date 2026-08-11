# src/farkle/utils/parallel.py
"""Parallel execution helpers used by simulations.

Small, testable utilities for seeding workers and mapping work with a
ProcessPoolExecutor. Keep simulation-specific logic outside utils.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import threading
import time
import weakref
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing.context import BaseContext
from typing import Any, Mapping

import psutil
from threadpoolctl import ThreadpoolController

_NATIVE_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
_NATIVE_LIMIT_LOCK = threading.Lock()
_NATIVE_LIMITER: Any | None = None
_NATIVE_LIMITER_CAP: int | None = None
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageParallelPolicy:
    """Resolved parallel budget for a specific stage."""

    total_cores: int
    process_workers: int
    python_threads: int
    arrow_threads: int
    native_threads_per_process: int
    configured_cpu_budget: int = 1
    cpu_worker_cap: int = 1
    memory_worker_cap: int = 1
    estimated_worker_memory_mb: int = 0
    scheduler_memory_budget_mb: int = 0
    parent_process_memory_mb: int = 0
    concurrent_roots: int = 1


class ResourceSafetyError(RuntimeError):
    """Raised before more work is scheduled outside the safe resource envelope."""


@dataclass(frozen=True)
class ResolvedResourcePolicy:
    """Detected and resolved machine-wide execution policy."""

    detected_logical_cpus: int
    detected_total_memory_mb: int
    detected_available_memory_mb: int
    requested_logical_cpu_budget: int
    resolved_logical_cpu_budget: int
    scheduler_memory_budget_mb: int
    process_tree_warning_threshold_mb: int
    aggregate_memory_hard_limit_mb: int
    minimum_system_available_memory_mb: int
    parent_process_memory_mb: int
    native_threads_per_worker: int

    def as_metadata(self, *, effective_hard_limit_mb: int | None = None) -> dict[str, Any]:
        """Return exact requested, resolved, and effective execution provenance."""

        return {
            "requested": {
                "logical_cpu_budget": self.requested_logical_cpu_budget,
                "scheduler_memory_budget_mb": self.scheduler_memory_budget_mb,
                "process_tree_warning_threshold_mb": self.process_tree_warning_threshold_mb,
                "aggregate_memory_hard_limit_mb": self.aggregate_memory_hard_limit_mb,
                "minimum_system_available_memory_mb": self.minimum_system_available_memory_mb,
                "parent_process_memory_mb": self.parent_process_memory_mb,
                "native_threads_per_worker": self.native_threads_per_worker,
            },
            "resolved": {
                "detected_logical_cpus": self.detected_logical_cpus,
                "detected_total_memory_mb": self.detected_total_memory_mb,
                "detected_available_memory_mb": self.detected_available_memory_mb,
                "logical_cpu_budget": self.resolved_logical_cpu_budget,
                "scheduler_memory_budget_mb": self.scheduler_memory_budget_mb,
                "process_tree_warning_threshold_mb": self.process_tree_warning_threshold_mb,
                "aggregate_memory_hard_limit_mb": self.aggregate_memory_hard_limit_mb,
                "minimum_system_available_memory_mb": self.minimum_system_available_memory_mb,
                "parent_process_memory_mb": self.parent_process_memory_mb,
                "native_threads_per_worker": self.native_threads_per_worker,
            },
            "effective": {
                "logical_cpu_budget": self.resolved_logical_cpu_budget,
                "scheduler_memory_budget_mb": self.scheduler_memory_budget_mb,
                "process_tree_warning_threshold_mb": self.process_tree_warning_threshold_mb,
                "aggregate_memory_hard_limit_mb": effective_hard_limit_mb,
                "minimum_system_available_memory_mb": self.minimum_system_available_memory_mb,
                "parent_process_memory_mb": self.parent_process_memory_mb,
                "native_threads_per_worker": self.native_threads_per_worker,
            },
        }


def resolve_resource_policy(
    resources: Any,
    *,
    logical_cpu_count: int | None = None,
    total_memory_mb: int | None = None,
    available_memory_mb: int | None = None,
    require_available_reserve: bool = False,
) -> ResolvedResourcePolicy:
    """Resolve and validate explicit resource controls against one detected host."""

    detected_cpus = max(1, int(logical_cpu_count or os.cpu_count() or 1))
    memory = psutil.virtual_memory()
    detected_total_mb = (
        int(memory.total // (1024 * 1024)) if total_memory_mb is None else int(total_memory_mb)
    )
    detected_available_mb = (
        int(memory.available // (1024 * 1024))
        if available_memory_mb is None
        else int(available_memory_mb)
    )
    requested_cpu = int(resources.logical_cpu_budget)
    resolved_cpu = detected_cpus if requested_cpu == 0 else requested_cpu
    if resolved_cpu > detected_cpus:
        raise ResourceSafetyError(
            "configured logical CPU budget is impossible on this machine: "
            f"requested {resolved_cpu}, detected {detected_cpus}"
        )
    hard_limit_mb = int(resources.aggregate_memory_hard_limit_mb)
    system_reserve_mb = int(resources.minimum_system_available_memory_mb)
    if hard_limit_mb + system_reserve_mb > detected_total_mb:
        raise ResourceSafetyError(
            "configured aggregate memory hard limit plus system-available reserve exceeds "
            f"detected memory: {hard_limit_mb} + {system_reserve_mb} > "
            f"{detected_total_mb} MiB"
        )
    if require_available_reserve and detected_available_mb < system_reserve_mb:
        raise ResourceSafetyError(
            "configured system-available memory reserve is not currently available: "
            f"{detected_available_mb} MiB available < {system_reserve_mb} MiB required"
        )
    return ResolvedResourcePolicy(
        detected_logical_cpus=detected_cpus,
        detected_total_memory_mb=detected_total_mb,
        detected_available_memory_mb=detected_available_mb,
        requested_logical_cpu_budget=requested_cpu,
        resolved_logical_cpu_budget=resolved_cpu,
        scheduler_memory_budget_mb=int(resources.scheduler_memory_budget_mb),
        process_tree_warning_threshold_mb=int(resources.process_tree_warning_threshold_mb),
        aggregate_memory_hard_limit_mb=hard_limit_mb,
        minimum_system_available_memory_mb=system_reserve_mb,
        parent_process_memory_mb=int(resources.parent_process_memory_mb),
        native_threads_per_worker=int(resources.native_threads_per_worker),
    )


@dataclass(slots=True, weakref_slot=True)
class ProcessTreeMemoryGuard:
    """Continuously sample current-process plus descendant resident memory.

    A threshold crossing is sticky for the lifetime of the guard.  This matters
    because a short-lived worker allocation can disappear before the next
    scheduling boundary; a later low sample must never make that execution
    eligible for successful publication.
    """

    rss_abort_mb: int
    rss_warning_mb: int | None = None
    minimum_system_available_memory_mb: int = 0
    sample_interval_seconds: float = 0.25
    pid: int | None = None
    last_rss_bytes: int = 0
    peak_rss_bytes: int = 0
    last_native_threads: int = 0
    peak_native_threads: int = 0
    _last_sample_at: float = 0.0
    tripped_rss_bytes: int = 0
    warned_rss_bytes: int = 0
    warning_emitted: bool = False
    monitoring_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _monitor_started: bool = field(default=False, init=False, repr=False)

    def _record_sample(self, rss: int, sampled_at: float, native_threads: int = 0) -> None:
        with self._lock:
            self.last_rss_bytes = rss
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.last_native_threads = native_threads
            self.peak_native_threads = max(self.peak_native_threads, native_threads)
            self._last_sample_at = sampled_at
            if rss >= self.rss_abort_mb * 1024 * 1024:
                self.tripped_rss_bytes = max(self.tripped_rss_bytes, rss)
            if self.rss_warning_mb is not None and rss >= self.rss_warning_mb * 1024 * 1024:
                self.warned_rss_bytes = max(self.warned_rss_bytes, rss)

    def _ensure_monitor(self) -> None:
        with self._lock:
            if self._monitor_started:
                return
            self._monitor_started = True
        guard_ref = weakref.ref(self)
        interval = float(self.sample_interval_seconds)

        def monitor() -> None:
            while True:
                guard = guard_ref()
                if guard is None:
                    return
                try:
                    rss = process_tree_rss_bytes(guard.pid)
                    native_threads = process_tree_native_thread_count(guard.pid)
                    guard._record_sample(rss, time.monotonic(), native_threads)
                except Exception as exc:  # noqa: BLE001 - monitoring must fail closed
                    with guard._lock:
                        guard.monitoring_error = f"{type(exc).__name__}: {exc}"
                    return
                finally:
                    # Do not retain the guard across the wait; this lets a
                    # completed stage release its daemon sampler naturally.
                    del guard
                time.sleep(interval)

        threading.Thread(
            target=monitor,
            name="farkle-process-tree-rss-guard",
            daemon=True,
        ).start()

    def sample(self, *, force: bool = False) -> int:
        """Return sampled process-tree RSS, reusing only a recent safe sample."""

        self._ensure_monitor()
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_sample_at < self.sample_interval_seconds:
                return self.last_rss_bytes
        rss = process_tree_rss_bytes(self.pid)
        native_threads = process_tree_native_thread_count(self.pid)
        self._record_sample(rss, now, native_threads)
        return rss

    def check_before_schedule(self, *, force: bool = False) -> int:
        """Warn at high water and fail closed at explicit process/host thresholds."""

        rss = self.sample(force=force)
        with self._lock:
            tripped = self.tripped_rss_bytes
            warned = self.warned_rss_bytes
            warning_emitted = self.warning_emitted
            monitoring_error = self.monitoring_error
            warning_mb = self.rss_warning_mb
        if warned and warning_mb is not None and not warning_emitted:
            LOGGER.warning(
                "process-tree RSS exceeded the configured high-water threshold; continuing "
                "until the explicit aggregate hard limit",
                extra={
                    "stage": "resource_safety",
                    "rss_mb": warned / (1024 * 1024),
                    "process_tree_warning_threshold_mb": warning_mb,
                    "aggregate_memory_hard_limit_mb": self.rss_abort_mb,
                },
            )
            with self._lock:
                self.warning_emitted = True
        if monitoring_error is not None:
            raise ResourceSafetyError(
                f"process-tree RSS monitoring failed closed: {monitoring_error}"
            )
        if tripped:
            raise ResourceSafetyError(
                "process-tree RSS safety threshold crossed: "
                f"{tripped / (1024 * 1024):.1f} MiB >= {self.rss_abort_mb} MiB"
            )
        if self.minimum_system_available_memory_mb:
            available_mb = int(psutil.virtual_memory().available // (1024 * 1024))
            if available_mb < self.minimum_system_available_memory_mb:
                raise ResourceSafetyError(
                    "system-available memory reserve would be violated before scheduling: "
                    f"{available_mb} MiB available < "
                    f"{self.minimum_system_available_memory_mb} MiB required"
                )
        return rss


def process_tree_rss_bytes(pid: int | None = None) -> int:
    """Return RSS for one process and all recursively reachable children."""

    root = psutil.Process(os.getpid() if pid is None else pid)
    processes = [root, *root.children(recursive=True)]
    rss = 0
    seen: set[int] = set()
    for process in processes:
        if process.pid in seen:
            continue
        seen.add(process.pid)
        try:
            rss += int(process.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss


def process_tree_native_thread_count(pid: int | None = None) -> int:
    """Return the aggregate native-thread count for a process tree."""

    root = psutil.Process(os.getpid() if pid is None else pid)
    processes = [root, *root.children(recursive=True)]
    threads = 0
    seen: set[int] = set()
    for process in processes:
        if process.pid in seen:
            continue
        seen.add(process.pid)
        try:
            threads += int(process.num_threads())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return threads


def _stage_resource_value(values: Mapping[str, int], stage: str, fallback: str) -> int:
    if stage in values:
        return int(values[stage])
    if fallback in values:
        return int(values[fallback])
    raise ValueError(f"resources do not define a budget for stage {stage!r}")


@dataclass(frozen=True)
class ParallelNestingContext:
    """Parallel context inherited by nested work units."""

    active_process_executor: bool = False
    parent_process_workers: int = 1
    total_cores: int | None = None


def resolve_mp_context(mp_start_method: str | None) -> BaseContext | None:
    """Resolve a multiprocessing context from a configured start-method name."""
    if mp_start_method is None:
        return None
    method = mp_start_method.strip().lower()
    if not method or method == "default":
        return None
    available = set(mp.get_all_start_methods())
    if method not in available:
        available_text = ", ".join(sorted(available))
        raise ValueError(
            f"Unsupported multiprocessing start method {mp_start_method!r}. "
            f"Expected one of: default, {available_text}."
        )
    return mp.get_context(method)


def normalize_n_jobs(
    value: int | None,
    cpu_count: int | None = None,
    *,
    default: int = 1,
) -> int:
    """Normalize ``n_jobs`` with explicit deterministic semantics.

    ``0`` resolves to all detected cores. ``None`` resolves to ``default``.
    """
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    cpu_count = max(1, int(cpu_count))
    if value is None:
        return max(1, int(default))
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"n_jobs must be >= 0 or None, got {value!r}")
    if resolved == 0:
        return cpu_count
    return max(1, resolved)


def resolve_stage_parallel_policy(
    stage: str,
    cfg: Any,
    outer_context: ParallelNestingContext | Mapping[str, Any] | None = None,
    *,
    n_jobs_override: int | None = None,
    resources: Any | None = None,
    concurrent_roots: int = 1,
) -> StageParallelPolicy:
    """Resolve per-stage parallel budgets with optional nesting awareness."""
    detected_cores = os.cpu_count() or 1
    total_cores = detected_cores
    context_total_cores: int | None = None
    active_process_executor = False
    parent_workers = 1
    if outer_context is not None:
        if isinstance(outer_context, ParallelNestingContext):
            active_process_executor = bool(outer_context.active_process_executor)
            parent_workers = max(1, int(outer_context.parent_process_workers))
            context_total_cores = outer_context.total_cores
        else:
            active_process_executor = bool(outer_context.get("active_process_executor", False))
            parent_workers = max(1, int(outer_context.get("parent_process_workers", 1)))
            total_value = outer_context.get("total_cores")
            context_total_cores = int(total_value) if total_value is not None else None

    if context_total_cores is not None:
        total_cores = max(1, context_total_cores)

    if os.environ.get("FARKLE_PROCESS_POOL_ACTIVE") == "1":
        active_process_executor = True

    concurrent_roots = max(1, int(concurrent_roots))
    if resources is not None:
        machine_policy = resolve_resource_policy(resources, logical_cpu_count=detected_cores)
        total_cores = machine_policy.resolved_logical_cpu_budget

    requested_n_jobs = (
        n_jobs_override if n_jobs_override is not None else getattr(cfg, "n_jobs", None)
    )
    process_workers = normalize_n_jobs(requested_n_jobs, cpu_count=total_cores, default=1)

    configured_cpu_budget = max(1, total_cores // concurrent_roots)
    cpu_worker_cap = configured_cpu_budget
    memory_worker_cap = process_workers
    estimated_worker_memory_mb = 0
    scheduler_memory_budget_mb = 0
    parent_process_memory_mb = 0
    if resources is not None:
        native_threads = max(1, int(resources.native_threads_per_worker))
        cpu_worker_cap = configured_cpu_budget // native_threads
        if cpu_worker_cap < 1:
            raise ResourceSafetyError(
                "configured CPU budget cannot support one worker at "
                f"{native_threads} native threads per worker"
            )
        fallback = "head2head" if "h2h" in stage else "analysis"
        estimated_worker_memory_mb = _stage_resource_value(
            resources.estimated_worker_memory_mb,
            stage,
            fallback,
        )
        scheduler_memory_budget_mb = int(resources.scheduler_memory_budget_mb)
        parent_process_memory_mb = int(resources.parent_process_memory_mb)
        available_mb = scheduler_memory_budget_mb - parent_process_memory_mb
        memory_worker_cap = available_mb // (estimated_worker_memory_mb * concurrent_roots)
        if memory_worker_cap < 1:
            raise ResourceSafetyError(
                f"stage {stage!r} estimated worker memory ({estimated_worker_memory_mb} MiB) "
                "exceeds its scheduler memory share "
                f"({available_mb // concurrent_roots} MiB after parent allowance)"
            )
        process_workers = min(process_workers, cpu_worker_cap, memory_worker_cap)
    if active_process_executor:
        process_workers = 1

    available_native_threads = (
        max(1, total_cores // parent_workers) if active_process_executor else total_cores
    )
    native_threads_per_process = (
        max(1, int(resources.native_threads_per_worker))
        if resources is not None
        else max(1, available_native_threads // max(1, process_workers))
    )
    python_threads = native_threads_per_process

    requested_arrow_threads = getattr(cfg, "arrow_threads", None)
    if requested_arrow_threads is None:
        arrow_threads = 1 if active_process_executor else native_threads_per_process
    else:
        requested_arrow_threads_i = int(requested_arrow_threads)
        if requested_arrow_threads_i < 0:
            raise ValueError(f"arrow_threads must be >= 0 or None, got {requested_arrow_threads!r}")
        if requested_arrow_threads_i == 0:
            arrow_threads = native_threads_per_process
        else:
            arrow_threads = max(1, requested_arrow_threads_i)

    return StageParallelPolicy(
        total_cores=total_cores,
        process_workers=process_workers,
        python_threads=python_threads,
        arrow_threads=arrow_threads,
        native_threads_per_process=native_threads_per_process,
        configured_cpu_budget=configured_cpu_budget,
        cpu_worker_cap=cpu_worker_cap,
        memory_worker_cap=memory_worker_cap,
        estimated_worker_memory_mb=estimated_worker_memory_mb,
        scheduler_memory_budget_mb=scheduler_memory_budget_mb,
        parent_process_memory_mb=parent_process_memory_mb,
        concurrent_roots=concurrent_roots,
    )


def apply_native_thread_limits(policy: StageParallelPolicy) -> None:
    """Apply environment-based native thread caps for the current process."""
    global _NATIVE_LIMITER, _NATIVE_LIMITER_CAP
    thread_cap = str(max(1, int(policy.native_threads_per_process)))
    for env_var in _NATIVE_THREAD_ENV_VARS:
        os.environ[env_var] = thread_cap
    os.environ["PYARROW_NUM_THREADS"] = str(max(1, int(policy.arrow_threads)))
    with _NATIVE_LIMIT_LOCK:
        resolved_cap = int(thread_cap)
        if resolved_cap != _NATIVE_LIMITER_CAP:
            _NATIVE_LIMITER = ThreadpoolController().limit(limits=resolved_cap)
            _NATIVE_LIMITER_CAP = resolved_cap


def _initialize_process_worker(initializer, initargs: tuple[Any, ...]) -> None:
    """Mark executor children so nested stage policies collapse to one process."""

    os.environ["FARKLE_PROCESS_POOL_ACTIVE"] = "1"
    if initializer is not None:
        initializer(*initargs)


def process_map(
    fn,
    items,
    *,
    n_jobs=None,
    initializer=None,
    initargs=None,
    window=0,
    mp_context: BaseContext | None = None,
    memory_guard: ProcessTreeMemoryGuard | None = None,
):
    """Map ``fn`` across ``items`` with optional multiprocessing support."""
    if initargs is None:
        initargs = ()
    resolved_jobs = normalize_n_jobs(n_jobs, default=1)
    if resolved_jobs == 1:
        # Single-process path: still run initializer so modules relying on
        # per-process globals (e.g., run_tournament._STATE) are set up.
        if initializer is not None:
            initializer(*tuple(initargs))
        for it in items:
            if memory_guard is not None:
                memory_guard.check_before_schedule()
            yield fn(it)
        return
    if window <= 0:
        window = resolved_jobs * 4

    with ProcessPoolExecutor(
        max_workers=resolved_jobs,
        initializer=_initialize_process_worker,
        initargs=(initializer, tuple(initargs)),
        mp_context=mp_context,
    ) as executor:
        it = iter(items)
        futs = []
        # prefill the window
        for _ in range(window):
            try:
                if memory_guard is not None:
                    memory_guard.check_before_schedule()
                futs.append(executor.submit(fn, next(it)))
            except StopIteration:
                break
        while futs:
            done = next(as_completed(futs))
            futs.remove(done)
            yield done.result()
            with contextlib.suppress(StopIteration):
                if memory_guard is not None:
                    memory_guard.check_before_schedule()
                futs.append(executor.submit(fn, next(it)))


__all__ = [
    "ParallelNestingContext",
    "ProcessTreeMemoryGuard",
    "ResourceSafetyError",
    "ResolvedResourcePolicy",
    "StageParallelPolicy",
    "apply_native_thread_limits",
    "normalize_n_jobs",
    "process_map",
    "process_tree_native_thread_count",
    "process_tree_rss_bytes",
    "resolve_mp_context",
    "resolve_resource_policy",
    "resolve_stage_parallel_policy",
]
