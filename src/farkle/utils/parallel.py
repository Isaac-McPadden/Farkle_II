# src/farkle/utils/parallel.py
"""Parallel execution helpers used by simulations.

Small, testable utilities for seeding workers and mapping work with a
ProcessPoolExecutor. Keep simulation-specific logic outside utils.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any, Mapping

_NATIVE_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class StageParallelPolicy:
    """Resolved parallel budget for a specific stage."""

    total_cores: int
    process_workers: int
    python_threads: int
    arrow_threads: int
    native_threads_per_process: int


@dataclass(frozen=True)
class ParallelNestingContext:
    """Parallel context inherited by nested work units."""

    active_process_pool: bool = False
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
) -> StageParallelPolicy:
    """Resolve per-stage parallel budgets with optional nesting awareness."""
    del stage  # stage remains part of API for future stage-specific rules.

    total_cores = os.cpu_count() or 1
    context_total_cores: int | None = None
    active_process_pool = False
    parent_workers = 1
    if outer_context is not None:
        if isinstance(outer_context, ParallelNestingContext):
            active_process_pool = bool(outer_context.active_process_pool)
            parent_workers = max(1, int(outer_context.parent_process_workers))
            context_total_cores = outer_context.total_cores
        else:
            active_process_pool = bool(outer_context.get("active_process_pool", False))
            parent_workers = max(1, int(outer_context.get("parent_process_workers", 1)))
            total_value = outer_context.get("total_cores")
            context_total_cores = int(total_value) if total_value is not None else None

    if context_total_cores is not None:
        total_cores = max(1, context_total_cores)

    requested_n_jobs = n_jobs_override if n_jobs_override is not None else getattr(cfg, "n_jobs", None)
    process_workers = normalize_n_jobs(requested_n_jobs, cpu_count=total_cores, default=1)
    if active_process_pool:
        process_workers = 1

    available_native_threads = (
        max(1, total_cores // parent_workers) if active_process_pool else total_cores
    )
    native_threads_per_process = max(1, available_native_threads // max(1, process_workers))
    python_threads = native_threads_per_process

    requested_arrow_threads = getattr(cfg, "arrow_threads", None)
    if requested_arrow_threads is None:
        arrow_threads = 1 if active_process_pool else native_threads_per_process
    else:
        requested_arrow_threads_i = int(requested_arrow_threads)
        if requested_arrow_threads_i < 0:
            raise ValueError(
                f"arrow_threads must be >= 0 or None, got {requested_arrow_threads!r}"
            )
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
    )


def apply_native_thread_limits(policy: StageParallelPolicy) -> None:
    """Apply environment-based native thread caps for the current process."""
    thread_cap = str(max(1, int(policy.native_threads_per_process)))
    for env_var in _NATIVE_THREAD_ENV_VARS:
        os.environ[env_var] = thread_cap
    os.environ["PYARROW_NUM_THREADS"] = str(max(1, int(policy.arrow_threads)))


def process_map(
    fn,
    items,
    *,
    n_jobs=None,
    initializer=None,
    initargs=None,
    window=0,
    mp_context: BaseContext | None = None,
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
            yield fn(it)
        return
    if window <= 0:
        window = resolved_jobs * 4

    with ProcessPoolExecutor(
        max_workers=resolved_jobs,
        initializer=initializer,
        initargs=tuple(initargs),
        mp_context=mp_context,
    ) as pool:
        it = iter(items)
        futs = []
        # prefill the window
        for _ in range(window):
            try:
                futs.append(pool.submit(fn, next(it)))
            except StopIteration:
                break
        while futs:
            done = next(as_completed(futs))
            futs.remove(done)
            yield done.result()
            with contextlib.suppress(StopIteration):
                futs.append(pool.submit(fn, next(it)))


__all__ = [
    "ParallelNestingContext",
    "StageParallelPolicy",
    "apply_native_thread_limits",
    "normalize_n_jobs",
    "process_map",
    "resolve_mp_context",
    "resolve_stage_parallel_policy",
]
