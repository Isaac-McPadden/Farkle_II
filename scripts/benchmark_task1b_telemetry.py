"""Bounded Task 1B scheduler/worker-telemetry overhead measurement.

This is intentionally not a pipeline benchmark.  It alternates identical
two-worker deterministic RNG-sort and H2H-like CPU chunks with and without the
Task 1B operational adapters, then reports medians and canonical result hashes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from multiprocessing.context import BaseContext

import numpy as np

from farkle.utils.parallel import process_map
from farkle.utils.telemetry import (
    SupervisorHeartbeatRecorder,
    install_worker_progress_endpoint,
    report_worker_progress,
)


def _rng_sort_unit(value: int) -> tuple[str, bool]:
    generator = np.random.Generator(np.random.PCG64(90_000 + value))
    values = generator.integers(0, 2**63, size=700_000, dtype=np.uint64)
    values.sort(kind="stable")
    sent = report_worker_progress(
        "rng_bounded_merge",
        counters={
            "spill_runs_created": 1,
            "spill_bytes_written": int(values.nbytes),
            "merge_passes_completed": 1,
        },
    )
    return hashlib.sha256(values.tobytes()).hexdigest(), sent


def _h2h_chunk_unit(value: int) -> tuple[str, bool]:
    state = (value + 1) * 0x9E3779B97F4A7C15
    digest = hashlib.sha256()
    for index in range(300_000):
        state = (state + 0x9E3779B97F4A7C15 + index) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 30
        state = (state * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 27
        digest.update((state & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))
    return digest.hexdigest(), False


@dataclass(frozen=True)
class _Measurement:
    wall_seconds: float
    parent_cpu_seconds: float
    digest: str
    sent_messages: int
    scheduler_events: int


def _run_fixture(
    worker: Callable[[int], tuple[str, bool]],
    *,
    telemetry_enabled: bool,
    context: BaseContext,
) -> _Measurement:
    logger = logging.getLogger("farkle.task1b.overhead")
    logger.handlers[:] = [logging.NullHandler()]
    logger.propagate = False
    recorder = SupervisorHeartbeatRecorder(
        logger,
        run="bounded_benchmark",
        interval_seconds=45.0,
    )
    scope = recorder.begin_scope(
        "bounded_fixture",
        run="bounded_benchmark",
        stage="rng" if worker is _rng_sort_unit else "h2h_execute",
        phase="bounded_work",
    )
    endpoint = (
        recorder.create_worker_progress_endpoint(scope.scope, mp_context=context)
        if telemetry_enabled and worker is _rng_sort_unit
        else None
    )
    event_count = 0

    def progress(event: Mapping[str, object]) -> None:
        nonlocal event_count
        event_count += 1
        scope.update(
            phase="bounded_work",
            progress={
                "submitted": event.get("submitted"),
                "completed": event.get("completed"),
                "pending": event.get("pending"),
            },
        )

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    results = list(
        process_map(
            worker,
            range(8),
            n_jobs=2,
            window=4,
            mp_context=context,
            initializer=(install_worker_progress_endpoint if endpoint is not None else None),
            initargs=((endpoint,) if endpoint is not None else ()),
            progress_callback=(progress if telemetry_enabled else None),
        )
    )
    parent_cpu = time.process_time() - started_cpu
    wall = time.perf_counter() - started_wall
    recorder.close_worker_progress_endpoint(endpoint)
    scope.finish(status="success")
    recorder.close()
    digest = hashlib.sha256("".join(sorted(item[0] for item in results)).encode()).hexdigest()
    return _Measurement(
        wall_seconds=wall,
        parent_cpu_seconds=parent_cpu,
        digest=digest,
        sent_messages=sum(bool(item[1]) for item in results),
        scheduler_events=event_count,
    )


def _summarize(name: str, measurements: list[tuple[_Measurement, _Measurement]]) -> dict[str, object]:
    disabled = [item[0] for item in measurements]
    enabled = [item[1] for item in measurements]
    disabled_wall = statistics.median(item.wall_seconds for item in disabled)
    enabled_wall = statistics.median(item.wall_seconds for item in enabled)
    overhead = (enabled_wall / disabled_wall - 1.0) * 100.0
    digests = {item.digest for pair in measurements for item in pair}
    return {
        "fixture": name,
        "repetitions": len(measurements),
        "disabled_median_wall_seconds": disabled_wall,
        "enabled_median_wall_seconds": enabled_wall,
        "median_wall_overhead_percent": overhead,
        "disabled_median_parent_cpu_seconds": statistics.median(
            item.parent_cpu_seconds for item in disabled
        ),
        "enabled_median_parent_cpu_seconds": statistics.median(
            item.parent_cpu_seconds for item in enabled
        ),
        "canonical_digest_equal": len(digests) == 1,
        "canonical_digest": next(iter(digests)) if len(digests) == 1 else sorted(digests),
        "enabled_worker_messages": [item.sent_messages for item in enabled],
        "enabled_scheduler_events": [item.scheduler_events for item in enabled],
        "worker_message_rate_per_second": statistics.median(
            item.sent_messages / item.wall_seconds for item in enabled
        ),
    }


def main() -> int:
    if "spawn" not in mp.get_all_start_methods():
        raise RuntimeError("Task 1B benchmark requires the Windows-compatible spawn context")
    context = mp.get_context("spawn")
    output: list[dict[str, object]] = []
    for name, worker in (("rng_sort_merge", _rng_sort_unit), ("h2h_chunks", _h2h_chunk_unit)):
        _run_fixture(worker, telemetry_enabled=False, context=context)
        _run_fixture(worker, telemetry_enabled=True, context=context)
        measurements: list[tuple[_Measurement, _Measurement]] = []
        for repetition in range(7):
            # Alternate order to reduce time/order bias without changing work.
            if repetition % 2:
                enabled = _run_fixture(worker, telemetry_enabled=True, context=context)
                disabled = _run_fixture(worker, telemetry_enabled=False, context=context)
            else:
                disabled = _run_fixture(worker, telemetry_enabled=False, context=context)
                enabled = _run_fixture(worker, telemetry_enabled=True, context=context)
            measurements.append((disabled, enabled))
        output.append(_summarize(name, measurements))
    print(json.dumps(output, indent=2, sort_keys=True))
    return int(any(not bool(item["canonical_digest_equal"]) for item in output))


if __name__ == "__main__":
    raise SystemExit(main())
