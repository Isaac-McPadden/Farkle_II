import csv
import multiprocessing as mp
import os
import threading
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Mapping, Sequence

import pytest
from pytest import MonkeyPatch

import farkle.utils.csv_files as csv_files
import farkle.utils.parallel as parallel
from farkle.config import ResourcesConfig


def _times_two(value: int) -> int:
    return value * 2


@pytest.fixture
def writer_queue() -> mp.Queue:  # type: ignore
    queue: mp.Queue = mp.Queue()
    try:
        yield queue  # type: ignore
    finally:
        queue.close()
        queue.join_thread()


def test_writer_worker_background_thread(tmp_path: Path, writer_queue: mp.Queue) -> None:
    header = ["a", "b"]
    out = tmp_path / "out.csv"

    worker = threading.Thread(
        target=csv_files._writer_worker, args=(writer_queue, str(out), header)
    )
    worker.start()

    writer_queue.put({"a": 1, "b": 2})
    writer_queue.put({"a": 3, "b": 4})
    writer_queue.put(None)

    worker.join(timeout=5)
    assert not worker.is_alive()

    with out.open(encoding="utf-8") as fh:
        text_lines = fh.read().splitlines()

    assert text_lines[0] == "a,b"
    assert text_lines.count("a,b") == 1
    assert len(text_lines) == 3

    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert rows == [
        {"a": "1", "b": "2"},
        {"a": "3", "b": "4"},
    ]


def test_writer_worker_flushes_when_buffer_full(
    tmp_path: Path, writer_queue: mp.Queue, monkeypatch: MonkeyPatch
) -> None:
    header = ["a", "b"]
    out = tmp_path / "out.csv"

    monkeypatch.setattr(csv_files, "BUFFER_SIZE", 1)

    batches: list[list[Mapping[str, object]]] = []

    class DummyWriter:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def writeheader(self) -> None:
            return None

        def writerows(self, rows: Sequence[Mapping[str, object]]) -> None:
            batches.append([dict(row) for row in rows])

    monkeypatch.setattr(csv_files.csv, "DictWriter", lambda *_args, **_kwargs: DummyWriter())

    worker = threading.Thread(
        target=csv_files._writer_worker, args=(writer_queue, str(out), header)
    )
    worker.start()

    writer_queue.put({"a": 1, "b": 2})
    writer_queue.put({"a": 3, "b": 4})
    writer_queue.put({"a": 5, "b": 6})
    writer_queue.put(None)

    worker.join(timeout=5)
    assert not worker.is_alive()

    assert len(batches) == 3
    assert all(len(batch) == 1 for batch in batches)


def test_writer_worker_respects_existing_header(tmp_path: Path, writer_queue: mp.Queue) -> None:
    header = ["a", "b"]
    out = tmp_path / "out.csv"
    out.write_text("a,b\n5,6\n", encoding="utf-8")

    worker = threading.Thread(
        target=csv_files._writer_worker, args=(writer_queue, str(out), header)
    )
    worker.start()

    writer_queue.put({"a": 7, "b": 8})
    writer_queue.put(None)

    worker.join(timeout=5)
    assert not worker.is_alive()

    with out.open(encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    assert lines[0] == "a,b"
    assert lines.count("a,b") == 1
    assert lines[1:] == ["5,6", "7,8"]


def test_writer_worker_handles_immediate_termination(
    tmp_path: Path, writer_queue: mp.Queue
) -> None:
    header = ["a", "b"]
    out = tmp_path / "out.csv"

    worker = threading.Thread(
        target=csv_files._writer_worker, args=(writer_queue, str(out), header)
    )
    worker.start()

    writer_queue.put(None)

    worker.join(timeout=5)
    assert not worker.is_alive()
    assert out.exists()

    with out.open(encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    assert lines == ["a,b"]


def test_writer_worker_detects_empty_existing_file(tmp_path: Path, writer_queue: mp.Queue) -> None:
    header = ["a", "b"]
    out = tmp_path / "out.csv"
    out.touch()

    worker = threading.Thread(
        target=csv_files._writer_worker, args=(writer_queue, str(out), header)
    )
    worker.start()

    writer_queue.put({"a": 1, "b": 2})
    writer_queue.put(None)

    worker.join(timeout=5)
    assert not worker.is_alive()

    with out.open(encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    assert lines[0] == "a,b"
    assert lines[1:] == ["1,2"]


def test_process_map_serial():
    items = [1, 2, 3]
    result = list(parallel.process_map(lambda x: x + 1, items, n_jobs=1))
    assert result == [2, 3, 4]


def test_process_map_executor(monkeypatch: MonkeyPatch):
    submitted = []
    executor_kwargs = {}

    class DummyFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class DummyExecutor:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs
            executor_kwargs.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):  # noqa: ANN002
            return False

        def submit(self, fn, item):
            submitted.append(item)
            return DummyFuture(fn(item))

        def shutdown(self, *, wait, cancel_futures):  # noqa: ANN001
            executor_kwargs["shutdown"] = (wait, cancel_futures)

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr(parallel, "as_completed", lambda futures: iter(futures))

    mp_context = (
        parallel.resolve_mp_context("spawn") if "spawn" in mp.get_all_start_methods() else None
    )
    result = list(
        parallel.process_map(
            _times_two,
            [1, 2, 3],
            n_jobs=2,
            window=2,
            mp_context=mp_context,
        )
    )

    assert result == [2, 4, 6]
    assert submitted == [1, 2, 3]
    assert executor_kwargs.get("mp_context") is mp_context


def test_resource_failure_cancels_pending_process_futures(monkeypatch: MonkeyPatch) -> None:
    cancelled: list[int] = []
    shutdown: list[tuple[bool, bool]] = []

    class DummyFuture:
        def __init__(self, item: int) -> None:
            self.item = item

        def result(self) -> int:
            if self.item == 1:
                raise MemoryError("synthetic worker allocation")
            return self.item

        def cancel(self) -> bool:
            cancelled.append(self.item)
            return True

    class DummyExecutor:
        def __init__(self, **_kwargs) -> None:  # noqa: ANN003
            pass

        def submit(self, _fn, item: int) -> DummyFuture:  # noqa: ANN001
            return DummyFuture(item)

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            shutdown.append((wait, cancel_futures))

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr(parallel, "as_completed", lambda futures: iter(futures))

    with pytest.raises(parallel.ResourceFailureError) as raised:
        list(parallel.process_map(_times_two, [1, 2, 3], n_jobs=2, window=3))

    assert raised.value.classification == "allocator_memory_error"
    assert cancelled == [2, 3]
    assert shutdown == [(True, True)]


def test_process_map_context_modes_identical_artifacts(tmp_path: Path) -> None:
    items = [1, 2, 3, 4]
    contexts: list[BaseContext | None] = [None]
    if "spawn" in mp.get_all_start_methods():
        contexts.append(parallel.resolve_mp_context("spawn"))

    artifact_paths: list[Path] = []
    for idx, context in enumerate(contexts):
        values = sorted(parallel.process_map(_times_two, items, n_jobs=2, mp_context=context))
        out_path = tmp_path / f"result_{idx}.csv"
        out_path.write_text("\n".join(str(v) for v in values) + "\n", encoding="utf-8")
        artifact_paths.append(out_path)

    baseline = artifact_paths[0].read_text(encoding="utf-8")
    for path in artifact_paths[1:]:
        assert path.read_text(encoding="utf-8") == baseline


def test_resolve_mp_context_none_default_and_invalid() -> None:
    assert parallel.resolve_mp_context(None) is None
    assert parallel.resolve_mp_context("  ") is None
    assert parallel.resolve_mp_context("default") is None

    with pytest.raises(ValueError, match="Unsupported multiprocessing start method"):
        parallel.resolve_mp_context("definitely-not-valid")


def test_process_map_serial_initializer_with_explicit_initargs() -> None:
    init_calls: list[tuple[int, int]] = []

    def initializer(a: int, b: int) -> None:
        init_calls.append((a, b))

    values = list(
        parallel.process_map(
            lambda x: x + 10,
            [1, 2],
            n_jobs=1,
            initializer=initializer,
            initargs=[7, 9],
        )
    )

    assert values == [11, 12]
    assert init_calls == [(7, 9)]


def test_normalize_n_jobs_semantics() -> None:
    assert parallel.normalize_n_jobs(None, cpu_count=8, default=3) == 3
    assert parallel.normalize_n_jobs(0, cpu_count=8) == 8
    assert parallel.normalize_n_jobs(1, cpu_count=8) == 1
    assert parallel.normalize_n_jobs(2, cpu_count=8) == 2

    with pytest.raises(ValueError, match="n_jobs must be >= 0"):
        parallel.normalize_n_jobs(-1, cpu_count=8)


def test_resolve_stage_parallel_policy_default_cfg() -> None:
    class DummyCfg:
        n_jobs: int | None = None

    policy = parallel.resolve_stage_parallel_policy("analysis", DummyCfg())
    assert policy.process_workers == 1
    assert policy.python_threads >= 1
    assert policy.arrow_threads >= 1
    assert policy.native_threads_per_process >= 1


def test_resolve_stage_parallel_policy_nested_context() -> None:
    class DummyCfg:
        n_jobs: int | None = 4

    outer = parallel.ParallelNestingContext(
        active_process_executor=True,
        parent_process_workers=3,
        total_cores=12,
    )
    policy = parallel.resolve_stage_parallel_policy("analysis", DummyCfg(), outer_context=outer)

    assert policy.total_cores == 12
    assert policy.process_workers == 1
    assert policy.native_threads_per_process == 4
    assert policy.python_threads == 4
    assert policy.arrow_threads == 1


def test_resolve_stage_parallel_policy_nested_context_with_zero_n_jobs() -> None:
    class DummyCfg:
        n_jobs: int | None = 0

    outer = parallel.ParallelNestingContext(
        active_process_executor=True,
        parent_process_workers=2,
        total_cores=10,
    )
    policy = parallel.resolve_stage_parallel_policy("metrics", DummyCfg(), outer_context=outer)

    assert policy.total_cores == 10
    assert policy.process_workers == 1
    assert policy.native_threads_per_process == 5
    assert policy.python_threads == 5
    assert policy.arrow_threads == 1


def test_apply_native_thread_limits(monkeypatch: MonkeyPatch) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "PYARROW_NUM_THREADS",
    ):
        monkeypatch.delenv(key, raising=False)

    policy = parallel.StageParallelPolicy(
        total_cores=8,
        process_workers=2,
        python_threads=4,
        arrow_threads=3,
        native_threads_per_process=4,
    )
    parallel.apply_native_thread_limits(policy)

    assert os.environ["OMP_NUM_THREADS"] == "4"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"
    assert os.environ["MKL_NUM_THREADS"] == "4"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "4"
    assert os.environ["VECLIB_MAXIMUM_THREADS"] == "4"
    assert os.environ["BLIS_NUM_THREADS"] == "4"
    assert os.environ["PYARROW_NUM_THREADS"] == "3"


def test_resource_policy_enforces_cpu_and_memory_caps() -> None:
    class DummyCfg:
        n_jobs: int | None = 20

    resources = ResourcesConfig(
        scheduler_memory_budget_mb=768,
        parent_process_memory_mb=192,
        logical_cpu_budget=12,
        native_threads_per_worker=2,
        estimated_worker_memory_mb={"analysis": 160},
        stage_batch_bytes={"analysis": 4096},
    )
    policy = parallel.resolve_stage_parallel_policy("analysis", DummyCfg(), resources=resources)

    assert policy.cpu_worker_cap == 6
    assert policy.memory_worker_cap == 3
    assert policy.process_workers == 3
    assert policy.process_workers <= policy.configured_cpu_budget // 2
    assert policy.process_workers <= (768 - 192) // 160
    assert policy.native_threads_per_process == 2


def test_machine_targeted_policy_resolves_fifteen_workers_on_sixteen_threads(
    monkeypatch: MonkeyPatch,
) -> None:
    class DummyCfg:
        n_jobs: int | None = 0

    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 16)
    resources = ResourcesConfig(
        scheduler_memory_budget_mb=8192,
        process_tree_warning_threshold_mb=8192,
        aggregate_memory_hard_limit_mb=12288,
        minimum_system_available_memory_mb=8192,
        parent_process_memory_mb=512,
        logical_cpu_budget=15,
        native_threads_per_worker=1,
        estimated_worker_memory_mb={"analysis": 192},
        stage_batch_bytes={"analysis": 4096},
    )

    resolved = parallel.resolve_resource_policy(
        resources,
        logical_cpu_count=16,
        total_memory_mb=32768,
        available_memory_mb=20000,
    )
    policy = parallel.resolve_stage_parallel_policy("analysis", DummyCfg(), resources=resources)

    assert resolved.resolved_logical_cpu_budget == 15
    assert policy.process_workers == 15
    assert policy.memory_worker_cap == 40


def test_native_thread_arithmetic_cannot_oversubscribe_logical_cpu_budget(
    monkeypatch: MonkeyPatch,
) -> None:
    class DummyCfg:
        n_jobs: int | None = 15

    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 16)
    resources = ResourcesConfig(
        scheduler_memory_budget_mb=8192,
        process_tree_warning_threshold_mb=8192,
        aggregate_memory_hard_limit_mb=12288,
        minimum_system_available_memory_mb=8192,
        parent_process_memory_mb=512,
        logical_cpu_budget=15,
        native_threads_per_worker=2,
        estimated_worker_memory_mb={"analysis": 192},
        stage_batch_bytes={"analysis": 4096},
    )

    policy = parallel.resolve_stage_parallel_policy("analysis", DummyCfg(), resources=resources)

    assert policy.process_workers == 7
    assert policy.process_workers * policy.native_threads_per_process <= 15


def test_explicit_resource_policy_is_validated_against_detected_capacity() -> None:
    resources = ResourcesConfig(
        aggregate_memory_hard_limit_mb=12288,
        minimum_system_available_memory_mb=8192,
        logical_cpu_budget=15,
    )
    with pytest.raises(parallel.ResourceSafetyError, match="requested 15, detected 8"):
        parallel.resolve_resource_policy(
            resources,
            logical_cpu_count=8,
            total_memory_mb=32768,
            available_memory_mb=20000,
        )
    with pytest.raises(parallel.ResourceSafetyError, match="exceeds detected memory"):
        parallel.resolve_resource_policy(
            resources,
            logical_cpu_count=16,
            total_memory_mb=16384,
            available_memory_mb=12000,
        )


def test_resource_policy_rejects_worker_that_cannot_fit_scheduler_share() -> None:
    class DummyCfg:
        n_jobs: int | None = 4

    resources = ResourcesConfig(
        scheduler_memory_budget_mb=768,
        parent_process_memory_mb=192,
        logical_cpu_budget=4,
        estimated_worker_memory_mb={"analysis": 700},
        stage_batch_bytes={"analysis": 4096},
    )
    with pytest.raises(parallel.ResourceSafetyError, match="scheduler memory share"):
        parallel.resolve_stage_parallel_policy("analysis", DummyCfg(), resources=resources)


def test_concurrent_roots_share_the_cpu_and_memory_envelope() -> None:
    class DummyCfg:
        n_jobs: int | None = 8

    resources = ResourcesConfig(
        logical_cpu_budget=8,
        native_threads_per_worker=1,
        estimated_worker_memory_mb={"analysis": 128},
        stage_batch_bytes={"analysis": 4096},
    )
    policy = parallel.resolve_stage_parallel_policy(
        "analysis", DummyCfg(), resources=resources, concurrent_roots=2
    )
    repeated = parallel.resolve_stage_parallel_policy(
        "analysis", DummyCfg(), resources=resources, concurrent_roots=2
    )
    assert policy.process_workers == 2
    assert policy.cpu_worker_cap == 4
    assert policy.memory_worker_cap == 2
    assert repeated == policy


def test_rss_is_not_a_sticky_hard_limit(monkeypatch: MonkeyPatch) -> None:
    guard = parallel.ProcessTreeMemoryGuard(aggregate_hard_limit_mb=950)
    # Keep this unit test synchronous; the public sample/check path remains the
    # behavior under test while the daemon sampler is covered by integration.
    guard._monitor_started = True
    monkeypatch.setattr(
        parallel,
        "process_tree_rss_bytes",
        lambda _pid=None: 951 * 1024 * 1024,
    )
    assert guard.check_before_schedule(force=True) == 951 * 1024 * 1024

    monkeypatch.setattr(
        parallel,
        "process_tree_rss_bytes",
        lambda _pid=None: 100 * 1024 * 1024,
    )
    assert guard.check_before_schedule(force=True) == 100 * 1024 * 1024


def test_process_tree_guard_enforces_system_available_reserve(
    monkeypatch: MonkeyPatch,
) -> None:
    guard = parallel.ProcessTreeMemoryGuard(
        aggregate_hard_limit_mb=2304,
        minimum_system_available_memory_mb=1024,
    )
    guard._monitor_started = True
    monkeypatch.setattr(parallel, "process_tree_rss_bytes", lambda _pid=None: 100)
    monkeypatch.setattr(
        parallel.psutil,
        "virtual_memory",
        lambda: type("Memory", (), {"available": 1023 * 1024 * 1024})(),
    )

    with pytest.raises(parallel.ResourceSafetyError, match="reserve would be violated"):
        guard.check_before_schedule(force=True)


def test_process_tree_guard_snapshot_is_read_only(monkeypatch: MonkeyPatch) -> None:
    guard = parallel.ProcessTreeMemoryGuard(aggregate_hard_limit_mb=2304, rss_warning_mb=768)
    guard.last_rss_bytes = 100
    guard.peak_rss_bytes = 200
    guard.last_native_threads = 3
    guard.peak_native_threads = 5
    guard.last_aggregate_memory_bytes = 300
    guard.peak_aggregate_memory_bytes = 400
    guard.aggregate_memory_source = "test-job"
    guard.aggregate_hard_limit_bytes = 1_000
    guard.warning_crossings = 2
    guard.backpressure_seconds = 1.5
    monkeypatch.setattr(
        parallel.psutil,
        "virtual_memory",
        lambda: type("Memory", (), {"available": 2_000})(),
    )

    snapshot = guard.snapshot()

    assert snapshot == {
        "process_tree_rss_bytes": 100,
        "peak_process_tree_rss_bytes": 200,
        "native_threads": 3,
        "peak_native_threads": 5,
        "aggregate_memory_bytes": 300,
        "peak_aggregate_memory_bytes": 400,
        "aggregate_memory_hard_limit_bytes": 1_000,
        "aggregate_memory_source": "test-job",
        "host_available_memory_bytes": 2_000,
        "warning_crossings": 2,
        "backpressure_seconds": 1.5,
        "near_hard_boundary": False,
        "monitoring_error": None,
    }
    assert guard.last_rss_bytes == 100
    assert guard.warning_crossings == 2


def test_process_tree_memory_warning_backpressures_until_memory_recedes(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    guard = parallel.ProcessTreeMemoryGuard(aggregate_hard_limit_mb=2304, rss_warning_mb=768)
    guard._monitor_started = True
    samples = iter((951, 951, 700))
    monkeypatch.setattr(
        parallel, "process_tree_rss_bytes", lambda _pid=None: next(samples) * 1024 * 1024
    )
    monkeypatch.setattr(parallel.time, "sleep", lambda _seconds: None)

    assert guard.check_before_schedule(force=True) == 700 * 1024 * 1024
    assert "exceeded the configured high-water threshold" in caplog.text
    assert guard.warning_crossings == 1
    assert not guard.warning_emitted


def test_resource_exceptions_are_separate_from_data_failures() -> None:
    assert parallel.classify_resource_exception(MemoryError("worker allocation")) == (
        "allocator_memory_error"
    )
    assert parallel.classify_resource_exception(RuntimeError("Arrow bad allocation")) == (
        "allocator_bad_allocation"
    )
    paging = OSError("paging file too small")
    paging.winerror = 1455  # type: ignore[attr-defined]
    assert parallel.classify_resource_exception(paging) == "windows_commit_exhaustion"
    assert parallel.classify_resource_exception(ValueError("invalid schema")) is None


def test_persistent_high_water_is_a_resource_failure(monkeypatch: MonkeyPatch) -> None:
    guard = parallel.ProcessTreeMemoryGuard(
        aggregate_hard_limit_mb=2304,
        rss_warning_mb=768,
        high_water_timeout_seconds=0.001,
        sample_interval_seconds=0.001,
    )
    guard._monitor_started = True
    monkeypatch.setattr(parallel, "process_tree_rss_bytes", lambda _pid=None: 800 * 1024 * 1024)
    with pytest.raises(parallel.ResourceFailureError) as raised:
        guard.check_before_schedule(force=True)
    assert raised.value.classification == "persistent_high_water"


def test_nested_executor_environment_prevents_process_pool() -> None:
    class DummyCfg:
        n_jobs: int | None = 8

    previous = os.environ.get("FARKLE_PROCESS_POOL_ACTIVE")
    os.environ["FARKLE_PROCESS_POOL_ACTIVE"] = "1"
    try:
        policy = parallel.resolve_stage_parallel_policy("analysis", DummyCfg())
    finally:
        if previous is None:
            os.environ.pop("FARKLE_PROCESS_POOL_ACTIVE", None)
        else:
            os.environ["FARKLE_PROCESS_POOL_ACTIVE"] = previous
    assert policy.process_workers == 1
