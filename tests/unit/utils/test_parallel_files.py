import csv
import multiprocessing as mp
import threading
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Mapping, Sequence

import pytest
from pytest import MonkeyPatch

import farkle.utils.csv_files as csv_files
import farkle.utils.parallel as parallel


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

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr(parallel, "as_completed", lambda futures: iter(futures))

    mp_context = (
        parallel.resolve_mp_context("spawn")
        if "spawn" in mp.get_all_start_methods()
        else None
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
