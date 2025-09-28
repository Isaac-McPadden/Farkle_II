import csv
import multiprocessing as mp
import threading
from pathlib import Path
from typing import Mapping, Sequence

import pytest
from pytest import MonkeyPatch

import farkle.utils.csv_files as csv_files
import farkle.utils.parallel as parallel


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

    class DummyFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class DummyExecutor:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *exc):  # noqa: ANN002
            return False

        def submit(self, fn, item):
            submitted.append(item)
            return DummyFuture(fn(item))

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr(parallel, "as_completed", lambda futures: iter(futures))

    result = list(parallel.process_map(lambda x: x * 2, [1, 2, 3], n_jobs=2, window=2))

    assert result == [2, 4, 6]
    assert submitted == [1, 2, 3]
