import csv
import multiprocessing as mp
import threading
from pathlib import Path

import pytest
from pytest import MonkeyPatch

import farkle.utils.csv_files as csv_files
import farkle.utils.parallel as parallel


@pytest.fixture
def writer_queue() -> mp.Queue:
    queue: mp.Queue = mp.Queue()
    try:
        yield queue
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


def test_writer_worker_respects_existing_header(
    tmp_path: Path, writer_queue: mp.Queue
) -> None:
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

