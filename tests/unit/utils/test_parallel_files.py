import csv
import multiprocessing as mp
from pathlib import Path

from pytest import MonkeyPatch

import farkle.utils.csv_files as csv_files
import farkle.utils.parallel as parallel


def test_writer_worker_appends(tmp_path: Path):
    header = ["a", "b"]
    out = tmp_path / "out.csv"

    q1 = mp.Queue()
    q1.put({"a": 1, "b": 2})
    q1.put(None)
    csv_files._writer_worker(q1, str(out), header)

    q2 = mp.Queue()
    q2.put({"a": 3, "b": 4})
    q2.put(None)
    csv_files._writer_worker(q2, str(out), header)

    with out.open() as fh:
        rows = list(csv.DictReader(fh))

    assert rows == [
        {"a": "1", "b": "2"},
        {"a": "3", "b": "4"},
    ]


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

