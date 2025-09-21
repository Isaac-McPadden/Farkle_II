import os
import threading

from pathlib import Path
import pytest

pa = pytest.importorskip("pyarrow")

from farkle.utils.manifest import (
    append_manifest_line,
    append_manifest_many,
    iter_manifest,
)
from farkle.utils import streaming_loop


def test_append_manifest_helpers(tmp_path):
    manifest_path = tmp_path / "manifest.ndjson"

    append_manifest_line(manifest_path, {"path": "alpha"})
    append_manifest_many(
        manifest_path,
        [{"path": "beta"}, {"path": "gamma"}],
    )

    with manifest_path.open("a", encoding="utf-8") as fh:
        fh.write("\n")

    records = list(iter_manifest(manifest_path))

    assert [r["path"] for r in records] == ["alpha", "beta", "gamma"]
    for record in records:
        assert "ts" in record and record["ts"]


def test_run_streaming_shard_invocation(tmp_path, monkeypatch):
    tables = [
        pa.table({"value": [1, 2]}),
        pa.table({"value": [3]}),
    ]
    schema = tables[0].schema
    out_path = tmp_path / "nested" / "out.parquet"
    manifest_path = tmp_path / "manifest.ndjson"

    captured_init = {}
    captured_batches = []

    class DummyWriter:
        def __init__(self, *, out_path, schema, compression, row_group_size):
            captured_init.update(
                {
                    "out_path": out_path,
                    "schema": schema,
                    "compression": compression,
                    "row_group_size": row_group_size,
                }
            )
            self.rows_written = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write_batches(self, batch_iterable):
            batches = list(batch_iterable)
            captured_batches.append(batches)
            self.rows_written = sum(tbl.num_rows for tbl in batches)

    monkeypatch.setattr(streaming_loop, "ParquetShardWriter", DummyWriter)

    manifest_calls = []

    def fake_append(path, record):
        manifest_calls.append((path, record))

    monkeypatch.setattr(streaming_loop, "append_manifest_line", fake_append)

    run_extra = {"block": 42}
    streaming_loop.run_streaming_shard(
        out_path=str(out_path),
        manifest_path=str(manifest_path),
        schema=schema,
        batch_iter=iter(tables),
        row_group_size=10,
        compression="zstd",
        manifest_extra=run_extra,
    )

    assert out_path.parent.exists()
    assert captured_init == {
        "out_path": str(out_path),
        "schema": schema,
        "compression": "zstd",
        "row_group_size": 10,
    }
    assert len(captured_batches) == 1
    assert len(captured_batches[0]) == len(tables)
    for expected, actual in zip(tables, captured_batches[0]):
        assert actual is expected

    assert manifest_calls and manifest_calls[0][0] == str(manifest_path)
    manifest_record = manifest_calls[0][1]
    # Compute relative path from the manifest's directory (matches production code).
    manifest_dir = os.fspath(Path(manifest_path).parent)
    try:
        expected_rel = os.path.relpath(os.fspath(out_path), start=manifest_dir)
    except ValueError:
        # Cross-drive fallback on Windows: use absolute path
        expected_rel = os.fspath(out_path)
    assert manifest_record["path"] == expected_rel
    assert manifest_record["rows"] == sum(tbl.num_rows for tbl in tables)
    for key, value in run_extra.items():
        assert manifest_record[key] == value


def test_writer_thread_forwards_manifest(monkeypatch):
    tables = [
        pa.table({"value": [1]}),
        pa.table({"value": [2, 3]}),
    ]
    queue = list(tables) + [None]
    pop_calls = []

    def fake_pop():
        value = queue.pop(0)
        pop_calls.append(value)
        return value

    captured = {}

    def fake_run_streaming_shard(**kwargs):
        batches = list(kwargs.pop("batch_iter"))
        captured.update(kwargs)
        captured["batches"] = batches

    monkeypatch.setattr(streaming_loop, "run_streaming_shard", fake_run_streaming_shard)

    manifest_extra = {"player": "A"}
    streaming_loop.writer_thread(
        fake_pop,
        out_path="result.parquet",
        manifest_path="manifest.ndjson",
        schema=tables[0].schema,
        row_group_size=5,
        compression="snappy",
        manifest_extra=manifest_extra,
    )

    assert pop_calls[-1] is None
    assert captured["out_path"] == "result.parquet"
    assert captured["manifest_path"] == "manifest.ndjson"
    assert captured["schema"].equals(tables[0].schema)
    assert captured["row_group_size"] == 5
    assert captured["compression"] == "snappy"
    assert captured["manifest_extra"] == manifest_extra
    assert len(captured["batches"]) == len(tables)
    for expected, actual in zip(tables, captured["batches"]):
        assert actual is expected


def test_bounded_queue_blocks_and_closes():
    queue = streaming_loop.BoundedQueue(maxsize=1)
    first = pa.table({"value": [1]})
    second = pa.table({"value": [2]})

    queue.push(first)

    started = threading.Event()
    finished = threading.Event()

    def push_second():
        started.set()
        queue.push(second)
        finished.set()

    thread = threading.Thread(target=push_second)
    thread.start()

    assert started.wait(timeout=1.0)
    assert not finished.wait(timeout=0.1)

    popped_first = queue.pop()
    assert first.equals(popped_first)

    assert finished.wait(timeout=1.0)
    thread.join(timeout=1.0)

    popped_second = queue.pop()
    assert second.equals(popped_second)

    queue.close()
    assert queue.pop() is None
