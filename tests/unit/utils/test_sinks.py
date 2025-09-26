from __future__ import annotations

from collections import Counter

import pytest

from farkle.utils.sinks import CsvSink, write_counter_csv


def test_csv_sink_context_manager_writes_and_exception_cleanup(tmp_path):
    ctx_dir = tmp_path / "ctx"
    data_path = ctx_dir / "data.csv"

    with CsvSink(data_path, header=["step", "value"]) as sink:
        sink.write_row({"step": 1, "value": 2})
        sink.write_rows([
            {"step": 2, "value": 3},
            {"step": 3, "value": 5},
        ])

    assert data_path.exists()
    lines = data_path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "step,value",
        "1,2",
        "2,3",
        "3,5",
    ]
    assert not any(ctx_dir.glob("._tmp_*")), "temporary files should be cleaned"

    failing_path = ctx_dir / "failing.csv"
    with pytest.raises(RuntimeError), CsvSink(failing_path, header=["value"]) as sink:
        def boom(row):
            raise RuntimeError("boom")

        sink._writer.writerow = boom  # type: ignore[assignment]
        sink.write_row({"value": 10})

    assert not failing_path.exists()
    assert not any(ctx_dir.glob("._tmp_*")), "atomic tmp should be removed on failure"


def test_csv_sink_manual_open_close_and_cleanup(tmp_path):
    manual_dir = tmp_path / "manual"
    manual_path = manual_dir / "metrics.csv"

    sink = CsvSink(manual_path, header=["id", "score"])
    sink.open()
    sink.write_row({"id": "A", "score": 10})
    sink.write_rows([
        {"id": "B", "score": 12},
        {"id": "C", "score": 14},
    ])
    sink.close()

    assert manual_path.exists()
    lines = manual_path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "id,score",
        "A,10",
        "B,12",
        "C,14",
    ]
    assert not any(manual_dir.glob("._tmp_*")), "temporary files should be cleaned"

    failing_path = manual_dir / "manual_failing.csv"
    sink = CsvSink(failing_path, header=["value"])
    sink.open()

    def boom(row):
        raise RuntimeError("boom")

    sink._writer.writerow = boom  # type: ignore[assignment]

    with pytest.raises(RuntimeError):
        try:
            sink.write_row({"value": 5})
        except RuntimeError as exc:
            sink.close(RuntimeError, exc, None)
            raise

    assert not failing_path.exists()
    assert not any(manual_dir.glob("._tmp_*")), "atomic tmp should be removed on failure"


def test_csv_sink_append_mode(tmp_path):
    append_path = tmp_path / "append" / "scores.csv"

    with CsvSink(append_path, header=["name"], mode="w") as sink:
        sink.write_row({"name": "first"})

    with CsvSink(append_path, header=["name"], mode="a") as sink:
        sink.write_rows([
            {"name": "second"},
            {"name": "third"},
        ])

    lines = append_path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "name",
        "first",
        "second",
        "third",
    ]


def test_write_counter_csv_creates_dirs_and_preserves_order(tmp_path):
    counter = Counter({"gamma": 5, "alpha": 3, "beta": 1})
    out_path = tmp_path / "nested" / "stats" / "wins.csv"

    write_counter_csv(counter, out_path)

    assert out_path.exists()
    assert out_path.parent.is_dir()

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "strategy,wins",
        "gamma,5",
        "alpha,3",
        "beta,1",
    ]
