from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from farkle.analysis import metrics


def test_update_batch_counters_accumulates_multiple_strategies_and_seats() -> None:
    arr_wstrat = np.array(["alpha", "beta", "alpha", "gamma"], dtype=object)
    arr_wseat = np.array(["P1", "P2", "P1", "P3"], dtype=object)
    arr_score = np.array([10, 20, 30, 40], dtype=np.int64)
    arr_rounds = np.array([5, 7, 6, 8], dtype=np.int64)

    wins_by_strategy = Counter({"alpha": 1})
    rounds_by_strategy = Counter({"alpha": 2})
    score_by_strategy = Counter({"alpha": 50})
    wins_by_seat = Counter({"P1": 1})

    metrics._update_batch_counters(
        arr_wstrat,
        arr_wseat,
        arr_score,
        arr_rounds,
        wins_by_strategy,
        rounds_by_strategy,
        score_by_strategy,
        wins_by_seat,
    )

    assert wins_by_strategy == Counter({"alpha": 3, "beta": 1, "gamma": 1})
    assert rounds_by_strategy == Counter({"alpha": 13, "beta": 7, "gamma": 8})
    assert score_by_strategy == Counter({"alpha": 90, "gamma": 40, "beta": 20})
    assert wins_by_seat == Counter({"P1": 3, "P2": 1, "P3": 1})


def test_write_parquet_delegates_to_atomic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def _capture(table: pa.Table, path: Path) -> None:
        captured["table"] = table
        captured["path"] = path

    monkeypatch.setattr(metrics, "write_parquet_atomic", _capture)

    schema = pa.schema([
        ("strategy", pa.string()),
        ("wins", pa.int64()),
    ])
    rows = [
        {"strategy": "alpha", "wins": 2},
        {"strategy": "beta", "wins": 1},
    ]
    out_path = tmp_path / "metrics.parquet"

    metrics._write_parquet(out_path, rows, schema)

    assert captured["path"] == out_path
    table = captured["table"]
    assert isinstance(table, pa.Table)
    assert table.schema == schema
    assert table.to_pylist() == rows
