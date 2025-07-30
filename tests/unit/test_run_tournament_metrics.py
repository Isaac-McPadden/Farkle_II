"""
Tests for the metric-collecting helpers in run_tournament.py.

These tests patch the heavy game logic so that no real games are played. They
focus on the aggregation logic in _run_chunk_metrics and its optional row
logging behaviour.
"""

from __future__ import annotations

import importlib
import pickle
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import farkle.run_tournament as rt

# Farkle depends on numba at import time. Provide a light-weight stub so the
# test suite can run without the real dependency installed.
sys.modules.setdefault(
    "numba",
    types.SimpleNamespace(jit=lambda *a, **k: (lambda f: f), njit=lambda *a, **k: (lambda f: f)),  # type: ignore  # noqa: ARG005
)
# Provide a minimal pyarrow stub so farkle.run_tournament imports without the real dependency.
pa_stub = types.ModuleType("pyarrow")
pq_stub = types.ModuleType("pyarrow.parquet")
pa_stub.Table = types.SimpleNamespace(from_pylist=lambda rows: None)  # type: ignore  # noqa: ARG005
pa_stub.parquet = pq_stub  # type: ignore
pa_stub.__version__ = "0.0.0"  # type: ignore
pq_stub.write_table = lambda *a, **k: None  # type: ignore  # noqa: ARG005
sys.modules.setdefault("pyarrow", pa_stub)
sys.modules.setdefault("pyarrow.parquet", pq_stub)
# Stats helpers depend on SciPy – provide a minimal stub
scipy_stats_stub = types.SimpleNamespace(norm=types.SimpleNamespace(ppf=lambda *a, **k: 0.0))  # noqa: ARG005
scipy_mod = types.ModuleType("scipy")
scipy_mod.stats = scipy_stats_stub  # type: ignore
sys.modules.setdefault("scipy", scipy_mod)
sys.modules.setdefault("scipy.stats", scipy_stats_stub)  # type: ignore


@pytest.fixture(autouse=True)
def silence_logging(monkeypatch):
    """Mute info/error logs from run_tournament during tests."""
    monkeypatch.setattr(rt.log, "info", lambda *a, **k: None)
    monkeypatch.setattr(rt.log, "error", lambda *a, **k: None)

# ---------------------------------------------------------------------------
# Dummy helper for deterministic results
# ---------------------------------------------------------------------------


def _fake_play_one_shuffle(seed: int, *, collect_rows: bool = False):
    """Return simple deterministic aggregates based on the seed."""
    winner = f"S{seed}"
    wins = Counter({winner: 1})
    sums = {label: defaultdict(float, {winner: float(seed)}) for label in rt.METRIC_LABELS}
    sq = {label: defaultdict(float, {winner: float(seed * seed)}) for label in rt.METRIC_LABELS}
    rows = [{"game_seed": seed, "winner_strategy": winner}] if collect_rows else []
    return wins, sums, sq, rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_chunk_metrics_accumulates(monkeypatch):
    monkeypatch.setattr(rt, "_play_one_shuffle", _fake_play_one_shuffle, raising=True)

    wins, sums, sqs = rt._run_chunk_metrics([1, 2], collect_rows=False)

    assert wins == Counter({"S1": 1, "S2": 1})
    for label in rt.METRIC_LABELS:
        assert sums[label]["S1"] == 1
        assert sums[label]["S2"] == 2
        assert sqs[label]["S1"] == 1
        assert sqs[label]["S2"] == 4


def test_run_chunk_metrics_row_logging(monkeypatch, tmp_path):
    monkeypatch.setattr(rt, "_play_one_shuffle", _fake_play_one_shuffle, raising=True)
    monkeypatch.setattr(rt, "getpid", lambda: 42)

    written = {}

    class DummyTable:
        @classmethod
        def from_pylist(cls, rows):
            written["rows"] = list(rows)
            return "tbl"

    pq_mod = types.SimpleNamespace()
    pa_mod = types.SimpleNamespace(
        Table=types.SimpleNamespace(from_pylist=DummyTable.from_pylist), parquet=pq_mod
    )
    pq_mod.write_table = lambda tbl, path: written.update({"path": Path(path), "tbl": tbl})
    monkeypatch.setattr(rt, "pa", pa_mod, raising=False)
    monkeypatch.setattr(rt, "pq", pq_mod, raising=False)

    wins, sums, sqs = rt._run_chunk_metrics([3], collect_rows=True, row_dir=tmp_path)

    assert written["rows"] == [{"game_seed": 3, "winner_strategy": "S3"}]
    assert written["tbl"] == "tbl"
    assert written["path"].parent == tmp_path
    assert written["path"].name == "rows_42_3.parquet"


def test_save_checkpoint_round_trip(tmp_path):
    wins = Counter({"A": 2})
    sums = {label: defaultdict(float, {"A": 1.0}) for label in rt.METRIC_LABELS}
    sqs = {label: defaultdict(float, {"A": 2.0}) for label in rt.METRIC_LABELS}

    ckpt = tmp_path / "ckpt.pkl"
    rt._save_checkpoint(ckpt, wins, sums, sqs)

    payload = pickle.loads(ckpt.read_bytes())

    assert payload["win_totals"] == wins
    assert payload["metric_sums"] == sums
    assert payload["metric_square_sums"] == sqs


def test_save_checkpoint_wins_only(tmp_path):
    """_save_checkpoint should omit metric keys when sums/sqs are None."""

    wins = Counter({"X": 7})
    ckpt = tmp_path / "ckpt.pkl"

    rt._save_checkpoint(ckpt, wins, None, None)

    payload = pickle.loads(ckpt.read_bytes())

    assert payload == {"win_totals": wins}
    

@pytest.mark.unit
def test_row_writer_flushes_and_shuts_down(monkeypatch, tmp_path: Path):
    """Verify that
       1. workers can push rows onto the Queue,
       2. the writer flushes once ROW_BUFFER rows are reached,
       3. the poison-pill terminates the loop,
       4. the parquet shard path is correct."""
    # ------------------------------------------------------------------
    # Arrange – patch pyarrow with a lightweight stub
    # ------------------------------------------------------------------
    writes: dict[str, object] = {}

    class DummyTable:
        @classmethod
        def from_pylist(cls, rows):
            writes["rows"] = list(rows)          # capture the payload
            return "dummy-table"

    pa_stub = ModuleType("pyarrow")
    pq_stub = ModuleType("pyarrow.parquet")
    pa_stub.Table = SimpleNamespace(from_pylist=DummyTable.from_pylist)  # type: ignore
    pa_stub.parquet = pq_stub  # type: ignore

    def fake_write_table(tbl, path):
        writes.setdefault("paths", []).append(Path(path))  # type: ignore
        writes["tbl"] = tbl

    pq_stub.write_table = fake_write_table  # type: ignore

    monkeypatch.setitem(sys.modules, "pyarrow", pa_stub)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq_stub)

    # ------------------------------------------------------------------
    # Import (or reload) target module *after* stubbing pyarrow
    # ------------------------------------------------------------------
    if "farkle.run_tournament" in sys.modules:
        rt = importlib.reload(sys.modules["farkle.run_tournament"])
    else:
        rt = importlib.import_module("farkle.run_tournament")

    # Tiny buffer so the test stays quick
    monkeypatch.setattr(rt, "ROW_BUFFER", 2, raising=True)

    # ------------------------------------------------------------------
    # Act – push two rows then the poison-pill, run writer synchronously
    # ------------------------------------------------------------------
    q = rt.mp.Queue()
    q.put({"game_seed": 1, "winner_strategy": "S1"})
    q.put({"game_seed": 2, "winner_strategy": "S2"})      # hits ROW_BUFFER == 2
    q.put(None)                                           # graceful stop

    rt._row_writer(q, tmp_path)                           # run inside this proc

    # ------------------------------------------------------------------
    # Assert – one shard written with both rows
    # ------------------------------------------------------------------
    assert writes["tbl"] == "dummy-table"
    assert writes["rows"] == [
        {"game_seed": 1, "winner_strategy": "S1"},
        {"game_seed": 2, "winner_strategy": "S2"},
    ]

    # exactly one parquet file in the tmp dir, with the expected naming scheme
    shard_paths = writes["paths"]
    assert len(shard_paths) == 1  # type: ignore
    shard = shard_paths[0]  # type: ignore
    assert shard.parent == tmp_path
    assert shard.name.startswith("rows_") and shard.suffix == ".parquet"
    