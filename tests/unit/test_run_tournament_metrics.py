"""
Tests for the metric-collecting helpers in run_tournament.py.

These tests patch the heavy game logic so that no real games are played. They
focus on the aggregation logic in _run_chunk_metrics and its optional row
logging behaviour.
"""

from __future__ import annotations

import pickle
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import farkle.run_tournament as rt

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
    monkeypatch.setattr(rt.mp, "current_process", lambda: types.SimpleNamespace(pid=42))
    monkeypatch.setattr(rt.time, "time_ns", lambda: 99)

    written = {}

    class DummyTable:
        @classmethod
        def from_pylist(cls, rows):
            written["rows"] = list(rows)
            return "tbl"

    pa_mod = types.ModuleType("pyarrow")
    pq_mod = types.ModuleType("pyarrow.parquet")
    pa_mod.Table = types.SimpleNamespace(from_pylist=DummyTable.from_pylist)  # type: ignore
    pa_mod.parquet = pq_mod  # type: ignore
    pq_mod.write_table = lambda tbl, path: written.update({"path": Path(path), "tbl": tbl})  # type: ignore
    monkeypatch.setitem(sys.modules, "pyarrow", pa_mod)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq_mod)

    wins, sums, sqs = rt._run_chunk_metrics([3], collect_rows=True, row_dir=tmp_path)

    assert written["rows"] == [{"game_seed": 3, "winner_strategy": "S3"}]
    assert written["tbl"] == "tbl"
    assert written["path"].parent == tmp_path
    assert written["path"].name == "rows_42_99.parquet"


def test_run_chunk_metrics_skips_on_missing_pyarrow(monkeypatch, tmp_path):
    monkeypatch.setattr(rt, "_play_one_shuffle", _fake_play_one_shuffle, raising=True)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("pyarrow"):
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    wins, sums, sqs = rt._run_chunk_metrics([4], collect_rows=True, row_dir=tmp_path)

    assert wins == Counter({"S4": 1})
    assert list(tmp_path.iterdir()) == []
    
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
    