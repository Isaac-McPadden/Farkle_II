"""
Tests for the metric-collecting helpers in run_tournament.py.

These tests patch the heavy game logic so that no real games are played. They
focus on the aggregation logic in _run_chunk_metrics and its optional row
logging behavior.
"""

from __future__ import annotations

import pickle
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from types import ModuleType, SimpleNamespace

import pytest

# Farkle depends on numba at import time. Provide a light-weight stub so the
# test suite can run without the real dependency installed.
sys.modules.setdefault(
    "numba",
    types.SimpleNamespace(jit=lambda *a, **k: (lambda f: f), njit=lambda *a, **k: (lambda f: f)),  # type: ignore  # noqa: ARG005
)
# Provide a minimal pyarrow stub so farkle.simulation.run_tournament imports without the real dependency.
pa_stub = types.ModuleType("pyarrow")
pq_stub = types.ModuleType("pyarrow.parquet")


class _DummyTable:
    def __init__(self, rows):
        self.rows = list(rows)
        self.schema = "dummy-schema"


def _from_pylist(rows):  # noqa: ANN001
    return _DummyTable(rows)


pa_stub.Table = types.SimpleNamespace(from_pylist=_from_pylist)  # type: ignore
pa_stub.parquet = pq_stub  # type: ignore
pa_stub.__version__ = "0.0.0"  # type: ignore
pq_stub.write_table = lambda *a, **k: None  # type: ignore  # noqa: ARG005
sys.modules["pyarrow"] = pa_stub
sys.modules["pyarrow.parquet"] = pq_stub
# Stats helpers depend on SciPy – provide a minimal stub


def _ppf(q, loc=0.0, scale=1.0):  # noqa: ANN001
    return NormalDist(mu=loc, sigma=scale).inv_cdf(q)


def _isf(q, loc=0.0, scale=1.0):  # noqa: ANN001
    return NormalDist(mu=loc, sigma=scale).inv_cdf(1 - q)


scipy_stats_stub = types.SimpleNamespace(
    norm=types.SimpleNamespace(ppf=_ppf, isf=_isf)
)
scipy_mod = types.ModuleType("scipy")
scipy_mod.stats = scipy_stats_stub  # type: ignore
sys.modules.setdefault("scipy", scipy_mod)
sys.modules.setdefault("scipy.stats", scipy_stats_stub)  # type: ignore

import farkle.simulation.run_tournament as rt


@pytest.fixture(autouse=True)
def silence_logging(monkeypatch):
    """Mute info/error logs from run_tournament during tests."""
    monkeypatch.setattr(rt.LOGGER, "info", lambda *a, **k: None)
    monkeypatch.setattr(rt.LOGGER, "error", lambda *a, **k: None)

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

    recorded: dict[str, object] = {}

    def fake_run_streaming_shard(**kwargs):
        batches = list(kwargs["batch_iter"])
        recorded.update(
            {
                "out_path": Path(kwargs["out_path"]),
                "manifest_path": Path(kwargs["manifest_path"]),
                "schema": kwargs["schema"],
                "batches": batches,
                "extra": kwargs.get("manifest_extra"),
            }
        )
        if batches:
            recorded["rows"] = batches[0].rows

    monkeypatch.setattr(rt, "run_streaming_shard", fake_run_streaming_shard)

    manifest = tmp_path / "manifest.jsonl"
    wins, sums, sqs = rt._run_chunk_metrics(
        [3], collect_rows=True, row_dir=tmp_path, manifest_path=manifest
    )

    assert recorded["rows"] == [{"game_seed": 3, "winner_strategy": "S3"}]
    assert recorded["out_path"] == tmp_path / "rows_42_3.parquet"
    assert recorded["manifest_path"] == manifest
    assert recorded["schema"] == "dummy-schema"
    assert recorded["batches"] and hasattr(recorded["batches"][0], "rows")
    assert recorded["extra"] == {
        "path": "rows_42_3.parquet",
        "n_players": None,
        "shuffle_seed": 3,
        "pid": 42,
    }


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
    