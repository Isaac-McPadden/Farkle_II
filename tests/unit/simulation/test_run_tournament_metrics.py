# ruff: noqa: ARG005 ARG003 ARG002 ARG001 E402
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
from types import ModuleType
from typing import Any

import pytest

# pytestmark = pytest.mark.xfail(
#     reason=(
#         "Metric chunk aggregation is currently non-deterministic; "
#         "tracked at https://github.com/Isaac-McPadden/Farkle_II/issues/201"
#     ),
#     strict=False,
# )

pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")

# Farkle depends on numba at import time. Provide a light-weight stub so the
# test suite can run without the real dependency installed.
sys.modules.setdefault(
    "numba",
    types.SimpleNamespace(jit=lambda *a, **k: (lambda f: f), njit=lambda *a, **k: (lambda f: f)),  # type: ignore  # noqa: ARG005
)
# Stats helpers depend on SciPy ?+? provide a minimal stub


def _ppf(q, loc=0.0, scale=1.0):  # noqa: ANN001
    return NormalDist(mu=loc, sigma=scale).inv_cdf(q)


def _isf(q, loc=0.0, scale=1.0):  # noqa: ANN001
    return NormalDist(mu=loc, sigma=scale).inv_cdf(1 - q)


scipy_stats_stub = types.SimpleNamespace(norm=types.SimpleNamespace(ppf=_ppf, isf=_isf))
scipy_mod = ModuleType("scipy")
scipy_mod.stats = scipy_stats_stub  # type: ignore
sys.modules.setdefault("scipy", scipy_mod)
sys.modules.setdefault("scipy.stats", scipy_stats_stub)  # type: ignore

import pyarrow as pa
import pyarrow.parquet as pq

import farkle.simulation.run_tournament as rt
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils import manifest


@pytest.fixture(autouse=True)
def silence_logging(monkeypatch):
    """Mute info/error logs from run_tournament during tests."""
    monkeypatch.setattr(rt.LOGGER, "info", lambda *a, **k: None)
    monkeypatch.setattr(rt.LOGGER, "error", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# Dummy helper for deterministic results
# ---------------------------------------------------------------------------


def _fake_play_one_shuffle(
    seed: int, *, collect_rows: bool = False
) -> tuple[
    Counter[int | str],
    dict[str, dict[int | str, float]],
    dict[str, dict[int | str, float]],
    list[dict[str, object]],
]:
    """Return simple deterministic aggregates based on the seed."""
    winner = f"S{seed}"
    wins: Counter[int | str] = Counter({winner: 1})
    sums: dict[str, dict[int | str, float]] = {
        label: defaultdict(float, {winner: float(seed)}) for label in rt.METRIC_LABELS
    }
    sq: dict[str, dict[int | str, float]] = {
        label: defaultdict(float, {winner: float(seed * seed)}) for label in rt.METRIC_LABELS
    }
    rows: list[dict[str, object]] = (
        [{"game_seed": seed, "winner_strategy": winner}] if collect_rows else []
    )
    return wins, sums, sq, rows


# ---------------------------------------------------------------------------
# Shared helpers for run_tournament tests
# ---------------------------------------------------------------------------


def _mini_strategies(count: int = 6) -> list[ThresholdStrategy]:
    return [ThresholdStrategy(40 + 5 * i, i % 3, True, True) for i in range(count)]


def _install_serial_process_map(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_process_map(func, iterable, *, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            initializer(*initargs)
        for item in iterable:
            yield func(item)

    monkeypatch.setattr(rt.parallel, "process_map", fake_process_map)


def _setup_serial_run(monkeypatch: pytest.MonkeyPatch) -> list[ThresholdStrategy]:
    strategies = _mini_strategies()
    monkeypatch.setattr(
        rt, "generate_strategy_grid", lambda *a, **kw: (strategies, None), raising=True
    )

    def fake_measure(
        sample_strategies, sample_games: int = 2_000, seed: int = 0
    ) -> float:  # noqa: ARG001
        n_players = max(1, len(sample_strategies))
        return 8_160 / n_players

    monkeypatch.setattr(rt, "_measure_throughput", fake_measure, raising=True)
    monkeypatch.setattr(
        rt.urandom,
        "spawn_seeds",
        lambda count, seed=0: list(range(seed, seed + count)),
        raising=True,
    )
    _install_serial_process_map(monkeypatch)
    return strategies


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

    recorded: dict[str, Any] = {}

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
            recorded["rows"] = batches[0].to_pylist()

    monkeypatch.setattr(rt, "run_streaming_shard", fake_run_streaming_shard)

    manifest = tmp_path / "manifest.jsonl"
    wins, sums, sqs = rt._run_chunk_metrics(
        [3], collect_rows=True, row_dir=tmp_path, manifest_path=manifest
    )

    assert recorded["rows"] == [{"game_seed": 3, "winner_strategy": "S3"}]
    assert recorded["out_path"] == tmp_path / "rows_42_3.parquet"
    assert recorded["manifest_path"] == manifest
    assert recorded["schema"] == pa.schema(
        [("game_seed", pa.int64()), ("winner_strategy", pa.string())]
    )
    assert recorded["batches"] and isinstance(recorded["batches"][0], pa.Table)
    assert recorded["extra"] == {
        "path": "rows_42_3.parquet",
        "n_players": None,
        "shuffle_seed": 3,
        "pid": 42,
    }


def test_save_checkpoint_round_trip(tmp_path):
    wins: Counter[int | str] = Counter({"A": 2})
    sums: dict[str, dict[int | str, float]] = {
        label: defaultdict(float, {"A": 1.0}) for label in rt.METRIC_LABELS
    }
    sqs: dict[str, dict[int | str, float]] = {
        label: defaultdict(float, {"A": 2.0}) for label in rt.METRIC_LABELS
    }

    ckpt = tmp_path / "ckpt.pkl"
    rt._save_checkpoint(ckpt, wins, sums, sqs)

    payload = pickle.loads(ckpt.read_bytes())

    assert payload["win_totals"] == wins
    assert payload["metric_sums"] == sums
    assert payload["metric_square_sums"] == sqs


def test_save_checkpoint_wins_only(tmp_path):
    """_save_checkpoint should omit metric keys when sums/sqs are None."""

    wins: Counter[int | str] = Counter({"X": 7})
    ckpt = tmp_path / "ckpt.pkl"

    rt._save_checkpoint(ckpt, wins, None, None)

    payload = pickle.loads(ckpt.read_bytes())

    assert payload == {"win_totals": wins}


def test_run_tournament_metric_chunks_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _setup_serial_run(monkeypatch)
    monkeypatch.setattr(rt, "_play_one_shuffle", _fake_play_one_shuffle, raising=True)

    checkpoint_path = tmp_path / "run" / "checkpoint.pkl"
    metrics_dir = tmp_path / "metrics"
    row_dir = tmp_path / "rows"

    config = rt.TournamentConfig(n_players=2, desired_sec_per_chunk=1, ckpt_every_sec=9999)

    rt.run_tournament(
        config=config,
        global_seed=0,
        checkpoint_path=checkpoint_path,
        n_jobs=1,
        collect_metrics=True,
        row_output_directory=row_dir,
        metric_chunk_directory=metrics_dir,
        num_shuffles=3,
    )

    chunk_files = sorted(metrics_dir.glob("metrics_*.parquet"))
    assert len(chunk_files) == 1

    metrics_manifest = metrics_dir / "metrics_manifest.jsonl"
    metric_records = list(manifest.iter_manifest(metrics_manifest))
    assert len(metric_records) == 1
    assert {record["path"] for record in metric_records} == {f.name for f in chunk_files}

    row_files = list(row_dir.glob("rows_*.parquet"))
    assert len(row_files) == config.num_shuffles
    row_records = list(manifest.iter_manifest(row_dir / "manifest.jsonl"))
    assert len(row_records) == config.num_shuffles
    assert {Path(record["path"]).name for record in row_records} == {f.name for f in row_files}
    assert all(record.get("rows") == 1 for record in row_records)

    metrics_path = checkpoint_path.with_name("2p_metrics.parquet")
    table = pq.read_table(metrics_path)
    assert table.num_rows == len(rt.METRIC_LABELS) * 3
    metrics_rows = {
        (
            row["metric"],
            row["strategy"],
        ): (row["sum"], row["square_sum"])
        for row in table.to_pylist()
    }
    for seed in range(3):
        for label in rt.METRIC_LABELS:
            assert metrics_rows[(label, f"S{seed}")] == (float(seed), float(seed * seed))


def test_run_tournament_checkpoint_cadence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _setup_serial_run(monkeypatch)
    monkeypatch.setattr(rt, "_play_one_shuffle", _fake_play_one_shuffle, raising=True)

    metrics_dir = tmp_path / "metrics"
    checkpoint_path = tmp_path / "ckpt.pkl"

    calls: list[tuple[Path, Counter[str], object, object]] = []

    def fake_save_checkpoint(path: Path, wins, sums, sqs, **kwargs):  # noqa: ANN001
        calls.append((Path(path), wins.copy(), sums, sqs))

    monkeypatch.setattr(rt, "_save_checkpoint", fake_save_checkpoint, raising=True)

    perf_state = {"value": -15.0}

    def fake_perf_counter():
        perf_state["value"] += 15.0
        return perf_state["value"]

    monkeypatch.setattr(rt.time, "perf_counter", fake_perf_counter, raising=True)
    monkeypatch.setattr(rt.time, "sleep", lambda *a, **k: None, raising=True)

    config = rt.TournamentConfig(n_players=2, desired_sec_per_chunk=1, ckpt_every_sec=10)

    rt.run_tournament(
        config=config,
        global_seed=0,
        checkpoint_path=checkpoint_path,
        n_jobs=1,
        collect_metrics=True,
        metric_chunk_directory=metrics_dir,
        num_shuffles=2,
    )

    assert len(calls) == 2  # one in-loop checkpoint plus the final flush
    mid_run_calls = calls[:-1]
    assert all(call[2] is None and call[3] is None for call in mid_run_calls)
    final_path, final_wins, final_sums, final_sq = calls[-1]
    assert final_path == checkpoint_path
    assert isinstance(final_wins, Counter)
    assert final_sums is not None and final_sq is not None


def test_run_tournament_reuses_existing_metric_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _setup_serial_run(monkeypatch)
    monkeypatch.setattr(rt, "_play_one_shuffle", _fake_play_one_shuffle, raising=True)

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()

    pre_rows = [
        {
            "metric": label,
            "strategy": "S_pre",
            "sum": 5.0,
            "square_sum": 25.0,
        }
        for label in rt.METRIC_LABELS
    ]
    pq.write_table(pa.Table.from_pylist(pre_rows), metrics_dir / "metrics_000010.parquet")

    checkpoint_path = tmp_path / "run" / "checkpoint.pkl"
    config = rt.TournamentConfig(n_players=2, desired_sec_per_chunk=1, ckpt_every_sec=9999)

    rt.run_tournament(
        config=config,
        global_seed=0,
        checkpoint_path=checkpoint_path,
        n_jobs=1,
        collect_metrics=True,
        metric_chunk_directory=metrics_dir,
        num_shuffles=1,
    )

    produced_chunks = sorted(metrics_dir.glob("metrics_*.parquet"))
    assert any(path.name == "metrics_000001.parquet" for path in produced_chunks)

    metrics_path = checkpoint_path.with_name("2p_metrics.parquet")
    table = pq.read_table(metrics_path)
    rows = {(row["metric"], row["strategy"]): row for row in table.to_pylist()}
    for label in rt.METRIC_LABELS:
        assert rows[(label, "S_pre")]["sum"] == 5.0
        assert rows[(label, "S_pre")]["square_sum"] == 25.0
        assert rows[(label, "S0")]["sum"] == 0.0
        assert rows[(label, "S0")]["square_sum"] == 0.0


def test_run_tournament_no_metrics_wins_only_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _setup_serial_run(monkeypatch)

    def fake_play_shuffle(seed: int) -> Counter[int | str]:
        return Counter({f"S{seed}": 1})

    monkeypatch.setattr(rt, "_play_shuffle", fake_play_shuffle, raising=True)

    checkpoint_path = tmp_path / "checkpoint.pkl"
    config = rt.TournamentConfig(n_players=2, desired_sec_per_chunk=1, ckpt_every_sec=9999)

    calls: list[tuple[Path, Counter[str], object, object]] = []

    def fake_save_checkpoint(path: Path, wins, sums, sqs, **kwargs):  # noqa: ANN001
        calls.append((Path(path), wins.copy(), sums, sqs))

    monkeypatch.setattr(rt, "_save_checkpoint", fake_save_checkpoint, raising=True)

    debug_logs: list[tuple[str, dict[str, object] | None]] = []

    def fake_debug(msg: str, *args, **kwargs):
        debug_logs.append((msg, kwargs.get("extra")))

    monkeypatch.setattr(rt.LOGGER, "debug", fake_debug)

    rt.run_tournament(
        config=config,
        global_seed=0,
        checkpoint_path=checkpoint_path,
        n_jobs=1,
        collect_metrics=False,
        num_shuffles=2,
    )

    assert len(calls) == 1
    path, wins, sums, sqs = calls[0]
    assert path == checkpoint_path
    assert wins == Counter({"S0": 1, "S1": 1})
    assert sums is None and sqs is None

    chunk_debugs = [extra for msg, extra in debug_logs if msg == "Chunk processed"]
    assert len(chunk_debugs) == 1
    for idx, extra in enumerate(chunk_debugs, start=1):
        assert extra is not None
        assert extra["chunk_index"] == idx
        assert extra["wins"] == 2
