from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.ingest import _fix_winner, _iter_shards, _process_block, run
from farkle.analysis.schema import expected_schema_for
from farkle.config import AppConfig


def _make_cfg(tmp_results_dir: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.io.results_dir = tmp_results_dir
    return cfg


def _create_shard(path: Path, rows: Iterable[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def test_iter_shards_consolidated(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="farkle.analysis.ingest")
    consolidated = tmp_path / "2p_rows.parquet"
    _create_shard(consolidated, [{"winner": "P1", "P1_strategy": "A"}])

    shards = list(_iter_shards(tmp_path, ("winner", "P1_strategy", "missing")))
    assert len(shards) == 1
    df, src = shards[0]
    assert src == consolidated
    assert list(df.columns) == ["winner", "P1_strategy"]
    assert any("Row file missing requested columns" in rec.message for rec in caplog.records)


def test_fix_winner_adds_strategy_column() -> None:
    df = pd.DataFrame(
        {
            "winner": ["P2"],
            "P1_strategy": ["alpha"],
            "P2_strategy": ["beta"],
            "P1_rank": [2],
            "P2_rank": [1],
        }
    )
    fixed = _fix_winner(df)
    assert fixed["winner_seat"].iloc[0] == "P2"
    assert fixed["winner_strategy"].iloc[0] == "beta"
    assert fixed["seat_ranks"].iloc[0] == ["P2", "P1"]


def test_process_block_writes_raw(tmp_results_dir: Path) -> None:
    cfg = _make_cfg(tmp_results_dir)
    block = cfg.results_dir / "3_players"
    block.mkdir(parents=True)
    _create_shard(block / "3p_rows.parquet", [{"winner": "P1", "P1_strategy": "A"}])

    written = _process_block(block, cfg)
    assert written == 1
    raw_out = cfg.ingested_rows_raw(3)
    manifest = raw_out.with_suffix(".manifest.jsonl")
    assert raw_out.exists()
    assert manifest.exists()

def test_run_process_pool_path(tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_cfg(tmp_results_dir)
    block = cfg.results_dir / "2_players"
    block.mkdir(parents=True)
    _create_shard(block / "2p_rows.parquet", [{"winner": "P1", "P1_strategy": "A"}])

    seen = {"workers": None}

    class DummyFuture:
        def __init__(self, value: int) -> None:
            self._value = value

        def result(self) -> int:
            return self._value

    class DummyPool:
        def __init__(self, max_workers: int) -> None:
            seen["workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return DummyFuture(fn(*args, **kwargs))

    monkeypatch.setattr("farkle.analysis.ingest.ProcessPoolExecutor", DummyPool)
    cfg.ingest.n_jobs = 2
    run(cfg)
    assert seen["workers"] == 2

def test_run_logging(tmp_results_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg = _make_cfg(tmp_results_dir)
    block = cfg.results_dir / "2_players"
    block.mkdir(parents=True)
    _create_shard(block / "2p_rows.parquet", [{"winner": "P1", "P1_strategy": "A"}])

    caplog.set_level(logging.INFO, logger="farkle.analysis.ingest")
    run(cfg)
    messages = [rec.message for rec in caplog.records]
    assert any("Ingest started" in msg for msg in messages)
    assert any("Ingest finished" in msg for msg in messages)
