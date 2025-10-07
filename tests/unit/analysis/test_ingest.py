import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from farkle.analysis.ingest import _fix_winner, _iter_shards, _process_block, run
from farkle.config import AppConfig, IOConfig, IngestConfig


def _make_cfg(tmp_results_dir: Path, *, ingest_overrides: dict[str, Any] | None = None) -> AppConfig:
    io_cfg = IOConfig(results_dir=tmp_results_dir, analysis_subdir="analysis", append_seed=False)
    ingest_cfg = IngestConfig(**(ingest_overrides or {}))
    return AppConfig(io=io_cfg, ingest=ingest_cfg)

# -------------------- _iter_shards -------------------------------------


def test_iter_shards_consolidated(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="farkle.analysis.ingest")
    block = tmp_path
    df = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"]})
    df.to_parquet(block / "2p_rows.parquet", index=False)

    shards = list(_iter_shards(block, ("winner", "P1_strategy", "missing")))
    assert len(shards) == 1
    shard_df, shard_path = shards[0]
    assert shard_path.name == "2p_rows.parquet"
    assert list(shard_df.columns) == ["winner", "P1_strategy"]

    messages = [
        record.message for record in caplog.records if record.name == "farkle.analysis.ingest"
    ]
    assert "Row file missing requested columns" in messages


def test_iter_shards_legacy_dirs(tmp_path):
    block = tmp_path
    row_dir = block / "2p_rows"
    row_dir.mkdir()
    df1 = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"]})
    df2 = pd.DataFrame({"winner": ["P2"], "P1_strategy": ["B"]})
    df1.to_parquet(row_dir / "a.parquet", index=False)
    df2.to_parquet(row_dir / "b.parquet", index=False)

    shards = list(_iter_shards(block, ("winner", "P1_strategy")))
    assert [p.name for _, p in shards] == ["a.parquet", "b.parquet"]


def test_iter_shards_csv_fallback(tmp_path):
    block = tmp_path
    df_pqt = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"]})
    df_csv = pd.DataFrame({"winner": ["P2"], "P1_strategy": ["B"]})
    df_pqt.to_parquet(block / "shard.parquet", index=False)
    df_csv.to_csv(block / "winners.csv", index=False)

    shards = list(_iter_shards(block, ("winner", "P1_strategy")))
    names = {p.name for _, p in shards}
    assert names == {"shard.parquet", "winners.csv"}


def test_iter_shards_subset_logs_missing(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="farkle.analysis.ingest")
    block = tmp_path
    df = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"]})
    df.to_parquet(block / "solo.parquet", index=False)

    shards = list(_iter_shards(block, ("winner", "P1_strategy", "missing")))
    assert len(shards) == 1
    shard_df, shard_path = shards[0]
    assert shard_path.name == "solo.parquet"
    assert list(shard_df.columns) == ["winner", "P1_strategy"]

    messages = [
        record.message for record in caplog.records if record.name == "farkle.analysis.ingest"
    ]
    assert "Shard missing requested columns" in messages


# -------------------- _fix_winner --------------------------------------


def test_fix_winner_with_ranks():
    df = pd.DataFrame(
        {
            "winner": ["P2"],
            "P1_strategy": ["A"],
            "P2_strategy": ["B"],
            "P1_rank": [2],
            "P2_rank": [1],
        }
    )
    result = _fix_winner(df)
    assert result["winner_strategy"].iloc[0] == "B"
    assert result["winner_seat"].iloc[0] == "P2"
    assert result["seat_ranks"].iloc[0] == ["P2", "P1"]
    assert "winner" not in result.columns


def test_fix_winner_without_ranks():
    df = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"], "P2_strategy": ["B"]})
    result = _fix_winner(df)
    assert result["winner_strategy"].iloc[0] == "A"
    assert result["winner_seat"].iloc[0] == "P1"
    assert result["seat_ranks"].iloc[0] == ["P1"]
    assert "winner" not in result.columns


def test_fix_winner_preserves_existing_fields():
    df = pd.DataFrame(
        {
            "winner": ["P1"],
            "winner_seat": ["P1"],
            "winner_strategy": ["kept"],
            "seat_ranks": [["P1", "P2"]],
        }
    )

    result = _fix_winner(df)

    assert "winner" not in result.columns
    assert result["winner_seat"].iloc[0] == "P1"
    assert result["winner_strategy"].iloc[0] == "kept"
    assert result["seat_ranks"].iloc[0] == ["P1", "P2"]


# -------------------- _process_block -----------------------------------


def test_process_block_skips_when_output_newer(tmp_results_dir, monkeypatch):
    cfg = _make_cfg(tmp_results_dir)
    block = cfg.results_dir / "3_players"
    block.mkdir(parents=True)

    shard = block / "3p_rows.parquet"
    df = pd.DataFrame(
        {"winner": ["P1"], "P1_strategy": ["A"], "n_rounds": [1], "winning_score": [10]}
    )
    df.to_parquet(shard, index=False)
    shard_mtime = shard.stat().st_mtime

    raw_out = cfg.ingested_rows_raw(3)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.write_text("existing")
    newer = shard_mtime + 10
    os.utime(raw_out, (newer, newer))

    calls: list[tuple[tuple, dict]] = []

    def fake_run_streaming_shard(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("farkle.analysis.ingest.run_streaming_shard", fake_run_streaming_shard)

    result = _process_block(block, cfg)

    assert result == 0
    assert not calls


def test_process_block_zero_rows_cleans_outputs(tmp_results_dir, monkeypatch):
    cfg = _make_cfg(tmp_results_dir)
    block = cfg.results_dir / "4_players"
    block.mkdir(parents=True)

    raw_out = cfg.ingested_rows_raw(4)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.write_text("stale")
    manifest = raw_out.with_suffix(".manifest.jsonl")
    manifest.write_text("old")

    def fake_iter_shards(block_path, cols):
        empty = pd.DataFrame(columns=cols)
        yield empty, block_path / "empty.parquet"

    calls: list[tuple[tuple, dict]] = []

    def fake_run_streaming_shard(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("farkle.analysis.ingest._iter_shards", fake_iter_shards)
    monkeypatch.setattr("farkle.analysis.ingest.run_streaming_shard", fake_run_streaming_shard)

    result = _process_block(block, cfg)

    assert result == 0
    assert not raw_out.exists()
    assert not manifest.exists()
    assert not calls


def test_process_block_handles_legacy_shards(tmp_results_dir, monkeypatch):
    cfg = _make_cfg(tmp_results_dir)
    block = cfg.results_dir / "3_players"
    row_dir = block / "3p_rows"
    row_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "winner": ["P2"],
            "P1_strategy": ["alpha"],
            "P1_rank": [2],
            "P2_strategy": ["beta"],
            "P2_rank": [1],
            "P3_strategy": ["gamma"],
            "P3_rank": [3],
            "n_rounds": [4],
            "winning_score": [250],
        }
    )
    shard_path = row_dir / "part-0.parquet"
    df.to_parquet(shard_path, index=False)

    batches: list[pd.DataFrame] = []

    def fake_run_streaming_shard(*, batch_iter, **kwargs):
        batches.extend(table.to_pandas() for table in batch_iter)

    monkeypatch.setattr("farkle.analysis.ingest.run_streaming_shard", fake_run_streaming_shard)

    result = _process_block(block, cfg)

    assert result == len(df)
    assert batches
    processed = batches[0]
    assert processed["winner_seat"].iloc[0] == "P2"
    assert processed["winner_strategy"].iloc[0] == "beta"
    assert processed["seat_ranks"].iloc[0][0] == "P2"


def test_process_block_zero_rows_without_outputs(tmp_results_dir, monkeypatch):
    cfg = _make_cfg(tmp_results_dir)
    block = cfg.results_dir / "5_players"
    block.mkdir(parents=True)

    def fake_iter_shards(block_path, cols):  # noqa: ARG001
        yield from ()

    monkeypatch.setattr("farkle.analysis.ingest._iter_shards", fake_iter_shards)
    monkeypatch.setattr(
        "farkle.analysis.ingest.run_streaming_shard",
        lambda *args, **kwargs: pytest.fail("unexpected streaming"),
    )

    result = _process_block(block, cfg)

    assert result == 0
    raw_out = cfg.ingested_rows_raw(5)
    assert not raw_out.exists()
    assert not raw_out.with_suffix(".manifest.jsonl").exists()


# -------------------- run integration ----------------------------------


def test_run_schema_mismatch_logs_and_closes(tmp_results_dir, caplog, monkeypatch):
    cfg = _make_cfg(tmp_results_dir)

    # create two block dirs so run() discovers them
    block1 = cfg.results_dir / "block1_players"
    block1.mkdir(parents=True)
    block2 = cfg.results_dir / "block2_players"
    block2.mkdir()

    calls = []

    def fake_run_streaming_shard(**kwargs):
        # Exhaust the iterator to mirror real behavior and update totals
        list(kwargs.get("batch_iter", ()))
        calls.append(kwargs)

    monkeypatch.setattr("farkle.analysis.ingest.run_streaming_shard", fake_run_streaming_shard)

    def fake_iter_shards(block, cols):  # noqa: ARG001
        if block.name.startswith("block1"):
            df = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"]})
            yield df, block / "good.parquet"
        else:
            df = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"], "bad": [1]})
            yield df, block / "bad.parquet"

    monkeypatch.setattr("farkle.analysis.ingest._iter_shards", fake_iter_shards)

    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        run(cfg)

    assert any("Schema mismatch" in rec.message for rec in caplog.records)
    assert len(calls) == 1


def test_run_process_pool_path(tmp_results_dir, monkeypatch):
    cfg = _make_cfg(tmp_results_dir, ingest_overrides={"n_jobs": 2})

    block1 = cfg.results_dir / "1_players"
    block2 = cfg.results_dir / "2_players"
    block1.mkdir(parents=True)
    block2.mkdir()

    calls: list[str] = []

    def fake_process_block(block, passed_cfg):
        assert passed_cfg is cfg
        calls.append(block.name)
        return {"1_players": 1, "2_players": 3}[block.name]

    monkeypatch.setattr("farkle.analysis.ingest._process_block", fake_process_block)

    seen = {}

    class DummyFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class DummyPool:
        def __init__(self, max_workers):
            seen["value"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            return DummyFuture(fn(*args, **kwargs))

    monkeypatch.setattr("farkle.analysis.ingest.ProcessPoolExecutor", DummyPool)

    run(cfg)

    assert seen["value"] == cfg.n_jobs_ingest
    assert calls == ["1_players", "2_players"]


def test_run_emits_logging(tmp_results_dir, caplog):
    cfg = _make_cfg(tmp_results_dir)
    block = cfg.results_dir / "2_players"
    block.mkdir(parents=True)
    df = pd.DataFrame(
        {"winner": ["P1"], "P1_strategy": ["A"], "n_rounds": [1], "winning_score": [100]}
    )
    df.to_parquet(block / "2p_rows.parquet", index=False)

    caplog.set_level(logging.INFO, logger="farkle.analysis.ingest")
    run(cfg)

    messages = [rec.message for rec in caplog.records]
    assert any("Ingest started" in msg for msg in messages)
    assert any("Ingest block complete" in msg for msg in messages)
