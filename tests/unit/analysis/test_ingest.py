import logging

import pandas as pd
import pytest

from farkle.analysis.analysis_config import PipelineCfg
from farkle.analysis.ingest import _fix_winner, _iter_shards, run

# -------------------- _iter_shards -------------------------------------

def test_iter_shards_consolidated(tmp_path):
    block = tmp_path
    df = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"]})
    df.to_parquet(block / "2p_rows.parquet", index=False)

    shards = list(_iter_shards(block, ("winner", "P1_strategy", "missing")))
    assert len(shards) == 1
    shard_df, shard_path = shards[0]
    assert shard_path.name == "2p_rows.parquet"
    assert list(shard_df.columns) == ["winner", "P1_strategy"]


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


# -------------------- run integration ----------------------------------

def test_run_schema_mismatch_logs_and_closes(tmp_path, caplog, monkeypatch):
    cfg = PipelineCfg(results_dir=tmp_path, analysis_subdir="analysis")

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


def test_run_emits_logging(tmp_path, caplog):
    cfg = PipelineCfg(results_dir=tmp_path, analysis_subdir="analysis")
    block = cfg.results_dir / "2_players"
    block.mkdir(parents=True)
    df = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"], "n_rounds": [1], "winning_score": [100]})
    df.to_parquet(block / "2p_rows.parquet", index=False)

    caplog.set_level(logging.INFO, logger="farkle.analysis.ingest")
    run(cfg)

    messages = [rec.message for rec in caplog.records]
    assert any("Ingest started" in msg for msg in messages)
    assert any("Ingest block complete" in msg for msg in messages)
