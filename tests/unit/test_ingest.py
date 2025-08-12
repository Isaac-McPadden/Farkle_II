import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.ingest import _iter_shards, _fix_winner, _coerce_schema, run
from farkle.analysis_config import PipelineCfg, expected_schema_for


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
    assert result["seat_ranks"].iloc[0] == ("P2", "P1")


def test_fix_winner_without_ranks():
    df = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"], "P2_strategy": ["B"]})
    result = _fix_winner(df)
    assert result["winner_strategy"].iloc[0] == "A"
    assert result["winner_seat"].iloc[0] == "P1"
    assert "seat_ranks" not in result.columns


# -------------------- _coerce_schema -----------------------------------

def test_coerce_schema_with_target():
    df = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"]})
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    target = expected_schema_for(2)
    coerced = _coerce_schema(tbl, target)
    assert coerced.schema == target
    assert coerced["P2_strategy"].to_pylist() == [None]


def test_coerce_schema_infer():
    df = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"]})
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    coerced = _coerce_schema(tbl)
    expected = expected_schema_for(1)
    assert coerced.schema == expected
    assert "P2_strategy" not in coerced.column_names


# -------------------- run integration ----------------------------------

def test_run_schema_mismatch_logs_and_closes(tmp_path, caplog, monkeypatch):
    cfg = PipelineCfg(results_dir=tmp_path, analysis_subdir="analysis")

    # create two block dirs so run() discovers them
    block1 = cfg.results_dir / "block1_players"
    block1.mkdir(parents=True)
    block2 = cfg.results_dir / "block2_players"
    block2.mkdir()

    created = []

    class DummyWriter:
        def __init__(self, path, schema, compression=None):
            self.path = path
            self.schema = schema
            self.compression = compression
            self.closed = False
            created.append(self)

        def write_table(self, table, row_group_size=None):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(pq, "ParquetWriter", DummyWriter)

    def fake_iter_shards(block, cols):
        if block.name.startswith("block1"):
            df = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"]})
            yield df, block / "good.parquet"
        else:
            df = pd.DataFrame({"winner": ["P1"], "P1_strategy": ["A"], "bad": [1]})
            yield df, block / "bad.parquet"

    monkeypatch.setattr("farkle.ingest._iter_shards", fake_iter_shards)

    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        run(cfg)

    assert any("Schema mismatch" in rec.message for rec in caplog.records)
    assert created and all(w.closed for w in created)
