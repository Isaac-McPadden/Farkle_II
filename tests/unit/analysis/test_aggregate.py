from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from farkle.analysis.analysis_config import PipelineCfg, expected_schema_for
from farkle.analysis import aggregate

def _write_curated(path: Path, schema: pa.Schema, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(tbl, path)

def test_aggregate_pads_and_counts(tmp_path: Path) -> None:
    cfg = PipelineCfg(results_dir=tmp_path)
    # create per-N curated files
    p1 = cfg.ingested_rows_curated(1)
    schema1 = expected_schema_for(1)
    _write_curated(p1, schema1, [
        {"winner": "P1", "n_rounds": 1, "winning_score": 100, "P1_strategy": "A", "P1_rank": 1},
    ])
    p2 = cfg.ingested_rows_curated(2)
    schema2 = expected_schema_for(2)
    _write_curated(p2, schema2, [
        {"winner": "P1", "n_rounds": 1, "winning_score": 200, "P1_strategy": "A", "P2_strategy": "B", "P1_rank": 1, "P2_rank": 2},
    ])
    # run aggregate
    aggregate.run(cfg)
    out = cfg.data_dir / "all_n_players_combined" / "all_ingested_rows.parquet"
    pf = pq.ParquetFile(out)
    assert pf.metadata.num_rows == 2
    assert pq.read_schema(out).names == expected_schema_for(12).names
