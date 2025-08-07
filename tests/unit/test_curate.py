import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis_config import PipelineCfg, expected_schema_for
from farkle.curate import _already_curated, _write_manifest


def test_already_curated_schema_hash(tmp_path):
    cfg = PipelineCfg(results_dir=tmp_path)

    schema0 = expected_schema_for(0)
    table1 = pa.table(
        {
            "winner": ["P1"],
            "winner_seat": ["1"],
            "winning_score": [100],
            "n_rounds": [1],
        },
        schema=schema0,
    )
    file1 = tmp_path / "file1.parquet"
    pq.write_table(table1, file1)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, rows=1, schema=schema0, cfg=cfg)

    assert _already_curated(file1, manifest)

    table2 = pa.table(
        {
            "winner": ["P1"],
            "winner_seat": ["1"],
            "winning_score": [100],
            "n_rounds": [1],
            "P1_score": [100],
        }
    )
    file2 = tmp_path / "file2.parquet"
    pq.write_table(table2, file2)

    assert not _already_curated(file2, manifest)
