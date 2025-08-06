import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis_config import PipelineCfg
from farkle.curate import _already_curated, _write_manifest


def test_already_curated_schema_checksum(tmp_path):
    cfg = PipelineCfg(results_dir=tmp_path)

    schema1 = pa.schema([("a", pa.int64())])
    table1 = pa.Table.from_pydict({"a": [1]})
    file1 = tmp_path / "file1.parquet"
    pq.write_table(table1, file1)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, rows=1, schema=schema1, cfg=cfg)

    assert _already_curated(file1, manifest)

    table2 = pa.Table.from_pydict({"b": [1]})
    file2 = tmp_path / "file2.parquet"
    pq.write_table(table2, file2)

    assert not _already_curated(file2, manifest)
