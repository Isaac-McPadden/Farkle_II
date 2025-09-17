import pyarrow as pa
import pytest

from farkle.utils.writer import ParquetShardWriter


def test_parquet_shard_writer_atomic(tmp_path):
    if getattr(pa, "__version__", "0.0.0") == "0.0.0":
        pytest.skip("pyarrow stub active")
    out_path = tmp_path / "sample.parquet"
    table = pa.Table.from_pydict({"winner": ["P1"], "score": [100]})

    with ParquetShardWriter(str(out_path)) as writer:
        writer.write_batch(table)

    assert out_path.exists()
    assert writer.rows_written == 1
    assert not any(tmp_path.glob("._tmp_*"))
