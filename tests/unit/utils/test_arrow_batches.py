from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.utils.arrow_batches import iter_parquet_tables_by_bytes


def test_projected_parquet_batches_are_byte_bounded(tmp_path: Path) -> None:
    path = tmp_path / "wide.parquet"
    table = pa.table(
        {
            "coordinate": pa.array(range(5000), type=pa.int32()),
            "wanted": [f"value-{index:05d}" * 3 for index in range(5000)],
            "unprojected": ["x" * 500 for _ in range(5000)],
        }
    )
    pq.write_table(table, path, row_group_size=5000)

    batches = list(
        iter_parquet_tables_by_bytes(
            path,
            columns=["coordinate", "wanted"],
            max_batch_bytes=4096,
            max_batch_rows=1000,
        )
    )
    assert sum(batch.num_rows for _rg, _bi, batch in batches) == 5000
    assert all(batch.nbytes <= 4096 for _rg, _bi, batch in batches)
    assert all(batch.column_names == ["coordinate", "wanted"] for _rg, _bi, batch in batches)
    assert len(batches) > 5
