from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.utils.arrow_batches import coalesce_arrow_tables, iter_parquet_tables_by_bytes


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


def test_coalescing_preserves_nested_nullable_values_and_bounds_chunks() -> None:
    table = pa.table({"id": range(400), "values": [[None, i] for i in range(400)]})
    batches = list(
        coalesce_arrow_tables(
            (table.slice(i, 1) for i in range(400)), max_buffer_bytes=10000, target_rows=200
        )
    )
    assert pa.concat_tables(batches).equals(table)
    assert all(b.num_rows <= 200 and b.nbytes <= 10000 for b in batches)
    assert all(column.num_chunks == 1 for batch in batches for column in batch.columns)
    assert len(batches) == 2


def test_coalescing_byte_ceiling_schema_and_input_admission() -> None:
    table = pa.table({"s": ["a" * 20] * 20})
    batches = list(
        coalesce_arrow_tables(
            (table.slice(i, 1) for i in range(20)), max_buffer_bytes=100, target_rows=20
        )
    )
    assert pa.concat_tables(batches).equals(table)
    assert all(batch.nbytes <= 100 for batch in batches)
    with pytest.raises(MemoryError, match="input exceeds"):
        list(coalesce_arrow_tables([table], max_buffer_bytes=10, target_rows=20))
    with pytest.raises(ValueError, match="exact Arrow schema"):
        list(
            coalesce_arrow_tables(
                [table, pa.table({"other": [1]})], max_buffer_bytes=1000, target_rows=20
            )
        )
