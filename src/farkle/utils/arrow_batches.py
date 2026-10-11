"""Projected Parquet iteration with deterministic byte-bounded Arrow batches."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class OversizedArrowRowError(MemoryError):
    """Raised when one Arrow row alone exceeds the configured batch ceiling."""


def coalesce_arrow_tables(
    tables: Iterable[pa.Table], *, max_buffer_bytes: int, target_rows: int
) -> Iterator[pa.Table]:
    """Combine small adjacent batches without retaining unbounded Arrow chunks.

    Inputs must already satisfy the byte ceiling. Pending decoded bytes and
    rows stay bounded; compact every 128 batches to bound chunk metadata too.
    Compaction can temporarily hold both old and new buffers, so callers must
    reserve twice this allowance in addition to their input/writer allocation.
    Row order, types and values are preserved, including nullable/list fields.
    """
    if max_buffer_bytes < 1 or target_rows < 1:
        raise ValueError("positive coalescing byte and row bounds required")
    pending: list[pa.Table] = []
    pending_bytes = pending_rows = 0
    schema: pa.Schema | None = None
    for table in tables:
        if schema is None:
            schema = table.schema
        elif not table.schema.equals(schema, check_metadata=True):
            raise ValueError("coalescing requires one exact Arrow schema")
        if not table.num_rows:
            continue
        if table.nbytes > max_buffer_bytes or table.num_rows > target_rows:
            raise MemoryError("coalescing input exceeds its byte or row allowance")
        if pending and (
            pending_bytes + table.nbytes > max_buffer_bytes
            or pending_rows + table.num_rows > target_rows
        ):
            yield pa.concat_tables(pending).combine_chunks()
            pending = []
            pending_bytes = pending_rows = 0
        pending.append(table)
        pending_bytes += table.nbytes
        pending_rows += table.num_rows
        if len(pending) >= 128:
            pending = [pa.concat_tables(pending).combine_chunks()]
    if pending:
        yield pa.concat_tables(pending).combine_chunks()


def _conservative_projected_row_width(schema: pa.Schema, columns: Sequence[str]) -> int:
    width = 0
    for name in columns:
        data_type = schema.field(name).type
        if pa.types.is_boolean(data_type):
            width += 2
        elif pa.types.is_integer(data_type) or pa.types.is_floating(data_type):
            width += max(1, int(data_type.bit_width) // 8) + 1
        elif (
            pa.types.is_string(data_type)
            or pa.types.is_large_string(data_type)
            or pa.types.is_binary(data_type)
            or pa.types.is_large_binary(data_type)
        ):
            width += 64
        else:
            width += 128
    return max(1, width)


def _split_record_batch(batch: pa.RecordBatch, max_batch_bytes: int) -> Iterator[pa.RecordBatch]:
    """Split without Python row materialization until each slice fits the byte ceiling."""

    pending = batch
    while pending.num_rows:
        if pending.nbytes <= max_batch_bytes:
            yield pending
            return
        if pending.num_rows == 1:
            raise OversizedArrowRowError(
                f"one projected Arrow row uses {pending.nbytes} bytes, exceeding "
                f"the {max_batch_bytes}-byte batch ceiling"
            )
        approximate = max(1, int(pending.num_rows * max_batch_bytes / pending.nbytes))
        candidate_rows = min(pending.num_rows - 1, approximate)
        candidate = pending.slice(0, candidate_rows)
        while candidate.nbytes > max_batch_bytes and candidate_rows > 1:
            candidate_rows = max(1, candidate_rows // 2)
            candidate = pending.slice(0, candidate_rows)
        if candidate.nbytes > max_batch_bytes:
            raise OversizedArrowRowError(
                f"one projected Arrow row exceeds the {max_batch_bytes}-byte batch ceiling"
            )
        yield candidate
        pending = pending.slice(candidate_rows)


def iter_parquet_tables_by_bytes(
    path: Path,
    *,
    columns: Sequence[str],
    max_batch_bytes: int,
    max_batch_rows: int,
    start_row_group: int = 0,
    start_batch_index: int = 0,
    use_threads: bool = False,
    on_decoded_batch: Callable[[int], None] | None = None,
) -> Iterator[tuple[int, int, pa.Table]]:
    """Yield projected tables without whole-row-group reads or Python row graphs.

    Batch indices are deterministic for fixed row and byte ceilings. Callers
    persisting those indices must also bind both execution controls in their
    checkpoint so a changed boundary restarts the bounded unit safely.
    """

    if max_batch_bytes < 1 or max_batch_rows < 1:
        raise ValueError("Arrow batch byte and row ceilings must be positive")
    parquet = pq.ParquetFile(path)
    projected_width = _conservative_projected_row_width(parquet.schema_arrow, columns)
    decode_rows = max(1, min(max_batch_rows, max_batch_bytes // projected_width))
    for row_group in range(start_row_group, parquet.num_row_groups):
        batch_index = 0
        raw_batches = parquet.iter_batches(
            batch_size=decode_rows,
            row_groups=[row_group],
            columns=list(columns),
            use_threads=use_threads,
        )
        for raw in raw_batches:
            if on_decoded_batch is not None:
                on_decoded_batch(raw.nbytes)
            for bounded in _split_record_batch(raw, max_batch_bytes):
                if row_group != start_row_group or batch_index >= start_batch_index:
                    yield row_group, batch_index, pa.Table.from_batches([bounded])
                batch_index += 1


__all__ = [
    "OversizedArrowRowError",
    "iter_parquet_tables_by_bytes",
]
