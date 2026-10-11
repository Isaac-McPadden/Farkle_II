"""Production-size high-k physical layout without playing production games."""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis import ingest
from farkle.config import AppConfig
from farkle.utils.partitioned_stage import PartitionedUnit
from farkle.utils.schema_helpers import raw_simulation_schema_for


def test_production_high_k_ingest_footer_is_admitted_without_row_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_property: Any
) -> None:
    """Stream 4,300 synthetic shuffle batches; never retain the game population."""
    cfg = AppConfig()
    cfg.ingest.row_group_size = 64000
    cfg.resources.stage_batch_bytes["ingest"] = 32 * 2**20
    schema = raw_simulation_schema_for(12)
    games = 430

    def values(field: pa.Field) -> pa.Array:
        if field.nullable:
            return pa.nulls(games, type=field.type)
        value: Any = (
            False
            if pa.types.is_boolean(field.type)
            else (
                "fixture"
                if pa.types.is_string(field.type)
                else [] if pa.types.is_list(field.type) else 0
            )
        )
        return pa.repeat(cast(Any, pa.scalar(value, type=field.type)), games)

    base = pa.Table.from_arrays([values(field) for field in schema], schema=schema)
    for name, value in (("root_seed", 11), ("k", 12), ("outcome_schema_version", 2)):
        field = schema.field(name)
        base = base.set_column(
            schema.get_field_index(name), field, pa.repeat(pa.scalar(value, type=field.type), games)
        )
    field = schema.field("game_index")
    base = base.set_column(
        schema.get_field_index("game_index"), field, pa.array(range(games), type=field.type)
    )
    max_input_bytes = 0

    def batches(_shards, *, columns, max_batch_bytes, max_batch_rows):
        nonlocal max_input_bytes
        assert list(columns) == schema.names
        assert max_batch_bytes == 8 * 2**20
        assert games <= max_batch_rows
        field = schema.field("shuffle_index")
        for shuffle in range(4300):
            table = base.set_column(
                schema.get_field_index("shuffle_index"),
                field,
                pa.repeat(pa.scalar(shuffle, type=field.type), games),
            )
            max_input_bytes = max(max_input_bytes, table.nbytes)
            yield table, tmp_path / "synthetic-shard.parquet"

    monkeypatch.setattr(ingest, "_iter_shards_arrow", batches)
    snapshot = cast(Any, SimpleNamespace(n_players=12, shards=()))
    output = tmp_path / "coalesced.parquet"
    ingest._IngestUnitWriter(cfg, (snapshot,))(PartitionedUnit((12,), output.name), output)
    with output.open("rb") as stream:
        stream.seek(-8, 2)
        footer_bytes, magic = struct.unpack("<I4s", stream.read(8))
    assert magic == b"PAR1"
    assert footer_bytes <= 32 * 2**20 // 4
    parquet = pq.ParquetFile(output)
    assert parquet.metadata.num_rows == 4300 * games
    assert parquet.metadata.num_row_groups < 4300 // 4
    # Independently check every ordered coordinate with bounded projected reads.
    ordinal = 0
    for batch in parquet.iter_batches(columns=["shuffle_index", "game_index"], batch_size=2048):
        for shuffle, game in zip(batch[0].to_pylist(), batch[1].to_pylist(), strict=True):
            assert (shuffle, game) == divmod(ordinal, games)
            ordinal += 1
    assert ordinal == 4300 * games
    record_property("synthetic_games", ordinal)
    record_property("row_groups", parquet.metadata.num_row_groups)
    record_property("footer_bytes", footer_bytes)
    record_property("largest_input_bytes", max_input_bytes)
