from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tests.helpers.artifact_sidecars import make_authenticated_v3_config, publish_v3_parquet

from farkle.analysis import combine
from farkle.utils.schema_helpers import expected_schema_for


def test_zero_row_partition_remains_required_and_authenticated(tmp_path: Path) -> None:
    cfg = make_authenticated_v3_config(
        tmp_path,
        name="combine-empty",
        root_seed=5,
        player_counts=(2,),
    )
    cfg.analysis.n_jobs = 1
    source = cfg.ingested_rows_curated(2)
    publish_v3_parquet(
        cfg,
        source,
        pa.Table.from_pylist([], schema=expected_schema_for(2)),
        stage_key="curate",
        producer="curate",
        operation="curate_game_rows",
        source_scope="by_k",
    )

    combine.run(cfg)

    partition = cfg.combined_rows_by_k(2)
    assert partition.exists()
    assert pq.read_metadata(partition).num_rows == 0
    records = [
        json.loads(line)
        for line in cfg.combined_manifest_path().read_text(encoding="utf-8").splitlines()
    ]
    assert records[1]["unit_metadata"]["row_count"] == 0
    assert records[-1]["required_units"] == 1


def test_dataset_adapter_exposes_target_schema_without_materializing(tmp_path: Path) -> None:
    cfg = make_authenticated_v3_config(
        tmp_path,
        name="combine-adapter",
        root_seed=5,
        player_counts=(1,),
    )
    cfg.analysis.n_jobs = 1
    source = cfg.ingested_rows_curated(1)
    publish_v3_parquet(
        cfg,
        source,
        pa.Table.from_pylist([{"k": 1, "winner_seat": "P1"}], schema=expected_schema_for(1)),
        stage_key="curate",
        producer="curate",
        operation="curate_game_rows",
        source_scope="by_k",
    )

    combine.run(cfg)
    dataset = combine.concat_ks_dataset(cfg)

    assert dataset.schema.equals(expected_schema_for(12), check_metadata=False)
    assert dataset.count_rows() == 1
    assert not cfg.curated_parquet.exists()
