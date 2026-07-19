from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis import curate
from farkle.config import AppConfig, IOConfig, SimConfig
from farkle.utils.artifact_contract import sidecar_path, validate_artifact_sidecar
from farkle.utils.schema_helpers import expected_schema_for


def test_curate_publishes_and_backfills_row_sidecars_without_recopying(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=7, seed_list=[7], n_players_list=[2]),
    )
    raw = cfg.ingested_rows_raw(2)
    raw.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([], schema=expected_schema_for(2)), raw)
    ingest_manifest = cfg.ingest_manifest(2)
    ingest_manifest.write_text("{}\n", encoding="utf-8")

    curate.run(cfg)

    output = cfg.ingested_rows_curated(2)
    original = output.read_bytes()
    validate_artifact_sidecar(
        output,
        expected={
            "scope": "by_k",
            "operation": "curate_game_rows",
            "player_counts": [2],
        },
    )

    sidecar_path(output).unlink()
    curate.run(cfg)

    assert output.read_bytes() == original
    validate_artifact_sidecar(output)
