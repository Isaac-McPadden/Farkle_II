from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.helpers.artifact_sidecars import (
    make_authenticated_v3_config,
    publish_v3_parquet,
    publish_v3_simulation_run,
)

from farkle.analysis import curate, ingest
from farkle.config import assign_config_sha
from farkle.simulation.simulation import _play_game, simulation_rows_to_table
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils.artifact_contract import sidecar_path, validate_artifact_sidecar
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose


def test_curate_publishes_and_backfills_row_sidecars_without_recopying(tmp_path: Path) -> None:
    cfg = make_authenticated_v3_config(
        tmp_path,
        name="curate",
        root_seed=7,
    )
    cfg.sim.row_dir = Path("rows")
    cfg.ingest.n_jobs = 1
    assign_config_sha(cfg)
    strategies = (
        ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=11),
        ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=12),
    )
    row = _play_game(
        123,
        list(strategies),
        target_score=200,
        provenance={
            "root_seed": cfg.sim.seed,
            "k": 2,
            "shuffle_index": 0,
            "game_index": 0,
            "deterministic_batch_id": 0,
            "shuffle_seed": 456,
            "game_seed": 123,
            "rng_scheme_version": RNG_SCHEME_VERSION,
            "rng_purpose_namespace": int(RandomPurpose.TOURNAMENT_GAME),
        },
    )
    publish_v3_simulation_run(
        cfg,
        simulation_rows_to_table([row], 2),
        strategies=strategies,
    )
    ingest.run(cfg)
    assert pq.read_table(cfg.ingested_rows_raw(2)).num_rows == 1

    curate.run(cfg)

    output = cfg.ingested_rows_curated(2)
    original = output.read_bytes()
    manifest = json.loads(cfg.manifest_for(2).read_text(encoding="utf-8"))
    assert manifest["representation"] in {"reflink", "physical_copy"}
    assert manifest["source_sha256"] == manifest["curated_sha256"]
    assert manifest["copy_buffer_bytes"] <= cfg.resources.stage_batch_bytes["partitioned_stage"]
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

    with cfg.ingested_rows_raw(2).open("ab") as handle:
        handle.write(b"post-creation mutation")
    with pytest.raises(RuntimeError):
        curate.run(cfg)

    incompatible = pq.read_table(output).set_column(
        0,
        pa.field("root_seed", pa.string()),
        pa.array([str(cfg.sim.seed)], type=pa.string()),
    )
    publish_v3_parquet(
        cfg,
        cfg.ingested_rows_raw(2),
        incompatible,
        stage_key="ingest",
        producer="ingest",
        operation="ingest_simulation_rows",
        source_scope="by_k",
    )
    with pytest.raises(ValueError, match="incompatible schema"):
        curate.run(cfg)
