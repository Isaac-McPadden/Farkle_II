from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from tests.helpers.artifact_sidecars import (
    make_authenticated_v3_config,
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
