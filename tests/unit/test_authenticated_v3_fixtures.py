from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tests.helpers.artifact_sidecars import (
    make_authenticated_v3_config,
    mutate_artifact_bytes,
    publish_v3_parquet,
    publish_v3_simulation_run,
    publish_v3_strategy_manifest,
)

from farkle.config import assign_config_sha
from farkle.simulation import runner
from farkle.simulation.simulation import _play_game, simulation_rows_to_table
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils.authenticated_contract import (
    load_immutable_manifest_sidecar,
    validate_authenticated_artifact_unbound,
)
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose
from farkle.utils.stage_completion import (
    CompletionState,
    resolve_stage_state,
    stage_done_path,
    write_stage_done,
)


def test_v3_config_and_strategy_manifest_use_accepted_identity(tmp_path: Path) -> None:
    cfg = make_authenticated_v3_config(tmp_path, root_seed=17)
    manifest = publish_v3_strategy_manifest(
        cfg,
        (
            ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=3),
            ThresholdStrategy(score_threshold=100, dice_threshold=5, strategy_id=7),
        ),
    )

    metadata = validate_authenticated_artifact_unbound(
        manifest,
        validate_provenance=False,
    )
    versions = metadata.versions
    assert (
        versions.artifact_contract_version,
        versions.rng_scheme_version,
        versions.outcome_schema_version,
        versions.schema_version,
        versions.estimand_version,
        versions.conditioning_version,
    ) == (3, 2, 2, 2, 2, 2)
    assert metadata.stage_identity.code.state == "clean"
    assert metadata.artifact.location.stage_key == "simulation"
    assert pq.read_table(manifest)["strategy_id"].to_pylist() == [3, 7]
    assert cfg.results_root.resolve().is_relative_to(tmp_path.resolve())


def test_v3_parquet_source_binding_has_valid_control_before_byte_mutation(
    tmp_path: Path,
) -> None:
    cfg = make_authenticated_v3_config(tmp_path)
    source = publish_v3_parquet(
        cfg,
        cfg.combined_rows_by_k(2),
        pa.table({"strategy": pa.array([1], type=pa.int32())}),
        stage_key="combine",
        producer="combine",
        operation="combine_test_rows",
    )
    output = publish_v3_parquet(
        cfg,
        cfg.metrics_all_player_batch_path(2),
        pa.table({"strategy": pa.array([1], type=pa.int32()), "wins": [1]}),
        stage_key="metrics",
        producer="metrics",
        operation="aggregate_test_rows",
        sources=(source,),
    )
    done = stage_done_path(cfg.metrics_stage_dir, "metrics")
    write_stage_done(done, inputs=[source], outputs=[output], cfg=cfg, stage="metrics")
    assert (
        resolve_stage_state(done, [source], [output], cfg=cfg, stage="metrics")
        is CompletionState.COMPLETE_VALID
    )

    mutate_artifact_bytes(source)

    assert (
        resolve_stage_state(done, [source], [output], cfg=cfg, stage="metrics")
        is CompletionState.COMPLETE_STALE
    )


def test_v3_simulation_fixture_publishes_authenticated_control(tmp_path: Path) -> None:
    cfg = make_authenticated_v3_config(tmp_path, root_seed=7)
    cfg.sim.row_dir = Path("rows")
    assign_config_sha(cfg)
    strategies = (
        ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=11),
        ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=12),
    )
    row = dict(
        _play_game(
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
    )

    published = publish_v3_simulation_run(
        cfg,
        simulation_rows_to_table([row], 2),
        strategies=strategies,
    )

    assert runner.simulation_is_complete(cfg, 2)
    for path in (
        published.shard,
        published.strategy_manifest,
        published.workload_plan,
    ):
        validate_authenticated_artifact_unbound(path, validate_provenance=False)
    load_immutable_manifest_sidecar(published.row_manifest)
    assert published.completion.is_file()
