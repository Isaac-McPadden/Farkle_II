"""Small deterministic, production-published sources for outcome qualification."""

from __future__ import annotations

import pyarrow as pa
from tests.helpers.artifact_sidecars import publish_v3_strategy_manifest

from farkle.analysis import curate, ingest
from farkle.config import AppConfig
from farkle.simulation import runner
from farkle.simulation.strategies import ThresholdStrategy
from farkle.simulation.workload_planner import TournamentWorkloadPlan
from farkle.utils.random import RandomPurpose
from farkle.utils.schema_helpers import raw_simulation_schema_for
from farkle.utils.streaming_loop import run_streaming_shard


def publish_outcome_source(
    cfg: AppConfig, rows: list[dict], *, roster: tuple[int, ...], width: int
) -> None:
    """Publish explicit toy outcomes through real simulation/ingest/curate contracts.

    Non-outcome counters are declared zero fixture values; the required outcome
    values and all game coordinates come from the caller, never imputation.
    """
    k = cfg.sim.n_players_list[0]
    games = len(roster) // k
    shuffles = len(rows) // games
    assert len(roster) % k == len(rows) % games == shuffles % width == 0
    manifest = publish_v3_strategy_manifest(
        cfg,
        tuple(
            ThresholdStrategy(score_threshold=sid + 100, dice_threshold=2, strategy_id=sid)
            for sid in roster
        ),
    )
    plan = TournamentWorkloadPlan(
        root_seed=cfg.sim.seed,
        k=k,
        strategy_count=len(roster),
        confidence=0.95,
        resolution_delta=0.9,
        required_shuffles_unrounded=shuffles,
        required_shuffles=shuffles,
        batch_count=shuffles // width,
        shuffles_per_batch=width,
        batch_construction="equal_contiguous",
        games_per_shuffle=games,
        required_games=len(rows),
        achieved_resolution=0.9,
        shuffle_cap=None,
        cap_exceeded=False,
        achieved_resolution_at_cap=None,
    )
    workload = cfg.simulation_workload_plan_path(k)
    runner._write_workload_plan_simulation_output(
        cfg, workload, plan, n_players=k, strategy_manifest=manifest
    )
    folder = cfg.simulation_row_dir(k)
    assert folder is not None
    folder.mkdir(parents=True, exist_ok=True)
    schema = raw_simulation_schema_for(k)
    for shuffle in range(shuffles):
        records = []
        for game in range(games):
            row = dict(rows[shuffle * games + game])
            row.update(
                shuffle_seed=1000 + shuffle,
                game_seed=100000 + shuffle * games + game,
                rng_scheme_version=2,
                rng_purpose_namespace=int(RandomPurpose.TOURNAMENT_GAME),
                n_rounds=1,
            )
            for field in schema:
                if field.name not in row:
                    assert field.name.startswith("P") and pa.types.is_integer(field.type)
                    row[field.name] = 0
            records.append(row)
        table = pa.Table.from_pylist(records, schema=schema)
        path = folder / f"rows_{shuffle:06d}.parquet"
        run_streaming_shard(
            out_path=str(path),
            manifest_path=str(folder / "manifest.jsonl"),
            schema=schema,
            batch_iter=(table,),
            sidecar=runner._simulation_output_sidecar(
                cfg,
                path,
                n_players=k,
                operation="publish_simulation_row_shard",
                sources=[manifest, workload],
            ),
            manifest_extra={
                "path": path.name,
                "rows": games,
                "root_seed": cfg.sim.seed,
                "n_players": k,
                "shuffle_index": shuffle,
                "shuffle_seed": 1000 + shuffle,
                "deterministic_batch_id": shuffle // width,
                "rng_scheme_version": 2,
                "outcome_schema_version": 2,
                "tournament_method_version": 2,
            },
        )
    runner.write_simulation_done(
        cfg,
        k,
        num_shuffles=shuffles,
        shuffles_per_batch=width,
        n_strategies=len(roster),
        outputs=[manifest, workload, folder],
    )
    assert runner.simulation_is_complete(cfg, k)
    ingest.run(cfg)
    curate.run(cfg)
