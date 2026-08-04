"""Integration coverage for manifest-backed simulation row ingestion."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.helpers.artifact_sidecars import (
    make_authenticated_v3_config,
    publish_v3_simulation_run,
)

from farkle.analysis import ingest
from farkle.config import AppConfig, assign_config_sha
from farkle.simulation.simulation import _play_game, simulation_rows_to_table
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils.artifact_contract import sidecar_path, validate_artifact_sidecar
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose
from farkle.utils.schema_helpers import raw_simulation_schema_for


def _strategies() -> tuple[ThresholdStrategy, ...]:
    return (
        ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=11),
        ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=12),
    )


def _valid_row_table(cfg: AppConfig) -> pa.Table:
    strategies = _strategies()
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
    return simulation_rows_to_table([row], 2)


def _publish_completed_row_run(cfg: AppConfig, table: pa.Table | None = None) -> Path:
    published = publish_v3_simulation_run(
        cfg,
        _valid_row_table(cfg) if table is None else table,
        strategies=_strategies(),
    )
    return published.shard


def _config(tmp_path: Path, *, workers: int, name: str = "review") -> AppConfig:
    cfg = make_authenticated_v3_config(tmp_path, name=name, root_seed=7)
    cfg.sim.row_dir = Path("rows")
    cfg.ingest.n_jobs = workers
    cfg.analysis.mp_start_method = "spawn"
    assign_config_sha(cfg)
    return cfg


def test_ingest_reads_manifest_backed_row_directory_through_spawn_worker(tmp_path: Path) -> None:
    cfg = _config(tmp_path, workers=2)
    _publish_completed_row_run(cfg)

    ingest.run(cfg)

    output = pq.read_table(cfg.ingested_rows_raw(2)).to_pandas()
    assert len(output) == 1
    assert output.loc[0, "root_seed"] == 7
    assert output.loc[0, "k"] == 2
    assert output.loc[0, "shuffle_index"] == 0
    assert output.loc[0, "winner_seat"] in {"P1", "P2"}
    assert "winner" not in output.columns
    assert pq.read_schema(cfg.ingested_rows_raw(2)).equals(
        raw_simulation_schema_for(2),
        check_metadata=False,
    )
    validate_artifact_sidecar(
        cfg.ingested_rows_raw(2),
        expected={
            "scope": "by_k",
            "operation": "ingest_simulation_rows",
            "player_counts": [2],
        },
    )

    original = cfg.ingested_rows_raw(2).read_bytes()
    sidecar_path(cfg.ingested_rows_raw(2)).unlink()
    ingest.run(cfg)
    assert cfg.ingested_rows_raw(2).read_bytes() == original
    validate_artifact_sidecar(cfg.ingested_rows_raw(2))


def test_ingest_rejects_retired_winner_field_in_new_row_shard(tmp_path: Path) -> None:
    def authenticated_run(name: str, *, retired_winner_field: bool) -> AppConfig:
        cfg = _config(tmp_path, workers=1, name=name)
        table = _valid_row_table(cfg)
        if retired_winner_field:
            winner = table["winner_seat"]
            table = table.drop(["winner_seat"]).append_column("winner", winner)
        _publish_completed_row_run(cfg, table)
        return cfg

    control = authenticated_run("control", retired_winner_field=False)
    ingest.run(control)
    assert pq.read_table(control.ingested_rows_raw(2)).num_rows == 1

    cfg = authenticated_run("retired_winner", retired_winner_field=True)

    with pytest.raises(ValueError, match="noncanonical columns.*winner"):
        ingest.run(cfg)


@pytest.mark.parametrize(
    ("corruption", "oracle"),
    [
        ("wrong_root", "internal root/k/shuffle/batch identity mismatch"),
        ("wrong_k", "internal root/k/shuffle/batch identity mismatch"),
        ("wrong_shuffle", "internal root/k/shuffle/batch identity mismatch"),
        ("wrong_batch", "internal root/k/shuffle/batch identity mismatch"),
        ("wrong_rng_version", "internal version/namespace mismatch"),
        ("wrong_purpose_namespace", "internal version/namespace mismatch"),
        ("wrong_outcome_version", "internal version/namespace mismatch"),
        ("wrong_game_index", "game_index support must be 0\\.\\.0"),
        ("duplicate_game_key", "duplicate or invalid game key"),
        ("invalid_winner", "exactly one winner matching its rank-1 seat"),
        ("invalid_rank", "exactly one winner matching its rank-1 seat"),
        ("invalid_termination", "Safety-limit simulation row must set hit_safety_limit=true"),
        ("repeated_strategy", "must seat distinct strategies"),
        ("bad_victory_margin", "inconsistent victory_margin"),
        ("bad_loss_margin", "inconsistent P2_loss_margin"),
        ("missing_identity", "noncanonical columns.*misses required columns.*root_seed"),
        ("nonnumeric_strategy", "exact canonical raw schema"),
        ("negative_strategy", "P1_strategy.*within"),
    ],
)
def test_ingest_rejects_internally_malformed_row_shards(
    tmp_path: Path,
    corruption: str,
    oracle: str,
) -> None:
    control = _config(tmp_path, workers=1, name="control")
    _publish_completed_row_run(control)
    ingest.run(control)
    assert pq.read_table(control.ingested_rows_raw(2)).num_rows == 1

    cfg = _config(tmp_path, workers=1, name=f"corrupt_{corruption}")
    table = _valid_row_table(cfg)
    rows = table.to_pylist()
    row = rows[0]

    if corruption == "wrong_root":
        row["root_seed"] += 1
    elif corruption == "wrong_k":
        row["k"] = 3
    elif corruption == "wrong_shuffle":
        row["shuffle_index"] = 1
    elif corruption == "wrong_batch":
        row["deterministic_batch_id"] = 1
    elif corruption == "wrong_rng_version":
        row["rng_scheme_version"] += 1
    elif corruption == "wrong_purpose_namespace":
        row["rng_purpose_namespace"] += 1
    elif corruption == "wrong_outcome_version":
        row["outcome_schema_version"] += 1
    elif corruption == "wrong_game_index":
        row["game_index"] = 1
    elif corruption == "duplicate_game_key":
        rows.append(dict(row))
    elif corruption == "invalid_winner":
        row["winner_seat"] = "P3"
    elif corruption == "invalid_rank":
        row["P1_rank"] = 1
        row["P2_rank"] = 1
    elif corruption == "invalid_termination":
        row["termination_status"] = "safety_limit"
    elif corruption == "repeated_strategy":
        row["P2_strategy"] = row["P1_strategy"]
    elif corruption == "bad_victory_margin":
        row["victory_margin"] += 50
    elif corruption == "bad_loss_margin":
        row["P2_loss_margin"] += 50
    elif corruption == "negative_strategy":
        row["P1_strategy"] = -1

    rewritten = pa.Table.from_pylist(rows, schema=raw_simulation_schema_for(2))
    if corruption == "missing_identity":
        rewritten = rewritten.drop(["root_seed"])
    elif corruption == "nonnumeric_strategy":
        index = rewritten.schema.get_field_index("P1_strategy")
        rewritten = rewritten.set_column(
            index,
            "P1_strategy",
            pa.array(["11"], type=pa.string()),
        )
    _publish_completed_row_run(cfg, rewritten)

    with pytest.raises(ValueError, match=oracle):
        ingest.run(cfg)
