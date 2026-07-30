"""Integration coverage for manifest-backed simulation row ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis import ingest
from farkle.config import AppConfig, IngestConfig, IOConfig, SimConfig
from farkle.simulation.simulation import _play_game, simulation_rows_to_table
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils.artifact_contract import sidecar_path, validate_artifact_sidecar
from farkle.utils.manifest import append_manifest_line
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose
from farkle.utils.schema_helpers import raw_simulation_schema_for


def _write_completed_row_run(cfg: AppConfig, *, retired_winner_field: bool = False) -> Path:
    block = cfg.n_dir(2)
    row_dir = cfg.simulation_row_dir(2)
    assert row_dir is not None
    row_dir.mkdir(parents=True)
    strategies = [
        ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=11),
        ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=12),
    ]
    row = dict(
        _play_game(
            123,
            strategies,
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
    shard = row_dir / "rows_test_456.parquet"
    table = simulation_rows_to_table([row], 2)
    if retired_winner_field:
        winner = table["winner_seat"]
        table = table.drop(["winner_seat"]).append_column("winner", winner)
    pq.write_table(table, shard)
    append_manifest_line(
        row_dir / "manifest.jsonl",
        {
            "path": shard.name,
            "rows": 1,
            "root_seed": cfg.sim.seed,
            "n_players": 2,
            "shuffle_index": 0,
            "shuffle_seed": 456,
            "deterministic_batch_id": 0,
            "rng_scheme_version": RNG_SCHEME_VERSION,
            "outcome_schema_version": 2,
            "tournament_method_version": 2,
        },
    )
    (block / "simulation.done.json").write_text(
        json.dumps(
            {
                "root_seed": cfg.sim.seed,
                "n_players": 2,
                "rng_scheme_version": RNG_SCHEME_VERSION,
                "outcome_schema_version": 2,
                "tournament_method_version": 2,
                "shuffle_index_start": 0,
                "shuffle_index_end": 0,
                "shuffles_per_batch": 1,
                "outputs": [str(row_dir)],
            }
        ),
        encoding="utf-8",
    )
    return shard


def _config(tmp_path: Path, *, workers: int) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "review"),
        sim=SimConfig(seed=7, seed_list=[7], n_players_list=[2], row_dir=Path("rows")),
        ingest=IngestConfig(n_jobs=workers),
    )
    cfg.analysis.mp_start_method = "spawn"
    return cfg


def test_ingest_reads_manifest_backed_row_directory_through_spawn_worker(tmp_path: Path) -> None:
    cfg = _config(tmp_path, workers=2)
    _write_completed_row_run(cfg)

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
    cfg = _config(tmp_path, workers=1)
    _write_completed_row_run(cfg, retired_winner_field=True)

    with pytest.raises(ValueError, match="noncanonical columns.*winner"):
        ingest.run(cfg)


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong_root",
        "wrong_k",
        "wrong_shuffle",
        "wrong_batch",
        "wrong_game_index",
        "duplicate_game_key",
        "invalid_winner",
        "invalid_rank",
        "invalid_termination",
        "repeated_strategy",
        "bad_victory_margin",
        "bad_loss_margin",
        "missing_identity",
        "nonnumeric_strategy",
    ],
)
def test_ingest_rejects_internally_malformed_row_shards(
    tmp_path: Path,
    corruption: str,
) -> None:
    cfg = _config(tmp_path, workers=1)
    shard = _write_completed_row_run(cfg)
    table = pq.read_table(shard)
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
    pq.write_table(rewritten, shard)

    if corruption == "duplicate_game_key":
        manifest_path = shard.parent / "manifest.jsonl"
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        record["rows"] = 2
        manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        ingest.run(cfg)
