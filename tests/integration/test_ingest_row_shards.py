"""Integration coverage for manifest-backed simulation row ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis import ingest
from farkle.config import AppConfig, IngestConfig, IOConfig, SimConfig
from farkle.simulation.simulation import _play_game
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils.artifact_contract import sidecar_path, validate_artifact_sidecar
from farkle.utils.manifest import append_manifest_line
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose


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
    if retired_winner_field:
        row["winner"] = row.pop("winner_seat")
    shard = row_dir / "rows_test_456.parquet"
    pq.write_table(pa.Table.from_pylist([row]), shard)
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
        },
    )
    (block / "simulation.done.json").write_text(
        json.dumps(
            {
                "root_seed": cfg.sim.seed,
                "n_players": 2,
                "rng_scheme_version": RNG_SCHEME_VERSION,
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
