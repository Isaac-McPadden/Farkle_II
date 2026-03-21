from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis import combine
from farkle.config import AppConfig, IOConfig
from farkle.utils.schema_helpers import expected_schema_for


def _write_curated(path: Path, schema: pa.Schema, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_migrate_combined_output_moves_manifest_when_present(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    legacy_dir = cfg.combine_stage_dir / f"{cfg.combine_max_players}p" / "pooled"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_file = legacy_dir / "all_ingested_rows.parquet"
    legacy_manifest = legacy_file.with_suffix(".manifest.jsonl")
    legacy_file.write_bytes(b"legacy")
    legacy_manifest.write_text('{"path":"legacy"}\n', encoding="utf-8")

    migrated = combine._migrate_combined_output(cfg)

    assert migrated.read_bytes() == b"legacy"
    assert (
        migrated.with_suffix(".manifest.jsonl").read_text(encoding="utf-8")
        == '{"path":"legacy"}\n'
    )
    assert not legacy_file.exists()
    assert not legacy_manifest.exists()


def test_write_partitioned_dataset_skips_invalid_dirs_and_reuses_uptodate_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    valid = cfg.ingested_rows_curated(2)
    invalid = tmp_path / "results" / "analysis" / "bogus" / cfg.curated_rows_name
    _write_curated(
        valid,
        expected_schema_for(2),
        [
            {
                "winner_seat": "P1",
                "winner_strategy": 1,
                "game_seed": 10,
                "seat_ranks": ["P1", "P2"],
                "n_rounds": 5,
                "winning_score": 200,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_rank": 1,
                "P2_rank": 2,
            }
        ],
    )
    _write_curated(invalid, expected_schema_for(1), [])

    monkeypatch.setattr(combine, "stage_is_up_to_date", lambda *args, **kwargs: True)

    outputs, manifests = combine._write_partitioned_dataset(
        cfg,
        [invalid, valid],
        expected_schema_for(12),
    )

    expected_output, expected_manifest = combine._partition_paths(cfg, 2)
    assert outputs == [expected_output]
    assert manifests == [expected_manifest]


def test_write_partitioned_dataset_cleans_zero_row_outputs(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    src = cfg.ingested_rows_curated(2)
    _write_curated(src, expected_schema_for(2), [])

    out_file, manifest_path = combine._partition_paths(cfg, 2)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("stale", encoding="utf-8")
    manifest_path.write_text("stale\n", encoding="utf-8")

    outputs, manifests = combine._write_partitioned_dataset(
        cfg,
        [src],
        expected_schema_for(12),
    )

    assert outputs == []
    assert manifests == []
    assert not out_file.exists()
    assert not manifest_path.exists()


def test_write_monolithic_compatibility_skips_empty_batches_and_drops_n_players(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))

    empty_batch = pa.record_batch(
        [
            pa.array([], type=pa.int64()),
            pa.array([], type=pa.int64()),
        ],
        names=["winner_strategy", "n_players"],
    )
    populated_batch = pa.record_batch(
        [
            pa.array([1], type=pa.int64()),
            pa.array([2], type=pa.int64()),
        ],
        names=["winner_strategy", "n_players"],
    )

    class DummyScanner:
        def to_batches(self):
            return iter([empty_batch, populated_batch])

    class DummyDataset:
        def scanner(self, **kwargs):
            return DummyScanner()

    captured_tables: list[pa.Table] = []

    def fake_run_streaming_shard(**kwargs) -> None:
        captured_tables.extend(list(kwargs["batch_iter"]))

    monkeypatch.setattr(combine.ds, "dataset", lambda *args, **kwargs: DummyDataset())
    monkeypatch.setattr(combine, "run_streaming_shard", fake_run_streaming_shard)

    total = combine._write_monolithic_compatibility_from_partitions(
        cfg,
        tmp_path / "combined.parquet",
        tmp_path / "combined.manifest.jsonl",
    )

    assert total == 1
    assert len(captured_tables) == 1
    assert "n_players" not in captured_tables[0].column_names
