# ruff: noqa: ARG005
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import farkle.analysis.curate as curate
from farkle.analysis.curate import (
    _already_curated,
    _schema_hash,
    _write_manifest,
)
from farkle.analysis.curate import (
    run as curate_run,
)
from farkle.config import AppConfig, IOConfig
from farkle.utils.schema_helpers import expected_schema_for


def _make_cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(io=IOConfig(results_dir_prefix=tmp_path))


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.table(
        {f.name: pa.array([], type=f.type) for f in schema},
        schema=schema,
    )


def _canonical_schema_hash(n_players: int) -> str:
    """Mirror curate's canonical schema serialization path for hash assertions."""
    schema = expected_schema_for(n_players)
    pa_serialize = getattr(pa.ipc, "serialize", None)
    if pa_serialize is not None:
        serialized = pa_serialize(schema).to_buffer().to_pybytes()
    else:
        serialized = schema.serialize().to_pybytes()
    return hashlib.sha256(serialized).hexdigest()


def test_schema_hash_known_value():
    expected = _canonical_schema_hash(2)
    assert expected == "81da1a048aa54451b2f1d7a93e945aa6adc061d416261f0ecbe4f29f38ac8902"
    assert _schema_hash(2) == expected


def test_schema_hash_uses_schema_serialize_when_pa_ipc_missing(monkeypatch):
    class DummySchema:
        def __init__(self):
            self.serialize_calls = 0

        def serialize(self):
            self.serialize_calls += 1

            class DummyBuffer:
                @staticmethod
                def to_pybytes():
                    return b"dummy-bytes"

            return DummyBuffer()

    dummy_schema = DummySchema()

    monkeypatch.setattr(pa.ipc, "serialize", None, raising=False)
    monkeypatch.setattr(
        "farkle.analysis.curate.expected_schema_for",
        lambda n_players: dummy_schema,
    )

    result = _schema_hash(3)

    assert dummy_schema.serialize_calls == 1
    assert result == hashlib.sha256(b"dummy-bytes").hexdigest()


def test_already_curated_schema_hash(tmp_path):
    cfg = _make_cfg(tmp_path)

    schema0 = expected_schema_for(0)
    table1 = pa.table(
        {
            "winner_seat": ["P1"],
            "winner_strategy": [0],
            "game_seed": [42],
            "seat_ranks": [[]],
            "winning_score": [100],
            "n_rounds": [1],
        },
        schema=schema0,
    )
    file1 = tmp_path / "file1.parquet"
    pq.write_table(table1, file1)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, rows=1, schema=schema0, cfg=cfg)

    assert _already_curated(file1, manifest)

    table2 = pa.table(
        {
            "winner_seat": ["P1"],
            "winning_score": [100],
            "n_rounds": [1],
            "P1_score": [100],
        }
    )
    file2 = tmp_path / "file2.parquet"
    pq.write_table(table2, file2)

    assert not _already_curated(file2, manifest)


def test_already_curated_missing_files(tmp_path):
    cfg = _make_cfg(tmp_path)
    schema = expected_schema_for(1)

    # Missing parquet file
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, rows=0, schema=schema, cfg=cfg)
    missing_parquet = tmp_path / "missing.parquet"
    assert not _already_curated(missing_parquet, manifest)

    # Missing manifest file
    table = _empty_table(schema)
    parquet_path = tmp_path / "curated.parquet"
    pq.write_table(table, parquet_path)
    missing_manifest = tmp_path / "missing.json"
    assert not _already_curated(parquet_path, missing_manifest)


def test_already_curated_manifest_failures(tmp_path):
    cfg = _make_cfg(tmp_path)
    schema = expected_schema_for(0)

    table = pa.table(
        {
            "winner_seat": ["P1"],
            "winner_strategy": [0],
            "game_seed": [42],
            "seat_ranks": [[]],
            "winning_score": [100],
            "n_rounds": [1],
        },
        schema=schema,
    )
    file = tmp_path / "file.parquet"
    pq.write_table(table, file)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, rows=1, schema=schema, cfg=cfg)
    assert _already_curated(file, manifest)

    manifest.write_text("not json")
    assert not _already_curated(file, manifest)

    _write_manifest(manifest, rows=99, schema=schema, cfg=cfg)
    assert not _already_curated(file, manifest)

    _write_manifest(manifest, rows=1, schema=schema, cfg=cfg)
    meta = json.loads(manifest.read_text())
    meta.pop("schema_hash", None)
    manifest.write_text(json.dumps(meta))
    assert not _already_curated(file, manifest)

    _write_manifest(manifest, rows=1, schema=schema, cfg=cfg)
    meta = json.loads(manifest.read_text())
    meta.pop("row_count", None)
    manifest.write_text(json.dumps(meta))
    assert not _already_curated(file, manifest)


def test_run_new_layout(tmp_path):
    cfg = _make_cfg(tmp_path)
    raw_files = {}
    for n in (1, 2):
        schema = expected_schema_for(n)
        raw_path = cfg.ingested_rows_raw(n)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_empty_table(schema), raw_path)
        raw_files[n] = raw_path

    curate_run(cfg)

    for n, raw_path in raw_files.items():
        curated = cfg.ingested_rows_curated(n)
        manifest = cfg.manifest_for(n)
        assert curated.exists()
        assert manifest.exists()
        assert raw_path.exists()
        meta = json.loads(manifest.read_text())
        assert meta["row_count"] == 0
        assert meta["schema_hash"] == _schema_hash(n)


def test_run_with_app_config(tmp_path):
    cfg = _make_cfg(tmp_path)

    schema = expected_schema_for(1)
    raw_path = cfg.ingested_rows_raw(1)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_empty_table(schema), raw_path)

    curate_run(cfg)

    curated = cfg.ingested_rows_curated(1)
    manifest = cfg.manifest_for(1)
    assert curated.exists()
    assert manifest.exists()
    assert raw_path.exists()
    meta = json.loads(manifest.read_text())
    assert meta["row_count"] == 0
    assert meta["schema_hash"] == _schema_hash(1)


def test_run_legacy_missing_raw(tmp_path):
    cfg = _make_cfg(tmp_path)
    legacy_dir = cfg.analysis_dir / "data"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    curated = legacy_dir / cfg.curated_rows_name
    curated.write_text("")

    with pytest.raises(FileNotFoundError):
        curate_run(cfg)


def test_write_manifest_includes_config_sha(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    monkeypatch.setattr(cfg, "config_sha", "abc123", raising=False)
    manifest = tmp_path / "manifest.json"
    schema = expected_schema_for(1)

    _write_manifest(manifest, rows=5, schema=schema, cfg=cfg)

    meta = json.loads(manifest.read_text())
    assert meta["row_count"] == 5
    assert meta["compression"] == cfg.parquet_codec
    assert meta["config_sha"] == "abc123"


def test_already_curated_logs_schema_mismatch(tmp_path, caplog):
    cfg = _make_cfg(tmp_path)
    schema = expected_schema_for(1)
    table = _empty_table(schema)
    parquet_path = tmp_path / "file.parquet"
    pq.write_table(table, parquet_path)
    manifest = tmp_path / "manifest.json"

    _write_manifest(manifest, rows=0, schema=schema, cfg=cfg)
    meta = json.loads(manifest.read_text())
    meta["schema_hash"] = "tampered"
    manifest.write_text(json.dumps(meta))

    with caplog.at_level("INFO"):
        assert not _already_curated(parquet_path, manifest)

    assert any("Curate schema mismatch detected" in message for message in caplog.messages)


def test_run_legacy_already_curated(tmp_path):
    cfg = _make_cfg(tmp_path)
    curated = cfg.curated_parquet
    curated.parent.mkdir(parents=True, exist_ok=True)
    schema = expected_schema_for(0)
    pq.write_table(_empty_table(schema), curated)
    manifest = cfg.analysis_dir / cfg.manifest_name
    _write_manifest(manifest, rows=0, schema=schema, cfg=cfg)
    assert _already_curated(curated, manifest)

    curate_run(cfg)

    assert curated.exists()
    assert manifest.exists()
    assert not curated.with_suffix(".raw.parquet").exists()


def test_run_new_layout_missing_manifest(tmp_path):
    cfg = _make_cfg(tmp_path)
    schema = expected_schema_for(1)
    curated = cfg.ingested_rows_curated(1)
    curated.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_empty_table(schema), curated)
    with pytest.raises(FileNotFoundError):
        curate_run(cfg)


def test_schema_hash_prefers_pa_ipc_serialize(monkeypatch):
    class DummySchema:
        def __init__(self):
            self.serialize_calls = 0

        def serialize(self):
            self.serialize_calls += 1
            raise AssertionError(
                "schema.serialize should not run when pa.ipc.serialize is available"
            )

    buffer_bytes = b"pa-ipc-bytes"

    def fake_serialize(schema):
        assert schema is dummy_schema

        class DummySerialized:
            def to_buffer(self):
                class DummyBuffer:
                    @staticmethod
                    def to_pybytes():
                        return buffer_bytes

                return DummyBuffer()

        return DummySerialized()

    dummy_schema = DummySchema()
    monkeypatch.setattr(
        "farkle.analysis.curate.expected_schema_for",
        lambda n_players: dummy_schema,
    )
    monkeypatch.setattr(pa.ipc, "serialize", fake_serialize, raising=False)

    result = _schema_hash(4)

    assert dummy_schema.serialize_calls == 0
    assert result == hashlib.sha256(buffer_bytes).hexdigest()


def test_already_curated_handles_metadata_error(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    schema = expected_schema_for(1)
    parquet_path = tmp_path / "broken.parquet"
    pq.write_table(_empty_table(schema), parquet_path)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, rows=0, schema=schema, cfg=cfg)

    def boom(_):
        raise RuntimeError("broken metadata")

    monkeypatch.setattr("farkle.analysis.curate.pq.read_metadata", boom)
    assert not _already_curated(parquet_path, manifest)


def test_run_existing_curated_manifest_allows_proceed(tmp_path):
    cfg = _make_cfg(tmp_path)

    schema_existing = expected_schema_for(2)
    curated_existing = cfg.ingested_rows_curated(2)
    curated_existing.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_empty_table(schema_existing), curated_existing)
    manifest_existing = cfg.manifest_for(2)
    _write_manifest(manifest_existing, rows=0, schema=schema_existing, cfg=cfg)

    schema_new = expected_schema_for(1)
    raw_path = cfg.ingested_rows_raw(1)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_empty_table(schema_new), raw_path)

    curate_run(cfg)

    curated_new = cfg.ingested_rows_curated(1)
    manifest_new = cfg.manifest_for(1)
    assert curated_new.exists()
    assert manifest_new.exists()
    assert raw_path.exists()
    assert curated_existing.exists()
    assert manifest_existing.exists()


def test_run_legacy_finalizes_raw_file(tmp_path):
    cfg = _make_cfg(tmp_path)
    dst_file = cfg.curated_parquet
    raw_file = dst_file.with_suffix(".raw.parquet")
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    schema = expected_schema_for(0)
    pq.write_table(_empty_table(schema), raw_file)

    curate_run(cfg)

    manifest = cfg.analysis_dir / cfg.manifest_name
    assert dst_file.exists()
    assert not raw_file.exists()
    meta = json.loads(manifest.read_text())
    assert meta["row_count"] == 0
    assert meta["schema_hash"] == _schema_hash(0)


def test_migrate_raw_and_curated_outputs(tmp_path):
    cfg = _make_cfg(tmp_path)

    legacy_raw = cfg.combine_stage_dir / "2p" / "2p_ingested_rows.raw.parquet"
    legacy_raw.parent.mkdir(parents=True, exist_ok=True)
    legacy_raw.write_bytes(b"raw")
    legacy_manifest = legacy_raw.with_suffix(".manifest.jsonl")
    legacy_manifest.write_text("{}")

    curated_legacy = cfg.combine_stage_dir / "3p" / cfg.curated_rows_name
    curated_legacy.parent.mkdir(parents=True, exist_ok=True)
    curated_legacy.write_bytes(b"curated")
    curated_manifest = curated_legacy.parent / cfg.manifest_name
    curated_manifest.write_text("{}")

    curate._migrate_raw_inputs(cfg)
    curate._migrate_curated_outputs(cfg)

    assert cfg.ingested_rows_raw(2).exists()
    assert cfg.ingest_manifest(2).exists()
    assert not legacy_raw.exists()
    assert cfg.ingested_rows_curated(3).exists()
    assert cfg.manifest_for(3).exists()
