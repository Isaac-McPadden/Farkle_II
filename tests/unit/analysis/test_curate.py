# ruff: noqa: ARG005
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.curate import _already_curated, _schema_hash, _write_manifest
from farkle.analysis.curate import run as curate_run
from farkle.analysis.schema import expected_schema_for
from farkle.config import AppConfig


def _make_cfg(results_dir: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.io.results_dir = results_dir
    return cfg


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.table({field.name: pa.array([], type=field.type) for field in schema}, schema=schema)


def test_schema_hash_known_value():
    assert _schema_hash(2) == "8d6a2409c58593937b2a9b7c69d12ca745fd16ad064e7e201bbdd1bb7e3a69cf"


def test_schema_hash_uses_schema_serialize_when_pa_ipc_missing(monkeypatch):
    class DummySchema:
        def __init__(self) -> None:
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
    monkeypatch.setattr("farkle.analysis.curate.expected_schema_for", lambda _: dummy_schema)

    result = _schema_hash(3)

    assert dummy_schema.serialize_calls == 1
    assert result == hashlib.sha256(b"dummy-bytes").hexdigest()


def test_already_curated_schema_hash(tmp_path):
    cfg = _make_cfg(tmp_path)

    schema0 = expected_schema_for(0)
    table1 = pa.table(
        {
            "winner_seat": ["P1"],
            "winner_strategy": ["none"],
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

    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, rows=0, schema=schema, cfg=cfg)
    missing_parquet = tmp_path / "missing.parquet"
    assert not _already_curated(missing_parquet, manifest)

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
            "winner_strategy": ["none"],
            "seat_ranks": [[]],
            "winning_score": [100],
            "n_rounds": [1],
        },
        schema=schema,
    )
    file = tmp_path / "file.parquet"
    pq.write_table(table, file)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("not json")
    assert not _already_curated(file, manifest)

    manifest.write_text(json.dumps({"row_count": 1}))
    assert not _already_curated(file, manifest)


def test_write_manifest_includes_config_sha(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path)
    monkeypatch.setattr(cfg, "config_sha", "abc123", raising=False)
    schema = expected_schema_for(1)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, rows=10, schema=schema, cfg=cfg)
    meta = json.loads(manifest.read_text())
    assert meta["config_sha"] == "abc123"


def test_run_new_layout(tmp_path):
    cfg = _make_cfg(tmp_path)
    schema = expected_schema_for(1)
    raw_path = cfg.ingested_rows_raw(1)
    pq.write_table(_empty_table(schema), raw_path)

    curate_run(cfg)

    curated_path = cfg.ingested_rows_curated(1)
    manifest_path = cfg.manifest_for(1)
    assert curated_path.exists()
    assert manifest_path.exists()
    assert not raw_path.exists()
    meta = json.loads(manifest_path.read_text())
    assert meta["row_count"] == 0


def test_run_new_layout_missing_manifest(tmp_path):
    cfg = _make_cfg(tmp_path)
    schema = expected_schema_for(1)
    curated = cfg.ingested_rows_curated(1)
    curated.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_empty_table(schema), curated)
    with pytest.raises(FileNotFoundError):
        curate_run(cfg)


def test_run_legacy_finalises_raw_file(tmp_path):
    cfg = _make_cfg(tmp_path)
    dst_file = cfg.curated_parquet
    raw_file = dst_file.with_suffix(".raw.parquet")
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    schema = expected_schema_for(0)
    pq.write_table(_empty_table(schema), raw_file)

    curate_run(cfg)

    manifest = cfg.analysis_dir / cfg.analysis.manifest_name
    assert dst_file.exists()
    assert not raw_file.exists()
    meta = json.loads(manifest.read_text())
    assert meta["row_count"] == 0
    assert meta["schema_hash"] == _schema_hash(0)


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
    pq.write_table(_empty_table(schema_new), raw_path)

    curate_run(cfg)

    curated_new = cfg.ingested_rows_curated(1)
    manifest_new = cfg.manifest_for(1)
    assert curated_new.exists()
    assert manifest_new.exists()
    assert not raw_path.exists()
    assert curated_existing.exists()
    assert manifest_existing.exists()


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


def test_run_legacy_already_curated(tmp_path):
    cfg = _make_cfg(tmp_path)
    legacy_dir = cfg.analysis_dir / "data"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    curated = legacy_dir / cfg.analysis.curated_rows_name
    schema = expected_schema_for(0)
    pq.write_table(_empty_table(schema), curated)
    manifest = cfg.analysis_dir / cfg.analysis.manifest_name
    _write_manifest(manifest, rows=0, schema=schema, cfg=cfg)
    assert _already_curated(curated, manifest)

    curate_run(cfg)

    assert curated.exists()
    assert manifest.exists()
    assert not curated.with_suffix(".raw.parquet").exists()
