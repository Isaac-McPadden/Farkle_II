import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.analysis_config import PipelineCfg, expected_schema_for
from farkle.analysis.curate import (
    _already_curated,
    _schema_hash,
    _write_manifest,
)
from farkle.analysis.curate import (
    run as curate_run,
)
from farkle.app_config import AppConfig


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.table(
        {f.name: pa.array([], type=f.type) for f in schema},
        schema=schema,
    )


def test_schema_hash_known_value():
    assert (
        _schema_hash(2)
        == "8d6a2409c58593937b2a9b7c69d12ca745fd16ad064e7e201bbdd1bb7e3a69cf"
    )


def test_already_curated_schema_hash(tmp_path):
    cfg = PipelineCfg(results_dir=tmp_path)

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


def test_already_curated_manifest_failures(tmp_path):
    cfg = PipelineCfg(results_dir=tmp_path)
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


def test_run_new_layout(tmp_path):
    cfg = PipelineCfg(results_dir=tmp_path)
    raw_files = {}
    for n in (1, 2):
        schema = expected_schema_for(n)
        raw_path = cfg.ingested_rows_raw(n)
        pq.write_table(_empty_table(schema), raw_path)
        raw_files[n] = raw_path

    curate_run(cfg)

    for n, raw_path in raw_files.items():
        curated = cfg.ingested_rows_curated(n)
        manifest = cfg.manifest_for(n)
        assert curated.exists()
        assert manifest.exists()
        assert not raw_path.exists()
        meta = json.loads(manifest.read_text())
        assert meta["row_count"] == 0
        assert meta["schema_hash"] == _schema_hash(n)


def test_run_with_app_config(tmp_path):
    cfg = PipelineCfg(results_dir=tmp_path)
    app_cfg = AppConfig(analysis=cfg)

    schema = expected_schema_for(1)
    raw_path = cfg.ingested_rows_raw(1)
    pq.write_table(_empty_table(schema), raw_path)

    curate_run(app_cfg)

    curated = cfg.ingested_rows_curated(1)
    manifest = cfg.manifest_for(1)
    assert curated.exists()
    assert manifest.exists()
    assert not raw_path.exists()
    meta = json.loads(manifest.read_text())
    assert meta["row_count"] == 0
    assert meta["schema_hash"] == _schema_hash(1)


def test_run_legacy_missing_raw(tmp_path):
    cfg = PipelineCfg(results_dir=tmp_path)
    legacy_dir = cfg.analysis_dir / "data"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    curated = legacy_dir / cfg.curated_rows_name
    curated.write_text("")

    with pytest.raises(FileNotFoundError):
        curate_run(cfg)


def test_write_manifest_includes_config_sha(tmp_path):
    cfg = PipelineCfg(results_dir=tmp_path)
    cfg.config_sha = "abc123"
    manifest = tmp_path / "manifest.json"
    schema = expected_schema_for(1)

    _write_manifest(manifest, rows=5, schema=schema, cfg=cfg)

    meta = json.loads(manifest.read_text())
    assert meta["row_count"] == 5
    assert meta["compression"] == cfg.parquet_codec
    assert meta["config_sha"] == "abc123"


def test_already_curated_logs_schema_mismatch(tmp_path, caplog):
    cfg = PipelineCfg(results_dir=tmp_path)
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
    cfg = PipelineCfg(results_dir=tmp_path)
    legacy_dir = cfg.analysis_dir / "data"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    curated = legacy_dir / cfg.curated_rows_name
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
    cfg = PipelineCfg(results_dir=tmp_path)
    schema = expected_schema_for(1)
    curated = cfg.ingested_rows_curated(1)
    curated.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_empty_table(schema), curated)
    with pytest.raises(FileNotFoundError):
        curate_run(cfg)
