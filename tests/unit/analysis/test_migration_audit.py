from __future__ import annotations

import json
from pathlib import Path

import pytest

from farkle.analysis import migration_audit
from farkle.analysis.release_audit import _audit_scientific_leaf, _IndexedArtifact
from farkle.config import AppConfig, IOConfig
from farkle.utils.artifact_contract import sidecar_path, validate_artifact_sidecar
from farkle.utils.authenticated_contract import load_authenticated_sidecar
from farkle.utils.scientific_invocation import scientific_invocation
from farkle.utils.scientific_reuse import launch_matches


def test_migration_audit_inventories_retired_files_without_reading_or_deleting(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    retired = cfg.results_root / "archive" / "metrics_weighted.parquet"
    retired.parent.mkdir(parents=True, exist_ok=True)
    retired.write_bytes(b"forensic bytes")
    canonical = cfg.performance_across_k_path()
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"canonical")

    output = migration_audit.run(cfg)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["migration_report_version"] == 3
    assert payload["ignored_artifact_count"] == 1
    assert payload["ignored_artifacts"][0]["path"] == "archive/metrics_weighted.parquet"
    assert payload["artifacts_deleted"] is False
    assert payload["current_code_reads_ignored_artifacts"] is False
    assert retired.read_bytes() == b"forensic bytes"
    assert canonical.read_bytes() == b"canonical"
    validate_artifact_sidecar(
        output,
        expected={
            "scope": "diagnostics",
            "operation": "inventory_ignored_on_disk_artifacts",
        },
    )


def test_migration_audit_is_idempotent_when_inventory_is_unchanged(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    retired = cfg.results_root / "tiers.json"
    retired.parent.mkdir(parents=True, exist_ok=True)
    retired.write_text("{}", encoding="utf-8")

    output = migration_audit.run(cfg)
    first_mtime = output.stat().st_mtime_ns
    second = migration_audit.run(cfg)

    assert second == output
    assert second.stat().st_mtime_ns == first_mtime


def test_mixed_audit_requires_the_canonical_migration_inventory(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    with scientific_invocation([(cfg, "reporting")]):
        output = migration_audit.run(cfg)
        meta = load_authenticated_sidecar(output)
        assert launch_matches(meta.stage_identity)
        record = _IndexedArtifact(output, sidecar_path(output), meta.artifact.location, {}, False)
        digest = cfg.stage_config_sha("reporting")
        _audit_scientific_leaf(cfg, record, meta, digest)

        (cfg.results_root / "tiers.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="migration inventory differs"):
            _audit_scientific_leaf(cfg, record, meta, digest)
