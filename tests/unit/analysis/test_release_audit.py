from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from farkle.analysis.release_audit import (
    audit_retired_entry_points,
    audit_runnable_configs,
    audit_sidecar_completeness,
    run_release_audits,
)
from farkle.config import AppConfig, ArtifactScope, IOConfig
from farkle.utils.artifact_contract import make_artifact_sidecar, sidecar_path
from farkle.utils.artifacts import write_parquet_artifact_atomic


def test_release_audits_accept_current_config_migration_and_sidecars(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        """
sim:
  n_players_list: [2]
screening:
  practical_delta_by_k: {2: 0.03}
  delta_across_k: 0.03
artifact_contract:
  artifact_contract_version: 3
  schema_version: 2
  estimand_version: 2
  conditioning_version: 2
""".strip()
        + "\n",
        encoding="utf-8",
    )
    assert audit_runnable_configs([config]) == []
    assert audit_retired_entry_points(repository) == []

    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    path = cfg.scope_path("metrics", ArtifactScope.DIAGNOSTICS, "example.parquet")
    table = pa.table({"value": [1]})
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=ArtifactScope.DIAGNOSTICS,
        source_scope=ArtifactScope.BY_K,
        operation="calculate_example_diagnostic",
        consistency_columns=table.schema.names,
        missing_cell_policy="not_applicable",
    )
    write_parquet_artifact_atomic(table, path, sidecar=sidecar)
    assert audit_sidecar_completeness(cfg.analysis_dir) == []

    sidecar_path(path).unlink()
    assert audit_sidecar_completeness(cfg.analysis_dir) == [f"missing sidecar: {path}"]


def test_release_audit_validates_stage_root_artifacts_with_sidecars(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    path = cfg.screening_path()
    table = pa.table({"value": [1]})
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="screening",
        scope=ArtifactScope.ACROSS_K,
        source_scope=ArtifactScope.BY_K,
        operation="equal_k_mean",
        consistency_columns=table.schema.names,
    )
    write_parquet_artifact_atomic(table, path, sidecar=sidecar)

    assert audit_sidecar_completeness(cfg.analysis_dir) == []
    assert cfg.screening_path().parent == cfg.across_k_dir("screening")
    sidecar_path(path).unlink()
    assert audit_sidecar_completeness(cfg.analysis_dir) == [f"missing sidecar: {path}"]


def test_release_audit_rejects_mixed_v2_v3_descendants(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    current = cfg.scope_path("metrics", ArtifactScope.DIAGNOSTICS, "current.parquet")
    table = pa.table({"value": [1]})
    write_parquet_artifact_atomic(
        table,
        current,
        sidecar=make_artifact_sidecar(
            cfg,
            current,
            producer="test",
            scope=ArtifactScope.DIAGNOSTICS,
            source_scope=ArtifactScope.BY_K,
            operation="current_v3",
            consistency_columns=table.schema.names,
            missing_cell_policy="not_applicable",
        ),
    )
    legacy_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"))
    legacy_cfg.artifact_contract.artifact_contract_version = 2
    legacy_cfg.artifact_contract.schema_version = 1
    legacy_cfg.artifact_contract.estimand_version = 1
    legacy_cfg.artifact_contract.conditioning_version = 1
    legacy = legacy_cfg.scope_path(
        "metrics",
        ArtifactScope.DIAGNOSTICS,
        "legacy.parquet",
    )
    write_parquet_artifact_atomic(
        table,
        legacy,
        sidecar=make_artifact_sidecar(
            legacy_cfg,
            legacy,
            producer="legacy_test",
            scope=ArtifactScope.DIAGNOSTICS,
            source_scope=ArtifactScope.BY_K,
            operation="legacy_v2",
            consistency_columns=table.schema.names,
            missing_cell_policy="not_applicable",
        ),
    )

    failures = audit_sidecar_completeness(cfg.analysis_dir)

    assert len(failures) == 1
    assert str(legacy) in failures[0]
    assert "contract exactly 3" in failures[0]


def test_release_preflight_requires_an_explicit_fresh_artifact_root(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    assert run_release_audits(
        repository,
        config_paths=[],
    ) == ["release audit requires at least one explicit fresh artifact root"]
