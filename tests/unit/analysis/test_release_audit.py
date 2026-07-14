from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from farkle.analysis.release_audit import (
    audit_retired_entry_points,
    audit_runnable_configs,
    audit_sidecar_completeness,
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
  artifact_contract_version: 2
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
