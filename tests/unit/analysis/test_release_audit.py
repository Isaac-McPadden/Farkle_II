from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.helpers.artifact_sidecars import (
    make_authenticated_v3_config,
    mutate_artifact_bytes,
    publish_v3_parquet,
)

import farkle.analysis.release_audit as release_audit
from farkle.analysis.release_audit import (
    audit_retired_entry_points,
    audit_runnable_configs,
    audit_sidecar_completeness,
    run_release_audits,
)
from farkle.config import AppConfig, ArtifactScope, IOConfig
from farkle.utils.artifact_contract import make_artifact_sidecar, sidecar_path
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.release_identity import write_v3_stage_completion


def _publish_indexed_outputs(
    tmp_path: Path,
    locations: list[tuple[ArtifactScope, int | None, str]],
    *,
    sources: tuple[Path, ...] = (),
) -> tuple[AppConfig, list[Path], Path]:
    cfg = make_authenticated_v3_config(tmp_path, player_counts=(2, 4))
    outputs: list[Path] = []
    for index, (scope, player_count, relative_path) in enumerate(locations):
        path = cfg.scope_path("metrics", scope, relative_path, k=player_count)
        publish_v3_parquet(
            cfg,
            path,
            pa.table({"value": [index]}),
            stage_key="metrics",
            producer="release_audit_test",
            operation="indexed_release_audit_test",
            sources=sources,
        )
        outputs.append(path)
    done_path = cfg.stage_dir("metrics") / "metrics.done.json"
    write_v3_stage_completion(
        done_path,
        cfg=cfg,
        stage_key="metrics",
        inputs=list(sources),
        outputs=outputs,
        status="success",
    )
    return cfg, outputs, done_path


def _mutate_completion_location(done_path: Path, **changes: Any) -> None:
    payload = json.loads(done_path.read_text(encoding="utf-8"))
    identity = payload["outputs"][0]["artifact"] or payload["outputs"][0]["manifest"]
    identity["location"].update(changes)
    done_path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_indexed_completion_canonical_lookup_succeeds(tmp_path: Path) -> None:
    cfg, _, _ = _publish_indexed_outputs(
        tmp_path,
        [(ArtifactScope.BY_K, 2, "nested/result.parquet")],
    )

    assert audit_sidecar_completeness(cfg.analysis_dir) == []


def test_indexed_completion_missing_output_fails(tmp_path: Path) -> None:
    cfg, outputs, _ = _publish_indexed_outputs(
        tmp_path,
        [(ArtifactScope.BY_K, 2, "missing.parquet")],
    )
    outputs[0].unlink()

    failures = audit_sidecar_completeness(cfg.analysis_dir)

    assert any("completion output is missing" in failure for failure in failures)


def test_duplicate_canonical_key_fails_during_index_construction(tmp_path: Path) -> None:
    cfg, outputs, _ = _publish_indexed_outputs(
        tmp_path,
        [(ArtifactScope.BY_K, 2, "duplicate.parquet")],
    )
    duplicate = cfg.analysis_dir / "unexpected_stage_copy" / "by_k" / "2p" / "duplicate.parquet"
    duplicate.parent.mkdir(parents=True)
    shutil.copyfile(outputs[0], duplicate)
    shutil.copyfile(sidecar_path(outputs[0]), sidecar_path(duplicate))

    _, failures = release_audit._build_audit_index(cfg.analysis_dir.resolve())

    assert len(failures) == 1
    assert "duplicate canonical artifact location" in failures[0]


def test_same_basename_in_different_scopes_resolves_independently(tmp_path: Path) -> None:
    cfg, _, _ = _publish_indexed_outputs(
        tmp_path,
        [
            (ArtifactScope.BY_K, 2, "shared.parquet"),
            (ArtifactScope.DIAGNOSTICS, None, "shared.parquet"),
        ],
    )

    assert audit_sidecar_completeness(cfg.analysis_dir) == []


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"scope": ArtifactScope.DIAGNOSTICS.value, "player_count": None}, "missing"),
        ({"player_count": 4}, "missing"),
        ({"relative_path": "nested/../result.parquet"}, "relative_path"),
        ({"relative_path": "nested//result.parquet"}, "canonical spelling"),
    ],
)
def test_completion_scope_k_or_noncanonical_relative_path_fails(
    tmp_path: Path,
    changes: dict[str, object],
    expected: str,
) -> None:
    cfg, _, done_path = _publish_indexed_outputs(
        tmp_path,
        [(ArtifactScope.BY_K, 2, "nested/result.parquet")],
    )
    _mutate_completion_location(done_path, **changes)

    failures = audit_sidecar_completeness(cfg.analysis_dir)

    assert any(expected in failure for failure in failures)


@pytest.mark.parametrize("mutation", ["bytes", "schema", "sidecar", "source"])
def test_indexed_lookup_preserves_content_schema_sidecar_and_source_failures(
    tmp_path: Path,
    mutation: str,
) -> None:
    source_cfg = make_authenticated_v3_config(
        tmp_path,
        name="source_fixture",
        player_counts=(2, 4),
    )
    source = source_cfg.scope_path("metrics", ArtifactScope.DIAGNOSTICS, "source.parquet")
    publish_v3_parquet(
        source_cfg,
        source,
        pa.table({"source_value": [1]}),
        stage_key="metrics",
        producer="release_audit_test",
        operation="indexed_release_audit_source",
    )
    cfg, outputs, _ = _publish_indexed_outputs(
        tmp_path,
        [(ArtifactScope.BY_K, 2, "authenticated.parquet")],
        sources=(source,),
    )
    # Put the source in the same audit root under its canonical metrics path so
    # the indexed graph authenticates the exact upstream binding.
    local_source = cfg.scope_path("metrics", ArtifactScope.DIAGNOSTICS, "source.parquet")
    local_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, local_source)
    shutil.copyfile(sidecar_path(source), sidecar_path(local_source))
    output = outputs[0]
    if mutation == "bytes":
        mutate_artifact_bytes(output)
    elif mutation == "schema":
        pq.write_table(pa.table({"changed_schema": [1]}), output)
    elif mutation == "sidecar":
        payload = json.loads(sidecar_path(output).read_text(encoding="utf-8"))
        payload["source_artifacts"][0]["logical_role"] = "changed_source_role"
        sidecar_path(output).write_text(json.dumps(payload), encoding="utf-8")
    else:
        publish_v3_parquet(
            cfg,
            local_source,
            pa.table({"source_value": [2]}),
            stage_key="metrics",
            producer="release_audit_test",
            operation="indexed_release_audit_source",
        )

    assert audit_sidecar_completeness(cfg.analysis_dir)


def test_many_output_audit_uses_one_tree_traversal_and_linear_lookups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_count = 128
    cfg, _, _ = _publish_indexed_outputs(
        tmp_path,
        [
            (ArtifactScope.BY_K, 2, f"many/output_{index:04d}.parquet")
            for index in range(output_count)
        ],
    )
    path_type = type(cfg.analysis_dir.resolve())
    original_rglob = path_type.rglob
    traversal_count = 0

    def counted_rglob(path: Path, pattern: str):
        nonlocal traversal_count
        traversal_count += 1
        return original_rglob(path, pattern)

    monkeypatch.setattr(path_type, "rglob", counted_rglob)

    index, build_failures = release_audit._build_audit_index(cfg.analysis_dir.resolve())
    validation_failures = release_audit._validate_audit_index(index)

    assert build_failures == []
    assert validation_failures == []
    assert traversal_count == 1
    assert len(index.by_location) == output_count
    assert index.lookup_count == output_count
