from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pytest

import farkle.utils.artifact_contract as contract_module
from farkle.config import AppConfig, ArtifactScope, assign_config_sha
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    ArtifactSidecar,
    load_artifact_sidecar,
    make_artifact_sidecar,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.artifacts import read_parquet_artifact, write_parquet_artifact_atomic
from farkle.utils.stage_completion import stage_is_up_to_date, write_stage_done


def _metadata(path: Path, **changes: object) -> ArtifactSidecar:
    metadata = ArtifactSidecar(
        artifact_contract_version=1,
        estimand_version=1,
        schema_version=1,
        artifact_name=path.name,
        producer="unit_test",
        scope="by_k",
        source_scope="by_k",
        operation="aggregate",
        baseline="none",
        weighted_quantity="none",
        k_aggregation_method="none",
        k_weights=None,
        support_count_role="raw_support_provenance",
        uncertainty_method="none",
        replication_unit="deterministic_shuffle_batch",
        conditioning="unconditional",
        consistency_columns=["strategy_id"],
        source_artifacts=["source.parquet"],
        grouping_keys=["strategy_id"],
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="single_root",
        rng_scheme_version=1,
        config_hash="abc",
        input_manifest_hashes=[],
        code_revision="test-revision",
    )
    return replace(metadata, **changes)


def test_sidecar_name_is_adjacent_and_preserves_data_suffix(tmp_path: Path) -> None:
    artifact = tmp_path / "summary.parquet"
    assert sidecar_path(artifact) == tmp_path / "summary.parquet.sidecar.json"


def test_parquet_and_sidecar_are_hash_bound_and_readable(tmp_path: Path) -> None:
    artifact = tmp_path / "summary.parquet"
    table = pa.table({"strategy_id": [1, 2], "wins": [3, 4]})

    written = write_parquet_artifact_atomic(table, artifact, sidecar=_metadata(artifact))

    assert written.artifact_sha256 == sha256_file(artifact)
    assert written.artifact_size_bytes == artifact.stat().st_size
    assert load_artifact_sidecar(artifact) == written
    assert read_parquet_artifact(
        artifact,
        expected_sidecar={"scope": "by_k", "operation": "aggregate"},
    ).equals(table)


def test_missing_stale_and_incompatible_sidecars_are_hard_failures(tmp_path: Path) -> None:
    artifact = tmp_path / "summary.parquet"
    table = pa.table({"value": [1]})
    write_parquet_artifact_atomic(table, artifact, sidecar=_metadata(artifact))

    with pytest.raises(ArtifactContractError, match="incompatible sidecar"):
        validate_artifact_sidecar(artifact, expected={"scope": "across_k"})

    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ArtifactContractError, match="size does not match"):
        validate_artifact_sidecar(artifact)

    sidecar_path(artifact).unlink()
    with pytest.raises(ArtifactContractError, match="missing sidecar"):
        validate_artifact_sidecar(artifact)


def test_unknown_or_missing_sidecar_fields_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "summary.parquet"
    artifact.write_bytes(b"data")
    payload = _metadata(artifact).__dict__.copy()
    payload.pop("conditioning")
    sidecar_path(artifact).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactContractError, match="invalid sidecar"):
        load_artifact_sidecar(artifact)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"scope": "wrong"}, "unsupported artifact scope"),
        ({"k_aggregation_method": "declared_mapping", "k_weights": None}, "requires non-empty"),
        ({"k_weights": {"2": 1.0}}, "requires k_weights to be null"),
        ({"player_counts": [4, 2]}, "sorted unique"),
        ({"player_counts": [2], "required_player_counts": [2, 4]}, "support is incomplete"),
    ],
)
def test_semantically_invalid_metadata_is_rejected(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    artifact = tmp_path / "summary.parquet"
    with pytest.raises(ArtifactContractError, match=message):
        write_parquet_artifact_atomic(
            pa.table({"value": [1]}),
            artifact,
            sidecar=_metadata(artifact, **changes),
        )
    assert not artifact.exists()
    assert not sidecar_path(artifact).exists()


def test_config_factory_records_versions_support_sources_and_manifest_hashes(
    tmp_path: Path,
) -> None:
    cfg = AppConfig()
    assign_config_sha(cfg)
    artifact = tmp_path / "concat.parquet"
    manifest = tmp_path / "input.manifest.jsonl"
    manifest.write_text('{"rows": 2}\n', encoding="utf-8")

    metadata = make_artifact_sidecar(
        cfg,
        artifact,
        producer="combine",
        scope=ArtifactScope.CONCAT_KS,
        source_scope=ArtifactScope.BY_K,
        operation="concat",
        source_artifacts=[tmp_path / "2p.parquet", tmp_path / "4p.parquet"],
        player_counts=[4, 2],
        required_player_counts=[2, 4],
        missing_cell_policy="fail",
        input_manifests=[manifest],
    )

    assert metadata.scope == "concat_ks"
    assert metadata.source_scope == "by_k"
    assert metadata.player_counts == [2, 4]
    assert metadata.input_manifest_hashes == [sha256_file(manifest)]
    assert metadata.config_hash == cfg.config_sha


def test_completion_is_not_written_until_all_sidecars_validate(tmp_path: Path) -> None:
    artifact = tmp_path / "summary.parquet"
    done = tmp_path / "stage.done.json"
    artifact.write_bytes(b"unpaired")

    with pytest.raises(ArtifactContractError, match="missing sidecar"):
        write_stage_done(done, inputs=[], outputs=[artifact], sidecar_artifacts=[artifact])
    assert not done.exists()

    write_parquet_artifact_atomic(
        pa.table({"value": [1]}), artifact, sidecar=_metadata(artifact)
    )
    write_stage_done(
        done,
        inputs=[],
        outputs=[artifact],
        stage_config_sha="test-stage-config",
        sidecar_artifacts=[artifact],
    )
    assert done.exists()

    sidecar_path(artifact).unlink()
    assert not stage_is_up_to_date(
        done,
        inputs=[],
        outputs=[artifact],
        stage_config_sha="test-stage-config",
        sidecar_artifacts=[artifact],
    )


def test_interruption_between_data_and_sidecar_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "summary.parquet"
    metadata_path = sidecar_path(artifact)
    real_replace = contract_module.os.replace

    def fail_sidecar_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == metadata_path:
            raise OSError("simulated interruption")
        real_replace(source, destination)

    monkeypatch.setattr(contract_module.os, "replace", fail_sidecar_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        write_parquet_artifact_atomic(
            pa.table({"value": [1]}), artifact, sidecar=_metadata(artifact)
        )

    assert artifact.exists()
    assert not metadata_path.exists()
    with pytest.raises(ArtifactContractError, match="missing sidecar"):
        validate_artifact_sidecar(artifact)
