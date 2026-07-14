"""Test builders for valid derived-artifact sidecars."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from farkle.utils.artifact_contract import ArtifactSidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic


def sidecar_metadata(path: Path, *, scope: str = "concat_ks") -> ArtifactSidecar:
    """Return minimal valid metadata for a test artifact."""

    return ArtifactSidecar(
        artifact_contract_version=2,
        estimand_version=1,
        schema_version=1,
        artifact_name=path.name,
        producer="test_fixture",
        scope=scope,
        source_scope="by_k",
        operation="concatenate",
        method_contract={"kind": "operation", "procedure": "concatenate"},
        baseline="none",
        weighted_quantity="none",
        k_aggregation_method="none",
        k_weights=None,
        support_count_role="raw_support_provenance",
        uncertainty_method="none",
        replication_unit="none",
        conditioning="unconditional",
        consistency_columns=[],
        source_artifacts=[],
        grouping_keys=[],
        player_counts=[1],
        required_player_counts=[1],
        missing_cell_policy="not_applicable",
        seed_scope="single_root",
        rng_scheme_version=1,
        config_hash="test-config",
        input_manifest_hashes=[],
        code_revision="test-revision",
    )


def write_parquet_test_artifact(
    table: pa.Table, path: Path, *, scope: str = "concat_ks"
) -> None:
    """Write a test Parquet with a compatible adjacent sidecar."""

    write_parquet_artifact_atomic(table, path, sidecar=sidecar_metadata(path, scope=scope))
