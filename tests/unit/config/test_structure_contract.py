from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from farkle.config import (
    AppConfig,
    ArtifactContractConfig,
    ArtifactScope,
    IOConfig,
    load_app_config,
)


@pytest.mark.parametrize(
    "filename",
    ["default_config.yaml", "farkle_mega_config.yaml", "fast_config.yaml"],
)
def test_active_configuration_examples_use_the_current_contract(filename: str) -> None:
    config_path = Path(__file__).resolve().parents[3] / "configs" / filename
    cfg = load_app_config(config_path)
    cfg.validate_statistical_contract(
        require_two_roots=filename in {"farkle_mega_config.yaml", "fast_config.yaml"}
    )
    assert (
        cfg.artifact_contract.artifact_contract_version,
        cfg.rng.scheme_version,
        2,
        cfg.artifact_contract.schema_version,
        cfg.artifact_contract.estimand_version,
        cfg.artifact_contract.conditioning_version,
    ) == (3, 2, 2, 2, 2, 2)


def test_public_artifact_contract_defaults_and_freshness_use_exact_release_identity() -> None:
    contract = ArtifactContractConfig()
    cfg = AppConfig()

    assert (
        contract.artifact_contract_version,
        cfg.rng.scheme_version,
        cfg.freshness_key()["outcome_schema_version"],
        contract.schema_version,
        contract.estimand_version,
        contract.conditioning_version,
    ) == (3, 2, 2, 2, 2, 2)
    assert {
        key: cfg.freshness_key()[key]
        for key in (
            "artifact_contract_version",
            "rng_scheme_version",
            "outcome_schema_version",
            "schema_version",
            "estimand_version",
            "conditioning_version",
        )
    } == {
        "artifact_contract_version": 3,
        "rng_scheme_version": 2,
        "outcome_schema_version": 2,
        "schema_version": 2,
        "estimand_version": 2,
        "conditioning_version": 2,
    }


def test_public_validation_lock_rejects_v2_and_mixed_release_identities() -> None:
    cfg = AppConfig()
    cfg.artifact_contract.artifact_contract_version = 2
    with pytest.raises(ValueError, match="must be exactly 3"):
        cfg.validate_statistical_contract()

    cfg.artifact_contract.artifact_contract_version = 3
    for field in ("schema_version", "estimand_version", "conditioning_version"):
        setattr(cfg.artifact_contract, field, 1)
        with pytest.raises(ValueError, match="requires estimand_version=2"):
            cfg.validate_statistical_contract()
        setattr(cfg.artifact_contract, field, 2)


@pytest.mark.parametrize(
    ("payload", "retired_key"),
    [
        ({"sim": {"n_players": 2}}, "sim.n_players"),
        ({"io": {"analysis_dir": "analysis"}}, "io.analysis_dir"),
        ({"metrics": {"enabled": True}}, "metrics"),
        ({"head2head": {"games_per_pair": 10}}, "head2head.games_per_pair"),
    ],
)
def test_retired_configuration_keys_fail_with_named_replacements(
    tmp_path: Path,
    payload: dict[str, object],
    retired_key: str,
) -> None:
    path = tmp_path / "retired.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=rf"Retired config (?:key|section) '{retired_key}'",
    ):
        load_app_config(path)


def test_canonical_scope_paths_are_disjoint_and_removed_helpers_stay_absent(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    paths = {
        cfg.scope_dir("metrics", ArtifactScope.BY_K, k=2),
        cfg.scope_dir("metrics", ArtifactScope.CONCAT_KS),
        cfg.scope_dir("metrics", ArtifactScope.ACROSS_K),
        cfg.scope_dir("metrics", ArtifactScope.CROSS_SEED),
        cfg.scope_dir("metrics", ArtifactScope.DIAGNOSTICS),
        cfg.scope_dir("metrics", ArtifactScope.H2H_2P),
    }
    assert len(paths) == 6
    assert not hasattr(cfg, "combine_combined_dir")
    assert not hasattr(cfg, "resolve_input_stage_dir")
    assert not hasattr(cfg, "meta_analysis_dir")
