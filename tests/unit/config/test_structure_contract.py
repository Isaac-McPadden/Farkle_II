from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from farkle.config import (
    AnalysisConfig,
    AppConfig,
    ArtifactContractConfig,
    ArtifactScope,
    BatchingConfig,
    CombineConfig,
    Head2HeadConfig,
    HGBConfig,
    IngestConfig,
    IOConfig,
    KAggregationConfig,
    OrchestrationConfig,
    ResourcesConfig,
    RNGConfig,
    RobustnessConfig,
    ScreeningConfig,
    SimConfig,
    TrueSkillConfig,
    compute_config_sha,
    effective_config_dict,
    load_app_config,
)
from farkle.utils.progress import ProgressLogConfig

CONFIG_SECTION_TYPES = {
    "io": IOConfig,
    "sim": SimConfig,
    "analysis": AnalysisConfig,
    "ingest": IngestConfig,
    "combine": CombineConfig,
    "trueskill": TrueSkillConfig,
    "head2head": Head2HeadConfig,
    "hgb": HGBConfig,
    "orchestration": OrchestrationConfig,
    "resources": ResourcesConfig,
    "rng": RNGConfig,
    "screening": ScreeningConfig,
    "batching": BatchingConfig,
    "robustness": RobustnessConfig,
    "artifact_contract": ArtifactContractConfig,
    "k_aggregation": KAggregationConfig,
}

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


@pytest.mark.parametrize(
    "filename",
    ["default_config.yaml", "farkle_mega_config.yaml", "fast_config.yaml"],
)
def test_active_configuration_examples_use_the_current_contract(filename: str) -> None:
    config_path = CONFIG_DIR / filename
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


@pytest.mark.parametrize(
    "filename",
    ["blank_config.yaml", "default_config.yaml", "farkle_mega_config.yaml", "fast_config.yaml"],
)
def test_checked_in_config_templates_list_every_public_field(filename: str) -> None:
    """Keep every checked-in config synchronized with the typed public schema."""

    payload = yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))
    assert set(payload) == set(CONFIG_SECTION_TYPES)
    for section_name, section_type in CONFIG_SECTION_TYPES.items():
        assert set(payload[section_name]) == {
            config_field.name for config_field in dataclasses.fields(section_type)
        }

    for section_name in ("sim", "analysis"):
        assert set(payload[section_name]["progress_logging"]) == {
            config_field.name for config_field in dataclasses.fields(ProgressLogConfig)
        }


def test_default_config_materializes_exact_app_config_defaults() -> None:
    config = load_app_config(CONFIG_DIR / "default_config.yaml")
    assert compute_config_sha(config) == compute_config_sha(AppConfig())


@pytest.mark.parametrize(
    "filename",
    ["default_config.yaml", "farkle_mega_config.yaml", "fast_config.yaml"],
)
def test_runnable_configs_round_trip_under_public_schema(tmp_path: Path, filename: str) -> None:
    loaded = load_app_config(CONFIG_DIR / filename)
    round_trip_path = tmp_path / filename
    round_trip_path.write_text(
        yaml.safe_dump(effective_config_dict(loaded), sort_keys=False),
        encoding="utf-8",
    )

    assert effective_config_dict(load_app_config(round_trip_path)) == effective_config_dict(loaded)


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
