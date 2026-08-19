from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from farkle.config import AppConfig, ProfileConfig, compute_config_sha, load_app_config
from farkle.orchestration.profile_metadata import resolved_profile_metadata

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def test_fast_config_resolves_exact_reduced_integration_workload() -> None:
    cfg = load_app_config(CONFIG_DIR / "fast_config.yaml", seed_list_len=2)
    cfg.validate_statistical_contract(require_two_roots=True)

    metadata = resolved_profile_metadata(cfg)

    assert metadata["purpose"] == "integration"
    assert metadata["reduced_resolution"] is True
    assert metadata["production_eligible"] is False
    assert metadata["release_eligible"] is False
    assert metadata["strategy_count"] == 80
    assert metadata["roots"] == [48, 49]
    assert metadata["player_counts"] == [2, 4, 5]
    assert metadata["configured_resolution"] == pytest.approx(0.08)
    assert metadata["bootstrap_replicates"] == 500
    assert metadata["rng_partitions"] == 8
    assert metadata["candidate_contribution_size_by_method"] == {
        "win_rate": 8,
        "trueskill": 8,
    }
    assert metadata["frozen_candidate_cap"] == 12
    assert metadata["maximum_candidate_count"] == 12
    assert metadata["maximum_pair_count"] == 66

    workload = metadata["workload_by_k"]
    assert set(workload) == {"2", "4", "5"}
    assert {cell["required_shuffles_unrounded"] for cell in workload.values()} == {597}
    assert {cell["required_shuffles"] for cell in workload.values()} == {3_000}
    assert {cell["batch_count"] for cell in workload.values()} == {100}
    assert {cell["shuffles_per_batch"] for cell in workload.values()} == {30}
    assert all(
        cell["achieved_resolution"] == pytest.approx(0.03576099446779917)
        for cell in workload.values()
    )
    assert {k: cell["required_games_per_root"] for k, cell in workload.items()} == {
        "2": 120_000,
        "4": 60_000,
        "5": 48_000,
    }

    h2h = metadata["projected_h2h"]
    assert h2h["candidate_count"] == 12
    assert h2h["unordered_pair_count"] == 66
    assert h2h["root_count"] == 2
    assert h2h["seat_orientation_count"] == 2
    assert h2h["n_completed_required_per_root_order_block"] == 1_372
    assert h2h["total_block_count"] == 264
    assert h2h["planned_completed_games"] == 362_208
    assert h2h["maximum_total_attempts"] == 724_416
    assert h2h["maximum_total_attempts"] <= cfg.head2head.total_game_cap
    assert cfg.resources.scheduler_memory_budget_mb == 8_192
    assert cfg.resources.process_tree_warning_threshold_mb == 8_192
    assert cfg.resources.aggregate_memory_hard_limit_mb == 12_288
    assert cfg.resources.minimum_system_available_memory_mb == 4_096
    assert cfg.resources.parent_process_memory_mb == 512
    assert cfg.resources.logical_cpu_budget == 15
    assert cfg.resources.native_threads_per_worker == 1
    assert compute_config_sha(cfg) == (
        "fe0e55de5613c2a296030c1fdc64c410a69131b70620e74a3d37430037c9fb3c"
    )
    with_old_host_reserve = replace(
        cfg,
        resources=replace(cfg.resources, minimum_system_available_memory_mb=2_048),
    )
    assert compute_config_sha(with_old_host_reserve) == compute_config_sha(cfg)


@pytest.mark.parametrize(
    ("filename", "expected_hash", "expected_player_counts"),
    [
        (
            "default_config.yaml",
            "2d00a21645ee90abe6c15505d984b5545aa555125835d6ff920a39730949ab80",
            [5],
        ),
        (
            "farkle_mega_config.yaml",
            "8df3969cc051aef72a541437ef803c924013a971ee8074e750f018d50d8cd7e4",
            [2, 3, 4, 5, 6, 8, 10, 12],
        ),
    ],
)
def test_production_configs_retain_statistical_settings_and_identity(
    filename: str,
    expected_hash: str,
    expected_player_counts: list[int],
) -> None:
    cfg = load_app_config(CONFIG_DIR / filename)

    assert cfg.profile == ProfileConfig()
    assert cfg.sim.n_players_list == expected_player_counts
    assert cfg.screening.resolution_delta == pytest.approx(0.03)
    assert cfg.screening.bootstrap_replicates == 2_000
    assert cfg.analysis.rng_diagnostic_partitions == 32
    assert cfg.screening.candidate_contribution_size == 75
    assert cfg.head2head.candidate_cap is None
    assert compute_config_sha(cfg) == expected_hash


def test_profile_labels_are_outside_statistical_compute_identity_but_bound_to_publishers() -> None:
    baseline = AppConfig()
    integration = replace(
        baseline,
        profile=ProfileConfig(
            purpose="integration",
            reduced_resolution=True,
            production_eligible=False,
            release_eligible=False,
        ),
    )

    assert compute_config_sha(baseline) == compute_config_sha(integration)
    assert baseline.stage_config_sha("simulation") == integration.stage_config_sha("simulation")
    assert baseline.stage_config_sha("h2h_power") != integration.stage_config_sha("h2h_power")
    assert baseline.stage_config_sha("reporting") != integration.stage_config_sha("reporting")


@pytest.mark.parametrize(
    "profile",
    [
        ProfileConfig(purpose="unknown"),
        ProfileConfig(purpose="production", reduced_resolution=True),
        ProfileConfig(
            purpose="integration",
            reduced_resolution=True,
            production_eligible=True,
            release_eligible=False,
        ),
    ],
)
def test_profile_semantics_reject_contradictory_claim_labels(profile: ProfileConfig) -> None:
    cfg = replace(AppConfig(), profile=profile)

    with pytest.raises((TypeError, ValueError), match="profile|production|integration"):
        cfg.validate_statistical_contract()
