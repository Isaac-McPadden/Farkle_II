from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from farkle.analysis.stage_registry import resolve_root_pair_stage_layout, resolve_stage_layout
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
    assert {cell["required_shuffles"] for cell in workload.values()} == {600}
    assert {cell["batch_count"] for cell in workload.values()} == {20}
    assert {cell["shuffles_per_batch"] for cell in workload.values()} == {30}
    assert all(
        cell["achieved_resolution"] == pytest.approx(0.07976027215162139)
        for cell in workload.values()
    )
    assert {k: cell["required_games_per_root"] for k, cell in workload.items()} == {
        "2": 24_000,
        "4": 12_000,
        "5": 9_600,
    }
    assert sum(cell["required_games_per_root"] for cell in workload.values()) == 45_600
    assert (
        len(metadata["roots"]) * sum(cell["required_games_per_root"] for cell in workload.values())
        == 91_200
    )

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
    assert cfg.batching.target_batches == 20
    assert cfg.batching.min_shuffles_per_batch == 30
    assert compute_config_sha(cfg) == (
        "0e2b6bad2a5fa07dcefa39466e96e170b4127fccf0f493e232daed79b9839b2e"
    )
    with_old_host_reserve = replace(
        cfg,
        resources=replace(cfg.resources, minimum_system_available_memory_mb=2_048),
    )
    assert compute_config_sha(with_old_host_reserve) == compute_config_sha(cfg)


def test_fast_config_keeps_pipeline_execution_mechanisms_and_principal_stages() -> None:
    cfg = load_app_config(CONFIG_DIR / "fast_config.yaml", seed_list_len=2)

    assert cfg.batching.target_batches > 1
    assert cfg.sim.n_jobs is not None and cfg.sim.n_jobs > 1
    assert cfg.analysis.n_jobs is not None and cfg.analysis.n_jobs > 1
    assert cfg.head2head.n_jobs == 0  # auto-resolved multiprocessing
    assert cfg.sim.ckpt_every_sec > 0
    assert cfg.sim.row_dir is not None
    assert cfg.analysis.disable_rng_diagnostics is False
    assert resolve_stage_layout(cfg).keys() == [
        "ingest",
        "curate",
        "combine",
        "metrics",
        "game_stats",
        "rng_diagnostics",
        "trueskill",
        "hgb",
        "screening",
        "candidate_freeze",
        "h2h_power",
        "h2h_execute",
        "h2h_inference",
        "h2h_digest",
        "agreement",
        "reporting",
    ]
    assert resolve_root_pair_stage_layout(cfg).keys() == [
        "root_stability",
        "trueskill",
        "candidate_freeze",
        "h2h_power",
        "h2h_execute",
        "h2h_inference",
        "h2h_digest",
        "agreement",
        "reporting",
    ]


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
    assert cfg.batching.target_batches == 100
    assert cfg.batching.min_shuffles_per_batch == 30
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
