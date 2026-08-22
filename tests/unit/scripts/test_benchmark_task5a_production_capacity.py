from __future__ import annotations

from pathlib import Path

import pytest
from scripts import benchmark_task5a_production_capacity as benchmark

from farkle.config import load_app_config


def test_merge_topology_distinguishes_current_three_pass_and_four_pass_fixture() -> None:
    current = benchmark.merge_topology(1_075)
    fixture = benchmark.merge_topology(32**3 + 1)

    assert current["depth"] == 3
    assert current["generations"] == [
        {"inputs": 1_075, "outputs": 34},
        {"inputs": 34, "outputs": 2},
        {"inputs": 2, "outputs": 1},
    ]
    assert fixture["depth"] == 4


def test_merge_topology_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        benchmark.merge_topology(-1)
    with pytest.raises(ValueError):
        benchmark.merge_topology(2, fan_in=1)


def test_current_production_dimensions_are_derived_from_executable_config() -> None:
    cfg = load_app_config(benchmark.PRODUCTION_CONFIG, seed_list_len=2)
    dimensions = benchmark.derive_dimensions(cfg)

    assert dimensions["strategy_count"] == {
        "value": 5_160,
        "evidence": "derived",
        "unit": "strategies",
    }
    assert dimensions["attempted_games_per_root"]["value"] == 39_013_900
    assert dimensions["player_exposures_per_root"]["value"] == 177_504_000
    assert dimensions["source_row_groups_per_root"]["value"] == 34_400
    assert dimensions["rng_route_units_per_route_per_root"]["value"] == 1_075
    assert dimensions["rng_count_route_records_per_root"]["value"] == 216_517_900
    assert dimensions["rng_reducer_opens_per_root"]["value"] == 68_800
    assert dimensions["maximum_candidate_count"]["value"] == 150
    assert dimensions["h2h"]["unordered_pair_count"]["value"] == 11_175
    assert dimensions["h2h"]["total_block_count"]["value"] == 44_700
    assert dimensions["h2h"]["planned_completed_games"]["value"] == 97_937_700


def test_integration_inventory_is_read_only_and_matches_accepted_tree() -> None:
    inventory = benchmark.integration_inventory(benchmark.INTEGRATION_ROOT)

    assert inventory["read_only"] is True
    assert inventory["run_elapsed_seconds"] == pytest.approx(1663.156)
    assert [root["source_row_groups"] for root in inventory["roots"]] == [1_800, 1_800]
    assert [root["route"]["01_count_route"]["rows"] for root in inventory["roots"]] == [
        189_600,
        189_600,
    ]


def test_projection_reconciles_fast_run_and_applies_stop_gate() -> None:
    cfg = load_app_config(benchmark.PRODUCTION_CONFIG, seed_list_len=2)
    dimensions = benchmark.derive_dimensions(cfg)
    inventory = benchmark.integration_inventory(benchmark.INTEGRATION_ROOT)
    projection = benchmark.build_projection(dimensions, inventory)

    assert projection["validation"]["passed"] is True
    assert abs(projection["validation"]["relative_residual"]) <= 0.10
    assert projection["production"]["plausible_lower_days"] > 2
    assert projection["production"]["conservative_planning_upper_days"] > 2
    assert projection["verdict"] == "not_capacity_ready"
    assert all(not band["planning_upper_fits"] for band in projection["budget_bands"])


def test_owned_root_requires_task5a_prefix(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="must use"):
        benchmark.run(tmp_path / "not-owned")
