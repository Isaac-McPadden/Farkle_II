from __future__ import annotations

from pathlib import Path

import pytest

from farkle.analysis.stage_registry import resolve_root_pair_stage_layout
from farkle.config import AppConfig, IOConfig, SimConfig


def _valid_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(
            seed=101,
            seed_list=[101, 202],
            n_players_list=[2, 4],
        ),
    )
    cfg.screening.practical_delta_by_k = {2: 0.03, 4: 0.02}
    cfg.screening.delta_across_k = 0.03
    cfg.set_stage_layout(resolve_root_pair_stage_layout(cfg))
    return cfg


def test_root_stability_paths_are_cross_seed_metric_artifacts(tmp_path: Path) -> None:
    cfg = _valid_cfg(tmp_path)
    base = cfg.cross_seed_dir("cross_seed")

    assert cfg.root_combined_performance_by_k_path(4).parent == base
    assert cfg.root_combined_performance_across_k_path().parent == base
    assert cfg.root_discrepancies_path().parent == base
    assert cfg.root_joint_discrepancy_path().parent == base
    assert cfg.root_rank_stability_path().parent == base
    assert cfg.root_top_n_stability_path().parent == base
    assert cfg.root_control_movement_path().parent == base
    assert cfg.root_shortlist_changes_path().parent == base
    assert cfg.root_matched_count_convergence_path().parent == base
    assert cfg.root_half_drift_path().parent == base


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("delta_seed_stability", 0.0, "delta_seed_stability"),
        ("joint_discrepancy_alpha", 0.0, "joint_discrepancy_alpha"),
        ("joint_discrepancy_alpha", 1.0, "joint_discrepancy_alpha"),
        ("matched_count_fractions", (), "matched_count_fractions"),
        ("matched_count_fractions", (0.5, 0.25, 1.0), "matched_count_fractions"),
        ("matched_count_fractions", (0.25, 0.5), "matched_count_fractions"),
    ],
)
def test_statistical_contract_validates_root_stability_settings(
    tmp_path: Path,
    field: str,
    value: float | tuple[float, ...],
    message: str,
) -> None:
    cfg = _valid_cfg(tmp_path)
    setattr(cfg.robustness, field, value)

    with pytest.raises(ValueError, match=message):
        cfg.validate_statistical_contract(require_two_roots=True)
