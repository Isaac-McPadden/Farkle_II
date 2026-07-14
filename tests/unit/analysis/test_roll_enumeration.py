from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from farkle.analysis.roll_enumeration import (
    build_exact_roll_enumeration,
    enumerate_ordered_roll_outcomes,
)
from farkle.config import AppConfig, IOConfig, SimConfig
from farkle.utils.artifact_contract import validate_artifact_sidecar


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=3, n_players_list=[2]),
    )


def test_exact_ordered_roll_oracle() -> None:
    distribution, summary = enumerate_ordered_roll_outcomes()

    assert summary["ordered_outcomes"].tolist() == [6**dice for dice in range(1, 7)]
    counts = distribution.groupby("dice_count")["ordered_outcome_count"].sum().tolist()
    assert counts == [6**dice for dice in range(1, 7)]

    one_die = summary.loc[summary["dice_count"] == 1].iloc[0]
    assert one_die["farkle_probability"] == pytest.approx(4 / 6)
    assert one_die["expected_max_immediate_score"] == pytest.approx(25)
    assert one_die["hot_dice_probability"] == pytest.approx(2 / 6)
    assert one_die["expected_scoring_dice"] == pytest.approx(2 / 6)

    two_dice = summary.loc[summary["dice_count"] == 2].iloc[0]
    assert two_dice["farkle_probability"] == pytest.approx(16 / 36)
    assert two_dice["expected_max_immediate_score"] == pytest.approx(50)
    assert two_dice["hot_dice_probability"] == pytest.approx(4 / 36)
    assert two_dice["expected_scoring_dice"] == pytest.approx(2 / 3)

    six_ones = distribution.loc[
        (distribution["dice_count"] == 6)
        & (distribution["max_immediate_score"] == 3000)
        & (distribution["scoring_dice"] == 6)
    ]
    assert six_ones["ordered_outcome_count"].sum() >= 1


def test_exact_roll_artifacts_have_diagnostic_sidecars(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    artifacts = build_exact_roll_enumeration(cfg)

    assert pq.read_table(artifacts.summary).num_rows == 6
    for path in artifacts.all_paths:
        validate_artifact_sidecar(
            path,
            expected={
                "scope": "diagnostics",
                "operation": "exact_enumeration",
                "seed_scope": "not_applicable",
            },
        )
