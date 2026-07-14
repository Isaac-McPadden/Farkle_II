from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.all_player_metrics import all_player_batch_schema
from farkle.analysis.performance import _pareto_membership, build_canonical_performance
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, n_players_list=[2, 4, 12]),
    )
    cfg.screening.delta_across_k = 0.03
    cfg.screening.practical_delta_by_k = {2: 0.03, 4: 0.03, 12: 0.03}
    cfg.screening.bootstrap_replicates = 40
    cfg.screening.candidate_contribution_size = 1
    cfg.screening.controls = [2]
    return cfg


def _metric_row(k: int, batch: int, strategy: int, wins: int, exposures: int) -> dict:
    row = {field.name: 0 for field in all_player_batch_schema()}
    row.update(
        {
            "root_seed": 11,
            "k": k,
            "deterministic_batch_id": batch,
            "strategy": strategy,
            "raw_player_game_exposures": exposures,
            "raw_wins": wins,
        }
    )
    for name in (
        "turn_return_turn_weighted",
        "turn_return_game_weighted_exact",
        "turn_return_round_proxy",
        "round_proxy_gap",
        "round_proxy_relative_gap",
        "turn_round_mismatch_prevalence",
    ):
        row[name] = None
    return row


def _write_batch_metrics(cfg: AppConfig, k: int, rows: list[dict]) -> None:
    path = cfg.metrics_all_player_batch_path(k)
    table = pa.Table.from_pylist(rows, schema=all_player_batch_schema())
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation="aggregate",
        conditioning="unconditional",
        player_counts=[k],
        required_player_counts=[k],
        missing_cell_policy="fail",
    )
    write_parquet_artifact_atomic(table, path, sidecar=sidecar)


def _write_inputs(cfg: AppConfig) -> None:
    values = {
        2: {1: (60, 80), 2: (45, 35)},
        4: {1: (30, 40), 2: (20, 20)},
        12: {1: (13, 14), 2: (8, 9)},
    }
    for k, strategies in values.items():
        rows = [
            _metric_row(k, batch, strategy, strategies[strategy][batch], 100)
            for batch in (0, 1)
            for strategy in (1, 2)
        ]
        if k == 2:
            rows.append(_metric_row(k, 0, 3, 50, 100))
        _write_batch_metrics(cfg, k, rows)


def test_canonical_performance_estimators_and_support_contract(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_inputs(cfg)

    artifacts = build_canonical_performance(cfg)

    by_k = {
        k: pq.read_table(path).to_pandas()
        for k, path in zip((2, 4, 12), artifacts.by_k, strict=True)
    }
    for k, frame in by_k.items():
        assert frame["chance_baseline"].unique().tolist() == pytest.approx([1 / k])
        assert frame["practical_delta_by_k"].unique().tolist() == pytest.approx([0.03])
        assert (frame["raw_exposures"] > 0).all()
        assert (frame["wilson_interval_width"] > 0).all()
    k2_strategy1 = by_k[2].set_index("strategy").loc[1]
    assert k2_strategy1["win_rate"] == pytest.approx(0.7)
    assert k2_strategy1["chance_delta"] == pytest.approx(0.2)
    assert k2_strategy1["batch_mcse"] == pytest.approx(0.1)

    across = pq.read_table(artifacts.across_k).to_pandas().set_index("strategy")
    expected_score = (0.2 + 0.10 + ((27 / 200) - 1 / 12)) / 3
    assert across.loc[1, "equal_k_score"] == pytest.approx(expected_score)
    assert across.loc[1, "practical_delta_across_k"] == pytest.approx(0.03)
    expected_mcse = ((0.1**2 + 0.05**2 + 0.005**2) / 9) ** 0.5
    assert across.loc[1, "equal_k_mcse"] == pytest.approx(expected_mcse)
    assert bool(across.loc[1, "pareto_member"])
    assert bool(across.loc[1, "maximin_leader"])
    assert across.loc[1, "worst_k"] == 12
    assert not bool(across.loc[3, "complete_support"])
    assert pd.isna(across.loc[3, "equal_k_score"])

    bootstrap = pq.read_table(artifacts.bootstrap).to_pandas().set_index("strategy")
    assert bootstrap.loc[1, "top_n_inclusion_probability"] == pytest.approx(1.0)
    assert bootstrap.loc[1, "shortlist_inclusion_probability"] == pytest.approx(1.0)
    contrasts = pq.read_table(artifacts.control_contrasts).to_pandas()
    observed = contrasts.loc[contrasts["strategy"] == 1].iloc[0]
    assert observed["observed_equal_k_contrast"] == pytest.approx(
        across.loc[1, "equal_k_score"] - across.loc[2, "equal_k_score"]
    )

    validate_artifact_sidecar(
        artifacts.across_k,
        expected={
            "scope": "across_k",
            "operation": "equal_k_mean",
            "k_aggregation_method": "equal_k",
        },
    )


def test_joint_batch_resampling_is_coordinate_deterministic(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.screening.controls = []
    _write_inputs(cfg)
    artifacts = build_canonical_performance(cfg)
    first = pq.read_table(artifacts.bootstrap).to_pandas()
    contrast_table = pq.read_table(artifacts.control_contrasts)
    assert contrast_table.num_rows == 0
    assert "control_strategy" in contrast_table.schema.names

    build_canonical_performance(cfg, force=True)
    second = pq.read_table(artifacts.bootstrap).to_pandas()

    pd.testing.assert_frame_equal(first, second)


def test_pareto_membership_preserves_tradeoffs_and_exact_ties() -> None:
    values = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.4, 0.4],
            [1.0, 0.0],
        ]
    )
    strategies = np.asarray([1, 2, 3, 4, 5])

    assert _pareto_membership(values, strategies).tolist() == [True, True, True, False, True]
