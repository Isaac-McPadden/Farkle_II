from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.all_player_metrics import all_player_batch_schema
from farkle.analysis.root_stability import (
    RootBatchCell,
    _ratio_mcse,
    build_two_root_stability,
)
from farkle.analysis.stage_registry import resolve_root_pair_stage_layout
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    make_artifact_sidecar,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.stage_completion import stage_done_path


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(
            seed=11,
            seed_list=[11, 22],
            n_players_list=[2, 4],
        ),
    )
    cfg.screening.practical_delta_by_k = {2: 0.03, 4: 0.03}
    cfg.screening.delta_across_k = 0.03
    cfg.screening.bootstrap_replicates = 20
    cfg.screening.candidate_contribution_size = 2
    cfg.screening.controls = [2]
    cfg.set_stage_layout(resolve_root_pair_stage_layout(cfg))
    return cfg


def _metric_row(
    root: int,
    k: int,
    batch: int,
    strategy: int,
    wins: int,
    exposures: int = 10,
) -> dict[str, object]:
    row: dict[str, object] = {field.name: 0 for field in all_player_batch_schema()}
    row.update(
        {
            "root_seed": root,
            "k": k,
            "deterministic_batch_id": batch,
            "strategy": strategy,
            "raw_player_game_exposures": exposures,
            "raw_completed_player_game_exposures": exposures,
            "raw_safety_limit_player_game_exposures": 0,
            "raw_wins": wins,
            "raw_losses": exposures - wins,
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


def _write_cell(
    cfg: AppConfig,
    tmp_path: Path,
    root: int,
    k: int,
    strategy_wins: dict[int, list[int]],
) -> RootBatchCell:
    rows = [
        _metric_row(root, k, batch, strategy, wins)
        for strategy, values in sorted(strategy_wins.items())
        for batch, wins in enumerate(values)
    ]
    path = tmp_path / "inputs" / f"root_{root}" / f"{k}p_metrics.parquet"
    table = pa.Table.from_pylist(rows, schema=all_player_batch_schema())
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation="aggregate_performance_by_strategy",
        conditioning="all_attempted_player_game_exposures_safety_limit_is_loss",
        consistency_columns=table.schema.names,
        player_counts=[k],
        required_player_counts=[k],
        missing_cell_policy="fail",
        seed_scope="single_root",
    )
    write_parquet_artifact_atomic(table, path, sidecar=sidecar)
    return RootBatchCell(root_seed=root, k=k, path=path)


def _write_inputs(cfg: AppConfig, tmp_path: Path) -> list[RootBatchCell]:
    values = {
        (11, 2): {1: [7, 8, 6, 9], 2: [5, 4, 6, 5], 3: [4, 5, 4, 5]},
        (22, 2): {1: [6, 7, 5, 8], 2: [5, 6, 4, 5], 3: [5, 4, 5, 4]},
        (11, 4): {1: [4, 3, 5, 4], 2: [3, 2, 3, 2], 3: [2, 3, 2, 3]},
        (22, 4): {1: [3, 4, 3, 4], 2: [2, 3, 2, 3], 3: [3, 2, 3, 2]},
    }
    return [
        _write_cell(cfg, tmp_path, root, k, strategy_wins)
        for (root, k), strategy_wins in sorted(values.items())
    ]


def test_ratio_mcse_matches_equal_exposure_batch_formula() -> None:
    wins = np.asarray([40.0, 50.0, 60.0, 70.0])
    exposures = np.full(4, 100.0)

    observed = _ratio_mcse(wins, exposures)

    assert observed == pytest.approx(np.std(wins / exposures, ddof=1) / np.sqrt(4))


def test_half_drift_preserves_raw_difference_when_one_batch_per_half_has_no_mcse(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    values = {
        (11, 2): {1: [7, 8], 2: [5, 4], 3: [4, 5]},
        (22, 2): {1: [6, 7], 2: [5, 6], 3: [5, 4]},
        (11, 4): {1: [4, 3], 2: [3, 2], 3: [2, 3]},
        (22, 4): {1: [3, 4], 2: [2, 3], 3: [3, 2]},
    }
    cells = [
        _write_cell(cfg, tmp_path, root, k, strategy_wins)
        for (root, k), strategy_wins in sorted(values.items())
    ]

    artifacts = build_two_root_stability(cfg, cells)
    drift = pq.read_table(artifacts.half_drift).to_pandas()

    assert len(drift) == 18
    assert drift["raw_difference"].notna().all()
    assert drift["expected_mcse"].isna().all()
    assert drift["standardized_drift"].isna().all()


def test_two_root_combination_and_stability_contract(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cells = _write_inputs(cfg, tmp_path)

    artifacts = build_two_root_stability(cfg, cells)

    k2 = pq.read_table(artifacts.combined_by_k[0]).to_pandas()
    strategy = k2.loc[(k2["estimate_scope"] == "combined_roots") & (k2["strategy"] == 1)].iloc[0]
    assert strategy["raw_wins"] == 56
    assert strategy["raw_exposures"] == 80
    assert strategy["win_rate"] == pytest.approx(56 / 80)
    assert strategy["chance_delta"] == pytest.approx(56 / 80 - 1 / 2)
    assert set(k2["estimate_scope"]) == {"root_11", "root_22", "combined_roots"}

    across = pq.read_table(artifacts.across_k).to_pandas()
    assert set(across["estimate_scope"]) == {"root_11", "root_22", "combined_roots"}
    combined = across.loc[
        (across["estimate_scope"] == "combined_roots") & (across["strategy"] == 1)
    ].iloc[0]
    expected = ((56 / 80 - 1 / 2) + (30 / 80 - 1 / 4)) / 2
    assert combined["across_k_score"] == pytest.approx(expected)
    assert bool(combined["complete_support"])

    discrepancies = pq.read_table(artifacts.discrepancies).to_pandas()
    assert {
        "raw_difference",
        "standardized_discrepancy",
        "threshold_fraction",
        "exceeds_joint_reference_quantile",
    }.issubset(discrepancies.columns)
    assert set(discrepancies["estimand_scope"]) == {"by_k", "across_k"}

    assert pq.read_table(artifacts.joint_discrepancy).num_rows == 1
    rank_stability = pq.read_table(artifacts.rank_stability).to_pandas()
    assert len(rank_stability) == 1
    assert rank_stability.iloc[0]["p95_absolute_rank_movement"] >= 0.0
    assert pq.read_table(artifacts.top_n_stability).num_rows == 4
    root_inclusion = pq.read_table(artifacts.bootstrap_top_n_inclusion).to_pandas()
    assert set(root_inclusion["root_seed"]) == {11, 22}
    assert len(root_inclusion) == 6
    assert root_inclusion["complete_support"].all()
    assert root_inclusion["bootstrap_top_n_inclusion_frequency"].between(0.0, 1.0).all()
    assert root_inclusion.groupby("root_seed")[
        "bootstrap_top_n_inclusion_frequency"
    ].sum().tolist() == pytest.approx([2.0, 2.0])
    assert pq.read_table(artifacts.control_movement).num_rows == 1
    assert pq.read_table(artifacts.shortlist_changes).num_rows == 3

    convergence = pq.read_table(artifacts.matched_count_convergence).to_pandas()
    assert convergence["cumulative_fraction"].tolist() == [0.25, 0.5, 0.75, 1.0]
    drift = pq.read_table(artifacts.half_drift).to_pandas()
    assert set(drift["estimand_scope"]) == {"by_k", "across_k"}
    assert set(drift["root_seed"]) == {11, 22}

    # This pointwise interval excludes chance, which historically triggered a
    # statistically_* label. It remains Monte Carlo precision only.
    assert strategy["batch_mc_precision_interval_low"] > strategy["chance_baseline"]
    assert strategy["practical_threshold_position"] == "above_positive_threshold"

    forbidden = {
        "tau_squared",
        "i_squared",
        "root_population_interval",
        "performance_classification",
        "root_a_classification",
        "root_b_classification",
        "combined_classification",
        "classification_changed",
        "joint_adjusted_diagnostic_p",
        "diagnostic_family_alpha",
        "joint_max_abs_standardized_critical",
        "joint_unusual",
        "flagged_estimands",
    }
    forbidden_language = ("statistically_", "significance", "rejection")
    for path in artifacts.all_paths:
        frame = pq.read_table(path).to_pandas()
        assert forbidden.isdisjoint(frame.columns)
        serialized = json.dumps(
            {
                "columns": frame.columns.tolist(),
                "string_values": {
                    column: frame[column].dropna().astype(str).tolist()
                    for column in frame.select_dtypes(include=["object", "string"]).columns
                },
                "sidecar": json.loads(sidecar_path(path).read_text(encoding="utf-8")),
            },
            sort_keys=True,
        ).lower()
        assert not any(label in serialized for label in forbidden_language)
        sidecar = validate_artifact_sidecar(path, expected={"scope": "cross_seed"})
        assert (sidecar.method_contract or {}).get("parameters") == {
            "method_version": 2,
            "design_interpretation": "fixed_design_descriptive_reproducibility",
            "interval_role": "monte_carlo_precision",
            "root_population_inference": "none",
            "multiple_testing_inference": "none",
        }
    completion = json.loads(
        stage_done_path(cfg.stage_dir("root_stability"), "root_stability").read_text(
            encoding="utf-8"
        )
    )
    assert completion["freshness_key"]["root_stability_method_version"] == 2

    first = pq.read_table(artifacts.discrepancies).to_pandas()
    build_two_root_stability(cfg, cells, force=True)
    second = pq.read_table(artifacts.discrepancies).to_pandas()
    pd.testing.assert_frame_equal(first, second)


def test_two_root_combination_rejects_incomplete_root_k_support(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cells = _write_inputs(cfg, tmp_path)

    with pytest.raises(ValueError, match="cover every root/k cell"):
        build_two_root_stability(cfg, cells[:-1])


def test_two_root_combination_rejects_scope_mismatched_input(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cells = _write_inputs(cfg, tmp_path)
    invalid = cells[0]
    table = pq.read_table(invalid.path)
    sidecar = make_artifact_sidecar(
        cfg,
        invalid.path,
        producer="test",
        scope=ArtifactScope.CONCAT_KS,
        source_scope=ArtifactScope.BY_K,
        operation="concatenate",
        conditioning="all_attempted_player_game_exposures_safety_limit_is_loss",
        consistency_columns=table.schema.names,
        player_counts=[invalid.k],
        required_player_counts=[invalid.k],
        missing_cell_policy="fail",
    )
    write_parquet_artifact_atomic(table, invalid.path, sidecar=sidecar)

    with pytest.raises(ArtifactContractError, match="scope"):
        build_two_root_stability(cfg, cells)


def test_root_stability_excludes_declared_zero_batch_and_records_it(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cells = _write_inputs(cfg, tmp_path)
    target = cells[0]
    rows = pq.read_table(target.path).to_pylist()
    rows.extend(
        _metric_row(target.root_seed, target.k, 4, strategy, 0, 0) for strategy in (1, 2, 3)
    )
    table = pa.Table.from_pylist(rows, schema=all_player_batch_schema())
    sidecar = make_artifact_sidecar(
        cfg,
        target.path,
        producer="test",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation="aggregate_performance_by_strategy",
        conditioning="all_attempted_player_game_exposures_safety_limit_is_loss",
        consistency_columns=table.schema.names,
        player_counts=[target.k],
        required_player_counts=[target.k],
        missing_cell_policy="fail",
        seed_scope="single_root",
    )
    write_parquet_artifact_atomic(table, target.path, sidecar=sidecar)

    artifacts = build_two_root_stability(cfg, cells)
    combined = pq.read_table(artifacts.combined_by_k[0]).to_pandas()
    root_rows = combined.loc[combined["estimate_scope"].eq("root_11")]

    assert set(root_rows["raw_declared_batches"]) == {5}
    assert set(root_rows["raw_batches"]) == {4}
    assert set(root_rows["excluded_zero_exposure_batch_cells"]) == {1}


def test_root_stability_rejects_missing_strategy_batch_cell(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cells = _write_inputs(cfg, tmp_path)
    target = cells[0]
    rows = [
        row
        for row in pq.read_table(target.path).to_pylist()
        if not (row["deterministic_batch_id"] == 3 and row["strategy"] == 3)
    ]
    table = pa.Table.from_pylist(rows, schema=all_player_batch_schema())
    sidecar = make_artifact_sidecar(
        cfg,
        target.path,
        producer="test",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation="aggregate_performance_by_strategy",
        conditioning="all_attempted_player_game_exposures_safety_limit_is_loss",
        consistency_columns=table.schema.names,
        player_counts=[target.k],
        required_player_counts=[target.k],
        missing_cell_policy="fail",
        seed_scope="single_root",
    )
    write_parquet_artifact_atomic(table, target.path, sidecar=sidecar)

    with pytest.raises(ValueError, match="missing declared rectangular strategy/batch cells"):
        build_two_root_stability(cfg, cells)
