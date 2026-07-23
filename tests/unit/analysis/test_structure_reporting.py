from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from farkle.analysis.structure_reporting import _claim_lines, render_markdown, run
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import write_json_artifact_atomic, write_parquet_artifact_atomic


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11], n_players_list=[2]),
    )
    cfg.screening.controls = [1]
    return cfg


def _write_frame(
    cfg: AppConfig,
    path: Path,
    frame: pd.DataFrame,
    *,
    scope: ArtifactScope,
    operation: str,
    player_counts: list[int] | None = None,
    k_aggregation_method: str = "none",
) -> None:
    counts = player_counts or [2]
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=scope,
        source_scope=scope,
        operation=operation,
        k_aggregation_method=k_aggregation_method,
        consistency_columns=frame.columns.tolist(),
        player_counts=counts,
        required_player_counts=counts,
        missing_cell_policy="fail",
        seed_scope="single_root",
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        sidecar=sidecar,
    )


def _write_json(
    cfg: AppConfig,
    path: Path,
    payload: dict[str, Any],
    operation: str,
) -> None:
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation=operation,
        consistency_columns=list(payload),
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="single_root",
    )
    write_json_artifact_atomic(payload, path, sidecar=sidecar)


def _publish_inputs(cfg: AppConfig) -> None:
    family_hash = "f" * 64
    family = {
        "family_hash": family_hash,
        "candidate_count": 2,
        "root_seeds": [11],
        "controls": ["1"],
        "mandatory_diagnostics": [],
        "initial_cutoffs": {"win_rate": 2, "trueskill": 2},
        "final_cutoffs": {"win_rate": 2, "trueskill": 2},
        "admission_counts": {"shared_methods": 2},
        "projected_workload": {"unordered_pair_count": 1},
    }
    _write_json(cfg, cfg.h2h_candidate_family_manifest_path(), family, "candidate_family_freeze")
    membership = pd.DataFrame(
        {
            "strategy": ["1", "2"],
            "final_family": [True, True],
            "family_hash": [family_hash, family_hash],
        }
    )
    _write_frame(
        cfg,
        cfg.h2h_candidate_family_path(),
        membership,
        scope=ArtifactScope.H2H_2P,
        operation="candidate_family_freeze",
    )
    agreement = {
        "family_hash": family_hash,
        "final_contribution_overlap": {"intersection_count": 2, "jaccard": 1.0},
        "rank_agreement": {"spearman": 1.0, "kendall": 1.0},
        "admission_counts": {"final_family": 2},
        "selection_conditioned_h2h": {"unordered_pair_count": 1},
        "root_specific_h2h_stability": {"available": False},
    }
    _write_json(
        cfg,
        cfg.structure_agreement_summary_path(),
        agreement,
        "selection_conditioned_method_agreement",
    )
    inference = pd.DataFrame(
        {
            "family_hash": [family_hash],
            "pair_id": [0],
            "strategy_a": ["1"],
            "strategy_b": ["2"],
            "games_attempted": [100],
            "games_completed": [100],
            "games_safety_limit": [0],
            "replacement_attempt_count": [0],
            "completion_game_rate": [1.0],
            "completion_status": ["complete"],
            "pair_inferentially_viable": [True],
            "pair_operationally_viable": [True],
            "pair_claim_eligible": [True],
            "strategy_a_completion_rate": [1.0],
            "strategy_b_completion_rate": [1.0],
            "strategy_a_games_attempted": [100],
            "strategy_b_games_attempted": [100],
            "strategy_a_games_completed": [100],
            "strategy_b_games_completed": [100],
            "strategy_a_games_safety_limit": [0],
            "strategy_b_games_safety_limit": [0],
            "strategy_a_replacement_attempt_count": [0],
            "strategy_b_replacement_attempt_count": [0],
            "strategy_a_operationally_viable": [True],
            "strategy_b_operationally_viable": [True],
            "strategy_a_inferentially_viable": [True],
            "strategy_b_inferentially_viable": [True],
            "min_candidate_completion_rate": [0.99],
            "d_ab": [0.1],
            "ordinary_d_low": [0.08],
            "ordinary_d_high": [0.12],
            "simultaneous_d_low": [0.06],
            "simultaneous_d_high": [0.14],
            "holm_adjusted_p": [0.001],
            "decision_class": ["practical_dominance_a"],
        }
    )
    _write_frame(
        cfg,
        cfg.h2h_pairwise_inference_path(),
        inference,
        scope=ArtifactScope.H2H_2P,
        operation="seat_adjusted_score_inference",
    )
    h2h_counts = pd.DataFrame(
        {
            "family_hash": [family_hash, family_hash],
            "pair_id": [0, 0],
            "strategy_a": ["1", "1"],
            "strategy_b": ["2", "2"],
            "root_seed": [11, 11],
            "order": [0, 1],
            "n_completed_required": [50, 50],
            "max_attempts": [100, 100],
            "games_attempted": [50, 50],
            "games_completed": [50, 50],
            "games_safety_limit": [0, 0],
            "replacement_attempt_count": [0, 0],
            "wins_a": [30, 30],
            "wins_b": [20, 20],
            "completion_status": ["complete", "complete"],
            "completion_game_rate": [1.0, 1.0],
            "safety_limit_game_rate": [0.0, 0.0],
        }
    )
    _write_frame(
        cfg,
        cfg.h2h_order_counts_path(),
        h2h_counts,
        scope=ArtifactScope.H2H_2P,
        operation="concatenate_root_order_blocks",
    )
    dominance = {
        "family_hash": family_hash,
        "practical_cycle_group_count": 0,
        "unique_best": "1",
        "unique_best_claim_permitted": True,
    }
    _write_json(
        cfg,
        cfg.h2h_dominance_summary_path(),
        dominance,
        "summarize_dominance_claims",
    )
    fronts = pd.DataFrame(
        {
            "strategy": ["1", "2"],
            "practical_front": [1, 2],
            "statistical_front": [1, 2],
            "practical_cycle_group": [None, None],
            "statistical_cycle_group": [None, None],
            "round_robin_mean_win_rate": [0.6, 0.4],
            "practical_net_wins": [1, -1],
            "tournament_screening_score": [0.1, 0.0],
        }
    )
    _write_frame(
        cfg,
        cfg.h2h_dominance_fronts_path(),
        fronts,
        scope=ArtifactScope.H2H_2P,
        operation="condensation_dag_fronts",
    )
    cycles = pd.DataFrame(
        columns=[
            "graph_type",
            "cycle_group",
            "cycle_size",
            "members_json",
            "strongest_internal_practical_winner",
            "strongest_internal_practical_loser",
            "strongest_internal_practical_distance",
            "weakest_internal_practical_winner",
            "weakest_internal_practical_loser",
            "weakest_internal_practical_distance",
            "representative_shortest_cycle_json",
        ]
    )
    _write_frame(
        cfg,
        cfg.h2h_cycle_groups_path(),
        cycles,
        scope=ArtifactScope.H2H_2P,
        operation="detect_strongly_connected_cycles",
    )
    root_agreement = pd.DataFrame(
        columns=[
            "pair_id",
            "agreement_available",
            "diagnostic_holm_decision_agreement",
            "effect_direction_agreement",
        ]
    )
    _write_frame(
        cfg,
        cfg.h2h_root_agreement_path(),
        root_agreement,
        scope=ArtifactScope.H2H_2P,
        operation="fixed_root_h2h_agreement_diagnostic",
    )
    across = pd.DataFrame(
        {
            "strategy": ["1", "2"],
            "complete_support": [True, True],
            "equal_k_score": [0.1, 0.0],
            "raw_attempted_exposures": [10, 10],
            "raw_completed_exposures": [10, 10],
            "raw_safety_limit_exposures": [0, 0],
            "safety_limit_exposure_rate": [0.0, 0.0],
        }
    )
    _write_frame(
        cfg,
        cfg.performance_across_k_path(),
        across,
        scope=ArtifactScope.ACROSS_K,
        operation="equal_k_mean",
        k_aggregation_method="equal_k",
    )
    by_k = pd.DataFrame(
        {
            "root_seed": [11, 11],
            "strategy": ["1", "2"],
            "chance_delta": [0.1, 0.0],
            "raw_attempted_exposures": [10, 10],
            "raw_completed_exposures": [10, 10],
            "raw_safety_limit_exposures": [0, 0],
        }
    )
    _write_frame(
        cfg,
        cfg.performance_by_k_path(2),
        by_k,
        scope=ArtifactScope.BY_K,
        operation="aggregate_performance_by_strategy",
    )


def test_reporting_writes_sidecar_validated_json_markdown_and_plot(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _publish_inputs(cfg)

    run(cfg, execution_scope="single_root")

    report = json.loads(cfg.structure_report_json_path().read_text(encoding="utf-8"))
    markdown = cfg.structure_report_markdown_path().read_text(encoding="utf-8")
    assert report["support"]["player_counts"] == [2]
    assert report["support"]["k_weights"] == {"2": 1.0}
    assert report["report_contract_version"] == 3
    assert report["performance"]["primary_rate"] == "win_rate_per_attempt"
    assert report["safety_limits"]["games_attempted"] == 10
    assert report["safety_limits"]["games_completed"] == 10
    assert report["safety_limits"]["games_safety_limit"] == 0
    assert report["h2h"]["role"] == "primary_two_player_finalist_inference"
    assert report["h2h"]["games_attempted"] == 100
    assert report["h2h"]["games_completed"] == 100
    assert report["h2h"]["games_safety_limit"] == 0
    assert report["h2h"]["unique_best_claim_permitted"] is True
    assert report["robustness"]["pareto_members"] == ["1"]
    assert "Unique best among the frozen two-player finalists" in markdown
    assert cfg.structure_report_plot_path().stat().st_size > 0
    for path in (
        cfg.structure_report_json_path(),
        cfg.structure_report_markdown_path(),
        cfg.structure_report_plot_path(),
    ):
        validate_artifact_sidecar(path, expected={"scope": "diagnostics"})


def _claim_report(
    *,
    unresolved: int = 0,
    cycles: int = 0,
    equivalent: int = 0,
    unique: str | None = None,
    permitted: bool = False,
) -> dict[str, Any]:
    return {
        "robustness": {
            "pareto_member_count": 2,
            "maximin_descriptive_leader": "A",
        },
        "h2h": {
            "unresolved_pair_count": unresolved,
            "cycle_group_count": cycles,
            "equivalent_pair_count": equivalent,
            "unique_best": unique,
            "unique_best_claim_permitted": permitted,
            "operationally_nonviable_candidates": [],
        },
    }


def test_claim_language_snapshots_cover_null_cycle_unresolved_equivalence_and_unique() -> None:
    assert _claim_lines(_claim_report())[-1] == (
        "No unique-best claim is permitted by the direct-dominance rule."
    )
    assert any(
        "cycle groups remain explicit" in line for line in _claim_lines(_claim_report(cycles=1))
    )
    assert any("remain unresolved" in line for line in _claim_lines(_claim_report(unresolved=2)))
    assert any(
        "configured equivalence rule" in line for line in _claim_lines(_claim_report(equivalent=1))
    )
    assert any(
        "Unique best among" in line
        for line in _claim_lines(_claim_report(unique="A", permitted=True))
    )
    assert any(
        "not a unique-best claim" in line
        for line in _claim_lines(_claim_report(unique="A", permitted=False))
    )


def test_markdown_uses_external_diagnostic_label_for_multi_k_claims() -> None:
    report = _claim_report(unique="A", permitted=False)
    report.update(
        {
            "execution_scope": "root_pair",
            "roots": [11, 22],
            "support": {"player_counts": [2, 4], "k_weights": {"2": 0.5, "4": 0.5}},
            "candidate_family": {"controls": []},
            "agreement": {
                "final_contribution_overlap": {},
                "rank_agreement": {},
                "admission_counts": {},
                "selection_conditioned_h2h": {},
            },
        }
    )
    report["h2h"].update(
        {
            "role": "external_two_player_finalist_diagnostic",
            "pair_intervals_artifact": "pairwise_inference.parquet",
        }
    )
    report["claim_language"] = _claim_lines(report)

    markdown = render_markdown(report)

    assert "external_two_player_finalist_diagnostic" in markdown
    assert "not a unique-best claim for the configured multi-k" in markdown
