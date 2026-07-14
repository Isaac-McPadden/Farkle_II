"""Sidecar-gated canonical reporting for screening, H2H, and stability evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import (
    ArtifactSidecar,
    make_artifact_sidecar,
    validate_artifact_sidecar,
    write_artifact_with_sidecar_atomic,
)
from farkle.utils.artifacts import write_json_artifact_atomic
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done

_PERFORMANCE_OPERATIONS: Final = {"equal_k_mean", "declared_k_weighted_mean"}


def _read_json(path: Path, *, operation: str) -> tuple[dict[str, Any], ArtifactSidecar]:
    sidecar = validate_artifact_sidecar(path, expected={"operation": operation})
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report input {path} must contain a JSON object")
    return payload, sidecar


def _read_frame(path: Path, *, operation: str) -> tuple[pd.DataFrame, ArtifactSidecar]:
    sidecar = validate_artifact_sidecar(path, expected={"operation": operation})
    return pq.read_table(path).to_pandas(), sidecar


def _roots(cfg: AppConfig) -> tuple[int, ...]:
    roots = tuple(int(root) for root in (cfg.sim.seed_list or [cfg.sim.seed]))
    if len(roots) not in {1, 2}:
        raise ValueError(f"reporting requires one or two roots, found {roots}")
    return roots


def _performance_source(cfg: AppConfig, roots: tuple[int, ...]) -> Path:
    return (
        cfg.root_combined_performance_across_k_path()
        if len(roots) == 2
        else cfg.performance_across_k_path()
    )


def _performance_frame(
    path: Path,
    roots: tuple[int, ...],
) -> tuple[pd.DataFrame, ArtifactSidecar, str]:
    sidecar = validate_artifact_sidecar(path)
    if sidecar.operation not in _PERFORMANCE_OPERATIONS:
        raise ValueError(f"report performance operation {sidecar.operation!r} is not canonical")
    if len(roots) == 2 and sidecar.scope != ArtifactScope.CROSS_SEED.value:
        raise ValueError("two-root report requires cross_seed performance")
    if len(roots) == 1 and sidecar.scope != ArtifactScope.ACROSS_K.value:
        raise ValueError("single-root report requires across_k performance")
    frame = pq.read_table(path).to_pandas()
    if len(roots) == 2:
        frame = frame.loc[frame["estimate_scope"].eq("combined_roots")].copy()
        score_column = "across_k_score"
    else:
        score_column = "equal_k_score"
    frame["strategy"] = frame["strategy"].astype(str)
    if frame.empty or frame["strategy"].duplicated().any():
        raise ValueError("report performance requires one nonempty row per strategy")
    if not frame["complete_support"].astype(bool).all():
        raise ValueError("report performance refuses incomplete configured k support")
    return frame, sidecar, score_column


def _by_k_vectors(
    cfg: AppConfig,
    roots: tuple[int, ...],
    player_counts: list[int],
) -> tuple[dict[str, list[float]], list[Path]]:
    values: dict[str, list[float]] = {}
    sources: list[Path] = []
    for k in player_counts:
        path = (
            cfg.root_combined_performance_by_k_path(k)
            if len(roots) == 2
            else cfg.performance_by_k_path(k)
        )
        expected_operation = (
            "within_k_exposure_combination"
            if len(roots) == 2
            else "aggregate_performance_by_strategy"
        )
        frame, _sidecar = _read_frame(path, operation=expected_operation)
        sources.append(path)
        if len(roots) == 2:
            frame = frame.loc[frame["estimate_scope"].eq("combined_roots")].copy()
        frame["strategy"] = frame["strategy"].astype(str)
        if frame["strategy"].duplicated().any():
            raise ValueError(f"per-k report evidence contains duplicate strategies for k={k}")
        for strategy, chance_delta in zip(
            frame["strategy"], frame["chance_delta"].astype(float), strict=True
        ):
            values.setdefault(strategy, []).append(float(chance_delta))
    incomplete = {
        strategy: len(vector)
        for strategy, vector in values.items()
        if len(vector) != len(player_counts)
    }
    if incomplete:
        raise ValueError(f"per-k report evidence lacks complete support: {incomplete}")
    return values, sources


def _robustness(vectors: dict[str, list[float]]) -> dict[str, Any]:
    strategies = sorted(vectors)
    pareto: list[str] = []
    for strategy in strategies:
        candidate = np.asarray(vectors[strategy], dtype=float)
        dominated = False
        for other in strategies:
            if other == strategy:
                continue
            comparison = np.asarray(vectors[other], dtype=float)
            if np.all(comparison >= candidate) and np.any(comparison > candidate):
                dominated = True
                break
        if not dominated:
            pareto.append(strategy)
    minima = {strategy: float(min(values)) for strategy, values in vectors.items()}
    maximin_value = max(minima.values())
    maximin_leader = min(
        strategy
        for strategy, value in minima.items()
        if np.isclose(value, maximin_value, rtol=0.0, atol=1e-15)
    )
    return {
        "pareto_members": pareto,
        "pareto_member_count": len(pareto),
        "maximin_descriptive_leader": maximin_leader,
        "maximin_value": maximin_value,
        "interpretation": "pareto_membership_and_maximin_are_distinct_descriptive_summaries",
    }


def _normalized_weights(sidecar: ArtifactSidecar, player_counts: list[int]) -> dict[str, float]:
    if sidecar.k_weights is None:
        equal = 1.0 / len(player_counts)
        return {str(k): equal for k in player_counts}
    weights = {str(k): float(sidecar.k_weights[str(k)]) for k in player_counts}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in cast(list[dict[str, Any]], frame[columns].to_dict(orient="records")):
        row: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
                row[key] = None
            elif isinstance(value, np.generic):
                row[key] = value.item()
            else:
                row[key] = value
        rows.append(row)
    return rows


def _cycle_records(cycles: pd.DataFrame) -> list[dict[str, Any]]:
    if cycles.empty:
        return []
    groups = cycles.drop_duplicates(["graph_type", "cycle_group"])
    columns = [
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
    output = _records(groups, columns)
    for row in output:
        row["members"] = json.loads(str(row.pop("members_json")))
        row["representative_shortest_cycle"] = json.loads(
            str(row.pop("representative_shortest_cycle_json"))
        )
    return output


def _claim_lines(report: dict[str, Any]) -> list[str]:
    h2h = report["h2h"]
    robustness = report["robustness"]
    lines = [
        "Tournament screening leaders are descriptive chance-adjusted score leaders.",
        f"Pareto membership contains {robustness['pareto_member_count']} strategy configurations.",
        (
            "The separate maximin descriptive leader is "
            f"{robustness['maximin_descriptive_leader']}."
        ),
    ]
    if h2h["unresolved_pair_count"]:
        lines.append(f"{h2h['unresolved_pair_count']} finalist comparisons remain unresolved.")
    if h2h["cycle_group_count"]:
        lines.append(f"{h2h['cycle_group_count']} directed cycle groups remain explicit.")
    if h2h["equivalent_pair_count"]:
        lines.append(
            f"{h2h['equivalent_pair_count']} comparisons satisfy the configured equivalence rule."
        )
    unique = h2h["unique_best"]
    if unique is not None and h2h["unique_best_claim_permitted"]:
        lines.append(
            f"Unique best among the frozen two-player finalists: {unique}, by direct practical "
            "dominance over every other finalist."
        )
    elif unique is not None:
        lines.append(
            f"The external two-player finalist diagnostic has direct practical dominator {unique}; "
            "this is not a unique-best claim for the configured multi-k performance estimand."
        )
    else:
        lines.append("No unique-best claim is permitted by the direct-dominance rule.")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """Render deterministic claim language from a machine-readable report."""

    support = report["support"]
    family = report["candidate_family"]
    h2h = report["h2h"]
    lines = [
        "# Structure analysis report",
        "",
        "This report is conditional on the simulated finite strategy grid.",
        "",
        "## Contract",
        "",
        f"- Execution scope: `{report['execution_scope']}`",
        f"- Roots: `{report['roots']}`",
        f"- Player-count support: `{support['player_counts']}`",
        f"- Declared k weights: `{support['k_weights']}`",
        f"- Controls: `{family['controls']}`",
        f"- H2H role: `{h2h['role']}`",
        "- Tournament performance is unconditional; H2H is conditioned on frozen finalist selection.",
        "",
        "## Distinct evidence summaries",
        "",
        *[f"- {line}" for line in report["claim_language"]],
        "",
        "## Method agreement",
        "",
        f"- Win-rate/TrueSkill final contribution overlap: `{report['agreement']['final_contribution_overlap']}`",
        f"- Common-population rank agreement: `{report['agreement']['rank_agreement']}`",
        f"- Admission counts: `{report['agreement']['admission_counts']}`",
        f"- Selection-conditioned H2H agreement: `{report['agreement']['selection_conditioned_h2h']}`",
        "",
        "## H2H decisions",
        "",
        f"- Unresolved pairs: {h2h['unresolved_pair_count']}",
        f"- Cycle groups: {h2h['cycle_group_count']}",
        f"- Ordinary intervals and simultaneous practical bounds are retained in `{h2h['pair_intervals_artifact']}`.",
        "",
    ]
    return "\n".join(lines)


def _write_text(path: Path, text: str, sidecar: ArtifactSidecar) -> None:
    def _writer(staged: Path) -> None:
        staged.write_text(text, encoding="utf-8")

    write_artifact_with_sidecar_atomic(
        path,
        sidecar,
        _writer,
    )


def _write_plot(
    path: Path,
    performance: pd.DataFrame,
    score_column: str,
    finalists: set[str],
    sidecar: ArtifactSidecar,
) -> None:
    import matplotlib.pyplot as plt

    selected = performance.sort_values(
        [score_column, "strategy"], ascending=[False, True], kind="mergesort"
    ).head(30)

    def write(staged: Path) -> None:
        fig, axis = plt.subplots(figsize=(10, 6))
        colors = [
            "#d95f02" if strategy in finalists else "#1b9e77" for strategy in selected["strategy"]
        ]
        axis.barh(selected["strategy"], selected[score_column], color=colors)
        axis.invert_yaxis()
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("chance-adjusted tournament screening score")
        axis.set_ylabel("strategy configuration")
        axis.set_title("Descriptive tournament screening scores")
        fig.tight_layout()
        fig.savefig(
            staged,
            format="png",
            dpi=150,
            bbox_inches="tight",
            metadata={"Software": "farkle"},
        )
        plt.close(fig)

    write_artifact_with_sidecar_atomic(path, sidecar, write)


def run(
    cfg: AppConfig,
    *,
    force: bool = False,
    execution_scope: str = "root_pair",
) -> None:
    """Build JSON, Markdown, and plot outputs from compatible canonical artifacts."""

    from farkle.analysis.migration_audit import run as run_migration_audit

    roots = _roots(cfg)
    migration_path = run_migration_audit(cfg)
    migration, _ = _read_json(
        migration_path,
        operation="inventory_ignored_on_disk_artifacts",
    )
    family_manifest_path = cfg.h2h_candidate_family_manifest_path()
    family_membership_path = cfg.h2h_candidate_family_path()
    agreement_path = cfg.structure_agreement_summary_path()
    inference_path = cfg.h2h_pairwise_inference_path()
    dominance_path = cfg.h2h_dominance_summary_path()
    fronts_path = cfg.h2h_dominance_fronts_path()
    cycles_path = cfg.h2h_cycle_groups_path()
    root_h2h_path = cfg.h2h_root_agreement_path()
    performance_path = _performance_source(cfg, roots)
    family, family_sidecar = _read_json(family_manifest_path, operation="candidate_family_freeze")
    membership, _ = _read_frame(family_membership_path, operation="candidate_family_freeze")
    agreement, agreement_sidecar = _read_json(
        agreement_path, operation="selection_conditioned_method_agreement"
    )
    inference, _ = _read_frame(inference_path, operation="seat_adjusted_score_inference")
    dominance, _ = _read_json(dominance_path, operation="summarize_dominance_claims")
    fronts, _ = _read_frame(fronts_path, operation="condensation_dag_fronts")
    cycles, _ = _read_frame(cycles_path, operation="detect_strongly_connected_cycles")
    _root_h2h, root_h2h_sidecar = _read_frame(
        root_h2h_path, operation="fixed_root_h2h_agreement_diagnostic"
    )
    performance, performance_sidecar, score_column = _performance_frame(performance_path, roots)
    expected_seed_scope = "both_roots_combined" if len(roots) == 2 else "single_root"
    if {
        family_sidecar.seed_scope,
        agreement_sidecar.seed_scope,
        root_h2h_sidecar.seed_scope,
    } != {expected_seed_scope}:
        raise ValueError("report inputs use incompatible root scopes")
    family_hash = str(family["family_hash"])
    inference_hashes = set(inference["family_hash"].astype(str))
    if inference_hashes != {family_hash}:
        raise ValueError("report H2H inference does not match the frozen family hash")
    if (
        str(agreement.get("family_hash")) != family_hash
        or str(dominance.get("family_hash")) != family_hash
    ):
        raise ValueError("report agreement or dominance does not match the frozen family hash")
    if tuple(int(root) for root in family["root_seeds"]) != roots:
        raise ValueError("report candidate roots do not match configured roots")
    player_counts = list(performance_sidecar.required_player_counts)
    vectors, per_k_sources = _by_k_vectors(cfg, roots, player_counts)
    sources = [
        family_manifest_path,
        family_membership_path,
        agreement_path,
        inference_path,
        dominance_path,
        fronts_path,
        cycles_path,
        root_h2h_path,
        performance_path,
        migration_path,
        *per_k_sources,
    ]
    tournament_stability: dict[str, Any]
    if len(roots) == 2:
        rank_path = cfg.root_rank_stability_path()
        inclusion_path = cfg.root_bootstrap_top_n_inclusion_path()
        rank, _ = _read_frame(rank_path, operation="rank_stability")
        inclusion, _ = _read_frame(
            inclusion_path, operation="root_specific_bootstrap_top_n_inclusion"
        )
        sources.extend([rank_path, inclusion_path])
        tournament_stability = {
            "available": True,
            "rank_summary": _records(rank, rank.columns.tolist())[0],
            "root_specific_bootstrap_top_n_inclusion": _records(
                inclusion,
                [
                    "root_seed",
                    "strategy",
                    "top_n_size",
                    "top_n_inclusion_probability",
                ],
            ),
        }
    else:
        tournament_stability = {
            "available": False,
            "interpretation": "single_root_no_combined_root_stability_claim",
        }
    output_json = cfg.structure_report_json_path()
    output_markdown = cfg.structure_report_markdown_path()
    output_plot = cfg.structure_report_plot_path()
    outputs = [output_json, output_markdown, output_plot]
    done = stage_done_path(cfg.stage_dir("reporting"), "structure_reporting")
    if not force and stage_is_up_to_date(
        done,
        inputs=sources,
        outputs=outputs,
        cfg=cfg,
        stage="reporting",
        sidecar_artifacts=outputs,
    ):
        return
    scores = performance.set_index("strategy")[score_column].astype(float)
    best_score = float(scores.max())
    screening_leaders = sorted(scores.index[np.isclose(scores, best_score)].tolist())
    decision_counts = inference["decision_class"].astype(str).value_counts().to_dict()
    if decision_counts.get("equivalent", 0) and cfg.head2head.delta_equivalence is None:
        raise ValueError("equivalent H2H decisions require an explicit equivalence margin")
    configured_only_2p = player_counts == [2]
    unique_dominator = dominance.get("unique_best")
    h2h_role = (
        "primary_two_player_finalist_inference"
        if configured_only_2p
        else "external_two_player_finalist_diagnostic"
    )
    pair_intervals = _records(
        inference,
        [
            "pair_id",
            "strategy_a",
            "strategy_b",
            "d_ab",
            "ordinary_d_low",
            "ordinary_d_high",
            "simultaneous_d_low",
            "simultaneous_d_high",
            "holm_adjusted_p",
            "decision_class",
        ],
    )
    report: dict[str, Any] = {
        "report_contract_version": 1,
        "execution_scope": execution_scope,
        "roots": list(roots),
        "finite_grid_conditionality": True,
        "support": {
            "player_counts": player_counts,
            "k_aggregation_operation": performance_sidecar.operation,
            "k_weights": _normalized_weights(performance_sidecar, player_counts),
            "chance_baseline": "1/k",
            "missing_cell_policy": performance_sidecar.missing_cell_policy,
        },
        "conditioning": {
            "tournament_performance": performance_sidecar.conditioning,
            "h2h": "frozen_finite_grid_candidate_family",
            "winner_conditioning": "unconditional_tournament_performance_not_winner_conditioned",
        },
        "candidate_family": {
            "family_hash": family["family_hash"],
            "candidate_count": family["candidate_count"],
            "controls": family["controls"],
            "mandatory_diagnostics": family["mandatory_diagnostics"],
            "initial_cutoffs": family["initial_cutoffs"],
            "final_cutoffs": family["final_cutoffs"],
            "admission_counts": family["admission_counts"],
            "projected_workload": family["projected_workload"],
        },
        "performance": {
            "screening_score_leaders": screening_leaders,
            "leader_score": best_score,
            "interpretation": "descriptive_complete_support_tournament_screening",
        },
        "robustness": _robustness(vectors),
        "agreement": agreement,
        "h2h": {
            "role": h2h_role,
            "family_hash": family["family_hash"],
            "decision_counts": {str(key): int(value) for key, value in decision_counts.items()},
            "unresolved_pair_count": int(decision_counts.get("unresolved", 0)),
            "equivalent_pair_count": int(decision_counts.get("equivalent", 0)),
            "cycle_group_count": int(dominance["practical_cycle_group_count"]),
            "cycles": _cycle_records(cycles),
            "fronts": _records(
                fronts,
                [
                    "strategy",
                    "practical_front",
                    "statistical_front",
                    "practical_cycle_group",
                    "statistical_cycle_group",
                    "round_robin_mean_win_rate",
                    "practical_net_wins",
                    "tournament_screening_score",
                ],
            ),
            "pair_intervals": pair_intervals,
            "pair_intervals_artifact": str(inference_path),
            "unique_best": unique_dominator,
            "unique_best_claim_permitted": bool(
                configured_only_2p and dominance["unique_best_claim_permitted"]
            ),
            "equivalence_enabled": cfg.head2head.delta_equivalence is not None,
            "root_specific_stability": agreement["root_specific_h2h_stability"],
        },
        "tournament_root_stability": tournament_stability,
        "migration": {
            "report": str(migration_path),
            "ignored_artifact_count": migration["ignored_artifact_count"],
            "artifacts_deleted": migration["artifacts_deleted"],
        },
        "artifact_sources": [str(path) for path in sources],
    }
    report["claim_language"] = _claim_lines(report)
    common: dict[str, Any] = {
        "producer": "structure_reporting",
        "scope": ArtifactScope.DIAGNOSTICS,
        "source_scope": ArtifactScope.H2H_2P,
        "weighted_quantity": "descriptive_and_inferential_evidence_summary",
        "k_aggregation_method": performance_sidecar.k_aggregation_method,
        "k_weights": performance_sidecar.k_weights,
        "support_count_role": "complete_configured_support_and_frozen_finalists",
        "uncertainty_method": "retain_source_intervals_and_decisions",
        "replication_unit": "source_specific",
        "conditioning": "finite_grid_with_selection_conditioned_h2h",
        "source_artifacts": sources,
        "grouping_keys": ["family_hash"],
        "player_counts": player_counts,
        "required_player_counts": player_counts,
        "missing_cell_policy": "fail",
        "seed_scope": family_sidecar.seed_scope,
    }
    json_sidecar = make_artifact_sidecar(
        cfg,
        output_json,
        operation="render_structure_report",
        consistency_columns=list(report),
        **common,
    )
    write_json_artifact_atomic(report, output_json, sidecar=json_sidecar)
    markdown = render_markdown(report)
    markdown_sidecar = make_artifact_sidecar(
        cfg,
        output_markdown,
        operation="render_structure_report",
        consistency_columns=["claim_language", "support", "agreement", "h2h"],
        **common,
    )
    _write_text(output_markdown, markdown, markdown_sidecar)
    plot_sidecar = make_artifact_sidecar(
        cfg,
        output_plot,
        operation="render_tournament_screening_score_plot",
        consistency_columns=["strategy", score_column],
        **common,
    )
    finalists = set(membership.loc[membership["final_family"].astype(bool), "strategy"].astype(str))
    _write_plot(output_plot, performance, score_column, finalists, plot_sidecar)
    write_stage_done(
        done,
        inputs=sources,
        outputs=outputs,
        cfg=cfg,
        stage="reporting",
        sidecar_artifacts=outputs,
    )


__all__ = ["render_markdown", "run"]
