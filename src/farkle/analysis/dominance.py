"""Cycle-preserving practical and statistical H2H dominance fronts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import (
    write_json_artifact_atomic,
    write_parquet_artifact_atomic,
)
from farkle.utils.stage_completion import (
    stage_done_path,
    stage_is_up_to_date,
    write_stage_done,
)

_ALLOWED_DECISIONS: Final = {
    "practical_dominance_a",
    "practical_dominance_b",
    "statistical_only_advantage_a",
    "statistical_only_advantage_b",
    "equivalent",
    "unresolved",
}
_SCREENING_SCORE_OPERATIONS: Final = {"equal_k_mean", "declared_k_weighted_mean"}


@dataclass(frozen=True)
class DominanceArtifacts:
    """Graph edges, cycle groups, fronts, and claim summary."""

    edges: Path
    cycles: Path
    fronts: Path
    summary: Path

    @property
    def all_paths(self) -> tuple[Path, Path, Path, Path]:
        return (self.edges, self.cycles, self.fronts, self.summary)


@dataclass(frozen=True)
class _GraphStructure:
    fronts: dict[str, int]
    cycle_groups: dict[str, str | None]
    cycles: list[tuple[str, ...]]


def _strongly_connected_components(
    nodes: set[str], adjacency: dict[str, set[str]]
) -> list[tuple[str, ...]]:
    """Return deterministic Tarjan strongly connected components."""

    next_index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal next_index
        indices[node] = next_index
        lowlinks[node] = next_index
        next_index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in sorted(adjacency[node]):
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])
        if lowlinks[node] != indices[node]:
            return
        members: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            members.append(member)
            if member == node:
                break
        components.append(tuple(sorted(members)))

    for node in sorted(nodes):
        if node not in indices:
            visit(node)
    return sorted(components)


def _graph_structure(
    nodes: set[str], edges: set[tuple[str, str]], graph_type: str
) -> _GraphStructure:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for winner, loser in edges:
        adjacency[winner].add(loser)
    components = _strongly_connected_components(nodes, adjacency)
    component_index = {
        member: index for index, component in enumerate(components) for member in component
    }
    component_edges: set[tuple[int, int]] = set()
    for winner, loser in edges:
        source = component_index[winner]
        target = component_index[loser]
        if source != target:
            component_edges.add((source, target))
    remaining = set(range(len(components)))
    component_front: dict[int, int] = {}
    front = 1
    while remaining:
        indegree = {
            component: sum(
                1
                for source, target in component_edges
                if target == component and source in remaining
            )
            for component in remaining
        }
        layer = sorted(component for component, degree in indegree.items() if degree == 0)
        if not layer:
            raise RuntimeError("condensation graph unexpectedly contains a cycle")
        for component in layer:
            component_front[component] = front
        remaining.difference_update(layer)
        front += 1
    cycles = [component for component in components if len(component) > 1]
    cycle_id_by_component = {
        component_index[component[0]]: f"{graph_type}_cycle_{index:03d}"
        for index, component in enumerate(cycles, 1)
    }
    return _GraphStructure(
        fronts={node: component_front[component_index[node]] for node in nodes},
        cycle_groups={node: cycle_id_by_component.get(component_index[node]) for node in nodes},
        cycles=cycles,
    )


def _read_inference(cfg: AppConfig) -> pd.DataFrame:
    path = cfg.h2h_pairwise_inference_path()
    validate_artifact_sidecar(
        path,
        expected={
            "scope": ArtifactScope.H2H_2P.value,
            "operation": "seat_adjusted_score_inference",
        },
    )
    required = {
        "family_hash",
        "pair_id",
        "strategy_a",
        "strategy_b",
        "d_ab",
        "balanced_a_win_rate_alias",
        "simultaneous_d_low",
        "simultaneous_d_high",
        "holm_reject",
        "practical_delta",
        "decision_class",
    }
    schema = pq.read_schema(path)
    missing = sorted(required.difference(schema.names))
    if missing:
        raise ValueError(f"H2H inference lacks dominance columns: {missing}")
    frame = pq.read_table(path, columns=sorted(required)).to_pandas()
    if frame.empty:
        raise ValueError("H2H dominance requires nonempty pairwise inference")
    if frame["pair_id"].duplicated().any():
        raise ValueError("H2H inference contains duplicate pair identifiers")
    if frame["strategy_a"].astype(str).eq(frame["strategy_b"].astype(str)).any():
        raise ValueError("H2H dominance refuses self comparisons")
    decisions = set(frame["decision_class"].astype(str))
    unknown = sorted(decisions.difference(_ALLOWED_DECISIONS))
    if unknown:
        raise ValueError(f"H2H inference contains unknown decisions: {unknown}")
    if frame["family_hash"].astype(str).nunique() != 1:
        raise ValueError("H2H inference contains multiple candidate families")
    pair_keys = frame.apply(
        lambda row: tuple(sorted((str(row["strategy_a"]), str(row["strategy_b"])))),
        axis=1,
    )
    if pair_keys.duplicated().any():
        raise ValueError("H2H inference contains duplicate unordered comparisons")
    nodes = set(frame["strategy_a"].astype(str)) | set(frame["strategy_b"].astype(str))
    expected_pairs = len(nodes) * (len(nodes) - 1) // 2
    if len(frame) != expected_pairs:
        raise ValueError(
            f"H2H inference is incomplete for {len(nodes)} finalists: "
            f"{len(frame)} != {expected_pairs}"
        )
    return frame


def _read_tournament_screening_scores(
    cfg: AppConfig,
    nodes: set[str],
) -> tuple[dict[str, float], Path]:
    """Read the canonical tournament score used only for within-front display."""

    roots = tuple(int(root) for root in (cfg.sim.seed_list or [cfg.sim.seed]))
    if len(roots) == 2:
        path = cfg.root_combined_performance_across_k_path()
        expected_scope = ArtifactScope.CROSS_SEED.value
        required = {"estimate_scope", "strategy", "across_k_score", "complete_support"}
        score_column = "across_k_score"
    elif len(roots) == 1:
        path = cfg.performance_across_k_path()
        expected_scope = ArtifactScope.ACROSS_K.value
        required = {"strategy", "equal_k_score", "complete_support"}
        score_column = "equal_k_score"
    else:
        raise ValueError(f"dominance requires one or two configured roots, found {roots}")
    sidecar = validate_artifact_sidecar(path, expected={"scope": expected_scope})
    if sidecar.operation not in _SCREENING_SCORE_OPERATIONS:
        raise ValueError(
            f"tournament screening score operation {sidecar.operation!r} is not canonical"
        )
    schema = pq.read_schema(path)
    missing_columns = sorted(required.difference(schema.names))
    if missing_columns:
        raise ValueError(f"tournament screening scores lack columns: {missing_columns}")
    frame = pq.read_table(path, columns=sorted(required)).to_pandas()
    if len(roots) == 2:
        frame = frame.loc[frame["estimate_scope"].eq("combined_roots")].copy()
    frame["strategy"] = frame["strategy"].astype(str)
    selected = frame.loc[frame["strategy"].isin(nodes)].copy()
    if selected["strategy"].duplicated().any():
        raise ValueError("tournament screening scores contain duplicate finalist rows")
    missing_nodes = sorted(nodes.difference(set(selected["strategy"])))
    if missing_nodes:
        raise ValueError(f"tournament screening scores lack finalists: {missing_nodes}")
    incomplete = selected.loc[~selected["complete_support"].astype(bool), "strategy"].tolist()
    if incomplete:
        raise ValueError(f"tournament screening scores lack complete support: {incomplete}")
    if selected[score_column].isna().any():
        raise ValueError("tournament screening scores contain unavailable finalist values")
    return (
        dict(
            zip(
                selected["strategy"],
                selected[score_column].astype(float),
                strict=True,
            )
        ),
        path,
    )


def _tournament_screening_score_path(cfg: AppConfig) -> Path:
    roots = tuple(int(root) for root in (cfg.sim.seed_list or [cfg.sim.seed]))
    if len(roots) == 2:
        return cfg.root_combined_performance_across_k_path()
    if len(roots) == 1:
        return cfg.performance_across_k_path()
    raise ValueError(f"dominance requires one or two configured roots, found {roots}")


def _edge_frames(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, set[tuple[str, str]], set[tuple[str, str]]]:
    rows: list[dict[str, Any]] = []
    practical_edges: set[tuple[str, str]] = set()
    statistical_edges: set[tuple[str, str]] = set()
    for raw in cast(list[dict[str, Any]], frame.to_dict(orient="records")):
        decision = str(raw["decision_class"])
        strategy_a = str(raw["strategy_a"])
        strategy_b = str(raw["strategy_b"])
        directed = decision not in {"unresolved", "equivalent"}
        if bool(raw["holm_reject"]) != directed:
            raise ValueError("Holm decisions and directional decision classes disagree")
        if decision.endswith("_a"):
            winner, loser = strategy_a, strategy_b
        elif decision.endswith("_b"):
            winner, loser = strategy_b, strategy_a
        else:
            continue
        is_practical = decision.startswith("practical_dominance")
        if is_practical and not bool(raw["holm_reject"]):
            raise ValueError("practical dominance must also satisfy the Holm decision")
        if bool(raw["holm_reject"]):
            statistical_edges.add((winner, loser))
            rows.append(
                {
                    "graph_type": "statistical",
                    "pair_id": int(raw["pair_id"]),
                    "winner": winner,
                    "loser": loser,
                    "decision_class": decision,
                    "d_ab": float(raw["d_ab"]),
                    "simultaneous_d_low": float(raw["simultaneous_d_low"]),
                    "simultaneous_d_high": float(raw["simultaneous_d_high"]),
                    "practical_delta": float(raw["practical_delta"]),
                    "practical_distance_beyond_threshold": None,
                }
            )
        if is_practical:
            threshold = float(raw["practical_delta"])
            distance = (
                float(raw["simultaneous_d_low"]) - threshold
                if decision.endswith("_a")
                else -float(raw["simultaneous_d_high"]) - threshold
            )
            practical_edges.add((winner, loser))
            rows.append(
                {
                    "graph_type": "practical",
                    "pair_id": int(raw["pair_id"]),
                    "winner": winner,
                    "loser": loser,
                    "decision_class": decision,
                    "d_ab": float(raw["d_ab"]),
                    "simultaneous_d_low": float(raw["simultaneous_d_low"]),
                    "simultaneous_d_high": float(raw["simultaneous_d_high"]),
                    "practical_delta": threshold,
                    "practical_distance_beyond_threshold": distance,
                }
            )
    columns = [
        "graph_type",
        "pair_id",
        "winner",
        "loser",
        "decision_class",
        "d_ab",
        "simultaneous_d_low",
        "simultaneous_d_high",
        "practical_delta",
        "practical_distance_beyond_threshold",
    ]
    return pd.DataFrame(rows, columns=columns), practical_edges, statistical_edges


def _descriptive_scores(
    frame: pd.DataFrame,
    nodes: set[str],
    practical_edges: set[tuple[str, str]],
    statistical_edges: set[tuple[str, str]],
    tournament_scores: dict[str, float],
) -> pd.DataFrame:
    rates: dict[str, list[float]] = {node: [] for node in nodes}
    for raw in cast(list[dict[str, Any]], frame.to_dict(orient="records")):
        strategy_a = str(raw["strategy_a"])
        strategy_b = str(raw["strategy_b"])
        a_rate = float(raw["balanced_a_win_rate_alias"])
        rates[strategy_a].append(a_rate)
        rates[strategy_b].append(1.0 - a_rate)
    rows: list[dict[str, Any]] = []
    opponent_count = len(nodes) - 1
    for strategy in sorted(nodes):
        practical_wins = sum(winner == strategy for winner, _ in practical_edges)
        practical_losses = sum(loser == strategy for _, loser in practical_edges)
        statistical_wins = sum(winner == strategy for winner, _ in statistical_edges)
        statistical_losses = sum(loser == strategy for _, loser in statistical_edges)
        rows.append(
            {
                "strategy": strategy,
                "round_robin_mean_win_rate": sum(rates[strategy]) / opponent_count,
                "practical_wins": practical_wins,
                "practical_losses": practical_losses,
                "practical_net_wins": practical_wins - practical_losses,
                "statistical_wins": statistical_wins,
                "statistical_losses": statistical_losses,
                "tournament_screening_score": tournament_scores[strategy],
            }
        )
    return pd.DataFrame(rows)


def _front_frame(
    scores: pd.DataFrame,
    practical: _GraphStructure,
    statistical: _GraphStructure,
) -> pd.DataFrame:
    output = scores.copy()
    output["practical_front"] = output["strategy"].map(practical.fronts)
    output["statistical_front"] = output["strategy"].map(statistical.fronts)
    output["practical_cycle_group"] = output["strategy"].map(practical.cycle_groups)
    output["statistical_cycle_group"] = output["strategy"].map(statistical.cycle_groups)
    sort_columns = [
        "round_robin_mean_win_rate",
        "practical_net_wins",
        "tournament_screening_score",
        "strategy",
    ]
    ascending = [False, False, False, True]
    for graph_type in ("practical", "statistical"):
        front_column = f"{graph_type}_front"
        position_column = f"{graph_type}_display_position_within_front"
        positions = pd.Series(index=output.index, dtype="int64")
        for _front, indices in output.groupby(front_column).groups.items():
            ordered = output.loc[list(indices)].sort_values(
                sort_columns,
                ascending=ascending,
                kind="mergesort",
            )
            positions.loc[ordered.index] = range(1, len(ordered) + 1)
        output[position_column] = positions.astype(int)
    output["display_order_is_inferential"] = False
    return output.sort_values(
        ["practical_front", "practical_display_position_within_front"],
        kind="mergesort",
    ).reset_index(drop=True)


def _canonical_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    """Rotate a directed cycle to its stable-ID-minimal representation."""

    return min(cycle[index:] + cycle[:index] for index in range(len(cycle)))


def _representative_shortest_cycle(
    members: tuple[str, ...],
    edges: set[tuple[str, str]],
) -> tuple[str, ...]:
    """Return a deterministic shortest directed cycle within one component."""

    member_set = set(members)
    adjacency = {
        member: sorted(loser for winner, loser in edges if winner == member and loser in member_set)
        for member in members
    }
    candidates: list[tuple[str, ...]] = []
    for start in members:
        queue: list[tuple[str, tuple[str, ...]]] = [
            (neighbor, (start, neighbor)) for neighbor in adjacency[start]
        ]
        visited = set(adjacency[start])
        cursor = 0
        while cursor < len(queue):
            node, path = queue[cursor]
            cursor += 1
            for neighbor in adjacency[node]:
                if neighbor == start:
                    candidates.append(_canonical_cycle(path))
                    queue.clear()
                    break
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + (neighbor,)))
    if not candidates:
        raise RuntimeError(f"strongly connected component lacks a directed cycle: {members}")
    return min(candidates, key=lambda cycle: (len(cycle), cycle))


def _cycle_frame(
    practical: _GraphStructure,
    statistical: _GraphStructure,
    edge_frame: pd.DataFrame,
    practical_edges: set[tuple[str, str]],
    statistical_edges: set[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for graph_type, structure, graph_edges in (
        ("practical", practical, practical_edges),
        ("statistical", statistical, statistical_edges),
    ):
        for members in structure.cycles:
            cycle_id = structure.cycle_groups[members[0]]
            member_set = set(members)
            internal_practical = edge_frame.loc[
                edge_frame["graph_type"].eq("practical")
                & edge_frame["winner"].isin(member_set)
                & edge_frame["loser"].isin(member_set)
            ].copy()
            strongest_order = internal_practical.sort_values(
                ["practical_distance_beyond_threshold", "winner", "loser"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            weakest_order = internal_practical.sort_values(
                ["practical_distance_beyond_threshold", "winner", "loser"],
                ascending=[True, True, True],
                kind="mergesort",
            )
            strongest = strongest_order.iloc[0] if not strongest_order.empty else None
            weakest = weakest_order.iloc[0] if not weakest_order.empty else None
            representative = _representative_shortest_cycle(members, graph_edges)
            for member in members:
                rows.append(
                    {
                        "graph_type": graph_type,
                        "cycle_group": cycle_id,
                        "cycle_size": len(members),
                        "members_json": json.dumps(list(members), separators=(",", ":")),
                        "strategy": member,
                        "front": structure.fronts[member],
                        "internal_practical_edge_evidence_available": strongest is not None,
                        "strongest_internal_practical_winner": (
                            None if strongest is None else str(strongest["winner"])
                        ),
                        "strongest_internal_practical_loser": (
                            None if strongest is None else str(strongest["loser"])
                        ),
                        "strongest_internal_practical_distance": (
                            None
                            if strongest is None
                            else float(strongest["practical_distance_beyond_threshold"])
                        ),
                        "weakest_internal_practical_winner": (
                            None if weakest is None else str(weakest["winner"])
                        ),
                        "weakest_internal_practical_loser": (
                            None if weakest is None else str(weakest["loser"])
                        ),
                        "weakest_internal_practical_distance": (
                            None
                            if weakest is None
                            else float(weakest["practical_distance_beyond_threshold"])
                        ),
                        "representative_shortest_cycle_json": json.dumps(
                            list(representative), separators=(",", ":")
                        ),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "graph_type",
            "cycle_group",
            "cycle_size",
            "members_json",
            "strategy",
            "front",
            "internal_practical_edge_evidence_available",
            "strongest_internal_practical_winner",
            "strongest_internal_practical_loser",
            "strongest_internal_practical_distance",
            "weakest_internal_practical_winner",
            "weakest_internal_practical_loser",
            "weakest_internal_practical_distance",
            "representative_shortest_cycle_json",
        ],
    )


def _write_parquet(
    cfg: AppConfig,
    frame: pd.DataFrame,
    path: Path,
    *,
    operation: str,
    sources: list[Path],
    grouping_keys: list[str],
    seed_scope: str,
) -> None:
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="h2h_dominance",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation=operation,
        baseline="seat_adjusted_pairwise_decision",
        weighted_quantity="directed_dominance",
        support_count_role="complete_unordered_candidate_pairs",
        uncertainty_method="score_holm_and_simultaneous_practical_bounds",
        replication_unit="unordered_candidate_pair",
        conditioning="frozen_finite_grid_candidate_family",
        consistency_columns=frame.columns.tolist(),
        source_artifacts=sources,
        grouping_keys=grouping_keys,
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope=seed_scope,
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        sidecar=sidecar,
        codec=cfg.parquet_codec,
    )


def build_dominance_outputs(cfg: AppConfig, *, force: bool = False) -> DominanceArtifacts:
    """Build partial dominance fronts without converting cycles into a total order."""

    source = cfg.h2h_pairwise_inference_path()
    artifacts = DominanceArtifacts(
        edges=cfg.h2h_dominance_edges_path(),
        cycles=cfg.h2h_cycle_groups_path(),
        fronts=cfg.h2h_dominance_fronts_path(),
        summary=cfg.h2h_dominance_summary_path(),
    )
    done = stage_done_path(cfg.stage_dir("h2h_digest"), "h2h_digest")
    tournament_score_source = _tournament_screening_score_path(cfg)
    if not force and stage_is_up_to_date(
        done,
        inputs=[source, tournament_score_source],
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="h2h_digest",
        sidecar_artifacts=list(artifacts.all_paths),
    ):
        return artifacts
    inference = _read_inference(cfg)
    nodes = set(inference["strategy_a"].astype(str)) | set(inference["strategy_b"].astype(str))
    tournament_scores, tournament_score_source = _read_tournament_screening_scores(cfg, nodes)
    sources = [source, tournament_score_source]
    edges, practical_edges, statistical_edges = _edge_frames(inference)
    practical = _graph_structure(nodes, practical_edges, "practical")
    statistical = _graph_structure(nodes, statistical_edges, "statistical")
    scores = _descriptive_scores(
        inference,
        nodes,
        practical_edges,
        statistical_edges,
        tournament_scores,
    )
    fronts = _front_frame(scores, practical, statistical)
    cycles = _cycle_frame(
        practical,
        statistical,
        edges,
        practical_edges,
        statistical_edges,
    )
    direct_practical_winners = sorted(
        strategy
        for strategy in nodes
        if sum(winner == strategy for winner, _ in practical_edges) == len(nodes) - 1
    )
    if len(direct_practical_winners) > 1:
        raise RuntimeError("more than one strategy directly dominates every finalist")
    decision_counts = inference["decision_class"].astype(str).value_counts().to_dict()
    family_hash = str(inference["family_hash"].iloc[0])
    seed_scope = validate_artifact_sidecar(source).seed_scope
    summary: dict[str, Any] = {
        "family_hash": family_hash,
        "finalist_count": len(nodes),
        "unordered_pair_count": len(inference),
        "decision_counts": {str(key): int(value) for key, value in decision_counts.items()},
        "practical_edge_count": len(practical_edges),
        "statistical_edge_count": len(statistical_edges),
        "practical_cycle_group_count": len(practical.cycles),
        "statistical_cycle_group_count": len(statistical.cycles),
        "practical_front_count": max(practical.fronts.values()),
        "statistical_front_count": max(statistical.fronts.values()),
        "unique_best": direct_practical_winners[0] if direct_practical_winners else None,
        "unique_best_claim_permitted": bool(direct_practical_winners),
        "unique_best_rule": "direct_practical_dominance_over_every_other_finalist",
        "display_order_is_inferential": False,
        "interpretation": (
            "Fronts are partial dominance layers; cycle members and unresolved pairs "
            "remain explicit and display order does not add inferential edges."
        ),
    }
    _write_parquet(
        cfg,
        edges,
        artifacts.edges,
        operation="construct_dominance_graphs",
        sources=sources,
        grouping_keys=["graph_type", "winner", "loser"],
        seed_scope=seed_scope,
    )
    _write_parquet(
        cfg,
        cycles,
        artifacts.cycles,
        operation="detect_strongly_connected_cycles",
        sources=sources,
        grouping_keys=["graph_type", "cycle_group", "strategy"],
        seed_scope=seed_scope,
    )
    _write_parquet(
        cfg,
        fronts,
        artifacts.fronts,
        operation="condensation_dag_fronts",
        sources=sources,
        grouping_keys=["strategy"],
        seed_scope=seed_scope,
    )
    summary_sidecar = make_artifact_sidecar(
        cfg,
        artifacts.summary,
        producer="h2h_dominance",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="summarize_dominance_claims",
        baseline="seat_adjusted_pairwise_decision",
        weighted_quantity="directed_dominance",
        support_count_role="complete_unordered_candidate_pairs",
        uncertainty_method="score_holm_and_simultaneous_practical_bounds",
        replication_unit="unordered_candidate_pair",
        conditioning="frozen_finite_grid_candidate_family",
        consistency_columns=list(summary),
        source_artifacts=sources,
        grouping_keys=["family_hash"],
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope=seed_scope,
    )
    write_json_artifact_atomic(summary, artifacts.summary, sidecar=summary_sidecar)
    write_stage_done(
        done,
        inputs=sources,
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="h2h_digest",
        sidecar_artifacts=list(artifacts.all_paths),
    )
    return artifacts


__all__ = ["DominanceArtifacts", "build_dominance_outputs"]
