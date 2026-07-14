# pyright: reportArgumentType=false
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.dominance import build_dominance_outputs
from farkle.analysis.h2h_inference import run_h2h_inference
from farkle.analysis.h2h_schedule import SCORE_TEST_ID
from farkle.analysis.stage_registry import resolve_root_pair_stage_layout
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import write_json_artifact_atomic, write_parquet_artifact_atomic


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(
            seed=11,
            seed_list=[11, 22],
            n_players_list=[2],
        ),
    )
    cfg.set_stage_layout(resolve_root_pair_stage_layout(cfg))
    return cfg


def _decision_row(
    pair_id: int,
    strategy_a: str,
    strategy_b: str,
    decision: str,
) -> dict[str, object]:
    favors_a = decision.endswith("_a")
    practical = decision.startswith("practical_dominance")
    directed = decision not in {"unresolved", "equivalent"}
    effect = (0.10 if practical else 0.01) * (1 if favors_a else -1) if directed else 0.0
    return {
        "family_hash": "c" * 64,
        "pair_id": pair_id,
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "d_ab": effect,
        "balanced_a_win_rate_alias": 0.5 + effect,
        "simultaneous_d_low": effect - 0.01,
        "simultaneous_d_high": effect + 0.01,
        "holm_reject": directed,
        "practical_delta": 0.03,
        "decision_class": decision,
    }


def _publish(
    cfg: AppConfig,
    strategies: tuple[str, ...],
    decisions: dict[tuple[str, str], str],
    *,
    scores: dict[str, float] | None = None,
    reverse_rows: bool = False,
) -> None:
    rows = [
        _decision_row(pair_id, a, b, decisions[(a, b)])
        for pair_id, (a, b) in enumerate(combinations(strategies, 2))
    ]
    if reverse_rows:
        rows.reverse()
    frame = pd.DataFrame(rows)
    path = cfg.h2h_pairwise_inference_path()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="seat_adjusted_score_inference",
        consistency_columns=frame.columns.tolist(),
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined",
    )
    write_parquet_artifact_atomic(table, path, sidecar=sidecar)
    _publish_scores(cfg, strategies, scores=scores)


def _publish_scores(
    cfg: AppConfig,
    strategies: tuple[str, ...],
    *,
    scores: dict[str, float] | None = None,
) -> None:
    score_frame = pd.DataFrame(
        [
            {
                "estimate_scope": "combined_roots",
                "strategy": strategy,
                "across_k_score": (scores or {}).get(strategy, 1.0 - 0.1 * index),
                "complete_support": True,
            }
            for index, strategy in enumerate(strategies)
        ]
    )
    score_path = cfg.root_combined_performance_across_k_path()
    score_sidecar = make_artifact_sidecar(
        cfg,
        score_path,
        producer="test",
        scope=ArtifactScope.CROSS_SEED,
        source_scope=ArtifactScope.BY_K,
        operation="equal_k_mean",
        consistency_columns=score_frame.columns.tolist(),
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined",
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(score_frame, preserve_index=False),
        score_path,
        sidecar=score_sidecar,
    )


def _publish_execution_inputs(
    cfg: AppConfig,
    strategies: tuple[str, ...],
    decisions: dict[tuple[str, str], str],
    *,
    reverse_pairs: bool = False,
) -> None:
    pairs = list(combinations(strategies, 2))
    if reverse_pairs:
        pairs.reverse()
    roots = cfg.sim.seed_list or [cfg.sim.seed]
    rows: list[dict[str, object]] = []
    games = 10_000
    for pair_id, (strategy_a, strategy_b) in enumerate(pairs):
        decision = decisions[tuple(sorted((strategy_a, strategy_b), key=strategies.index))]
        favors_a = decision.endswith("_a")
        q_ab, q_ba = (0.60, 0.40) if favors_a else (0.40, 0.60)
        for root_index, root_seed in enumerate(roots):
            for order, seat1_rate in ((0, q_ab), (1, q_ba)):
                wins_seat1 = int(round(seat1_rate * games))
                rows.append(
                    {
                        "family_hash": "c" * 64,
                        "pair_id": pair_id,
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "root_seed": root_seed,
                        "root_index": root_index,
                        "order": order,
                        "order_label": "a_b" if order == 0 else "b_a",
                        "games_required": games,
                        "games_completed": games,
                        "wins_seat1": wins_seat1,
                        "wins_seat2": games - wins_seat1,
                        "score_test_id": SCORE_TEST_ID,
                    }
                )
    counts = pd.DataFrame(rows)
    counts_path = cfg.h2h_order_counts_path()
    counts_sidecar = make_artifact_sidecar(
        cfg,
        counts_path,
        producer="test",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="concatenate_root_order_blocks",
        uncertainty_method=SCORE_TEST_ID,
        consistency_columns=counts.columns.tolist(),
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined",
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(counts, preserve_index=False),
        counts_path,
        sidecar=counts_sidecar,
    )
    plan = {
        "family_hash": "c" * 64,
        "planning_state": "complete_valid",
        "execution_state": "complete_valid",
        "root_seeds": roots,
        "unordered_pair_count": len(pairs),
        "target_power": 0.80,
        "worst_scenario_achieved_power": 0.81,
    }
    plan_path = cfg.h2h_power_plan_path()
    plan_sidecar = make_artifact_sidecar(
        cfg,
        plan_path,
        producer="test",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="score_test_power_plan",
        consistency_columns=list(plan),
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined",
    )
    write_json_artifact_atomic(plan, plan_path, sidecar=plan_sidecar)


def _cycle_decisions(strategies: tuple[str, str, str, str]) -> dict[tuple[str, str], str]:
    a, b, c, d = strategies
    return {
        (a, b): "practical_dominance_a",
        (a, c): "practical_dominance_b",
        (a, d): "practical_dominance_a",
        (b, c): "practical_dominance_a",
        (b, d): "practical_dominance_a",
        (c, d): "practical_dominance_a",
    }


def test_cycles_remain_explicit_and_fronts_use_condensation_dag(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    strategies = ("A", "B", "C", "D")
    _publish(cfg, strategies, _cycle_decisions(strategies))

    artifacts = build_dominance_outputs(cfg)

    cycles = pq.read_table(artifacts.cycles).to_pandas()
    practical_cycle = cycles.loc[cycles["graph_type"] == "practical"]
    assert set(practical_cycle["strategy"]) == {"A", "B", "C"}
    assert practical_cycle["cycle_group"].nunique() == 1
    assert practical_cycle["cycle_size"].eq(3).all()
    assert practical_cycle["members_json"].nunique() == 1
    assert json.loads(practical_cycle["members_json"].iloc[0]) == ["A", "B", "C"]
    assert json.loads(practical_cycle["representative_shortest_cycle_json"].iloc[0]) == [
        "A",
        "B",
        "C",
    ]
    assert practical_cycle["strongest_internal_practical_distance"].tolist() == pytest.approx(
        [0.06, 0.06, 0.06]
    )
    assert practical_cycle["weakest_internal_practical_distance"].tolist() == pytest.approx(
        [0.06, 0.06, 0.06]
    )

    fronts = pq.read_table(artifacts.fronts).to_pandas().set_index("strategy")
    assert fronts.loc["A", "practical_front"] == 1
    assert fronts.loc["B", "practical_front"] == 1
    assert fronts.loc["C", "practical_front"] == 1
    assert fronts.loc["D", "practical_front"] == 2
    assert not fronts["display_order_is_inferential"].any()

    summary = json.loads(artifacts.summary.read_text(encoding="utf-8"))
    assert summary["practical_cycle_group_count"] == 1
    assert summary["unique_best"] is None
    assert summary["unique_best_claim_permitted"] is False

    edges = pq.read_table(artifacts.edges).to_pandas()
    assert len(edges.loc[edges["graph_type"] == "practical"]) == 6
    assert len(edges.loc[edges["graph_type"] == "statistical"]) == 6
    for path in artifacts.all_paths:
        validate_artifact_sidecar(path, expected={"scope": "h2h_2p"})


def test_unique_best_requires_direct_practical_dominance_over_every_finalist(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    strategies = ("A", "B", "C", "D")
    decisions = {
        ("A", "B"): "practical_dominance_a",
        ("A", "C"): "practical_dominance_a",
        ("A", "D"): "practical_dominance_a",
        ("B", "C"): "unresolved",
        ("B", "D"): "unresolved",
        ("C", "D"): "unresolved",
    }
    _publish(cfg, strategies, decisions)

    artifacts = build_dominance_outputs(cfg)
    summary = json.loads(artifacts.summary.read_text(encoding="utf-8"))
    fronts = pq.read_table(artifacts.fronts).to_pandas().set_index("strategy")

    assert summary["unique_best"] == "A"
    assert summary["unique_best_claim_permitted"] is True
    assert fronts.loc["A", "practical_front"] == 1
    assert set(fronts.loc[["B", "C", "D"], "practical_front"]) == {2}
    assert summary["decision_counts"]["unresolved"] == 3


def test_screening_score_is_the_third_display_key(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    strategies = ("A", "B", "C")
    decisions = dict.fromkeys(combinations(strategies, 2), "unresolved")
    _publish(
        cfg,
        strategies,
        decisions,
        scores={"A": 0.1, "B": 0.3, "C": 0.2},
    )

    fronts = pq.read_table(build_dominance_outputs(cfg).fronts).to_pandas()

    assert fronts.sort_values("practical_display_position_within_front")["strategy"].tolist() == [
        "B",
        "C",
        "A",
    ]
    assert "tournament_score" not in fronts.columns
    assert fronts.set_index("strategy").loc["B", "tournament_screening_score"] == pytest.approx(0.3)


def test_identifier_renaming_does_not_change_graph_structure(tmp_path: Path) -> None:
    first_cfg = _cfg(tmp_path / "first")
    first = ("A", "B", "C", "D")
    first_scores = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    _publish(first_cfg, first, _cycle_decisions(first), scores=first_scores)
    first_artifacts = build_dominance_outputs(first_cfg)

    renamed_cfg = _cfg(tmp_path / "renamed")
    renamed = ("W", "X", "Y", "Z")
    mapping = dict(zip(first, renamed, strict=True))
    renamed_scores = {mapping[key]: value for key, value in first_scores.items()}
    _publish(
        renamed_cfg,
        renamed,
        _cycle_decisions(renamed),
        scores=renamed_scores,
        reverse_rows=True,
    )
    renamed_artifacts = build_dominance_outputs(renamed_cfg)

    first_fronts = pq.read_table(first_artifacts.fronts).to_pandas()
    renamed_fronts = pq.read_table(renamed_artifacts.fronts).to_pandas()
    assert sorted(first_fronts["practical_front"].tolist()) == sorted(
        renamed_fronts["practical_front"].tolist()
    )
    first_cycles = pq.read_table(first_artifacts.cycles).to_pandas()
    renamed_cycles = pq.read_table(renamed_artifacts.cycles).to_pandas()
    assert sorted(first_cycles["cycle_size"].tolist()) == sorted(
        renamed_cycles["cycle_size"].tolist()
    )
    first_summary = json.loads(first_artifacts.summary.read_text(encoding="utf-8"))
    renamed_summary = json.loads(renamed_artifacts.summary.read_text(encoding="utf-8"))
    assert (
        first_summary["unique_best_claim_permitted"]
        == renamed_summary["unique_best_claim_permitted"]
    )
    mapped_first_edges = {
        (row.graph_type, mapping[row.winner], mapping[row.loser])
        for row in pq.read_table(first_artifacts.edges).to_pandas().itertuples()
    }
    renamed_edges = {
        (row.graph_type, row.winner, row.loser)
        for row in pq.read_table(renamed_artifacts.edges).to_pandas().itertuples()
    }
    assert mapped_first_edges == renamed_edges
    mapped_fronts = {
        mapping[row.strategy]: (row.practical_front, row.statistical_front)
        for row in first_fronts.itertuples()
    }
    assert mapped_fronts == {
        row.strategy: (row.practical_front, row.statistical_front)
        for row in renamed_fronts.itertuples()
    }
    mapped_cycles = {
        (
            row.graph_type,
            tuple(sorted(mapping[item] for item in json.loads(row.members_json))),
        )
        for row in first_cycles.itertuples()
    }
    assert mapped_cycles == {
        (row.graph_type, tuple(sorted(json.loads(row.members_json))))
        for row in renamed_cycles.itertuples()
    }


def test_relabelled_permuted_inference_and_dominance_are_invariant(tmp_path: Path) -> None:
    first = ("A", "B", "C", "D")
    renamed = ("W", "X", "Y", "Z")
    mapping = dict(zip(first, renamed, strict=True))
    first_decisions = _cycle_decisions(first)
    renamed_decisions = _cycle_decisions(renamed)
    first_scores = {strategy: 0.4 - 0.1 * index for index, strategy in enumerate(first)}
    renamed_scores = {mapping[key]: value for key, value in first_scores.items()}

    first_cfg = _cfg(tmp_path / "first_e2e")
    _publish_execution_inputs(first_cfg, first, first_decisions)
    first_inference = run_h2h_inference(first_cfg)
    _publish_scores(first_cfg, first, scores=first_scores)
    first_dominance = build_dominance_outputs(first_cfg)

    renamed_cfg = _cfg(tmp_path / "renamed_e2e")
    _publish_execution_inputs(
        renamed_cfg,
        renamed,
        renamed_decisions,
        reverse_pairs=True,
    )
    renamed_inference = run_h2h_inference(renamed_cfg)
    _publish_scores(renamed_cfg, renamed, scores=renamed_scores)
    renamed_dominance = build_dominance_outputs(renamed_cfg)

    def mapped_decisions(
        path: Path,
        rename: dict[str, str] | None = None,
    ) -> set[tuple[str, str, str]]:
        frame = pq.read_table(path).to_pandas()
        return {
            (
                (rename or {}).get(str(row.strategy_a), str(row.strategy_a)),
                (rename or {}).get(str(row.strategy_b), str(row.strategy_b)),
                str(row.decision_class),
            )
            for row in frame.itertuples()
        }

    assert mapped_decisions(first_inference.pairwise_inference, mapping) == mapped_decisions(
        renamed_inference.pairwise_inference
    )
    first_edges = pq.read_table(first_dominance.edges).to_pandas()
    renamed_edges = pq.read_table(renamed_dominance.edges).to_pandas()
    assert {
        (row.graph_type, mapping[row.winner], mapping[row.loser])
        for row in first_edges.itertuples()
    } == {(row.graph_type, row.winner, row.loser) for row in renamed_edges.itertuples()}
    first_fronts = pq.read_table(first_dominance.fronts).to_pandas()
    renamed_fronts = pq.read_table(renamed_dominance.fronts).to_pandas()
    assert {
        mapping[row.strategy]: (row.practical_front, row.statistical_front)
        for row in first_fronts.itertuples()
    } == {
        row.strategy: (row.practical_front, row.statistical_front)
        for row in renamed_fronts.itertuples()
    }
    first_cycles = pq.read_table(first_dominance.cycles).to_pandas()
    renamed_cycles = pq.read_table(renamed_dominance.cycles).to_pandas()
    assert {
        (
            row.graph_type,
            tuple(mapping[item] for item in json.loads(row.representative_shortest_cycle_json)),
        )
        for row in first_cycles.itertuples()
    } == {
        (row.graph_type, tuple(json.loads(row.representative_shortest_cycle_json)))
        for row in renamed_cycles.itertuples()
    }
    first_summary = json.loads(first_dominance.summary.read_text(encoding="utf-8"))
    renamed_summary = json.loads(renamed_dominance.summary.read_text(encoding="utf-8"))
    assert (
        first_summary["unique_best_claim_permitted"]
        == renamed_summary["unique_best_claim_permitted"]
    )


def test_dominance_rejects_incomplete_candidate_pairs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    strategies = ("A", "B", "C", "D")
    decisions = _cycle_decisions(strategies)
    decisions.pop(("C", "D"))
    rows = [
        _decision_row(pair_id, a, b, decision)
        for pair_id, ((a, b), decision) in enumerate(decisions.items())
    ]
    frame = pd.DataFrame(rows)
    path = cfg.h2h_pairwise_inference_path()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="seat_adjusted_score_inference",
        consistency_columns=frame.columns.tolist(),
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined",
    )
    write_parquet_artifact_atomic(table, path, sidecar=sidecar)

    with pytest.raises(ValueError, match="incomplete"):
        build_dominance_outputs(cfg)
