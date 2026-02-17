"""Targeted tests for pairwise aggregation and post-H2H output stability."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from farkle.analysis import h2h_analysis
from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig, IOConfig


def _make_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    return cfg


def test_aggregate_pairwise_single_strategy_and_asymmetric_rows_have_sorted_pair_keys() -> None:
    df = pd.DataFrame(
        [
            {"a": "Gamma", "b": "Alpha", "wins_a": 3, "wins_b": 1, "games": 4},
            {"a": "Alpha", "b": "Gamma", "wins_a": 2, "wins_b": 0, "games": 2},
            {"a": "Alpha", "b": "Beta", "wins_a": 5, "wins_b": 2, "games": 7},
            {"a": "Solo", "b": "Solo", "wins_a": 4, "wins_b": 0, "games": 4},
        ]
    )

    aggregated = h2h_analysis._aggregate_pairwise(df)

    assert aggregated[["a", "b"]].to_records(index=False).tolist() == [
        ("Alpha", "Beta"),
        ("Alpha", "Gamma"),
        ("Solo", "Solo"),
    ]

    alpha_gamma = aggregated[(aggregated["a"] == "Alpha") & (aggregated["b"] == "Gamma")].iloc[0]
    assert alpha_gamma["wins_a"] == 3
    assert alpha_gamma["wins_b"] == 3
    assert alpha_gamma["games"] == 6

    solo = aggregated[(aggregated["a"] == "Solo") & (aggregated["b"] == "Solo")].iloc[0]
    assert solo["P1_win_rate"] == 1.0
    assert solo["P2_win_rate"] == 0.0


def test_holm_bonferroni_tie_heavy_input_is_stable_and_non_significant() -> None:
    df = pd.DataFrame(
        [
            {"a": "C", "b": "A", "wins_a": 5, "wins_b": 5, "games": 10},
            {"a": "B", "b": "A", "wins_a": 2, "wins_b": 2, "games": 4},
            {"a": "D", "b": "E", "wins_a": 6, "wins_b": 6, "games": 12},
        ]
    )

    decisions = h2h_analysis.holm_bonferroni(df, alpha=0.05, tie_policy="neutral_edge")

    order = decisions[["a", "b", "dir"]].to_records(index=False).tolist()
    assert order == [
        ("C", "A", "tie"),
        ("B", "A", "tie"),
        ("D", "E", "tie"),
    ]
    assert order == h2h_analysis.holm_bonferroni(df, alpha=0.05, tie_policy="neutral_edge")[["a", "b", "dir"]].to_records(index=False).tolist()
    assert decisions["is_sig"].eq(False).all()
    assert decisions["adj_p"].eq(1.0).all()
    assert decisions["tie_break"].eq(False).all()


def test_holm_bonferroni_adjustment_and_significance_filtering_branch() -> None:
    df = pd.DataFrame(
        [
            {"a": "A", "b": "B", "wins_a": 19, "wins_b": 1, "games": 20},
            {"a": "C", "b": "D", "wins_a": 12, "wins_b": 8, "games": 20},
            {"a": "E", "b": "F", "wins_a": 10, "wins_b": 10, "games": 20},
        ]
    )

    decisions = h2h_analysis.holm_bonferroni(df, alpha=0.05)
    graph = h2h_analysis.build_significant_graph(decisions)

    assert graph.has_edge("A", "B")
    assert not graph.has_edge("C", "D")
    assert ("E", "F") in graph.graph["neutral_pairs"]


def test_run_post_h2h_writes_expected_schema_and_filters_union_candidates(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)

    pairwise_path = cfg.head2head_stage_dir / "bonferroni_pairwise.parquet"
    pairwise_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"a": "B", "b": "A", "wins_a": 0, "wins_b": 20, "games": 20},
            {"a": "A", "b": "C", "wins_a": 20, "wins_b": 0, "games": 20},
        ]
    ).to_parquet(pairwise_path)

    (cfg.head2head_stage_dir / "h2h_union_candidates.json").write_text(
        json.dumps({"candidates": ["A", "B", "ghost"]}, sort_keys=True)
    )

    h2h_analysis.run_post_h2h(cfg)

    ranking_df = pd.read_csv(cfg.post_h2h_stage_dir / "h2h_significant_ranking.csv")
    assert ranking_df["strategy"].tolist() == ["A", "B", "C"]

    s_tiers = json.loads((cfg.post_h2h_stage_dir / "h2h_s_tiers.json").read_text())
    assert "ghost" not in s_tiers
    assert [name for name in s_tiers if name != "_meta"] == ["A", "B"]

    decisions_df = pd.read_parquet(cfg.post_h2h_stage_dir / "bonferroni_decisions.parquet")
    assert decisions_df.columns.tolist() == [
        "a",
        "b",
        "wins_a",
        "wins_b",
        "games",
        "P1_win_rate",
        "P2_win_rate",
        "pval",
        "adj_p",
        "is_sig",
        "dir",
        "tie_break",
        "tie_policy",
    ]


def test_topological_order_is_deterministic_for_equal_indegree_nodes() -> None:
    graph = h2h_analysis.build_significant_graph(
        pd.DataFrame(
            [
                {"a": "B", "b": "D", "dir": "a>b", "is_sig": True, "pval": 0.001, "adj_p": 0.001},
                {"a": "A", "b": "D", "dir": "a>b", "is_sig": True, "pval": 0.001, "adj_p": 0.001},
            ]
        )
    )

    assert h2h_analysis._topological_order(graph) == ["A", "B", "D"]
