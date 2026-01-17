# src/farkle/analysis/h2h_analysis.py
"""Post head-to-head significance (Holm-Bonferroni) plus graph utilities.

Loads head-to-head decision tables, performs Holm-Bonferroni correction, and
derives graph-based tiering orders for reporting and downstream comparisons.
"""

from __future__ import annotations

import bisect
import json
import logging
from typing import cast

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import binomtest

from farkle.config import AppConfig
from farkle.utils.artifacts import write_csv_atomic, write_parquet_atomic
from farkle.utils.manifest import append_manifest_line
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)

_DECISION_SCHEMA = pa.schema(
    [
        pa.field("a", pa.string()),
        pa.field("b", pa.string()),
        pa.field("wins_a", pa.int64()),
        pa.field("wins_b", pa.int64()),
        pa.field("games", pa.int64()),
        pa.field("P1_win_rate", pa.float64()),
        pa.field("P2_win_rate", pa.float64()),
        pa.field("pval", pa.float64()),
        pa.field("adj_p", pa.float64()),
        pa.field("is_sig", pa.bool_()),
        pa.field("dir", pa.string()),
        pa.field("tie_break", pa.bool_()),
        pa.field("tie_policy", pa.string()),
    ]
)


def holm_bonferroni(
    df_pairs: pd.DataFrame,
    alpha: float,
    *,
    tie_policy: str = "neutral_edge",
    tie_break_seed: int | None = None,
) -> pd.DataFrame:
    """Apply Holm-Bonferroni to pairwise p-values with deterministic ordering."""
    if tie_policy not in {"neutral_edge", "simulate_game"}:
        raise ValueError(f"Unknown tie_policy {tie_policy!r}")

    rng = np.random.default_rng(tie_break_seed) if tie_policy == "simulate_game" else None
    required = {"a", "b", "wins_a", "wins_b", "games"}
    missing = required - set(df_pairs.columns)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise ValueError(f"pairwise dataframe missing columns: {missing_csv}")

    base = df_pairs.copy()
    base["a"] = base["a"].astype(str)
    base["b"] = base["b"].astype(str)

    played = base["wins_a"] + base["wins_b"]
    mismatched = base.loc[played != base["games"], ["a", "b", "games", "wins_a", "wins_b"]]
    if not mismatched.empty:
        raise RuntimeError(
            f"Detected wins != games for pairs: {mismatched.to_dict(orient='records')}"
        )

    ties = base.loc[base["wins_a"] == base["wins_b"], ["a", "b", "wins_a"]]
    tie_decisions = pd.DataFrame(
        columns=["a", "b", "pval", "adj_p", "is_sig", "dir", "tie_break", "tie_policy"],
        dtype=object,
    )
    if not ties.empty:
        LOGGER.warning(
            "Ties detected in head-to-head results; marking as non-significant",
            extra={
                "stage": "post_h2h",
                "tie_pairs": ties.to_dict(orient="records"),
                "tie_count": len(ties),
                "tie_policy": tie_policy,
            },
        )
        if tie_policy == "neutral_edge":
            tie_decisions = pd.DataFrame(
                {
                    "a": ties["a"].tolist(),
                    "b": ties["b"].tolist(),
                    "pval": [1.0] * len(ties),
                    "adj_p": [1.0] * len(ties),
                    "is_sig": [False] * len(ties),
                    "dir": ["tie"] * len(ties),
                    "tie_break": [False] * len(ties),
                    "tie_policy": [tie_policy] * len(ties),
                }
            )
        else:
            assert rng is not None  # for mypy
            outcomes: list[str] = []
            tie_break_flags: list[bool] = []
            for _ in ties.itertuples(index=False):
                winner_is_a = bool(rng.integers(0, 2) == 0)
                outcomes.append("a>b" if winner_is_a else "b>a")
                tie_break_flags.append(True)

            tie_decisions = pd.DataFrame(
                {
                    "a": ties["a"].tolist(),
                    "b": ties["b"].tolist(),
                    "pval": [1.0] * len(ties),
                    "adj_p": [1.0] * len(ties),
                    "is_sig": [False] * len(ties),
                    "dir": outcomes,
                    "tie_break": tie_break_flags,
                    "tie_policy": [tie_policy] * len(ties),
                }
            )

    base = base.loc[base["wins_a"] != base["wins_b"]].copy()
    if base.empty:
        return tie_decisions.astype(
            {
                "a": "string",
                "b": "string",
                "pval": "float64",
                "adj_p": "float64",
                "is_sig": "bool",
                "dir": "string",
                "tie_break": "bool",
                "tie_policy": "string",
            }
        )

    winner_is_a: pd.Series = base["wins_a"] > base["wins_b"]
    winner_wins = np.where(winner_is_a, base["wins_a"], base["wins_b"]).astype(int)
    total_games = base["games"].astype(int)
    pvals = [
        float(binomtest(wins, games, alternative="greater").pvalue)
        for wins, games in zip(winner_wins.tolist(), total_games.tolist(), strict=False)
    ]
    decisions = pd.DataFrame(
        {
            "a": base["a"].tolist(),
            "b": base["b"].tolist(),
            "pval": pvals,
            "dir": np.where(winner_is_a, "a>b", "b>a"),
        }
    )

    decisions["tie_break"] = False
    decisions["tie_policy"] = tie_policy

    if not tie_decisions.dropna(how="all").empty:
        decisions = pd.concat([decisions, tie_decisions], ignore_index=True)
    decisions = decisions.sort_values(by=["a", "b", "dir", "pval"], kind="mergesort")
    ordered = (
        decisions.reset_index(names="_idx")
        .sort_values(by=["pval", "a", "b", "dir"], kind="mergesort")
        .reset_index(drop=True)
    )
    m = len(ordered)
    running = 0.0
    adjusted: list[float] = []
    for rank, row in enumerate(ordered.itertuples(index=False), start=1):
        mult = float(m - rank + 1)
        pval = float(cast(float, row.pval))
        adj = min(pval * mult, 1.0)
        running = max(running, adj)
        adjusted.append(running)
    ordered["adj_p"] = adjusted
    adj_map = ordered.set_index("_idx")["adj_p"]
    decisions["adj_p"] = decisions.index.to_series().map(adj_map).astype(float)
    decisions["is_sig"] = decisions["adj_p"] <= float(alpha)
    decisions["tie_break"] = decisions["tie_break"].astype(bool)
    decisions["tie_policy"] = decisions["tie_policy"].astype(str)
    return decisions[["a", "b", "pval", "adj_p", "is_sig", "dir", "tie_break", "tie_policy"]]


def _aggregate_pairwise(df_pairs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mirrored matchups so seat order does not skew win rates."""

    if df_pairs.empty:
        return df_pairs.assign(P1_win_rate=pd.Series(dtype="float64"), P2_win_rate=pd.Series(dtype="float64"))

    base = df_pairs.copy()
    base["a"] = base["a"].astype(str)
    base["b"] = base["b"].astype(str)
    base["wins_a"] = pd.to_numeric(base["wins_a"], errors="coerce").fillna(0).astype(int)
    base["wins_b"] = pd.to_numeric(base["wins_b"], errors="coerce").fillna(0).astype(int)
    base["games"] = pd.to_numeric(base["games"], errors="coerce").fillna(0).astype(int)

    canonical_is_ab = base["a"] <= base["b"]
    base["p1"] = np.where(canonical_is_ab, base["a"], base["b"])
    base["p2"] = np.where(canonical_is_ab, base["b"], base["a"])
    base["p1_wins"] = np.where(canonical_is_ab, base["wins_a"], base["wins_b"])
    base["p2_wins"] = np.where(canonical_is_ab, base["wins_b"], base["wins_a"])

    grouped = (
        base.groupby(["p1", "p2"], as_index=False)[["p1_wins", "p2_wins", "games"]]
        .sum()
        .rename(
            columns={
                "p1": "a",
                "p2": "b",
                "p1_wins": "wins_a",
                "p2_wins": "wins_b",
            }
        )
    )
    grouped["P1_win_rate"] = (grouped["wins_a"] / grouped["games"]).fillna(0.0)
    grouped["P2_win_rate"] = (grouped["wins_b"] / grouped["games"]).fillna(0.0)
    return grouped[["a", "b", "wins_a", "wins_b", "games", "P1_win_rate", "P2_win_rate"]]


def build_significant_graph(
    df_adj: pd.DataFrame,
    *,
    tie_policy: str = "neutral_edge",
    tie_break_seed: int | None = None,
) -> nx.DiGraph:
    """Construct a directed graph from Holm-adjusted significance decisions."""
    expected = {"a", "b", "dir", "is_sig", "pval", "adj_p"}
    if not expected.issubset(df_adj.columns):
        raise ValueError("Adjusted dataframe missing required columns")

    graph: nx.DiGraph = nx.DiGraph()
    graph.graph["tie_policy"] = tie_policy
    graph.graph["tie_break_seed"] = tie_break_seed
    neutral_pairs: list[tuple[str, str]] = []
    tie_break_edges: list[dict[str, object]] = []
    for row in df_adj.itertuples(index=False):
        graph.add_node(row.a)
        graph.add_node(row.b)
        row_policy = getattr(row, "tie_policy", tie_policy)
        row_tie_break = bool(getattr(row, "tie_break", False))
        if row.dir == "tie":
            neutral_pairs.append((str(row.a), str(row.b)))
            continue
        if not bool(row.is_sig) and not row_tie_break:
            continue
        if row.dir == "a>b":
            source, target = row.a, row.b
        elif row.dir == "b>a":
            source, target = row.b, row.a
        else:
            raise ValueError(f"Unknown direction {row.dir!r}")
        if graph.has_edge(source, target):
            raise RuntimeError(f"Duplicate edge detected: {source}->{target}")
        pval = float(cast(float, row.pval))
        adj_p = float(cast(float, row.adj_p))
        edge_data = {
            "pval": pval,
            "adj_p": adj_p,
            "tie_break": row_tie_break,
            "tie_policy": row_policy,
        }
        graph.add_edge(source, target, **edge_data)
        if row_tie_break:
            tie_break_edges.append({"source": source, "target": target, **edge_data})
    graph.graph["neutral_pairs"] = neutral_pairs
    graph.graph["tie_break_edges"] = tie_break_edges
    return graph


def derive_sig_ranking(G: nx.DiGraph) -> list[str]:
    """Derive a deterministic ranking by topological order of significant edges."""
    if G.number_of_nodes() == 0:
        return []
    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError("Significant graph contains cycles; ranking undefined")

    indegree = {node: G.in_degree(node) for node in G.nodes}
    available: list[str] = sorted(node for node, deg in indegree.items() if deg == 0)
    ranking: list[str] = []
    while available:
        node = available.pop(0)
        ranking.append(node)
        for successor in sorted(G.successors(node)):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                bisect.insort(available, successor)
    if len(ranking) != G.number_of_nodes():
        raise RuntimeError("Ranking incomplete; graph may contain unreachable nodes")
    return ranking


def run_post_h2h(cfg: AppConfig) -> None:
    """Execute the full post head-to-head Holm + ranking workflow."""
    analysis_dir = cfg.head2head_stage_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tie_policy = getattr(cfg.head2head, "tie_break_policy", "neutral_edge")
    tie_break_seed = (
        cfg.head2head.tie_break_seed
        if getattr(cfg.head2head, "tie_break_seed", None) is not None
        else int(cfg.sim.seed)
    )
    pairwise_candidates = [
        analysis_dir / "bonferroni_pairwise.parquet",
        cfg.analysis_dir / "bonferroni_pairwise.parquet",
    ]
    pairwise_path = next((p for p in pairwise_candidates if p.exists()), pairwise_candidates[0])
    if not pairwise_path.exists():
        raise FileNotFoundError(pairwise_path)

    df_pairs = pd.read_parquet(pairwise_path, columns=["a", "b", "wins_a", "wins_b", "games"])
    df_pairs = _aggregate_pairwise(df_pairs)
    alpha = _resolve_alpha(cfg)
    LOGGER.info(
        "Post H2H: adjusting Holm-Bonferroni",
        extra={
            "stage": "post_h2h",
            "alpha": alpha,
            "rows": len(df_pairs),
            "tie_policy": tie_policy,
            "tie_break_seed": tie_break_seed,
        },
    )

    df_adj = holm_bonferroni(
        df_pairs, alpha, tie_policy=tie_policy, tie_break_seed=tie_break_seed
    )
    decisions = df_adj.merge(df_pairs, on=["a", "b"], how="left")
    decisions = decisions[[field.name for field in _DECISION_SCHEMA]]
    decisions_tbl = pa.Table.from_pandas(decisions, schema=_DECISION_SCHEMA, preserve_index=False)
    decisions_path = analysis_dir / "bonferroni_decisions.parquet"
    write_parquet_atomic(decisions_tbl, decisions_path)

    graph = build_significant_graph(df_adj, tie_policy=tie_policy, tie_break_seed=tie_break_seed)
    graph_path = analysis_dir / "h2h_significant_graph.json"
    _write_graph_json(graph, graph_path)

    ranking = derive_sig_ranking(graph)
    ranking_records = [
        {
            "rank": idx,
            "strategy": node,
            "in_degree": graph.in_degree(node),
            "out_degree": graph.out_degree(node),
        }
        for idx, node in enumerate(ranking, start=1)
    ]
    ranking_df = pd.DataFrame(
        ranking_records, columns=["rank", "strategy", "in_degree", "out_degree"]
    )
    ranking_path = analysis_dir / "h2h_significant_ranking.csv"
    write_csv_atomic(ranking_df, ranking_path)

    LOGGER.info(
        "Post H2H completed",
        extra={
            "stage": "post_h2h",
            "decisions": decisions_tbl.num_rows,
            "edges": graph.number_of_edges(),
            "ranking_nodes": len(ranking),
            "tie_policy": tie_policy,
        },
    )

    manifest_path = cfg.analysis_dir / cfg.manifest_name
    append_manifest_line(
        manifest_path,
        {
            "event": "post_h2h",
            "decisions_path": str(decisions_path),
            "graph_path": str(graph_path),
            "tie_policy": tie_policy,
            "tie_break_seed": tie_break_seed,
        },
    )


def _resolve_alpha(cfg: AppConfig) -> float:
    """Resolve the significance threshold from configuration defaults.

    Args:
        cfg: Application configuration containing head-to-head settings.

    Returns:
        Alpha value sourced from Bonferroni design or FDR fallback.
    """
    design = dict(getattr(cfg.head2head, "bonferroni_design", {}) or {})
    for key in ("control", "alpha"):
        if key in design and design[key] is not None:
            return float(design[key])
    fallback = getattr(cfg.head2head, "fdr_q", None)
    return float(fallback if fallback is not None else 0.05)


def _write_graph_json(graph: nx.DiGraph, path) -> None:
    """Serialize a head-to-head significance graph to JSON.

    Args:
        graph: Directed graph of pairwise comparisons with p-values.
        path: Destination file path for the JSON artifact.
    """
    payload = {
        "tie_policy": graph.graph.get("tie_policy"),
        "tie_break_seed": graph.graph.get("tie_break_seed"),
        "nodes": sorted(graph.nodes()),
        "edges": [
            {
                "source": source,
                "target": target,
                "pval": float(data.get("pval", float("nan"))),
                "adj_p": float(data.get("adj_p", float("nan"))),
                "tie_break": bool(data.get("tie_break", False)),
                "tie_policy": data.get("tie_policy"),
            }
            for source, target, data in sorted(
                graph.edges(data=True), key=lambda item: (item[0], item[1])
            )
        ],
        "neutral_pairs": [
            {"a": a, "b": b}
            for a, b in sorted(graph.graph.get("neutral_pairs", []), key=lambda pair: (pair[0], pair[1]))
        ],
        "tie_break_edges": sorted(
            graph.graph.get("tie_break_edges", []),
            key=lambda item: (str(item.get("source", "")), str(item.get("target", ""))),
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as tmp_path, open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
