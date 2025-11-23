# src/farkle/analysis/h2h_analysis.py
"""Post head-to-head significance (Holm-Bonferroni) plus graph utilities.

Loads head-to-head decision tables, performs Holm-Bonferroni correction, and
derives graph-based tiering orders for reporting and downstream comparisons.
"""

from __future__ import annotations

import bisect
import json
import logging

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import binomtest

from farkle.config import AppConfig
from farkle.utils.artifacts import write_csv_atomic, write_parquet_atomic
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)

_DECISION_SCHEMA = pa.schema(
    [
        pa.field("a", pa.string()),
        pa.field("b", pa.string()),
        pa.field("pval", pa.float64()),
        pa.field("adj_p", pa.float64()),
        pa.field("is_sig", pa.bool_()),
        pa.field("dir", pa.string()),
    ]
)


def holm_bonferroni(df_pairs: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Apply Holm-Bonferroni to pairwise p-values with deterministic ordering."""
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
    if not ties.empty:
        raise RuntimeError(
            f"Ties detected in head-to-head results: {ties.to_dict(orient='records')}"
        )

    if base.empty:
        return pd.DataFrame(
            {
                "a": pd.Series(dtype="string"),
                "b": pd.Series(dtype="string"),
                "pval": pd.Series(dtype="float64"),
                "adj_p": pd.Series(dtype="float64"),
                "is_sig": pd.Series(dtype="bool"),
                "dir": pd.Series(dtype="string"),
            }
        )

    winner_is_a = base["wins_a"] > base["wins_b"]
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

    ordered = (
        decisions.reset_index(names="_idx")
        .sort_values(by=["pval", "a", "b", "dir"], kind="mergesort")
        .reset_index(drop=True)
    )
    m = len(ordered)
    running = 0.0
    adjusted: list[float] = []
    for rank, row in enumerate(ordered.itertuples(index=False), start=1):
        mult = m - rank + 1
        adj = min(row.pval * mult, 1.0)
        running = max(running, adj)
        adjusted.append(running)
    ordered["adj_p"] = adjusted
    adj_map = ordered.set_index("_idx")["adj_p"]
    decisions["adj_p"] = decisions.index.to_series().map(adj_map).astype(float)
    decisions["is_sig"] = decisions["adj_p"] <= float(alpha)
    return decisions[["a", "b", "pval", "adj_p", "is_sig", "dir"]]


def build_significant_graph(df_adj: pd.DataFrame) -> nx.DiGraph:
    """Construct a directed graph from Holm-adjusted significance decisions."""
    expected = {"a", "b", "dir", "is_sig", "pval", "adj_p"}
    if not expected.issubset(df_adj.columns):
        raise ValueError("Adjusted dataframe missing required columns")

    graph = nx.DiGraph()
    for row in df_adj.itertuples(index=False):
        graph.add_node(row.a)
        graph.add_node(row.b)
        if not bool(row.is_sig):
            continue
        if row.dir == "a>b":
            source, target = row.a, row.b
        elif row.dir == "b>a":
            source, target = row.b, row.a
        else:
            raise ValueError(f"Unknown direction {row.dir!r}")
        if graph.has_edge(source, target):
            raise RuntimeError(f"Duplicate edge detected: {source}->{target}")
        graph.add_edge(source, target, pval=float(row.pval), adj_p=float(row.adj_p))
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
    analysis_dir = cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    pairwise_path = analysis_dir / "bonferroni_pairwise.parquet"
    if not pairwise_path.exists():
        raise FileNotFoundError(pairwise_path)

    df_pairs = pd.read_parquet(pairwise_path, columns=["a", "b", "wins_a", "wins_b", "games"])
    alpha = _resolve_alpha(cfg)
    LOGGER.info(
        "Post H2H: adjusting Holm-Bonferroni",
        extra={"stage": "post_h2h", "alpha": alpha, "rows": len(df_pairs)},
    )

    df_adj = holm_bonferroni(df_pairs, alpha)
    decisions_tbl = pa.Table.from_pandas(df_adj, schema=_DECISION_SCHEMA, preserve_index=False)
    decisions_path = analysis_dir / "bonferroni_decisions.parquet"
    write_parquet_atomic(decisions_tbl, decisions_path)

    graph = build_significant_graph(df_adj)
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
        "nodes": sorted(graph.nodes()),
        "edges": [
            {
                "source": source,
                "target": target,
                "pval": float(data.get("pval", float("nan"))),
                "adj_p": float(data.get("adj_p", float("nan"))),
            }
            for source, target, data in sorted(
                graph.edges(data=True), key=lambda item: (item[0], item[1])
            )
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as tmp_path, open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
