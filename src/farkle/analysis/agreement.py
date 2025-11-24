# src/farkle/analysis/agreement.py
"""Cross-method agreement metrics between TrueSkill, frequentist, and H2H outputs.

This module loads per-player results from multiple analytical approaches,
computes agreement metrics across rankings and tiers, and writes JSON payloads
summarizing consistency between methods.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from farkle.config import AppConfig
from farkle.utils.writer import atomic_path

try:  # optional dependency for head-to-head tiers
    from networkx import DiGraph as nx_digraph
except ModuleNotFoundError:  # pragma: no cover - optional import
    nx = None  # type: ignore[assignment]

try:  # optional dependency: h2h analysis module may require networkx
    from farkle.analysis.h2h_analysis import build_significant_graph, derive_sig_ranking
except ModuleNotFoundError:  # pragma: no cover - optional import
    build_significant_graph = None  # type: ignore[assignment]
    derive_sig_ranking = None  # type: ignore[assignment]

try:  # optional dependency for clustering agreement
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
except ModuleNotFoundError:  # pragma: no cover - optional import
    adjusted_rand_score = None  # type: ignore[assignment]
    normalized_mutual_info_score = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass
class MethodData:
    """Container for agreement inputs tied to one analytical method."""

    scores: pd.Series
    tiers: dict[str, int] | None
    per_seed_scores: list[pd.Series]


def run(cfg: AppConfig) -> None:
    """Compute rank/tier agreement metrics and write JSON payloads per player count."""

    analysis_dir = cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    player_counts = sorted({int(n) for n in cfg.sim.n_players_list}) or [0]

    for players in player_counts:
        payload = _build_payload(analysis_dir, players)
        payload["players"] = players
        out_path = analysis_dir / f"agreement_{players}p.json"
        with atomic_path(str(out_path)) as tmp_path:
            Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
        LOGGER.info(
            "Agreement metrics written",
            extra={
                "stage": "agreement",
                "players": players,
                "path": str(out_path),
            },
        )


def _build_payload(analysis_dir: Path, players: int) -> dict[str, object]:
    """Assemble agreement metrics for the requested player count.

    Args:
        analysis_dir: Directory containing prior analytics outputs.
        players: Number of players for which to compute agreement.

    Returns:
        Dictionary of correlation, stability, and coverage metrics keyed by method.
    """
    methods: dict[str, MethodData] = {}

    ts = _load_trueskill(analysis_dir, players)
    if ts is None:
        raise FileNotFoundError(analysis_dir / "ratings_pooled.parquet")
    methods["trueskill"] = ts

    freq = _load_frequentist(analysis_dir, players)
    if freq is not None:
        methods["frequentist"] = freq

    h2h = _load_head2head(analysis_dir)
    if h2h is not None:
        methods["h2h"] = h2h

    score_vectors = {name: data.scores for name, data in methods.items()}
    for name, series in score_vectors.items():
        _assert_no_ties(series, f"{name} scores")

    spearman, kendall, coverage = _rank_correlations(score_vectors)

    tier_maps = {name: _normalize_tiers(data.tiers) for name, data in methods.items() if data.tiers}
    ari, nmi = _tier_agreements(tier_maps)

    stability = {
        name: _summarize_seed_stability(data.per_seed_scores) for name, data in methods.items()
    }

    return {
        "methods": sorted(methods),
        "strategy_counts": {name: int(len(data.scores)) for name, data in methods.items()},
        "coverage": coverage or None,
        "spearman": spearman,
        "kendall": kendall,
        "ari": ari,
        "nmi": nmi,
        "seed_stability": stability,
    }


def _load_trueskill(analysis_dir: Path, players: int) -> MethodData | None:
    """Load pooled TrueSkill ratings and optional tiers for a given player count.

    Args:
        analysis_dir: Directory containing rating parquet files.
        players: Number of players to filter the ratings to.

    Returns:
        Prepared ``MethodData`` or ``None`` when no ratings are available.
    """
    path = analysis_dir / "ratings_pooled.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df = _filter_by_players(df, players)
    if "strategy" not in df.columns:
        raise ValueError("ratings_pooled.parquet missing 'strategy' column")
    if "mu" not in df.columns:
        raise ValueError("ratings_pooled.parquet missing 'mu' column")

    series = df.set_index(df["strategy"].astype(str))["mu"].astype(float).sort_index()

    tiers_path = analysis_dir / "tiers.json"
    tiers = None
    if tiers_path.exists():
        raw = json.loads(tiers_path.read_text())
        if not isinstance(raw, Mapping):
            raise ValueError("tiers.json must map strategy to tier")
        tiers = {str(k): int(v) for k, v in raw.items()}

    per_seed: list[pd.Series] = []
    for seed_path in sorted(analysis_dir.glob("ratings_pooled_seed*.parquet")):
        seed_df = pd.read_parquet(seed_path)
        seed_df = _filter_by_players(seed_df, players)
        if "strategy" not in seed_df.columns or "mu" not in seed_df.columns:
            continue
        per_seed.append(
            seed_df.set_index(seed_df["strategy"].astype(str))["mu"].astype(float).sort_index()
        )

    return MethodData(scores=series, tiers=tiers, per_seed_scores=per_seed)


def _load_frequentist(analysis_dir: Path, players: int) -> MethodData | None:
    """Load frequentist scoring outputs and optional tiers.

    Args:
        analysis_dir: Directory containing frequentist parquet outputs.
        players: Number of players to filter the scores to.

    Returns:
        Populated ``MethodData`` or ``None`` when the file is absent or empty.
    """
    path = analysis_dir / "frequentist_scores.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df = _filter_by_players(df, players)
    if df.empty:
        return None
    if "strategy" not in df.columns:
        raise ValueError("frequentist_scores.parquet missing 'strategy' column")

    score_col = _select_score_column(df, ["win_rate", "score", "estimate", "value"])
    series = df.set_index(df["strategy"].astype(str))[score_col].astype(float).sort_index()

    tier_cols = [c for c in ("tier", "tier_label", "mdd_tier") if c in df.columns]
    tiers = None
    if tier_cols:
        tier_series = df.set_index(df["strategy"].astype(str))[tier_cols[0]].dropna().astype(float)
        if not tier_series.empty:
            tiers = {str(k): int(v) for k, v in tier_series.items()}

    per_seed: list[pd.Series] = []
    for seed_path in sorted(analysis_dir.glob("frequentist_scores_seed*.parquet")):
        seed_df = pd.read_parquet(seed_path)
        seed_df = _filter_by_players(seed_df, players)
        if "strategy" not in seed_df.columns or score_col not in seed_df.columns:
            continue
        per_seed.append(
            seed_df.set_index(seed_df["strategy"].astype(str))[score_col].astype(float).sort_index()
        )

    return MethodData(scores=series, tiers=tiers, per_seed_scores=per_seed)


def _load_head2head(analysis_dir: Path) -> MethodData | None:
    """Translate head-to-head significance results into score/tier data.

    Args:
        analysis_dir: Directory containing head-to-head decision artifacts.

    Returns:
        ``MethodData`` when ranking information can be derived, otherwise ``None``.
    """
    if build_significant_graph is None or derive_sig_ranking is None or nx is None:
        LOGGER.info(
            "Agreement: skipping head-to-head inputs (networkx unavailable)",
            extra={"stage": "agreement"},
        )
        return None

    path = analysis_dir / "bonferroni_decisions.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    if df.empty:
        return None
    graph = build_significant_graph(df)
    if graph.number_of_nodes() == 0:
        return None

    ranking = derive_sig_ranking(graph)
    if not ranking:
        return None

    size = len(ranking)
    scores = pd.Series(
        {strategy: float(size - idx) for idx, strategy in enumerate(ranking)},
        dtype=float,
    ).sort_index()

    tiers = _tiers_from_graph(graph)

    return MethodData(scores=scores, tiers=tiers or None, per_seed_scores=[])


def _filter_by_players(df: pd.DataFrame, players: int) -> pd.DataFrame:
    """Restrict a DataFrame to rows matching the requested player count.

    Args:
        df: Source DataFrame containing a players column.
        players: Expected number of players.

    Returns:
        Filtered DataFrame containing only rows that match ``players``.
    """
    for column in ("players", "n_players"):
        if column in df.columns:
            mask = df[column].astype(int) == int(players)
            df = df.loc[mask]
    return df


def _select_score_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """Pick a score-like column from preferred candidates or numeric columns.

    Args:
        df: DataFrame of scores and metadata.
        candidates: Ordered column names to try first.

    Returns:
        Name of the chosen column.

    Raises:
        ValueError: If no numeric columns are available for selection.
    """
    for col in candidates:
        if col in df.columns:
            return col
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("frequentist_scores.parquet lacks a numeric score column")
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError(
        "frequentist_scores.parquet has multiple numeric columns; specify score column"
    )


def _assert_no_ties(series: pd.Series, label: str) -> None:
    """Validate that a score series does not contain duplicate values.

    Args:
        series: Scores indexed by strategy.
        label: Human-readable identifier used in error messages.

    Raises:
        ValueError: If duplicate values are detected in the series.
    """
    values = series.to_numpy()
    if np.unique(values).size != values.size:
        raise ValueError(f"Ties detected in {label}")


def _rank_correlations(
    score_vectors: Mapping[str, pd.Series],
) -> tuple[dict | None, dict | None, dict]:
    """Compute pairwise correlation statistics between scoring methods.

    Args:
        score_vectors: Mapping of method name to score series keyed by strategy.

    Returns:
        Tuple of (Spearman correlations, Kendall correlations, coverage counts).
    """
    spearman: dict[str, float | None] = {}
    kendall: dict[str, float | None] = {}
    coverage: dict[str, dict[str, int]] = {}

    for a, b in combinations(sorted(score_vectors), 2):
        series_a = score_vectors[a]
        series_b = score_vectors[b]
        common = sorted(set(series_a.index) & set(series_b.index))
        only_a = len(set(series_a.index) - set(common))
        only_b = len(set(series_b.index) - set(common))
        coverage[f"{a}_vs_{b}"] = {
            "common": len(common),
            f"only_{a}": only_a,
            f"only_{b}": only_b,
        }

        if len(common) < 2:
            spearman[f"{a}_vs_{b}"] = None
            kendall[f"{a}_vs_{b}"] = None
            continue

        a_vals = series_a.loc[common].to_numpy()
        b_vals = series_b.loc[common].to_numpy()
        corr_s = list(spearmanr(a_vals, b_vals))[0]
        corr_k = list(kendalltau(a_vals, b_vals))[0]
        corr_s_value = float(corr_s)
        corr_k_value = float(corr_k)
        spearman[f"{a}_vs_{b}"] = corr_s_value if np.isfinite(corr_s_value) else None
        kendall[f"{a}_vs_{b}"] = corr_k_value if np.isfinite(corr_k_value) else None

    return (spearman or None, kendall or None, coverage)


def _normalize_tiers(tiers: Mapping[str, int] | None) -> dict[str, int] | None:
    """Normalize arbitrary tier labels into zero-based consecutive integers.

    Args:
        tiers: Mapping of strategy to tier label.

    Returns:
        Normalized tier mapping or ``None`` when no tiers are provided.
    """
    if not tiers:
        return None
    normalized: dict[str, int] = {}
    label_map: dict[str, int] = {}
    for strategy, tier in tiers.items():
        key = str(tier)
        label_map.setdefault(key, len(label_map))
        normalized[str(strategy)] = label_map[key]
    return normalized


def _tier_agreements(
    tier_maps: Mapping[str, dict[str, int] | None],
) -> tuple[dict | None, dict | None]:
    """Calculate clustering agreement metrics for overlapping tier maps.

    Args:
        tier_maps: Mapping of method name to normalized tier assignments.

    Returns:
        Tuple of (adjusted Rand index map, normalized mutual information map).
    """
    if adjusted_rand_score is None or normalized_mutual_info_score is None:
        return (None, None)
    ari: dict[str, float | None] = {}
    nmi: dict[str, float | None] = {}

    for a, b in combinations(sorted(tier_maps), 2):
        tiers_a = tier_maps[a]
        tiers_b = tier_maps[b]
        if tiers_a is None or tiers_b is None:
            ari[f"{a}_vs_{b}"] = None
            nmi[f"{a}_vs_{b}"] = None
            continue

        common = sorted(set(tiers_a) & set(tiers_b))
        if len(common) < 2:
            ari[f"{a}_vs_{b}"] = None
            nmi[f"{a}_vs_{b}"] = None
            continue

        labels_a = [tiers_a[s] for s in common]
        labels_b = [tiers_b[s] for s in common]
        if len(set(labels_a)) <= 1 or len(set(labels_b)) <= 1:
            ari[f"{a}_vs_{b}"] = None
            nmi[f"{a}_vs_{b}"] = None
            continue

        ari_val = adjusted_rand_score(labels_a, labels_b)
        nmi_val = normalized_mutual_info_score(labels_a, labels_b)
        ari[f"{a}_vs_{b}"] = float(ari_val)
        nmi[f"{a}_vs_{b}"] = float(nmi_val)

    return (ari or None, nmi or None)


def _summarize_seed_stability(per_seed: list[pd.Series]) -> dict[str, object] | None:
    """Summarize variability of scores across seeds for common strategies.

    Args:
        per_seed: Score series per seed, indexed by strategy.

    Returns:
        Statistics describing score dispersion or ``None`` when insufficient data.
    """
    if not per_seed:
        return None

    common = set(per_seed[0].index)
    for series in per_seed[1:]:
        common &= set(series.index)
    if not common:
        return None

    order = sorted(common)
    data = np.stack([series.loc[order].to_numpy() for series in per_seed], axis=1)
    std = data.std(axis=1, ddof=0)

    if std.size == 0:
        return None

    summary = {
        "seeds": len(per_seed),
        "strategies": len(order),
        "mean_stddev": float(std.mean()),
        "median_stddev": float(np.median(std)),
        "max_stddev": float(std.max()),
        "p95_stddev": float(np.quantile(std, 0.95)),
        "top_strategies": [
            {"strategy": order[idx], "stddev": float(val)}
            for idx, val in sorted(enumerate(std), key=lambda item: item[1], reverse=True)[:5]
        ],
    }
    return summary


def _tiers_from_graph(graph: nx_digraph) -> dict[str, int]:
    """Derive tier labels from a condensed graph of significant results.

    Args:
        graph: Directed graph representing significant pairwise outcomes.

    Returns:
        Mapping of strategy identifiers to tier index, defaulting to empty when unavailable.
    """
    if nx is None:
        return {}
    if graph.number_of_nodes() == 0:
        return {}
    condensed = nx.condensation(graph)
    order = list(nx.topological_sort(condensed))
    tiers: dict[str, int] = {}
    for tier_idx, comp in enumerate(order, start=1):
        members = condensed.nodes[comp].get("members", frozenset())
        for member in sorted(members):
            tiers[str(member)] = tier_idx
    return tiers
