# src/farkle/analysis/agreement.py
"""Cross-method agreement metrics between TrueSkill, frequentist, and H2H outputs.

This module loads per-player results from multiple analytical approaches,
computes agreement metrics across rankings and tiers, and writes JSON payloads
summarizing consistency between methods.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from farkle.analysis import StageLogger, stage_logger
from farkle.analysis.h2h_analysis import build_significant_graph, derive_sig_ranking
from farkle.config import AppConfig
from farkle.utils.analysis_shared import TierMap, tiers_to_map
from farkle.utils.tiers import load_tier_payload, tier_mapping_from_payload
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)
_SEED_SUFFIX_RE = re.compile(r"_seed(\d+)$")


@dataclass
class MethodData:
    """Container for agreement inputs tied to one analytical method."""

    scores: pd.Series
    tiers: TierMap | None
    per_seed_scores: list[pd.Series]


def run(cfg: AppConfig) -> None:
    """Compute rank/tier agreement metrics and write JSON payloads per player count."""

    stage_log = stage_logger("agreement", logger=LOGGER)
    stage_log.start()

    stage_dir = cfg.agreement_stage_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    player_counts = cfg.agreement_players()

    wrote_payload = False
    summary_rows: list[dict[str, object]] = []
    for players in player_counts:
        payload = _build_payload(cfg, players=players, pooled_scope=False, stage_log=stage_log)
        if payload is None:
            continue
        payload["players"] = players
        out_path = cfg.agreement_output_path(players)
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
        summary_rows.append(_flatten_payload(payload))
        wrote_payload = True

    if cfg.agreement_include_pooled():
        payload = _build_payload(cfg, players=0, pooled_scope=True, stage_log=stage_log)
        if payload is not None:
            payload["players"] = "pooled"
            out_path = cfg.agreement_output_path_pooled()
            with atomic_path(str(out_path)) as tmp_path:
                Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
            LOGGER.info(
                "Agreement metrics written",
                extra={
                    "stage": "agreement",
                    "players": "pooled",
                    "path": str(out_path),
                },
            )
            summary_rows.append(_flatten_payload(payload))
            wrote_payload = True

    if not wrote_payload:
        stage_log.missing_input("no agreement payloads generated")
        return

    summary_path = stage_dir / "agreement_summary.parquet"
    summary_df = pd.DataFrame(summary_rows)
    with atomic_path(str(summary_path)) as tmp_path:
        summary_df.to_parquet(tmp_path, index=False)
    LOGGER.info(
        "Agreement summary table written",
        extra={
            "stage": "agreement",
            "path": str(summary_path),
            "rows": len(summary_df),
        },
    )


def _build_payload(
    cfg: AppConfig, players: int, pooled_scope: bool, stage_log: StageLogger
) -> dict[str, object] | None:
    """Assemble agreement metrics for the requested player count.

    Args:
        cfg: Application configuration with analytics output paths.
        players: Number of players for which to compute agreement (0 when pooled).

    Returns:
        Dictionary of correlation, stability, and coverage metrics keyed by method.

    Notes:
        Score ties are permitted because downstream correlation metrics already
        account for them; ties are only logged for visibility.
    """
    methods: dict[str, MethodData] = {}
    comparison_scope = "pooled" if pooled_scope else "per_k"

    try:
        ts = _load_trueskill(cfg, players, pooled_scope=pooled_scope)
    except (FileNotFoundError, ValueError) as exc:
        stage_log.missing_input(str(exc), players=players)
        return None
    if ts is None:
        stage_log.missing_input(
            "missing TrueSkill ratings",
            players=players,
            path=(
                str(cfg.trueskill_path("ratings_pooled.parquet"))
                if pooled_scope
                else f"{players}p ratings parquet"
            ),
        )
        return None
    methods["trueskill"] = ts

    try:
        freq = _load_frequentist(cfg, players)
    except ValueError as exc:
        stage_log.missing_input(str(exc), players=players)
    else:
        if freq is not None:
            methods["frequentist"] = freq

    if pooled_scope:
        try:
            h2h = _load_head2head(cfg)
        except ValueError as exc:
            stage_log.missing_input(str(exc), players=players)
        else:
            if h2h is not None:
                methods["h2h"] = h2h

    strategy_filter = getattr(cfg.analysis, "agreement_strategies", None)
    if strategy_filter:
        allowed = [str(strategy) for strategy in strategy_filter]
        methods = {
            name: _filter_method_to_strategies(data, allowed, name)
            for name, data in methods.items()
        }

    score_vectors = {name: data.scores for name, data in methods.items()}
    for name, series in score_vectors.items():
        _assert_no_ties(series, f"{name} scores")

    spearman, kendall, coverage = _rank_correlations(score_vectors)

    tier_maps = {name: data.tiers for name, data in methods.items() if data.tiers}
    ari, nmi = _tier_agreements(tier_maps)

    stability = {
        name: _summarize_seed_stability(data.per_seed_scores) for name, data in methods.items()
    }

    return {
        "methods": sorted(methods),
        "comparison_scope": {
            "mode": comparison_scope,
            "trueskill": "pooled" if pooled_scope else f"{players}p",
            "frequentist": "pooled" if pooled_scope else f"{players}p",
            "h2h": "pooled" if pooled_scope and "h2h" in methods else None,
        },
        "strategy_counts": {name: int(len(data.scores)) for name, data in methods.items()},
        "coverage": coverage or None,
        "spearman": spearman,
        "kendall": kendall,
        "ari": ari,
        "nmi": nmi,
        "seed_stability": stability,
    }


def _load_trueskill(
    cfg: AppConfig, players: int | str, *, pooled_scope: bool = False
) -> MethodData | None:
    """Load TrueSkill ratings and optional tiers for a given player count.

    Args:
        cfg: Application configuration used to locate outputs.
        players: Number of players to filter the ratings to, or pooled sentinel.

    Returns:
        Prepared ``MethodData`` or ``None`` when no ratings are available.
    """
    path = (
        cfg.trueskill_path("ratings_pooled.parquet")
        if pooled_scope
        else _resolve_trueskill_per_k_path(cfg, players)
    )
    if path is None or not path.exists():
        return None

    df = pd.read_parquet(path)
    df = _filter_by_players(df, players)
    if "strategy" not in df.columns:
        raise ValueError("ratings_pooled.parquet missing 'strategy' column")
    if "mu" not in df.columns:
        raise ValueError("ratings_pooled.parquet missing 'mu' column")

    series = df.set_index(df["strategy"].astype(str))["mu"].astype(float).sort_index()

    tiers_path = cfg.preferred_tiers_path()
    tiers_payload = load_tier_payload(tiers_path)
    tiers = tier_mapping_from_payload(tiers_payload, prefer=str(players)) or None

    per_seed: list[pd.Series] = []
    seed_candidates = _resolve_trueskill_seed_paths(cfg, players, pooled_scope=pooled_scope)
    for seed_path in sorted(seed_candidates):
        seed_df = pd.read_parquet(seed_path)
        seed_df = _filter_by_players(seed_df, players)
        if "strategy" not in seed_df.columns or "mu" not in seed_df.columns:
            continue
        per_seed.append(
            seed_df.set_index(seed_df["strategy"].astype(str))["mu"].astype(float).sort_index()
        )

    return MethodData(scores=series, tiers=tiers, per_seed_scores=per_seed)


def _resolve_trueskill_per_k_path(cfg: AppConfig, players: int | str) -> Path | None:
    """Locate per-player TrueSkill ratings parquet for a given player count."""
    if AppConfig.is_pooled_players(players):
        return None
    players_int = int(players)
    stage_dir = cfg.stage_dir_if_active("trueskill")
    roots = [root for root in (stage_dir, cfg.analysis_dir) if root is not None]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / f"{players_int}p" / f"ratings_{players_int}.parquet",
                root / f"ratings_{players_int}.parquet",
                root / "data" / f"{players_int}p" / f"ratings_{players_int}.parquet",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _resolve_trueskill_seed_paths(
    cfg: AppConfig, players: int | str, *, pooled_scope: bool
) -> list[Path]:
    """Collect candidate TrueSkill seed parquet paths for a scope."""
    seed_candidates: list[Path] = []
    if pooled_scope:
        seed_candidates.extend(cfg.analysis_dir.glob("ratings_pooled_seed*.parquet"))
        trueskill_pooled_dir = cfg.stage_dir_if_active("trueskill", "pooled")
        if trueskill_pooled_dir is not None:
            seed_candidates.extend(trueskill_pooled_dir.glob("ratings_pooled_seed*.parquet"))
        trueskill_stage_dir = cfg.stage_dir_if_active("trueskill")
        if trueskill_stage_dir is not None:
            seed_candidates.extend(trueskill_stage_dir.glob("ratings_pooled_seed*.parquet"))
        return _select_seed_paths(
            seed_candidates,
            key_fn=lambda path: (
                0 if path.parent.name == "pooled" else 1,
                str(path),
            ),
        )

    if AppConfig.is_pooled_players(players):
        return []
    players_int = int(players)
    stage_dir = cfg.stage_dir_if_active("trueskill")
    roots = [root for root in (stage_dir, cfg.analysis_dir) if root is not None]
    for root in roots:
        seed_candidates.extend(root.glob(f"{players_int}p/ratings_{players_int}_seed*.parquet"))
        seed_candidates.extend(root.glob(f"ratings_{players_int}_seed*.parquet"))
        seed_candidates.extend(root.glob(f"trueskill_{players_int}p_seed*.parquet"))
        seed_candidates.extend(
            root.glob(f"data/{players_int}p/ratings_{players_int}_seed*.parquet")
        )
    return _select_seed_paths(
        seed_candidates,
        key_fn=lambda path: (
            _trueskill_seed_path_priority(path, players_int),
            str(path),
        ),
    )


def _seed_from_path(path: Path) -> int | None:
    match = _SEED_SUFFIX_RE.search(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _select_seed_paths(paths: Iterable[Path], key_fn: Callable[[Path], tuple[int, str]]) -> list[Path]:
    selected_by_seed: dict[int, Path] = {}
    selected_keys: dict[int, tuple[int, str]] = {}
    for path in sorted(set(paths), key=str):
        seed = _seed_from_path(path)
        if seed is None:
            continue
        key = key_fn(path)
        current_key = selected_keys.get(seed)
        if current_key is None or key < current_key:
            selected_by_seed[seed] = path
            selected_keys[seed] = key
    return [selected_by_seed[seed] for seed in sorted(selected_by_seed)]


def _trueskill_seed_path_priority(path: Path, players: int) -> int:
    name = path.name
    if path.parent.name == f"{players}p" and name.startswith(f"ratings_{players}_seed"):
        return 0
    if name.startswith(f"ratings_{players}_seed"):
        return 1
    if name.startswith(f"trueskill_{players}p_seed"):
        return 2
    return 3


def _load_frequentist(cfg: AppConfig, players: int | str) -> MethodData | None:
    """Load frequentist scoring outputs and optional tiers.

    Args:
        cfg: Application configuration used to locate frequentist outputs.
        players: Number of players to filter the scores to, or pooled sentinel.

    Returns:
        Populated ``MethodData`` or ``None`` when the file is absent or empty.
    """
    path = cfg.tiering_path("frequentist_scores.parquet")
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
    seed_candidates = {*cfg.analysis_dir.glob("frequentist_scores_seed*.parquet")}
    tiering_dir = cfg.stage_dir_if_active("tiering")
    if tiering_dir is not None:
        seed_candidates.update(tiering_dir.glob("frequentist_scores_seed*.parquet"))
    for seed_path in sorted(seed_candidates):
        seed_df = pd.read_parquet(seed_path)
        seed_df = _filter_by_players(seed_df, players)
        if "strategy" not in seed_df.columns or score_col not in seed_df.columns:
            continue
        per_seed.append(
            seed_df.set_index(seed_df["strategy"].astype(str))[score_col].astype(float).sort_index()
        )

    return MethodData(scores=series, tiers=tiers, per_seed_scores=per_seed)


def _load_head2head(cfg: AppConfig) -> MethodData | None:
    """Translate head-to-head significance results into score/tier data.

    Args:
        cfg: Application configuration containing head-to-head decision paths.

    Returns:
        ``MethodData`` when ranking information can be derived, otherwise ``None``.
    """
    path = cfg.post_h2h_path("bonferroni_decisions.parquet")
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    if df.empty:
        return None
    graph = build_significant_graph(df)
    if graph.number_of_nodes() == 0:
        return None

    tier_lists = derive_sig_ranking(graph)
    if not tier_lists:
        return None

    tier_count = len(tier_lists)
    score_map: dict[str, float] = {}
    for tier_idx, strategies in enumerate(tier_lists, start=1):
        score_value = float(tier_count - tier_idx + 1)
        for strategy in strategies:
            score_map[str(strategy)] = score_value
    scores = pd.Series(score_map, dtype=float).sort_index()

    # Boundary conversion: legacy rankings are list[list[str]] and are
    # normalized once here into canonical strategy→tier mapping.
    tier_map: TierMap = tiers_to_map(tier_lists)

    return MethodData(scores=scores, tiers=tier_map or None, per_seed_scores=[])


def _filter_by_players(df: pd.DataFrame, players: int | str) -> pd.DataFrame:
    """Restrict a DataFrame to rows matching the requested player count.

    Args:
        df: Source DataFrame containing a players column.
        players: Expected number of players, or pooled sentinel.

    Returns:
        Filtered DataFrame containing only rows that match ``players``.
    """
    if "players" not in df.columns and "n_players" in df.columns:
        df = df.rename(columns={"n_players": "players"})
    if AppConfig.is_pooled_players(players):
        if "players" not in df.columns:
            return df
        mask = df["players"].astype(int) == 0
        if not mask.any():
            raise ValueError("pooled agreement requested, but no players == 0 rows found")
        return df.loc[mask]
    if "players" in df.columns:
        mask = df["players"].astype(int) == int(players)
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
    """Log a warning when a score series contains duplicate values.

    Args:
        series: Scores indexed by strategy.
        label: Human-readable identifier used in error messages.
    """
    values = series.to_numpy()
    if np.unique(values).size != values.size:
        LOGGER.warning("Ties detected in %s; correlation metrics handle ties.", label)


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
        corr_s_value = float(spearmanr(a_vals, b_vals).statistic)
        corr_k_value = float(kendalltau(a_vals, b_vals).statistic)
        spearman[f"{a}_vs_{b}"] = corr_s_value if np.isfinite(corr_s_value) else None
        kendall[f"{a}_vs_{b}"] = corr_k_value if np.isfinite(corr_k_value) else None

    return (spearman or None, kendall or None, coverage)


def _filter_method_to_strategies(
    method: MethodData, strategies: Sequence[str], method_name: str
) -> MethodData:
    """Restrict method data to a predefined set of strategies.

    Args:
        method: Score/tier collections for a single analytical method.
        strategies: Strategies that should be retained.
        method_name: Identifier used when logging missing strategies.

    Returns:
        New ``MethodData`` limited to the requested strategies.
    """

    allowed = {str(strategy) for strategy in strategies}
    present = set(method.scores.index.astype(str))
    missing = sorted(allowed - present)
    if missing:
        LOGGER.warning(
            "Agreement filtering: %s missing %d strategies: %s",
            method_name,
            len(missing),
            ", ".join(missing),
        )

    scores = method.scores[method.scores.index.isin(allowed)]
    tiers = None
    if method.tiers:
        tiers = {strategy: tier for strategy, tier in method.tiers.items() if strategy in allowed}

    per_seed = [series[series.index.isin(allowed)] for series in method.per_seed_scores]

    return MethodData(scores=scores, tiers=tiers, per_seed_scores=per_seed)



def _tier_agreements(
    tier_maps: Mapping[str, TierMap | None],
) -> tuple[dict | None, dict | None]:
    """Calculate clustering agreement metrics for overlapping tier maps.

    Args:
        tier_maps: Mapping of method name to normalized tier assignments.

    Returns:
        Tuple of (adjusted Rand index map, normalized mutual information map).
    """
    ari: dict[str, float | None] = {}
    nmi: dict[str, float | None] = {}

    for a, b in combinations(sorted(tier_maps), 2):
        tiers_a = tier_maps[a]
        tiers_b = tier_maps[b]
        if tiers_a is None or tiers_b is None:
            ari[f"{a}_vs_{b}"] = None
            nmi[f"{a}_vs_{b}"] = None
            continue

        assert tiers_a is not None
        assert tiers_b is not None

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



def _flatten_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Flatten nested payload metrics into a single-level dict."""
    flat: dict[str, object] = {}
    for key, value in payload.items():
        _flatten_value(flat, str(key), value)
    return flat


def _flatten_value(flat: dict[str, object], prefix: str, value: object) -> None:
    """Recursively flatten dict values into ``flat`` using ``prefix`` names."""
    if isinstance(value, Mapping):
        for subkey, subvalue in value.items():
            _flatten_value(flat, f"{prefix}_{subkey}", subvalue)
        return
    if isinstance(value, list):
        flat[prefix] = json.dumps(value, sort_keys=True)
        return
    flat[prefix] = value
