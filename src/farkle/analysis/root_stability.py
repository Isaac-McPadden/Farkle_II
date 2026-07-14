"""Two-root count combination, reproducibility, convergence, and drift diagnostics.

The two roots are independent RNG domains for one fixed simulation design. They
are combined from raw wins and exposures within k. Root differences diagnose
reproducibility; they do not estimate a root superpopulation or random effect.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import kendalltau, norm, spearmanr, t

from farkle.analysis.all_player_metrics import validate_unconditional_all_player_schema
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.random import RandomPurpose, coordinate_rng

_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "root_seed",
    "k",
    "deterministic_batch_id",
    "strategy",
    "raw_wins",
    "raw_player_game_exposures",
)


@dataclass(frozen=True)
class RootBatchCell:
    """One canonical root/k batch-metric input."""

    root_seed: int
    k: int
    path: Path


@dataclass(frozen=True)
class RootStabilityArtifacts:
    """Artifacts published by the two-root stability stage."""

    combined_by_k: tuple[Path, ...]
    across_k: Path
    discrepancies: Path
    joint_discrepancy: Path
    rank_stability: Path
    top_n_stability: Path
    bootstrap_top_n_inclusion: Path
    control_movement: Path
    shortlist_changes: Path
    matched_count_convergence: Path
    half_drift: Path

    @property
    def all_paths(self) -> tuple[Path, ...]:
        """Return every output in deterministic publication order."""

        return (
            *self.combined_by_k,
            self.across_k,
            self.discrepancies,
            self.joint_discrepancy,
            self.rank_stability,
            self.top_n_stability,
            self.bootstrap_top_n_inclusion,
            self.control_movement,
            self.shortlist_changes,
            self.matched_count_convergence,
            self.half_drift,
        )


@dataclass(frozen=True)
class _BatchMatrix:
    """Dense batch arrays aligned to a complete strategy support."""

    strategies: np.ndarray
    batch_ids: np.ndarray
    wins: np.ndarray
    exposures: np.ndarray


def _read_cell(cell: RootBatchCell) -> pd.DataFrame:
    """Read and strictly validate one unconditional root/k input."""

    validate_artifact_sidecar(
        cell.path,
        expected={
            "scope": ArtifactScope.BY_K.value,
            "conditioning": "unconditional",
        },
    )
    schema = pq.read_schema(cell.path)
    validate_unconditional_all_player_schema(schema)
    missing = sorted(set(_INPUT_COLUMNS).difference(schema.names))
    if missing:
        raise ValueError(f"{cell.path} lacks two-root inputs: {missing}")
    frame = pq.read_table(cell.path, columns=list(_INPUT_COLUMNS)).to_pandas()
    observed_roots = sorted(frame["root_seed"].dropna().astype(int).unique().tolist())
    observed_k = sorted(frame["k"].dropna().astype(int).unique().tolist())
    if observed_roots != [cell.root_seed] or observed_k != [cell.k]:
        raise ValueError(
            f"{cell.path} contains root/k {observed_roots}/{observed_k}; "
            f"expected [{cell.root_seed}]/[{cell.k}]"
        )
    keys = ["root_seed", "k", "deterministic_batch_id", "strategy"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{cell.path} contains duplicate root/k/batch/strategy cells")
    exposures = frame["raw_player_game_exposures"]
    wins = frame["raw_wins"]
    if (exposures <= 0).any() or (wins < 0).any() or (wins > exposures).any():
        raise ValueError(f"{cell.path} contains invalid wins or exposure support")
    return frame


def _ratio_mcse(wins: np.ndarray, exposures: np.ndarray) -> float | None:
    """Cluster-ratio MCSE using deterministic batches as independent units."""

    batch_count = len(wins)
    total_exposures = float(exposures.sum())
    if batch_count < 2 or total_exposures <= 0:
        return None
    rate = float(wins.sum() / total_exposures)
    residuals = wins - rate * exposures
    variance = batch_count / (batch_count - 1.0) * float(np.sum(residuals * residuals))
    return sqrt(max(variance, 0.0)) / total_exposures


def _classification(effect: float, mcse: float | None, practical_delta: float) -> str:
    """Classify performance relative to chance without implying equivalence."""

    if effect >= practical_delta:
        return "meaningfully_above"
    if effect <= -practical_delta:
        return "meaningfully_below"
    if mcse is not None and mcse > 0.0:
        critical = float(norm.ppf(0.975))
        if effect - critical * mcse > 0.0:
            return "statistically_above_below_practical"
    return "unresolved"


def _estimate_k(
    frame: pd.DataFrame,
    *,
    k: int,
    estimate_scope: str,
    root_seed: int | None,
    practical_delta: float,
) -> pd.DataFrame:
    """Estimate one root or the raw-count combination for one k."""

    rows: list[dict[str, object]] = []
    chance = 1.0 / k
    for strategy, group in frame.groupby("strategy", sort=True):
        batch_wins = group["raw_wins"].to_numpy(dtype=float)
        batch_exposures = group["raw_player_game_exposures"].to_numpy(dtype=float)
        wins = int(batch_wins.sum())
        exposures = int(batch_exposures.sum())
        rate = wins / exposures
        mcse = _ratio_mcse(batch_wins, batch_exposures)
        if mcse is None:
            interval_low = None
            interval_high = None
        else:
            degrees = len(batch_wins) - 1
            critical = float(t.ppf(0.975, degrees))
            interval_low = max(0.0, rate - critical * mcse)
            interval_high = min(1.0, rate + critical * mcse)
        effect = rate - chance
        rows.append(
            {
                "estimate_scope": estimate_scope,
                "root_seed": root_seed,
                "k": k,
                "strategy": int(cast(int, strategy)),
                "chance_baseline": chance,
                "raw_wins": wins,
                "raw_exposures": exposures,
                "raw_batches": int(
                    group[["root_seed", "deterministic_batch_id"]].drop_duplicates().shape[0]
                ),
                "win_rate": rate,
                "chance_delta": effect,
                "batch_mcse": mcse,
                "batch_interval_low": interval_low,
                "batch_interval_high": interval_high,
                "practical_delta": practical_delta,
                "performance_classification": _classification(effect, mcse, practical_delta),
            }
        )
    return pd.DataFrame(rows)


def _k_weights(cfg: AppConfig, required_k: list[int]) -> dict[int, float]:
    """Return normalized declared k weights with complete support."""

    if cfg.k_aggregation.method == "equal-k":
        weight = 1.0 / len(required_k)
        return dict.fromkeys(required_k, weight)
    declared = cfg.k_aggregation.k_weights
    if declared is None or {int(k) for k in declared} != set(required_k):
        raise ValueError("declared k weights must cover complete configured support")
    weights = {int(k): float(value) for k, value in declared.items()}
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise ValueError("declared k weights must sum to one")
    return weights


def _estimate_across_k(
    estimates: dict[int, pd.DataFrame],
    *,
    required_k: list[int],
    weights: dict[int, float],
    estimate_scope: str,
    root_seed: int | None,
    practical_delta: float,
) -> pd.DataFrame:
    """Calculate a complete-support declared-k score and independent-k MCSE."""

    support_sets = [set(frame["strategy"].astype(int)) for frame in estimates.values()]
    if not support_sets or any(support != support_sets[0] for support in support_sets[1:]):
        raise ValueError("root combination requires identical complete strategy support across k")
    strategies = sorted(support_sets[0])
    delta_maps = {
        k: frame.set_index("strategy")["chance_delta"].astype(float).to_dict()
        for k, frame in estimates.items()
    }
    mcse_maps = {
        k: frame.set_index("strategy")["batch_mcse"].astype(float).to_dict()
        for k, frame in estimates.items()
    }
    rows: list[dict[str, object]] = []
    critical = float(norm.ppf(0.975))
    for strategy in strategies:
        values = np.asarray([delta_maps[k][strategy] for k in required_k], dtype=float)
        variances = np.asarray([mcse_maps[k][strategy] ** 2 for k in required_k], dtype=float)
        weight_array = np.asarray([weights[k] for k in required_k], dtype=float)
        score = float(np.dot(weight_array, values))
        mcse = float(sqrt(np.dot(weight_array * weight_array, variances)))
        worst_position = int(np.argmin(values))
        rows.append(
            {
                "estimate_scope": estimate_scope,
                "root_seed": root_seed,
                "strategy": strategy,
                "required_k_count": len(required_k),
                "support_k_count": len(required_k),
                "complete_support": True,
                "k_aggregation_method": cfg_method_name(weights, required_k),
                "across_k_score": score,
                "across_k_mcse": mcse,
                "across_k_interval_low": score - critical * mcse,
                "across_k_interval_high": score + critical * mcse,
                "minimum_chance_delta": float(values[worst_position]),
                "worst_k": required_k[worst_position],
                "practical_delta": practical_delta,
                "performance_classification": _classification(score, mcse, practical_delta),
            }
        )
    return pd.DataFrame(rows)


def cfg_method_name(weights: dict[int, float], required_k: list[int]) -> str:
    """Name equal or declared weighting from exact normalized values."""

    equal = 1.0 / len(required_k)
    if all(abs(weights[k] - equal) <= 1e-15 for k in required_k):
        return "equal_k_mean"
    return "declared_k_weighted_mean"


def _build_matrix(frame: pd.DataFrame, strategies: np.ndarray) -> _BatchMatrix:
    """Pivot one root/k input into aligned batch matrices."""

    batch_ids = np.sort(frame["deterministic_batch_id"].astype(int).unique())
    wins = frame.pivot(index="deterministic_batch_id", columns="strategy", values="raw_wins")
    exposures = frame.pivot(
        index="deterministic_batch_id",
        columns="strategy",
        values="raw_player_game_exposures",
    )
    wins = wins.reindex(index=batch_ids, columns=strategies)
    exposures = exposures.reindex(index=batch_ids, columns=strategies)
    if wins.isna().any().any() or exposures.isna().any().any():
        raise ValueError("every root/k batch must contain every strategy exactly once")
    return _BatchMatrix(
        strategies=strategies,
        batch_ids=batch_ids,
        wins=wins.to_numpy(dtype=float),
        exposures=exposures.to_numpy(dtype=float),
    )


def _rank_vector(frame: pd.DataFrame, score_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Return stable strategy order and one-based ranks."""

    ordered = frame.sort_values(
        [score_column, "strategy"], ascending=[False, True], kind="mergesort"
    )
    strategies = ordered["strategy"].to_numpy(dtype=np.int64)
    ranks = np.arange(1, len(ordered) + 1, dtype=np.int64)
    return strategies, ranks


def _rank_map(frame: pd.DataFrame, score_column: str) -> dict[int, int]:
    strategies, ranks = _rank_vector(frame, score_column)
    return dict(zip(strategies.tolist(), ranks.tolist(), strict=True))


def _correlations(rank_a: dict[int, int], rank_b: dict[int, int]) -> tuple[float, float]:
    """Return Spearman and Kendall correlations on identical strategy support."""

    if set(rank_a) != set(rank_b):
        raise ValueError("rank stability requires identical strategy support")
    strategies = sorted(rank_a)
    a = np.asarray([rank_a[strategy] for strategy in strategies], dtype=float)
    b = np.asarray([rank_b[strategy] for strategy in strategies], dtype=float)
    if len(strategies) < 2:
        return 1.0, 1.0
    return float(spearmanr(a, b).statistic), float(kendalltau(a, b).statistic)


def _rank_and_selection_stability(
    cfg: AppConfig,
    roots: tuple[int, int],
    across_by_scope: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build rank, top-N, control, and shortlist reproducibility artifacts."""

    root_a, root_b = roots
    frame_a = across_by_scope[f"root_{root_a}"]
    frame_b = across_by_scope[f"root_{root_b}"]
    combined = across_by_scope["combined_roots"]
    rank_a = _rank_map(frame_a, "across_k_score")
    rank_b = _rank_map(frame_b, "across_k_score")
    rank_combined = _rank_map(combined, "across_k_score")
    spearman, kendall = _correlations(rank_a, rank_b)
    movements = np.asarray([abs(rank_a[s] - rank_b[s]) for s in sorted(rank_a)], dtype=float)
    cutoff = min(cfg.screening.candidate_contribution_size, len(rank_combined))
    combined_top = {strategy for strategy, rank in rank_combined.items() if rank <= cutoff}
    top_movements = np.asarray([abs(rank_a[s] - rank_b[s]) for s in combined_top], dtype=float)
    rank_summary = pd.DataFrame(
        [
            {
                "root_a": root_a,
                "root_b": root_b,
                "strategy_count": len(rank_a),
                "spearman_rank_correlation": spearman,
                "kendall_rank_correlation": kendall,
                "median_absolute_rank_movement": float(np.median(movements)),
                "p95_absolute_rank_movement": float(np.quantile(movements, 0.95)),
                "maximum_absolute_rank_movement": float(movements.max(initial=0.0)),
                "combined_candidate_cutoff": cutoff,
                "combined_top_median_absolute_rank_movement": float(np.median(top_movements)),
                "combined_top_maximum_absolute_rank_movement": float(
                    top_movements.max(initial=0.0)
                ),
            }
        ]
    )

    requested = sorted({10, 25, 50, cfg.screening.candidate_contribution_size})
    top_rows: list[dict[str, int | float]] = []
    for top_n in requested:
        effective = min(top_n, len(rank_a))
        set_a = {s for s, rank in rank_a.items() if rank <= effective}
        set_b = {s for s, rank in rank_b.items() if rank <= effective}
        set_combined = {s for s, rank in rank_combined.items() if rank <= effective}
        intersection = set_a & set_b
        union = set_a | set_b
        top_rows.append(
            {
                "requested_top_n": top_n,
                "effective_top_n": effective,
                "root_overlap_count": len(intersection),
                "root_jaccard": len(intersection) / len(union) if union else 1.0,
                "root_a_combined_overlap_count": len(set_a & set_combined),
                "root_b_combined_overlap_count": len(set_b & set_combined),
            }
        )

    score_maps: dict[str, dict[int, float]] = {
        scope: {
            int(cast(int, strategy)): float(score)
            for strategy, score in frame.set_index("strategy")["across_k_score"].items()
        }
        for scope, frame in across_by_scope.items()
    }
    control_rows: list[dict[str, int | float]] = []
    for control in sorted(set(cfg.screening.controls)):
        if control not in rank_combined:
            raise ValueError(f"declared control {control} lacks complete two-root support")
        control_rows.append(
            {
                "strategy": control,
                "root_a_rank": rank_a[control],
                "root_b_rank": rank_b[control],
                "combined_rank": rank_combined[control],
                "absolute_rank_movement": abs(rank_a[control] - rank_b[control]),
                "root_a_score": score_maps[f"root_{root_a}"][control],
                "root_b_score": score_maps[f"root_{root_b}"][control],
                "combined_score": score_maps["combined_roots"][control],
                "raw_score_difference": score_maps[f"root_{root_a}"][control]
                - score_maps[f"root_{root_b}"][control],
            }
        )

    delta = cfg.screening.delta_across_k
    if delta is None:
        raise ValueError("screening.delta_across_k is required for shortlist stability")
    shortlist_sets: dict[str, set[int]] = {}
    for scope, scores in score_maps.items():
        leader = max(scores.values())
        shortlist_sets[scope] = {
            strategy for strategy, score in scores.items() if score >= leader - delta
        }
    shortlist_rows = []
    for strategy in sorted(rank_combined):
        in_a = strategy in shortlist_sets[f"root_{root_a}"]
        in_b = strategy in shortlist_sets[f"root_{root_b}"]
        in_combined = strategy in shortlist_sets["combined_roots"]
        shortlist_rows.append(
            {
                "strategy": strategy,
                "root_a_shortlist": in_a,
                "root_b_shortlist": in_b,
                "combined_shortlist": in_combined,
                "root_shortlist_changed": in_a != in_b,
                "combined_changed_from_either_root": in_combined != in_a or in_combined != in_b,
            }
        )
    return (
        rank_summary,
        pd.DataFrame(top_rows),
        pd.DataFrame(control_rows),
        pd.DataFrame(shortlist_rows),
    )


def _root_bootstrap_top_n_inclusion(
    cfg: AppConfig,
    cells: dict[tuple[int, int], RootBatchCell],
    roots: tuple[int, int],
    required_k: list[int],
) -> pd.DataFrame:
    """Estimate root-specific top-N inclusion by joint complete-support resampling."""

    weights = _k_weights(cfg, required_k)
    replicates = cfg.screening.bootstrap_replicates
    reference = _read_cell(cells[(roots[0], required_k[0])])
    strategies = np.sort(reference["strategy"].astype(int).unique())
    if strategies.size == 0:
        raise ValueError("root bootstrap top-N inclusion requires strategy support")
    matrices: dict[tuple[int, int], _BatchMatrix] = {}
    for root in roots:
        for k in required_k:
            matrices[(root, k)] = _build_matrix(_read_cell(cells[(root, k)]), strategies)
    top_n = min(cfg.screening.candidate_contribution_size, len(strategies))
    rows: list[dict[str, object]] = []
    for root in roots:
        counts = np.zeros(len(strategies), dtype=np.int64)
        for replicate in range(replicates):
            scores = np.zeros(len(strategies), dtype=float)
            for k in required_k:
                matrix = matrices[(root, k)]
                rng = coordinate_rng(
                    RandomPurpose.ROOT_STABILITY_BOOTSTRAP,
                    root_seed=root,
                    k=k,
                    replicate_index=replicate,
                )
                selected = rng.integers(0, len(matrix.batch_ids), size=len(matrix.batch_ids))
                wins = matrix.wins[selected].sum(axis=0)
                exposures = matrix.exposures[selected].sum(axis=0)
                if np.any(exposures <= 0):
                    raise ValueError("root bootstrap produced zero complete-support exposure")
                scores += weights[k] * (wins / exposures - 1.0 / k)
            order = np.lexsort((strategies, -scores))
            counts[order[:top_n]] += 1
        for strategy, count in zip(strategies, counts, strict=True):
            rows.append(
                {
                    "root_seed": root,
                    "strategy": int(strategy),
                    "required_k_count": len(required_k),
                    "complete_support": True,
                    "k_aggregation_method": cfg_method_name(weights, required_k),
                    "bootstrap_replicates": replicates,
                    "top_n_size": top_n,
                    "top_n_inclusion_probability": float(count / replicates),
                }
            )
    return pd.DataFrame(rows)


def _scope_estimates(
    cfg: AppConfig,
    cells: dict[tuple[int, int], RootBatchCell],
    roots: tuple[int, int],
    required_k: list[int],
    *,
    maximum_batches: int | None = None,
) -> tuple[dict[int, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Build estimates while retaining at most one k's raw inputs in memory."""

    practical_by_k = cfg.screening.practical_delta_by_k
    if practical_by_k is None:
        raise ValueError("screening.practical_delta_by_k is required")
    if cfg.screening.delta_across_k is None:
        raise ValueError("screening.delta_across_k is required")
    weights = _k_weights(cfg, required_k)
    per_scope_k: dict[str, dict[int, pd.DataFrame]] = {f"root_{root}": {} for root in roots}
    per_scope_k["combined_roots"] = {}
    for k in required_k:
        root_frames: dict[int, pd.DataFrame] = {}
        for root in roots:
            frame = _read_cell(cells[(root, k)])
            if maximum_batches is not None:
                batch_ids = np.sort(frame["deterministic_batch_id"].astype(int).unique())
                selected = set(batch_ids[:maximum_batches].tolist())
                frame = frame.loc[frame["deterministic_batch_id"].isin(selected)].copy()
            root_frames[root] = frame
            scope = f"root_{root}"
            per_scope_k[scope][k] = _estimate_k(
                frame,
                k=k,
                estimate_scope=scope,
                root_seed=root,
                practical_delta=float(practical_by_k[k]),
            )
        combined_frame = pd.concat([root_frames[root] for root in roots], ignore_index=True)
        per_scope_k["combined_roots"][k] = _estimate_k(
            combined_frame,
            k=k,
            estimate_scope="combined_roots",
            root_seed=None,
            practical_delta=float(practical_by_k[k]),
        )

    by_k_tables = {
        k: pd.concat(
            [per_scope_k[f"root_{root}"][k] for root in roots] + [per_scope_k["combined_roots"][k]],
            ignore_index=True,
        )
        for k in required_k
    }
    across_by_scope = {
        scope: _estimate_across_k(
            estimates,
            required_k=required_k,
            weights=weights,
            estimate_scope=scope,
            root_seed=(int(scope.removeprefix("root_")) if scope.startswith("root_") else None),
            practical_delta=cfg.screening.delta_across_k,
        )
        for scope, estimates in per_scope_k.items()
    }
    return by_k_tables, across_by_scope


def _safe_standardized(raw_difference: float, expected_mcse: float) -> float:
    """Return a stable standardized discrepancy for zero-noise edge cases."""

    if expected_mcse > 0.0:
        return raw_difference / expected_mcse
    if raw_difference == 0.0:
        return 0.0
    return float(np.copysign(np.inf, raw_difference))


def _at_float(frame: pd.DataFrame, strategy: Hashable, column: str) -> float:
    """Read one known numeric scalar from a strategy-indexed frame."""

    return float(cast(float, frame.at[strategy, column]))


def _at_str(frame: pd.DataFrame, strategy: Hashable, column: str) -> str:
    """Read one known string scalar from a strategy-indexed frame."""

    return str(cast(str, frame.at[strategy, column]))


def _discrepancies(
    cfg: AppConfig,
    roots: tuple[int, int],
    by_k_tables: dict[int, pd.DataFrame],
    across_by_scope: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare root-specific performance without a root-population interval."""

    root_a, root_b = roots
    threshold = cfg.robustness.delta_seed_stability
    rows: list[dict[str, object]] = []
    for k, table in sorted(by_k_tables.items()):
        indexed = {
            scope: frame.set_index("strategy")
            for scope, frame in table.groupby("estimate_scope", sort=False)
        }
        a = indexed[f"root_{root_a}"]
        b = indexed[f"root_{root_b}"]
        combined = indexed["combined_roots"]
        if set(a.index) != set(b.index) or set(a.index) != set(combined.index):
            raise ValueError(f"root discrepancy support differs for k={k}")
        for strategy in sorted(a.index.astype(int)):
            raw = _at_float(a, strategy, "chance_delta") - _at_float(b, strategy, "chance_delta")
            expected = sqrt(
                _at_float(a, strategy, "batch_mcse") ** 2
                + _at_float(b, strategy, "batch_mcse") ** 2
            )
            rows.append(
                {
                    "estimand_scope": "by_k",
                    "k": k,
                    "strategy": strategy,
                    "root_a": root_a,
                    "root_b": root_b,
                    "root_a_estimate": _at_float(a, strategy, "chance_delta"),
                    "root_b_estimate": _at_float(b, strategy, "chance_delta"),
                    "combined_estimate": _at_float(combined, strategy, "chance_delta"),
                    "raw_difference": raw,
                    "expected_mcse": expected,
                    "standardized_discrepancy": _safe_standardized(raw, expected),
                    "stability_threshold": threshold,
                    "threshold_fraction": abs(raw) / threshold,
                    "root_a_classification": _at_str(a, strategy, "performance_classification"),
                    "root_b_classification": _at_str(b, strategy, "performance_classification"),
                    "combined_classification": _at_str(
                        combined, strategy, "performance_classification"
                    ),
                    "classification_changed": (
                        _at_str(a, strategy, "performance_classification")
                        != _at_str(b, strategy, "performance_classification")
                    ),
                }
            )

    a = across_by_scope[f"root_{root_a}"].set_index("strategy")
    b = across_by_scope[f"root_{root_b}"].set_index("strategy")
    combined = across_by_scope["combined_roots"].set_index("strategy")
    if set(a.index) != set(b.index) or set(a.index) != set(combined.index):
        raise ValueError("across-k root discrepancy support differs")
    for strategy in sorted(a.index.astype(int)):
        raw = _at_float(a, strategy, "across_k_score") - _at_float(b, strategy, "across_k_score")
        expected = sqrt(
            _at_float(a, strategy, "across_k_mcse") ** 2
            + _at_float(b, strategy, "across_k_mcse") ** 2
        )
        rows.append(
            {
                "estimand_scope": "across_k",
                "k": None,
                "strategy": strategy,
                "root_a": root_a,
                "root_b": root_b,
                "root_a_estimate": _at_float(a, strategy, "across_k_score"),
                "root_b_estimate": _at_float(b, strategy, "across_k_score"),
                "combined_estimate": _at_float(combined, strategy, "across_k_score"),
                "raw_difference": raw,
                "expected_mcse": expected,
                "standardized_discrepancy": _safe_standardized(raw, expected),
                "stability_threshold": threshold,
                "threshold_fraction": abs(raw) / threshold,
                "root_a_classification": _at_str(a, strategy, "performance_classification"),
                "root_b_classification": _at_str(b, strategy, "performance_classification"),
                "combined_classification": _at_str(
                    combined, strategy, "performance_classification"
                ),
                "classification_changed": (
                    _at_str(a, strategy, "performance_classification")
                    != _at_str(b, strategy, "performance_classification")
                ),
            }
        )
    return pd.DataFrame(rows)


def _joint_discrepancy_bootstrap(
    cfg: AppConfig,
    cells: dict[tuple[int, int], RootBatchCell],
    roots: tuple[int, int],
    required_k: list[int],
    discrepancies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate dependent discrepancy flags from joint batch-vector resampling."""

    replicates = cfg.screening.bootstrap_replicates
    alpha = cfg.robustness.joint_discrepancy_alpha
    weights = _k_weights(cfg, required_k)
    strategies = np.sort(discrepancies["strategy"].astype(int).unique())
    matrices: dict[tuple[int, int], _BatchMatrix] = {}
    for root in roots:
        for k in required_k:
            matrices[(root, k)] = _build_matrix(_read_cell(cells[(root, k)]), strategies)
    root_a, root_b = roots
    by_k_rows = {
        k: discrepancies.loc[discrepancies["k"].eq(k)].sort_values("strategy") for k in required_k
    }
    across_rows = discrepancies.loc[discrepancies["estimand_scope"].eq("across_k")].sort_values(
        "strategy"
    )
    observed_by_k = {
        k: frame["raw_difference"].to_numpy(dtype=float) for k, frame in by_k_rows.items()
    }
    expected_by_k = {
        k: frame["expected_mcse"].to_numpy(dtype=float) for k, frame in by_k_rows.items()
    }
    observed_across = across_rows["raw_difference"].to_numpy(dtype=float)
    expected_across = across_rows["expected_mcse"].to_numpy(dtype=float)
    maxima = np.zeros(replicates, dtype=float)

    for replicate in range(replicates):
        replicate_rates: dict[tuple[int, int], np.ndarray] = {}
        standardized_parts: list[np.ndarray] = []
        for root in roots:
            for k in required_k:
                matrix = matrices[(root, k)]
                rng = coordinate_rng(
                    RandomPurpose.ROOT_STABILITY_BOOTSTRAP,
                    root_seed=root,
                    k=k,
                    replicate_index=replicate,
                )
                selected = rng.integers(0, len(matrix.batch_ids), size=len(matrix.batch_ids))
                wins = matrix.wins[selected].sum(axis=0)
                exposures = matrix.exposures[selected].sum(axis=0)
                if np.any(exposures <= 0):
                    raise ValueError("joint root bootstrap produced zero strategy exposure")
                replicate_rates[(root, k)] = wins / exposures - 1.0 / k
        for k in required_k:
            replicate_difference = replicate_rates[(root_a, k)] - replicate_rates[(root_b, k)]
            centered = replicate_difference - observed_by_k[k]
            valid = expected_by_k[k] > 0.0
            standardized_parts.append(np.abs(centered[valid] / expected_by_k[k][valid]))
        replicate_across = sum(
            weights[k] * (replicate_rates[(root_a, k)] - replicate_rates[(root_b, k)])
            for k in required_k
        )
        centered_across = replicate_across - observed_across
        valid_across = expected_across > 0.0
        standardized_parts.append(
            np.abs(centered_across[valid_across] / expected_across[valid_across])
        )
        nonempty = [part for part in standardized_parts if part.size]
        maxima[replicate] = max((float(part.max()) for part in nonempty), default=0.0)

    critical = float(np.quantile(maxima, 1.0 - alpha, method="higher"))
    enriched = discrepancies.copy()
    enriched["joint_max_abs_standardized_critical"] = critical
    enriched["joint_discrepancy_flag"] = enriched["standardized_discrepancy"].abs() > critical
    enriched["joint_adjusted_diagnostic_p"] = [
        (1.0 + float(np.count_nonzero(maxima >= abs(value)))) / (replicates + 1.0)
        for value in enriched["standardized_discrepancy"].astype(float)
    ]
    finite_observed = enriched["standardized_discrepancy"].replace([np.inf, -np.inf], np.nan)
    observed_max = float(finite_observed.abs().max()) if finite_observed.notna().any() else np.inf
    summary = pd.DataFrame(
        [
            {
                "root_a": root_a,
                "root_b": root_b,
                "bootstrap_replicates": replicates,
                "diagnostic_family_alpha": alpha,
                "maximum_absolute_standardized_discrepancy": observed_max,
                "joint_max_abs_standardized_critical": critical,
                "joint_unusual": observed_max > critical,
                "flagged_estimands": int(enriched["joint_discrepancy_flag"].sum()),
                "interpretation": "reproducibility_diagnostic_not_root_random_effect",
            }
        ]
    )
    return enriched, summary


def _matched_count_convergence(
    cfg: AppConfig,
    cells: dict[tuple[int, int], RootBatchCell],
    roots: tuple[int, int],
    required_k: list[int],
    final_across: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Recompute headline stability at matched cumulative batch counts."""

    minimum_batches = min(
        _read_cell(cell)["deterministic_batch_id"].nunique() for cell in cells.values()
    )
    final_combined = final_across["combined_roots"]
    final_rank = _rank_map(final_combined, "across_k_score")
    delta = cfg.screening.delta_across_k
    if delta is None:
        raise ValueError("screening.delta_across_k is required")
    final_scores = final_combined.set_index("strategy")["across_k_score"].astype(float).to_dict()
    final_leader = max(final_scores.values())
    final_shortlist = {
        strategy for strategy, score in final_scores.items() if score >= final_leader - delta
    }
    cutoff = min(cfg.screening.candidate_contribution_size, len(final_rank))
    rows: list[dict[str, object]] = []
    for fraction in cfg.robustness.matched_count_fractions:
        matched_count = min(minimum_batches, max(1, ceil(minimum_batches * fraction)))
        _, partial_across = _scope_estimates(
            cfg,
            cells,
            roots,
            required_k,
            maximum_batches=matched_count,
        )
        root_a_scope = f"root_{roots[0]}"
        root_b_scope = f"root_{roots[1]}"
        rank_a = _rank_map(partial_across[root_a_scope], "across_k_score")
        rank_b = _rank_map(partial_across[root_b_scope], "across_k_score")
        rank_combined = _rank_map(partial_across["combined_roots"], "across_k_score")
        root_spearman, root_kendall = _correlations(rank_a, rank_b)
        root_a_final_spearman, _ = _correlations(rank_a, final_rank)
        root_b_final_spearman, _ = _correlations(rank_b, final_rank)
        combined_final_spearman, combined_final_kendall = _correlations(rank_combined, final_rank)
        set_a = {strategy for strategy, rank in rank_a.items() if rank <= cutoff}
        set_b = {strategy for strategy, rank in rank_b.items() if rank <= cutoff}
        partial_scores = (
            partial_across["combined_roots"]
            .set_index("strategy")["across_k_score"]
            .astype(float)
            .to_dict()
        )
        partial_leader = max(partial_scores.values())
        partial_shortlist = {
            strategy
            for strategy, score in partial_scores.items()
            if score >= partial_leader - delta
        }
        root_a_scores = (
            partial_across[root_a_scope].set_index("strategy")["across_k_score"].astype(float)
        )
        root_b_scores = (
            partial_across[root_b_scope].set_index("strategy")["across_k_score"].astype(float)
        )
        combined_mcse = partial_across["combined_roots"]["across_k_mcse"].astype(float)
        rows.append(
            {
                "cumulative_fraction": fraction,
                "matched_batches_per_root_k": matched_count,
                "root_spearman_rank_correlation": root_spearman,
                "root_kendall_rank_correlation": root_kendall,
                "root_a_spearman_with_final_combined": root_a_final_spearman,
                "root_b_spearman_with_final_combined": root_b_final_spearman,
                "partial_combined_spearman_with_final": combined_final_spearman,
                "partial_combined_kendall_with_final": combined_final_kendall,
                "candidate_cutoff": cutoff,
                "root_candidate_overlap_count": len(set_a & set_b),
                "root_candidate_jaccard": len(set_a & set_b) / len(set_a | set_b),
                "shortlist_symmetric_difference_from_final": len(
                    partial_shortlist ^ final_shortlist
                ),
                "median_combined_interval_half_width": float(
                    np.median(float(norm.ppf(0.975)) * combined_mcse)
                ),
                "maximum_root_raw_discrepancy": float((root_a_scores - root_b_scores).abs().max()),
                "interpretation": "matched_count_convergence_not_additional_roots",
            }
        )
    return pd.DataFrame(rows)


def _half_drift(
    cfg: AppConfig,
    cells: dict[tuple[int, int], RootBatchCell],
    roots: tuple[int, int],
    required_k: list[int],
) -> pd.DataFrame:
    """Compare contiguous first and second halves within each root."""

    practical_by_k = cfg.screening.practical_delta_by_k
    across_delta = cfg.screening.delta_across_k
    if practical_by_k is None or across_delta is None:
        raise ValueError("practical performance thresholds are required for drift diagnostics")
    weights = _k_weights(cfg, required_k)
    rows: list[dict[str, object]] = []
    for root in roots:
        half_estimates: dict[str, dict[int, pd.DataFrame]] = {
            "first_half": {},
            "second_half": {},
        }
        for k in required_k:
            frame = _read_cell(cells[(root, k)])
            batch_ids = np.sort(frame["deterministic_batch_id"].astype(int).unique())
            if len(batch_ids) < 2:
                raise ValueError(f"root {root}, k={k} needs at least two batches for drift")
            midpoint = len(batch_ids) // 2
            halves = {
                "first_half": set(batch_ids[:midpoint].tolist()),
                "second_half": set(batch_ids[midpoint:].tolist()),
            }
            for half, selected in halves.items():
                half_frame = frame.loc[frame["deterministic_batch_id"].isin(selected)]
                half_estimates[half][k] = _estimate_k(
                    half_frame,
                    k=k,
                    estimate_scope=half,
                    root_seed=root,
                    practical_delta=float(practical_by_k[k]),
                )
            first = half_estimates["first_half"][k].set_index("strategy")
            second = half_estimates["second_half"][k].set_index("strategy")
            for strategy in sorted(first.index.astype(int)):
                raw = _at_float(first, strategy, "chance_delta") - _at_float(
                    second, strategy, "chance_delta"
                )
                expected = sqrt(
                    _at_float(first, strategy, "batch_mcse") ** 2
                    + _at_float(second, strategy, "batch_mcse") ** 2
                )
                rows.append(
                    {
                        "root_seed": root,
                        "estimand_scope": "by_k",
                        "k": k,
                        "strategy": strategy,
                        "first_half_estimate": _at_float(first, strategy, "chance_delta"),
                        "second_half_estimate": _at_float(second, strategy, "chance_delta"),
                        "raw_difference": raw,
                        "expected_mcse": expected,
                        "standardized_drift": _safe_standardized(raw, expected),
                        "threshold_fraction": abs(raw) / cfg.robustness.delta_seed_stability,
                        "classification_changed": (
                            _at_str(first, strategy, "performance_classification")
                            != _at_str(second, strategy, "performance_classification")
                        ),
                        "interpretation": "within_root_drift_not_additional_root",
                    }
                )
        across_halves = {
            half: _estimate_across_k(
                estimates,
                required_k=required_k,
                weights=weights,
                estimate_scope=half,
                root_seed=root,
                practical_delta=across_delta,
            ).set_index("strategy")
            for half, estimates in half_estimates.items()
        }
        first_across = across_halves["first_half"]
        second_across = across_halves["second_half"]
        for strategy in sorted(first_across.index.astype(int)):
            raw = _at_float(first_across, strategy, "across_k_score") - _at_float(
                second_across, strategy, "across_k_score"
            )
            expected = sqrt(
                _at_float(first_across, strategy, "across_k_mcse") ** 2
                + _at_float(second_across, strategy, "across_k_mcse") ** 2
            )
            rows.append(
                {
                    "root_seed": root,
                    "estimand_scope": "across_k",
                    "k": None,
                    "strategy": strategy,
                    "first_half_estimate": _at_float(first_across, strategy, "across_k_score"),
                    "second_half_estimate": _at_float(second_across, strategy, "across_k_score"),
                    "raw_difference": raw,
                    "expected_mcse": expected,
                    "standardized_drift": _safe_standardized(raw, expected),
                    "threshold_fraction": abs(raw) / cfg.robustness.delta_seed_stability,
                    "classification_changed": (
                        _at_str(first_across, strategy, "performance_classification")
                        != _at_str(second_across, strategy, "performance_classification")
                    ),
                    "interpretation": "within_root_drift_not_additional_root",
                }
            )
    return pd.DataFrame(rows)


def _write_frame(
    cfg: AppConfig,
    frame: pd.DataFrame,
    path: Path,
    *,
    operation: str,
    sources: list[Path],
    player_counts: list[int],
    grouping_keys: list[str],
    uncertainty_method: str,
    k_aggregation_method: str = "none",
    seed_scope: str = "root_pair_stability",
) -> None:
    """Publish one hash-bound cross-seed artifact."""

    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="root_stability",
        scope=ArtifactScope.CROSS_SEED,
        source_scope=ArtifactScope.BY_K,
        operation=operation,
        baseline="chance_1_over_k",
        weighted_quantity="win_rate_minus_chance",
        k_aggregation_method=k_aggregation_method,
        k_weights=(
            cfg.k_aggregation.k_weights if k_aggregation_method == "declared_mapping" else None
        ),
        support_count_role="raw_player_game_exposures",
        uncertainty_method=uncertainty_method,
        replication_unit="deterministic_shuffle_batch",
        conditioning="unconditional_fixed_simulation_design",
        consistency_columns=frame.columns.tolist(),
        source_artifacts=sources,
        grouping_keys=grouping_keys,
        player_counts=player_counts,
        required_player_counts=player_counts,
        missing_cell_policy="fail",
        seed_scope=seed_scope,
    )
    table = pa.Table.from_pandas(frame, preserve_index=False)
    write_parquet_artifact_atomic(table, path, sidecar=sidecar, codec=cfg.parquet_codec)


def _artifact_paths(cfg: AppConfig, required_k: list[int]) -> RootStabilityArtifacts:
    return RootStabilityArtifacts(
        combined_by_k=tuple(cfg.root_combined_performance_by_k_path(k) for k in required_k),
        across_k=cfg.root_combined_performance_across_k_path(),
        discrepancies=cfg.root_discrepancies_path(),
        joint_discrepancy=cfg.root_joint_discrepancy_path(),
        rank_stability=cfg.root_rank_stability_path(),
        top_n_stability=cfg.root_top_n_stability_path(),
        bootstrap_top_n_inclusion=cfg.root_bootstrap_top_n_inclusion_path(),
        control_movement=cfg.root_control_movement_path(),
        shortlist_changes=cfg.root_shortlist_changes_path(),
        matched_count_convergence=cfg.root_matched_count_convergence_path(),
        half_drift=cfg.root_half_drift_path(),
    )


def build_two_root_stability(
    cfg: AppConfig,
    cells: list[RootBatchCell],
    *,
    force: bool = False,
) -> RootStabilityArtifacts:
    """Combine exactly two roots and publish reproducibility diagnostics."""

    required_k = sorted({int(k) for k in cfg.sim.n_players_list})
    roots = tuple(sorted({int(cell.root_seed) for cell in cells}))
    if len(roots) != 2:
        raise ValueError(f"two-root stability requires exactly two roots, found {roots}")
    root_pair = cast(tuple[int, int], roots)
    expected = {(root, k) for root in root_pair for k in required_k}
    observed = {(int(cell.root_seed), int(cell.k)) for cell in cells}
    if observed != expected or len(cells) != len(expected):
        missing = sorted(expected.difference(observed))
        extra = sorted(observed.difference(expected))
        raise ValueError(
            f"two-root inputs must cover every root/k cell; missing={missing}, extra={extra}"
        )
    cell_map = {(cell.root_seed, cell.k): cell for cell in cells}
    sources = [cell_map[key].path for key in sorted(cell_map)]
    artifacts = _artifact_paths(cfg, required_k)
    done = stage_done_path(cfg.cross_seed_dir("metrics"), "two_root_stability")
    if not force and stage_is_up_to_date(
        done,
        inputs=sources,
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=list(artifacts.all_paths),
    ):
        return artifacts

    by_k_tables, across_by_scope = _scope_estimates(cfg, cell_map, root_pair, required_k)
    discrepancies = _discrepancies(cfg, root_pair, by_k_tables, across_by_scope)
    discrepancies, joint_summary = _joint_discrepancy_bootstrap(
        cfg,
        cell_map,
        root_pair,
        required_k,
        discrepancies,
    )
    rank, top_n, controls, shortlist = _rank_and_selection_stability(
        cfg,
        root_pair,
        across_by_scope,
    )
    bootstrap_top_n = _root_bootstrap_top_n_inclusion(
        cfg,
        cell_map,
        root_pair,
        required_k,
    )
    convergence = _matched_count_convergence(
        cfg,
        cell_map,
        root_pair,
        required_k,
        across_by_scope,
    )
    drift = _half_drift(cfg, cell_map, root_pair, required_k)
    across = pd.concat(
        [across_by_scope[f"root_{root}"] for root in root_pair]
        + [across_by_scope["combined_roots"]],
        ignore_index=True,
    )
    aggregation_method = "equal_k" if cfg.k_aggregation.method == "equal-k" else "declared_mapping"

    for k, path in zip(required_k, artifacts.combined_by_k, strict=True):
        _write_frame(
            cfg,
            by_k_tables[k],
            path,
            operation="within_k_exposure_combination",
            sources=[cell_map[(root, k)].path for root in root_pair],
            player_counts=[k],
            grouping_keys=["estimate_scope", "root_seed", "k", "strategy"],
            uncertainty_method="within_root_and_combined_batch_ratio_mcse",
        )
    _write_frame(
        cfg,
        across,
        artifacts.across_k,
        operation=cfg_method_name(_k_weights(cfg, required_k), required_k),
        sources=sources,
        player_counts=required_k,
        grouping_keys=["estimate_scope", "root_seed", "strategy"],
        uncertainty_method="independent_k_variance_sum",
        k_aggregation_method=aggregation_method,
    )
    _write_frame(
        cfg,
        bootstrap_top_n,
        artifacts.bootstrap_top_n_inclusion,
        operation="root_specific_bootstrap_top_n_inclusion",
        sources=sources,
        player_counts=required_k,
        grouping_keys=["root_seed", "strategy"],
        uncertainty_method="root_specific_joint_deterministic_batch_resampling",
        k_aggregation_method=aggregation_method,
    )
    diagnostic_frames = (
        (
            discrepancies,
            artifacts.discrepancies,
            "root_difference",
            ["estimand_scope", "k", "strategy"],
            "joint_max_standardized_batch_resampling",
        ),
        (
            joint_summary,
            artifacts.joint_discrepancy,
            "joint_discrepancy_diagnostic",
            ["root_a", "root_b"],
            "joint_max_standardized_batch_resampling",
        ),
        (
            rank,
            artifacts.rank_stability,
            "rank_stability",
            ["root_a", "root_b"],
            "descriptive_rank_comparison",
        ),
        (
            top_n,
            artifacts.top_n_stability,
            "top_n_overlap",
            ["requested_top_n"],
            "descriptive_set_overlap",
        ),
        (
            controls,
            artifacts.control_movement,
            "control_movement",
            ["strategy"],
            "descriptive_root_difference",
        ),
        (
            shortlist,
            artifacts.shortlist_changes,
            "shortlist_change",
            ["strategy"],
            "declared_delta_membership",
        ),
        (
            convergence,
            artifacts.matched_count_convergence,
            "matched_count_convergence",
            ["cumulative_fraction"],
            "contiguous_batch_prefix",
        ),
        (
            drift,
            artifacts.half_drift,
            "first_half_second_half_drift",
            ["root_seed", "estimand_scope", "k", "strategy"],
            "contiguous_batch_halves",
        ),
    )
    for frame, path, operation, grouping_keys, uncertainty in diagnostic_frames:
        _write_frame(
            cfg,
            frame,
            path,
            operation=operation,
            sources=sources,
            player_counts=required_k,
            grouping_keys=grouping_keys,
            uncertainty_method=uncertainty,
        )
    write_stage_done(
        done,
        inputs=sources,
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=list(artifacts.all_paths),
    )
    return artifacts


__all__ = ["RootBatchCell", "RootStabilityArtifacts", "build_two_root_stability"]
