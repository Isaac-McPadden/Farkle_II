"""Canonical chance-adjusted performance estimators and joint batch resampling."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
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
from farkle.utils.artifact_contract import (
    make_artifact_sidecar,
    validate_artifact_sidecar,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.random import RandomPurpose, coordinate_rng
from farkle.utils.stats import wilson_ci

_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "root_seed",
    "k",
    "deterministic_batch_id",
    "strategy",
    "raw_wins",
    "raw_player_game_exposures",
)


@dataclass(frozen=True)
class PerformanceArtifacts:
    """Paths published by the canonical performance estimator."""

    by_k: tuple[Path, ...]
    across_k: Path
    bootstrap: Path
    control_contrasts: Path
    player_count_effects: Path

    @property
    def all_paths(self) -> tuple[Path, ...]:
        """Return every artifact path in deterministic order."""

        return (
            *self.by_k,
            self.across_k,
            self.bootstrap,
            self.control_contrasts,
            self.player_count_effects,
        )


def _read_batch_metrics(path: Path, k: int) -> pd.DataFrame:
    validate_artifact_sidecar(
        path,
        expected={
            "scope": ArtifactScope.BY_K.value,
            "conditioning": "unconditional",
        },
    )
    schema = pq.read_schema(path)
    validate_unconditional_all_player_schema(schema)
    missing = sorted(set(_INPUT_COLUMNS).difference(schema.names))
    if missing:
        raise ValueError(f"{path} lacks canonical performance inputs: {missing}")
    frame = pq.read_table(path, columns=list(_INPUT_COLUMNS)).to_pandas()
    observed_k = sorted(frame["k"].dropna().astype(int).unique().tolist())
    if observed_k != [k]:
        raise ValueError(f"{path} has k support {observed_k}, expected [{k}]")
    roots = sorted(frame["root_seed"].dropna().astype(int).unique().tolist())
    if len(roots) != 1:
        raise ValueError(f"{path} must contain exactly one root, found {roots}")
    if frame.duplicated(["root_seed", "k", "deterministic_batch_id", "strategy"]).any():
        raise ValueError(f"{path} contains duplicate root/k/batch/strategy cells")
    if (frame["raw_player_game_exposures"] <= 0).any():
        raise ValueError(f"{path} contains nonpositive exposure support")
    if (frame["raw_wins"] < 0).any() or (
        frame["raw_wins"] > frame["raw_player_game_exposures"]
    ).any():
        raise ValueError(f"{path} contains impossible win counts")
    return frame


def _estimate_one_k(
    frame: pd.DataFrame,
    k: int,
    resolution_delta: float,
    practical_delta: float,
) -> pd.DataFrame:
    chance = 1.0 / k
    alpha = 0.05
    rows: list[dict[str, int | float | bool | None]] = []
    for strategy, group in frame.groupby("strategy", sort=True):
        wins = int(group["raw_wins"].sum())
        exposures = int(group["raw_player_game_exposures"].sum())
        batches = int(group["deterministic_batch_id"].nunique())
        rate = wins / exposures
        batch_rates = group["raw_wins"].to_numpy(dtype=float) / group[
            "raw_player_game_exposures"
        ].to_numpy(dtype=float)
        if batches >= 2:
            mcse = float(np.std(batch_rates, ddof=1) / sqrt(batches))
            critical = float(t.ppf(1.0 - alpha / 2.0, batches - 1))
            interval_low = max(0.0, rate - critical * mcse)
            interval_high = min(1.0, rate + critical * mcse)
        else:
            mcse = None
            interval_low = None
            interval_high = None
        wilson_low, wilson_high = wilson_ci(wins, exposures, alpha=alpha)
        width = wilson_high - wilson_low
        rows.append(
            {
                "root_seed": int(group["root_seed"].iloc[0]),
                "k": k,
                "strategy": int(cast(int, strategy)),
                "chance_baseline": chance,
                "raw_wins": wins,
                "raw_exposures": exposures,
                "raw_batches": batches,
                "win_rate": rate,
                "chance_delta": rate - chance,
                "wilson_interval_low": wilson_low,
                "wilson_interval_high": wilson_high,
                "wilson_interval_width": width,
                "screening_resolution_delta": resolution_delta,
                "practical_delta_by_k": practical_delta,
                "wilson_resolution_met": width <= resolution_delta,
                "batch_mcse": mcse,
                "batch_interval_low": interval_low,
                "batch_interval_high": interval_high,
            }
        )
    return pd.DataFrame(rows)


def _pareto_membership(values: np.ndarray, strategies: np.ndarray) -> np.ndarray:
    """Return exact Pareto membership using an incrementally maintained frontier."""

    count = len(strategies)
    membership = np.zeros(count, dtype=bool)
    if count == 0:
        return membership
    order = np.lexsort((strategies, -values[:, 0]))
    frontier: list[int] = []
    for candidate in order:
        point = values[candidate]
        dominated = False
        retained: list[int] = []
        for incumbent in frontier:
            incumbent_point = values[incumbent]
            if np.all(incumbent_point >= point) and np.any(incumbent_point > point):
                dominated = True
                break
            if not (np.all(point >= incumbent_point) and np.any(point > incumbent_point)):
                retained.append(incumbent)
        if dominated:
            continue
        frontier = retained
        frontier.append(int(candidate))
    membership[np.asarray(frontier, dtype=int)] = True
    return membership


def _across_k_estimates(
    by_k: dict[int, pd.DataFrame], required_k: list[int], practical_delta: float
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    all_strategies = sorted(
        set().union(*(set(frame["strategy"].astype(int).tolist()) for frame in by_k.values()))
    )
    rows: list[dict[str, int | float | bool | None]] = []
    complete_strategies: list[int] = []
    complete_vectors: list[list[float]] = []
    root_seed = int(next(iter(by_k.values()))["root_seed"].iloc[0])
    delta_maps = {
        k: dict(
            zip(
                frame["strategy"].astype(int).tolist(),
                frame["chance_delta"].astype(float).tolist(),
                strict=True,
            )
        )
        for k, frame in by_k.items()
    }
    mcse_maps = {
        k: dict(
            zip(
                frame["strategy"].astype(int).tolist(),
                frame["batch_mcse"].astype(float).tolist(),
                strict=True,
            )
        )
        for k, frame in by_k.items()
    }
    for strategy in all_strategies:
        support = [k for k in required_k if strategy in delta_maps[k]]
        complete = support == required_k
        row: dict[str, int | float | bool | None] = {
            "root_seed": root_seed,
            "strategy": strategy,
            "required_k_count": len(required_k),
            "support_k_count": len(support),
            "complete_support": complete,
            "practical_delta_across_k": practical_delta,
            "equal_k_score": None,
            "equal_k_mcse": None,
            "equal_k_interval_low": None,
            "equal_k_interval_high": None,
            "minimum_chance_delta": None,
            "worst_k": None,
            "pareto_member": False,
            "maximin_value": None,
            "maximin_leader": False,
        }
        if complete:
            deltas = np.array([delta_maps[k][strategy] for k in required_k], dtype=np.float64)
            variances = np.array(
                [mcse_maps[k][strategy] ** 2 for k in required_k], dtype=np.float64
            )
            score = float(deltas.mean())
            mcse = float(sqrt(np.sum(variances) / (len(required_k) ** 2)))
            critical = float(norm.ppf(0.975))
            worst_index = int(np.argmin(deltas))
            minimum = float(deltas[worst_index])
            row.update(
                {
                    "equal_k_score": score,
                    "equal_k_mcse": mcse,
                    "equal_k_interval_low": score - critical * mcse,
                    "equal_k_interval_high": score + critical * mcse,
                    "minimum_chance_delta": minimum,
                    "worst_k": required_k[worst_index],
                    "maximin_value": minimum,
                }
            )
            complete_strategies.append(strategy)
            complete_vectors.append(deltas.tolist())
        rows.append(row)

    output = pd.DataFrame(rows)
    strategies_array = np.asarray(complete_strategies, dtype=np.int64)
    vectors = np.asarray(complete_vectors, dtype=float)
    if len(complete_strategies):
        pareto = _pareto_membership(vectors, strategies_array)
        output.loc[output["strategy"].isin(strategies_array[pareto]), "pareto_member"] = True
        minima = vectors.min(axis=1)
        best_minimum = float(minima.max())
        tied = strategies_array[np.isclose(minima, best_minimum, rtol=0.0, atol=1e-15)]
        leader = int(tied.min())
        output.loc[output["strategy"] == leader, "maximin_leader"] = True
    return output, strategies_array, vectors


def _batch_arrays(
    frames: dict[int, pd.DataFrame], strategies: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    strategy_list = strategies.tolist()
    for k, frame in frames.items():
        wins = (
            frame.pivot(index="deterministic_batch_id", columns="strategy", values="raw_wins")
            .reindex(columns=strategy_list, fill_value=0)
            .fillna(0)
        )
        exposures = (
            frame.pivot(
                index="deterministic_batch_id",
                columns="strategy",
                values="raw_player_game_exposures",
            )
            .reindex(index=wins.index, columns=strategy_list, fill_value=0)
            .fillna(0)
        )
        arrays[k] = (wins.to_numpy(dtype=float), exposures.to_numpy(dtype=float))
    return arrays


def _joint_batch_resampling(
    cfg: AppConfig,
    frames: dict[int, pd.DataFrame],
    across: pd.DataFrame,
    strategies: np.ndarray,
    required_k: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    replicates = cfg.screening.bootstrap_replicates
    delta = cfg.screening.delta_across_k
    if delta is None:
        raise ValueError("screening.delta_across_k is required for shortlist resampling")
    arrays = _batch_arrays(frames, strategies)
    strategy_count = len(strategies)
    top_n = min(cfg.screening.candidate_contribution_size, strategy_count)
    rank_sum = np.zeros(strategy_count, dtype=float)
    rank_square_sum = np.zeros(strategy_count, dtype=float)
    top_counts = np.zeros(strategy_count, dtype=np.int64)
    shortlist_counts = np.zeros(strategy_count, dtype=np.int64)
    controls = np.asarray(sorted(set(cfg.screening.controls)), dtype=np.int64)
    missing_controls = sorted(set(controls.tolist()).difference(strategies.tolist()))
    if missing_controls:
        raise ValueError(f"declared controls lack complete k support: {missing_controls}")
    control_indices = [int(np.flatnonzero(strategies == control)[0]) for control in controls]
    contrast_sum = np.zeros((len(controls), strategy_count), dtype=float)
    contrast_square_sum = np.zeros((len(controls), strategy_count), dtype=float)
    root_seed = int(across["root_seed"].iloc[0])

    for replicate in range(replicates):
        replicate_scores = np.zeros(strategy_count, dtype=float)
        for k in required_k:
            wins, exposures = arrays[k]
            batch_count = wins.shape[0]
            rng = coordinate_rng(
                RandomPurpose.BOOTSTRAP,
                root_seed=root_seed,
                k=k,
                replicate_index=replicate,
            )
            selected = rng.integers(0, batch_count, size=batch_count)
            total_wins = wins[selected].sum(axis=0)
            total_exposures = exposures[selected].sum(axis=0)
            if np.any(total_exposures <= 0):
                raise ValueError("joint batch resampling produced zero complete-support exposure")
            replicate_scores += total_wins / total_exposures - 1.0 / k
        replicate_scores /= len(required_k)
        order = np.lexsort((strategies, -replicate_scores))
        ranks = np.empty(strategy_count, dtype=np.int64)
        ranks[order] = np.arange(1, strategy_count + 1)
        rank_sum += ranks
        rank_square_sum += ranks * ranks
        top_counts[order[:top_n]] += 1
        shortlist_counts += replicate_scores >= replicate_scores.max() - delta
        for control_position, control_index in enumerate(control_indices):
            contrasts = replicate_scores - replicate_scores[control_index]
            contrast_sum[control_position] += contrasts
            contrast_square_sum[control_position] += contrasts * contrasts

    divisor = float(replicates)
    rank_mean = rank_sum / divisor
    rank_variance = np.maximum(rank_square_sum / divisor - rank_mean * rank_mean, 0.0)
    bootstrap = pd.DataFrame(
        {
            "root_seed": root_seed,
            "strategy": strategies,
            "bootstrap_replicates": replicates,
            "bootstrap_rank_mean": rank_mean,
            "bootstrap_rank_sd": np.sqrt(rank_variance),
            "top_n_size": top_n,
            "top_n_inclusion_probability": top_counts / divisor,
            "shortlist_delta": delta,
            "shortlist_inclusion_probability": shortlist_counts / divisor,
        }
    )

    complete_across = across.loc[across["complete_support"]]
    observed = dict(
        zip(
            complete_across["strategy"].astype(int).tolist(),
            complete_across["equal_k_score"].astype(float).tolist(),
            strict=True,
        )
    )
    contrast_rows: list[dict[str, int | float]] = []
    for control_position, control in enumerate(controls):
        means = contrast_sum[control_position] / divisor
        variances = np.maximum(
            contrast_square_sum[control_position] / divisor - means * means,
            0.0,
        )
        for index, strategy in enumerate(strategies):
            contrast_rows.append(
                {
                    "root_seed": root_seed,
                    "strategy": int(strategy),
                    "control_strategy": int(control),
                    "observed_equal_k_contrast": observed[int(strategy)]
                    - observed[int(control)],
                    "bootstrap_contrast_mean": float(means[index]),
                    "bootstrap_contrast_sd": float(sqrt(variances[index])),
                    "bootstrap_replicates": replicates,
                }
            )
    return bootstrap, pd.DataFrame(
        contrast_rows,
        columns=[
            "root_seed",
            "strategy",
            "control_strategy",
            "observed_equal_k_contrast",
            "bootstrap_contrast_mean",
            "bootstrap_contrast_sd",
            "bootstrap_replicates",
        ],
    )


def _declared_k_weights(cfg: AppConfig, required_k: list[int]) -> dict[int, float]:
    """Return the complete-support player-count weights declared by config."""

    if cfg.k_aggregation.method == "equal-k":
        weight = 1.0 / len(required_k)
        return dict.fromkeys(required_k, weight)
    weights = cfg.k_aggregation.k_weights
    if weights is None or {int(k) for k in weights} != set(required_k):
        raise ValueError("declared player-count weights must cover complete configured support")
    return {k: float(weights[k]) for k in required_k}


def _chance_relative_log_odds(win_rate: float, k: int) -> float | None:
    """Return finite log odds relative to chance, or ``None`` at a boundary."""

    if not 0.0 < win_rate < 1.0:
        return None
    chance = 1.0 / k
    return float(np.log(win_rate / (1.0 - win_rate)) - np.log(chance / (1.0 - chance)))


def _player_count_effect_diagnostics(
    cfg: AppConfig,
    estimates: dict[int, pd.DataFrame],
    required_k: list[int],
) -> pd.DataFrame:
    """Build complete-support relative effects, spreads, and cross-k rank checks."""

    complete_strategies = sorted(
        set.intersection(
            *(set(frame["strategy"].astype(int).tolist()) for frame in estimates.values())
        )
    )
    if not complete_strategies:
        raise ValueError("player-count diagnostics require complete configured strategy support")
    weights = _declared_k_weights(cfg, required_k)
    root_seed = int(next(iter(estimates.values()))["root_seed"].iloc[0])
    values: dict[tuple[int, int], float | None] = {}
    rows: list[dict[str, object]] = []

    def base_row(diagnostic_type: str) -> dict[str, object]:
        return {
            "diagnostic_type": diagnostic_type,
            "root_seed": root_seed,
            "strategy": None,
            "k": None,
            "k_a": None,
            "k_b": None,
            "k_weight": None,
            "k_weight_a": None,
            "k_weight_b": None,
            "win_rate": None,
            "chance_baseline": None,
            "chance_relative_log_odds": None,
            "effect_available": None,
            "unavailable_reason": None,
            "log_odds_contrast": None,
            "finite_strategy_count": None,
            "boundary_unavailable_count": None,
            "log_odds_sd": None,
            "log_odds_iqr": None,
            "log_odds_top_minus_median": None,
            "common_finite_strategy_count": None,
            "spearman_rank_correlation": None,
            "kendall_rank_correlation": None,
            "complete_configured_k_support": True,
            "declared_k_method": cfg.k_aggregation.method,
        }

    indexed = {k: frame.set_index("strategy") for k, frame in estimates.items()}
    for k in required_k:
        for strategy in complete_strategies:
            rate = float(indexed[k].loc[strategy, "win_rate"])
            effect = _chance_relative_log_odds(rate, k)
            values[(k, strategy)] = effect
            row = base_row("strategy_k_chance_relative_log_odds")
            row.update(
                {
                    "strategy": strategy,
                    "k": k,
                    "k_weight": weights[k],
                    "win_rate": rate,
                    "chance_baseline": 1.0 / k,
                    "chance_relative_log_odds": effect,
                    "effect_available": effect is not None,
                    "unavailable_reason": (
                        None if effect is not None else "boundary_win_rate_log_odds_unavailable"
                    ),
                }
            )
            rows.append(row)

        finite = np.asarray(
            [
                values[(k, strategy)]
                for strategy in complete_strategies
                if values[(k, strategy)] is not None
            ],
            dtype=float,
        )
        spread = base_row("within_k_strategy_spread")
        spread.update(
            {
                "k": k,
                "k_weight": weights[k],
                "finite_strategy_count": int(finite.size),
                "boundary_unavailable_count": len(complete_strategies) - int(finite.size),
                "log_odds_sd": float(np.std(finite, ddof=1)) if finite.size >= 2 else None,
                "log_odds_iqr": (
                    float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25))
                    if finite.size
                    else None
                ),
                "log_odds_top_minus_median": (
                    float(np.max(finite) - np.median(finite)) if finite.size else None
                ),
            }
        )
        rows.append(spread)

    for left_index, k_a in enumerate(required_k):
        for k_b in required_k[left_index + 1 :]:
            common = [
                strategy
                for strategy in complete_strategies
                if values[(k_a, strategy)] is not None and values[(k_b, strategy)] is not None
            ]
            for strategy in complete_strategies:
                left = values[(k_a, strategy)]
                right = values[(k_b, strategy)]
                row = base_row("strategy_pairwise_k_contrast")
                row.update(
                    {
                        "strategy": strategy,
                        "k_a": k_a,
                        "k_b": k_b,
                        "k_weight_a": weights[k_a],
                        "k_weight_b": weights[k_b],
                        "effect_available": left is not None and right is not None,
                        "unavailable_reason": (
                            None
                            if left is not None and right is not None
                            else "boundary_win_rate_log_odds_unavailable"
                        ),
                        "log_odds_contrast": (
                            float(left - right)
                            if left is not None and right is not None
                            else None
                        ),
                    }
                )
                rows.append(row)
            rank_row = base_row("pairwise_k_rank_agreement")
            rank_row.update(
                {
                    "k_a": k_a,
                    "k_b": k_b,
                    "k_weight_a": weights[k_a],
                    "k_weight_b": weights[k_b],
                    "common_finite_strategy_count": len(common),
                    "spearman_rank_correlation": (
                        float(
                            spearmanr(
                                [values[(k_a, strategy)] for strategy in common],
                                [values[(k_b, strategy)] for strategy in common],
                            ).statistic
                        )
                        if len(common) >= 2
                        else None
                    ),
                    "kendall_rank_correlation": (
                        float(
                            kendalltau(
                                [values[(k_a, strategy)] for strategy in common],
                                [values[(k_b, strategy)] for strategy in common],
                            ).statistic
                        )
                        if len(common) >= 2
                        else None
                    ),
                }
            )
            rows.append(rank_row)
    return pd.DataFrame(rows)


def _write_frame(
    cfg: AppConfig,
    frame: pd.DataFrame,
    path: Path,
    *,
    scope: ArtifactScope,
    operation: str,
    sources: list[Path],
    player_counts: list[int],
    grouping_keys: list[str],
    uncertainty_method: str,
    k_aggregation_method: str = "none",
) -> None:
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="performance",
        scope=scope,
        source_scope=ArtifactScope.BY_K,
        operation=operation,
        baseline="chance_1_over_k",
        weighted_quantity="chance_adjusted_win_rate",
        k_aggregation_method=k_aggregation_method,
        k_weights=(
            cfg.k_aggregation.k_weights
            if k_aggregation_method == "declared_mapping"
            else None
        ),
        support_count_role="raw_player_game_exposures",
        uncertainty_method=uncertainty_method,
        replication_unit="deterministic_shuffle_batch",
        conditioning="unconditional",
        consistency_columns=frame.columns.tolist(),
        source_artifacts=sources,
        grouping_keys=grouping_keys,
        player_counts=player_counts,
        required_player_counts=player_counts,
        missing_cell_policy="fail",
    )
    table = pa.Table.from_pandas(frame, preserve_index=False)
    write_parquet_artifact_atomic(table, path, sidecar=sidecar, codec=cfg.parquet_codec)


def build_canonical_performance(cfg: AppConfig, *, force: bool = False) -> PerformanceArtifacts:
    """Build per-k, complete-support equal-k, and joint-resampling estimates."""

    required_k = sorted({int(k) for k in cfg.sim.n_players_list})
    practical_by_k = cfg.screening.practical_delta_by_k
    if practical_by_k is None or {int(k) for k in practical_by_k} != set(required_k):
        raise ValueError("screening.practical_delta_by_k must cover complete configured k support")
    if cfg.screening.delta_across_k is None:
        raise ValueError("screening.delta_across_k must be explicitly configured")
    sources = [cfg.metrics_all_player_batch_path(k) for k in required_k]
    missing = [path for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing unconditional all-player batch metrics: {missing}")
    by_k_paths = tuple(cfg.performance_by_k_path(k) for k in required_k)
    artifacts = PerformanceArtifacts(
        by_k=by_k_paths,
        across_k=cfg.performance_across_k_path(),
        bootstrap=cfg.performance_bootstrap_path(),
        control_contrasts=cfg.performance_control_contrasts_path(),
        player_count_effects=cfg.performance_player_count_effects_path(),
    )
    done = stage_done_path(cfg.metrics_stage_dir, "canonical_performance")
    if not force and stage_is_up_to_date(
        done,
        inputs=sources,
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=list(artifacts.all_paths),
    ):
        return artifacts

    frames = {k: _read_batch_metrics(path, k) for k, path in zip(required_k, sources, strict=True)}
    roots = {int(frame["root_seed"].iloc[0]) for frame in frames.values()}
    if len(roots) != 1:
        raise ValueError(f"single-root performance inputs disagree on root: {sorted(roots)}")
    estimates = {
        k: _estimate_one_k(
            frame,
            k,
            cfg.screening.resolution_delta,
            float(practical_by_k[k]),
        )
        for k, frame in frames.items()
    }
    for k, path in zip(required_k, by_k_paths, strict=True):
        _write_frame(
            cfg,
            estimates[k],
            path,
            scope=ArtifactScope.BY_K,
            operation="aggregate_performance_by_strategy",
            sources=[cfg.metrics_all_player_batch_path(k)],
            player_counts=[k],
            grouping_keys=["root_seed", "k", "strategy"],
            uncertainty_method="wilson_and_batch_t_interval",
        )

    across, strategies, _ = _across_k_estimates(
        estimates,
        required_k,
        cfg.screening.delta_across_k,
    )
    if not len(strategies):
        raise ValueError("no strategies have complete configured k support")
    bootstrap, contrasts = _joint_batch_resampling(cfg, frames, across, strategies, required_k)
    player_count_effects = _player_count_effect_diagnostics(cfg, estimates, required_k)
    _write_frame(
        cfg,
        across,
        artifacts.across_k,
        scope=ArtifactScope.ACROSS_K,
        operation="equal_k_mean",
        sources=sources,
        player_counts=required_k,
        grouping_keys=["root_seed", "strategy"],
        uncertainty_method="independent_k_variance_sum",
        k_aggregation_method="equal_k",
    )
    _write_frame(
        cfg,
        bootstrap,
        artifacts.bootstrap,
        scope=ArtifactScope.ACROSS_K,
        operation="equal_k_mean",
        sources=sources,
        player_counts=required_k,
        grouping_keys=["root_seed", "strategy"],
        uncertainty_method="joint_deterministic_batch_resampling",
        k_aggregation_method="equal_k",
    )
    _write_frame(
        cfg,
        contrasts,
        artifacts.control_contrasts,
        scope=ArtifactScope.ACROSS_K,
        operation="equal_k_mean",
        sources=sources,
        player_counts=required_k,
        grouping_keys=["root_seed", "strategy", "control_strategy"],
        uncertainty_method="joint_deterministic_batch_resampling",
        k_aggregation_method="equal_k",
    )
    diagnostic_method = (
        "equal_k" if cfg.k_aggregation.method == "equal-k" else "declared_mapping"
    )
    _write_frame(
        cfg,
        player_count_effects,
        artifacts.player_count_effects,
        scope=ArtifactScope.DIAGNOSTICS,
        operation="calculate_player_count_effect_diagnostics",
        sources=list(by_k_paths),
        player_counts=required_k,
        grouping_keys=["diagnostic_type", "strategy", "k", "k_a", "k_b"],
        uncertainty_method="descriptive_complete_support_rank_and_spread",
        k_aggregation_method=diagnostic_method,
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


__all__ = ["PerformanceArtifacts", "build_canonical_performance"]
