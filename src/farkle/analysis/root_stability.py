"""Two-root count combination, reproducibility, convergence, and drift diagnostics.

The two roots are independent RNG domains for one fixed simulation design. They
are combined from raw wins and exposures within k. Root differences diagnose
reproducibility; they do not estimate a root superpopulation or random effect.
"""

from __future__ import annotations

import hashlib
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import kendalltau, norm, spearmanr, t

from farkle.analysis.all_player_metrics import (
    ATTEMPT_CONDITIONING,
    validate_unconditional_all_player_schema,
)
from farkle.analysis.batch_support import (
    RECTANGULAR_SUPPORT_POLICY,
)
from farkle.analysis.performance import (
    _BATCH_MATRIX_DTYPE,
    _validate_matrix_array,
    _write_batch_matrix,
)
from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import (
    ArtifactSidecar,
    make_artifact_sidecar,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
    write_artifact_with_sidecar_atomic,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.parallel import (
    ProcessTreeMemoryGuard,
    apply_native_thread_limits,
    resolve_stage_parallel_policy,
)
from farkle.utils.partitioned_stage import (
    PartitionedStageIdentity,
    PartitionedUnit,
    resolved_code_identity_sha256,
    run_partitioned_stage,
    validate_final_manifest,
)
from farkle.utils.random import RandomPurpose, coordinate_rng
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.strategy_ids import canonical_strategy_ids, require_strategy_id_field

_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "root_seed",
    "k",
    "deterministic_batch_id",
    "strategy",
    "raw_wins",
    "raw_player_game_exposures",
    "raw_completed_player_game_exposures",
    "raw_safety_limit_player_game_exposures",
    "raw_losses",
)
ROOT_STABILITY_METHOD_VERSION: Final = 3
_ROOT_BOOTSTRAP_RANGE_SIZE: Final = 50


@dataclass(frozen=True)
class RootBatchCell:
    """One canonical root/k batch-metric input."""

    root_seed: int
    k: int
    path: Path
    matrix_path: Path | None = None


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


def _validate_cell_source(cell: RootBatchCell) -> None:
    """Validate source metadata without materializing its whole Parquet table."""

    validate_artifact_sidecar(
        cell.path,
        expected={
            "scope": ArtifactScope.BY_K.value,
            "conditioning": ATTEMPT_CONDITIONING,
        },
    )
    schema = pq.read_schema(cell.path)
    validate_unconditional_all_player_schema(schema)
    require_strategy_id_field(schema, "strategy", context=str(cell.path))
    missing = sorted(set(_INPUT_COLUMNS).difference(schema.names))
    if missing:
        raise ValueError(f"{cell.path} lacks two-root inputs: {missing}")


def _matrix_sidecar(cfg: AppConfig, source: Path, destination: Path, k: int) -> ArtifactSidecar:
    """Describe the fallback with the same immutable layout as performance."""

    return make_artifact_sidecar(
        cfg,
        destination,
        producer="root_stability",
        scope=ArtifactScope.CROSS_SEED,
        source_scope=ArtifactScope.BY_K,
        operation="materialize_root_stability_batch_matrix",
        method_contract={
            "kind": "root_combination",
            "procedure": "materialize_root_stability_batch_matrix",
            "parameters": {
                "method_version": ROOT_STABILITY_METHOD_VERSION,
                "numpy_dtype": _BATCH_MATRIX_DTYPE.descr,
                "layout": "deterministic_batch_by_strategy",
            },
        },
        source_artifacts=[source],
        consistency_columns=list(_BATCH_MATRIX_DTYPE.names or ()),
        grouping_keys=["root_seed", "k", "deterministic_batch_id", "strategy"],
        player_counts=[k],
        required_player_counts=[k],
        missing_cell_policy="fail",
        replication_unit="deterministic_shuffle_batch",
        conditioning=ATTEMPT_CONDITIONING,
        seed_scope="root_pair_stability",
    )


def _matrix_cell(
    cfg: AppConfig,
    cell: RootBatchCell,
    *,
    force: bool,
    guard: ProcessTreeMemoryGuard,
) -> RootBatchCell:
    """Use the published performance matrix, or atomically create one fallback.

    The fallback is deliberately the byte-compatible performance matrix format;
    it is an immutable cache, not a second dataframe representation.
    """

    _validate_cell_source(cell)
    shared = cell.path.with_name("performance_batch_matrix.npy")
    candidate = (
        shared if shared.exists() else cfg.root_stability_matrix_path(cell.root_seed, cell.k)
    )
    if shared.exists():
        validate_artifact_sidecar(
            shared, expected={"operation": "materialize_performance_batch_matrix"}
        )
    else:
        done = stage_done_path(candidate.parent, "root_stability_batch_matrix")
        if force or not stage_is_up_to_date(
            done,
            inputs=[cell.path],
            outputs=[candidate],
            cfg=cfg,
            stage="root_stability",
            sidecar_artifacts=[candidate],
        ):
            sidecar = _matrix_sidecar(cfg, cell.path, candidate, cell.k)
            write_artifact_with_sidecar_atomic(
                candidate,
                sidecar,
                lambda staged: _write_batch_matrix(cfg, cell.path, staged, k=cell.k, guard=guard),
            )
            write_stage_done(
                done,
                inputs=[cell.path],
                outputs=[candidate],
                cfg=cfg,
                stage="root_stability",
                sidecar_artifacts=[candidate],
            )
    matrix = np.load(candidate, mmap_mode="r", allow_pickle=False)
    _validate_matrix_array(matrix, path=candidate, k=cell.k)
    roots = np.unique(matrix["root_seed"])
    if roots.tolist() != [cell.root_seed]:
        raise ValueError(
            f"{candidate} has root support {roots.tolist()}, expected [{cell.root_seed}]"
        )
    del matrix
    return RootBatchCell(cell.root_seed, cell.k, cell.path, candidate)


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


def _practical_threshold_position(effect: float, practical_delta: float) -> str:
    """Describe an estimate's position relative to the declared practical thresholds."""

    if effect >= practical_delta:
        return "above_positive_threshold"
    if effect <= -practical_delta:
        return "below_negative_threshold"
    return "between_thresholds"


def _root_stability_freshness_key(cfg: AppConfig) -> dict[str, object]:
    """Bind fixed-design diagnostic semantics to the stage lifecycle."""

    return {
        **cfg.freshness_key(),
        "root_stability_method_version": ROOT_STABILITY_METHOD_VERSION,
    }


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
        positive = group.loc[group["raw_player_game_exposures"] > 0]
        batch_wins = positive["raw_wins"].to_numpy(dtype=float)
        batch_exposures = positive["raw_player_game_exposures"].to_numpy(dtype=float)
        wins = int(batch_wins.sum())
        exposures = int(batch_exposures.sum())
        completed_exposures = int(group["raw_completed_player_game_exposures"].sum())
        safety_exposures = int(group["raw_safety_limit_player_game_exposures"].sum())
        losses = int(group["raw_losses"].sum())
        if exposures <= 0:
            raise ValueError(
                f"strategy {strategy} has no positive exposure cells after explicit "
                "zero-exposure exclusion"
            )
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
                "raw_attempted_exposures": exposures,
                "raw_completed_exposures": completed_exposures,
                "raw_safety_limit_exposures": safety_exposures,
                "raw_losses": losses,
                "raw_declared_batches": int(
                    group[["root_seed", "deterministic_batch_id"]].drop_duplicates().shape[0]
                ),
                "raw_batches": int(
                    positive[["root_seed", "deterministic_batch_id"]].drop_duplicates().shape[0]
                ),
                "excluded_zero_exposure_batch_cells": int(
                    group["raw_player_game_exposures"].eq(0).sum()
                ),
                "batch_support_policy": RECTANGULAR_SUPPORT_POLICY,
                "win_rate": rate,
                "win_rate_per_attempt": rate,
                "win_rate_given_completion": (
                    wins / completed_exposures if completed_exposures else None
                ),
                "safety_limit_exposure_rate": safety_exposures / exposures,
                "chance_delta": effect,
                "batch_mcse": mcse,
                "batch_mc_precision_interval_low": interval_low,
                "batch_mc_precision_interval_high": interval_high,
                "practical_delta": practical_delta,
                "practical_threshold_position": _practical_threshold_position(
                    effect, practical_delta
                ),
            }
        )
    return pd.DataFrame(rows)


def _matrix_batch_count(cell: RootBatchCell) -> int:
    if cell.matrix_path is None:
        raise ValueError("root stability requires a prepared batch matrix")
    matrix = np.load(cell.matrix_path, mmap_mode="r", allow_pickle=False)
    try:
        return int(matrix.shape[0])
    finally:
        del matrix


def _estimate_matrix_cells(
    cfg: AppConfig,
    cells: Sequence[RootBatchCell],
    *,
    k: int,
    estimate_scope: str,
    root_seed: int | None,
    practical_delta: float,
    row_bounds: tuple[int | None, int | None] = (None, None),
) -> pd.DataFrame:
    """Estimate one scope from matrix slices without rebuilding raw pandas rows."""

    if not cells:
        raise ValueError("root stability estimate requires at least one matrix cell")
    paths = [cast(Path, cell.matrix_path) for cell in cells]
    first = np.load(paths[0], mmap_mode="r", allow_pickle=False)
    try:
        strategies = np.asarray(first["strategy"][0], dtype=np.int32).copy()
    finally:
        del first
    strategy_count = len(strategies)
    totals = {
        name: np.zeros(strategy_count, dtype=np.int64)
        for name in (
            "raw_wins",
            "raw_player_game_exposures",
            "raw_completed_player_game_exposures",
            "raw_safety_limit_player_game_exposures",
            "raw_losses",
        )
    }
    positive_batches = np.zeros(strategy_count, dtype=np.int64)
    sum_w2 = np.zeros(strategy_count, dtype=np.float64)
    sum_we = np.zeros(strategy_count, dtype=np.float64)
    sum_e2 = np.zeros(strategy_count, dtype=np.float64)
    declared_batches = 0
    max_bytes = int(
        cfg.resources.stage_batch_bytes.get(
            "performance", cfg.resources.stage_batch_bytes["analysis"]
        )
    )
    for path in paths:
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if not np.array_equal(matrix["strategy"][0], strategies):
                raise ValueError("root stability strategy support differs across matrices")
            start = 0 if row_bounds[0] is None else int(row_bounds[0])
            stop = (
                matrix.shape[0]
                if row_bounds[1] is None
                else min(matrix.shape[0], int(row_bounds[1]))
            )
            if start < 0 or stop <= start:
                raise ValueError("root stability matrix slice is empty or invalid")
            declared_batches += stop - start
            bytes_per_cell = 7 * np.dtype("<i8").itemsize + 1
            chunk = max(
                1,
                min(
                    strategy_count,
                    max_bytes // max(1, (stop - start) * bytes_per_cell),
                ),
            )
            for column_start in range(0, strategy_count, chunk):
                column_stop = min(strategy_count, column_start + chunk)
                for name, output in totals.items():
                    output[column_start:column_stop] += np.asarray(
                        matrix[name][start:stop, column_start:column_stop]
                    ).sum(axis=0, dtype=np.int64)
                batch_wins = np.asarray(
                    matrix["raw_wins"][start:stop, column_start:column_stop],
                    dtype=np.float64,
                )
                batch_exposures = np.asarray(
                    matrix["raw_player_game_exposures"][start:stop, column_start:column_stop],
                    dtype=np.float64,
                )
                positive = batch_exposures > 0
                positive_batches[column_start:column_stop] += positive.sum(axis=0, dtype=np.int64)
                batch_wins = np.where(positive, batch_wins, 0.0)
                batch_exposures = np.where(positive, batch_exposures, 0.0)
                sum_w2[column_start:column_stop] += np.sum(batch_wins * batch_wins, axis=0)
                sum_we[column_start:column_stop] += np.sum(batch_wins * batch_exposures, axis=0)
                sum_e2[column_start:column_stop] += np.sum(
                    batch_exposures * batch_exposures, axis=0
                )
        finally:
            del matrix

    wins = totals["raw_wins"]
    exposures = totals["raw_player_game_exposures"]
    completed = totals["raw_completed_player_game_exposures"]
    safety = totals["raw_safety_limit_player_game_exposures"]
    losses = totals["raw_losses"]
    if np.any(exposures <= 0):
        raise ValueError("root stability scope contains a strategy without positive exposure")
    rates = wins / exposures
    mcse = np.full(strategy_count, np.nan, dtype=np.float64)
    eligible = positive_batches >= 2
    if np.any(eligible):
        residual_squares = (
            sum_w2[eligible]
            - 2.0 * rates[eligible] * sum_we[eligible]
            + rates[eligible] ** 2 * sum_e2[eligible]
        )
        variance = positive_batches[eligible] / (positive_batches[eligible] - 1.0)
        variance *= np.maximum(residual_squares, 0.0)
        mcse[eligible] = np.sqrt(variance) / exposures[eligible]
    interval_low = np.full(strategy_count, np.nan, dtype=np.float64)
    interval_high = np.full(strategy_count, np.nan, dtype=np.float64)
    if np.any(eligible):
        critical = t.ppf(0.975, positive_batches[eligible] - 1)
        interval_low[eligible] = np.maximum(0.0, rates[eligible] - critical * mcse[eligible])
        interval_high[eligible] = np.minimum(1.0, rates[eligible] + critical * mcse[eligible])
    effects = rates - 1.0 / k
    return pd.DataFrame(
        {
            "estimate_scope": estimate_scope,
            "root_seed": root_seed,
            "k": k,
            "strategy": strategies,
            "chance_baseline": 1.0 / k,
            "raw_wins": wins,
            "raw_exposures": exposures,
            "raw_attempted_exposures": exposures,
            "raw_completed_exposures": completed,
            "raw_safety_limit_exposures": safety,
            "raw_losses": losses,
            "raw_declared_batches": declared_batches,
            "raw_batches": positive_batches,
            "excluded_zero_exposure_batch_cells": declared_batches - positive_batches,
            "batch_support_policy": RECTANGULAR_SUPPORT_POLICY,
            "win_rate": rates,
            "win_rate_per_attempt": rates,
            "win_rate_given_completion": np.divide(
                wins,
                completed,
                out=np.full(strategy_count, np.nan, dtype=np.float64),
                where=completed > 0,
            ),
            "safety_limit_exposure_rate": safety / exposures,
            "chance_delta": effects,
            "batch_mcse": mcse,
            "batch_mc_precision_interval_low": interval_low,
            "batch_mc_precision_interval_high": interval_high,
            "practical_delta": practical_delta,
            "practical_threshold_position": [
                _practical_threshold_position(float(effect), practical_delta) for effect in effects
            ],
        }
    )


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
    count_columns = (
        "raw_wins",
        "raw_attempted_exposures",
        "raw_completed_exposures",
        "raw_safety_limit_exposures",
        "raw_losses",
        "raw_declared_batches",
        "raw_batches",
        "excluded_zero_exposure_batch_cells",
    )
    count_maps = {
        column: {
            k: frame.set_index("strategy")[column].astype(int).to_dict()
            for k, frame in estimates.items()
        }
        for column in count_columns
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
                **{
                    column: sum(count_maps[column][k][strategy] for k in required_k)
                    for column in count_columns
                },
                "k_aggregation_method": cfg_method_name(weights, required_k),
                "across_k_score": score,
                "across_k_mcse": mcse,
                "across_k_mc_precision_interval_low": score - critical * mcse,
                "across_k_mc_precision_interval_high": score + critical * mcse,
                "minimum_chance_delta": float(values[worst_position]),
                "worst_k": required_k[worst_position],
                "practical_delta": practical_delta,
                "practical_threshold_position": _practical_threshold_position(
                    score, practical_delta
                ),
            }
        )
        rows[-1]["safety_limit_exposure_rate"] = int(
            cast(int, rows[-1]["raw_safety_limit_exposures"])
        ) / int(cast(int, rows[-1]["raw_attempted_exposures"]))
    return pd.DataFrame(rows)


def cfg_method_name(weights: dict[int, float], required_k: list[int]) -> str:
    """Name equal or declared weighting from exact normalized values."""

    equal = 1.0 / len(required_k)
    if all(abs(weights[k] - equal) <= 1e-15 for k in required_k):
        return "equal_k_mean"
    return "declared_k_weighted_mean"


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


def _root_bootstrap_units(
    replicates: int, range_size: int = _ROOT_BOOTSTRAP_RANGE_SIZE
) -> tuple[PartitionedUnit, ...]:
    return tuple(
        PartitionedUnit(
            (start, min(replicates, start + range_size)),
            f"replicates_{start:08d}_{min(replicates, start + range_size):08d}.npy",
        )
        for start in range(0, replicates, range_size)
    )


def _root_bootstrap_identity(
    cfg: AppConfig, cells: dict[tuple[int, int], RootBatchCell], roots: tuple[int, int], family: str
) -> PartitionedStageIdentity:
    inputs = tuple(
        sorted(
            (
                f"root_{root}_k_{k}",
                hashlib.sha256(
                    (
                        sha256_file(cast(Path, cells[(root, k)].matrix_path))
                        + sha256_file(sidecar_path(cast(Path, cells[(root, k)].matrix_path)))
                    ).encode("ascii")
                ).hexdigest(),
            )
            for root in roots
            for k in sorted({int(value) for value in cfg.sim.n_players_list})
        )
    )
    return PartitionedStageIdentity(
        stage_name=f"root_stability_{family}",
        root_seed=roots[0],
        input_identities=inputs,
        statistical_config_sha256=cfg.stage_config_sha("root_stability"),
        code_identity_sha256=resolved_code_identity_sha256(cfg),
        schema_version=1,
        method_version=ROOT_STABILITY_METHOD_VERSION,
    )


def _bootstrap_ranges_validate(
    cfg: AppConfig,
    cells: dict[tuple[int, int], RootBatchCell],
    roots: tuple[int, int],
) -> bool:
    """A root-stage completion is valid only when both range families validate."""

    units = _root_bootstrap_units(int(cfg.screening.bootstrap_replicates))
    unit_source = lambda: iter(units)  # noqa: E731 - stable reusable source
    return all(
        validate_final_manifest(
            directory / "partition_manifest.jsonl",
            root=directory,
            identity=_root_bootstrap_identity(cfg, cells, roots, family),
            unit_source=unit_source,
        )
        is not None
        for directory, family in (
            (cfg.root_stability_top_n_ranges_dir(), "top_n"),
            (cfg.root_stability_joint_ranges_dir(), "joint_discrepancy"),
        )
    )


@dataclass(frozen=True)
class _RootTopNRangeWriter:
    matrix_paths: tuple[Path, ...]
    roots: tuple[int, int]
    required_k: tuple[int, ...]
    strategies: tuple[int, ...]
    weights: tuple[float, ...]
    top_n: int
    max_batch_bytes: int

    def __call__(self, unit: PartitionedUnit, path: Path) -> None:
        start, stop = (int(value) for value in unit.key)
        strategy_ids = np.asarray(self.strategies, dtype=np.int64)
        matrices: dict[tuple[int, int], np.ndarray] = {}
        eligible_batches: dict[tuple[int, int], np.ndarray] = {}
        try:
            for index, (root, k) in enumerate((r, p) for r in self.roots for p in self.required_k):
                matrix = np.load(self.matrix_paths[index], mmap_mode="r", allow_pickle=False)
                _validate_matrix_array(matrix, path=self.matrix_paths[index], k=k)
                if not np.array_equal(matrix["strategy"][0], strategy_ids):
                    raise ValueError(
                        "root bootstrap strategy support differs across root/k matrices"
                    )
                matrices[(root, k)] = matrix
                eligible = np.asarray(
                    [
                        batch
                        for batch in range(matrix.shape[0])
                        if np.all(matrix["raw_player_game_exposures"][batch] > 0)
                    ],
                    dtype=np.intp,
                )
                if not eligible.size:
                    raise ValueError("root bootstrap has no positive-exposure batch vectors")
                eligible_batches[(root, k)] = eligible
            output = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.uint8,
                shape=(stop - start, len(self.roots), len(strategy_ids)),
            )
            output.fill(0)
            for row, replicate in enumerate(range(start, stop)):
                for root_index, root in enumerate(self.roots):
                    scores = np.zeros(len(strategy_ids), dtype=np.float64)
                    for k, weight in zip(self.required_k, self.weights, strict=True):
                        matrix = matrices[(root, k)]
                        eligible = eligible_batches[(root, k)]
                        rng = coordinate_rng(
                            RandomPurpose.ROOT_STABILITY_BOOTSTRAP,
                            root_seed=root,
                            k=k,
                            replicate_index=replicate,
                        )
                        selected = rng.integers(0, len(eligible), size=len(eligible))
                        counts = np.bincount(selected, minlength=len(eligible)).astype(
                            np.int64, copy=False
                        )
                        strategy_chunk = max(
                            1,
                            min(
                                len(strategy_ids),
                                self.max_batch_bytes
                                // max(1, 3 * len(eligible) * np.dtype("<i8").itemsize),
                            ),
                        )
                        for column_start in range(0, len(strategy_ids), strategy_chunk):
                            column_stop = min(len(strategy_ids), column_start + strategy_chunk)
                            wins = counts @ np.asarray(
                                matrix["raw_wins"][eligible, column_start:column_stop],
                                dtype=np.int64,
                            )
                            exposures = counts @ np.asarray(
                                matrix["raw_player_game_exposures"][
                                    eligible, column_start:column_stop
                                ],
                                dtype=np.int64,
                            )
                            if np.any(exposures <= 0):
                                raise ValueError(
                                    "root bootstrap produced zero complete-support exposure"
                                )
                            scores[column_start:column_stop] += weight * (
                                wins / exposures - 1.0 / k
                            )
                    order = np.lexsort((strategy_ids, -scores))
                    output[row, root_index, order[: self.top_n]] = 1
            output.flush()
            del output
        finally:
            matrices.clear()


def _root_bootstrap_top_n_inclusion(
    cfg: AppConfig,
    cells: dict[tuple[int, int], RootBatchCell],
    roots: tuple[int, int],
    required_k: list[int],
    *,
    force: bool,
    guard: ProcessTreeMemoryGuard,
) -> pd.DataFrame:
    """Reduce authenticated semantic-coordinate bootstrap ranges in stable order."""

    strategies = np.load(
        cast(Path, cells[(roots[0], required_k[0])].matrix_path), mmap_mode="r", allow_pickle=False
    )["strategy"][0].astype(np.int64, copy=True)
    weights = _k_weights(cfg, required_k)
    replicates = int(cfg.screening.bootstrap_replicates)
    units = _root_bootstrap_units(replicates)
    writer = _RootTopNRangeWriter(
        tuple(cast(Path, cells[(root, k)].matrix_path) for root in roots for k in required_k),
        roots,
        tuple(required_k),
        tuple(int(value) for value in strategies),
        tuple(weights[k] for k in required_k),
        min(cfg.screening.candidate_contribution_size, len(strategies)),
        int(
            cfg.resources.stage_batch_bytes.get(
                "performance", cfg.resources.stage_batch_bytes["analysis"]
            )
        ),
    )
    result = run_partitioned_stage(
        root=cfg.root_stability_top_n_ranges_dir(),
        identity=_root_bootstrap_identity(cfg, cells, roots, "top_n"),
        unit_source=lambda: iter(units),
        writer=writer,
        resources=cfg.resources,
        requested_workers=cfg.analysis.n_jobs,
        mp_start_method=cfg.analysis.mp_start_method,
        force=force,
        memory_guard=guard,
    )
    if result.required_units != len(units):
        raise RuntimeError("root top-N manifest does not cover every configured replicate")
    counts = np.zeros((len(roots), len(strategies)), dtype=np.int64)
    expected = 0
    for unit in units:
        start, stop = (int(value) for value in unit.key)
        if start != expected:
            raise RuntimeError("root top-N ranges are not contiguous")
        shard = np.load(
            cfg.root_stability_top_n_ranges_dir() / "units" / unit.relative_output,
            mmap_mode="r",
            allow_pickle=False,
        )
        if shard.dtype != np.dtype("u1") or shard.shape != (
            stop - start,
            len(roots),
            len(strategies),
        ):
            raise ValueError("invalid root top-N bootstrap range")
        counts += shard.sum(axis=0, dtype=np.int64)
        expected = stop
        del shard
    if expected != replicates:
        raise RuntimeError("root top-N reduction ended before every replicate")
    top_n = min(cfg.screening.candidate_contribution_size, len(strategies))
    return pd.DataFrame(
        [
            {
                "root_seed": root,
                "strategy": int(strategy),
                "required_k_count": len(required_k),
                "complete_support": True,
                "k_aggregation_method": cfg_method_name(weights, required_k),
                "bootstrap_replicates": replicates,
                "top_n_size": top_n,
                "bootstrap_top_n_inclusion_frequency": float(
                    counts[root_index, strategy_index] / replicates
                ),
            }
            for root_index, root in enumerate(roots)
            for strategy_index, strategy in enumerate(strategies)
        ]
    )


def _scope_estimates(
    cfg: AppConfig,
    cells: dict[tuple[int, int], RootBatchCell],
    roots: tuple[int, int],
    required_k: list[int],
    *,
    maximum_batches: int | None = None,
) -> tuple[dict[int, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Build estimates directly from prepared matrix slices."""

    practical_by_k = cfg.screening.practical_delta_by_k
    if practical_by_k is None:
        raise ValueError("screening.practical_delta_by_k is required")
    if cfg.screening.delta_across_k is None:
        raise ValueError("screening.delta_across_k is required")
    weights = _k_weights(cfg, required_k)
    per_scope_k: dict[str, dict[int, pd.DataFrame]] = {f"root_{root}": {} for root in roots}
    per_scope_k["combined_roots"] = {}
    for k in required_k:
        row_bounds = (None, maximum_batches)
        for root in roots:
            scope = f"root_{root}"
            per_scope_k[scope][k] = _estimate_matrix_cells(
                cfg,
                [cells[(root, k)]],
                k=k,
                estimate_scope=scope,
                root_seed=root,
                practical_delta=float(practical_by_k[k]),
                row_bounds=row_bounds,
            )
        per_scope_k["combined_roots"][k] = _estimate_matrix_cells(
            cfg,
            [cells[(root, k)] for root in roots],
            k=k,
            estimate_scope="combined_roots",
            root_seed=None,
            practical_delta=float(practical_by_k[k]),
            row_bounds=row_bounds,
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


def _safe_standardized(
    raw_difference: float,
    expected_mcse: float | None,
) -> float | None:
    """Return a stable standardized discrepancy for zero-noise edge cases."""

    if expected_mcse is None or not np.isfinite(expected_mcse):
        return None
    if expected_mcse > 0.0:
        return raw_difference / expected_mcse
    if raw_difference == 0.0:
        return 0.0
    return float(np.copysign(np.inf, raw_difference))


def _at_float(frame: pd.DataFrame, strategy: Hashable, column: str) -> float:
    """Read one known numeric scalar from a strategy-indexed frame."""

    return float(cast(float, frame.at[strategy, column]))


def _combined_optional_mcse(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    strategy: Hashable,
    column: str,
) -> float | None:
    """Combine independent MCSEs when both half-sample estimates are available."""

    value_a = frame_a.at[strategy, column]
    value_b = frame_b.at[strategy, column]
    if pd.isna(value_a) or pd.isna(value_b):
        return None
    return sqrt(float(cast(float, value_a)) ** 2 + float(cast(float, value_b)) ** 2)


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
                    "root_a_practical_threshold_position": _at_str(
                        a, strategy, "practical_threshold_position"
                    ),
                    "root_b_practical_threshold_position": _at_str(
                        b, strategy, "practical_threshold_position"
                    ),
                    "combined_practical_threshold_position": _at_str(
                        combined, strategy, "practical_threshold_position"
                    ),
                    "practical_threshold_position_changed": (
                        _at_str(a, strategy, "practical_threshold_position")
                        != _at_str(b, strategy, "practical_threshold_position")
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
                "root_a_practical_threshold_position": _at_str(
                    a, strategy, "practical_threshold_position"
                ),
                "root_b_practical_threshold_position": _at_str(
                    b, strategy, "practical_threshold_position"
                ),
                "combined_practical_threshold_position": _at_str(
                    combined, strategy, "practical_threshold_position"
                ),
                "practical_threshold_position_changed": (
                    _at_str(a, strategy, "practical_threshold_position")
                    != _at_str(b, strategy, "practical_threshold_position")
                ),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class _JointDiscrepancyRangeWriter:
    """Write one independently authenticated range of joint bootstrap maxima."""

    matrix_paths: tuple[Path, ...]
    roots: tuple[int, int]
    required_k: tuple[int, ...]
    strategies: tuple[int, ...]
    weights: tuple[float, ...]
    observed_by_k: tuple[tuple[int, tuple[float, ...], tuple[float, ...]], ...]
    observed_across: tuple[float, ...]
    expected_across: tuple[float, ...]
    max_batch_bytes: int

    def __call__(self, unit: PartitionedUnit, path: Path) -> None:
        start, stop = (int(value) for value in unit.key)
        strategy_ids = np.asarray(self.strategies, dtype=np.int64)
        observed = {k: np.asarray(values) for k, values, _expected in self.observed_by_k}
        expected = {k: np.asarray(values) for k, _values, values in self.observed_by_k}
        matrices: dict[tuple[int, int], np.ndarray] = {}
        eligible_batches: dict[tuple[int, int], np.ndarray] = {}
        try:
            for index, (root, k) in enumerate((r, p) for r in self.roots for p in self.required_k):
                matrix = np.load(self.matrix_paths[index], mmap_mode="r", allow_pickle=False)
                _validate_matrix_array(matrix, path=self.matrix_paths[index], k=k)
                if not np.array_equal(matrix["strategy"][0], strategy_ids):
                    raise ValueError(
                        "joint bootstrap strategy support differs across root/k matrices"
                    )
                matrices[(root, k)] = matrix
                eligible = np.asarray(
                    [
                        batch
                        for batch in range(matrix.shape[0])
                        if np.all(matrix["raw_player_game_exposures"][batch] > 0)
                    ],
                    dtype=np.intp,
                )
                if not eligible.size:
                    raise ValueError("joint bootstrap has no positive-exposure batch vectors")
                eligible_batches[(root, k)] = eligible
            maxima = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.dtype("<f8"), shape=(stop - start,)
            )
            for output_row, replicate in enumerate(range(start, stop)):
                rates: dict[tuple[int, int], np.ndarray] = {}
                for root in self.roots:
                    for k in self.required_k:
                        matrix = matrices[(root, k)]
                        eligible = eligible_batches[(root, k)]
                        rng = coordinate_rng(
                            RandomPurpose.ROOT_STABILITY_BOOTSTRAP,
                            root_seed=root,
                            k=k,
                            replicate_index=replicate,
                        )
                        selected = rng.integers(0, len(eligible), size=len(eligible))
                        counts = np.bincount(selected, minlength=len(eligible)).astype(
                            np.int64, copy=False
                        )
                        rate = np.empty(len(strategy_ids), dtype=np.float64)
                        strategy_chunk = max(
                            1,
                            min(
                                len(strategy_ids),
                                self.max_batch_bytes
                                // max(1, 3 * len(eligible) * np.dtype("<i8").itemsize),
                            ),
                        )
                        for column_start in range(0, len(strategy_ids), strategy_chunk):
                            column_stop = min(len(strategy_ids), column_start + strategy_chunk)
                            wins = counts @ np.asarray(
                                matrix["raw_wins"][eligible, column_start:column_stop],
                                dtype=np.int64,
                            )
                            exposures = counts @ np.asarray(
                                matrix["raw_player_game_exposures"][
                                    eligible, column_start:column_stop
                                ],
                                dtype=np.int64,
                            )
                            if np.any(exposures <= 0):
                                raise ValueError(
                                    "joint root bootstrap produced zero strategy exposure"
                                )
                            rate[column_start:column_stop] = wins / exposures - 1.0 / k
                        rates[(root, k)] = rate
                standardized: list[np.ndarray] = []
                for k in self.required_k:
                    valid = expected[k] > 0.0
                    centered = rates[(self.roots[0], k)] - rates[(self.roots[1], k)] - observed[k]
                    standardized.append(np.abs(centered[valid] / expected[k][valid]))
                across = sum(
                    weight * (rates[(self.roots[0], k)] - rates[(self.roots[1], k)])
                    for k, weight in zip(self.required_k, self.weights, strict=True)
                )
                expected_across = np.asarray(self.expected_across)
                valid_across = expected_across > 0.0
                standardized.append(
                    np.abs(
                        (across - np.asarray(self.observed_across))[valid_across]
                        / expected_across[valid_across]
                    )
                )
                maxima[output_row] = max(
                    (float(part.max()) for part in standardized if part.size), default=0.0
                )
            maxima.flush()
            del maxima
        finally:
            matrices.clear()


def _joint_discrepancy_bootstrap(
    cfg: AppConfig,
    cells: dict[tuple[int, int], RootBatchCell],
    roots: tuple[int, int],
    required_k: list[int],
    discrepancies: pd.DataFrame,
    *,
    force: bool,
    guard: ProcessTreeMemoryGuard,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate dependent discrepancy flags from joint batch-vector resampling."""

    replicates = cfg.screening.bootstrap_replicates
    reference_upper_tail_fraction = cfg.robustness.joint_discrepancy_alpha
    weights = _k_weights(cfg, required_k)
    strategies = np.sort(discrepancies["strategy"].astype(int).unique())
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
    units = _root_bootstrap_units(replicates)
    writer = _JointDiscrepancyRangeWriter(
        tuple(cast(Path, cells[(root, k)].matrix_path) for root in roots for k in required_k),
        roots,
        tuple(required_k),
        tuple(int(value) for value in strategies),
        tuple(weights[k] for k in required_k),
        tuple((k, tuple(observed_by_k[k]), tuple(expected_by_k[k])) for k in required_k),
        tuple(observed_across),
        tuple(expected_across),
        int(
            cfg.resources.stage_batch_bytes.get(
                "performance", cfg.resources.stage_batch_bytes["analysis"]
            )
        ),
    )
    result = run_partitioned_stage(
        root=cfg.root_stability_joint_ranges_dir(),
        identity=_root_bootstrap_identity(cfg, cells, roots, "joint_discrepancy"),
        unit_source=lambda: iter(units),
        writer=writer,
        resources=cfg.resources,
        requested_workers=cfg.analysis.n_jobs,
        mp_start_method=cfg.analysis.mp_start_method,
        force=force,
        memory_guard=guard,
    )
    if result.required_units != len(units):
        raise RuntimeError("joint discrepancy manifest does not cover every configured replicate")
    maxima = np.empty(replicates, dtype=np.float64)
    expected = 0
    for unit in units:
        start, stop = (int(value) for value in unit.key)
        if start != expected:
            raise RuntimeError("joint discrepancy ranges are not contiguous")
        shard = np.load(
            cfg.root_stability_joint_ranges_dir() / "units" / unit.relative_output,
            mmap_mode="r",
            allow_pickle=False,
        )
        if shard.dtype != np.dtype("<f8") or shard.shape != (stop - start,):
            raise ValueError("invalid joint discrepancy bootstrap range")
        maxima[start:stop] = shard
        expected = stop
        del shard
    if expected != replicates:
        raise RuntimeError("joint discrepancy reduction ended before every replicate")

    reference_quantile = float(
        np.quantile(maxima, 1.0 - reference_upper_tail_fraction, method="higher")
    )
    enriched = discrepancies.copy()
    enriched["joint_max_abs_standardized_reference_quantile"] = reference_quantile
    enriched["exceeds_joint_reference_quantile"] = (
        enriched["standardized_discrepancy"].abs() > reference_quantile
    )
    enriched["joint_bootstrap_exceedance_frequency"] = [
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
                "joint_reference_upper_tail_fraction": reference_upper_tail_fraction,
                "maximum_absolute_standardized_discrepancy": observed_max,
                "joint_max_abs_standardized_reference_quantile": reference_quantile,
                "observed_max_exceeds_joint_reference_quantile": (
                    observed_max > reference_quantile
                ),
                "estimands_exceeding_joint_reference_quantile": int(
                    enriched["exceeds_joint_reference_quantile"].sum()
                ),
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

    minimum_batches = min(_matrix_batch_count(cell) for cell in cells.values())
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
            cell = cells[(root, k)]
            batch_count = _matrix_batch_count(cell)
            if batch_count < 2:
                raise ValueError(f"root {root}, k={k} needs at least two batches for drift")
            midpoint = batch_count // 2
            halves = {
                "first_half": (0, midpoint),
                "second_half": (midpoint, batch_count),
            }
            for half, bounds in halves.items():
                half_estimates[half][k] = _estimate_matrix_cells(
                    cfg,
                    [cell],
                    k=k,
                    estimate_scope=half,
                    root_seed=root,
                    practical_delta=float(practical_by_k[k]),
                    row_bounds=bounds,
                )
            first = half_estimates["first_half"][k].set_index("strategy")
            second = half_estimates["second_half"][k].set_index("strategy")
            for strategy in sorted(first.index.astype(int)):
                raw = _at_float(first, strategy, "chance_delta") - _at_float(
                    second, strategy, "chance_delta"
                )
                expected = _combined_optional_mcse(
                    first,
                    second,
                    strategy,
                    "batch_mcse",
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
                        "practical_threshold_position_changed": (
                            _at_str(first, strategy, "practical_threshold_position")
                            != _at_str(second, strategy, "practical_threshold_position")
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
            expected = _combined_optional_mcse(
                first_across,
                second_across,
                strategy,
                "across_k_mcse",
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
                    "practical_threshold_position_changed": (
                        _at_str(first_across, strategy, "practical_threshold_position")
                        != _at_str(second_across, strategy, "practical_threshold_position")
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

    frame = frame.copy()
    if "strategy" in frame:
        frame["strategy"] = canonical_strategy_ids(
            frame["strategy"],
            nullable=bool(frame["strategy"].isna().any()),
            context=f"{operation} strategy",
        )
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="root_stability",
        scope=ArtifactScope.CROSS_SEED,
        source_scope=ArtifactScope.BY_K,
        operation=operation,
        method_contract={
            "kind": "root_combination",
            "procedure": operation,
            "parameters": {
                "method_version": ROOT_STABILITY_METHOD_VERSION,
                "design_interpretation": "fixed_design_descriptive_reproducibility",
                "interval_role": "monte_carlo_precision",
                "root_population_inference": "none",
                "multiple_testing_inference": "none",
            },
        },
        baseline="chance_1_over_k",
        weighted_quantity="win_rate_minus_chance",
        k_aggregation_method=k_aggregation_method,
        k_weights=(
            cfg.k_aggregation.k_weights if k_aggregation_method == "declared_mapping" else None
        ),
        support_count_role="raw_player_game_exposures",
        uncertainty_method=uncertainty_method,
        replication_unit="deterministic_shuffle_batch",
        conditioning=f"{ATTEMPT_CONDITIONING}_fixed_simulation_design",
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
    done = stage_done_path(cfg.stage_dir("root_stability"), "root_stability")
    stage_current = not force and stage_is_up_to_date(
        done,
        inputs=sources,
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="root_stability",
        freshness_key=_root_stability_freshness_key(cfg),
        sidecar_artifacts=list(artifacts.all_paths),
    )

    cfg.validate_resource_contract()
    policy = resolve_stage_parallel_policy("analysis", cfg.analysis, resources=cfg.resources)
    apply_native_thread_limits(policy)
    guard = ProcessTreeMemoryGuard(
        cfg.resources.hard_memory_limit_mb,
        rss_warning_mb=cfg.resources.target_memory_mb,
        sample_interval_seconds=cfg.resources.rss_sample_interval_seconds,
    )
    guard.check_before_schedule(force=True)
    matrix_cells = {
        key: _matrix_cell(cfg, cell, force=force, guard=guard)
        for key, cell in sorted(cell_map.items())
    }
    if stage_current and _bootstrap_ranges_validate(cfg, matrix_cells, root_pair):
        return artifacts

    by_k_tables, across_by_scope = _scope_estimates(cfg, matrix_cells, root_pair, required_k)
    discrepancies = _discrepancies(cfg, root_pair, by_k_tables, across_by_scope)
    discrepancies, joint_summary = _joint_discrepancy_bootstrap(
        cfg,
        matrix_cells,
        root_pair,
        required_k,
        discrepancies,
        force=force,
        guard=guard,
    )
    rank, top_n, controls, shortlist = _rank_and_selection_stability(
        cfg,
        root_pair,
        across_by_scope,
    )
    bootstrap_top_n = _root_bootstrap_top_n_inclusion(
        cfg,
        matrix_cells,
        root_pair,
        required_k,
        force=force,
        guard=guard,
    )
    convergence = _matched_count_convergence(
        cfg,
        matrix_cells,
        root_pair,
        required_k,
        across_by_scope,
    )
    drift = _half_drift(cfg, matrix_cells, root_pair, required_k)
    guard.check_before_schedule(force=True)
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
            uncertainty_method="descriptive_fixed_design_batch_ratio_mc_precision",
        )
    _write_frame(
        cfg,
        across,
        artifacts.across_k,
        operation=cfg_method_name(_k_weights(cfg, required_k), required_k),
        sources=sources,
        player_counts=required_k,
        grouping_keys=["estimate_scope", "root_seed", "strategy"],
        uncertainty_method="descriptive_fixed_design_independent_k_mc_precision",
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
        uncertainty_method="descriptive_fixed_root_joint_batch_resampling",
        k_aggregation_method=aggregation_method,
    )
    diagnostic_frames = (
        (
            discrepancies,
            artifacts.discrepancies,
            "root_difference",
            ["estimand_scope", "k", "strategy"],
            "descriptive_fixed_design_joint_max_batch_resampling",
        ),
        (
            joint_summary,
            artifacts.joint_discrepancy,
            "joint_discrepancy_diagnostic",
            ["root_a", "root_b"],
            "descriptive_fixed_design_joint_max_batch_resampling",
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
    guard.check_before_schedule(force=True)
    write_stage_done(
        done,
        inputs=sources,
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="root_stability",
        freshness_key=_root_stability_freshness_key(cfg),
        sidecar_artifacts=list(artifacts.all_paths),
    )
    return artifacts


__all__ = [
    "ROOT_STABILITY_METHOD_VERSION",
    "RootBatchCell",
    "RootStabilityArtifacts",
    "build_two_root_stability",
]
