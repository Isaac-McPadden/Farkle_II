"""Seat-adjusted H2H score inference over the frozen candidate family."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import norm
from statsmodels.stats.proportion import confint_proportions_2indep

from farkle.analysis.h2h_schedule import SCORE_TEST_ID
from farkle.analysis.stage_state import (
    CompletionState,
    stage_done_path,
    stage_is_up_to_date,
    write_stage_done,
)
from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic

_INTERVAL_METHOD: Final = "independent_two_proportion_score_inversion_v1"


@dataclass(frozen=True)
class H2HInferenceArtifacts:
    """Root-combined order counts and pairwise seat-adjusted decisions."""

    combined_order_counts: Path
    pairwise_inference: Path
    root_pairwise_diagnostics: Path
    root_agreement: Path

    @property
    def all_paths(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.combined_order_counts,
            self.pairwise_inference,
            self.root_pairwise_diagnostics,
            self.root_agreement,
        )


@dataclass(frozen=True)
class _ScoreResult:
    difference: float
    null_proportion: float
    statistic: float
    p_value: float


def two_proportion_score_test(
    count1: int,
    nobs1: int,
    count2: int,
    nobs2: int,
) -> _ScoreResult:
    """Test equality of two independent proportions using the constrained null."""

    if nobs1 <= 0 or nobs2 <= 0:
        raise ValueError("two-proportion score tests require positive sample sizes")
    if not 0 <= count1 <= nobs1 or not 0 <= count2 <= nobs2:
        raise ValueError("two-proportion score counts must lie within their sample sizes")
    rate1 = count1 / nobs1
    rate2 = count2 / nobs2
    difference = rate1 - rate2
    null_proportion = (count1 + count2) / (nobs1 + nobs2)
    variance = null_proportion * (1.0 - null_proportion) * (1.0 / nobs1 + 1.0 / nobs2)
    if variance > 0.0:
        statistic = difference / math.sqrt(variance)
        p_value = float(2.0 * norm.sf(abs(statistic)))
    elif difference == 0.0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = math.copysign(math.inf, difference)
        p_value = 0.0
    return _ScoreResult(
        difference=difference,
        null_proportion=null_proportion,
        statistic=statistic,
        p_value=p_value,
    )


def score_difference_interval(
    count1: int,
    nobs1: int,
    count2: int,
    nobs2: int,
    *,
    alpha: float,
) -> tuple[float, float]:
    """Invert the uncorrected independent-proportion score test for a difference."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("score interval alpha must be between zero and one")
    low, high = confint_proportions_2indep(
        count1,
        nobs1,
        count2,
        nobs2,
        method="score",
        compare="diff",
        alpha=alpha,
        correction=False,
    )
    return max(-1.0, float(low)), min(1.0, float(high))


def _holm_adjust(p_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Holm adjusted p-values and stable sorted positions."""

    count = len(p_values)
    order = np.argsort(p_values, kind="mergesort")
    adjusted_sorted = np.maximum.accumulate(
        np.asarray(
            [(count - index) * p_values[position] for index, position in enumerate(order)],
            dtype=float,
        )
    )
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = np.minimum(1.0, adjusted_sorted)
    positions = np.empty(count, dtype=np.int64)
    positions[order] = np.arange(1, count + 1, dtype=np.int64)
    return adjusted, positions


def _read_counts(cfg: AppConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts_path = cfg.h2h_order_counts_path()
    plan_path = cfg.h2h_power_plan_path()
    validate_artifact_sidecar(
        counts_path,
        expected={
            "scope": ArtifactScope.H2H_2P.value,
            "operation": "concatenate_root_order_blocks",
            "uncertainty_method": SCORE_TEST_ID,
        },
    )
    validate_artifact_sidecar(
        plan_path,
        expected={
            "scope": ArtifactScope.H2H_2P.value,
            "operation": "score_test_power_plan",
        },
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("planning_state") != CompletionState.COMPLETE_VALID.value:
        raise RuntimeError("seat-adjusted inference requires a complete valid H2H power plan")
    if plan.get("execution_state") != CompletionState.COMPLETE_VALID.value:
        raise RuntimeError("seat-adjusted inference requires complete valid H2H execution")
    required = {
        "family_hash",
        "pair_id",
        "strategy_a",
        "strategy_b",
        "root_seed",
        "order",
        "order_label",
        "games_required",
        "games_completed",
        "wins_seat1",
        "wins_seat2",
        "score_test_id",
    }
    schema = pq.read_schema(counts_path)
    missing = sorted(required.difference(schema.names))
    if missing:
        raise ValueError(f"H2H root/order counts lack inference columns: {missing}")
    frame = pq.read_table(counts_path, columns=sorted(required)).to_pandas()
    if frame.empty:
        raise ValueError("H2H root/order counts are empty")
    if set(frame["family_hash"].astype(str)) != {str(plan["family_hash"])}:
        raise ValueError("H2H counts do not match the power-plan family hash")
    if set(frame["score_test_id"].astype(str)) != {SCORE_TEST_ID}:
        raise ValueError("H2H counts were not scheduled for the accepted score procedure")
    if not (frame["games_required"].astype(int) == frame["games_completed"].astype(int)).all():
        raise ValueError("H2H inference refuses incomplete root/order blocks")
    if not (
        frame["wins_seat1"].astype(int) + frame["wins_seat2"].astype(int)
        == frame["games_completed"].astype(int)
    ).all():
        raise ValueError("H2H root/order wins do not equal completed games")
    keys = ["pair_id", "root_seed", "order"]
    if frame.duplicated(keys).any():
        raise ValueError("H2H root/order counts contain duplicate immutable blocks")
    expected_roots = {int(root) for root in plan["root_seeds"]}
    if set(frame["root_seed"].astype(int)) != expected_roots:
        raise ValueError("H2H root/order counts do not cover the planned roots")
    return frame, plan


def _combine_within_order(frame: pd.DataFrame, plan: Mapping[str, Any]) -> pd.DataFrame:
    """Combine raw counts across roots without mixing the two seat orders."""

    root_count = len(cast(list[object], plan["root_seeds"]))
    expected_cells = root_count * 2
    pair_sizes = frame.groupby("pair_id").size()
    if not pair_sizes.eq(expected_cells).all():
        invalid = pair_sizes.loc[~pair_sizes.eq(expected_cells)].to_dict()
        raise ValueError(f"H2H pairs lack complete root/order support: {invalid}")
    order_counts = frame.groupby("pair_id")["order"].nunique()
    if not order_counts.eq(2).all() or set(frame["order"].astype(int)) != {0, 1}:
        raise ValueError("every H2H pair must contain both independent seat orders")
    identity_counts = frame.groupby("pair_id").agg(
        strategy_a_count=("strategy_a", "nunique"),
        strategy_b_count=("strategy_b", "nunique"),
    )
    if not (
        identity_counts["strategy_a_count"].eq(1) & identity_counts["strategy_b_count"].eq(1)
    ).all():
        raise ValueError("H2H pair strategy identities vary across root/order blocks")
    combined = (
        frame.groupby(
            [
                "family_hash",
                "pair_id",
                "strategy_a",
                "strategy_b",
                "order",
                "order_label",
            ],
            as_index=False,
            sort=True,
        )
        .agg(
            root_count=("root_seed", "nunique"),
            games=("games_completed", "sum"),
            wins_seat1=("wins_seat1", "sum"),
            wins_seat2=("wins_seat2", "sum"),
        )
        .sort_values(["pair_id", "order"], kind="mergesort")
        .reset_index(drop=True)
    )
    if not combined["root_count"].eq(root_count).all():
        raise ValueError("root combination changed support between seat orders")
    return combined


def _pairwise_estimates(
    cfg: AppConfig,
    combined: pd.DataFrame,
    plan: Mapping[str, Any],
) -> pd.DataFrame:
    pair_count = int(plan["unordered_pair_count"])
    if combined["pair_id"].nunique() != pair_count:
        raise ValueError("combined H2H counts do not cover the planned unordered pairs")
    ordinary_alpha = cfg.head2head.family_alpha
    simultaneous_alpha = ordinary_alpha / pair_count
    rows: list[dict[str, Any]] = []
    for pair_id, group in combined.groupby("pair_id", sort=True):
        ordered = group.set_index("order")
        if set(ordered.index.astype(int)) != {0, 1}:
            raise ValueError(f"pair {pair_id} lacks both seat orders")
        ab = ordered.loc[0]
        ba = ordered.loc[1]
        n_ab = int(cast(int, ab["games"]))
        n_ba = int(cast(int, ba["games"]))
        if n_ab != n_ba:
            raise ValueError(f"pair {pair_id} is not exactly balanced between seat orders")
        x_ab = int(cast(int, ab["wins_seat1"]))
        x_ba = int(cast(int, ba["wins_seat1"]))
        result = two_proportion_score_test(x_ab, n_ab, x_ba, n_ba)
        ordinary_low, ordinary_high = score_difference_interval(
            x_ab,
            n_ab,
            x_ba,
            n_ba,
            alpha=ordinary_alpha,
        )
        simultaneous_low, simultaneous_high = score_difference_interval(
            x_ab,
            n_ab,
            x_ba,
            n_ba,
            alpha=simultaneous_alpha,
        )
        q_ab = x_ab / n_ab
        q_ba = x_ba / n_ba
        effect = 0.5 * result.difference
        balanced_a_wins = x_ab + (n_ba - x_ba)
        balanced_a_rate = balanced_a_wins / (n_ab + n_ba)
        alias = 0.5 + effect
        if not math.isclose(balanced_a_rate, alias, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError("balanced A-win alias disagrees with the seat-order estimator")
        rows.append(
            {
                "family_hash": str(ab["family_hash"]),
                "pair_id": int(cast(int, pair_id)),
                "strategy_a": str(ab["strategy_a"]),
                "strategy_b": str(ab["strategy_b"]),
                "n_ab": n_ab,
                "n_ba": n_ba,
                "seat1_a_wins_ab": x_ab,
                "seat1_b_wins_ba": x_ba,
                "q_ab": q_ab,
                "q_ba": q_ba,
                "raw_order_rate_difference": result.difference,
                "d_ab": effect,
                "score_null_proportion": result.null_proportion,
                "score_z": result.statistic,
                "score_p_value": result.p_value,
                "ordinary_alpha": ordinary_alpha,
                "ordinary_d_low": 0.5 * ordinary_low,
                "ordinary_d_high": 0.5 * ordinary_high,
                "bonferroni_alpha_per_pair": simultaneous_alpha,
                "simultaneous_d_low": 0.5 * simultaneous_low,
                "simultaneous_d_high": 0.5 * simultaneous_high,
                "balanced_a_wins": balanced_a_wins,
                "balanced_total_games": n_ab + n_ba,
                "balanced_a_win_rate_alias": balanced_a_rate,
                "balanced_alias_checked": True,
                "score_test_id": SCORE_TEST_ID,
                "interval_method_id": _INTERVAL_METHOD,
                "planned_target_power": float(plan["target_power"]),
                "planned_worst_scenario_power": float(plan["worst_scenario_achieved_power"]),
            }
        )
    output = pd.DataFrame(rows).sort_values("pair_id", kind="mergesort").reset_index(drop=True)
    adjusted, positions = _holm_adjust(output["score_p_value"].to_numpy(dtype=float))
    output["holm_order"] = positions
    output["holm_adjusted_p"] = adjusted
    output["holm_reject"] = adjusted <= ordinary_alpha
    practical = cfg.head2head.practical_delta
    equivalence = cfg.head2head.delta_equivalence
    classifications: list[str] = []
    decision_rows = cast(list[dict[str, Any]], output.to_dict(orient="records"))
    for row in decision_rows:
        simultaneous_low = float(row["simultaneous_d_low"])
        simultaneous_high = float(row["simultaneous_d_high"])
        effect = float(row["d_ab"])
        if simultaneous_low > practical:
            classification = "practical_dominance_a"
        elif simultaneous_high < -practical:
            classification = "practical_dominance_b"
        elif bool(row["holm_reject"]):
            classification = (
                "statistical_only_advantage_a" if effect > 0.0 else "statistical_only_advantage_b"
            )
        elif (
            equivalence is not None
            and simultaneous_low > -equivalence
            and simultaneous_high < equivalence
        ):
            classification = "equivalent"
        else:
            classification = "unresolved"
        classifications.append(classification)
    output["practical_delta"] = practical
    output["delta_equivalence"] = equivalence
    output["equivalence_enabled"] = equivalence is not None
    output["decision_class"] = classifications
    output["multiplicity_method"] = "holm"
    return output


def _root_specific_diagnostics(
    cfg: AppConfig,
    counts: pd.DataFrame,
    plan: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate fixed-root score diagnostics and cross-root agreement summaries."""

    root_frames: list[pd.DataFrame] = []
    roots = [int(root) for root in cast(list[object], plan["root_seeds"])]
    for root in roots:
        selected = counts.loc[counts["root_seed"].astype(int).eq(root)].copy()
        root_orders = selected.rename(
            columns={
                "games_completed": "games",
            }
        )
        root_orders["root_count"] = 1
        root_inference = _pairwise_estimates(cfg, root_orders, plan)
        root_inference.insert(0, "root_seed", root)
        root_inference["diagnostic_holm_decision"] = np.where(
            root_inference["holm_reject"].astype(bool),
            np.where(
                root_inference["d_ab"].astype(float) > 0.0,
                "diagnostic_advantage_a",
                "diagnostic_advantage_b",
            ),
            "diagnostic_no_adjusted_rejection",
        )
        root_inference["inference_role"] = "fixed_root_diagnostic_not_root_population"
        root_frames.append(
            root_inference[
                [
                    "root_seed",
                    "family_hash",
                    "pair_id",
                    "strategy_a",
                    "strategy_b",
                    "n_ab",
                    "n_ba",
                    "seat1_a_wins_ab",
                    "seat1_b_wins_ba",
                    "q_ab",
                    "q_ba",
                    "d_ab",
                    "score_null_proportion",
                    "score_z",
                    "score_p_value",
                    "ordinary_alpha",
                    "ordinary_d_low",
                    "ordinary_d_high",
                    "holm_order",
                    "holm_adjusted_p",
                    "holm_reject",
                    "diagnostic_holm_decision",
                    "score_test_id",
                    "interval_method_id",
                    "inference_role",
                ]
            ]
        )
    diagnostics = pd.concat(root_frames, ignore_index=True).sort_values(
        ["pair_id", "root_seed"], kind="mergesort"
    )

    agreement_rows: list[dict[str, Any]] = []
    for pair_id, group in diagnostics.groupby("pair_id", sort=True):
        ordered = group.sort_values("root_seed", kind="mergesort")
        first = ordered.iloc[0]
        row: dict[str, Any] = {
            "family_hash": str(first["family_hash"]),
            "pair_id": int(cast(int, pair_id)),
            "strategy_a": str(first["strategy_a"]),
            "strategy_b": str(first["strategy_b"]),
            "root_a": int(first["root_seed"]),
            "root_a_d_ab": float(first["d_ab"]),
            "root_a_diagnostic_holm_decision": str(first["diagnostic_holm_decision"]),
            "root_b": None,
            "root_b_d_ab": None,
            "root_b_diagnostic_holm_decision": None,
            "effect_discrepancy_a_minus_b": None,
            "absolute_effect_discrepancy": None,
            "diagnostic_holm_decision_agreement": None,
            "effect_direction_agreement": None,
            "agreement_available": False,
            "interpretation": "single_root_diagnostic_no_cross_root_stability_claim",
        }
        if len(ordered) == 2:
            second = ordered.iloc[1]
            first_effect = float(first["d_ab"])
            second_effect = float(second["d_ab"])
            discrepancy = first_effect - second_effect
            row.update(
                {
                    "root_b": int(second["root_seed"]),
                    "root_b_d_ab": second_effect,
                    "root_b_diagnostic_holm_decision": str(
                        second["diagnostic_holm_decision"]
                    ),
                    "effect_discrepancy_a_minus_b": discrepancy,
                    "absolute_effect_discrepancy": abs(discrepancy),
                    "diagnostic_holm_decision_agreement": str(
                        first["diagnostic_holm_decision"]
                    )
                    == str(second["diagnostic_holm_decision"]),
                    "effect_direction_agreement": int(np.sign(first_effect))
                    == int(np.sign(second_effect)),
                    "agreement_available": True,
                    "interpretation": "fixed_root_reproducibility_diagnostic_not_population_inference",
                }
            )
        agreement_rows.append(row)
    return diagnostics.reset_index(drop=True), pd.DataFrame(agreement_rows)


def _write_frame(
    cfg: AppConfig,
    frame: pd.DataFrame,
    path: Path,
    *,
    operation: str,
    sources: list[Path],
    grouping_keys: list[str],
    roots: list[int],
    weighted_quantity: str = "seat_adjusted_h2h_effect",
    uncertainty_method: str = f"{SCORE_TEST_ID}_holm",
) -> None:
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="h2h_inference",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation=operation,
        baseline="equal_seat_order_rates",
        weighted_quantity=weighted_quantity,
        support_count_role="independent_games_per_seat_order",
        uncertainty_method=uncertainty_method,
        replication_unit="independent_h2h_game",
        conditioning="frozen_finite_grid_candidate_family",
        consistency_columns=frame.columns.tolist(),
        source_artifacts=sources,
        grouping_keys=grouping_keys,
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined" if len(roots) == 2 else "single_root",
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        sidecar=sidecar,
        codec=cfg.parquet_codec,
    )


def run_h2h_inference(cfg: AppConfig, *, force: bool = False) -> H2HInferenceArtifacts:
    """Combine roots within orders, infer pair effects, and apply Holm decisions."""

    counts_path = cfg.h2h_order_counts_path()
    plan_path = cfg.h2h_power_plan_path()
    sources = [counts_path, plan_path]
    artifacts = H2HInferenceArtifacts(
        combined_order_counts=cfg.h2h_combined_order_counts_path(),
        pairwise_inference=cfg.h2h_pairwise_inference_path(),
        root_pairwise_diagnostics=cfg.h2h_root_pairwise_diagnostics_path(),
        root_agreement=cfg.h2h_root_agreement_path(),
    )
    done = stage_done_path(cfg.h2h_2p_dir(), "h2h_seat_adjusted_inference")
    if not force and stage_is_up_to_date(
        done,
        inputs=sources,
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="head2head",
        sidecar_artifacts=list(artifacts.all_paths),
    ):
        return artifacts
    counts, plan = _read_counts(cfg)
    combined = _combine_within_order(counts, plan)
    inference = _pairwise_estimates(cfg, combined, plan)
    root_diagnostics, root_agreement = _root_specific_diagnostics(cfg, counts, plan)
    roots = [int(root) for root in plan["root_seeds"]]
    _write_frame(
        cfg,
        combined,
        artifacts.combined_order_counts,
        operation="combine_roots_within_seat_order",
        sources=sources,
        grouping_keys=["pair_id", "order"],
        roots=roots,
    )
    _write_frame(
        cfg,
        inference,
        artifacts.pairwise_inference,
        operation="seat_adjusted_score_inference",
        sources=[artifacts.combined_order_counts, plan_path],
        grouping_keys=["pair_id"],
        roots=roots,
    )
    _write_frame(
        cfg,
        root_diagnostics,
        artifacts.root_pairwise_diagnostics,
        operation="root_specific_score_diagnostic",
        sources=[counts_path, plan_path],
        grouping_keys=["root_seed", "pair_id"],
        roots=roots,
    )
    _write_frame(
        cfg,
        root_agreement,
        artifacts.root_agreement,
        operation="fixed_root_h2h_agreement_diagnostic",
        sources=[artifacts.root_pairwise_diagnostics],
        grouping_keys=["pair_id"],
        roots=roots,
        weighted_quantity="fixed_root_effect_discrepancy",
        uncertainty_method="descriptive_fixed_root_decision_comparison",
    )
    write_stage_done(
        done,
        inputs=sources,
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="head2head",
        sidecar_artifacts=list(artifacts.all_paths),
    )
    return artifacts


__all__ = [
    "H2HInferenceArtifacts",
    "run_h2h_inference",
    "score_difference_interval",
    "two_proportion_score_test",
]
