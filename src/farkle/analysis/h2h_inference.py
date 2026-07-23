"""Seat-adjusted H2H score inference over the frozen candidate family."""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import brentq
from scipy.stats import norm
from statsmodels.stats.proportion import confint_proportions_2indep

from farkle.analysis.h2h_schedule import (
    H2H_CONDITIONING,
    H2H_METHOD_VERSION,
    SCORE_TEST_ID,
    h2h_method_contract,
)
from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.stage_completion import (
    CompletionState,
    stage_done_path,
    stage_is_up_to_date,
    write_stage_done,
)

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
    if nobs1 <= 0 or nobs2 <= 0:
        raise ValueError("score intervals require positive sample sizes")
    if not 0 <= count1 <= nobs1 or not 0 <= count2 <= nobs2:
        raise ValueError("score interval counts must lie within their sample sizes")
    try:
        # Statsmodels uses a fast closed-form constrained estimate, but its
        # endpoint evaluations and guessed Brent brackets are undefined for
        # valid 0/1 outcomes. Keep its established results whenever available.
        with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
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
        observed = count1 / nobs1 - count2 / nobs2
        if math.isfinite(low) and math.isfinite(high) and low <= observed <= high:
            return max(-1.0, float(low)), min(1.0, float(high))
    except ValueError:
        pass
    return _score_difference_interval_fallback(count1, nobs1, count2, nobs2, alpha=alpha)


def _score_statistic_at_difference(
    count1: int,
    nobs1: int,
    count2: int,
    nobs2: int,
    difference: float,
) -> float:
    """Evaluate Statsmodels' uncorrected constrained score statistic robustly."""

    observed = count1 / nobs1 - count2 / nobs2
    if difference <= -1.0:
        null_prop1, null_prop2 = 0.0, 1.0
    elif difference >= 1.0:
        null_prop1, null_prop2 = 1.0, 0.0
    elif difference == 0.0:
        constrained_null_rate = (count1 + count2) / (nobs1 + nobs2)
        null_prop1 = null_prop2 = constrained_null_rate
    else:
        total_n = nobs1 + nobs2
        total_count = count1 + count2
        cubic_2 = (nobs1 + 2 * nobs2) * difference - total_n - total_count
        cubic_1 = (count2 * difference - total_n - 2 * count2) * difference + total_count
        cubic_0 = count2 * difference * (1.0 - difference)
        q_value = (
            (cubic_2 / (3 * total_n)) ** 3
            - cubic_1 * cubic_2 / (6 * total_n**2)
            + cubic_0 / (2 * total_n)
        )
        radicand = (cubic_2 / (3 * total_n)) ** 2 - cubic_1 / (3 * total_n)
        cubic_p = math.copysign(math.sqrt(max(0.0, radicand)), q_value) if q_value != 0.0 else 0.0
        if cubic_p == 0.0:
            null_prop2 = -cubic_2 / (3 * total_n)
        else:
            cosine_argument = max(-1.0, min(1.0, q_value / cubic_p**3))
            angle = (math.pi + math.acos(cosine_argument)) / 3.0
            null_prop2 = 2.0 * cubic_p * math.cos(angle) - cubic_2 / (3 * total_n)
        null_prop1 = null_prop2 + difference
        null_prop1 = max(0.0, min(1.0, null_prop1))
        null_prop2 = max(0.0, min(1.0, null_prop2))

    variance = null_prop1 * (1.0 - null_prop1) / nobs1 + null_prop2 * (1.0 - null_prop2) / nobs2
    numerator = observed - difference
    if variance > 0.0:
        return numerator / math.sqrt(variance)
    if numerator == 0.0:
        return 0.0
    return math.copysign(math.inf, numerator)


def _score_interval_bound(
    count1: int,
    nobs1: int,
    count2: int,
    nobs2: int,
    *,
    observed: float,
    endpoint: float,
    critical_value: float,
) -> float:
    """Find the first score-test rejection moving outward from the estimate."""

    if observed == endpoint:
        return endpoint

    def objective(difference: float) -> float:
        statistic = _score_statistic_at_difference(count1, nobs1, count2, nobs2, difference)
        if math.isnan(statistic):
            raise RuntimeError("score interval fallback produced an undefined statistic")
        if math.isinf(statistic):
            return 1.0
        return abs(statistic) - critical_value

    previous = observed
    span = endpoint - observed
    for exponent in range(52, -1, -1):
        candidate = observed + span * 2.0**-exponent
        if candidate == previous:
            continue
        if objective(candidate) >= 0.0:
            return float(
                brentq(
                    objective,
                    min(previous, candidate),
                    max(previous, candidate),
                    xtol=1e-12,
                    rtol=1e-14,
                )
            )
        previous = candidate
    raise RuntimeError("score interval fallback could not bracket a confidence bound")


def _score_difference_interval_fallback(
    count1: int,
    nobs1: int,
    count2: int,
    nobs2: int,
    *,
    alpha: float,
) -> tuple[float, float]:
    """Complete score inversion for boundary outcomes and failed library brackets."""

    observed = count1 / nobs1 - count2 / nobs2
    if observed > 0.0:
        swapped_low, swapped_high = _score_difference_interval_fallback(
            count2,
            nobs2,
            count1,
            nobs1,
            alpha=alpha,
        )
        return -swapped_high, -swapped_low

    critical_value = float(norm.isf(alpha / 2.0))
    low = _score_interval_bound(
        count1,
        nobs1,
        count2,
        nobs2,
        observed=observed,
        endpoint=-1.0,
        critical_value=critical_value,
    )
    high = _score_interval_bound(
        count1,
        nobs1,
        count2,
        nobs2,
        observed=observed,
        endpoint=1.0,
        critical_value=critical_value,
    )
    if count1 == count2 and nobs1 == nobs2:
        symmetric = max(abs(low), abs(high))
        return -symmetric, symmetric
    return low, high


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
    schedule_path = cfg.h2h_block_manifest_path()
    state_path = cfg.h2h_execution_state_path()
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
    validate_artifact_sidecar(
        schedule_path,
        expected={
            "scope": ArtifactScope.H2H_2P.value,
            "operation": "construct_pair_root_order_blocks",
        },
    )
    validate_artifact_sidecar(
        state_path,
        expected={
            "scope": ArtifactScope.H2H_2P.value,
            "operation": "h2h_execution_state",
        },
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    execution = json.loads(state_path.read_text(encoding="utf-8"))
    if plan.get("planning_state") != CompletionState.COMPLETE_VALID.value:
        raise RuntimeError("seat-adjusted inference requires a complete valid H2H power plan")
    if int(plan.get("h2h_method_version", -1)) != H2H_METHOD_VERSION:
        raise ValueError("H2H power plan uses an incompatible method version")
    if execution.get("execution_state") != CompletionState.COMPLETE_VALID.value:
        raise RuntimeError("seat-adjusted inference requires complete valid H2H execution")
    for field in ("family_hash", "schedule_hash", "total_block_count"):
        if execution.get(field) != plan.get(field):
            raise ValueError(f"H2H execution state does not match power-plan {field}")
    if execution.get("completed_block_count") != execution.get("total_block_count"):
        raise RuntimeError("H2H execution state does not cover every scheduled block")
    required = {
        "family_hash",
        "schedule_hash",
        "block_id",
        "pair_id",
        "strategy_a",
        "strategy_b",
        "root_seed",
        "root_index",
        "order",
        "order_label",
        "seat1_strategy",
        "seat2_strategy",
        "n_completed_required",
        "max_attempts",
        "games_attempted",
        "games_completed",
        "games_safety_limit",
        "replacement_attempt_count",
        "completion_status",
        "wins_seat1",
        "wins_seat2",
        "wins_a",
        "wins_b",
        "rng_scheme_version",
        "outcome_schema_version",
        "h2h_method_version",
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
    if set(frame["schedule_hash"].astype(str)) != {str(plan["schedule_hash"])}:
        raise ValueError("H2H counts do not match the power-plan schedule hash")
    if set(frame["score_test_id"].astype(str)) != {SCORE_TEST_ID}:
        raise ValueError("H2H counts were not scheduled for the accepted score procedure")
    if set(frame["h2h_method_version"].astype(int)) != {H2H_METHOD_VERSION}:
        raise ValueError("H2H counts use an incompatible method version")
    if not (
        frame["games_attempted"].astype(int)
        == frame["games_completed"].astype(int) + frame["games_safety_limit"].astype(int)
    ).all():
        raise ValueError("H2H root/order counts violate attempt conservation")
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
    schedule = pq.read_table(schedule_path).to_pandas()
    if schedule.empty:
        raise ValueError("H2H block manifest is empty")
    schedule_required = {
        "family_hash",
        "schedule_hash",
        "block_id",
        "pair_id",
        "strategy_a",
        "strategy_b",
        "root_seed",
        "root_index",
        "order",
        "order_label",
        "seat1_strategy",
        "seat2_strategy",
        "n_completed_required",
        "max_attempts",
        "rng_scheme_version",
        "outcome_schema_version",
        "h2h_method_version",
        "score_test_id",
    }
    missing_schedule = sorted(schedule_required.difference(schedule.columns))
    if missing_schedule:
        raise ValueError(f"H2H schedule lacks authentication columns: {missing_schedule}")
    if schedule.duplicated(keys).any():
        raise ValueError("H2H schedule contains duplicate immutable blocks")
    if (
        not schedule["n_completed_required"]
        .astype(int)
        .eq(int(plan["n_completed_required_per_root_order_block"]))
        .all()
    ):
        raise ValueError("H2H schedule changed the frozen completed-game target")
    if (
        not schedule["max_attempts"]
        .astype(int)
        .eq(int(plan["max_attempts_per_root_order_block"]))
        .all()
    ):
        raise ValueError("H2H schedule changed the frozen maximum-attempt rule")
    if set(map(tuple, frame[keys].to_numpy())) != set(map(tuple, schedule[keys].to_numpy())):
        raise ValueError("H2H counts do not cover the exact frozen root/order schedule")
    schedule_indexed = schedule.set_index(keys).sort_index()
    frame_indexed = frame.set_index(keys).sort_index()
    static_columns = sorted(schedule_required.difference(keys))
    for column in static_columns:
        if not frame_indexed[column].astype(str).equals(schedule_indexed[column].astype(str)):
            raise ValueError(f"H2H counts disagree with the frozen schedule column {column}")
    terminal_statuses = {"complete", "unresolved_nonviable"}
    if not set(frame["completion_status"].astype(str)).issubset(terminal_statuses):
        raise ValueError("H2H aggregate contains a nonterminal root/order block")
    complete = frame["completion_status"].astype(str).eq("complete")
    unresolved = frame["completion_status"].astype(str).eq("unresolved_nonviable")
    if not (
        frame.loc[complete, "games_completed"].astype(int)
        == frame.loc[complete, "n_completed_required"].astype(int)
    ).all():
        raise ValueError("complete H2H blocks do not reach their completed-game target")
    if not (
        (
            frame.loc[unresolved, "games_attempted"].astype(int)
            == frame.loc[unresolved, "max_attempts"].astype(int)
        )
        & (
            frame.loc[unresolved, "games_completed"].astype(int)
            < frame.loc[unresolved, "n_completed_required"].astype(int)
        )
    ).all():
        raise ValueError("unresolved H2H blocks did not exhaust their declared attempt cap")
    return frame, plan


def _combine_within_order(
    frame: pd.DataFrame,
    plan: Mapping[str, Any],
    *,
    expected_root_count: int | None = None,
) -> pd.DataFrame:
    """Combine raw counts across roots without mixing the two seat orders."""

    root_count = (
        len(cast(list[object], plan["root_seeds"]))
        if expected_root_count is None
        else expected_root_count
    )
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
            root_order_cell_count=("root_seed", "size"),
            resolved_root_order_cell_count=(
                "completion_status",
                lambda values: int(pd.Series(values).astype(str).eq("complete").sum()),
            ),
            n_completed_required=("n_completed_required", "sum"),
            max_attempts=("max_attempts", "sum"),
            games_attempted=("games_attempted", "sum"),
            games_completed=("games_completed", "sum"),
            games_safety_limit=("games_safety_limit", "sum"),
            replacement_attempt_count=("replacement_attempt_count", "sum"),
            wins_seat1=("wins_seat1", "sum"),
            wins_seat2=("wins_seat2", "sum"),
            wins_a=("wins_a", "sum"),
            wins_b=("wins_b", "sum"),
        )
        .sort_values(["pair_id", "order"], kind="mergesort")
        .reset_index(drop=True)
    )
    if not combined["root_count"].eq(root_count).all():
        raise ValueError("root combination changed support between seat orders")
    combined["order_support_complete"] = combined["resolved_root_order_cell_count"].eq(
        combined["root_order_cell_count"]
    ) & combined["games_completed"].eq(combined["n_completed_required"])
    combined["completion_game_rate"] = combined["games_completed"] / combined["games_attempted"]
    combined["safety_limit_game_rate"] = (
        combined["games_safety_limit"] / combined["games_attempted"]
    )
    return combined


def _viability_status(
    counts: pd.DataFrame,
    plan: Mapping[str, Any],
) -> tuple[dict[int, bool], dict[str, dict[str, Any]]]:
    """Compute frozen-pair support and incident-attempt candidate viability."""

    pair_viable = {
        int(cast(Any, pair_id)): bool(
            group["completion_status"].astype(str).eq("complete").all()
            and (
                group["games_completed"].astype(int) == group["n_completed_required"].astype(int)
            ).all()
        )
        for pair_id, group in counts.groupby("pair_id", sort=True)
    }
    incident_rows: list[dict[str, Any]] = []
    for row in cast(list[dict[str, Any]], counts.to_dict(orient="records")):
        for strategy in (str(row["strategy_a"]), str(row["strategy_b"])):
            incident_rows.append(
                {
                    "strategy": strategy,
                    "pair_id": int(row["pair_id"]),
                    "games_attempted": int(row["games_attempted"]),
                    "games_completed": int(row["games_completed"]),
                    "games_safety_limit": int(row["games_safety_limit"]),
                    "replacement_attempt_count": int(row["replacement_attempt_count"]),
                }
            )
    incident = pd.DataFrame(incident_rows)
    threshold = float(plan["min_candidate_completion_rate"])
    status: dict[str, dict[str, Any]] = {}
    for strategy_key, group in incident.groupby("strategy", sort=True):
        strategy = str(cast(Any, strategy_key))
        attempted = int(group["games_attempted"].sum())
        completed = int(group["games_completed"].sum())
        safety_limit = int(group["games_safety_limit"].sum())
        replacements = int(group["replacement_attempt_count"].sum())
        completion_rate = completed / attempted if attempted else None
        incident_pairs = sorted(set(group["pair_id"].astype(int)))
        inferentially_viable = all(pair_viable[pair_id] for pair_id in incident_pairs)
        operationally_viable = completion_rate is not None and completion_rate >= threshold
        status[strategy] = {
            "strategy": strategy,
            "games_attempted": attempted,
            "games_completed": completed,
            "games_safety_limit": safety_limit,
            "replacement_attempt_count": replacements,
            "completion_rate": completion_rate,
            "safety_limit_rate": safety_limit / attempted if attempted else None,
            "min_candidate_completion_rate": threshold,
            "operationally_viable": operationally_viable,
            "inferentially_viable": inferentially_viable,
            "candidate_status": (
                "viable"
                if operationally_viable and inferentially_viable
                else (
                    "operationally_nonviable"
                    if not operationally_viable
                    else "inferentially_nonviable"
                )
            ),
        }
    return pair_viable, status


def _pairwise_estimates(
    cfg: AppConfig,
    combined: pd.DataFrame,
    plan: Mapping[str, Any],
    *,
    pair_viable: Mapping[int, bool],
    candidate_status: Mapping[str, Mapping[str, Any]],
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
        ab = cast(pd.Series, ordered.loc[0])
        ba = cast(pd.Series, ordered.loc[1])
        pair_id_int = int(cast(int, pair_id))
        n_ab = int(cast(int, ab["games_completed"]))
        n_ba = int(cast(int, ba["games_completed"]))
        x_ab = int(cast(int, ab["wins_a"]))
        x_ba = int(cast(int, ba["wins_b"]))
        a_wins_ba = int(cast(int, ba["wins_a"]))
        strategy_a = str(ab["strategy_a"])
        strategy_b = str(ab["strategy_b"])
        a_status = candidate_status[strategy_a]
        b_status = candidate_status[strategy_b]
        inferentially_viable = bool(pair_viable[pair_id_int])
        if inferentially_viable and n_ab != n_ba:
            raise ValueError(f"pair {pair_id} is not exactly balanced between seat orders")
        pair_operationally_viable = bool(
            a_status["operationally_viable"] and b_status["operationally_viable"]
        )
        row: dict[str, Any] = {
            "family_hash": str(ab["family_hash"]),
            "pair_id": pair_id_int,
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "games_attempted": int(ab["games_attempted"]) + int(ba["games_attempted"]),
            "games_completed": n_ab + n_ba,
            "games_safety_limit": int(ab["games_safety_limit"]) + int(ba["games_safety_limit"]),
            "replacement_attempt_count": int(ab["replacement_attempt_count"])
            + int(ba["replacement_attempt_count"]),
            "completion_game_rate": (
                (n_ab + n_ba) / (int(ab["games_attempted"]) + int(ba["games_attempted"]))
            ),
            "n_completed_required": int(ab["n_completed_required"])
            + int(ba["n_completed_required"]),
            "pair_inferentially_viable": inferentially_viable,
            "pair_operationally_viable": pair_operationally_viable,
            "pair_claim_eligible": inferentially_viable and pair_operationally_viable,
            "completion_status": ("complete" if inferentially_viable else "unresolved_nonviable"),
            "strategy_a_completion_rate": a_status["completion_rate"],
            "strategy_b_completion_rate": b_status["completion_rate"],
            "strategy_a_games_attempted": a_status["games_attempted"],
            "strategy_b_games_attempted": b_status["games_attempted"],
            "strategy_a_games_completed": a_status["games_completed"],
            "strategy_b_games_completed": b_status["games_completed"],
            "strategy_a_games_safety_limit": a_status["games_safety_limit"],
            "strategy_b_games_safety_limit": b_status["games_safety_limit"],
            "strategy_a_replacement_attempt_count": a_status["replacement_attempt_count"],
            "strategy_b_replacement_attempt_count": b_status["replacement_attempt_count"],
            "strategy_a_operationally_viable": a_status["operationally_viable"],
            "strategy_b_operationally_viable": b_status["operationally_viable"],
            "strategy_a_inferentially_viable": a_status["inferentially_viable"],
            "strategy_b_inferentially_viable": b_status["inferentially_viable"],
            "min_candidate_completion_rate": float(plan["min_candidate_completion_rate"]),
            "n_ab": n_ab if inferentially_viable else None,
            "n_ba": n_ba if inferentially_viable else None,
            "seat1_a_wins_ab": x_ab if inferentially_viable else None,
            "seat1_b_wins_ba": x_ba if inferentially_viable else None,
            "seat2_a_wins_ba": a_wins_ba if inferentially_viable else None,
            "descriptive_a_wins_completed": x_ab + a_wins_ba,
            "descriptive_completed_games": n_ab + n_ba,
            "descriptive_a_completed_win_rate": (
                (x_ab + a_wins_ba) / (n_ab + n_ba) if n_ab + n_ba else None
            ),
            "q_ab": None,
            "q_ba": None,
            "raw_order_rate_difference": None,
            "d_ab": None,
            "score_null_proportion": None,
            "score_z": None,
            "score_p_value": None,
            "ordinary_alpha": ordinary_alpha,
            "ordinary_d_low": None,
            "ordinary_d_high": None,
            "bonferroni_alpha_per_pair": simultaneous_alpha,
            "simultaneous_d_low": None,
            "simultaneous_d_high": None,
            "balanced_a_wins": None,
            "balanced_total_games": None,
            "balanced_a_win_rate_alias": None,
            "balanced_alias_checked": False,
            "formal_test_performed": inferentially_viable,
            "multiplicity_family_member": True,
            "no_test_p_value_convention": (
                None if inferentially_viable else "null_reported_treated_as_one_for_holm"
            ),
            "score_test_id": SCORE_TEST_ID,
            "interval_method_id": _INTERVAL_METHOD,
            "h2h_method_version": H2H_METHOD_VERSION,
            "planned_target_power": float(plan["target_power"]),
            "planned_worst_scenario_power": float(plan["worst_scenario_achieved_power"]),
        }
        if inferentially_viable:
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
            balanced_a_wins = x_ab + a_wins_ba
            balanced_a_rate = balanced_a_wins / (n_ab + n_ba)
            alias = 0.5 + effect
            if not math.isclose(balanced_a_rate, alias, rel_tol=0.0, abs_tol=1e-15):
                raise RuntimeError("actual A-win alias disagrees with the seat-order estimator")
            row.update(
                {
                    "q_ab": q_ab,
                    "q_ba": q_ba,
                    "raw_order_rate_difference": result.difference,
                    "d_ab": effect,
                    "score_null_proportion": result.null_proportion,
                    "score_z": result.statistic,
                    "score_p_value": result.p_value,
                    "ordinary_d_low": 0.5 * ordinary_low,
                    "ordinary_d_high": 0.5 * ordinary_high,
                    "simultaneous_d_low": 0.5 * simultaneous_low,
                    "simultaneous_d_high": 0.5 * simultaneous_high,
                    "balanced_a_wins": balanced_a_wins,
                    "balanced_total_games": n_ab + n_ba,
                    "balanced_a_win_rate_alias": balanced_a_rate,
                    "balanced_alias_checked": True,
                }
            )
        rows.append(row)
    output = pd.DataFrame(rows).sort_values("pair_id", kind="mergesort").reset_index(drop=True)
    performed = output["formal_test_performed"].astype(bool).to_numpy()
    working_p_values = np.where(
        performed,
        pd.to_numeric(output["score_p_value"], errors="coerce").fillna(1.0).to_numpy(dtype=float),
        1.0,
    )
    adjusted, positions = _holm_adjust(working_p_values)
    output["holm_order"] = pd.array(
        [
            int(position) if valid else None
            for position, valid in zip(positions, performed, strict=True)
        ],
        dtype="Int64",
    )
    output["holm_adjusted_p"] = np.where(performed, adjusted, np.nan)
    output["holm_reject_before_viability"] = performed & (adjusted <= ordinary_alpha)
    output["holm_reject"] = output["holm_reject_before_viability"].astype(bool) & output[
        "pair_claim_eligible"
    ].astype(bool)
    practical = cfg.head2head.practical_delta
    equivalence = cfg.head2head.delta_equivalence
    classifications: list[str] = []
    decision_rows = cast(list[dict[str, Any]], output.to_dict(orient="records"))
    for row in decision_rows:
        if not bool(row["pair_claim_eligible"]):
            classifications.append("unresolved_nonviable")
            continue
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
    *,
    pair_viable: Mapping[int, bool],
    candidate_status: Mapping[str, Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate fixed-root score diagnostics and cross-root agreement summaries."""

    root_frames: list[pd.DataFrame] = []
    roots = [int(cast(Any, root)) for root in cast(list[object], plan["root_seeds"])]
    for root in roots:
        selected = counts.loc[counts["root_seed"].astype(int).eq(root)].copy()
        root_orders = _combine_within_order(
            selected,
            plan,
            expected_root_count=1,
        )
        root_inference = _pairwise_estimates(
            cfg,
            root_orders,
            plan,
            pair_viable=pair_viable,
            candidate_status=candidate_status,
        )
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
                    "games_attempted",
                    "games_completed",
                    "games_safety_limit",
                    "replacement_attempt_count",
                    "completion_game_rate",
                    "completion_status",
                    "pair_inferentially_viable",
                    "pair_operationally_viable",
                    "pair_claim_eligible",
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
            "root_a_d_ab": (None if pd.isna(first["d_ab"]) else float(first["d_ab"])),
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
            first_effect = None if pd.isna(first["d_ab"]) else float(first["d_ab"])
            second_effect = None if pd.isna(second["d_ab"]) else float(second["d_ab"])
            discrepancy = (
                None
                if first_effect is None or second_effect is None
                else first_effect - second_effect
            )
            row.update(
                {
                    "root_b": int(second["root_seed"]),
                    "root_b_d_ab": second_effect,
                    "root_b_diagnostic_holm_decision": str(second["diagnostic_holm_decision"]),
                    "effect_discrepancy_a_minus_b": discrepancy,
                    "absolute_effect_discrepancy": (
                        None if discrepancy is None else abs(discrepancy)
                    ),
                    "diagnostic_holm_decision_agreement": (
                        str(first["diagnostic_holm_decision"])
                        == str(second["diagnostic_holm_decision"])
                        if first_effect is not None and second_effect is not None
                        else None
                    ),
                    "effect_direction_agreement": (
                        None
                        if first_effect is None or second_effect is None
                        else int(np.sign(first_effect)) == int(np.sign(second_effect))
                    ),
                    "agreement_available": (first_effect is not None and second_effect is not None),
                    "interpretation": (
                        "fixed_root_reproducibility_diagnostic_not_population_inference"
                        if first_effect is not None and second_effect is not None
                        else "unavailable_for_unresolved_nonviable_pair"
                    ),
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
    plan: Mapping[str, Any],
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
        support_count_role="attempted_and_completed_games_per_seat_order",
        uncertainty_method=uncertainty_method,
        replication_unit="independent_h2h_game",
        conditioning=H2H_CONDITIONING,
        consistency_columns=frame.columns.tolist(),
        source_artifacts=sources,
        grouping_keys=grouping_keys,
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined" if len(roots) == 2 else "single_root",
        method_contract=h2h_method_contract(
            cfg,
            operation,
            family_hash=str(plan["family_hash"]),
            schedule_hash=str(plan["schedule_hash"]),
        ),
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
    schedule_path = cfg.h2h_block_manifest_path()
    state_path = cfg.h2h_execution_state_path()
    sources = [counts_path, plan_path, schedule_path, state_path]
    artifacts = H2HInferenceArtifacts(
        combined_order_counts=cfg.h2h_combined_order_counts_path(),
        pairwise_inference=cfg.h2h_pairwise_inference_path(),
        root_pairwise_diagnostics=cfg.h2h_root_pairwise_diagnostics_path(),
        root_agreement=cfg.h2h_root_agreement_path(),
    )
    done = stage_done_path(cfg.stage_dir("h2h_inference"), "h2h_inference")
    if not force and stage_is_up_to_date(
        done,
        inputs=sources,
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="h2h_inference",
        sidecar_artifacts=list(artifacts.all_paths),
    ):
        return artifacts
    counts, plan = _read_counts(cfg)
    combined = _combine_within_order(counts, plan)
    pair_viable, candidate_status = _viability_status(counts, plan)
    inference = _pairwise_estimates(
        cfg,
        combined,
        plan,
        pair_viable=pair_viable,
        candidate_status=candidate_status,
    )
    root_diagnostics, root_agreement = _root_specific_diagnostics(
        cfg,
        counts,
        plan,
        pair_viable=pair_viable,
        candidate_status=candidate_status,
    )
    roots = [int(root) for root in plan["root_seeds"]]
    _write_frame(
        cfg,
        combined,
        artifacts.combined_order_counts,
        operation="combine_roots_within_seat_order",
        sources=sources,
        grouping_keys=["pair_id", "order"],
        roots=roots,
        plan=plan,
    )
    _write_frame(
        cfg,
        inference,
        artifacts.pairwise_inference,
        operation="seat_adjusted_score_inference",
        sources=[artifacts.combined_order_counts, plan_path, schedule_path, state_path],
        grouping_keys=["pair_id"],
        roots=roots,
        plan=plan,
    )
    _write_frame(
        cfg,
        root_diagnostics,
        artifacts.root_pairwise_diagnostics,
        operation="root_specific_score_diagnostic",
        sources=[counts_path, plan_path, schedule_path, state_path],
        grouping_keys=["root_seed", "pair_id"],
        roots=roots,
        plan=plan,
    )
    _write_frame(
        cfg,
        root_agreement,
        artifacts.root_agreement,
        operation="fixed_root_h2h_agreement_diagnostic",
        sources=[artifacts.root_pairwise_diagnostics],
        grouping_keys=["pair_id"],
        roots=roots,
        plan=plan,
        weighted_quantity="fixed_root_effect_discrepancy",
        uncertainty_method="descriptive_fixed_root_decision_comparison",
    )
    write_stage_done(
        done,
        inputs=sources,
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="h2h_inference",
        sidecar_artifacts=list(artifacts.all_paths),
    )
    return artifacts


__all__ = [
    "H2HInferenceArtifacts",
    "run_h2h_inference",
    "score_difference_interval",
    "two_proportion_score_test",
]
