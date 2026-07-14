"""Canonical method agreement conditioned on the frozen H2H finalist family."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import kendalltau, spearmanr

from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import write_json_artifact_atomic, write_parquet_artifact_atomic


def _read_frame(path: Path, *, operation: str) -> tuple[pd.DataFrame, str]:
    sidecar = validate_artifact_sidecar(
        path,
        expected={"scope": ArtifactScope.H2H_2P.value, "operation": operation},
    )
    return pq.read_table(path).to_pandas(), sidecar.seed_scope


def _correlation(frame: pd.DataFrame, method: str) -> float | None:
    selected = frame.loc[
        frame["scored_by_both_methods"].astype(bool),
        ["win_rate_rank", "trueskill_rank"],
    ].dropna()
    if len(selected) < 2:
        return None
    statistic = (
        spearmanr(selected["win_rate_rank"], selected["trueskill_rank"]).statistic
        if method == "spearman"
        else kendalltau(selected["win_rate_rank"], selected["trueskill_rank"]).statistic
    )
    value = float(statistic)
    return value if pd.notna(value) else None


def _set_overlap(left: set[str], right: set[str]) -> dict[str, int | float | None]:
    union = left | right
    intersection = left & right
    return {
        "win_rate_count": len(left),
        "trueskill_count": len(right),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard": (len(intersection) / len(union)) if union else None,
    }


def _preference(rank_a: object, rank_b: object) -> str | None:
    value_a = float(cast(Any, rank_a))
    value_b = float(cast(Any, rank_b))
    if pd.isna(value_a) or pd.isna(value_b):
        return None
    if value_a == value_b:
        return None
    return "a" if value_a < value_b else "b"


def _h2h_direction(decision: str) -> str | None:
    if decision.endswith("_a"):
        return "a"
    if decision.endswith("_b"):
        return "b"
    return None


def _pair_agreement(membership: pd.DataFrame, inference: pd.DataFrame) -> pd.DataFrame:
    membership = membership.copy()
    membership["strategy"] = membership["strategy"].astype(str)
    if membership["strategy"].duplicated().any():
        raise ValueError("candidate provenance contains duplicate strategies")
    finalists = set(membership.loc[membership["final_family"].astype(bool), "strategy"])
    inference_nodes = set(inference["strategy_a"].astype(str)) | set(
        inference["strategy_b"].astype(str)
    )
    if inference_nodes != finalists:
        raise ValueError("H2H inference finalist support differs from candidate provenance")
    ranks = membership.set_index("strategy")
    rows: list[dict[str, Any]] = []
    for raw in cast(list[dict[str, Any]], inference.to_dict(orient="records")):
        strategy_a = str(raw["strategy_a"])
        strategy_b = str(raw["strategy_b"])
        if strategy_a not in ranks.index or strategy_b not in ranks.index:
            raise ValueError("H2H inference contains a strategy outside candidate provenance")
        row_a = ranks.loc[strategy_a]
        row_b = ranks.loc[strategy_b]
        win_preference = _preference(row_a["win_rate_rank"], row_b["win_rate_rank"])
        trueskill_preference = _preference(row_a["trueskill_rank"], row_b["trueskill_rank"])
        decision = str(raw["decision_class"])
        h2h_direction = _h2h_direction(decision)
        rows.append(
            {
                "family_hash": str(raw["family_hash"]),
                "pair_id": int(raw["pair_id"]),
                "strategy_a": strategy_a,
                "strategy_b": strategy_b,
                "decision_class": decision,
                "h2h_direction": h2h_direction,
                "win_rate_preference": win_preference,
                "trueskill_preference": trueskill_preference,
                "win_rate_trueskill_agree": (
                    win_preference == trueskill_preference
                    if win_preference is not None and trueskill_preference is not None
                    else None
                ),
                "h2h_agrees_with_win_rate": (
                    h2h_direction == win_preference
                    if h2h_direction is not None and win_preference is not None
                    else None
                ),
                "h2h_agrees_with_trueskill": (
                    h2h_direction == trueskill_preference
                    if h2h_direction is not None and trueskill_preference is not None
                    else None
                ),
                "selection_conditioning": "frozen_finite_grid_candidate_family",
            }
        )
    return pd.DataFrame(rows).sort_values("pair_id", kind="mergesort").reset_index(drop=True)


def _fraction(frame: pd.DataFrame, column: str) -> float | None:
    available = frame[column].dropna()
    return float(available.astype(bool).mean()) if len(available) else None


def _summary(
    membership: pd.DataFrame,
    pairs: pd.DataFrame,
    root_agreement: pd.DataFrame,
    *,
    execution_scope: str,
) -> dict[str, Any]:
    initial_win = set(
        membership.loc[membership["initial_win_rate_contribution"].astype(bool), "strategy"].astype(
            str
        )
    )
    initial_ts = set(
        membership.loc[
            membership["initial_trueskill_contribution"].astype(bool), "strategy"
        ].astype(str)
    )
    final_win = set(
        membership.loc[membership["final_win_rate_contribution"].astype(bool), "strategy"].astype(
            str
        )
    )
    final_ts = set(
        membership.loc[membership["final_trueskill_contribution"].astype(bool), "strategy"].astype(
            str
        )
    )
    final_family = membership.loc[membership["final_family"].astype(bool)]
    directional = pairs.loc[pairs["h2h_direction"].notna()]
    agreement_available = root_agreement.loc[root_agreement["agreement_available"].astype(bool)]
    family_hashes = set(final_family["family_hash"].astype(str))
    if len(family_hashes) != 1:
        raise ValueError("candidate provenance must contain one final family hash")
    return {
        "family_hash": next(iter(family_hashes)),
        "execution_scope": execution_scope,
        "conditioning": "agreement_is_conditioned_on_the_frozen_finalist_family",
        "common_scored_population_count": int(membership["scored_by_both_methods"].sum()),
        "rank_agreement": {
            "spearman": _correlation(membership, "spearman"),
            "kendall": _correlation(membership, "kendall"),
        },
        "initial_contribution_overlap": _set_overlap(initial_win, initial_ts),
        "final_contribution_overlap": _set_overlap(final_win, final_ts),
        "admission_counts": {
            "final_family": int(final_family.shape[0]),
            "win_rate": int(final_family["final_win_rate_contribution"].sum()),
            "trueskill": int(final_family["final_trueskill_contribution"].sum()),
            "shared": int(final_family["final_shared_contribution"].sum()),
            "controls": int(final_family["configured_control"].sum()),
            "mandatory_diagnostics": int(final_family["mandatory_diagnostic"].sum()),
        },
        "selection_conditioned_h2h": {
            "unordered_pair_count": int(len(pairs)),
            "directional_pair_count": int(len(directional)),
            "unresolved_pair_count": int(pairs["decision_class"].eq("unresolved").sum()),
            "equivalent_pair_count": int(pairs["decision_class"].eq("equivalent").sum()),
            "win_rate_trueskill_direction_agreement_fraction": _fraction(
                pairs, "win_rate_trueskill_agree"
            ),
            "h2h_win_rate_direction_agreement_fraction": _fraction(
                directional, "h2h_agrees_with_win_rate"
            ),
            "h2h_trueskill_direction_agreement_fraction": _fraction(
                directional, "h2h_agrees_with_trueskill"
            ),
        },
        "root_specific_h2h_stability": {
            "available": bool(len(agreement_available)),
            "pair_count": int(len(root_agreement)),
            "diagnostic_decision_agreement_fraction": _fraction(
                agreement_available, "diagnostic_holm_decision_agreement"
            ),
            "effect_direction_agreement_fraction": _fraction(
                agreement_available, "effect_direction_agreement"
            ),
            "interpretation": "fixed_root_reproducibility_not_root_population_inference",
        },
    }


def run(
    cfg: AppConfig,
    *,
    force: bool = False,
    execution_scope: str = "root_pair",
) -> None:
    """Write canonical method-agreement artifacts after H2H inference."""

    membership_path = cfg.h2h_candidate_family_path()
    inference_path = cfg.h2h_pairwise_inference_path()
    root_agreement_path = cfg.h2h_root_agreement_path()
    sources = [membership_path, inference_path, root_agreement_path]
    membership, seed_scope = _read_frame(membership_path, operation="candidate_family_freeze")
    inference, inference_seed_scope = _read_frame(
        inference_path, operation="seat_adjusted_score_inference"
    )
    root_agreement, root_seed_scope = _read_frame(
        root_agreement_path, operation="fixed_root_h2h_agreement_diagnostic"
    )
    if {seed_scope, inference_seed_scope, root_seed_scope} != {seed_scope}:
        raise ValueError("agreement inputs use incompatible root scopes")
    pairs_path = cfg.structure_agreement_pairs_path()
    summary_path = cfg.structure_agreement_summary_path()
    outputs = [pairs_path, summary_path]
    done = stage_done_path(cfg.stage_dir("agreement"), "structure_agreement")
    if not force and stage_is_up_to_date(
        done,
        inputs=sources,
        outputs=outputs,
        cfg=cfg,
        stage="agreement",
        sidecar_artifacts=outputs,
    ):
        return
    pairs = _pair_agreement(membership, inference)
    summary = _summary(
        membership,
        pairs,
        root_agreement,
        execution_scope=execution_scope,
    )
    common: dict[str, Any] = {
        "producer": "structure_agreement",
        "scope": ArtifactScope.H2H_2P,
        "source_scope": ArtifactScope.H2H_2P,
        "operation": "selection_conditioned_method_agreement",
        "weighted_quantity": "pair_direction_agreement",
        "support_count_role": "frozen_unordered_finalist_pairs",
        "uncertainty_method": "descriptive_only",
        "replication_unit": "unordered_finalist_pair",
        "conditioning": "frozen_finite_grid_candidate_family",
        "source_artifacts": sources,
        "player_counts": [2],
        "required_player_counts": [2],
        "missing_cell_policy": "fail",
        "seed_scope": seed_scope,
    }
    pairs_sidecar = make_artifact_sidecar(
        cfg,
        pairs_path,
        consistency_columns=pairs.columns.tolist(),
        grouping_keys=["family_hash", "pair_id"],
        **common,
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(pairs, preserve_index=False),
        pairs_path,
        sidecar=pairs_sidecar,
        codec=cfg.parquet_codec,
    )
    summary_sidecar = make_artifact_sidecar(
        cfg,
        summary_path,
        consistency_columns=list(summary),
        grouping_keys=["family_hash"],
        **common,
    )
    write_json_artifact_atomic(summary, summary_path, sidecar=summary_sidecar)
    write_stage_done(
        done,
        inputs=sources,
        outputs=outputs,
        cfg=cfg,
        stage="agreement",
        sidecar_artifacts=outputs,
    )


__all__ = ["run"]
