"""Freeze the canonical finite-grid family admitted to two-player H2H."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.trueskill_screening import (
    TRUESKILL_CONDITIONING,
    trueskill_method_contract,
)
from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    ArtifactSidecar,
    make_artifact_sidecar,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.artifacts import (
    write_json_artifact_atomic,
    write_parquet_artifact_atomic,
)
from farkle.utils.stage_completion import (
    stage_done_path,
    stage_is_up_to_date,
    write_stage_done,
)

_WIN_RATE_OPERATIONS: Final = {"equal_k_mean", "declared_k_weighted_mean"}


@dataclass(frozen=True)
class CandidateFamilyArtifacts:
    """Canonical candidate-membership table and immutable identity manifest."""

    membership: Path
    manifest: Path

    @property
    def all_paths(self) -> tuple[Path, Path]:
        """Return outputs in publication order."""

        return (self.membership, self.manifest)


@dataclass(frozen=True)
class _RankedContribution:
    """One complete method-level ranking with stable strategy identifiers."""

    ranks: dict[str, int]
    scores: dict[str, float]
    score_name: str
    sidecar: ArtifactSidecar

    @property
    def ordered(self) -> list[str]:
        """Return strategy identifiers in declared rank order."""

        return sorted(self.ranks, key=self.ranks.__getitem__)


def _strategy_key(value: object) -> str:
    """Normalize numeric and textual strategy identifiers without guessing aliases."""

    if isinstance(value, bool) or value is None:
        raise ValueError(f"invalid strategy identifier: {value!r}")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"invalid numeric strategy identifier: {value!r}")
        return str(int(numeric))
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("strategy identifiers must not be blank")
    return normalized


def _configured_roots(cfg: AppConfig) -> tuple[int, ...]:
    """Resolve the declared one- or two-root execution contract."""

    if cfg.sim.seed_list is not None:
        roots = tuple(int(root) for root in cfg.sim.seed_list)
    else:
        roots = (int(cfg.sim.seed),)
    if len(roots) not in {1, 2} or len(set(roots)) != len(roots):
        raise ValueError(f"candidate freezing requires one or two distinct roots, found {roots}")
    return roots


def _validate_complete_support(frame: pd.DataFrame, *, label: str) -> None:
    if "complete_support" not in frame:
        raise ValueError(f"{label} contribution lacks complete_support")
    incomplete = frame.loc[~frame["complete_support"].astype(bool)]
    if not incomplete.empty:
        examples = incomplete["strategy"].map(_strategy_key).head(20).tolist()
        raise ValueError(f"{label} contribution contains incomplete configured support: {examples}")


def _rank_contribution(
    frame: pd.DataFrame,
    *,
    score_column: str,
    label: str,
    sidecar: ArtifactSidecar,
    declared_rank_column: str | None = None,
) -> _RankedContribution:
    """Validate one canonical contribution and return its complete rank mapping."""

    required = {"strategy", "complete_support", score_column}
    if declared_rank_column is not None:
        required.add(declared_rank_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} contribution lacks columns: {missing}")
    _validate_complete_support(frame, label=label)
    ranked = frame.loc[:, sorted(required)].copy()
    ranked["strategy"] = ranked["strategy"].map(_strategy_key)
    if ranked["strategy"].duplicated().any():
        duplicates = ranked.loc[ranked["strategy"].duplicated(), "strategy"].tolist()
        raise ValueError(f"{label} contribution has duplicate strategies: {duplicates[:20]}")
    ranked[score_column] = pd.to_numeric(ranked[score_column], errors="raise")
    if not np.isfinite(ranked[score_column].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contribution contains non-finite scores")

    if declared_rank_column is None:
        ranked = ranked.sort_values(
            [score_column, "strategy"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ranked["_source_rank"] = np.arange(1, len(ranked) + 1, dtype=np.int64)
    else:
        source_ranks = pd.to_numeric(ranked[declared_rank_column], errors="raise")
        if not np.equal(source_ranks, np.floor(source_ranks)).all():
            raise ValueError(f"{label} source ranks must be integers")
        ranked["_source_rank"] = source_ranks.astype(int)
        expected = list(range(1, len(ranked) + 1))
        if sorted(ranked["_source_rank"].tolist()) != expected:
            raise ValueError(f"{label} source ranks must be unique and contiguous")
        ranked = ranked.sort_values("_source_rank", kind="mergesort").reset_index(drop=True)
        ordered_scores = ranked[score_column].to_numpy(dtype=float)
        if len(ordered_scores) > 1 and np.any(ordered_scores[:-1] < ordered_scores[1:]):
            raise ValueError(f"{label} source ranks disagree with descending scores")

    strategies = ranked["strategy"].tolist()
    ranks = dict(zip(strategies, ranked["_source_rank"].astype(int).tolist(), strict=True))
    scores = dict(zip(strategies, ranked[score_column].astype(float).tolist(), strict=True))
    return _RankedContribution(
        ranks=ranks,
        scores=scores,
        score_name=score_column,
        sidecar=sidecar,
    )


def _load_win_rate_contribution(
    path: Path,
    roots: tuple[int, ...],
) -> _RankedContribution:
    """Load the canonical combined-root or explicitly single-root contribution."""

    sidecar = validate_artifact_sidecar(path)
    if sidecar.operation not in _WIN_RATE_OPERATIONS:
        raise ArtifactContractError(
            f"win-rate contribution operation {sidecar.operation!r} is not canonical"
        )
    schema = pq.read_schema(path)
    if len(roots) == 2:
        if sidecar.scope != ArtifactScope.CROSS_SEED.value:
            raise ArtifactContractError(
                "two-root candidate freezing requires the cross_seed win-rate contribution"
            )
        required = {"estimate_scope", "strategy", "across_k_score", "complete_support"}
        missing = sorted(required.difference(schema.names))
        if missing:
            raise ValueError(f"combined-root win-rate contribution lacks columns: {missing}")
        frame = pq.read_table(path, columns=sorted(required)).to_pandas()
        frame = frame.loc[frame["estimate_scope"].eq("combined_roots")].copy()
        if frame.empty:
            raise ValueError("combined-root win-rate contribution lacks combined_roots rows")
        return _rank_contribution(
            frame,
            score_column="across_k_score",
            label="win-rate",
            sidecar=sidecar,
        )

    if sidecar.scope != ArtifactScope.ACROSS_K.value:
        raise ArtifactContractError(
            "single-root candidate freezing requires the across_k win-rate contribution"
        )
    required = {"strategy", "equal_k_score", "complete_support"}
    missing = sorted(required.difference(schema.names))
    if missing:
        raise ValueError(f"single-root win-rate contribution lacks columns: {missing}")
    frame = pq.read_table(path, columns=sorted(required)).to_pandas()
    return _rank_contribution(
        frame,
        score_column="equal_k_score",
        label="win-rate",
        sidecar=sidecar,
    )


def _load_trueskill_contribution(path: Path, roots: tuple[int, ...]) -> _RankedContribution:
    """Load the screening-only complete-root/k TrueSkill contribution."""

    expected_seed_scope = "both_roots_combined" if len(roots) == 2 else "single_root"
    sidecar = validate_artifact_sidecar(
        path,
        expected={
            "scope": ArtifactScope.ACROSS_K.value,
            "operation": "equal_root_k_percentile_mean",
            "seed_scope": expected_seed_scope,
            "conditioning": TRUESKILL_CONDITIONING,
            "method_contract": trueskill_method_contract("equal_root_k_percentile_mean"),
        },
    )
    required = {
        "strategy",
        "mean_percentile_rank",
        "candidate_contribution_rank",
        "complete_support",
    }
    schema = pq.read_schema(path)
    missing = sorted(required.difference(schema.names))
    if missing:
        raise ValueError(f"TrueSkill contribution lacks columns: {missing}")
    frame = pq.read_table(path, columns=sorted(required)).to_pandas()
    return _rank_contribution(
        frame,
        score_column="mean_percentile_rank",
        label="TrueSkill",
        sidecar=sidecar,
        declared_rank_column="candidate_contribution_rank",
    )


def _top_set(contribution: _RankedContribution, cutoff: int) -> set[str]:
    return {strategy for strategy, rank in contribution.ranks.items() if rank <= cutoff}


def _family_at_cutoffs(
    win_rate: _RankedContribution,
    trueskill: _RankedContribution,
    cutoffs: dict[str, int],
    protected: set[str],
) -> set[str]:
    return (
        _top_set(win_rate, cutoffs["win_rate"])
        | _top_set(trueskill, cutoffs["trueskill"])
        | protected
    )


def _contract_family(
    *,
    win_rate: _RankedContribution,
    trueskill: _RankedContribution,
    contribution_size: int,
    candidate_cap: int | None,
    protected: set[str],
) -> tuple[set[str], dict[str, int], dict[str, int], list[dict[str, Any]], dict[str, int]]:
    """Apply simultaneous method-tail contraction and retain removal provenance."""

    initial_cutoffs = {
        "win_rate": min(contribution_size, len(win_rate.ranks)),
        "trueskill": min(contribution_size, len(trueskill.ranks)),
    }
    cutoffs = dict(initial_cutoffs)
    family = _family_at_cutoffs(win_rate, trueskill, cutoffs, protected)
    history: list[dict[str, Any]] = [
        {
            "round": 0,
            "win_rate_cutoff": cutoffs["win_rate"],
            "trueskill_cutoff": cutoffs["trueskill"],
            "family_size": len(family),
            "removed": [],
        }
    ]
    removal_round: dict[str, int] = {}
    if candidate_cap is not None and len(protected) > candidate_cap:
        raise ValueError(
            "head2head.candidate_cap is smaller than the protected control and "
            f"diagnostic family ({candidate_cap} < {len(protected)})"
        )
    round_index = 0
    while candidate_cap is not None and len(family) > candidate_cap:
        if cutoffs["win_rate"] == 0 and cutoffs["trueskill"] == 0:
            raise RuntimeError("candidate cap contraction cannot reduce the protected family")
        previous = family
        cutoffs = {method: max(0, cutoff - 1) for method, cutoff in cutoffs.items()}
        family = _family_at_cutoffs(win_rate, trueskill, cutoffs, protected)
        round_index += 1
        removed = sorted(previous.difference(family))
        removal_round.update(dict.fromkeys(removed, round_index))
        history.append(
            {
                "round": round_index,
                "win_rate_cutoff": cutoffs["win_rate"],
                "trueskill_cutoff": cutoffs["trueskill"],
                "family_size": len(family),
                "removed": removed,
            }
        )
    if len(family) < 2:
        raise ValueError(
            f"frozen H2H candidate family needs at least two strategies, found {len(family)}"
        )
    return family, initial_cutoffs, cutoffs, history, removal_round


def _overlap_summary(left: set[str], right: set[str]) -> dict[str, int | float]:
    intersection = left & right
    union = left | right
    smaller = min(len(left), len(right))
    return {
        "win_rate_count": len(left),
        "trueskill_count": len(right),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "overlap_coefficient": len(intersection) / smaller if smaller else 1.0,
    }


def _family_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _admission_reasons(
    strategy: str,
    *,
    win_set: set[str],
    trueskill_set: set[str],
    controls: set[str],
    diagnostics: set[str],
) -> list[str]:
    reasons: list[str] = []
    if strategy in win_set:
        reasons.append("win_rate_contribution")
    if strategy in trueskill_set:
        reasons.append("trueskill_contribution")
    if strategy in controls:
        reasons.append("configured_control")
    if strategy in diagnostics:
        reasons.append("mandatory_diagnostic")
    return reasons


def _membership_frame(
    *,
    win_rate: _RankedContribution,
    trueskill: _RankedContribution,
    family: set[str],
    initial_cutoffs: dict[str, int],
    final_cutoffs: dict[str, int],
    controls: set[str],
    diagnostics: set[str],
    removal_round: dict[str, int],
    contraction_rounds: int,
    family_hash: str,
) -> pd.DataFrame:
    population = sorted(set(win_rate.ranks) | set(trueskill.ranks))
    initial_win = _top_set(win_rate, initial_cutoffs["win_rate"])
    initial_ts = _top_set(trueskill, initial_cutoffs["trueskill"])
    final_win = _top_set(win_rate, final_cutoffs["win_rate"])
    final_ts = _top_set(trueskill, final_cutoffs["trueskill"])
    rows: list[dict[str, Any]] = []
    for strategy in population:
        initial_reasons = _admission_reasons(
            strategy,
            win_set=initial_win,
            trueskill_set=initial_ts,
            controls=controls,
            diagnostics=diagnostics,
        )
        final_reasons = _admission_reasons(
            strategy,
            win_set=final_win,
            trueskill_set=final_ts,
            controls=controls,
            diagnostics=diagnostics,
        )
        rows.append(
            {
                "strategy": strategy,
                "win_rate_rank": win_rate.ranks.get(strategy),
                "win_rate_score": win_rate.scores.get(strategy),
                "trueskill_rank": trueskill.ranks.get(strategy),
                "trueskill_mean_percentile": trueskill.scores.get(strategy),
                "scored_by_both_methods": strategy in win_rate.ranks
                and strategy in trueskill.ranks,
                "initial_win_rate_contribution": strategy in initial_win,
                "initial_trueskill_contribution": strategy in initial_ts,
                "initial_shared_contribution": strategy in initial_win and strategy in initial_ts,
                "configured_control": strategy in controls,
                "mandatory_diagnostic": strategy in diagnostics,
                "protected": strategy in controls or strategy in diagnostics,
                "initial_family": bool(initial_reasons),
                "final_win_rate_contribution": strategy in final_win,
                "final_trueskill_contribution": strategy in final_ts,
                "final_shared_contribution": strategy in final_win and strategy in final_ts,
                "final_family": strategy in family,
                "removed_by_cap": strategy in removal_round,
                "removal_round": removal_round.get(strategy),
                "cap_contraction_rounds": contraction_rounds,
                "initial_admission_reasons": json.dumps(initial_reasons),
                "final_admission_reasons": json.dumps(final_reasons),
                "family_hash": family_hash,
            }
        )
    frame = pd.DataFrame(rows)
    for column in ("win_rate_rank", "trueskill_rank", "removal_round"):
        frame[column] = frame[column].astype("Int64")
    return frame


def _default_win_rate_path(cfg: AppConfig, roots: tuple[int, ...]) -> Path:
    if len(roots) == 2:
        return cfg.root_combined_performance_across_k_path()
    return cfg.performance_across_k_path()


def freeze_h2h_candidate_family(
    cfg: AppConfig,
    *,
    win_rate_path: Path | None = None,
    trueskill_path: Path | None = None,
    force: bool = False,
) -> CandidateFamilyArtifacts:
    """Freeze one balanced union of canonical win-rate and TrueSkill contributions."""

    roots = _configured_roots(cfg)
    win_path = win_rate_path or _default_win_rate_path(cfg, roots)
    ts_path = trueskill_path or cfg.trueskill_candidate_contribution_path()
    sources = [win_path, ts_path]
    missing = [path for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"canonical candidate contributions are missing: {missing}")
    artifacts = CandidateFamilyArtifacts(
        membership=cfg.h2h_candidate_family_path(),
        manifest=cfg.h2h_candidate_family_manifest_path(),
    )
    done = stage_done_path(cfg.stage_dir("candidate_freeze"), "candidate_freeze")
    if not force and stage_is_up_to_date(
        done,
        inputs=sources,
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="candidate_freeze",
        sidecar_artifacts=list(artifacts.all_paths),
    ):
        return artifacts

    win_rate = _load_win_rate_contribution(win_path, roots)
    trueskill = _load_trueskill_contribution(ts_path, roots)
    controls = {_strategy_key(strategy) for strategy in cfg.screening.controls}
    diagnostics = {_strategy_key(strategy) for strategy in cfg.screening.mandatory_diagnostics}
    protected = controls | diagnostics
    scored_population = set(win_rate.ranks) | set(trueskill.ranks)
    missing_protected = sorted(protected.difference(scored_population))
    if missing_protected:
        raise ValueError(
            "protected H2H strategies are absent from both canonical contributions: "
            f"{missing_protected}"
        )

    family, initial_cutoffs, final_cutoffs, history, removal_round = _contract_family(
        win_rate=win_rate,
        trueskill=trueskill,
        contribution_size=cfg.screening.candidate_contribution_size,
        candidate_cap=cfg.head2head.candidate_cap,
        protected=protected,
    )
    initial_win = _top_set(win_rate, initial_cutoffs["win_rate"])
    initial_ts = _top_set(trueskill, initial_cutoffs["trueskill"])
    final_win = _top_set(win_rate, final_cutoffs["win_rate"])
    final_ts = _top_set(trueskill, final_cutoffs["trueskill"])
    candidates = sorted(family)
    pair_count = len(candidates) * (len(candidates) - 1) // 2
    source_identity = {
        "win_rate": {
            "artifact_name": win_rate.sidecar.artifact_name,
            "sha256": win_rate.sidecar.artifact_sha256,
        },
        "trueskill": {
            "artifact_name": trueskill.sidecar.artifact_name,
            "sha256": trueskill.sidecar.artifact_sha256,
            "sidecar_sha256": sha256_file(sidecar_path(ts_path)),
            "artifact_contract_version": trueskill.sidecar.artifact_contract_version,
            "estimand_version": trueskill.sidecar.estimand_version,
            "schema_version": trueskill.sidecar.schema_version,
            "rng_scheme_version": trueskill.sidecar.rng_scheme_version,
            "method_contract": trueskill.sidecar.method_contract,
            "conditioning": trueskill.sidecar.conditioning,
        },
    }
    identity = {
        "candidate_family_version": cfg.artifact_contract.candidate_family_version,
        "candidates": candidates,
        "source_identity": source_identity,
        "contribution_size": cfg.screening.candidate_contribution_size,
        "candidate_cap": cfg.head2head.candidate_cap,
        "candidate_cap_policy": cfg.head2head.candidate_cap_policy,
        "initial_cutoffs": initial_cutoffs,
        "final_cutoffs": final_cutoffs,
        "controls": sorted(controls),
        "mandatory_diagnostics": sorted(diagnostics),
    }
    family_hash = _family_hash(identity)
    admission_counts = {
        "win_rate_only": len((final_win - final_ts) & family),
        "trueskill_only": len((final_ts - final_win) & family),
        "shared_methods": len(final_win & final_ts & family),
        "protected_without_final_method_contribution": len(
            protected.difference(final_win | final_ts)
        ),
    }
    manifest: dict[str, Any] = {
        **identity,
        "family_hash": family_hash,
        "source_paths": {
            "win_rate": str(win_path),
            "trueskill": str(ts_path),
        },
        "candidate_count": len(candidates),
        "root_seeds": list(roots),
        "single_root_execution": len(roots) == 1,
        "cutoff_rounds": len(history) - 1,
        "cutoff_history": history,
        "initial_overlap": _overlap_summary(initial_win, initial_ts),
        "final_overlap": _overlap_summary(final_win, final_ts),
        "admission_counts": admission_counts,
        "projected_workload": {
            "unordered_pair_count": pair_count,
            "root_count": len(roots),
            "seat_order_blocks": pair_count * len(roots) * 2,
            "selfplay_root_blocks": len(candidates) * len(roots),
            "game_allocation_status": "pending_power_plan",
        },
        "interpretation": (
            "Finite-grid H2H family frozen before scheduling; source ranks are screening "
            "contributions, not inferential H2H order."
        ),
    }
    membership = _membership_frame(
        win_rate=win_rate,
        trueskill=trueskill,
        family=family,
        initial_cutoffs=initial_cutoffs,
        final_cutoffs=final_cutoffs,
        controls=controls,
        diagnostics=diagnostics,
        removal_round=removal_round,
        contraction_rounds=len(history) - 1,
        family_hash=family_hash,
    )
    seed_scope = "both_roots_combined" if len(roots) == 2 else "single_root"
    common: dict[str, Any] = {
        "producer": "candidate_family",
        "scope": ArtifactScope.H2H_2P,
        "source_scope": win_rate.sidecar.scope,
        "operation": "candidate_family_freeze",
        "weighted_quantity": "candidate_membership",
        "support_count_role": "canonical_source_rank",
        "uncertainty_method": "not_applicable",
        "replication_unit": "strategy_configuration",
        "conditioning": "finite_grid_screening_selection",
        "source_artifacts": sources,
        "grouping_keys": ["strategy"],
        "player_counts": [2],
        "required_player_counts": [2],
        "missing_cell_policy": "fail",
        "seed_scope": seed_scope,
    }
    membership_sidecar = make_artifact_sidecar(
        cfg,
        artifacts.membership,
        consistency_columns=membership.columns.tolist(),
        **common,
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(membership, preserve_index=False),
        artifacts.membership,
        sidecar=membership_sidecar,
        codec=cfg.parquet_codec,
    )
    manifest_sidecar = make_artifact_sidecar(
        cfg,
        artifacts.manifest,
        consistency_columns=list(manifest),
        **common,
    )
    write_json_artifact_atomic(
        manifest,
        artifacts.manifest,
        sidecar=manifest_sidecar,
    )
    write_stage_done(
        done,
        inputs=sources,
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="candidate_freeze",
        sidecar_artifacts=list(artifacts.all_paths),
    )
    return artifacts


__all__ = ["CandidateFamilyArtifacts", "freeze_h2h_candidate_family"]
