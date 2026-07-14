from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.stage_registry import resolve_root_pair_stage_layout
from farkle.analysis.structure_agreement import run
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2]),
    )
    cfg.set_stage_layout(resolve_root_pair_stage_layout(cfg))
    return cfg


def _write(cfg: AppConfig, path: Path, frame: pd.DataFrame, operation: str) -> None:
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation=operation,
        consistency_columns=frame.columns.tolist(),
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined",
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        sidecar=sidecar,
    )


def _inputs(cfg: AppConfig) -> None:
    membership = pd.DataFrame(
        {
            "strategy": ["A", "B", "C"],
            "win_rate_rank": [1, 2, 3],
            "trueskill_rank": [2, 1, 3],
            "scored_by_both_methods": [True, True, True],
            "initial_win_rate_contribution": [True, True, False],
            "initial_trueskill_contribution": [False, True, True],
            "final_win_rate_contribution": [True, True, False],
            "final_trueskill_contribution": [False, True, True],
            "final_shared_contribution": [False, True, False],
            "configured_control": [False, False, False],
            "mandatory_diagnostic": [False, False, False],
            "final_family": [True, True, True],
            "family_hash": ["f" * 64] * 3,
        }
    )
    _write(cfg, cfg.h2h_candidate_family_path(), membership, "candidate_family_freeze")
    inference = pd.DataFrame(
        {
            "family_hash": ["f" * 64] * 3,
            "pair_id": [0, 1, 2],
            "strategy_a": ["A", "A", "B"],
            "strategy_b": ["B", "C", "C"],
            "decision_class": ["practical_dominance_a", "unresolved", "practical_dominance_a"],
        }
    )
    _write(cfg, cfg.h2h_pairwise_inference_path(), inference, "seat_adjusted_score_inference")
    root_agreement = pd.DataFrame(
        {
            "pair_id": [0, 1, 2],
            "agreement_available": [True, True, True],
            "diagnostic_holm_decision_agreement": [True, False, True],
            "effect_direction_agreement": [True, True, True],
        }
    )
    _write(
        cfg,
        cfg.h2h_root_agreement_path(),
        root_agreement,
        "fixed_root_h2h_agreement_diagnostic",
    )


def test_agreement_reports_overlap_ranks_admissions_and_h2h_conditioning(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    _inputs(cfg)

    run(cfg)

    summary_path = cfg.structure_agreement_summary_path()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pairs = pq.read_table(cfg.structure_agreement_pairs_path()).to_pandas()
    assert summary["final_contribution_overlap"]["intersection_count"] == 1
    assert summary["final_contribution_overlap"]["jaccard"] == pytest.approx(1 / 3)
    assert summary["rank_agreement"]["spearman"] == pytest.approx(0.5)
    assert summary["admission_counts"]["final_family"] == 3
    assert summary["selection_conditioned_h2h"][
        "h2h_win_rate_direction_agreement_fraction"
    ] == pytest.approx(1.0)
    assert summary["selection_conditioned_h2h"][
        "h2h_trueskill_direction_agreement_fraction"
    ] == pytest.approx(0.5)
    assert summary["root_specific_h2h_stability"][
        "diagnostic_decision_agreement_fraction"
    ] == pytest.approx(2 / 3)
    assert set(pairs["selection_conditioning"]) == {"frozen_finite_grid_candidate_family"}
    for path in (summary_path, cfg.structure_agreement_pairs_path()):
        validate_artifact_sidecar(
            path,
            expected={
                "scope": "h2h_2p",
                "operation": "selection_conditioned_method_agreement",
            },
        )


def test_agreement_rejects_finalist_support_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _inputs(cfg)
    membership_path = cfg.h2h_candidate_family_path()
    membership = pq.read_table(membership_path).to_pandas()
    membership.loc[membership["strategy"].eq("C"), "final_family"] = False
    _write(cfg, membership_path, membership, "candidate_family_freeze")

    with pytest.raises(ValueError, match="finalist support"):
        run(cfg)
