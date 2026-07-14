from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.candidate_family import freeze_h2h_candidate_family
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    make_artifact_sidecar,
    validate_artifact_sidecar,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic


def _cfg(tmp_path: Path, *, cap: int | None = 5) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(
            seed=11,
            seed_list=[11, 22],
            seed_pair=(11, 22),
            n_players_list=[2, 4],
        ),
    )
    cfg.screening.candidate_contribution_size = 3
    cfg.screening.controls = [8]
    cfg.screening.mandatory_diagnostics = [7]
    cfg.head2head.candidate_cap = cap
    return cfg


def _write_frame(
    cfg: AppConfig,
    path: Path,
    frame: pd.DataFrame,
    *,
    scope: ArtifactScope,
    operation: str,
    seed_scope: str,
) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=scope,
        source_scope=scope,
        operation=operation,
        consistency_columns=table.schema.names,
        player_counts=[2, 4],
        required_player_counts=[2, 4],
        missing_cell_policy="fail",
        seed_scope=seed_scope,
    )
    write_parquet_artifact_atomic(table, path, sidecar=sidecar)


def _write_inputs(cfg: AppConfig, *, reverse_rows: bool = False) -> None:
    win_rows = [
        {
            "estimate_scope": "combined_roots",
            "strategy": strategy,
            "across_k_score": score,
            "complete_support": True,
        }
        for strategy, score in zip(
            range(1, 9),
            (0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2),
            strict=True,
        )
    ]
    ts_order = (3, 4, 5, 1, 2, 6, 7, 8)
    ts_rows = [
        {
            "strategy": strategy,
            "mean_percentile_rank": 1.0 - (rank - 1) / 10,
            "candidate_contribution_rank": rank,
            "complete_support": True,
        }
        for rank, strategy in enumerate(ts_order, 1)
    ]
    if reverse_rows:
        win_rows.reverse()
        ts_rows.reverse()
    _write_frame(
        cfg,
        cfg.root_combined_performance_across_k_path(),
        pd.DataFrame(win_rows),
        scope=ArtifactScope.CROSS_SEED,
        operation="equal_k_mean",
        seed_scope="root_pair_stability",
    )
    _write_frame(
        cfg,
        cfg.trueskill_candidate_contribution_path(),
        pd.DataFrame(ts_rows),
        scope=ArtifactScope.ACROSS_K,
        operation="equal_root_k_percentile_mean",
        seed_scope="both_roots_combined",
    )


def test_candidate_family_balanced_tail_contraction_and_provenance(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_inputs(cfg)

    artifacts = freeze_h2h_candidate_family(cfg)

    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["candidates"] == ["1", "3", "7", "8"]
    assert manifest["candidate_count"] == 4
    assert manifest["initial_cutoffs"] == {"win_rate": 3, "trueskill": 3}
    assert manifest["final_cutoffs"] == {"win_rate": 1, "trueskill": 1}
    assert manifest["cutoff_rounds"] == 2
    assert manifest["cutoff_history"][1]["removed"] == ["5"]
    assert manifest["cutoff_history"][2]["removed"] == ["2", "4"]
    assert manifest["initial_overlap"]["intersection_count"] == 1
    assert manifest["projected_workload"] == {
        "game_allocation_status": "pending_power_plan",
        "root_count": 2,
        "seat_order_blocks": 24,
        "selfplay_root_blocks": 8,
        "unordered_pair_count": 6,
    }
    assert len(manifest["family_hash"]) == 64

    membership = pq.read_table(artifacts.membership).to_pandas().set_index("strategy")
    assert membership.index.tolist() == [str(strategy) for strategy in range(1, 9)]
    assert int(cast(int, membership.loc["1", "win_rate_rank"])) == 1
    assert int(cast(int, membership.loc["3", "trueskill_rank"])) == 1
    assert int(cast(int, membership.loc["5", "removal_round"])) == 1
    assert int(cast(int, membership.loc["2", "removal_round"])) == 2
    assert bool(membership.loc["7", "protected"])
    assert bool(membership.loc["7", "final_family"])
    assert json.loads(cast(str, membership.loc["7", "final_admission_reasons"])) == [
        "mandatory_diagnostic"
    ]
    assert not bool(membership.loc["6", "initial_family"])
    assert membership["family_hash"].nunique() == 1
    assert membership["family_hash"].iloc[0] == manifest["family_hash"]

    for path in artifacts.all_paths:
        validate_artifact_sidecar(
            path,
            expected={
                "scope": "h2h_2p",
                "operation": "candidate_family_freeze",
                "seed_scope": "both_roots_combined",
            },
        )

    freeze_h2h_candidate_family(cfg, force=True)
    replay = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert replay["family_hash"] == manifest["family_hash"]


def test_candidate_family_without_cap_keeps_complete_declared_union(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, cap=None)
    _write_inputs(cfg, reverse_rows=True)

    artifacts = freeze_h2h_candidate_family(cfg)
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))

    assert manifest["candidates"] == ["1", "2", "3", "4", "5", "7", "8"]
    assert manifest["cutoff_rounds"] == 0
    assert manifest["final_cutoffs"] == manifest["initial_cutoffs"]


def test_candidate_family_rejects_missing_or_over_cap_protected_set(tmp_path: Path) -> None:
    missing_cfg = _cfg(tmp_path / "missing")
    missing_cfg.screening.controls = [99]
    _write_inputs(missing_cfg)

    with pytest.raises(ValueError, match="absent from both canonical contributions"):
        freeze_h2h_candidate_family(missing_cfg)

    cap_cfg = _cfg(tmp_path / "cap", cap=2)
    cap_cfg.screening.controls = [6, 8]
    cap_cfg.screening.mandatory_diagnostics = [7]
    _write_inputs(cap_cfg)

    with pytest.raises(ValueError, match="smaller than the protected"):
        freeze_h2h_candidate_family(cap_cfg)


def test_two_root_family_rejects_single_root_win_rate_scope(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_inputs(cfg)
    win_path = cfg.root_combined_performance_across_k_path()
    frame = pq.read_table(win_path).to_pandas()
    _write_frame(
        cfg,
        win_path,
        frame,
        scope=ArtifactScope.ACROSS_K,
        operation="equal_k_mean",
        seed_scope="single_root",
    )

    with pytest.raises(ArtifactContractError, match="cross_seed"):
        freeze_h2h_candidate_family(cfg)


def test_candidate_family_supports_explicit_single_root_label(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11], n_players_list=[2, 4]),
    )
    cfg.screening.candidate_contribution_size = 2
    win = pd.DataFrame(
        {
            "strategy": [1, 2, 3],
            "equal_k_score": [0.3, 0.2, 0.1],
            "complete_support": [True, True, True],
        }
    )
    ts = pd.DataFrame(
        {
            "strategy": [2, 3, 1],
            "mean_percentile_rank": [1.0, 0.8, 0.6],
            "candidate_contribution_rank": [1, 2, 3],
            "complete_support": [True, True, True],
        }
    )
    _write_frame(
        cfg,
        cfg.performance_across_k_path(),
        win,
        scope=ArtifactScope.ACROSS_K,
        operation="equal_k_mean",
        seed_scope="single_root",
    )
    _write_frame(
        cfg,
        cfg.trueskill_candidate_contribution_path(),
        ts,
        scope=ArtifactScope.ACROSS_K,
        operation="equal_root_k_percentile_mean",
        seed_scope="single_root",
    )

    artifacts = freeze_h2h_candidate_family(cfg)
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))

    assert manifest["single_root_execution"] is True
    assert manifest["root_seeds"] == [11]
    validate_artifact_sidecar(
        artifacts.manifest,
        expected={"seed_scope": "single_root", "scope": "h2h_2p"},
    )
