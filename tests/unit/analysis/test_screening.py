from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis import screening
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=5, n_players_list=[2, 4]),
    )
    cfg.screening.practical_delta_by_k = {2: 0.03, 4: 0.02}
    cfg.screening.delta_across_k = 0.025
    cfg.screening.candidate_contribution_size = 1
    cfg.screening.controls = [2]
    cfg.screening.mandatory_diagnostics = [1]
    return cfg


def _write_artifact(
    cfg: AppConfig,
    path: Path,
    frame: pd.DataFrame,
    *,
    scope: ArtifactScope,
    operation: str,
    player_counts: list[int],
    uncertainty_method: str,
) -> None:
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=scope,
        source_scope=ArtifactScope.BY_K,
        operation=operation,
        k_aggregation_method="equal_k" if scope is ArtifactScope.ACROSS_K else "none",
        uncertainty_method=uncertainty_method,
        conditioning="unconditional",
        player_counts=player_counts,
        required_player_counts=player_counts,
        missing_cell_policy="fail",
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(frame, preserve_index=False), path, sidecar=sidecar
    )


def _write_inputs(cfg: AppConfig) -> None:
    across = pd.DataFrame(
        {
            "root_seed": [5, 5],
            "strategy": [1, 2],
            "complete_support": [True, True],
            "equal_k_score": [0.10, 0.08],
            "pareto_member": [True, False],
            "maximin_leader": [True, False],
        }
    )
    _write_artifact(
        cfg,
        cfg.performance_across_k_path(),
        across,
        scope=ArtifactScope.ACROSS_K,
        operation="equal_k_mean",
        player_counts=[2, 4],
        uncertainty_method="independent_k_variance_sum",
    )
    bootstrap = pd.DataFrame(
        {
            "root_seed": [5, 5],
            "strategy": [1, 2],
            "top_n_inclusion_probability": [0.9, 0.1],
            "shortlist_inclusion_probability": [0.8, 0.4],
        }
    )
    _write_artifact(
        cfg,
        cfg.performance_bootstrap_path(),
        bootstrap,
        scope=ArtifactScope.ACROSS_K,
        operation="equal_k_mean",
        player_counts=[2, 4],
        uncertainty_method="joint_deterministic_batch_resampling",
    )
    for k, deltas, rates in (
        (2, [0.12, 0.10], [0.62, 0.60]),
        (4, [0.08, 0.03], [0.33, 0.28]),
    ):
        frame = pd.DataFrame(
            {
                "root_seed": [5, 5],
                "strategy": [1, 2],
                "chance_delta": deltas,
                "win_rate": rates,
                "raw_exposures": [1000, 1000],
            }
        )
        _write_artifact(
            cfg,
            cfg.performance_by_k_path(k),
            frame,
            scope=ArtifactScope.BY_K,
            operation="aggregate_screening_inputs",
            player_counts=[k],
            uncertainty_method="wilson_and_batch_t_interval",
        )


def test_screening_reports_evidence_without_inferential_tiers(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_inputs(cfg)

    screening.run(cfg)

    output = cfg.screening_path()
    frame = pq.read_table(output).to_pandas().set_index("strategy")
    assert frame.loc[1, "score_order_position"] == 1
    assert bool(frame.loc[1, "pareto_member"])
    assert bool(frame.loc[1, "maximin_leader"])
    assert bool(frame.loc[2, "declared_control"])
    assert bool(frame.loc[1, "mandatory_diagnostic"])
    assert bool(frame.loc[2, "within_across_k_practical_band"])
    assert not bool(frame.loc[2, "within_every_k_practical_band"])
    assert not any("tier" in column.lower() for column in frame.columns)

    report = json.loads(cfg.screening_path("descriptive_screening.json").read_text())
    assert "not tests of equality" in report["interpretation"]
    assert "final tiers" in report["interpretation"]
    validate_artifact_sidecar(
        output,
        expected={"scope": "across_k", "operation": "equal_k_mean"},
    )


def test_screening_rejects_incomplete_k_support(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_inputs(cfg)
    path = cfg.performance_across_k_path()
    frame = pq.read_table(path).to_pandas()
    frame.loc[frame["strategy"] == 2, "complete_support"] = False
    _write_artifact(
        cfg,
        path,
        frame,
        scope=ArtifactScope.ACROSS_K,
        operation="equal_k_mean",
        player_counts=[2, 4],
        uncertainty_method="independent_k_variance_sum",
    )

    try:
        screening.run(cfg)
    except ValueError as exc:
        assert "complete configured k support" in str(exc)
    else:
        raise AssertionError("incomplete k support was accepted")
