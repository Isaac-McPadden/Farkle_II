from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.seat_analysis import build_canonical_seat_analysis
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    make_artifact_sidecar,
    validate_artifact_sidecar,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.schema_helpers import expected_schema_for


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, n_players_list=[2, 4]),
    )


def _row(k: int, game: int, strategies: list[int], winner_seat: int) -> dict[str, object]:
    row: dict[str, object] = {
        "root_seed": 11,
        "k": k,
        "shuffle_index": game,
        "game_index": game,
        "deterministic_batch_id": 0,
        "winner_seat": f"P{winner_seat}",
        "winner_strategy": strategies[winner_seat - 1],
        "termination_status": "completed",
        "outcome_schema_version": 2,
    }
    row.update({f"P{seat}_strategy": strategy for seat, strategy in enumerate(strategies, 1)})
    return row


def _write_source(
    cfg: AppConfig,
    k: int,
    rows: list[dict[str, object]],
    *,
    operation: str = "concatenate_rows_within_k",
) -> Path:
    path = cfg.combined_rows_by_k(k)
    table = pa.Table.from_pylist(rows, schema=expected_schema_for(k))
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation=operation,
        consistency_columns=table.schema.names,
        player_counts=[k],
        required_player_counts=[k],
        missing_cell_policy="fail",
    )
    write_parquet_artifact_atomic(table, path, sidecar=sidecar)
    return path


def _write_inputs(cfg: AppConfig) -> None:
    _write_source(
        cfg,
        2,
        [
            _row(2, 0, [10, 20], 1),
            _row(2, 1, [20, 10], 2),
            _row(2, 2, [10, 10], 1),
            _row(2, 3, [30, 10], 2),
        ],
    )
    _write_source(
        cfg,
        4,
        [
            _row(4, 0, [10, 20, 10, 20], 2),
            _row(4, 1, [20, 10, 20, 10], 1),
        ],
    )


def test_canonical_seat_estimators_and_diagnostics(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_inputs(cfg)

    artifacts = build_canonical_seat_analysis(cfg)

    counts_2 = pq.read_table(artifacts.batch_counts[0]).to_pandas()
    cell = counts_2.loc[(counts_2["strategy"] == 10) & (counts_2["seat"] == 1)].iloc[0]
    assert cell["raw_wins"] == 2
    assert cell["raw_exposures"] == 2

    effects_2 = pq.read_table(artifacts.by_k[0]).to_pandas()
    cell = effects_2.loc[(effects_2["strategy"] == 10) & (effects_2["seat"] == 1)].iloc[0]
    assert cell["chance_baseline"] == pytest.approx(0.5)
    assert cell["seat_effect"] == pytest.approx(0.5)

    standardized = pq.read_table(artifacts.standardized_across_k).to_pandas()
    strategy_cell = standardized.loc[
        (standardized["effect_scope"] == "strategy")
        & (standardized["strategy"] == 10)
        & (standardized["seat"] == 1)
    ].iloc[0]
    assert strategy_cell["common_k_support"].tolist() == [2, 4]
    assert strategy_cell["standardized_seat_effect"] == pytest.approx(0.125)
    assert 30 not in standardized["strategy"].dropna().tolist()
    population_cell = standardized.loc[
        (standardized["effect_scope"] == "population") & (standardized["seat"] == 1)
    ].iloc[0]
    assert population_cell["standardized_seat_effect"] == pytest.approx(0.125)

    mixture = pq.read_table(artifacts.exposure_mixture_diagnostic).to_pandas()
    mixture_cell = mixture.loc[
        (mixture["effect_scope"] == "strategy")
        & (mixture["strategy"] == 10)
        & (mixture["seat"] == 1)
    ].iloc[0]
    assert mixture_cell["common_k_support"].tolist() == [2, 4]
    assert mixture_cell["exposure_weighted_baseline"] == pytest.approx(1.25 / 3)
    assert mixture_cell["exposure_weighted_seat_effect"] == pytest.approx(0.25)

    selfplay = pq.read_table(artifacts.selfplay_diagnostic).to_pandas()
    assert selfplay.loc[0, "p1_effect_vs_chance"] == pytest.approx(0.5)
    mirrored = pq.read_table(artifacts.mirrored_diagnostic).to_pandas()
    assert mirrored.loc[0, "paired_mirrored_games"] == 1
    assert mirrored.loc[0, "mean_p1_win_difference"] == pytest.approx(1.0)

    validate_artifact_sidecar(
        artifacts.standardized_across_k,
        expected={
            "scope": "across_k",
            "operation": "equal_k_mean",
            "missing_cell_policy": "declared_common_support",
        },
    )
    validate_artifact_sidecar(
        artifacts.exposure_mixture_diagnostic,
        expected={
            "scope": "diagnostics",
            "operation": "within_k_exposure_combination",
        },
    )


def test_declared_k_weights_are_used_for_standardization(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.k_aggregation.method = "declared-mapping"
    cfg.k_aggregation.k_weights = {2: 0.25, 4: 0.75}
    _write_inputs(cfg)

    artifacts = build_canonical_seat_analysis(cfg)
    standardized = pq.read_table(artifacts.standardized_across_k).to_pandas()
    cell = standardized.loc[
        (standardized["effect_scope"] == "strategy")
        & (standardized["strategy"] == 10)
        & (standardized["seat"] == 1)
    ].iloc[0]
    assert cell["standardized_seat_effect"] == pytest.approx(-0.0625)
    validate_artifact_sidecar(
        artifacts.standardized_across_k,
        expected={
            "operation": "declared_k_weighted_mean",
            "k_aggregation_method": "declared_mapping",
            "k_weights": {"2": 0.25, "4": 0.75},
        },
    )


def test_incompatible_input_scope_fails_before_estimation(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_inputs(cfg)
    _write_source(cfg, 4, [_row(4, 0, [10, 20, 10, 20], 2)], operation="concatenate")

    with pytest.raises(ArtifactContractError, match="operation"):
        build_canonical_seat_analysis(cfg)


def test_mismatched_root_support_fails_before_estimation(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_inputs(cfg)
    row = _row(4, 0, [10, 20, 10, 20], 2)
    row["root_seed"] = 12
    _write_source(cfg, 4, [row])

    with pytest.raises(ValueError, match="configured-root support"):
        build_canonical_seat_analysis(cfg)
