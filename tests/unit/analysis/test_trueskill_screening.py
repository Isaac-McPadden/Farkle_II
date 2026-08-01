from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.trueskill_screening import (
    MU_SOFTMAX_HEURISTIC,
    MU_SOFTMAX_HEURISTIC_CLAIM,
    MU_SOFTMAX_HEURISTIC_OPERATION,
    TRUESKILL_CONDITIONING,
    ScreeningRatingCell,
    build_percentile_contribution,
    build_screening_diagnostics,
    classify_trueskill_row,
    diagnose_rating_cell,
    mu_softmax_heuristic_probabilities,
    publish_rating_cell_contract,
    trueskill_diagnostic_method_contract,
)
from farkle.config import AppConfig, IOConfig, SimConfig
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.schema_helpers import expected_schema_for
from farkle.utils.stage_completion import read_stage_done, stage_done_path


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 17], n_players_list=[2, 4]),
    )


def _ratings(path: Path, values: dict[str, tuple[float, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    strategy_count = len(values)
    updates = [1] * strategy_count
    for index in range(max(0, 4 - sum(updates))):
        updates[index % strategy_count] += 1
    pq.write_table(
        pa.table(
            {
                "strategy": pa.array([int(value) for value in values], type=pa.int32()),
                "mu": [values[key][0] for key in values],
                "sigma": [values[key][1] for key in values],
                "strategy_attempted_exposures": updates,
                "strategy_completed_exposures": updates,
                "strategy_excluded_safety_limit_exposures": [0] * strategy_count,
                "strategy_performed_updates": updates,
                "rating_status": ["evidence_backed_completed_games"] * strategy_count,
                "cell_games_attempted": [2] * strategy_count,
                "cell_games_completed": [2] * strategy_count,
                "cell_games_excluded_safety_limit": [0] * strategy_count,
                "cell_performed_updates": [2] * strategy_count,
            }
        ),
        path,
    )
    return path


def test_percentile_contribution_requires_complete_root_k_support(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cells: list[ScreeningRatingCell] = []
    for root in (11, 17):
        for k in (2, 4):
            values = {"1": (30.0 + root / 100, 2.0), "2": (20.0, 3.0)}
            if (root, k) == (11, 2):
                values["3"] = (40.0, 1.0)
            path = _ratings(tmp_path / f"ratings_{root}_{k}.parquet", values)
            cell = ScreeningRatingCell(root, k, path)
            publish_rating_cell_contract(
                cfg,
                cell,
                completed_artifact_sha256=sha256_file(path),
            )
            cells.append(cell)

    output = build_percentile_contribution(cfg, cells)
    frame = pq.read_table(output).to_pandas().set_index("strategy")

    assert frame.loc[1, "complete_support"]
    assert frame.loc[1, "rating_cells_present"] == 4
    assert frame.loc[1, "candidate_contribution_rank"] == 1
    assert frame.loc[1, "mean_percentile_rank"] == pytest.approx((2 / 3 + 1 + 1 + 1) / 4)
    assert frame.loc[2, "mean_percentile_rank"] == pytest.approx((1 / 3 + 0.5 + 0.5 + 0.5) / 4)
    assert 3 not in frame.index
    assert "sigma" not in frame.columns
    validate_artifact_sidecar(
        output,
        expected={
            "scope": "across_k",
            "operation": "equal_root_k_percentile_mean",
            "seed_scope": "both_roots_combined",
        },
    )
    publish_rating_cell_contract(
        cfg,
        cells[0],
        completed_artifact_sha256=sha256_file(cells[0].ratings_path),
    )
    validate_artifact_sidecar(
        cells[0].ratings_path,
        expected={
            "scope": "by_k",
            "operation": "sequential_rating",
            "uncertainty_method": "trueskill_model_sigma_screening_only",
        },
    )


def test_percentile_contribution_excludes_prior_only_rows(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11], n_players_list=[2, 4]),
    )
    cells: list[ScreeningRatingCell] = []
    for k in (2, 4):
        path = tmp_path / f"ratings_{k}.parquet"
        pq.write_table(
            pa.table(
                {
                    "strategy": pa.array([1, 2, 3], type=pa.int32()),
                    "mu": [30.0, 20.0, 25.0],
                    "sigma": [2.0, 3.0, 25.0 / 3.0],
                    "strategy_performed_updates": [2, 2, 0],
                    "strategy_attempted_exposures": [3, 2, 1],
                    "strategy_completed_exposures": [2, 2, 0],
                    "strategy_excluded_safety_limit_exposures": [1, 0, 1],
                    "rating_status": [
                        "evidence_backed_completed_games",
                        "evidence_backed_completed_games",
                        "prior_only_unrated",
                    ],
                    "cell_games_attempted": [3, 3, 3],
                    "cell_games_completed": [2, 2, 2],
                    "cell_games_excluded_safety_limit": [1, 1, 1],
                    "cell_performed_updates": [2, 2, 2],
                }
            ),
            path,
        )
        cell = ScreeningRatingCell(11, k, path)
        publish_rating_cell_contract(
            cfg,
            cell,
            completed_artifact_sha256=sha256_file(path),
        )
        cells.append(cell)

    contribution = pq.read_table(build_percentile_contribution(cfg, cells)).to_pandas()

    assert contribution["strategy"].tolist() == [1, 2]
    assert contribution["mean_percentile_rank"].tolist() == [1.0, 0.5]


def test_rating_cell_contract_does_not_repair_a_present_corrupt_sidecar(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    ratings = _ratings(tmp_path / "ratings.parquet", {"1": (30.0, 2.0), "2": (20.0, 3.0)})
    cell = ScreeningRatingCell(11, 2, ratings)
    with pytest.raises(ArtifactContractError, match="independent cell completion"):
        publish_rating_cell_contract(cfg, cell)
    publish_rating_cell_contract(
        cfg,
        cell,
        completed_artifact_sha256=sha256_file(ratings),
    )
    metadata = sidecar_path(ratings)
    original_bytes = metadata.read_bytes()
    corrupt_bytes = original_bytes.replace(b'"scope": "by_k"', b'"scope": "nope"')
    assert corrupt_bytes != original_bytes
    metadata.write_bytes(corrupt_bytes)

    with pytest.raises(ArtifactContractError):
        publish_rating_cell_contract(cfg, cell)

    assert metadata.read_bytes() == corrupt_bytes


def test_prechange_rating_schema_is_rejected_as_stale(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    ratings = tmp_path / "prechange_ratings.parquet"
    pq.write_table(
        pa.table(
            {
                "strategy": pa.array([1, 2], type=pa.int32()),
                "mu": [30.0, 20.0],
                "sigma": [2.0, 3.0],
            }
        ),
        ratings,
    )

    with pytest.raises(ArtifactContractError, match="rating support is missing"):
        publish_rating_cell_contract(
            cfg,
            ScreeningRatingCell(11, 2, ratings),
            completed_artifact_sha256=sha256_file(ratings),
        )


def _game_rows(path: Path, games: int = 10) -> Path:
    rows: list[dict[str, object]] = []
    for game in range(games):
        p1_wins = game % 2 == 0
        rows.append(
            {
                "root_seed": 11,
                "k": 2,
                "shuffle_index": game,
                "game_index": game,
                "deterministic_batch_id": 0,
                "termination_status": "completed",
                "hit_safety_limit": False,
                "outcome_schema_version": 2,
                "winner_seat": "P1" if p1_wins else "P2",
                "winner_strategy": 1 if p1_wins else 2,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_rank": 1 if p1_wins else 2,
                "P2_rank": 2 if p1_wins else 1,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=expected_schema_for(2)), path)
    return path


def test_mu_softmax_heuristic_probability_assumptions_are_explicit() -> None:
    equal_mu = mu_softmax_heuristic_probabilities(
        [(25.0, 1.0), (25.0, 10.0)],
        beta=5.0,
    )
    different_mu = mu_softmax_heuristic_probabilities(
        [(30.0, 4.0), (20.0, 4.0)],
        beta=5.0,
    )

    assert equal_mu.tolist() == pytest.approx([0.5, 0.5])
    assert different_mu.tolist() == pytest.approx(
        [1.0 / (1.0 + math.exp(-2.0)), 1.0 / (1.0 + math.exp(2.0))]
    )


def test_mu_softmax_heuristic_claim_and_method_contract_are_exact() -> None:
    expected_claim = (
        "Held-out descriptive scores use mu_softmax_heuristic probabilities computed "
        "as softmax(mu / beta). TrueSkill sigma is ignored; these are heuristic "
        "probabilities, not TrueSkill predictive probabilities."
    )

    assert MU_SOFTMAX_HEURISTIC == "mu_softmax_heuristic"
    assert expected_claim == MU_SOFTMAX_HEURISTIC_CLAIM
    assert trueskill_diagnostic_method_contract().get("parameters") == {
        "method_version": 3,
        "outcome_schema_version": 2,
        "conditioning": "termination_status == completed",
        "safety_limit_policy": "excluded_without_update_or_rank_imputation",
        "diagnostic_method_version": 1,
        "heldout_probability_method": "mu_softmax_heuristic",
        "heldout_probability_formula": "softmax(mu / beta)",
        "heldout_probability_sigma_policy": "ignored",
        "heldout_fraction": 0.2,
        "heldout_rating_policy": "freeze_after_chronological_training_prefix",
        "heldout_target": "unique_rank_1_completed_game_winner",
        "interpretation": expected_claim,
    }


def test_tau_order_and_heldout_diagnostics(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    rating_path = _ratings(
        tmp_path / "ratings.parquet",
        {"1": (25.0, 2.0), "2": (25.0, 2.0)},
    )
    game_path = _game_rows(tmp_path / "games.parquet")
    cell = ScreeningRatingCell(11, 2, rating_path, game_path)

    row = diagnose_rating_cell(
        cell,
        beta=cfg.trueskill.beta,
        tau=cfg.trueskill.tau,
        draw_probability=cfg.trueskill.draw_probability,
    )
    assert row["tau_zero_games"] == 10
    assert row["reversed_order_games"] == 10
    assert row["mu_softmax_heuristic_holdout_games"] == 2
    assert row["mu_softmax_heuristic_heldout_log_loss"] is not None
    assert row["mu_softmax_heuristic_heldout_brier_score"] is not None
    assert row["mu_softmax_heuristic_uniform_reference_log_loss"] == pytest.approx(math.log(2))
    assert row["mu_softmax_heuristic_claim"] == MU_SOFTMAX_HEURISTIC_CLAIM

    output = build_screening_diagnostics(cfg, [cell])
    assert output is not None
    diagnostics = pq.read_table(output).to_pandas()
    assert diagnostics.loc[0, "mu_softmax_heuristic_holdout_games"] == 2
    validate_artifact_sidecar(
        output,
        expected={
            "scope": "diagnostics",
            "operation": MU_SOFTMAX_HEURISTIC_OPERATION,
            "weighted_quantity": (
                "trueskill_screening_sensitivity_and_mu_softmax_heuristic_scores"
            ),
            "uncertainty_method": "descriptive_replay_and_mu_softmax_heuristic_scoring",
            "conditioning": TRUESKILL_CONDITIONING,
            "method_contract": trueskill_diagnostic_method_contract(),
        },
    )
    completion = read_stage_done(
        stage_done_path(cfg.trueskill_stage_dir, "trueskill_screening_diagnostics")
    )
    freshness_key = completion["freshness_key"]
    assert isinstance(freshness_key, dict)
    assert freshness_key["trueskill_diagnostic_method_version"] == 1


def test_diagnostics_report_mixed_support_and_prior_only_strategy(tmp_path: Path) -> None:
    rating_path = tmp_path / "ratings.parquet"
    pq.write_table(
        pa.table(
            {
                    "strategy": pa.array([1, 2, 3], type=pa.int32()),
                "mu": [30.0, 20.0, 25.0],
                "sigma": [5.0, 5.0, 25.0 / 3.0],
                "strategy_performed_updates": [1, 1, 0],
                "strategy_attempted_exposures": [2, 1, 1],
                "strategy_completed_exposures": [1, 1, 0],
                "strategy_excluded_safety_limit_exposures": [1, 0, 1],
                "rating_status": [
                    "evidence_backed_completed_games",
                    "evidence_backed_completed_games",
                    "prior_only_unrated",
                ],
                "cell_games_attempted": [2, 2, 2],
                "cell_games_completed": [1, 1, 1],
                "cell_games_excluded_safety_limit": [1, 1, 1],
                "cell_performed_updates": [1, 1, 1],
            }
        ),
        rating_path,
    )
    game_path = tmp_path / "games.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "termination_status": "completed",
                    "outcome_schema_version": 2,
                    "winner_seat": "P1",
                    "P1_strategy": 1,
                    "P2_strategy": 2,
                    "P1_rank": 1,
                    "P2_rank": 2,
                },
                {
                    "termination_status": "safety_limit",
                    "outcome_schema_version": 2,
                    "winner_seat": None,
                    "P1_strategy": 1,
                    "P2_strategy": 3,
                    "P1_rank": None,
                    "P2_rank": None,
                },
            ]
        ),
        game_path,
    )

    row = diagnose_rating_cell(
        ScreeningRatingCell(11, 2, rating_path, game_path),
        beta=25.0,
        tau=0.1,
        draw_probability=0.0,
    )

    assert row["attempted_games"] == 2
    assert row["completed_games"] == 1
    assert row["excluded_safety_limit_games"] == 1
    assert row["performed_update_games"] == 1
    assert row["prior_only_strategy_count"] == 1
    assert row["prior_only_strategies"] == "3"


@pytest.mark.parametrize("malformed", [None, "1", 1.0, True])
def test_trueskill_row_rejects_noncanonical_strategy_ids(malformed: object) -> None:
    row: dict[str, object] = {
        "termination_status": "completed",
        "outcome_schema_version": 2,
        "winner_seat": "P1",
        "P1_strategy": malformed,
        "P2_strategy": 2,
        "P1_rank": 1,
        "P2_rank": 2,
    }

    with pytest.raises(ValueError, match="P1_strategy"):
        classify_trueskill_row(row, 2)
