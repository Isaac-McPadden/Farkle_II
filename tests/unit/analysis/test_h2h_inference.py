from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.stats import norm
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
)
from statsmodels.stats.proportion import (
    test_proportions_2indep as statsmodels_score_test,
)
from tests.helpers.artifact_sidecars import clean_test_code_identity
from tests.helpers.h2h_oracle import execute_oracle, plan_oracle
from tests.unit.analysis.test_h2h_schedule import _write_frozen_family

from farkle.analysis import h2h_schedule
from farkle.analysis.h2h_inference import (
    run_h2h_inference,
    score_difference_interval,
    two_proportion_score_test,
)
from farkle.analysis.h2h_schedule import SCORE_TEST_ID
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import (
    write_json_artifact_atomic,
    write_parquet_artifact_atomic,
)
from farkle.utils.scientific_runtime import numerical_observation


def _cfg(tmp_path: Path, *, roots: tuple[int, ...] = (11, 22)) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(
            seed=roots[0],
            seed_list=list(roots),
            n_players_list=[2],
        ),
    )
    cfg.screening.practical_delta_by_k = {2: 0.03}
    cfg.screening.delta_across_k = 0.03
    cfg._code_identity = clean_test_code_identity()
    return cfg


@numerical_observation()
def _publish_inputs(
    cfg: AppConfig,
    pairs: list[tuple[str, str, int, int, int]],
    *,
    cell_overrides: dict[tuple[int, int, int], dict[str, object]] | None = None,
) -> None:
    """Publish pairs as (A, B, games/order/root, x_ab/root, x_ba/root)."""

    candidates = sorted({int(sid) for a, b, *_ in pairs for sid in (a, b)})
    outcomes = {(int(a), int(b)): (games, x_ab, x_ba) for a, b, games, x_ab, x_ba in pairs}
    block_games = pairs[0][2]
    assert all(games == block_games for games, _, _ in outcomes.values())
    _write_frozen_family(cfg, tuple(candidates))
    with (
        patch.object(h2h_schedule, "_minimum_block_games", return_value=block_games),
        patch.object(h2h_schedule, "_worst_scenario_power", return_value=0.81),
        patch.object(h2h_schedule, "implemented_score_test_power", return_value=0.81),
    ):
        plan_oracle(cfg)

    def fixed_counts(block: dict[str, Any], _path: Path, _limit: int) -> dict[str, Any]:
        order = block["order"]
        games, x_ab, x_ba = outcomes[block["strategy_a"], block["strategy_b"]]
        wins = x_ab if order == 0 else x_ba
        return {
            **block,
            "games_attempted": games,
            "games_completed": games,
            "wins_seat1": wins,
            "wins_seat2": games - wins,
            **(cell_overrides or {}).get((block["pair_id"], block["root_seed"], order), {}),
        }

    execute_oracle(cfg, block_runner=fixed_counts, n_jobs=1, chunk_games=2 * block_games)


def _rewrite_execution_state(cfg: AppConfig, **updates: object) -> None:
    path = cfg.h2h_execution_state_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    roots = cfg.sim.seed_list or [cfg.sim.seed]
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="h2h_execution_state",
        uncertainty_method=SCORE_TEST_ID,
        consistency_columns=list(payload),
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined" if len(roots) == 2 else "single_root",
    )
    write_json_artifact_atomic(payload, path, sidecar=sidecar)


def test_score_test_uses_constrained_null_proportion() -> None:
    result = two_proportion_score_test(60, 100, 40, 100)
    expected_z = 0.2 / ((0.5 * 0.5 * (1 / 100 + 1 / 100)) ** 0.5)

    assert result.difference == pytest.approx(0.2)
    assert result.null_proportion == pytest.approx(0.5)
    assert result.statistic == pytest.approx(expected_z)
    assert result.p_value == pytest.approx(2 * norm.sf(abs(expected_z)))


def test_inference_rejects_partial_execution_state(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _publish_inputs(cfg, [("1", "2", 100, 55, 45)])
    _rewrite_execution_state(cfg, execution_state="partial_resumable")

    with pytest.raises(RuntimeError, match="complete valid H2H execution"):
        run_h2h_inference(cfg)


def test_inference_rejects_execution_schedule_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _publish_inputs(cfg, [("1", "2", 100, 55, 45)])
    _rewrite_execution_state(cfg, schedule_hash="e" * 64)

    with pytest.raises(ValueError, match="power-plan schedule_hash"):
        run_h2h_inference(cfg)


def test_score_interval_contains_observed_difference() -> None:
    low, high = score_difference_interval(60, 100, 40, 100, alpha=0.02)

    assert low < 0.2 < high
    assert low > -1.0
    assert high < 1.0


def test_score_test_and_interval_match_statsmodels_oracles() -> None:
    observed = two_proportion_score_test(73, 120, 51, 115)
    expected = cast(
        Any,
        statsmodels_score_test(
            73,
            120,
            51,
            115,
            value=0,
            method="score",
            compare="diff",
            alternative="two-sided",
            correction=False,
            return_results=True,
        ),
    )
    expected_interval = confint_proportions_2indep(
        73,
        120,
        51,
        115,
        method="score",
        compare="diff",
        alpha=0.02,
        correction=False,
    )

    assert observed.statistic == pytest.approx(expected.statistic)
    assert observed.p_value == pytest.approx(expected.pvalue)
    assert score_difference_interval(73, 120, 51, 115, alpha=0.02) == pytest.approx(
        expected_interval
    )


def test_score_interval_completes_boundary_outcomes_symmetrically() -> None:
    negative = score_difference_interval(0, 100, 100, 100, alpha=0.05)
    positive = score_difference_interval(100, 100, 0, 100, alpha=0.05)
    all_success = score_difference_interval(100, 100, 100, 100, alpha=0.05)
    all_failure = score_difference_interval(0, 100, 0, 100, alpha=0.05)

    assert negative[0] == -1.0
    assert -1.0 < negative[1] < 0.0
    assert positive == pytest.approx((-negative[1], -negative[0]))
    assert all_success[0] == pytest.approx(-all_success[1])
    assert all_failure[0] == pytest.approx(-all_failure[1])
    assert all_success == pytest.approx(all_failure)


def test_score_interval_rebrackets_tiny_alpha_without_changing_the_test() -> None:
    alpha = 0.05 / 2_926
    observed = 2_209 / 3_966 - 2_019 / 3_966

    low, high = score_difference_interval(2_209, 3_966, 2_019, 3_966, alpha=alpha)

    assert math.isfinite(low)
    assert math.isfinite(high)
    assert low < observed < high


def test_h2h_inference_accepts_complete_boundary_counts(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _publish_inputs(
        cfg,
        [
            ("1", "2", 100, 0, 100),
            ("1", "3", 100, 100, 100),
            ("2", "3", 100, 0, 0),
        ],
    )

    artifacts = run_h2h_inference(cfg)

    inference = pq.read_table(artifacts.pairwise_inference).to_pandas()
    diagnostics = pq.read_table(artifacts.root_pairwise_diagnostics).to_pandas()
    assert len(inference) == 3
    assert len(diagnostics) == 6
    assert np.isfinite(inference[["ordinary_d_low", "ordinary_d_high"]]).all().all()
    assert np.isfinite(diagnostics[["ordinary_d_low", "ordinary_d_high"]]).all().all()


def test_seat_adjusted_inference_holm_and_decision_classes(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _publish_inputs(
        cfg,
        [
            ("1", "2", 5_000, 3_000, 2_000),
            ("1", "3", 5_000, 2_600, 2_500),
            ("2", "3", 5_000, 2_500, 2_500),
        ],
    )

    artifacts = run_h2h_inference(cfg)

    combined = pq.read_table(artifacts.combined_order_counts).to_pandas()
    assert len(combined) == 6
    assert combined["root_count"].eq(2).all()
    first_ab = combined.loc[(combined["pair_id"] == 0) & (combined["order"] == 0)].iloc[0]
    assert first_ab["games_completed"] == 10_000
    assert first_ab["wins_seat1"] == 6_000

    inference = pq.read_table(artifacts.pairwise_inference).to_pandas().set_index("pair_id")
    assert inference.loc[0, "q_ab"] == pytest.approx(0.6)
    assert inference.loc[0, "q_ba"] == pytest.approx(0.4)
    assert inference.loc[0, "d_ab"] == pytest.approx(0.1)
    assert inference.loc[0, "balanced_a_win_rate_alias"] == pytest.approx(0.6)
    assert bool(inference.loc[0, "balanced_alias_checked"])
    assert inference.loc[0, "decision_class"] == "practical_dominance_a"
    assert inference.loc[1, "decision_class"] == "statistical_only_advantage_a"
    assert inference.loc[2, "decision_class"] == "unresolved"
    assert bool(inference.loc[0, "holm_reject"])
    assert bool(inference.loc[1, "holm_reject"])
    assert not bool(inference.loc[2, "holm_reject"])
    assert not inference["equivalence_enabled"].any()
    assert (inference["simultaneous_d_low"] <= inference["ordinary_d_low"]).all()
    assert (inference["simultaneous_d_high"] >= inference["ordinary_d_high"]).all()

    root_diagnostics = pq.read_table(artifacts.root_pairwise_diagnostics).to_pandas()
    assert len(root_diagnostics) == 6
    assert set(root_diagnostics["root_seed"]) == {11, 22}
    assert set(root_diagnostics["inference_role"]) == {"fixed_root_diagnostic_not_root_population"}
    assert (root_diagnostics["ordinary_d_low"] <= root_diagnostics["d_ab"]).all()
    assert (root_diagnostics["d_ab"] <= root_diagnostics["ordinary_d_high"]).all()
    agreement = pq.read_table(artifacts.root_agreement).to_pandas()
    assert len(agreement) == 3
    assert agreement["agreement_available"].all()
    assert agreement["diagnostic_holm_decision_agreement"].all()
    assert agreement["absolute_effect_discrepancy"].eq(0.0).all()
    assert set(agreement["interpretation"]) == {
        "fixed_root_reproducibility_diagnostic_not_population_inference"
    }

    validate_artifact_sidecar(
        artifacts.pairwise_inference,
        expected={
            "scope": "h2h_2p",
            "operation": "seat_adjusted_score_inference",
            "uncertainty_method": f"{SCORE_TEST_ID}_holm",
        },
    )
    validate_artifact_sidecar(
        artifacts.root_agreement,
        expected={
            "scope": "h2h_2p",
            "operation": "fixed_root_h2h_agreement_diagnostic",
            "uncertainty_method": "descriptive_fixed_root_decision_comparison",
        },
    )


@pytest.mark.parametrize("seat1_advantage", [0.03, 0.06])
def test_common_first_seat_effect_does_not_create_strategy_effect(
    tmp_path: Path,
    seat1_advantage: float,
) -> None:
    cfg = _cfg(tmp_path)
    games = 10_000
    common_wins = int(round((0.5 + seat1_advantage) * games))
    _publish_inputs(cfg, [("1", "2", games, common_wins, common_wins)])

    artifacts = run_h2h_inference(cfg)
    primary = pq.read_table(artifacts.pairwise_inference).to_pandas().iloc[0]
    root_diagnostics = pq.read_table(artifacts.root_pairwise_diagnostics).to_pandas()

    assert primary["q_ab"] == pytest.approx(0.5 + seat1_advantage)
    assert primary["q_ba"] == pytest.approx(0.5 + seat1_advantage)
    assert primary["d_ab"] == pytest.approx(0.0)
    assert primary["score_p_value"] == pytest.approx(1.0)
    assert primary["decision_class"] == "unresolved"
    assert not root_diagnostics["holm_reject"].any()


def test_single_root_diagnostics_are_explicitly_labeled(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, roots=(11,))
    _publish_inputs(cfg, [("1", "2", 5_000, 2_700, 2_400)])

    artifacts = run_h2h_inference(cfg)
    root_diagnostics = pq.read_table(artifacts.root_pairwise_diagnostics).to_pandas()
    agreement = pq.read_table(artifacts.root_agreement).to_pandas().iloc[0]

    assert root_diagnostics["root_seed"].tolist() == [11]
    assert not bool(agreement["agreement_available"])
    assert agreement["interpretation"] == "single_root_diagnostic_no_cross_root_stability_claim"
    validate_artifact_sidecar(
        artifacts.root_agreement,
        expected={"scope": "h2h_2p", "seed_scope": "single_root"},
    )


def test_equivalence_requires_explicit_margin(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.head2head.delta_equivalence = 0.02
    _publish_inputs(cfg, [("1", "2", 50_000, 25_000, 25_000)])

    artifacts = run_h2h_inference(cfg)
    inference = pq.read_table(artifacts.pairwise_inference).to_pandas().iloc[0]

    assert bool(inference["equivalence_enabled"])
    assert inference["decision_class"] == "equivalent"
    assert inference["simultaneous_d_low"] > -0.02
    assert inference["simultaneous_d_high"] < 0.02


def test_mixed_completed_and_safety_attempts_use_completed_score_counts_and_actual_a_wins(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    overrides = {
        (0, root, order): {
            "games_attempted": 101,
            "games_completed": 100,
            "games_safety_limit": 1,
            "replacement_attempt_count": 1,
            "completion_status": "complete",
        }
        for root in (11, 22)
        for order in (0, 1)
    }
    _publish_inputs(
        cfg,
        [("1", "2", 100, 60, 40)],
        cell_overrides=overrides,
    )

    artifacts = run_h2h_inference(cfg)
    inference = pq.read_table(artifacts.pairwise_inference).to_pandas().iloc[0]

    assert inference["games_attempted"] == 404
    assert inference["games_completed"] == 400
    assert inference["games_safety_limit"] == 4
    assert inference["replacement_attempt_count"] == 4
    assert inference["q_ab"] == pytest.approx(0.60)
    assert inference["q_ba"] == pytest.approx(0.40)
    assert inference["balanced_a_wins"] == 240
    assert inference["seat2_a_wins_ba"] == 120
    assert inference["balanced_a_win_rate_alias"] == pytest.approx(0.60)
    assert bool(inference["pair_inferentially_viable"])
    assert bool(inference["pair_operationally_viable"])
    assert inference["power_support_status"] == "completed_support_achieved"


def test_zero_completed_pair_is_no_test_and_never_equivalent(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.head2head.delta_equivalence = 0.20
    overrides = {
        (0, root, order): {
            "games_attempted": 200,
            "games_completed": 0,
            "games_safety_limit": 200,
            "replacement_attempt_count": 100,
            "completion_status": "unresolved_nonviable",
            "wins_seat1": 0,
            "wins_seat2": 0,
        }
        for root in (11, 22)
        for order in (0, 1)
    }
    _publish_inputs(
        cfg,
        [("1", "2", 100, 0, 0)],
        cell_overrides=overrides,
    )

    artifacts = run_h2h_inference(cfg)
    inference = pq.read_table(artifacts.pairwise_inference).to_pandas().iloc[0]

    assert inference["completion_status"] == "unresolved_nonviable"
    assert not bool(inference["formal_test_performed"])
    assert pd.isna(inference["q_ab"])
    assert pd.isna(inference["q_ba"])
    assert pd.isna(inference["d_ab"])
    assert pd.isna(inference["score_p_value"])
    assert pd.isna(inference["holm_adjusted_p"])
    assert not bool(inference["holm_reject"])
    assert inference["decision_class"] == "unresolved_nonviable"
    assert not bool(inference["strategy_a_operationally_viable"])
    assert not bool(inference["strategy_b_operationally_viable"])
    assert bool(inference["multiplicity_family_member"])
    assert inference["no_test_p_value_convention"] == "null_reported_treated_as_one_for_holm"
    assert inference["power_support_status"] == "unresolved_required_completed_support_not_achieved"


def test_below_threshold_candidate_retains_valid_pair_estimate_but_gets_no_claim(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    cfg.head2head.delta_equivalence = 0.20
    overrides = {
        (0, root, order): {
            "games_attempted": 102,
            "games_completed": 100,
            "games_safety_limit": 2,
            "replacement_attempt_count": 2,
            "completion_status": "complete",
        }
        for root in (11, 22)
        for order in (0, 1)
    }
    _publish_inputs(
        cfg,
        [("1", "2", 100, 50, 50)],
        cell_overrides=overrides,
    )

    artifacts = run_h2h_inference(cfg)
    inference = pq.read_table(artifacts.pairwise_inference).to_pandas().iloc[0]

    assert bool(inference["formal_test_performed"])
    assert inference["score_p_value"] == pytest.approx(1.0)
    assert inference["d_ab"] == pytest.approx(0.0)
    assert not bool(inference["pair_operationally_viable"])
    assert not bool(inference["pair_claim_eligible"])
    assert inference["decision_class"] == "unresolved_nonviable"


def test_no_test_pair_retains_frozen_holm_family_size(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    overrides = {
        (1, root, order): {
            "games_attempted": 200,
            "games_completed": 0,
            "games_safety_limit": 200,
            "replacement_attempt_count": 100,
            "completion_status": "unresolved_nonviable",
            "wins_seat1": 0,
            "wins_seat2": 0,
        }
        for root in (11, 22)
        for order in (0, 1)
    }
    _publish_inputs(
        cfg,
        [
            ("1", "2", 100, 80, 20),
            ("1", "3", 100, 0, 0),
            ("2", "3", 100, 50, 50),
        ],
        cell_overrides=overrides,
    )

    artifacts = run_h2h_inference(cfg)
    inference = pq.read_table(artifacts.pairwise_inference).to_pandas().set_index("pair_id")

    assert pd.isna(inference.loc[1, "score_p_value"])
    assert pd.isna(inference.loc[1, "holm_adjusted_p"])
    assert not bool(inference.loc[1, "formal_test_performed"])
    assert inference.loc[0, "holm_adjusted_p"] == pytest.approx(
        min(1.0, 3.0 * float(cast(Any, inference.loc[0, "score_p_value"])))
    )
    assert inference["multiplicity_family_member"].all()


def test_statistical_contract_rejects_invalid_equivalence_margin(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.head2head.delta_equivalence = 0.0

    with pytest.raises(ValueError, match="delta_equivalence"):
        cfg.validate_statistical_contract(require_two_roots=True)


def test_inference_rejects_counts_that_do_not_authenticate_frozen_schedule(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    _publish_inputs(cfg, [("1", "2", 5_000, 2_600, 2_400)])
    counts_path = cfg.h2h_order_counts_path()
    counts = pq.read_table(counts_path).to_pandas()
    mask = (counts["root_seed"] == 22) & (counts["order"] == 1)
    counts.loc[mask, "n_completed_required"] = 5_001
    table = pa.Table.from_pandas(counts, preserve_index=False)
    sidecar = make_artifact_sidecar(
        cfg,
        counts_path,
        producer="test",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="concatenate_root_order_blocks",
        uncertainty_method=SCORE_TEST_ID,
        consistency_columns=counts.columns.tolist(),
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined",
    )
    write_parquet_artifact_atomic(table, counts_path, sidecar=sidecar)

    with pytest.raises(ValueError, match="frozen schedule column n_completed_required"):
        run_h2h_inference(cfg)


def test_inference_rejects_compensated_root_order_imbalance(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    _publish_inputs(cfg, [("1", "2", 100, 50, 50)])
    counts_path = cfg.h2h_order_counts_path()
    counts = pq.read_table(counts_path).to_pandas()
    expanded = ((counts["root_seed"] == 11) & (counts["order"] == 1)) | (
        (counts["root_seed"] == 22) & (counts["order"] == 0)
    )
    counts.loc[expanded, "n_completed_required"] = 200
    counts.loc[expanded, "games_attempted"] = 200
    counts.loc[expanded, "games_completed"] = 200
    counts.loc[expanded, "wins_seat1"] = 100
    counts.loc[expanded, "wins_seat2"] = 100
    counts.loc[expanded, "wins_a"] = 100
    counts.loc[expanded, "wins_b"] = 100
    counts.loc[expanded, "replacement_attempt_count"] = 0
    table = pa.Table.from_pandas(counts, preserve_index=False)
    sidecar = make_artifact_sidecar(
        cfg,
        counts_path,
        producer="test",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="concatenate_root_order_blocks",
        uncertainty_method=SCORE_TEST_ID,
        consistency_columns=counts.columns.tolist(),
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined",
    )
    write_parquet_artifact_atomic(table, counts_path, sidecar=sidecar)

    with pytest.raises(ValueError, match="frozen schedule column n_completed_required"):
        run_h2h_inference(cfg)


def test_inference_accepts_exact_current_style_root_order_support(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    _publish_inputs(cfg, [("1", "2", 1_974, 1_020, 954)])

    artifacts = run_h2h_inference(cfg)

    combined = pq.read_table(artifacts.combined_order_counts).to_pandas()
    inference = pq.read_table(artifacts.pairwise_inference).to_pandas().iloc[0]
    assert len(combined) == 2
    assert combined["games_completed"].eq(3_948).all()
    assert combined["root_count"].eq(2).all()
    assert inference["games_completed"] == 7_896
    assert inference["power_support_status"] == "completed_support_achieved"
