from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.stats import norm

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


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(
            seed=11,
            seed_list=[11, 22],
            seed_pair=(11, 22),
            n_players_list=[2],
        ),
    )
    cfg.screening.practical_delta_by_k = {2: 0.03}
    cfg.screening.delta_across_k = 0.03
    return cfg


def _publish_inputs(
    cfg: AppConfig,
    pairs: list[tuple[str, str, int, int, int]],
) -> None:
    """Publish pairs as (A, B, games/order/root, x_ab/root, x_ba/root)."""

    family_hash = "b" * 64
    rows: list[dict[str, object]] = []
    for pair_id, (strategy_a, strategy_b, games, x_ab, x_ba) in enumerate(pairs):
        for root_index, root_seed in enumerate((11, 22)):
            for order, wins_seat1 in ((0, x_ab), (1, x_ba)):
                rows.append(
                    {
                        "family_hash": family_hash,
                        "pair_id": pair_id,
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "root_seed": root_seed,
                        "root_index": root_index,
                        "order": order,
                        "order_label": "a_b" if order == 0 else "b_a",
                        "games_required": games,
                        "games_completed": games,
                        "wins_seat1": wins_seat1,
                        "wins_seat2": games - wins_seat1,
                        "score_test_id": SCORE_TEST_ID,
                    }
                )
    counts = pd.DataFrame(rows)
    counts_path = cfg.h2h_order_counts_path()
    counts_table = pa.Table.from_pandas(counts, preserve_index=False)
    counts_sidecar = make_artifact_sidecar(
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
    write_parquet_artifact_atomic(counts_table, counts_path, sidecar=counts_sidecar)

    plan = {
        "family_hash": family_hash,
        "schedule_state": "ready",
        "root_seeds": [11, 22],
        "unordered_pair_count": len(pairs),
        "target_power": 0.80,
        "worst_scenario_achieved_power": 0.81,
    }
    plan_path = cfg.h2h_power_plan_path()
    plan_sidecar = make_artifact_sidecar(
        cfg,
        plan_path,
        producer="test",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="score_test_power_plan",
        consistency_columns=list(plan),
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined",
    )
    write_json_artifact_atomic(plan, plan_path, sidecar=plan_sidecar)


def test_score_test_uses_constrained_null_proportion() -> None:
    result = two_proportion_score_test(60, 100, 40, 100)
    expected_z = 0.2 / ((0.5 * 0.5 * (1 / 100 + 1 / 100)) ** 0.5)

    assert result.difference == pytest.approx(0.2)
    assert result.null_proportion == pytest.approx(0.5)
    assert result.statistic == pytest.approx(expected_z)
    assert result.p_value == pytest.approx(2 * norm.sf(abs(expected_z)))


def test_score_interval_contains_observed_difference() -> None:
    low, high = score_difference_interval(60, 100, 40, 100, alpha=0.02)

    assert low < 0.2 < high
    assert low > -1.0
    assert high < 1.0


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
    assert first_ab["games"] == 10_000
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

    validate_artifact_sidecar(
        artifacts.pairwise_inference,
        expected={
            "scope": "h2h_2p",
            "operation": "seat_adjusted_score_inference",
            "uncertainty_method": f"{SCORE_TEST_ID}_holm",
        },
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


def test_statistical_contract_rejects_invalid_equivalence_margin(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.head2head.delta_equivalence = 0.0

    with pytest.raises(ValueError, match="delta_equivalence"):
        cfg.validate_statistical_contract(require_two_roots=True)


def test_inference_rejects_unbalanced_order_allocation(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _publish_inputs(cfg, [("1", "2", 5_000, 2_600, 2_400)])
    counts_path = cfg.h2h_order_counts_path()
    counts = pq.read_table(counts_path).to_pandas()
    mask = (counts["root_seed"] == 22) & (counts["order"] == 1)
    counts.loc[mask, "games_required"] = 4_999
    counts.loc[mask, "games_completed"] = 4_999
    counts.loc[mask, "wins_seat2"] = 2_599
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

    with pytest.raises(ValueError, match="not exactly balanced"):
        run_h2h_inference(cfg)
