from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from scripts.task4_h2h_example import numerical_example
from tests.unit.analysis.test_dominance import _cfg as dominance_config
from tests.unit.analysis.test_dominance import _publish as publish_decisions
from tests.unit.analysis.test_h2h_inference import _cfg, _publish_inputs
from tests.unit.analysis.test_h2h_schedule import _write_frozen_family
from tests.unit.analysis.test_structure_agreement import _inputs as agreement_inputs

from farkle.analysis import structure_agreement
from farkle.analysis.dominance import build_dominance_outputs
from farkle.analysis.h2h_inference import (
    pair_conclusions,
    run_h2h_inference,
    validate_pair_conclusions,
)
from farkle.analysis.h2h_schedule import plan_h2h_schedule
from farkle.analysis.structure_reporting import _h2h_method_notes
from farkle.config import load_app_config


def conclusion(
    *,
    low=-0.01,
    high=0.01,
    effect=0.0,
    holm=False,
    margin: float | None = 0.03,
    eligible=True,
    practical=0.03,
):
    return pair_conclusions(
        eligible=eligible,
        holm_reject=holm,
        effect=effect,
        low=low,
        high=high,
        practical_delta=practical,
        delta_equivalence=margin,
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_small_significant_advantage_can_also_be_equivalent(sign):
    row = conclusion(
        low=0.004 if sign > 0 else -0.02,
        high=0.02 if sign > 0 else -0.004,
        effect=sign * 0.01,
        holm=True,
    )
    direction = "a" if sign > 0 else "b"
    assert row[f"statistical_advantage_{direction}"]
    assert row["equivalent"] and not row["unresolved"] and not row["nonviable"]
    assert not row["practical_dominance_a"] and not row["practical_dominance_b"]
    assert row["decision_class"] == f"statistical_only_advantage_{direction}_and_equivalent"


@pytest.mark.parametrize(
    "low,high,effect,holm,expected",
    [
        (0.04, 0.08, 0.06, True, "practical_dominance_a"),
        (-0.08, -0.04, -0.06, True, "practical_dominance_b"),
        (-0.1, 0.1, 0.0, False, "unresolved"),
        (-0.03, 0.02, 0.0, False, "unresolved"),
        (-0.02, 0.03, 0.0, False, "unresolved"),
        (0.03, 0.05, 0.04, True, "statistical_only_advantage_a"),
        (-0.05, -0.03, -0.04, True, "statistical_only_advantage_b"),
    ],
)
def test_strict_boundaries_and_distinct_conclusions(low, high, effect, holm, expected):
    row = conclusion(low=low, high=high, effect=effect, holm=holm)
    assert row["decision_class"] == expected
    assert not row["equivalent"]


def test_nondefault_margin_independent_of_practical_threshold_and_viability():
    assert not conclusion(margin=0.005)["equivalent"]
    assert not conclusion(margin=None)["equivalent"]
    assert conclusion(margin=0.02)["equivalent"]
    both = conclusion(low=0.04, high=0.06, effect=0.05, holm=True, margin=0.07)
    assert both["practical_dominance_a"] and both["equivalent"]
    assert both["decision_class"] == "practical_dominance_a_and_equivalent"
    unavailable = conclusion(
        low=math.nan, high=math.nan, effect=math.nan, eligible=False, holm=True
    )
    assert unavailable["nonviable"]
    assert sum(value is True for value in unavailable.values()) == 1


def test_real_score_inference_publishes_both_flags_and_resumes(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.head2head.delta_equivalence = 0.03
    _publish_inputs(cfg, [("1", "2", 50000, 26000, 25000)])
    artifacts = run_h2h_inference(cfg)
    frame = pq.read_table(artifacts.pairwise_inference).to_pandas()
    row = frame.iloc[0]
    assert row.statistical_advantage_a and row.equivalent
    assert row.decision_class == "statistical_only_advantage_a_and_equivalent"
    assert row.simultaneous_d_low > 0 and row.simultaneous_d_high < 0.03
    before = {p: p.stat().st_mtime_ns for p in artifacts.all_paths}
    run_h2h_inference(cfg)
    assert before == {p: p.stat().st_mtime_ns for p in artifacts.all_paths}
    frame.loc[0, "equivalent"] = False
    with pytest.raises(ValueError, match="disagree"):
        validate_pair_conclusions(frame)


def test_graph_preserves_statistical_edge_and_does_not_merge_equivalence_chain(tmp_path):
    cfg = dominance_config(tmp_path)
    decisions = {
        ("A", "B"): "statistical_only_advantage_a_and_equivalent",
        ("B", "C"): "equivalent",
        ("A", "C"): "unresolved",
    }
    publish_decisions(cfg, ("A", "B", "C"), decisions)
    artifacts = build_dominance_outputs(cfg)
    edges = pq.read_table(artifacts.edges).to_pandas()
    assert list(zip(edges.graph_type, edges.winner, edges.loser, strict=True)) == [
        ("statistical", "A", "B")
    ]
    assert edges.equivalent.tolist() == [True]
    summary = json.loads(artifacts.summary.read_text())
    assert summary["equivalent_pair_count"] == 2
    assert summary["statistical_advantage_and_equivalence_pair_count"] == 1
    assert summary["equivalence_is_pairwise_not_transitive"]
    assert summary["unique_best"] is None
    fronts = pq.read_table(artifacts.fronts).to_pandas()
    assert fronts.practical_front.eq(1).all()


def test_agreement_retains_direction_for_equivalent_advantage(tmp_path):
    cfg = dominance_config(tmp_path)
    agreement_inputs(cfg)
    membership = pq.read_table(cfg.h2h_candidate_family_path()).to_pandas()
    inference = pq.read_table(cfg.h2h_pairwise_inference_path()).to_pandas()
    inference.loc[0, "decision_class"] = "statistical_only_advantage_a_and_equivalent"
    inference.loc[0, "equivalent"] = True
    inference.loc[0, "practical_dominance_a"] = False
    pairs = structure_agreement._pair_agreement(membership, inference)
    assert pairs.loc[0, "h2h_direction"] == "a"
    assert pairs.loc[0, "equivalent"]
    assert pairs.loc[0, "h2h_agrees_with_win_rate"]
    roots = pq.read_table(cfg.h2h_root_agreement_path()).to_pandas()
    summary = structure_agreement._summary(membership, pairs, roots, execution_scope="root_pair")
    assert summary["selection_conditioned_h2h"]["equivalent_pair_count"] == 1
    assert (
        summary["selection_conditioned_h2h"]["statistical_advantage_and_equivalence_pair_count"]
        == 1
    )


def test_equivalence_margin_changes_no_frozen_schedule_counts(tmp_path):
    plans, schedules = [], []
    for index, margin in enumerate((None, 0.015, 0.03)):
        cfg = _cfg(tmp_path / str(index))
        cfg.head2head.delta_equivalence = margin
        cfg.head2head.sensitivity_deltas = (0.03,)
        _write_frozen_family(cfg)
        result = plan_h2h_schedule(cfg)
        plans.append(json.loads(result.power_plan.read_text()))
        schedules.append(pq.read_table(result.block_manifest).to_pandas())
    for key in (
        "schedule_hash",
        "n_completed_required_per_root_order_block",
        "total_completed_required",
        "maximum_total_attempts",
    ):
        assert len({p[key] for p in plans}) == 1
    pd.testing.assert_frame_equal(schedules[0], schedules[1])
    pd.testing.assert_frame_equal(schedules[0], schedules[2])


def test_report_notes_are_configured_and_do_not_claim_joint_error_control(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.head2head.delta_equivalence = 0.017
    notes = " ".join(
        _h2h_method_notes(cfg, {"target_power": 0.8, "target_effect": 0.03, "family_alpha": 0.02})
    )
    for phrase in (
        "+/-1.7",
        "80%",
        "2%",
        "zero-advantage",
        "not power",
        "No single joint",
        "not transitive",
        "inverse-square",
        "not target sample size",
        "completed games",
    ):
        assert phrase in notes


def test_actual_150_finalist_plan_detects_zero_without_proving_practical_or_equivalent():
    cfg = load_app_config(Path(__file__).resolve().parents[3] / "configs/farkle_mega_config.yaml")
    result = numerical_example(cfg, 150)
    assert result["pair_count"] == 11175
    assert result["worst_scenario_rejection_power"] >= cfg.head2head.target_power
    assert result["total_planned_completed_games"] == (
        result["pair_count"] * 2 * result["n_completed_per_order"]
    )
    example = result["example"]
    assert 0 < example["simultaneous_d_low"] < cfg.head2head.practical_delta
    assert example["simultaneous_d_high"] > cfg.head2head.delta_equivalence
    assert example["statistical_advantage_a"]
    assert not example["practical_dominance_a"] and not example["equivalent"]
    assert example["decision_class"] == "statistical_only_advantage_a"
