from __future__ import annotations

import copy
import itertools
import json
from typing import Any, cast

import pytest

from farkle.analysis.parameter_responses import response_rows, response_schema, selected_levels
from farkle.simulation.strategies import ThresholdStrategy, build_strategy_manifest


def _fixture(interaction=0.0):
    strategies = [
        ThresholdStrategy(
            score_threshold=score,
            dice_threshold=dice,
            auto_hot_dice=hot,
            run_up_score=run,
            strategy_id=i,
        )
        for i, (score, dice, hot, run) in enumerate(
            itertools.product((100, 200), (1, 2), (False, True), (False, True))
        )
    ]
    manifest = cast(list[dict[str, Any]], build_strategy_manifest(strategies).to_dict("records"))
    performance, predictions = [], []
    for row in manifest:
        sid = int(row["strategy_id"])
        value = (
            0.1
            + 0.1 * (row["score_threshold"] == 200)
            + 0.05 * (row["dice_threshold"] == 2)
            + interaction * (row["score_threshold"] == 200 and row["dice_threshold"] == 2)
        )
        performance.append({"strategy": sid, "win_rate": value, "k": 2, "root_seed": 11})
        predictions.append(
            {
                "strategy": sid,
                "observed_win_rate": value,
                "predicted_win_rate": value + 0.2 * (sid % 2),
                "fold": sid % 2,
                "root_seed": 11,
                "players": 2,
            }
        )
    folds = [
        {"fold": f, "root_seed": 11, "players": 2, "train_strategies": 8, "heldout_strategies": 8}
        for f in (0, 1)
    ]
    return manifest, performance, predictions, folds


def _result(data, **kwargs):
    return list(response_rows(*data, root=11, k=2, source_identity={"fixture": "a" * 64}, **kwargs))


@pytest.mark.parametrize("interaction", [0.0, 0.125])
def test_observed_matched_contrast_and_separate_fold_means(interaction):
    data = _fixture(interaction)
    result = _result(data)
    two = [r for p, r in result if p == "two_way" and r["parameter_a"] == "score_threshold"]
    assert len(two) == 1
    assert two[0]["matched_backgrounds"] == 4
    assert two[0]["observed_nonadditivity"] == pytest.approx(interaction)
    assert not any(
        "interaction" in n or "nonadditivity" in n
        for n in response_schema("two_way").names
        if n.startswith("oof")
    )
    support = [r for p, r in result if p == "support" and r["support_id"] == two[0]["support_id"]]
    assert len(support) == 16
    assert {r["fold"] for r in support} == {0, 1}
    assert all(json.loads(r["specification_json"])["matching_attributes"] for r in support)
    one = [r for p, r in result if p == "one_way" and r["parameter"] == "score_threshold"]
    assert {r["matched_backgrounds"] for r in one} == {8}
    assert one[1]["observed_mean"] - one[0]["observed_mean"] == pytest.approx(0.1 + interaction / 2)


def test_missing_required_input_fails_not_support_shrink():
    for index in (0, 1, 2):
        data = copy.deepcopy(_fixture())
        data[index].pop()
        with pytest.raises(ValueError, match="support|predictions"):
            _result(data)


def test_impossible_smart_quartet_is_unestimable():
    data = _fixture()
    # All smart-five values are False: no rectangle can be claimed.
    result = _result(data)
    support = [
        r
        for p, r in result
        if p == "support"
        and json.loads(r["specification_json"]).get("parameters") == ["smart_five", "auto_hot_dice"]
    ]
    assert len(support) == 1
    assert support[0]["status"] == "unestimable_common_support"
    assert support[0]["strategy"] is None


def test_selection_bound_inactive_levels_and_illegal_settings():
    assert selected_levels(list(range(101))) == [i * 100 // 24 for i in range(25)]
    data = copy.deepcopy(_fixture())
    data[0][0]["smart_one"] = True
    with pytest.raises(ValueError, match="smart_one"):
        _result(data)
    data = copy.deepcopy(_fixture())
    for r in data[0]:
        r["consider_score"] = False
    with pytest.raises(ValueError, match="equivalent"):
        _result(data)


def test_declared_prediction_unavailable_preserves_observed_contrasts():
    m, p, _, _ = _fixture(0.125)
    result = _result((m, p, [], []), prediction_unavailable=True)
    row = next(r for product, r in result if product == "two_way")
    assert row["status"] == "prediction_unavailable"
    assert row["observed_nonadditivity"] == pytest.approx(0.125)
    assert row["oof_mean_00"] is None


def test_changed_sources_change_support_identity():
    data = _fixture()
    first = _result(data)
    second = list(response_rows(*data, root=11, k=2, source_identity={"fixture": "b" * 64}))
    assert first[0][1]["support_id"] != second[0][1]["support_id"]


def test_adjacent_rectangle_without_common_background_is_not_zero():
    m, p, predictions, folds = _fixture()
    removed = {
        r["strategy_id"] for r in m if r["score_threshold"] == 200 and r["dice_threshold"] == 2
    }
    m = [r for r in m if r["strategy_id"] not in removed]
    p = [r for r in p if r["strategy"] not in removed]
    predictions = [r for r in predictions if r["strategy"] not in removed]
    for f in folds:
        f["heldout_strategies"] = sum(r["fold"] == f["fold"] for r in predictions)
        f["train_strategies"] = len(m) - f["heldout_strategies"]
    rows = _result((m, p, predictions, folds))
    contrast = next(
        r for kind, r in rows if kind == "two_way" and r["parameter_a"] == "score_threshold"
    )
    assert contrast["matched_backgrounds"] == 0
    assert contrast["observed_nonadditivity"] is None
    assert contrast["status"] == "unestimable_common_support"


def test_favor_alias_varies_without_holding_it_fixed():
    from farkle.simulation.strategies import FavorDiceOrScore

    m = build_strategy_manifest(
        [
            ThresholdStrategy(smart_five=True, favor_dice_or_score=favor, strategy_id=i)
            for i, favor in enumerate((FavorDiceOrScore.DICE, FavorDiceOrScore.SCORE))
        ]
    ).to_dict("records")
    p = [{"strategy": i, "win_rate": v} for i, v in enumerate((0.2, 0.7))]
    result = _result((m, p, [], []), prediction_unavailable=True)
    values = [r for kind, r in result if kind == "one_way" and r["parameter"] == "favor_score"]
    assert [r["level"] for r in values] == [0, 1]
    assert [r["observed_mean"] for r in values] == [0.2, 0.7]
    assert [r["matched_backgrounds"] for r in values] == [1, 1]


def test_unknown_type_is_retained_as_explicit_exclusion():
    m, p, predictions, folds = _fixture()
    m[0]["strategy_type"] = "unrecognized_custom_strategy"
    result = _result((m, p, predictions, folds))
    record = next(r for kind, r in result if kind == "support" and r["product"] == "excluded_type")
    assert record["status"] == "unsupported_type"
    assert json.loads(record["specification_json"])["excluded_strategy_ids"] == [0]
