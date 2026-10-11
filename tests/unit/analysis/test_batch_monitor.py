from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts.task5b_capability import (
    _unit,
    blind_spots,
    evaluate_fixture,
    fixture_counts,
    integrity_and_support,
    verify_evidence,
    write_json,
)

from farkle.analysis import batch_monitor as monitor
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.authenticated_contract import identity_sha256


@pytest.fixture
def recipe():
    return json.loads(
        (
            Path(__file__).resolve().parents[3] / "configs/validation/task5b_capability.json"
        ).read_text()
    )


def test_statistics_match_independent_direct_calculations():
    values = np.array([1.0, 3.0, 2.0, 4.0, 8.0] * 20)
    lag, drift = monitor.diagnose_series(values * 10, np.full(100, 10))
    assert lag.statistic == pytest.approx(np.corrcoef(values[:-1], values[1:])[0, 1])
    assert drift.statistic == pytest.approx(
        (sum(values[50:]) / 50 - sum(values[:50]) / 50)
        / np.sqrt(sum((values - sum(values) / 100) ** 2) / 99)
    )
    assert lag.effect_threshold == 0.5


@pytest.mark.parametrize("boundary", [-0.5, 0.5])
def test_effect_boundary_is_strict(monkeypatch, boundary):
    monkeypatch.setattr(monitor, "lag_one_correlation", lambda _: boundary)
    lag, _ = monitor.diagnose_series(np.arange(100), np.ones(100))
    assert lag.result == "no_concern_detected"
    monkeypatch.setattr(monitor, "lag_one_correlation", lambda _: boundary * 1.01)
    lag, _ = monitor.diagnose_series(np.arange(100), np.ones(100))
    assert lag.result == "investigate"


def test_descriptive_band_excursion_is_not_automatically_an_alert(monkeypatch):
    monkeypatch.setattr(monitor, "lag_one_correlation", lambda _: 0.3)
    lag, _ = monitor.diagnose_series(np.arange(100), np.ones(100))
    assert 1.96 / np.sqrt(99) < 0.3
    assert lag.result == "no_concern_detected"


@pytest.mark.parametrize(
    "n,reason",
    [
        (29, "fewer_than_30_batches"),
        (31, "batch_count_outside_capability_recipe"),
        (101, "batch_count_outside_capability_recipe"),
    ],
)
def test_unvalidated_or_short_support_withholds(n, reason):
    assert all(row.reason == reason for row in monitor.diagnose_series(np.arange(n), np.ones(n)))


def test_no_dropping_zero_support_or_claiming_constant_series_independence():
    den = np.ones(100)
    den[40] = 0
    rows = monitor.diagnose_series(np.arange(100), den)
    assert all(row.support == 100 and row.result == "insufficient_evidence" for row in rows)
    assert all(row.reason == "zero_denominator_batch_not_dropped" for row in rows)
    assert all(
        row.reason == "constant_series"
        for row in monitor.diagnose_series(np.zeros(100), np.ones(100))
    )
    # A nonconstant whole vector can still have a constant lag segment.
    lag, drift = monitor.diagnose_series(np.r_[np.zeros(99), 1.0], np.ones(100))
    assert lag.result == "insufficient_evidence" and drift.statistic is not None


@pytest.mark.parametrize(
    "num,den", [([1], [1, 2]), ([np.nan], [1]), ([1], [-1]), ([np.inf], [1]), ([], [])]
)
def test_malformed_vectors_fail(num, den):
    with pytest.raises(ValueError):
        monitor.diagnose_series(np.asarray(num), np.asarray(den))


def test_policy_changes_require_new_capability():
    with pytest.raises(ValueError, match="new capability"):
        monitor.MonitorPolicy(lag_effect_floor=0.1)


def test_seeded_fixtures_replay_and_preserve_clean_counterparts(recipe):
    clean, fault = recipe["fixtures"][:2]
    a = fixture_counts(recipe, 51001, 100, 0, clean)
    b = fixture_counts(recipe, 51001, 100, 0, clean)
    for field in a:
        np.testing.assert_array_equal(a[field], b[field])
    # Only sentinel zero is injected; unrelated semantic streams are unchanged.
    a = fixture_counts(recipe, 51001, 100, 1, clean)
    b = fixture_counts(recipe, 51001, 100, 1, fault)
    for field in a:
        np.testing.assert_array_equal(a[field], b[field])


def test_integrity_faults_raise_and_unsupported_cases_withhold(recipe):
    result = integrity_and_support(recipe)
    assert result["clean_integrity_passed"]
    assert len(result["faults"]) == 9
    assert all(row["detected"] for row in result["faults"].values())
    assert all(
        row["result"] == "insufficient_evidence"
        for pair in result["support"].values()
        for row in pair
    )


def test_complete_inventory_is_required_even_when_counts_are_unchanged(recipe):
    counts = fixture_counts(recipe, 51001, 100, 0, recipe["fixtures"][0])
    with pytest.raises(ValueError, match="coverage/order"):
        monitor.diagnose_counts(np.r_[np.arange(99), 98], counts, expected_batches=100)
    counts.pop("wins")
    with pytest.raises(ValueError, match="missing required"):
        monitor.diagnose_counts(np.arange(100), counts, expected_batches=100)


def test_within_batch_and_shared_root_faults_expose_distinct_blind_spots(recipe):
    result = blind_spots(recipe)
    hidden = result["within_batch"]
    assert hidden["batch_totals_identical"] and hidden["batch_results_identical"]
    assert abs(hidden["clean_game_lag_one"]) < 0.1
    assert hidden["fault_game_lag_one"] > 0.9
    assert all(row["result"] == "no_concern_detected" for row in hidden["batch_results"])
    assert result["common_fault_roots"]["between_root_difference"] == 0
    assert result["common_fault_roots"]["within_each_root_warning_count"] > 0


def test_atomic_unit_resume_corruption_and_publication_interruption(recipe, tmp_path, monkeypatch):
    import scripts.task5b_capability as driver

    task = (recipe, "p" * 64, str(tmp_path), 51001, 100, recipe["fixtures"][0], False)
    relative, expected, reused = _unit(task)
    assert not reused
    path = tmp_path / relative
    original = path.read_bytes()
    mtime = path.stat().st_mtime_ns
    assert _unit(task)[2]
    assert path.stat().st_mtime_ns == mtime
    path.write_text("corrupt")
    assert _unit(task)[1] == expected
    assert path.read_bytes() == original
    path.unlink()
    real_write = driver.write_json
    monkeypatch.setattr(
        driver, "write_json", lambda *_: (_ for _ in ()).throw(RuntimeError("interrupted"))
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        _unit(task)
    assert not path.exists()
    monkeypatch.setattr(driver, "write_json", real_write)
    assert _unit(task)[1] == expected


def test_fixture_records_family_denominator_and_fault_localization(recipe):
    row = evaluate_fixture(recipe, 51001, 100, recipe["fixtures"][1])
    assert row["series_count"] == 96
    assert row["check_opportunities"] == 192
    assert row["injected_lag_alert"]
    assert len(row["checks"]) == 192


def test_evidence_requires_complete_inventory_and_exact_summary(recipe, tmp_path):
    recipe = {
        **recipe,
        "fixtures": recipe["fixtures"][:1],
        "fixture_root_seeds": [51001],
        "batch_counts": [100],
    }
    plan = {"recipe": recipe}
    plan_hash = identity_sha256(plan)
    write_json(tmp_path / "plan.json", plan)
    relative, _, _ = _unit(
        (recipe, plan_hash, str(tmp_path), 51001, 100, recipe["fixtures"][0], False)
    )
    summary = {
        "plan_sha256": plan_hash,
        "units": [{"path": relative, "sha256": sha256_file(tmp_path / relative)}],
    }
    write_json(tmp_path / "summary.json", summary)
    assert not verify_evidence(tmp_path)  # Final digest has not been committed.
    write_json(
        tmp_path / "evidence_digest.json",
        {"plan_sha256": plan_hash, "summary_sha256": sha256_file(tmp_path / "summary.json")},
    )
    assert verify_evidence(tmp_path)
    write_json(tmp_path / "summary.json", {**summary, "changed": True})
    assert not verify_evidence(tmp_path)
    write_json(tmp_path / "summary.json", summary)
    assert verify_evidence(tmp_path)
    (tmp_path / relative).unlink()
    assert not verify_evidence(tmp_path)


def test_driver_completion_order_cannot_rewrite_scientific_summary(recipe, tmp_path, monkeypatch):
    import scripts.task5b_capability as driver

    output = tmp_path / "evidence"
    reduced = {
        **recipe,
        "sentinel_count": 1,
        "fixture_root_seeds": [51001],
        "output_dir": str(output),
    }
    path = tmp_path / "recipe.json"
    write_json(path, reduced)
    reverse = False

    def simulated_completion_order(fn, items, **_kwargs):
        completed = [fn(item) for item in items]
        yield from reversed(completed) if reverse else completed

    monkeypatch.setattr(driver, "process_map", simulated_completion_order)
    first = driver.run(path, workers=1)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*.json")}
    reverse = True
    second = driver.run(path, workers=1)
    assert second == first
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == original for p, original in before.items())
    assert verify_evidence(output)
