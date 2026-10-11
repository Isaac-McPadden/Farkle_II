"""Bounded seeded capability evidence; no production simulation or stage wiring.

Each fixture is one atomic, hashed result unit. Compatible units resume exactly;
the small summary retains counts and their denominators, not all fixture arrays.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from farkle.analysis.batch_monitor import (
    METHOD_ID,
    METRICS,
    MonitorPolicy,
    diagnose_counts,
    diagnose_series,
    lag_one_correlation,
)
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.authenticated_contract import (
    CodeIdentityPolicy,
    identity_sha256,
    resolve_code_identity,
)
from farkle.utils.parallel import (
    StageParallelPolicy,
    apply_native_thread_limits,
    process_map,
    resolve_mp_context,
)
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose, coordinate_rng
from farkle.utils.writer import atomic_path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as staged:
        Path(staged).write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def _json_matches(path: Path, expected: dict) -> bool:
    try:
        return json.loads(path.read_text()) == expected
    except (OSError, ValueError):
        return False


def fixture_counts(
    recipe: dict, seed: int, batches: int, sentinel: int, fault: dict
) -> dict[str, np.ndarray]:
    # Config seed + sentinel + B own the fixture stream. Fault and clean share
    # innovations deliberately, enabling paired capability comparisons.
    rng = coordinate_rng(
        RandomPurpose.INDEXED_SEED, root_seed=seed, game_index=sentinel, replicate_index=batches
    )
    burn = recipe["burn_in"]
    innovations = rng.normal(size=(burn + batches, recipe["latent_dimensions"]))
    latent = innovations.copy()
    if sentinel == 0 and fault["kind"] == "ar1":
        rho = fault["strength"]
        for index in range(1, len(latent)):
            latent[index] = rho * latent[index - 1] + np.sqrt(1 - rho * rho) * innovations[index]
    latent = latent[burn:]
    if sentinel == 0 and fault["kind"] == "step":
        latent[: batches // 2] -= fault["strength"] / 2
        latent[batches // 2 :] += fault["strength"] / 2
    attempted = np.full(batches, recipe["attempted_exposures_per_batch"], dtype=np.int64)
    safety = np.clip(np.rint(10 + 3 * latent[:, 1]), 0, attempted).astype(np.int64)
    completed = attempted - safety
    wins = np.clip(np.rint(completed * 0.5 + 100 * latent[:, 0]), 1, completed - 1).astype(np.int64)
    losses = completed - wins
    losing_mean = np.maximum(0, np.rint(1000 + 50 * latent[:, 2])).astype(np.int64)
    winning_mean = np.maximum(0, np.rint(600 + 40 * latent[:, 3])).astype(np.int64)
    return {
        "attempted": attempted,
        "safety": safety,
        "completed": completed,
        "wins": wins,
        "losses_completed": losses,
        "deficit_sum": losses * losing_mean,
        "victory_margin_sum": wins * winning_mean,
    }


def evaluate_fixture(recipe: dict, seed: int, batches: int, fault: dict) -> dict:
    checks = []
    for sentinel in range(recipe["sentinel_count"]):
        counts = fixture_counts(recipe, seed, batches, sentinel, fault)
        results = diagnose_counts(np.arange(batches), counts, expected_batches=batches)
        for metric, pair in results.items():
            for item in pair:
                checks.append({"sentinel": sentinel, "metric": metric, **item.payload()})
    warnings = [row for row in checks if row["result"] == "investigate"]
    return {
        "seed": seed,
        "batches": batches,
        "fixture": fault["id"],
        "series_count": recipe["sentinel_count"] * len(METRICS),
        "check_opportunities": len(checks),
        "warning_count": len(warnings),
        "family_alert": bool(warnings),
        "injected_sentinel_alert": any(row["sentinel"] == 0 for row in warnings),
        "injected_lag_alert": any(
            row["sentinel"] == 0 and row["check_id"] == "lag_one" for row in warnings
        ),
        "injected_drift_alert": any(
            row["sentinel"] == 0 and row["check_id"] == "half_drift" for row in warnings
        ),
        "insufficient_count": sum(row["result"] == "insufficient_evidence" for row in checks),
        "checks": checks,
    }


def _worker_init() -> None:
    apply_native_thread_limits(StageParallelPolicy(2, 2, 1, 1, 1))


def _unit(task: tuple) -> tuple[str, dict, bool]:
    recipe, plan_hash, folder, seed, batches, fault, force = task
    relative = f"units/{fault['id']}/b{batches}/seed{seed}.json"
    path = Path(folder) / relative
    expected = {"seed": seed, "batches": batches, "fixture": fault["id"]}
    if path.exists() and not force:
        try:
            unit = json.loads(path.read_text())
            result = unit["result"]
            if (
                unit["plan_sha256"] == plan_hash
                and unit["result_sha256"] == identity_sha256(result)
                and all(result[k] == v for k, v in expected.items())
            ):
                return relative, result, True
        except (OSError, ValueError, KeyError, TypeError):
            pass  # Only this absent/corrupt unit is recomputed.
    result = evaluate_fixture(recipe, seed, batches, fault)
    write_json(
        path, {"plan_sha256": plan_hash, "result_sha256": identity_sha256(result), "result": result}
    )
    return relative, result, False


def integrity_and_support(recipe: dict) -> dict:
    clean = recipe["fixtures"][0]
    baseline = fixture_counts(recipe, recipe["fixture_root_seeds"][0], 100, 0, clean)
    outcomes = {}
    diagnose_counts(np.arange(100), baseline, expected_batches=100)
    for fault in recipe["acceptance"]["integrity_faults"]:
        counts = {key: values.copy() for key, values in baseline.items()}
        ids = np.arange(100)
        if fault == "missing_batch":
            ids = ids[:-1]
            counts = {key: values[:-1] for key, values in counts.items()}
        elif fault == "duplicate_batch":
            ids[-1] = ids[-2]
        elif fault == "out_of_order":
            ids[[2, 3]] = ids[[3, 2]]
        elif fault == "negative_count":
            counts["wins"][0] = -1
        elif fault == "noninteger_count":
            counts["wins"] = counts["wins"].astype(float)
            counts["wins"][0] += 0.5
        elif fault == "completion_conservation":
            counts["completed"][0] -= 1
        elif fault == "win_loss_conservation":
            counts["wins"][0] -= 1
        elif fault == "zero_loss_nonzero_deficit":
            counts["losses_completed"][0] = 0
            counts["wins"][0] = counts["completed"][0]
        elif fault == "zero_win_nonzero_margin":
            counts["wins"][0] = 0
            counts["losses_completed"][0] = counts["completed"][0]
        else:
            raise ValueError(f"unimplemented integrity fault: {fault}")
        try:
            diagnose_counts(ids, counts, expected_batches=100)
        except ValueError as exc:
            outcomes[fault] = {"detected": True, "reason": str(exc)}
        else:
            outcomes[fault] = {"detected": False}
    support = {}
    for case in recipe["acceptance"]["required_support_cases"]:
        n = {"short": 29, "unvalidated_batch_count": 31}.get(case, 100)
        numerator = np.arange(n, dtype=float)
        denominator = np.ones(n)
        if case == "constant":
            numerator[:] = 0
        if case == "zero_denominator":
            denominator[40] = 0
        rows = diagnose_series(numerator, denominator)
        support[case] = [r.payload() for r in rows]
    return {"clean_integrity_passed": True, "faults": outcomes, "support": support}


def blind_spots(recipe: dict) -> dict:
    batches, size = 100, 1000
    rng = coordinate_rng(RandomPurpose.INDEXED_SEED, root_seed=recipe["blind_spot_seed"])
    wins = np.clip(np.rint(500 + 15 * rng.normal(size=batches)), 1, size - 1).astype(int)
    ordered = np.concatenate([np.r_[np.zeros(size - count), np.ones(count)] for count in wins])
    clean = np.concatenate(
        [rng.permutation(ordered[i * size : (i + 1) * size]) for i in range(batches)]
    )
    assert np.array_equal(
        clean.reshape(batches, size).sum(axis=1), ordered.reshape(batches, size).sum(axis=1)
    )
    batch_results = [row.payload() for row in diagnose_series(wins, np.full(batches, size))]
    step = next(f for f in recipe["fixtures"] if f["id"] == "half_drift")
    common = fixture_counts(recipe, recipe["common_fault_seed"], 100, 0, step)
    # Deliberate identical corruption in two nominal root datasets. Root
    # agreement is perfect, although each root has the same drift fault.
    root_results = diagnose_counts(np.arange(100), common, expected_batches=100)
    other_root = {key: value.copy() for key, value in common.items()}
    other_results = diagnose_counts(np.arange(100), other_root, expected_batches=100)
    return {
        "within_batch": {
            "seed": recipe["blind_spot_seed"],
            "batches": batches,
            "games_per_batch": size,
            "batch_totals_identical": True,
            "clean_game_lag_one": lag_one_correlation(clean),
            "fault_game_lag_one": lag_one_correlation(ordered),
            "batch_results_identical": True,
            "batch_results": batch_results,
            "interpretation": "Game order is hidden by unchanged batch totals; this is outside batch-monitor coverage.",
        },
        "common_fault_roots": {
            "seed": recipe["common_fault_seed"],
            "construction": "same corrupted summaries copied into both nominal roots, not two independent realizations",
            "between_root_difference": max(
                int(np.max(np.abs(common[key] - other_root[key]))) for key in common
            ),
            "within_each_root_warning_count": sum(
                row.result == "investigate" for pair in root_results.values() for row in pair
            ),
            "other_root_warning_count": sum(
                row.result == "investigate" for pair in other_results.values() for row in pair
            ),
            "interpretation": "Agreement on a common injected fault is not independent proof of correctness.",
        },
    }


def verify_evidence(output: Path) -> bool:
    """Verify this standalone evidence graph, not a production PE completion."""
    try:
        plan = json.loads((output / "plan.json").read_text())
        summary = json.loads((output / "summary.json").read_text())
        digest = json.loads((output / "evidence_digest.json").read_text())
        plan_hash = identity_sha256(plan)
        if (
            summary["plan_sha256"] != plan_hash
            or digest["plan_sha256"] != plan_hash
            or digest["summary_sha256"] != sha256_file(output / "summary.json")
        ):
            return False
        recipe = plan["recipe"]
        expected = {
            f"units/{fault['id']}/b{n}/seed{seed}.json"
            for fault in recipe["fixtures"]
            for n in recipe["batch_counts"]
            for seed in recipe["fixture_root_seeds"]
        }
        paths = [unit["path"] for unit in summary["units"]]
        if len(paths) != len(expected) or set(paths) != expected:
            return False
        for entry in summary["units"]:
            path = output / entry["path"]
            if sha256_file(path) != entry["sha256"]:
                return False
            unit = json.loads(path.read_text())
            if unit["plan_sha256"] != plan_hash or unit["result_sha256"] != identity_sha256(
                unit["result"]
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run(recipe_path: Path, *, force: bool = False, workers: int | None = None) -> dict:
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe["method_id"] != METHOD_ID or recipe["metric_ids"] != list(METRICS):
        raise ValueError("recipe method/metrics differ from implementation")
    if (
        not 1 <= recipe["sentinel_count"] <= 16
        or recipe["latent_dimensions"] != 4
        or not 1 <= len(recipe["fixture_root_seeds"]) <= 64
        or len(set(recipe["fixture_root_seeds"])) != len(recipe["fixture_root_seeds"])
        or recipe["batch_counts"] != [30, 100]
        or recipe["attempted_exposures_per_batch"] != 10000
        or recipe["execution"]["workers"] != 2
        or recipe["execution"]["native_threads"] != 1
    ):
        raise ValueError("recipe lies outside the bounded fixture contract")
    p = recipe["policy"]
    MonitorPolicy(
        p["minimum_batches"],
        tuple(p["supported_batch_counts"]),
        p["lag_effect_floor"],
        p["lag_support_multiplier"],
        p["half_drift_threshold"],
    )
    output = ROOT / recipe["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    code = asdict(resolve_code_identity(ROOT, policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY))
    plan = {
        "recipe": recipe,
        "recipe_sha256": sha256_file(recipe_path),
        "code_identity": code,
        "rng_scheme": RNG_SCHEME_VERSION,
        "bit_generator": "PCG64DXSM",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/farkle/analysis/batch_monitor.py",
                ROOT / "src/farkle/utils/random.py",
            )
        },
    }
    plan_hash = identity_sha256(plan)
    plan_path = output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan and not force:
        raise ValueError("capability identity changed; use --force for an explicit rebuild")
    if not plan_path.exists() or force:
        write_json(plan_path, plan)
    started = time.perf_counter()
    tasks = (
        (recipe, plan_hash, str(output), seed, n, fault, force)
        for fault in recipe["fixtures"]
        for n in recipe["batch_counts"]
        for seed in recipe["fixture_root_seeds"]
    )
    groups: dict[str, Counter] = {}
    manifest, reused = [], 0
    jobs = recipe["execution"]["workers"] if workers is None else workers
    if not 1 <= jobs <= recipe["execution"]["workers"]:
        raise ValueError("workers exceed the frozen execution bound")
    _worker_init()
    for relative, result, cached in process_map(
        _unit,
        tasks,
        n_jobs=jobs,
        initializer=_worker_init,
        mp_context=resolve_mp_context("spawn"),
        window=2 * jobs,
    ):
        reused += int(cached)
        group = groups.setdefault(f"{result['fixture']}/b{result['batches']}", Counter())
        group.update(
            fixtures=1,
            family_alerts=int(result["family_alert"]),
            injected_sentinel_detections=int(result["injected_sentinel_alert"]),
            injected_lag_detections=int(result["injected_lag_alert"]),
            injected_drift_detections=int(result["injected_drift_alert"]),
            warning_count=result["warning_count"],
            insufficient_count=result["insufficient_count"],
            check_opportunities=result["check_opportunities"],
        )
        manifest.append({"path": relative, "sha256": sha256_file(output / relative)})
    integrity, limitations = integrity_and_support(recipe), blind_spots(recipe)
    acceptance = recipe["acceptance"]
    primary = acceptance["primary_batches"]
    gates = {
        "clean_family_alert_budget": groups[f"clean/b{primary}"]["family_alerts"]
        <= acceptance["maximum_clean_family_alerts"]
    }
    for name in acceptance["strong_fixture_ids"]:
        key = "injected_drift_detections" if name == "half_drift" else "injected_lag_detections"
        gates[name] = (
            groups[f"{name}/b{primary}"][key] >= acceptance["minimum_strong_fault_detections"]
        )
    gates["integrity_faults"] = all(row["detected"] for row in integrity["faults"].values())
    gates["unsupported_cases"] = all(
        row["result"] == "insufficient_evidence"
        for pair in integrity["support"].values()
        for row in pair
    )
    gates["batch_blind_spot_exposed"] = bool(
        limitations["within_batch"]["fault_game_lag_one"] > 0.9
        and all(
            row["result"] == "no_concern_detected"
            for row in limitations["within_batch"]["batch_results"]
        )
    )
    gates["common_root_fault_exposed"] = (
        limitations["common_fault_roots"]["within_each_root_warning_count"] > 0
    )
    group_summaries = {
        key: {
            **dict(group),
            "family_alert_rate": group["family_alerts"] / group["fixtures"],
            "injected_sentinel_detection_rate": group["injected_sentinel_detections"]
            / group["fixtures"],
        }
        for key, group in groups.items()
    }
    summary = {
        "contract_id": recipe["contract_id"],
        "recipe_id": recipe["recipe_id"],
        "method_id": METHOD_ID,
        "plan_sha256": plan_hash,
        "status": (
            "declared_capability_demonstrated" if all(gates.values()) else "capability_gate_failed"
        ),
        "gates": gates,
        "groups": group_summaries,
        "integrity_and_support": integrity,
        "blind_spots": limitations,
        "units": sorted(manifest, key=lambda entry: entry["path"]),
        "multiplicity": recipe["multiplicity"],
        "production_preflight_artifact": False,
    }
    # Preserve scientific summary bytes/mtime on compatible no-force replay.
    summary_path = output / "summary.json"
    if not _json_matches(summary_path, summary):
        write_json(summary_path, summary)
    digest_path = output / "evidence_digest.json"
    digest = {"plan_sha256": plan_hash, "summary_sha256": sha256_file(summary_path)}
    if not _json_matches(digest_path, digest):
        write_json(digest_path, digest)
    if not verify_evidence(output):
        raise RuntimeError("capability evidence inventory failed verification")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "gates": gates,
                "unit_count": len(manifest),
                "reused_units": reused,
                "workers": jobs,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe", type=Path, default=ROOT / "configs/validation/task5b_capability.json"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    summary = run(args.recipe, force=args.force, workers=args.workers)
    if summary["status"] != "declared_capability_demonstrated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
