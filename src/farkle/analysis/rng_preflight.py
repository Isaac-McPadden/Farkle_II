"""Bounded, validation-only first-release preflight with durable check units.

No validation game or capability fixture is an input to production inference.
The full matrix uses the real shuffle/game implementation, with one shuffle as
the publication unit. Existing production publication fault oracles supplement
this deliberately small matrix.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.rng_evidence import BLIND_SPOTS, CONTRACT_ID, finish, inventory, read, reusable
from farkle.config import AppConfig, effective_config_dict, load_app_config
from farkle.simulation.game_profile import GameProfile, TournamentMaxRoundsOverride
from farkle.simulation.run_tournament import (
    ShuffleTask,
    TournamentConfig,
    _init_worker,
    _play_one_shuffle,
)
from farkle.simulation.runner import _resolve_strategies
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
    process_tree_rss_bytes,
    resolve_mp_context,
)
from farkle.utils.random import (
    RNG_SCHEME_VERSION,
    RandomPurpose,
    coordinate_entropy,
    coordinate_seed,
)

ROOT = Path(__file__).resolve().parents[3]
KS = (2, 3, 4, 5, 6, 8, 10, 12)
ROOTS = (1101, 2202)


def environment(cfg: AppConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dependencies": {
            name: version(name) for name in ("numpy", "pyarrow", "pandas", "scipy", "psutil")
        },
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "start_method": "spawn",
        "simulation_workers": cfg.sim.n_jobs,
        "native_threads": 1,
        "storage": str(cfg.results_root.anchor),
        "scratch": str(cfg.io.temp_dir),
    }


def _limits() -> None:
    apply_native_thread_limits(StageParallelPolicy(2, 2, 1, 1, 1))


def _matrix_unit(task: tuple) -> tuple[str, dict[str, Any]]:
    key, root, k, shuffle, strategies = task
    _limits()
    profile = GameProfile(
        default_target_score=10000,
        default_max_rounds=200,
        tournament_max_rounds_overrides=(TournamentMaxRoundsOverride(root, k, 0, 0, 0),),
    )
    config = TournamentConfig(
        n_players=k, num_shuffles=4, n_strategies=120, deterministic_batch_size=2
    )
    _init_worker(strategies, config, profile)
    work = ShuffleTask(
        root,
        k,
        shuffle,
        int(
            coordinate_seed(
                RandomPurpose.SHUFFLE_PERMUTATION, root_seed=root, k=k, shuffle_index=shuffle
            )
        ),
        shuffle // 2,
    )
    wins, sums, squares, rows = _play_one_shuffle(work, collect_rows=True)
    # Normalize NumPy scalars while retaining exact integer and float values.
    payload = json.loads(
        json.dumps(
            {"rows": rows, "wins": dict(wins), "sums": sums, "squares": squares},
            default=lambda value: value.item(),
        )
    )
    expected = {(root, k, shuffle, game) for game in range(120 // k)}
    observed = [(r["root_seed"], r["k"], r["shuffle_index"], r["game_index"]) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("preflight game coordinate ownership failure")
    expected_roster = sorted(int(s.strategy_id) for s in strategies)
    roster = sorted(int(r[f"P{seat}_strategy"]) for r in rows for seat in range(1, k + 1))
    if roster != expected_roster:
        raise ValueError("preflight roster multiplicity failure")
    completed = sum(r["termination_status"] == "completed" for r in rows)
    safety = len(rows) - completed
    if sum(wins.values()) != completed or (shuffle == 0 and safety < 1):
        raise ValueError("preflight completion/win/safety conservation failure")
    for row in rows:
        if row["termination_status"] == "safety_limit" and row["winner_seat"] is not None:
            raise ValueError("safety attempt acquired a winner")
    payload["conservation"] = {
        "attempted": 120,
        "completed": k * completed,
        "safety": k * safety,
        "wins": completed,
        "losses_completed": (k - 1) * completed,
    }
    return key, payload


def _capability_driver():
    path = ROOT / "scripts/task5b_capability.py"
    spec = importlib.util.spec_from_file_location("task5b_capability", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capability_unit(task: tuple) -> tuple[str, dict[str, Any]]:
    seed, recipe = task
    _limits()
    driver = _capability_driver()
    return str(seed), {
        "fixtures": [
            driver.evaluate_fixture(recipe, seed, b, fault)
            for fault in recipe["fixtures"]
            for b in recipe["batch_counts"]
        ]
    }


def capability_schema() -> pa.Schema:
    """Keep all per-check evidence typed in a bounded twelve-fixture seed unit."""
    check = pa.struct(
        [
            ("sentinel", pa.int32()),
            ("metric", pa.string()),
            ("check_id", pa.string()),
            ("support", pa.int64()),
            ("statistic", pa.float64()),
            ("effect_threshold", pa.float64()),
            ("result", pa.string()),
            ("reason", pa.string()),
            ("method_id", pa.string()),
        ]
    )
    fields: list[tuple[str, pa.DataType]] = [
        ("seed", pa.int64()),
        ("batches", pa.int32()),
        ("fixture", pa.string()),
        ("series_count", pa.int32()),
        ("check_opportunities", pa.int32()),
        ("warning_count", pa.int32()),
        ("family_alert", pa.bool_()),
        ("injected_sentinel_alert", pa.bool_()),
        ("injected_lag_alert", pa.bool_()),
        ("injected_drift_alert", pa.bool_()),
        ("insufficient_count", pa.int32()),
        ("checks", pa.list_(check)),
    ]
    return pa.schema(fields)


def _deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("preflight exceeded its declared 1800-second execution ceiling")


def _tests(cfg: AppConfig, plan: Path, recipe: dict[str, Any], deadline: float) -> Path:
    path = cfg.rng_preflight_dir() / "correctness.json"
    if reusable(cfg, path, [plan]):
        return path
    nodes = sorted({node for check in recipe["checks"] for node in check["existing_test_nodes"]})
    with tempfile.TemporaryDirectory(prefix="farkle-preflight-", dir=cfg.io.temp_dir) as temporary:
        report = Path(temporary) / "tests.xml"
        command = [sys.executable, "-m", "pytest", "-q", *nodes, f"--junitxml={report}"]
        started = time.monotonic()
        test_environment = dict(os.environ)
        test_environment.pop("PYTEST_ADDOPTS", None)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=test_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            output, _ = process.communicate(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            process.kill()
            process.communicate()
            raise TimeoutError("preflight correctness workload timed out") from None
        cases = (
            []
            if not report.exists()
            else [
                {
                    "classname": case.attrib.get("classname"),
                    "name": case.attrib.get("name"),
                    "seconds": case.attrib.get("time"),
                    "result": (
                        "fail"
                        if any(case.find(n) is not None for n in ("failure", "error", "skipped"))
                        else "pass"
                    ),
                }
                for case in ET.parse(report).iter("testcase")
            ]
        )
        payload = {
            "invocation": command[:-1],
            "selected_nodes": nodes,
            "cases": cases,
            "elapsed_seconds": time.monotonic() - started,
            "exit_code": process.returncode,
            "log_tail": output.decode("utf-8", errors="replace")[-100000:],
        }
        executed = {
            str(case["classname"]).replace(".", "/") + ".py::" + str(case["name"]).split("[")[0]
            for case in cases
        }
        if (
            process.returncode != 0
            or not cases
            or any(c["result"] != "pass" for c in cases)
            or set(nodes) != executed
        ):
            from farkle.analysis.rng_evidence import publish

            publish(cfg, cfg.rng_preflight_dir() / "_failures/correctness.json", payload, [plan])
            raise RuntimeError(
                "preflight correctness checks failed; see preflight/_failures/correctness.json"
            )
        finish(cfg, path, payload, [plan])
    return path


def run(cfg: AppConfig, *, force: bool = False) -> tuple[Path, dict[str, Any]]:
    """Execute/reuse the frozen matrix, supplemental oracles and capability units."""
    started = time.monotonic()
    deadline = started + 1800
    peak_rss = process_tree_rss_bytes()
    io_start = psutil.Process().io_counters()._asdict()
    recipe = read(ROOT / "docs/implementation/task5a_recipe.json")
    capability = read(ROOT / "configs/validation/task5b_capability.json")
    lock = read(ROOT / "docs/implementation/task5b_protocol_lock.json")
    for relative, digest in lock["sha256"].items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError(
                "frozen capability protocol changed; a versioned reassessment is required"
            )
    mega = load_app_config(ROOT / "configs/farkle_mega_config.yaml")
    strategies = sorted(_resolve_strategies(mega, None)[0], key=lambda s: cast(int, s.strategy_id))
    if len(strategies) < 120:
        raise ValueError("preflight requires the frozen legal mega roster")
    selected = [strategies[i * (len(strategies) - 1) // 119] for i in range(120)]
    if cfg.sim.n_jobs is None:
        raise ValueError("preflight requires an explicit simulation worker count")
    workers = int(cfg.sim.n_jobs)
    if workers < 1 or workers > 16:
        raise ValueError("preflight topology extension supports explicit worker counts 1..16 only")
    # Resolve once per invocation, before work. Individual publication helpers
    # consume this same captured identity; a later invocation resolves afresh.
    configured_code = getattr(cfg, "_code_identity", None)
    policy = (
        CodeIdentityPolicy(configured_code.policy)
        if configured_code
        else CodeIdentityPolicy.DEVELOPMENT_DIRTY
    )
    current_code = resolve_code_identity(ROOT, policy=policy)
    if configured_code is not None and configured_code != current_code:
        raise ValueError("preflight code differs from the authenticated current run")
    cfg._code_identity = current_code
    code = asdict(current_code)
    specification = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "recipe": recipe,
        "capability_recipe": capability,
        "code": code,
        "environment": environment(cfg),
        "target_config": effective_config_dict(cfg),
        "target_game_profile_sha256": cfg._game_profile_sha256,
        "roster": [str(s) for s in selected],
        "roster_ids": [s.strategy_id for s in selected],
        "mega_config_sha256": sha256_file(ROOT / "configs/farkle_mega_config.yaml"),
        "topology_extension": {
            "id": "target_workers_v1",
            "workers": workers,
            "additional_attempts": 1688 if workers > 2 else 0,
            "aggregate_worker_bound": max(2, workers),
            "maximum_total_game_attempts": 5486 + (1688 if workers > 2 else 0),
            "maximum_total_player_exposures": 24960 + (7680 if workers > 2 else 0),
        },
        "rng": {
            "scheme": RNG_SCHEME_VERSION,
            "engine": "PCG64DXSM",
            "purposes": {p.name: int(p) for p in RandomPurpose},
            "coordinate_layout": [
                "root_seed",
                "k",
                "shuffle_index",
                "pair_id",
                "order",
                "game_index",
                "seat_index",
                "replicate_index",
            ],
        },
        "driver_sha256": sha256_file(ROOT / "scripts/task5b_capability.py"),
        "test_source_sha256": {
            relative: sha256_file(ROOT / relative)
            for relative in sorted(
                {
                    node.split("::")[0]
                    for check in recipe["checks"]
                    for node in check["existing_test_nodes"]
                }
            )
        },
    }
    variant_names = ["one", "two", "resume"] + (["target"] if workers > 2 else [])
    specification["required_source_paths"] = sorted(
        [
            "plan.json",
            "correctness.json",
            "resources.json",
            *(
                f"matrix/{v}-{root}-{k}-{s}.json"
                for v in variant_names
                for root in ROOTS
                for k in KS
                for s in range(4)
            ),
            *(
                f"matrix/{kind}-{root}-{k}.json"
                for kind in ("injection", "injection-attempt")
                for root in ROOTS
                for k in KS
            ),
            *(f"capability/{seed}.parquet" for seed in capability["fixture_root_seeds"]),
        ]
    )
    specification = json.loads(json.dumps(specification))
    plan = cfg.rng_preflight_dir() / "plan.json"
    if force or not reusable(cfg, plan, []) or read(plan) != specification:
        finish(cfg, plan, specification, [])
    paths = [plan]
    # Entire small coordinate entropy set, rather than adjacent-only comparison.
    entropies = [
        tuple(coordinate_entropy(p, root_seed=root, k=k, shuffle_index=s, seat_index=seat))
        for p in (RandomPurpose.TOURNAMENT_GAME, RandomPurpose.TOURNAMENT_PLAYER)
        for root in ROOTS
        for k in KS
        for s in range(4)
        for seat in range(1, k + 1)
    ]
    if len(set(entropies)) != len(entropies):
        raise ValueError("preflight full coordinate entropy collision")
    variants = [("one", 1), ("two", 2), ("resume", 2)]
    if workers > 2:
        variants.append(("target", workers))
    for variant, jobs in variants:
        # The resumed variant authenticates 0/1, stages and interrupts shuffle 2,
        # then changes to one worker. Only the unauthenticated shuffle is replayed.
        phases = [(range(2), jobs), (range(2, 4), 1)] if variant == "resume" else [(range(4), jobs)]
        for phase_index, (shuffles, phase_jobs) in enumerate(phases):
            if variant == "resume" and phase_index == 1:
                for root in ROOTS:
                    for k in KS:
                        marker = cfg.rng_preflight_dir() / "matrix" / f"injection-{root}-{k}.json"
                        attempted = marker.with_name(f"injection-attempt-{root}-{k}.json")
                        paths.append(attempted)
                        if not reusable(cfg, marker, [plan]):
                            if reusable(cfg, attempted, [plan]) and not force:
                                raise RuntimeError(
                                    "preflight injection execution incomplete; explicit force required, no automatic reinjection"
                                )
                            finish(
                                cfg, attempted, {"maximum_injections": 1, "started": True}, [plan]
                            )
                            _, staged = _matrix_unit(("injected", root, k, 2, selected))
                            # Exercise the same atomic data+sidecar publisher's
                            # before-ownership failure boundary, without canonical bytes.
                            from farkle.analysis.rng_evidence import metadata
                            from farkle.utils.artifact_contract import (
                                write_artifact_with_sidecar_atomic,
                            )

                            target = marker.with_name(f"interrupted-{root}-{k}.json")

                            def interrupted(destination: Path, staged=staged) -> None:
                                destination.write_text(json.dumps(staged), encoding="utf-8")
                                raise InterruptedError("declared preflight fault")

                            with contextlib.suppress(InterruptedError):
                                write_artifact_with_sidecar_atomic(
                                    target, metadata(cfg, target, [plan], []), interrupted
                                )
                            if target.exists():
                                raise ValueError("interrupted preflight unit became canonical")
                            finish(cfg, marker, {"injected": True, "unowned_shuffle": 2}, [plan])
                        paths.append(marker)
            tasks: list[tuple[Any, ...]] = []
            for root in ROOTS:
                for k in KS:
                    for shuffle in shuffles:
                        key = f"{variant}-{root}-{k}-{shuffle}"
                        path = cfg.rng_preflight_dir() / "matrix" / f"{key}.json"
                        paths.append(path)
                        if force or not reusable(cfg, path, [plan]):
                            tasks.append((key, root, k, shuffle, selected))
            _deadline(deadline)
            for key, payload in process_map(
                _matrix_unit,
                tasks,
                n_jobs=phase_jobs,
                mp_context=resolve_mp_context("spawn"),
                window=phase_jobs,
            ):
                _deadline(deadline)
                peak_rss = max(peak_rss, process_tree_rss_bytes())
                finish(cfg, cfg.rng_preflight_dir() / "matrix" / f"{key}.json", payload, [plan])
    for root in ROOTS:
        for k in KS:
            for shuffle in range(4):
                baseline = read(
                    cfg.rng_preflight_dir() / "matrix" / f"one-{root}-{k}-{shuffle}.json"
                )
                for variant, _ in variants[1:]:
                    if (
                        read(
                            cfg.rng_preflight_dir()
                            / "matrix"
                            / f"{variant}-{root}-{k}-{shuffle}.json"
                        )
                        != baseline
                    ):
                        raise ValueError(
                            "worker/resume logical rows or scientific summaries differ"
                        )
    paths.append(_tests(cfg, plan, recipe, deadline))
    tasks = []
    capability_paths = []
    for seed in capability["fixture_root_seeds"]:
        path = cfg.rng_preflight_dir() / "capability" / f"{seed}.parquet"
        capability_paths.append(path)
        if force or not reusable(cfg, path, [plan]):
            tasks.append((seed, capability))
    for seed, payload in process_map(
        _capability_unit, tasks, n_jobs=2, mp_context=resolve_mp_context("spawn"), window=2
    ):
        _deadline(deadline)
        peak_rss = max(peak_rss, process_tree_rss_bytes())
        table = pa.Table.from_pylist(payload["fixtures"], schema=capability_schema())
        finish(cfg, cfg.rng_preflight_dir() / "capability" / f"{seed}.parquet", table, [plan])
    paths.extend(capability_paths)
    groups: dict[str, Counter] = {}
    for path in capability_paths:
        for result in pq.read_table(path).to_pylist():
            group = groups.setdefault(f"{result['fixture']}/b{result['batches']}", Counter())
            group.update(
                fixtures=1,
                family_alerts=int(result["family_alert"]),
                lag_detections=int(result["injected_lag_alert"]),
                drift_detections=int(result["injected_drift_alert"]),
            )
    acceptance = capability["acceptance"]
    gate = groups["clean/b100"]["family_alerts"] <= acceptance["maximum_clean_family_alerts"]
    for name in acceptance["strong_fixture_ids"]:
        field = "drift_detections" if name == "half_drift" else "lag_detections"
        gate &= groups[f"{name}/b100"][field] >= acceptance["minimum_strong_fault_detections"]
    driver = _capability_driver()
    integrity, blind = driver.integrity_and_support(capability), driver.blind_spots(capability)
    gate &= all(row["detected"] for row in integrity["faults"].values())
    gate &= all(
        row["result"] == "insufficient_evidence"
        for pair in integrity["support"].values()
        for row in pair
    )
    gate &= blind["within_batch"]["fault_game_lag_one"] > 0.9
    gate &= all(
        r["result"] == "no_concern_detected" for r in blind["within_batch"]["batch_results"]
    )
    gate &= blind["common_fault_roots"]["within_each_root_warning_count"] > 0
    if not gate:
        raise ValueError("preflight capability acceptance gate failed")
    resources = cfg.rng_preflight_dir() / "resources.json"
    if force or not reusable(cfg, resources, [plan]):
        io_stop = psutil.Process().io_counters()._asdict()
        finish(
            cfg,
            resources,
            {
                "elapsed_seconds": time.monotonic() - started,
                "peak_sampled_process_tree_rss_bytes": peak_rss,
                "sampling": "at_matrix_and_capability_unit_completions",
                "parent_io_delta": {key: io_stop[key] - io_start[key] for key in io_start},
                "worker_topologies": sorted({1, 2, workers}),
                "note": "parent I/O excludes child process I/O",
            },
            [plan],
        )
    paths.append(resources)
    summary = cfg.rng_preflight_dir() / "publication.json"
    result = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "complete_valid",
        "code_identity": identity_sha256(code),
        "workload_identity": identity_sha256(specification),
        "environment": environment(cfg),
        "groups": groups,
        "integrity_and_support": integrity,
        "blind_spot_results": blind,
        "blind_spots": BLIND_SPOTS,
        "statistical_matrix_result": "insufficient_evidence",
        "matrix_batches": 2,
        "covered_ks": list(KS),
        "covered_workers": sorted({1, 2, workers}),
        "sources": inventory(paths, cfg.rng_preflight_dir()),
    }
    if not reusable(cfg, summary, paths) or read(summary) != result:
        finish(cfg, summary, result, paths)
    return summary, result
