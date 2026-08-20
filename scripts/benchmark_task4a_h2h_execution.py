"""Bounded real-path benchmark for Task 4A H2H checkpoint execution.

The benchmark builds disposable, authenticated H2H plans and calls the real
``execute_h2h_schedule`` process path.  Only the exact-power search is replaced
during fixture construction so bounded fixtures can hold review-selected block
targets without manufacturing game outcomes.  Every attempted game still uses
the production engine and semantic RNG coordinates.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import shutil
import statistics
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis import h2h_schedule
from farkle.analysis.h2h_schedule import execute_h2h_schedule, plan_h2h_schedule
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.simulation.game_profile import GameProfile, H2HMaxRoundsOverride
from farkle.simulation.strategies import ThresholdStrategy, build_strategy_manifest
from farkle.utils.artifact_contract import make_artifact_sidecar
from farkle.utils.artifacts import write_json_artifact_atomic, write_parquet_artifact_atomic
from farkle.utils.telemetry import SupervisorHeartbeatRecorder, use_supervisor_recorder
from farkle.utils.writer import atomic_path

BENCHMARK_SCHEMA_VERSION = 2
OWNERSHIP_MARKER = ".task4a-benchmark-owned.json"


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    target: int
    profile_kind: str = "normal"


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    workers: int = 2
    repetitions: int = 1
    targets: tuple[int, ...] = (1_372, 1_974, 2_191)
    include_exceptional: bool = True
    heartbeat_seconds: float = 0.1


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_digest(paths: Sequence[Path]) -> str:
    payload = [(path.name, _sha256(path)) for path in sorted(paths)]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _write_owned_marker(root: Path, settings: BenchmarkSettings) -> None:
    root.mkdir(parents=True, exist_ok=True)
    marker = root / OWNERSHIP_MARKER
    payload = {
        "benchmark": "task4a_h2h_execution",
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "settings": asdict(settings),
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_owned_root(root: Path, settings: BenchmarkSettings) -> None:
    marker = root / OWNERSHIP_MARKER
    if not marker.is_file():
        raise RuntimeError(f"refusing to operate on unowned benchmark root: {root}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("benchmark") != "task4a_h2h_execution":
        raise RuntimeError(f"invalid Task 4A ownership marker: {marker}")
    if payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise RuntimeError(f"stale Task 4A benchmark schema in ownership marker: {marker}")
    if _canonical_json(payload.get("settings")) != _canonical_json(asdict(settings)):
        raise RuntimeError(f"Task 4A benchmark settings conflict with ownership marker: {marker}")


def _cfg(root: Path, *, profile: GameProfile | None, workers: int) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=root / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2, 4]),
    )
    cfg.head2head.n_jobs = workers
    cfg.head2head.total_game_cap = None
    cfg.analysis.mp_start_method = "spawn"
    cfg.resources.minimum_system_available_memory_mb = min(
        cfg.resources.minimum_system_available_memory_mb,
        256,
    )
    cfg._game_profile_sha256 = profile.sha256 if profile is not None else None
    return cfg


def _write_frozen_family(cfg: AppConfig) -> None:
    family_hash = "4" * 64
    candidates = [1, 2]
    membership = pd.DataFrame(
        {
            "strategy": pd.array(candidates, dtype="Int32"),
            "final_family": [True, True],
            "family_hash": [family_hash, family_hash],
        }
    )
    manifest: dict[str, object] = {
        "family_hash": family_hash,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "root_seeds": [11, 22],
        "single_root_execution": False,
    }
    common: dict[str, Any] = {
        "producer": "task4a_benchmark",
        "scope": ArtifactScope.H2H_2P,
        "source_scope": ArtifactScope.CROSS_SEED,
        "operation": "candidate_family_freeze",
        "player_counts": [2],
        "required_player_counts": [2],
        "missing_cell_policy": "fail",
        "seed_scope": "both_roots_combined",
    }
    membership_path = cfg.h2h_candidate_family_path()
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(membership, preserve_index=False),
        membership_path,
        sidecar=make_artifact_sidecar(
            cfg,
            membership_path,
            consistency_columns=membership.columns.tolist(),
            **common,
        ),
    )
    manifest_path = cfg.h2h_candidate_family_manifest_path()
    write_json_artifact_atomic(
        manifest,
        manifest_path,
        sidecar=make_artifact_sidecar(
            cfg,
            manifest_path,
            consistency_columns=list(manifest),
            **common,
        ),
    )
    strategies = [
        ThresholdStrategy(score_threshold=300, dice_threshold=3, strategy_id=1),
        ThresholdStrategy(score_threshold=700, dice_threshold=2, strategy_id=2),
    ]
    strategy_manifest = cfg.strategy_manifest_root_path()
    strategy_manifest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(build_strategy_manifest(strategies), preserve_index=False),
        strategy_manifest,
    )


@contextlib.contextmanager
def _bounded_power_fixture(target: int) -> Iterator[None]:
    """Replace expensive design search only while publishing a disposable plan."""

    original_minimum = h2h_schedule._minimum_block_games
    original_worst = h2h_schedule._worst_scenario_power
    original_grid = h2h_schedule._power_grid
    h2h_schedule._minimum_block_games = lambda **_kwargs: target
    h2h_schedule._worst_scenario_power = lambda **_kwargs: 0.8
    h2h_schedule._power_grid = lambda *_args, **_kwargs: []
    try:
        yield
    finally:
        h2h_schedule._minimum_block_games = original_minimum
        h2h_schedule._worst_scenario_power = original_worst
        h2h_schedule._power_grid = original_grid


def _profile_for(scenario: Scenario) -> GameProfile | None:
    if scenario.profile_kind == "normal":
        return None
    cap = scenario.target * 2
    if scenario.profile_kind == "low_safety":
        overrides = tuple(
            H2HMaxRoundsOverride(root, 0, order, attempt, 0)
            for root in (11, 22)
            for order in (0, 1)
            for attempt in range(0, cap, 97)
        )
        return GameProfile(h2h_max_rounds_overrides=overrides)
    if scenario.profile_kind == "nonviable":
        return GameProfile(default_max_rounds=0)
    if scenario.profile_kind == "mixed_tail":
        overrides = tuple(H2HMaxRoundsOverride(22, 0, 1, attempt, 0) for attempt in range(cap))
        return GameProfile(h2h_max_rounds_overrides=overrides)
    raise ValueError(f"unknown Task 4A profile kind: {scenario.profile_kind}")


def _policies(target: int) -> tuple[tuple[str, int], ...]:
    return (
        ("baseline_fixed_1000", 1_000),
        ("target_aligned", target),
        ("selected_cap_bounded_5000", 5_000),
    )


def _measurement_path(root: Path, scenario: Scenario, policy: str, repetition: int) -> Path:
    return root / scenario.name / policy / f"rep-{repetition:02d}" / "measurement.json"


def _scenario_root(root: Path, scenario: Scenario, policy: str, repetition: int) -> Path:
    return root / scenario.name / policy / f"rep-{repetition:02d}"


def _measure_once(
    root: Path,
    scenario: Scenario,
    policy: str,
    attempt_limit: int,
    repetition: int,
    settings: BenchmarkSettings,
) -> dict[str, object]:
    measurement_path = _measurement_path(root, scenario, policy, repetition)
    if measurement_path.is_file():
        existing = cast(dict[str, object], json.loads(measurement_path.read_text(encoding="utf-8")))
        if existing.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
            raise RuntimeError(f"stale Task 4A benchmark measurement: {measurement_path}")
        return existing
    run_root = _scenario_root(root, scenario, policy, repetition)
    if run_root.exists():
        shutil.rmtree(run_root)
    profile = _profile_for(scenario)
    cfg = _cfg(run_root, profile=profile, workers=settings.workers)
    _write_frozen_family(cfg)
    with _bounded_power_fixture(scenario.target):
        plan_h2h_schedule(cfg)

    recorder = SupervisorHeartbeatRecorder(
        logging.getLogger("task4a.benchmark"),
        run=f"{scenario.name}:{policy}:{repetition}",
        interval_seconds=settings.heartbeat_seconds,
    )
    scope = recorder.begin_scope(
        "task4a",
        run=f"{scenario.name}:{policy}:{repetition}",
        stage="h2h_execute",
        phase="benchmark",
    )
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        with use_supervisor_recorder(recorder, scope):
            artifacts = execute_h2h_schedule(
                cfg,
                n_jobs=settings.workers,
                chunk_games=attempt_limit,
                oracle_game_profile=profile,
            )
    except BaseException:
        scope.finish(status="failed")
        recorder.close()
        raise
    wall_seconds = time.perf_counter() - started
    cpu_seconds = time.process_time() - cpu_started
    scope.finish(status="success")
    recorder.close()
    summary = recorder.summary()
    completion = cast(
        dict[str, object],
        cast(dict[str, dict[str, object]], summary["completed_progress"])["task4a:h2h_execute"],
    )
    counts = pq.read_table(artifacts.order_counts).to_pandas()
    block_paths = list(artifacts.block_paths)
    measurement: dict[str, object] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "scenario": asdict(scenario),
        "policy": policy,
        "attempt_limit": attempt_limit,
        "repetition": repetition,
        "workers": settings.workers,
        "wall_seconds": wall_seconds,
        "parent_cpu_seconds": cpu_seconds,
        "telemetry": completion,
        "resource_summary": summary["resource_summary"],
        "counts": {
            "blocks": len(counts),
            "attempted": int(counts["games_attempted"].sum()),
            "completed": int(counts["games_completed"].sum()),
            "safety": int(counts["games_safety_limit"].sum()),
            "replacements": int(counts["replacement_attempt_count"].sum()),
            "wins_seat1": int(counts["wins_seat1"].sum()),
            "wins_seat2": int(counts["wins_seat2"].sum()),
            "nonviable": int(counts["completion_status"].eq("unresolved_nonviable").sum()),
        },
        "digests": {
            "aggregate_parquet": _sha256(artifacts.order_counts),
            "block_parquets": _bundle_digest(block_paths),
            "aggregate_schema": hashlib.sha256(
                str(pq.read_schema(artifacts.order_counts)).encode("utf-8")
            ).hexdigest(),
            "logical_rows": hashlib.sha256(
                _canonical_json(counts.to_dict(orient="records"))
            ).hexdigest(),
        },
    }
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(measurement_path)) as staged:
        Path(staged).write_text(
            json.dumps(measurement, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return measurement


def _validate_equivalence(measurements: Sequence[dict[str, object]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for measurement in measurements:
        scenario = cast(dict[str, object], measurement["scenario"])
        key = (str(scenario["name"]), cast(int, measurement["repetition"]))
        grouped.setdefault(key, []).append(measurement)
    for key, group in grouped.items():
        counts = {_canonical_json(item["counts"]) for item in group}
        logical = {str(cast(dict[str, object], item["digests"])["logical_rows"]) for item in group}
        aggregate = {
            str(cast(dict[str, object], item["digests"])["aggregate_parquet"]) for item in group
        }
        blocks = {str(cast(dict[str, object], item["digests"])["block_parquets"]) for item in group}
        if len(counts) != 1 or len(logical) != 1 or len(aggregate) != 1 or len(blocks) != 1:
            raise AssertionError(f"Task 4A policy outputs differ for {key}")


def summarize(measurements: Sequence[dict[str, object]]) -> dict[str, object]:
    _validate_equivalence(measurements)
    by_policy: dict[str, list[dict[str, object]]] = {}
    for item in measurements:
        by_policy.setdefault(str(item["policy"]), []).append(item)
    policies: dict[str, object] = {}
    for policy, rows in sorted(by_policy.items()):
        walls = [cast(float, row["wall_seconds"]) for row in rows]
        telemetry_rows = [cast(dict[str, object], row["telemetry"]) for row in rows]
        policies[policy] = {
            "wall_seconds": walls,
            "median_wall_seconds": statistics.median(walls),
            "scheduled_chunks": sum(cast(int, row["scheduled_chunks"]) for row in telemetry_rows),
            "checkpoint_writes": sum(cast(int, row["checkpoint_writes"]) for row in telemetry_rows),
            "pool_generations": sum(cast(int, row["pool_generations"]) for row in telemetry_rows),
            "worker_initializer_loads": sum(
                cast(int, row["worker_initializer_loads"]) for row in telemetry_rows
            ),
        }
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "measurement_count": len(measurements),
        "exact_policy_equivalence": True,
        "policies": policies,
        "measurements": list(measurements),
    }


def run_benchmark(root: Path, settings: BenchmarkSettings) -> dict[str, object]:
    resolved = root.resolve()
    if resolved.exists():
        _require_owned_root(resolved, settings)
    else:
        if not resolved.name.startswith("farkle-task4a-"):
            raise ValueError("Task 4A benchmark root name must start with 'farkle-task4a-'")
        _write_owned_marker(resolved, settings)
    scenarios = [Scenario(f"normal-{target}", target) for target in settings.targets]
    if settings.include_exceptional:
        scenarios.extend(
            [
                Scenario("low-safety-256", 256, "low_safety"),
                Scenario("mixed-tail-128", 128, "mixed_tail"),
                Scenario("nonviable-128", 128, "nonviable"),
            ]
        )
    measurements: list[dict[str, object]] = []
    # Alternate policy order across repetitions to expose order effects.
    for repetition in range(1, settings.repetitions + 1):
        for scenario in scenarios:
            policies = list(_policies(scenario.target))
            if repetition % 2 == 0:
                policies.reverse()
            for policy, limit in policies:
                measurements.append(
                    _measure_once(
                        resolved,
                        scenario,
                        policy,
                        limit,
                        repetition,
                        settings,
                    )
                )
    summary = summarize(measurements)
    summary_path = resolved / "summary.json"
    with atomic_path(str(summary_path)) as staged:
        Path(staged).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = BenchmarkSettings(
        workers=args.workers,
        repetitions=args.repetitions,
        targets=(12,) if args.quick else (1_372, 1_974, 2_191),
        include_exceptional=not args.quick,
    )
    summary = run_benchmark(args.output, settings)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
