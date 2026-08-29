"""Reproducible real-path benchmark for Task 7B H2H checkpoint bundling.

The driver is intentionally layout-neutral.  A baseline phase is captured with
the pre-Task-7B ``execute_h2h_schedule`` implementation and retained unchanged;
the candidate phase exercises the edited implementation with the same fixture
coordinates.  Fixture power search is replaced only while publishing a bounded
authenticated plan.  Every game is still played by the production engine.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from farkle.analysis import h2h_schedule
from farkle.analysis.h2h_inference import run_h2h_inference
from farkle.analysis.h2h_schedule import execute_h2h_schedule, plan_h2h_schedule
from farkle.analysis.release_audit import audit_sidecar_completeness
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig, assign_config_sha
from farkle.simulation.strategies import ThresholdStrategy, build_strategy_manifest
from farkle.utils.artifact_contract import make_artifact_sidecar, sidecar_path
from farkle.utils.artifacts import write_json_artifact_atomic, write_parquet_artifact_atomic
from farkle.utils.authenticated_contract import CodeIdentityPolicy, resolve_code_identity
from farkle.utils.authentication_telemetry import (
    AuthenticationTelemetry,
    use_authentication_telemetry,
)
from farkle.utils.telemetry import SupervisorHeartbeatRecorder, use_supervisor_recorder
from farkle.utils.writer import atomic_path

BENCHMARK_SCHEMA_VERSION: Final = 1
SAFE_ROOT_NAME: Final = "farkle-task7b-h2h-bundling-v1"
MARKER_NAME: Final = "task7b_ownership.json"
DEFAULT_CASES: Final = (
    ("accepted-180-block-shape", 10, 1_372),
    ("production-shaped-760-block-inventory", 20, 1_372),
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One deterministic real-engine schedule shape."""

    name: str
    candidate_count: int
    completed_target: int

    @property
    def expected_pairs(self) -> int:
        return self.candidate_count * (self.candidate_count - 1) // 2

    @property
    def expected_blocks(self) -> int:
        return self.expected_pairs * 4


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    """Settings shared by baseline and candidate phases."""

    workers: int = 15
    repetitions: int = 3
    heartbeat_seconds: float = 1.0
    cases: tuple[BenchmarkCase, ...] = tuple(BenchmarkCase(*row) for row in DEFAULT_CASES)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def expected_checkpoint_count(phase: str, logical_blocks: int) -> int:
    """Return the physical-layout oracle for one benchmark phase."""

    if phase == "baseline":
        return logical_blocks
    if phase == "candidate":
        return (logical_blocks + 15) // 16
    raise ValueError("Task 7B benchmark phase must be baseline or candidate")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _marker_settings(settings: BenchmarkSettings) -> dict[str, object]:
    return cast(dict[str, object], json.loads(_canonical_bytes(asdict(settings))))


def prepare_owned_root(root: Path, settings: BenchmarkSettings) -> Path:
    """Create or validate the exact Task 7B owned root without broad cleanup."""

    resolved = root.resolve()
    if resolved.name != SAFE_ROOT_NAME:
        raise ValueError(f"Task 7B benchmark root must have exact name {SAFE_ROOT_NAME!r}")
    marker = resolved / MARKER_NAME
    if resolved.exists():
        if not marker.is_file():
            raise RuntimeError(f"refusing to use unowned Task 7B root: {resolved}")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        expected = {
            "task": "7B",
            "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
            "resolved_root": str(resolved),
            "settings": _marker_settings(settings),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Task 7B ownership marker or settings mismatch: {marker}")
        return resolved
    resolved.mkdir(parents=True)
    payload = {
        "task": "7B",
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "resolved_root": str(resolved),
        "settings": _marker_settings(settings),
        "starting_head": _head(),
        "created_ns": time.time_ns(),
    }
    with atomic_path(str(marker)) as staged:
        Path(staged).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return resolved


def _cfg(root: Path, *, workers: int) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=root / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 22], n_players_list=[2, 4]),
    )
    cfg.head2head.n_jobs = workers
    cfg.head2head.total_game_cap = None
    cfg.analysis.mp_start_method = "spawn"
    # Match the machine-targeted production topology instead of AppConfig's
    # deliberately small portable defaults.
    cfg.resources.scheduler_memory_budget_mb = 8_192
    cfg.resources.process_tree_warning_threshold_mb = 8_192
    cfg.resources.aggregate_memory_hard_limit_mb = 12_288
    # Keep a conservative reserve while allowing the bounded benchmark on the
    # currently loaded 32-GiB host.  The aggregate hard limit and worker policy
    # remain production-shaped.
    cfg.resources.minimum_system_available_memory_mb = 4_096
    cfg.resources.parent_process_memory_mb = 512
    cfg.resources.logical_cpu_budget = 15
    cfg._code_identity = resolve_code_identity(
        Path(__file__).resolve().parents[1],
        policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY,
    )
    assign_config_sha(cfg)
    return cfg


def _write_frozen_family(cfg: AppConfig, candidate_count: int) -> None:
    family_hash = hashlib.sha256(f"task7b-family:{candidate_count}".encode("ascii")).hexdigest()
    candidates = list(range(1, candidate_count + 1))
    membership = pd.DataFrame(
        {
            "strategy": pd.array(candidates, dtype="Int32"),
            "final_family": [True] * candidate_count,
            "family_hash": [family_hash] * candidate_count,
        }
    )
    manifest: dict[str, object] = {
        "family_hash": family_hash,
        "candidates": candidates,
        "candidate_count": candidate_count,
        "root_seeds": [11, 22],
        "single_root_execution": False,
    }
    common: dict[str, Any] = {
        "producer": "task7b_benchmark",
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
        ThresholdStrategy(
            score_threshold=250 + 17 * strategy_id,
            dice_threshold=1 + strategy_id % 6,
            strategy_id=strategy_id,
        )
        for strategy_id in candidates
    ]
    strategy_manifest = cfg.strategy_manifest_root_path()
    strategy_manifest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(build_strategy_manifest(strategies), preserve_index=False),
        strategy_manifest,
    )


@contextlib.contextmanager
def _bounded_power_fixture(target: int) -> Iterator[None]:
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


def _arrow_digest(table: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _path_digest(path: Path) -> dict[str, object]:
    table = pq.read_table(path)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "schema": str(table.schema),
        "logical_arrow_sha256": _arrow_digest(table),
        "rows": table.num_rows,
    }


def _file_inventory(root: Path) -> dict[str, object]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    parquets = [path for path in files if path.suffix.lower() == ".parquet"]
    sidecars = [path for path in files if path.name.endswith(".sidecar.json")]
    unit_stamps = [path for path in files if path.name.endswith(".unit.done.json")]
    manifests = [path for path in files if "manifest" in path.name.lower()]
    footer_bytes = 0
    for path in parquets:
        metadata = pq.ParquetFile(path).metadata
        footer_bytes += int(metadata.serialized_size)
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "parquet_files": len(parquets),
        "parquet_bytes": sum(path.stat().st_size for path in parquets),
        "footer_bytes": footer_bytes,
        "sidecar_files": len(sidecars),
        "sidecar_bytes": sum(path.stat().st_size for path in sidecars),
        "unit_stamp_files": len(unit_stamps),
        "unit_stamp_bytes": sum(path.stat().st_size for path in unit_stamps),
        "manifest_files": len(manifests),
        "manifest_bytes": sum(path.stat().st_size for path in manifests),
    }


def _measurement_path(
    root: Path,
    phase: str,
    case: BenchmarkCase,
    repetition: int,
) -> Path:
    return root / phase / case.name / f"rep-{repetition:02d}" / "measurement.json"


def _run_root(root: Path, phase: str, case: BenchmarkCase, repetition: int) -> Path:
    return root / phase / case.name / f"rep-{repetition:02d}" / "run"


def _measure_once(
    root: Path,
    *,
    phase: str,
    case: BenchmarkCase,
    repetition: int,
    settings: BenchmarkSettings,
) -> dict[str, object]:
    measurement_path = _measurement_path(root, phase, case, repetition)
    if measurement_path.is_file():
        payload = cast(
            dict[str, object],
            json.loads(measurement_path.read_text(encoding="utf-8")),
        )
        if payload.get("benchmark_schema_version") != BENCHMARK_SCHEMA_VERSION:
            raise RuntimeError(f"stale Task 7B measurement: {measurement_path}")
        return payload
    run_root = _run_root(root, phase, case, repetition)
    if run_root.exists():
        raise RuntimeError(f"refusing to overwrite incomplete Task 7B run: {run_root}")
    cfg = _cfg(run_root, workers=settings.workers)
    _write_frozen_family(cfg, case.candidate_count)
    with _bounded_power_fixture(case.completed_target):
        schedule_artifacts = plan_h2h_schedule(cfg)
    schedule = pq.read_table(schedule_artifacts.block_manifest)
    if schedule.num_rows != case.expected_blocks:
        raise AssertionError(
            f"executable schedule produced {schedule.num_rows}, expected {case.expected_blocks}"
        )

    recorder = SupervisorHeartbeatRecorder(
        logging.getLogger("task7b.benchmark"),
        run=f"{phase}:{case.name}:{repetition}",
        interval_seconds=settings.heartbeat_seconds,
    )
    scope = recorder.begin_scope(
        "task7b",
        run=f"{phase}:{case.name}:{repetition}",
        stage="h2h_execute",
        phase="benchmark",
    )
    authentication = AuthenticationTelemetry()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        with (
            use_authentication_telemetry(authentication),
            use_supervisor_recorder(recorder, scope),
        ):
            execution = execute_h2h_schedule(cfg, n_jobs=settings.workers)
            inference_started = time.perf_counter()
            inference = run_h2h_inference(cfg)
            inference_seconds = time.perf_counter() - inference_started
            audit_started = time.perf_counter()
            audit_failures = audit_sidecar_completeness(cfg.analysis_dir)
            audit_seconds = time.perf_counter() - audit_started
    except BaseException:
        scope.finish(status="failed")
        recorder.close()
        raise
    wall_seconds = time.perf_counter() - wall_started
    parent_cpu_seconds = time.process_time() - cpu_started
    scope.finish(status="success")
    recorder.close()
    if audit_failures:
        raise AssertionError(f"Task 7B fixture release audit failed: {audit_failures}")
    recorder_summary = recorder.summary()
    completion = cast(
        dict[str, object],
        cast(dict[str, dict[str, object]], recorder_summary["completed_progress"])[
            "task7b:h2h_execute"
        ],
    )
    counts_table = pq.read_table(execution.order_counts)
    counts = counts_table.to_pandas()
    checkpoint_paths = list(execution.block_paths)
    expected_physical_count = expected_checkpoint_count(phase, case.expected_blocks)
    if len(checkpoint_paths) != expected_physical_count:
        raise AssertionError(
            f"{phase} produced {len(checkpoint_paths)} physical checkpoints; "
            f"expected {expected_physical_count}"
        )
    inference_paths = list(inference.all_paths)
    measurement: dict[str, object] = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "phase": phase,
        "case": asdict(case),
        "repetition": repetition,
        "workers": settings.workers,
        "head": _head(),
        "wall_seconds": wall_seconds,
        "parent_cpu_seconds": parent_cpu_seconds,
        "inference_seconds": inference_seconds,
        "release_audit_seconds": audit_seconds,
        "release_audit_failures": audit_failures,
        "telemetry": completion,
        "resource_summary": recorder_summary["resource_summary"],
        "authentication_telemetry": authentication.as_metadata(),
        "schedule": {
            "rows": schedule.num_rows,
            "schema": str(schedule.schema),
            "logical_arrow_sha256": _arrow_digest(schedule),
            "parquet_sha256": _sha256(schedule_artifacts.block_manifest),
            "records": schedule.to_pylist(),
        },
        "logical_counts": {
            "rows": counts_table.num_rows,
            "schema": str(counts_table.schema),
            "logical_arrow_sha256": _arrow_digest(counts_table),
            "records": counts_table.to_pylist(),
            "attempted": int(counts["games_attempted"].sum()),
            "completed": int(counts["games_completed"].sum()),
            "safety": int(counts["games_safety_limit"].sum()),
            "replacements": int(counts["replacement_attempt_count"].sum()),
            "wins_seat1": int(counts["wins_seat1"].sum()),
            "wins_seat2": int(counts["wins_seat2"].sum()),
            "nonviable": int(counts["completion_status"].eq("unresolved_nonviable").sum()),
        },
        "canonical_artifacts": {
            "execution_state": {
                "bytes": execution.execution_state.stat().st_size,
                "sha256": _sha256(execution.execution_state),
                "payload": json.loads(execution.execution_state.read_text(encoding="utf-8")),
            },
            "aggregate": _path_digest(execution.order_counts),
            "inference": [_path_digest(path) for path in inference_paths],
        },
        "physical_checkpoints": {
            "count": len(checkpoint_paths),
            "paths": [path.name for path in checkpoint_paths],
            "data_bytes": sum(path.stat().st_size for path in checkpoint_paths),
            "sidecar_bytes": sum(sidecar_path(path).stat().st_size for path in checkpoint_paths),
            "inventory_sha256": hashlib.sha256(
                _canonical_bytes(
                    [
                        (path.name, _sha256(path), _sha256(sidecar_path(path)))
                        for path in checkpoint_paths
                    ]
                )
            ).hexdigest(),
        },
        "file_inventory": _file_inventory(cfg.analysis_dir),
    }
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(measurement_path)) as staged:
        Path(staged).write_text(
            json.dumps(measurement, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return measurement


def _validate_phase_equivalence(rows: Sequence[Mapping[str, object]]) -> None:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        case = cast(Mapping[str, object], row["case"])
        grouped.setdefault(str(case["name"]), []).append(row)
    for name, group in grouped.items():
        logical = {
            str(cast(Mapping[str, object], row["logical_counts"])["logical_arrow_sha256"])
            for row in group
        }
        schedules = {
            str(cast(Mapping[str, object], row["schedule"])["logical_arrow_sha256"])
            for row in group
        }
        inference = {
            _canonical_bytes(cast(Mapping[str, object], row["canonical_artifacts"])["inference"])
            for row in group
        }
        if len(logical) != 1 or len(schedules) != 1 or len(inference) != 1:
            raise AssertionError(f"Task 7B repetitions differ logically for {name}")


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    _validate_phase_equivalence(rows)
    cases: dict[str, object] = {}
    for case_name in sorted({str(cast(Mapping[str, object], row["case"])["name"]) for row in rows}):
        group = [
            row for row in rows if str(cast(Mapping[str, object], row["case"])["name"]) == case_name
        ]
        walls = [float(cast(float, row["wall_seconds"])) for row in group]
        checkpoint_walls = [
            float(
                cast(
                    float,
                    cast(Mapping[str, object], row["telemetry"]).get(
                        "bundle_publication_seconds",
                        cast(Mapping[str, object], row["telemetry"]).get(
                            "block_checkpoint_seconds", 0.0
                        ),
                    ),
                )
            )
            for row in group
        ]
        cases[case_name] = {
            "wall_seconds": walls,
            "median_wall_seconds": statistics.median(walls),
            "checkpoint_publication_seconds": checkpoint_walls,
            "median_checkpoint_publication_seconds": statistics.median(checkpoint_walls),
        }
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "measurement_count": len(rows),
        "exact_within_phase_equivalence": True,
        "cases": cases,
        "measurements": list(rows),
    }


def _run_warmup(root: Path, phase: str, settings: BenchmarkSettings) -> None:
    marker = root / phase / "warmup.complete.json"
    if marker.is_file():
        return
    warmup = BenchmarkCase("warmup-12-block", 3, 2)
    _measure_once(
        root,
        phase=phase,
        case=warmup,
        repetition=0,
        settings=replace(settings, workers=min(4, settings.workers)),
    )
    with atomic_path(str(marker)) as staged:
        Path(staged).write_text(
            json.dumps({"phase": phase, "completed_ns": time.time_ns()}, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def run_phase(root: Path, *, phase: str, settings: BenchmarkSettings) -> dict[str, object]:
    """Run one immutable baseline or candidate phase and publish its summary."""

    if phase not in {"baseline", "candidate"}:
        raise ValueError("Task 7B benchmark phase must be baseline or candidate")
    resolved = prepare_owned_root(root, settings)
    _run_warmup(resolved, phase, settings)
    rows: list[dict[str, object]] = []
    for repetition in range(1, settings.repetitions + 1):
        cases = list(settings.cases)
        reverse = repetition % 2 == 0
        if phase == "candidate":
            reverse = not reverse
        if reverse:
            cases.reverse()
        for case in cases:
            rows.append(
                _measure_once(
                    resolved,
                    phase=phase,
                    case=case,
                    repetition=repetition,
                    settings=settings,
                )
            )
    summary = summarize(rows)
    summary.update(
        {
            "phase": phase,
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "pid": os.getpid(),
                "pyarrow": pa.__version__,
            },
        }
    )
    summary_path = resolved / phase / "summary.json"
    with atomic_path(str(summary_path)) as staged:
        Path(staged).write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = (
        (BenchmarkCase("quick-12-block", 3, 2),)
        if args.quick
        else tuple(BenchmarkCase(*row) for row in DEFAULT_CASES)
    )
    settings = BenchmarkSettings(
        workers=args.workers,
        repetitions=args.repetitions,
        heartbeat_seconds=0.0 if args.quick else 1.0,
        cases=cases,
    )
    print(
        json.dumps(
            run_phase(args.output, phase=args.phase, settings=settings),
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
