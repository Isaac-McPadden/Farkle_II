"""Bounded Task 4C protected-CLI simulation execution benchmark.

The driver compares intentional serial execution with Windows-spawn process
pools, records live topology and process-tree resources, and exercises a normal
console interruption followed by exact resume.  Every output is confined to an
explicit benchmark-owned root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import psutil
import yaml  # type: ignore[import-untyped]

from farkle.config import AppConfig, assign_config_sha, effective_config_dict, load_app_config
from farkle.utils.manifest import iter_manifest
from farkle.utils.writer import atomic_path

BENCHMARK_SCHEMA_VERSION = 1
OWNED_PREFIX = "farkle-task4c-"
OWNERSHIP_MARKER = ".task4c-owned.json"


@dataclass(frozen=True, slots=True)
class CaseResult:
    name: str
    workers: int
    wall_seconds: float
    simulation_wall_seconds: float
    process_tree_cpu_seconds: float
    throughput_games_per_second: float
    simulation_throughput_games_per_second: float
    requested_workers: int
    resolved_workers: int
    created_workers: int
    peak_live_workers: int
    cleanly_terminated_workers: int
    full_pool_observed_seconds: float | None
    worker_drain_to_quiescence_seconds: float | None
    worker_pids: tuple[int, ...]
    peak_process_tree_rss_bytes: int
    peak_native_threads: int
    checkpoint_count: int
    checkpoint_interval_seconds: float
    canonical_bundle_sha256: str
    logical_checkpoint_sha256: str
    returncode: int
    console_path: str
    heartbeat_examples: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(self).items()
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_owned_root(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.name.startswith(OWNED_PREFIX):
        raise ValueError(f"Task 4C output root must start with {OWNED_PREFIX!r}: {resolved}")
    if resolved == Path.cwd().resolve() or Path.cwd().resolve() not in resolved.parents:
        raise ValueError("Task 4C output root must be a repository-contained child")
    return resolved


def _prepare_owned_root(path: Path, *, force: bool) -> Path:
    root = _safe_owned_root(path)
    marker = root / OWNERSHIP_MARKER
    if root.exists() and force:
        if not marker.exists():
            raise ValueError(f"refusing to remove unmarked benchmark root {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        with atomic_path(str(marker)) as temporary:
            Path(temporary).write_text(
                json.dumps({"benchmark": "task4c", "schema_version": BENCHMARK_SCHEMA_VERSION}),
                encoding="utf-8",
            )
    return root


def _fixture_config(
    *,
    prefix: Path,
    workers: int,
    row_output: bool,
    production_grid: bool = False,
) -> AppConfig:
    cfg = load_app_config(Path("configs/fast_config.yaml"))
    cfg.io.results_dir_prefix = prefix
    cfg.sim.seed = 94_512
    cfg.sim.seed_list = [94_512]
    cfg.sim.n_players_list = [2]
    cfg.sim.n_jobs = int(workers)
    cfg.sim.mp_start_method = "spawn"
    cfg.sim.expanded_metrics = bool(row_output)
    cfg.sim.row_dir = Path("rows") if row_output else None
    cfg.sim.metric_chunk_dir = None
    cfg.sim.ckpt_every_sec = 5
    if not production_grid:
        cfg.sim.score_thresholds = [250]
        cfg.sim.dice_thresholds = [0, 1]
    cfg.sim.smart_five_opts = [True]
    cfg.sim.smart_one_opts = [True]
    cfg.sim.consider_score_opts = [True]
    cfg.sim.consider_dice_opts = [True]
    cfg.sim.auto_hot_dice_opts = [True]
    cfg.sim.run_up_score_opts = [True]
    cfg.screening.resolution_delta = 0.20
    cfg.screening.practical_delta_by_k = {2: 0.03}
    cfg.screening.bootstrap_replicates = 20
    cfg.screening.candidate_contribution_size = 2
    cfg.screening.max_shuffles_per_root_k = 600
    cfg.resources.logical_cpu_budget = min(15, max(12, int(workers)))
    cfg.resources.scheduler_memory_budget_mb = 4096
    cfg.resources.process_tree_warning_threshold_mb = 5120
    cfg.resources.aggregate_memory_hard_limit_mb = 6144
    cfg.resources.minimum_system_available_memory_mb = 1024
    cfg.resources.parent_process_memory_mb = 256
    cfg.resources.estimated_worker_memory_mb["simulation"] = 128
    cfg.orchestration.parallel_seeds = False
    cfg.validate_statistical_contract(require_two_roots=False)
    assign_config_sha(cfg)
    return cfg


def _write_config(path: Path, cfg: AppConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as temporary:
        Path(temporary).write_text(
            yaml.safe_dump(effective_config_dict(cfg), sort_keys=True),
            encoding="utf-8",
        )


def _checkpoint_payload(results_root: Path) -> dict[str, Any] | None:
    checkpoint = results_root / "2_players" / "2p_checkpoint.pkl"
    if not checkpoint.exists():
        return None
    try:
        value = pickle.loads(checkpoint.read_bytes())
    except (OSError, EOFError, pickle.UnpicklingError):
        return None
    return value if isinstance(value, dict) else None


def _logical_checkpoint_sha256(results_root: Path) -> str:
    payload = _checkpoint_payload(results_root)
    if payload is None:
        raise FileNotFoundError(f"completed checkpoint is unavailable under {results_root}")
    logical = {
        "win_totals": sorted(
            (str(key), int(value)) for key, value in payload["win_totals"].items()
        ),
        "outcome_counts": payload.get("outcome_counts"),
        "completed_shuffle_indices": payload.get("meta", {}).get("completed_shuffle_indices", []),
        "completed_process_block_indices": payload.get("meta", {}).get(
            "completed_process_block_indices", []
        ),
    }
    return hashlib.sha256(
        json.dumps(logical, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _canonical_bundle_sha256(results_root: Path) -> str:
    entries: list[tuple[str, str]] = []
    for path in sorted((results_root / "2_players").rglob("*")):
        if (
            not path.is_file()
            or path.name.startswith("._")
            or path.name.endswith(".sidecar.json")
            or path.name in {"simulation.done.json", "manifest.jsonl"}
            or path.suffix not in {".parquet", ".json"}
        ):
            continue
        entries.append((path.relative_to(results_root).as_posix(), _sha256_file(path)))
    return hashlib.sha256(
        json.dumps(entries, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _durable_row_identities(row_dir: Path) -> dict[str, tuple[int, str]]:
    """Return identities only for row shards committed to the durable manifest."""

    manifest_path = row_dir / "manifest.jsonl"
    identities: dict[str, tuple[int, str]] = {}
    for record in iter_manifest(manifest_path):
        raw_path = record.get("path")
        if not isinstance(raw_path, str):
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = row_dir / path
        if path.is_file():
            identities[path.name] = (path.stat().st_mtime_ns, _sha256_file(path))
    return identities


def _simulation_wall_seconds(console_text: str) -> float:
    """Extract the bounded tournament interval from rendered operational logs."""

    started: datetime | None = None
    completed: datetime | None = None
    for line in console_text.splitlines():
        try:
            timestamp = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if "Tournament run start" in line and started is None:
            started = timestamp
        elif "Tournament run complete" in line:
            completed = timestamp
    if started is None or completed is None or completed <= started:
        raise RuntimeError("benchmark console did not contain a valid tournament interval")
    return (completed - started).total_seconds()


def _command(config_path: Path, *, force: bool) -> list[str]:
    command = [sys.executable, "-m", "farkle", "--config", str(config_path), "run"]
    if force:
        command.append("--force")
    return command


def _run_case(
    *,
    name: str,
    cfg: AppConfig,
    config_path: Path,
    console_path: Path,
    force: bool,
    interrupt_after_rows: int | None = None,
) -> tuple[CaseResult | None, dict[str, object]]:
    _write_config(config_path, cfg)
    results_root = cfg.results_root.resolve()
    command = _command(config_path, force=force)
    started = time.monotonic()
    cpu_by_pid: dict[int, float] = {}
    worker_pids: set[int] = set()
    peak_live_workers = 0
    peak_rss = 0
    peak_threads = 0
    interruption_triggered_at: float | None = None
    full_pool_observed_at: float | None = None
    worker_shutdown_started_at: float | None = None
    worker_quiesced_at: float | None = None
    rows_at_interrupt = 0
    console_path.parent.mkdir(parents=True, exist_ok=True)
    with console_path.open("w", encoding="utf-8") as console:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=console,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        while process.poll() is None:
            try:
                descendants = psutil.Process(process.pid).children(recursive=True)
            except psutil.Error:
                descendants = []
            live_workers = 0
            rss = 0
            threads = 0
            for descendant in descendants:
                try:
                    cpu = descendant.cpu_times()
                    cpu_by_pid[descendant.pid] = max(
                        cpu_by_pid.get(descendant.pid, 0.0), float(cpu.user + cpu.system)
                    )
                    rss += int(descendant.memory_info().rss)
                    threads += int(descendant.num_threads())
                    if "spawn_main" in " ".join(descendant.cmdline()):
                        worker_pids.add(descendant.pid)
                        live_workers += 1
                except psutil.Error:
                    continue
            peak_live_workers = max(peak_live_workers, live_workers)
            if live_workers >= int(cfg.sim.n_jobs or 1) and full_pool_observed_at is None:
                full_pool_observed_at = time.monotonic()
            if (
                full_pool_observed_at is not None
                and live_workers < int(cfg.sim.n_jobs or 1)
                and worker_shutdown_started_at is None
            ):
                worker_shutdown_started_at = time.monotonic()
            if worker_shutdown_started_at is not None and live_workers == 0:
                worker_quiesced_at = time.monotonic()
            peak_rss = max(peak_rss, rss)
            peak_threads = max(peak_threads, threads)
            if interrupt_after_rows is not None and interruption_triggered_at is None:
                row_dir = results_root / "2_players" / "2p_rows"
                rows_at_interrupt = (
                    len(tuple(row_dir.glob("rows_*.parquet"))) if row_dir.exists() else 0
                )
                if rows_at_interrupt >= interrupt_after_rows:
                    interruption_triggered_at = time.monotonic()
                    if os.name == "nt":
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:  # pragma: no cover - Windows acceptance path
                        process.send_signal(signal.SIGINT)
            time.sleep(0.1)
        returncode = int(process.wait())
    wall = time.monotonic() - started
    if worker_shutdown_started_at is not None and worker_quiesced_at is None:
        worker_quiesced_at = time.monotonic()
    quiescence_started = interruption_triggered_at
    while quiescence_started is not None and time.monotonic() - quiescence_started < 15.0:
        if not any(psutil.pid_exists(pid) for pid in worker_pids):
            break
        time.sleep(0.1)
    shutdown_latency = (
        time.monotonic() - interruption_triggered_at
        if interruption_triggered_at is not None
        else None
    )
    interruption = {
        "triggered": interruption_triggered_at is not None,
        "rows_at_interrupt": rows_at_interrupt,
        "shutdown_latency_seconds": shutdown_latency,
        "orphan_worker_pids": sorted(pid for pid in worker_pids if psutil.pid_exists(pid)),
        "returncode": returncode,
    }
    if interruption_triggered_at is not None:
        return None, interruption
    if returncode != 0:
        raise RuntimeError(f"benchmark case {name!r} failed with exit code {returncode}")
    console_text = console_path.read_text(encoding="utf-8")
    simulation_wall = _simulation_wall_seconds(console_text)
    heartbeat_examples = tuple(line for line in console_text.splitlines() if "Heartbeat:" in line)[
        -3:
    ]
    payload = _checkpoint_payload(results_root)
    if payload is None:
        raise RuntimeError(f"benchmark case {name!r} did not publish a checkpoint")
    games = int(payload.get("outcome_counts", {}).get("games_attempted", 0))
    checkpoint = results_root / "2_players" / "2p_checkpoint.pkl"
    result = CaseResult(
        name=name,
        workers=int(cfg.sim.n_jobs or 1),
        wall_seconds=wall,
        simulation_wall_seconds=simulation_wall,
        process_tree_cpu_seconds=sum(cpu_by_pid.values()),
        throughput_games_per_second=games / wall if wall > 0 else 0.0,
        simulation_throughput_games_per_second=games / simulation_wall,
        requested_workers=int(cfg.sim.n_jobs or 1),
        resolved_workers=int(cfg.sim.n_jobs or 1),
        created_workers=len(worker_pids),
        peak_live_workers=peak_live_workers,
        cleanly_terminated_workers=(
            len(worker_pids) if not any(psutil.pid_exists(pid) for pid in worker_pids) else 0
        ),
        full_pool_observed_seconds=(
            None if full_pool_observed_at is None else full_pool_observed_at - started
        ),
        worker_drain_to_quiescence_seconds=(
            None
            if worker_shutdown_started_at is None or worker_quiesced_at is None
            else worker_quiesced_at - worker_shutdown_started_at
        ),
        worker_pids=tuple(sorted(worker_pids)),
        peak_process_tree_rss_bytes=peak_rss,
        peak_native_threads=peak_threads,
        checkpoint_count=int(checkpoint.exists()),
        checkpoint_interval_seconds=float(cfg.sim.ckpt_every_sec),
        canonical_bundle_sha256=_canonical_bundle_sha256(results_root),
        logical_checkpoint_sha256=_logical_checkpoint_sha256(results_root),
        returncode=returncode,
        console_path=str(console_path),
        heartbeat_examples=heartbeat_examples,
    )
    return result, interruption


def run_benchmark(output_root: Path, *, force: bool) -> dict[str, object]:
    root = _prepare_owned_root(output_root, force=force)
    cases: list[CaseResult] = []
    for workers in (1, 2, 12):
        prefix = root / f"cpu_n_jobs_{workers}"
        cfg = _fixture_config(
            prefix=prefix,
            workers=workers,
            row_output=False,
            production_grid=True,
        )
        result, _ = _run_case(
            name=f"cpu_n_jobs_{workers}",
            cfg=cfg,
            config_path=root / "configs" / f"cpu_n_jobs_{workers}.yaml",
            console_path=root / "logs" / f"cpu_n_jobs_{workers}.txt",
            force=True,
        )
        assert result is not None
        cases.append(result)

    reference_cfg = _fixture_config(
        prefix=root / "resume_reference_n_jobs_12",
        workers=12,
        row_output=True,
    )
    reference, _ = _run_case(
        name="resume_reference_n_jobs_12",
        cfg=reference_cfg,
        config_path=root / "configs" / "resume_reference_n_jobs_12.yaml",
        console_path=root / "logs" / "resume_reference_n_jobs_12.txt",
        force=True,
    )
    assert reference is not None

    resume_cfg = _fixture_config(
        prefix=root / "interrupted_resume_n_jobs_12",
        workers=12,
        row_output=True,
    )
    _, interruption = _run_case(
        name="interrupted_resume_n_jobs_12",
        cfg=resume_cfg,
        config_path=root / "configs" / "interrupted_resume_n_jobs_12.yaml",
        console_path=root / "logs" / "interrupted_resume_interrupt.txt",
        force=True,
        interrupt_after_rows=72,
    )
    row_dir = resume_cfg.results_root.resolve() / "2_players" / "2p_rows"
    visible_rows_before_resume = len(tuple(row_dir.glob("rows_*.parquet")))
    preserved = _durable_row_identities(row_dir)
    resumed, _ = _run_case(
        name="interrupted_resume_n_jobs_12",
        cfg=resume_cfg,
        config_path=root / "configs" / "interrupted_resume_n_jobs_12.yaml",
        console_path=root / "logs" / "interrupted_resume_complete.txt",
        force=False,
    )
    assert resumed is not None
    reused_rows = sum(
        path.exists() and (path.stat().st_mtime_ns, _sha256_file(path)) == identity
        for name, identity in preserved.items()
        for path in [row_dir / name]
    )
    interruption.update(
        {
            "durable_rows_before_resume": len(preserved),
            "durable_rows_reused": reused_rows,
            "incomplete_rows_before_resume": visible_rows_before_resume - len(preserved),
            "maximum_recomputed_games": 12 * 4,
            "canonical_equivalent_to_reference": (
                resumed.canonical_bundle_sha256 == reference.canonical_bundle_sha256
            ),
            "logical_equivalent_to_reference": (
                resumed.logical_checkpoint_sha256 == reference.logical_checkpoint_sha256
            ),
        }
    )
    serial = cases[0]
    twelve = cases[-1]
    shutdown_latency = interruption.get("shutdown_latency_seconds")
    summary: dict[str, object] = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "platform": sys.platform,
        "python": sys.version,
        "multiprocessing_start_method": "spawn",
        "protected_launcher": True,
        "cpu_fixture": {
            "root_seed": 94_512,
            "player_count": 2,
            "strategies": 80,
            "shuffles": 600,
            "games_per_shuffle": 40,
            "games": 24_000,
            "pending_process_blocks": 20,
        },
        "interruption_fixture": {
            "root_seed": 94_512,
            "player_count": 2,
            "strategies": 8,
            "shuffles": 600,
            "games_per_shuffle": 4,
            "games": 2_400,
            "pending_process_blocks": 20,
        },
        "cases": [case.as_dict() for case in cases],
        "resume_reference": reference.as_dict(),
        "interrupted_resume": resumed.as_dict(),
        "interruption": interruption,
        "speedup_12_vs_1": serial.wall_seconds / twelve.wall_seconds,
        "simulation_speedup_12_vs_1": (
            serial.simulation_wall_seconds / twelve.simulation_wall_seconds
        ),
        "throughput_ratio_12_vs_1": (
            twelve.throughput_games_per_second / serial.throughput_games_per_second
        ),
        "acceptance": {
            "twelve_workers_created": twelve.created_workers == 12,
            "twelve_workers_peak_live": twelve.peak_live_workers == 12,
            "all_workers_cleanly_terminated": all(
                case.cleanly_terminated_workers == case.created_workers for case in cases
            ),
            "multiprocessing_materially_faster": serial.wall_seconds > twelve.wall_seconds,
            "speedup_at_least_4x": serial.wall_seconds / twelve.wall_seconds >= 4.0,
            "simulation_speedup_at_least_4x": (
                serial.simulation_wall_seconds / twelve.simulation_wall_seconds >= 4.0
            ),
            "exact_logical_worker_equivalence": len(
                {case.logical_checkpoint_sha256 for case in cases}
            )
            == 1,
            "byte_identical_canonical_worker_outputs": len(
                {case.canonical_bundle_sha256 for case in cases}
            )
            == 1,
            "resume_exact_logical_equivalence": bool(
                interruption["logical_equivalent_to_reference"]
            ),
            "resume_byte_identical_canonical_outputs": bool(
                interruption["canonical_equivalent_to_reference"]
            ),
            "durable_rows_reused": reused_rows == len(preserved),
            "shutdown_within_15_seconds": bool(
                isinstance(shutdown_latency, (int, float))
                and shutdown_latency <= 15.0
                and not interruption["orphan_worker_pids"]
            ),
        },
    }
    summary_path = root / "task4c_simulation_execution.json"
    with atomic_path(str(summary_path)) as temporary:
        Path(temporary).write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/farkle-task4c-simulation-execution-v1"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_benchmark(args.output_root, force=bool(args.force))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
