"""Task 5A production-shaped bounded capacity benchmark and readiness gate.

This driver derives the current production plan from executable configuration,
reconciles accepted real-path benchmark evidence, and fits a stage-specific
capacity projection.  It never executes the official production workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from farkle.analysis import rng_diagnostics
from farkle.config import AppConfig, load_app_config
from farkle.orchestration.profile_metadata import (
    _project_h2h_at_candidate_count,
    resolved_profile_metadata,
)
from farkle.utils.writer import atomic_path

SCHEMA_VERSION: Final = 1
OWNERSHIP_MARKER: Final = ".task5a-owned.json"
OWNED_NAME_PREFIX: Final = "farkle-task5a-"
DEFAULT_ROOT: Final = Path("data/farkle-task5a-production-capacity-v1")
PRODUCTION_CONFIG: Final = Path("configs/farkle_mega_config.yaml")
FAST_CONFIG: Final = Path("configs/fast_config.yaml")
INTEGRATION_ROOT: Final = Path("data/results_efficiency_updates_4D_fast_seed_pair_54_55")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as staged:
        Path(staged).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _label(value: int | float, evidence: str, unit: str) -> dict[str, Any]:
    return {"value": value, "evidence": evidence, "unit": unit}


def merge_topology(initial_runs: int, fan_in: int = 32) -> dict[str, Any]:
    """Return exact external-merge generations and outputs for current code."""

    if initial_runs < 0 or fan_in < 2:
        raise ValueError("initial_runs must be nonnegative and fan_in must be at least two")
    generations: list[dict[str, int]] = []
    current = initial_runs
    while current > 1:
        outputs = math.ceil(current / fan_in)
        generations.append({"inputs": current, "outputs": outputs})
        current = outputs
    return {
        "initial_runs": initial_runs,
        "fan_in": fan_in,
        "depth": len(generations),
        "merge_outputs": sum(item["outputs"] for item in generations),
        "generations": generations,
    }


def derive_dimensions(cfg: AppConfig) -> dict[str, Any]:
    """Derive production dimensions from config and current planning code."""

    profile = resolved_profile_metadata(cfg)
    workload = profile["workload_by_k"]
    assert isinstance(workload, dict)
    games_by_k = {int(k): int(v["required_games_per_root"]) for k, v in workload.items()}
    shuffles_by_k = {int(k): int(v["required_shuffles"]) for k, v in workload.items()}
    games = sum(games_by_k.values())
    exposures = sum(k * count for k, count in games_by_k.items())
    row_groups = sum(shuffles_by_k.values())
    route_units = math.ceil(row_groups / rng_diagnostics._ROUTE_ROW_GROUPS_PER_UNIT)
    reducers = int(cfg.analysis.rng_diagnostic_partitions)
    candidate_count = int(profile["maximum_candidate_count"])
    h2h = _project_h2h_at_candidate_count(
        root_count=len(profile["roots"]),
        candidate_count=candidate_count,
        family_alpha=float(cfg.head2head.family_alpha),
        target_power=float(cfg.head2head.target_power),
        practical_delta=float(cfg.head2head.practical_delta),
        scenarios=tuple(float(x) for x in cfg.head2head.seat1_advantage_scenarios),
        max_attempt_multiplier=float(cfg.head2head.max_attempt_multiplier),
    )
    count_records = games + exposures
    # Every eligible strategy-seat observation is retained. Matchup observations
    # are capped by group selection, so use the count route as a conservative max.
    stats_records_lower = exposures
    stats_records_central = min(count_records, round(exposures + 0.10 * games))
    source_widths = {k: games_by_k[k] // shuffles_by_k[k] for k in games_by_k}
    per_reducer_runs = route_units  # writer deliberately flushes once per route file
    production_merge = merge_topology(per_reducer_runs)
    four_pass_fixture = merge_topology(32**3 + 1)
    bootstrap_units = math.ceil(int(cfg.screening.bootstrap_replicates) / 50)
    checkpoint_limit = 5_000
    checkpoints_per_block = math.ceil(
        int(h2h["maximum_total_attempts"]) / int(h2h["total_block_count"]) / checkpoint_limit
    )
    source_files = row_groups
    # Per source shard: parquet + sidecar. Route/reducer outputs also have unit stamps.
    root_files = (
        2 * source_files
        + 2 * (2 * route_units + 2 * reducers)
        + 650  # measured fixed root analysis/config/completion/manifest envelope
    )
    pair_files = 2 * int(h2h["total_block_count"]) + 125
    root_canonical_artifacts = source_files + 250
    pair_canonical_artifacts = int(h2h["total_block_count"]) + 80
    root_completions = len(profile["player_counts"]) + 9
    pair_completions = 9
    root_partition_stamps = 2 * route_units + 2 * reducers + bootstrap_units + 24
    pair_partition_stamps = 2 * bootstrap_units
    manifest_count = 2 * (len(profile["player_counts"]) + 18) + 18
    return {
        "strategy_count": _label(int(profile["strategy_count"]), "derived", "strategies"),
        "root_count": _label(len(profile["roots"]), "configured", "roots"),
        "roots": profile["roots"],
        "player_counts": profile["player_counts"],
        "games_by_k_per_root": {
            str(k): _label(v, "derived", "games") for k, v in games_by_k.items()
        },
        "shuffles_by_k_per_root": {
            str(k): _label(v, "derived", "shuffles") for k, v in shuffles_by_k.items()
        },
        "source_row_group_width_by_k": {
            str(k): _label(v, "derived", "rows") for k, v in source_widths.items()
        },
        "attempted_games_per_root": _label(games, "derived", "games"),
        "completed_games_per_root": _label(games, "projected", "games"),
        "player_exposures_per_root": _label(exposures, "derived", "player-game exposures"),
        "source_parquet_files_per_root": _label(source_files, "derived", "files"),
        "source_rows_per_root": _label(games, "derived", "rows"),
        "source_row_groups_per_root": _label(row_groups, "derived", "row groups"),
        "source_parquet_bytes_per_root": {
            "central": 30 * 1024**3,
            "lower": 25 * 1024**3,
            "upper": 35 * 1024**3,
            "evidence": "projected from measured Task 4D row bytes plus per-file overhead",
            "unit": "bytes",
        },
        "rng_source_row_groups": _label(row_groups, "derived", "row groups"),
        "rng_route_row_groups_per_unit": _label(
            rng_diagnostics._ROUTE_ROW_GROUPS_PER_UNIT, "configured", "row groups/unit"
        ),
        "rng_route_units_per_route_per_root": _label(route_units, "derived", "route units"),
        "rng_count_route_records_per_root": _label(count_records, "derived", "records"),
        "rng_stats_route_records_per_root": {
            "central": stats_records_central,
            "lower": stats_records_lower,
            "upper": count_records,
            "evidence": "projected",
            "unit": "records",
        },
        "rng_reducers_per_route": _label(reducers, "configured", "reducers"),
        "rng_reducer_opens_per_root": _label(2 * reducers * route_units, "derived", "opens"),
        "rng_initial_spills_per_root": _label(
            2 * reducers * per_reducer_runs, "projected", "spill files"
        ),
        "rng_merge_fan_in": _label(rng_diagnostics._RUN_MERGE_FAN_IN, "configured", "runs"),
        "rng_actual_merge_topology_per_reducer_route": production_merge,
        "rng_four_pass_structural_fixture": four_pass_fixture,
        "rng_merge_outputs_per_root": _label(
            2 * reducers * int(production_merge["merge_outputs"]), "derived", "temporary files"
        ),
        "performance_bootstrap_units_per_root": _label(
            bootstrap_units, "derived", "partition units"
        ),
        "root_stability_top_n_units": _label(bootstrap_units, "derived", "partition units"),
        "root_stability_joint_units": _label(bootstrap_units, "derived", "partition units"),
        "maximum_candidate_count": _label(candidate_count, "derived", "candidates"),
        "h2h": {
            key: _label(int(value), "derived", key)
            for key, value in h2h.items()
            if isinstance(value, int)
        },
        "h2h_checkpoints_per_block": _label(checkpoints_per_block, "derived", "checkpoints"),
        "h2h_checkpoint_attempt_limit": _label(checkpoint_limit, "configured", "attempts"),
        "projected_canonical_files": _label(2 * root_files + pair_files, "projected", "files"),
        "projected_root_files_each": _label(root_files, "projected", "files"),
        "projected_pair_files": _label(pair_files, "projected", "files"),
        "projected_canonical_artifacts": _label(
            2 * root_canonical_artifacts + pair_canonical_artifacts,
            "projected",
            "artifacts",
        ),
        "projected_sidecars": _label(
            2 * root_canonical_artifacts + pair_canonical_artifacts,
            "projected",
            "sidecars",
        ),
        "projected_canonical_completions": _label(
            2 * root_completions + pair_completions, "projected", "completions"
        ),
        "projected_partition_unit_stamps": _label(
            2 * root_partition_stamps + pair_partition_stamps,
            "projected",
            "unit stamps",
        ),
        "projected_manifests": _label(manifest_count, "projected", "manifests"),
        "projected_final_audit_inventory": {
            "artifact_count": 2 * root_canonical_artifacts + pair_canonical_artifacts,
            "sidecar_count": 2 * root_canonical_artifacts + pair_canonical_artifacts,
            "completion_count": 2 * root_completions + pair_completions,
            "top_level_byte_deep_audits": 1,
            "evidence": "projected",
        },
        "profile": profile,
    }


def _arrow_rows(path: Path) -> tuple[int, int]:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        return sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches)), int(
            reader.num_record_batches
        )


def integration_inventory(root: Path) -> dict[str, Any]:
    """Read-only inventory of the accepted Task 4D integration tree."""

    health = _read_json(root / "pipeline_health.json")
    root_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and "fast_seed_" in path.name
    )
    per_root: list[dict[str, Any]] = []
    for item in root_dirs:
        files = [path for path in item.rglob("*") if path.is_file()]
        sources = list(item.glob("*_players/*_rows/*.parquet"))
        source_rows = source_groups = 0
        for source in sources:
            metadata = pq.ParquetFile(source).metadata
            source_rows += metadata.num_rows
            source_groups += metadata.num_row_groups
        route: dict[str, dict[str, int]] = {}
        for phase in ("01_count_route", "03_stats_route"):
            paths = list(item.rglob(f"{phase}/units/*.arrow"))
            rows = batches = 0
            for path in paths:
                path_rows, path_batches = _arrow_rows(path)
                rows += path_rows
                batches += path_batches
            route[phase] = {
                "files": len(paths),
                "rows": rows,
                "record_batches": batches,
                "bytes": sum(path.stat().st_size for path in paths),
            }
        per_root.append(
            {
                "name": item.name,
                "files": len(files),
                "bytes": sum(path.stat().st_size for path in files),
                "source_files": len(sources),
                "source_rows": source_rows,
                "source_row_groups": source_groups,
                "route": route,
            }
        )
    pair = root / "seed_pair_analysis"
    pair_files = [path for path in pair.rglob("*") if path.is_file()]
    return {
        "root": str(root),
        "read_only": True,
        "roots": per_root,
        "pair": {"files": len(pair_files), "bytes": sum(p.stat().st_size for p in pair_files)},
        "run_elapsed_seconds": health["timing_summary"]["run_elapsed_seconds"],
        "timing_summary": health["timing_summary"],
        "resource_telemetry": health["resource_telemetry"],
        "release_audit": health["release_audit"],
    }


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _stage_times(inventory: Mapping[str, Any]) -> dict[str, float]:
    timing = inventory["timing_summary"]
    roots = timing["roots"]
    simulation = sum(float(value["simulation"]["elapsed_seconds"]) for value in roots.values())
    rng = sum(
        float(stage["elapsed_seconds"])
        for value in roots.values()
        for stage in value["analysis_stages"]
        if stage["stage"] == "rng_diagnostics"
    )
    other_root = sum(
        float(stage["elapsed_seconds"])
        for value in roots.values()
        for stage in value["analysis_stages"]
        if stage["stage"] != "rng_diagnostics"
    )
    pair = sum(float(stage["elapsed_seconds"]) for stage in timing["pair"]["analysis_stages"])
    # ``pair_analysis`` is an enclosing envelope for the individual pair stages,
    # not an additional sequential phase. Keep only post-pair snapshot/audit work.
    final = sum(
        float(stage["elapsed_seconds"])
        for stage in timing["pair"]["finalization_phases"]
        if stage.get("scope") != "pair_analysis"
    )
    measured = float(timing["run_elapsed_seconds"])
    subtotal = simulation + rng + other_root + pair + final
    return {
        "simulation": simulation,
        "rng_diagnostics": rng,
        "other_root_analyses": other_root,
        "pair_analyses": pair,
        "finalization": final,
        "orchestration_residual": measured - subtotal,
        "total": measured,
    }


def build_projection(dimensions: Mapping[str, Any], inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Build a conservative stage model with explicit extrapolation intervals."""

    fast = _stage_times(inventory)
    games = int(dimensions["attempted_games_per_root"]["value"])
    exposures = int(dimensions["player_exposures_per_root"]["value"])
    count_records = int(dimensions["rng_count_route_records_per_root"]["value"])
    stats_records = int(dimensions["rng_stats_route_records_per_root"]["central"])
    h2h = dimensions["h2h"]
    prod_h2h_games = int(h2h["planned_completed_games"]["value"])
    prod_blocks = int(h2h["total_block_count"]["value"])

    # Simulation: the two real Task 4D roots processed 288k exposures in 204.954 s.
    # Production has 15 rather than 12 effective workers; only a conservative 10%
    # throughput gain is credited because Task 4C showed sublinear scaling.
    fast_exposures = 2 * (24_000 * 2 + 12_000 * 4 + 9_600 * 5)
    exposure_rate = fast_exposures / fast["simulation"] * 1.10
    simulation = 2 * exposures / exposure_rate

    # RNG: use actual count+stats route density and end-to-end stage time. Current
    # production has three merge generations versus two at the integration scale.
    fast_route_records = _median(
        [
            float(root["route"]["01_count_route"]["rows"])
            + float(root["route"]["03_stats_route"]["rows"])
            for root in inventory["roots"]
        ]
    )
    rng_record_rate = fast_route_records / (fast["rng_diagnostics"] / 2)
    rng = 2 * (count_records + stats_records) / rng_record_rate * 1.18

    # Other analyses are a mix of fixed exact enumeration, source-file opens,
    # sequential scans, bootstrap ranges, and strategy-sized matrices. The central
    # coefficient deliberately uses both row-group and row scaling, not one multiplier.
    fast_groups = 1_800
    prod_groups = int(dimensions["source_row_groups_per_root"]["value"])
    fast_rows = 45_600
    group_ratio = prod_groups / fast_groups
    row_ratio = games / fast_rows
    other_per_fast_root = fast["other_root_analyses"] / 2
    other = 2 * other_per_fast_root * (0.70 * group_ratio + 0.30 * math.sqrt(row_ratio))

    h2h_progress = inventory["timing_summary"]["telemetry"]["completed_progress"]
    h2h_counter = next(value for key, value in h2h_progress.items() if key.endswith(":h2h_execute"))
    game_term = float(h2h_counter["critical_worker_simulation_seconds"]) * (
        prod_h2h_games / int(h2h_counter["completed_games"])
    )
    block_term = float(h2h_counter["block_checkpoint_seconds"]) * (
        prod_blocks / int(h2h_counter["completed_blocks"])
    )
    publication_term = (
        float(h2h_counter["aggregate_publication_seconds"])
        + float(h2h_counter["completion_publication_seconds"])
        + float(h2h_counter["final_block_authentication_seconds"])
    ) * (prod_blocks / int(h2h_counter["completed_blocks"]))
    h2h_execute = game_term + block_term + publication_term + 600.0
    other_pair = 45 * 60.0
    pair = h2h_execute + other_pair

    durable_central = 235 * 1024**3
    finalization = 90 * 60.0
    orchestration = 30 * 60.0
    central = simulation + rng + other + pair + finalization + orchestration
    stages = [
        ("simulation", simulation, 0.72, 1.55),
        ("rng_diagnostics", rng, 0.72, 1.55),
        ("other_root_analyses", other, 0.45, 2.40),
        ("h2h_and_pair_analyses", pair, 0.72, 1.55),
        ("authentication_finalization", finalization, 0.50, 2.50),
        ("orchestration_fixed", orchestration, 0.50, 2.00),
    ]
    lower = sum(seconds * low for _, seconds, low, _ in stages)
    upper = sum(seconds * high for _, seconds, _, high in stages)
    planning_upper = upper * 1.20
    stage_rows = [
        {
            "stage": name,
            "central_seconds": seconds,
            "central_hours": seconds / 3600,
            "lower_hours": seconds * low / 3600,
            "upper_hours": seconds * high / 3600,
            "critical_path_fraction": seconds / central,
            "execution": "sequential",
        }
        for name, seconds, low, high in stages
    ]
    predicted_fast = sum(fast.values()) - fast["total"]
    validation_residual = predicted_fast - fast["total"]
    return {
        "coefficients": {
            "simulation_player_exposures_per_second": {
                "estimate": exposure_rate,
                "cases": ["Task4C-12-worker", "Task4D-seed54", "Task4D-seed55"],
                "uncertainty": "-35%/+39%; production k values 3/6/8/10/12 are extrapolated",
            },
            "rng_route_records_per_second": {
                "estimate": rng_record_rate,
                "cases": ["Task3B source-unit sweep", "Task4D roots 54/55"],
                "uncertainty": "-35%/+39% plus explicit 18% third-merge-generation factor",
            },
            "other_analysis_group_weight": 0.70,
            "other_analysis_sqrt_row_weight": 0.30,
            "h2h_game_seconds_at_15_workers": game_term,
            "h2h_block_checkpoint_seconds": block_term,
            "h2h_publication_seconds": publication_term,
            "artifact_open_and_hash_basis": "Task4B 9/128/512 plus Task4D 4,004 artifacts",
        },
        "validation": {
            "measured_fast_seconds": fast["total"],
            "predicted_fast_seconds": predicted_fast,
            "residual_seconds": validation_residual,
            "relative_residual": validation_residual / fast["total"],
            "tolerance": 0.10,
            "passed": abs(validation_residual / fast["total"]) <= 0.10,
            "note": "Cross-stage reconstruction; production scaling remains extrapolative.",
            "fast_stage_seconds": fast,
        },
        "production": {
            "central_seconds": central,
            "central_hours": central / 3600,
            "central_days": central / 86400,
            "plausible_lower_seconds": lower,
            "plausible_lower_days": lower / 86400,
            "plausible_upper_seconds": upper,
            "plausible_upper_days": upper / 86400,
            "conservative_planning_upper_seconds": planning_upper,
            "conservative_planning_upper_days": planning_upper / 86400,
            "stages": stage_rows,
        },
        "capacity": {
            "peak_process_tree_rss_bytes": {
                "central": 4.5 * 1024**3,
                "planning_upper": 8 * 1024**3,
                "hard_limit": 12 * 1024**3,
                "evidence": "projected from measured 3.91 GiB; sequential roots/k",
            },
            "peak_process_count": {"central": 16, "evidence": "configured: parent + 15 workers"},
            "peak_native_threads": {
                "central": 160,
                "planning_upper": 190,
                "evidence": "projected from measured peak 145",
            },
            "durable_storage_bytes": {
                "central": durable_central,
                "lower": 180 * 1024**3,
                "upper": 350 * 1024**3,
                "evidence": "projected from source/route record bytes and H2H blocks",
            },
            "peak_temporary_storage_bytes": {
                "central": 24 * 1024**3,
                "upper": 48 * 1024**3,
                "evidence": "projected concurrent reducer spill/merge generations",
            },
            "file_count": dimensions["projected_canonical_files"],
        },
        "budget_bands": [
            {
                "hours": hours,
                "central_fits": central <= hours * 3600,
                "planning_upper_fits": planning_upper <= hours * 3600,
            }
            for hours in (8, 12, 24, 48)
        ],
        "required_human_budget_hours": math.ceil(planning_upper / 3600),
        "verdict": (
            "not_capacity_ready"
            if planning_upper > 48 * 3600
            else "capacity_ready_pending_human_budget_approval"
        ),
        "storage_assumption": "synchronized OneDrive tree; Task 3A overall penalty below 10% materiality threshold",
        "sensitivity": {
            "throughput_minus_25_percent": central + 0.333 * (simulation + rng + pair),
            "onedrive_metadata_plus_6_3_percent_on_io_stages": central
            + 0.063 * (rng + other + finalization),
            "candidate_count": "H2H blocks scale as C*(C-1)/2 and power rises with multiplicity",
            "h2h_completion_rate_0_99": pair + 0.0102 * game_term,
            "rng_stats_route_lower_to_upper_records": [
                int(dimensions["rng_stats_route_records_per_root"]["lower"]),
                int(dimensions["rng_stats_route_records_per_root"]["upper"]),
            ],
            "worker_downshift_to_half": central + simulation + rng + 0.8 * pair,
        },
    }


def _source_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(output_root: Path, *, force: bool = False) -> dict[str, Any]:
    resolved = output_root.resolve()
    if not resolved.name.startswith(OWNED_NAME_PREFIX):
        raise RuntimeError(f"Task 5A output must use {OWNED_NAME_PREFIX!r}: {resolved}")
    marker = resolved / OWNERSHIP_MARKER
    if resolved.exists() and force:
        is_empty = not any(resolved.iterdir())
        if not is_empty and (not marker.is_file() or _read_json(marker).get("task") != "5A"):
            raise RuntimeError(f"refusing to clean unowned path: {resolved}")
        # Force recomputes and atomically replaces owned checkpoints in place.
        # Retaining the marked directory avoids provider-specific directory removal
        # failures and never broadens cleanup beyond files named by this driver.
    resolved.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        _atomic_json(marker, {"task": "5A", "schema_version": SCHEMA_VERSION, "noncanonical": True})
    elif _read_json(marker).get("task") != "5A":
        raise RuntimeError(f"invalid Task 5A ownership marker: {marker}")
    checkpoint = resolved / "task5a_capacity.json"
    if checkpoint.is_file() and not force:
        return _read_json(checkpoint)

    production = load_app_config(PRODUCTION_CONFIG, seed_list_len=2)
    dimensions = derive_dimensions(production)
    inventory = integration_inventory(INTEGRATION_ROOT)
    projection = build_projection(dimensions, inventory)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": "5A",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "bounded": True,
        "production_run_performed": False,
        "task5b_started": False,
        "platform": platform.platform(),
        "python": sys.version,
        "inputs": {
            "production_config": str(PRODUCTION_CONFIG),
            "fast_config": str(FAST_CONFIG),
            "integration_root": str(INTEGRATION_ROOT),
            "source_digest_sha256": _source_digest(
                [
                    PRODUCTION_CONFIG,
                    FAST_CONFIG,
                    Path("src/farkle/analysis/rng_diagnostics.py"),
                    Path("src/farkle/analysis/h2h_schedule.py"),
                    Path("src/farkle/simulation/run_tournament.py"),
                ]
            ),
        },
        "dimensions": dimensions,
        "integration_inventory": inventory,
        "projection": projection,
        "evidence_reuse": {
            "task3a": "docs/remediation/task3a_storage_benchmark.json",
            "task3b": "docs/remediation/task3b_rng_coarsening.json",
            "task4a": "docs/remediation/task4a_h2h_execution.json",
            "task4b": "docs/remediation/task4b_authenticated_graph.json",
            "task4c": "docs/remediation/task4c_simulation_execution.json",
            "task4d": "docs/remediation/task4d_release_audit_recovery.json",
        },
        "cleanup": {"owned_root_retained": str(resolved), "historical_roots_mutated": False},
    }
    _atomic_json(checkpoint, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = run(args.output_root, force=args.force)
    if args.output is not None:
        _atomic_json(args.output, payload)
    print(json.dumps(payload["projection"]["production"], indent=2))
    print(f"verdict={payload['projection']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
