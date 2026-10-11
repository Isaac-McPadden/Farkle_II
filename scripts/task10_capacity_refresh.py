"""Refresh conditional first-release runtime/storage scenarios without simulation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.benchmark_task5a_production_capacity import derive_dimensions

from farkle.config import load_app_config
from farkle.utils.artifact_contract import sha256_file
from farkle.utils.writer import atomic_path


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected measurement object: {path}")
    return value


def interpolate(k: int, low: float, high: float) -> float:
    return low + (high - low) * (k - 2) / 10


def scenario(
    dimensions: dict[str, Any],
    endpoints: tuple[dict[str, Any], dict[str, Any]],
    precision: dict[str, Any],
    *,
    bins: int,
    source_fraction: float,
    merge_factor: float,
) -> dict[str, float]:
    """Use disclosed source, dense-grid, tile and activated-resampling drivers."""
    strategies = int(dimensions["strategy_count"]["value"])
    roots = int(dimensions["root_count"]["value"])
    batches = 100
    outcome_seconds = behavior_seconds = 0.0
    outcome_bytes = behavior_bytes = 0.0
    total_cells = 0
    for k in dimensions["player_counts"]:
        exposures = k * int(dimensions["games_by_k_per_root"][str(k)]["value"])
        cells = strategies * (k + 1)
        total_cells += cells
        per_endpoint = []
        for fixture in endpoints:
            fixture_cells = fixture["strategies"] * (fixture["k"] + 1)
            source_ratio = exposures / fixture["exposures"]
            seat_grid_ratio = (
                strategies
                * k
                * batches
                / (fixture["strategies"] * fixture["k"] * fixture["batches"])
            )
            mixed_ratio = source_fraction * source_ratio + (1 - source_fraction) * seat_grid_ratio
            op = fixture["outcome_phase_seconds"]
            scan = op["source"] + op["normalized"]
            merges = sum(v for name, v in op.items() if "merge" in name)
            output_rows = cells * (44 + bins + 1)
            sample_outputs = fixture["measurements"]["outcomes"]["outputs"]
            sample_rows = sum(
                sample_outputs[n]["rows"]
                for n in (
                    "outcome_frequencies.parquet",
                    "outcome_summaries.parquet",
                    "outcome_support.parquet",
                )
            )
            residual = max(0.0, fixture["measurements"]["outcomes"]["seconds"] - sum(op.values()))
            outcome = (
                scan * source_ratio
                + merges * source_ratio * merge_factor
                + op["certified_batches"] * mixed_ratio
                + op["publish_batches"] * seat_grid_ratio
                + op["publish_summaries"] * output_rows / sample_rows
                + residual * batches / fixture["batches"]
            )
            bp = fixture["behavior_phase_seconds"]

            def tile(n: int, allowance: int = int(fixture["analysis_batch_bytes"])) -> int:
                return max(1, allowance // (4 * (n + 1) * 2 * 2048))

            tiles = math.ceil(strategies / tile(k))
            fixture_tiles = math.ceil(fixture["strategies"] / tile(fixture["k"]))
            tile_ratio = (
                cells * batches * tiles / (fixture_cells * fixture["batches"] * fixture_tiles)
            )
            final_and_batch_rows = 81 * cells + 2 * strategies * k * batches
            sample_behavior = fixture["measurements"]["behavior"]["outputs"]
            sample_behavior_rows = (
                sample_behavior["seat_behavior_summaries.parquet"]["rows"]
                + sample_behavior["seat_behavior_batch.parquet"]["rows"]
            )
            residual_b = max(0.0, fixture["measurements"]["behavior"]["seconds"] - sum(bp.values()))
            behavior = (
                bp["source"] * source_ratio
                + bp.get("merge", 0) * source_ratio * merge_factor
                + bp["certify_batches"] * mixed_ratio
                + bp["summaries"] * tile_ratio
                + bp["publish_rows"] * final_and_batch_rows / sample_behavior_rows
                + bp["reconcile"] * seat_grid_ratio
                + residual_b * batches / fixture["batches"]
            )
            # Retained units mix source records and dense products. This is a
            # sensitivity estimate, not a proof of a storage upper bound.
            checkpoint_ratio = (
                source_fraction * source_ratio + (1 - source_fraction) * seat_grid_ratio
            )
            ob = fixture["measurements"]["outcomes"]["retained_unit_bytes"] * checkpoint_ratio
            ob += sum(v["bytes"] for v in sample_outputs.values()) * output_rows / sample_rows
            bb = fixture["measurements"]["behavior"]["retained_unit_bytes"] * checkpoint_ratio
            bb += (
                sum(v["bytes"] for v in sample_behavior.values())
                * final_and_batch_rows
                / sample_behavior_rows
            )
            per_endpoint.append((outcome, behavior, ob, bb))
        values = [interpolate(k, per_endpoint[0][i], per_endpoint[1][i]) for i in range(4)]
        outcome_seconds += values[0]
        behavior_seconds += values[1]
        outcome_bytes += values[2]
        behavior_bytes += values[3]
    maximum_primary_cells = roots * total_cells * 4
    # Dense fixture actually activates 4,800 primary cells; withheld seat cells
    # from the old 30-batch fixture cannot stand in for active production work.
    precision_ratio = maximum_primary_cells / int(precision["resampled_primary_cells"])
    precision_ratio *= batches / int(precision["batches"])
    precision_seconds = float(precision["elapsed_seconds"]) * precision_ratio
    precision_bytes = float(precision["retained_checkpoint_bytes"]) * precision_ratio
    return {
        "outcomes_hours": roots * outcome_seconds / 3600,
        "behavior_hours": roots * behavior_seconds / 3600,
        "precision_hours": precision_seconds / 3600,
        "added_durable_bytes": roots * (outcome_bytes + behavior_bytes) + precision_bytes,
        "maximum_primary_resampled_cells": maximum_primary_cells,
        "precision_raw_replicate_value_bytes": maximum_primary_cells * 2000 * 8,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        print(f"existing capacity record preserved: {args.output}")
        return
    repo = Path(__file__).resolve().parents[1]
    config = repo / "configs/farkle_mega_config.yaml"
    cfg = load_app_config(config, seed_list_len=2)
    cfg.validate_statistical_contract(require_two_roots=True)
    cfg.validate_resource_contract()
    if cfg.batching.target_batches != 100 or cfg.screening.bootstrap_replicates != 2000:
        raise ValueError("Task 10 capacity coefficients require 100 batches and 2,000 replicates")
    dimensions = derive_dimensions(cfg)
    paths = [args.measurements_dir / f"phase-costs-k{k}-final.json" for k in (2, 12)]
    precision_path = args.measurements_dir / "dense-precision-final.json"
    baseline_path = repo / "docs/remediation/task7a_production_capacity.json"
    low, high = [read(p) for p in paths]
    precision, baseline = read(precision_path), read(baseline_path)
    for record in (low, high):
        for name, digest in record["code_sha256"].items():
            if sha256_file(repo / name) != digest:
                raise ValueError(f"measurement source mismatch: {name}")
    for name, digest in precision["source_sha256"].items():
        if sha256_file(repo / name) != digest:
            raise ValueError(f"precision source mismatch: {name}")
    cases = []
    settings = (
        ("lower_cost", 30, 0.0, 1.0, 0.75, 0.75, 1.0, "plausible_lower_hours", "lower"),
        ("central_serial_coefficients", 150, 0.5, 1.5, 1.0, 1.0, 3.0, "central_hours", "central"),
        ("high_cost", 500, 1.0, 2.0, 2.0, 2.0, 8.0, "conservative_planning_upper_hours", "upper"),
    )
    for name, bins, fraction, merge, io, storage_factor, fixed, old_hours, storage in settings:
        result = scenario(
            dimensions,
            (low, high),
            precision,
            bins=bins,
            source_fraction=fraction,
            merge_factor=merge,
        )
        added = (
            io * sum(result[k] for k in ("outcomes_hours", "behavior_hours", "precision_hours"))
            + fixed
        )
        cases.append(
            {
                "name": name,
                "assumptions": {
                    "frequency_bins_per_final_cell": bins,
                    "mixed_phase_source_fraction": fraction,
                    "merge_generation_factor": merge,
                    "host_io_factor": io,
                    "durable_compression_support_factor": storage_factor,
                    "additional_monitor_parameter_guide_audit_hours": fixed,
                },
                "added": result,
                "legacy_base_hours": baseline["projection"][old_hours],
                "added_hours_after_factors": added,
                "total_hours": baseline["projection"][old_hours] + added,
                "total_days": (baseline["projection"][old_hours] + added) / 24,
                "total_durable_bytes": baseline["capacity"]["durable_storage_bytes"][storage]
                + storage_factor * result["added_durable_bytes"],
            }
        )
    inputs = [config, *paths, precision_path, baseline_path]
    result = {
        "schema_version": 1,
        "contract_id": "first_release_task0_v1",
        "task": "10",
        "production_executed": False,
        "dimensions": dimensions,
        "input_evidence": [{"path": str(p), "sha256": sha256_file(p)} for p in inputs],
        "scenarios": cases,
        "method": "serial measured coefficients with explicit workload-driver and engineering sensitivities",
        "limitations": [
            "Scenarios are not confidence intervals, guaranteed bounds or observed production timings.",
            "No unmeasured parallel speedup or ingest-coalescing improvement is credited.",
            "Legacy base is historical OneDrive evidence; new fixtures use local C: and concurrent validation can confound timings.",
            "Final bin cardinality, compression, merge shrinkage and active conditional support remain unknown.",
            "Precision assumes every primary cell can resample; actual sparse/degenerate support can lower work.",
            "Monitor/parameter/guide/audit overhead is an explicit sensitivity allowance, not a measured production coefficient.",
            "Retained canonical checkpoints count as durable storage. Disposable scratch and cross-volume bridges need additional peak headroom.",
            "Storage estimates do not bound uncompressible data or arbitrarily broad empirical support.",
        ],
        "operational_off_ramps": {
            "h2h_configured_attempt_cap": cfg.head2head.total_game_cap,
            "h2h_envelope_maximum_attempts": dimensions["h2h"]["maximum_total_attempts"]["value"],
            "authorization": "exact frozen plan consent or blocked_by_cap exit 3; no cap edit",
            "source_metadata": "ingest physical layout 3 coalesces within existing byte budget; footer admission still fails closed",
            "production_authorized": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(args.output)) as staged:
        Path(staged).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
