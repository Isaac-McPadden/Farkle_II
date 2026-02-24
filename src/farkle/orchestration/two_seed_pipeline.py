"""Two-seed simulation + analysis pipeline orchestrator."""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from farkle import analysis
from farkle.analysis import combine, curate, game_stats, ingest, metrics
from farkle.analysis.stage_runner import StagePlanItem, StageRunContext, StageRunner
from farkle.config import (
    AppConfig,
    apply_dot_overrides,
    assign_config_sha,
    load_app_config,
)
from farkle.orchestration.run_contexts import InterseedRunContext, SeedRunContext
from farkle.orchestration.seed_utils import (
    prepare_seed_config,
    seed_has_completion_markers,
    seed_pair_meta_root,
    seed_pair_root,
    seed_pair_seed_root,
    write_active_config,
)
from farkle.simulation import runner
from farkle.utils.logging import setup_info_logging
from farkle.utils.manifest import append_manifest_line
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SeedRunStatus:
    seed: int
    context: SeedRunContext
    analysis_ok: bool
    error: str | None = None


def _per_seed_worker_budget(total_workers: int, seed_count: int) -> int:
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    return max(1, total_workers // seed_count)


def _derive_per_seed_job_budgets(cfg: AppConfig, seed_count: int) -> tuple[int, int, int]:
    available_workers = cfg.sim.n_jobs if cfg.sim.n_jobs and cfg.sim.n_jobs > 0 else (os.cpu_count() or 1)
    per_seed_workers = _per_seed_worker_budget(int(available_workers), seed_count)
    simulation_workers = per_seed_workers
    ingest_workers = min(per_seed_workers, max(1, int(cfg.ingest.n_jobs)))
    analysis_workers = per_seed_workers
    return simulation_workers, ingest_workers, analysis_workers


def _build_seed_cfg(
    cfg: AppConfig,
    *,
    seed_pair: tuple[int, int],
    seed: int,
    meta_dir: Path,
    simulation_workers: int,
    ingest_workers: int,
    analysis_workers: int,
) -> AppConfig:
    seed_cfg = prepare_seed_config(
        cfg,
        seed=seed,
        base_results_dir=seed_pair_seed_root(cfg, seed_pair, seed),
        meta_analysis_dir=meta_dir,
    )
    seed_cfg.sim.n_jobs = simulation_workers
    seed_cfg.ingest.n_jobs = ingest_workers
    seed_cfg.analysis.n_jobs = analysis_workers
    return seed_cfg


def _run_one_seed(
    cfg: AppConfig,
    *,
    seed: int,
    seed_pair: tuple[int, int],
    meta_dir: Path,
    manifest_path: Path,
    force: bool,
    simulation_workers: int,
    ingest_workers: int,
    analysis_workers: int,
) -> _SeedRunStatus:
    seed_cfg = _build_seed_cfg(
        cfg,
        seed_pair=seed_pair,
        seed=seed,
        meta_dir=meta_dir,
        simulation_workers=simulation_workers,
        ingest_workers=ingest_workers,
        analysis_workers=analysis_workers,
    )
    seed_context = SeedRunContext.from_config(seed_cfg)
    active_config_path = seed_context.active_config_path

    append_manifest_line(
        manifest_path,
        {
            "event": "seed_start",
            "seed": seed,
            "results_dir": str(seed_cfg.results_root),
            "active_config": str(active_config_path),
            "config_sha": seed_cfg.config_sha,
        },
    )

    write_active_config(seed_cfg)
    LOGGER.info(
        "Using resolved config",
        extra={
            "stage": "orchestration",
            "seed": seed,
            "results_dir": str(seed_cfg.results_root),
            "active_config": str(active_config_path),
        },
    )

    if not force and seed_has_completion_markers(seed_cfg):
        LOGGER.info(
            "Skipping seed run (completion markers found)",
            extra={
                "stage": "orchestration",
                "seed": seed,
                "results_dir": str(seed_cfg.results_root),
            },
        )
        append_manifest_line(
            manifest_path,
            {
                "event": "seed_simulation_skipped",
                "seed": seed,
                "results_dir": str(seed_context.results_root),
            },
        )
    else:
        LOGGER.info(
            "Running simulation",
            extra={
                "stage": "orchestration",
                "seed": seed,
                "results_dir": str(seed_cfg.results_root),
            },
        )
        runner.run_tournament(seed_cfg, force=force)
        append_manifest_line(
            manifest_path,
            {
                "event": "seed_simulation_complete",
                "seed": seed,
                "results_dir": str(seed_context.results_root),
            },
        )

    LOGGER.info(
        "Running per-seed analysis",
        extra={
            "stage": "orchestration",
            "seed": seed,
            "results_dir": str(seed_cfg.results_root),
        },
    )
    try:
        _run_per_seed_analysis(seed_cfg, manifest_path=manifest_path, seed=seed)
        append_manifest_line(
            manifest_path,
            {
                "event": "seed_analysis_complete",
                "seed": seed,
                "results_dir": str(seed_context.results_root),
            },
        )
        return _SeedRunStatus(seed=seed, context=seed_context, analysis_ok=True)
    except Exception as exc:  # noqa: BLE001
        append_manifest_line(
            manifest_path,
            {
                "event": "seed_analysis_failed",
                "seed": seed,
                "results_dir": str(seed_context.results_root),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return _SeedRunStatus(
            seed=seed,
            context=seed_context,
            analysis_ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


@dataclass(frozen=True)
class _PipelineStageContract:
    name: str
    required_outputs: tuple[Path, ...]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ResolvedStageStatus:
    status: str
    diagnostics: list[str]
    required_outputs: tuple[Path, ...]
    missing_outputs: tuple[Path, ...]


def _is_valid_artifact(path: Path) -> tuple[bool, str | None]:
    if not path.exists():
        return False, "missing"
    if path.stat().st_size <= 0:
        return False, "empty"
    if path.suffix in {".json", ".jsonl"}:
        try:
            if path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if stripped:
                        json.loads(stripped)
                        break
                else:
                    return False, "empty jsonl"
            else:
                json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return False, "unparseable metadata"
    return True, None


def _resolve_stage_contract_status(
    contract: _PipelineStageContract,
    *,
    dependency_statuses: Sequence[str] = (),
    explicit_error: str | None = None,
) -> _ResolvedStageStatus:
    diagnostics: list[str] = []
    if explicit_error:
        diagnostics.append(explicit_error)
    if dependency_statuses and any(status != "success" for status in dependency_statuses):
        diagnostics.append(f"upstream incomplete: {', '.join(dependency_statuses)}")

    missing_outputs: list[Path] = []
    for output_path in contract.required_outputs:
        valid, reason = _is_valid_artifact(output_path)
        if not valid:
            missing_outputs.append(output_path)
            diagnostics.append(f"{output_path}: {reason}")

    if explicit_error or any("unparseable metadata" in reason for reason in diagnostics):
        status = "failed"
    elif missing_outputs:
        status = "missing"
    else:
        status = "success"
    return _ResolvedStageStatus(
        status=status,
        diagnostics=diagnostics,
        required_outputs=contract.required_outputs,
        missing_outputs=tuple(missing_outputs),
    )


def _resolve_seed_family_statuses(
    seed: int,
    *,
    seed_cfg: AppConfig,
    analysis_error: str | None,
) -> dict[str, _ResolvedStageStatus]:
    analysis_manifest_candidates = (
        seed_cfg.analysis_dir / seed_cfg.manifest_name,
        seed_cfg.analysis_dir / "analysis_manifest.jsonl",
    )
    analysis_manifest = next(
        (path for path in analysis_manifest_candidates if path.exists()),
        analysis_manifest_candidates[0],
    )
    stage_contracts = (
        _PipelineStageContract(
            name=f"seed_{seed}.analysis",
            required_outputs=(analysis_manifest,),
        ),
        _PipelineStageContract(
            name=f"seed_{seed}.seed_symmetry",
            required_outputs=(seed_cfg.seed_symmetry_stage_dir / "seed_symmetry_summary.parquet",),
            depends_on=(f"seed_{seed}.analysis",),
        ),
        _PipelineStageContract(
            name=f"seed_{seed}.post_h2h",
            required_outputs=(seed_cfg.post_h2h_stage_dir / "h2h_s_tiers.json",),
            depends_on=(f"seed_{seed}.analysis",),
        ),
    )
    analysis_status = _resolve_stage_contract_status(stage_contracts[0], explicit_error=analysis_error)
    return {
        stage_contracts[0].name: analysis_status,
        stage_contracts[1].name: _resolve_stage_contract_status(
            stage_contracts[1], dependency_statuses=(analysis_status.status,)
        ),
        stage_contracts[2].name: _resolve_stage_contract_status(
            stage_contracts[2], dependency_statuses=(analysis_status.status,)
        ),
    }


def _resolve_interseed_family_statuses(
    *,
    interseed_cfg: AppConfig,
    seed_pair: tuple[int, int],
    stage_statuses: dict[str, dict[str, Any]],
    interseed_error: str | None,
    h2h_error: str | None,
) -> dict[str, _ResolvedStageStatus]:
    per_seed_post_h2h_stages = tuple(f"seed_{seed}.post_h2h" for seed in seed_pair)
    interseed_contract = _PipelineStageContract(
        name="interseed_analysis",
        required_outputs=(interseed_cfg.interseed_stage_dir / "interseed_summary.json",),
        depends_on=per_seed_post_h2h_stages,
    )
    interseed_status = _resolve_stage_contract_status(
        interseed_contract,
        dependency_statuses=tuple(stage_statuses.get(name, {}).get("status", "missing") for name in per_seed_post_h2h_stages),
        explicit_error=interseed_error,
    )

    h2h_contract = _PipelineStageContract(
        name="h2h_tier_trends",
        required_outputs=(interseed_cfg.stage_dir("h2h_tier_trends") / "s_tier_trends.parquet",),
        depends_on=("interseed_analysis", *per_seed_post_h2h_stages),
    )
    h2h_status = _resolve_stage_contract_status(
        h2h_contract,
        dependency_statuses=(interseed_status.status,),
        explicit_error=h2h_error,
    )
    return {
        interseed_contract.name: interseed_status,
        h2h_contract.name: h2h_status,
    }


def _write_pipeline_health(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")




def _validate_required_config_sha_outputs(
    *,
    expected_sha: str,
    manifest_path: Path,
    seed_contexts: Mapping[int, SeedRunContext],
    interseed_cfg: AppConfig,
) -> list[str]:
    errors: list[str] = []

    def _check_json(path: Path) -> None:
        if not path.exists():
            errors.append(f"missing metadata: {path}")
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid metadata {path}: {type(exc).__name__}: {exc}")
            return
        value = payload.get("config_sha") if isinstance(payload, dict) else None
        if value != expected_sha:
            errors.append(f"{path} config_sha={value!r} expected {expected_sha!r}")

    if not manifest_path.exists():
        errors.append(f"missing metadata: {manifest_path}")
    else:
        records: list[dict[str, Any]] = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"invalid metadata {manifest_path}: {type(exc).__name__}: {exc}")
                continue
            if isinstance(parsed, dict):
                records.append(parsed)

        run_start_idx = 0
        for idx in range(len(records) - 1, -1, -1):
            record = records[idx]
            if record.get("event") == "run_start" and "seed_pair" in record:
                run_start_idx = idx
                break
        current_run_records = records[run_start_idx:]

        for record in current_run_records:
            event = record.get("event")
            if event in {"run_start", "run_end", "stage-end", "seed_start"} and record.get(
                "config_sha"
            ) != expected_sha:
                errors.append(f"manifest event {event} has config_sha={record.get('config_sha')!r}")

    for seed_context in seed_contexts.values():
        _check_json(seed_context.active_config_path.with_suffix(".done.json"))

    rng_stage_dir = interseed_cfg.stage_dir_if_active("rng_diagnostics")
    if rng_stage_dir is not None:
        rng_done = rng_stage_dir / "rng_diagnostics.done.json"
        if rng_done.exists():
            _check_json(rng_done)

    return errors

def _resolve_seed_pair(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[int, int] | None:
    if args.seed_pair and (args.seed_a is not None or args.seed_b is not None):
        parser.error("Use --seed-pair or --seed-a/--seed-b, not both.")
    if (args.seed_a is None) ^ (args.seed_b is None):
        parser.error("--seed-a and --seed-b must be provided together.")
    if args.seed_pair:
        return (int(args.seed_pair[0]), int(args.seed_pair[1]))
    if args.seed_a is not None and args.seed_b is not None:
        return (int(args.seed_a), int(args.seed_b))
    return None


def _shared_meta_dir(cfg: AppConfig, pair_root: Path, seed_pair: tuple[int, int]) -> Path:
    configured_meta = seed_pair_meta_root(cfg, seed_pair)
    if configured_meta is not None:
        return configured_meta
    return pair_root / "interseed_analysis" / "seed_summaries_meta"


def _run_per_seed_analysis(
    cfg: AppConfig,
    *,
    manifest_path: Path,
    seed: int,
) -> None:
    plan: list[StagePlanItem] = [
        StagePlanItem("ingest", ingest.run),
        StagePlanItem("curate", curate.run),
        StagePlanItem("combine", combine.run),
        StagePlanItem("metrics", metrics.run),
    ]
    plan.append(StagePlanItem("game_stats", game_stats.run))
    plan.append(
        StagePlanItem(
            "single_seed_analysis",
            lambda cfg: analysis.run_single_seed_analysis(cfg, manifest_path=manifest_path),
        )
    )
    context = StageRunContext(
        config=cfg,
        manifest_path=manifest_path,
        run_label=f"per_seed_pipeline_{seed}",
        run_metadata={
            "seed": seed,
            "results_dir": str(cfg.results_root),
            "analysis_dir": str(cfg.analysis_dir),
            "config_sha": cfg.config_sha,
        },
        run_end_metadata={"config_sha": cfg.config_sha},
        continue_on_error=False,
        logger=LOGGER,
    )
    StageRunner.run(plan, context, raise_on_failure=True)


def run_pipeline(
    cfg: AppConfig,
    *,
    seed_pair: tuple[int, int],
    force: bool = False,
) -> None:
    if cfg.config_sha is None:
        assign_config_sha(cfg)
    pair_root = seed_pair_root(cfg, seed_pair)
    meta_dir = _shared_meta_dir(cfg, pair_root, seed_pair)
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pair_root / "two_seed_pipeline_manifest.jsonl"
    pipeline_health_path = pair_root / "pipeline_health.json"
    stage_statuses: dict[str, dict[str, Any]] = {}
    stage_errors: dict[str, str] = {}

    append_manifest_line(
        manifest_path,
        {
            "event": "run_start",
            "seed_pair": list(seed_pair),
            "results_dir": str(pair_root),
            "meta_analysis_dir": str(meta_dir),
            "config_sha": cfg.config_sha,
        },
    )

    seed_contexts: dict[int, SeedRunContext] = {}
    simulation_workers, ingest_workers, analysis_workers = _derive_per_seed_job_budgets(
        cfg,
        len(seed_pair),
    )

    if cfg.orchestration.parallel_seeds:
        with ThreadPoolExecutor(max_workers=len(seed_pair)) as executor:
            futures = {
                seed: executor.submit(
                    _run_one_seed,
                    cfg,
                    seed=seed,
                    seed_pair=seed_pair,
                    meta_dir=meta_dir,
                    manifest_path=manifest_path,
                    force=force,
                    simulation_workers=simulation_workers,
                    ingest_workers=ingest_workers,
                    analysis_workers=analysis_workers,
                )
                for seed in seed_pair
            }
            seed_results = {seed: futures[seed].result() for seed in seed_pair}
    else:
        seed_results = {
            seed: _run_one_seed(
                cfg,
                seed=seed,
                seed_pair=seed_pair,
                meta_dir=meta_dir,
                manifest_path=manifest_path,
                force=force,
                simulation_workers=simulation_workers,
                ingest_workers=ingest_workers,
                analysis_workers=analysis_workers,
            )
            for seed in seed_pair
        }

    for seed in seed_pair:
        seed_result = seed_results[seed]
        seed_context = seed_result.context
        seed_contexts[seed] = seed_context
        seed_cfg = seed_context.config
        if seed_result.error:
            stage_errors[f"seed_{seed}.analysis"] = seed_result.error

        seed_resolved = _resolve_seed_family_statuses(
            seed,
            seed_cfg=seed_cfg,
            analysis_error=seed_result.error,
        )
        for stage_name, resolved in seed_resolved.items():
            stage_statuses[stage_name] = {
                "status": resolved.status,
                "required_outputs": [str(path) for path in resolved.required_outputs],
                "missing_required_outputs": [str(path) for path in resolved.missing_outputs],
                "diagnostics": resolved.diagnostics,
            }
            append_manifest_line(
                manifest_path,
                {
                    "event": "stage-end",
                    "stage": stage_name,
                    "status": resolved.status,
                    "diagnostics": resolved.diagnostics,
                    "missing_required_outputs": [str(path) for path in resolved.missing_outputs],
                    "config_sha": cfg.config_sha,
                },
            )

    interseed_seed = seed_pair[0]
    interseed_seed_context = seed_contexts[interseed_seed]
    interseed_context = InterseedRunContext.from_seed_context(
        interseed_seed_context,
        seed_pair=seed_pair,
        analysis_root=pair_root / "interseed_analysis",
    )
    interseed_cfg = interseed_context.config

    per_seed_post_h2h_stages = tuple(f"seed_{seed}.post_h2h" for seed in seed_pair)
    interseed_should_run = all(
        stage_statuses.get(stage, {}).get("status") == "success" for stage in per_seed_post_h2h_stages
    )
    if interseed_should_run:
        LOGGER.info(
            "Running interseed analysis",
            extra={
                "stage": "orchestration",
                "seed": interseed_seed,
                "results_dir": str(interseed_cfg.results_root),
            },
        )
        append_manifest_line(
            manifest_path,
            {
                "event": "interseed_start",
                "seed": interseed_seed,
                "results_dir": str(interseed_cfg.results_root),
                "active_config": str(interseed_cfg.results_root / "active_config.yaml"),
            },
        )
        try:
            analysis.run_interseed_analysis(
                interseed_cfg,
                force=force,
                manifest_path=manifest_path,
            )
            append_manifest_line(
                manifest_path,
                {
                    "event": "interseed_complete",
                    "seed": interseed_seed,
                    "results_dir": str(interseed_cfg.results_root),
                },
            )
        except Exception as exc:  # noqa: BLE001
            append_manifest_line(
                manifest_path,
                {
                    "event": "interseed_failed",
                    "seed": interseed_seed,
                    "results_dir": str(interseed_cfg.results_root),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            stage_errors["interseed_analysis"] = f"{type(exc).__name__}: {exc}"
    seed_s_tier_paths = [
        seed_contexts[seed].config.post_h2h_stage_dir / "h2h_s_tiers.json" for seed in seed_pair
    ]
    missing_tier_inputs = [path for path in seed_s_tier_paths if not path.exists()]
    tier_should_run = interseed_should_run and "interseed_analysis" not in stage_errors and not missing_tier_inputs
    if tier_should_run:
        LOGGER.info(
            "Running head-to-head tier trends",
            extra={
                "stage": "orchestration",
                "seed": interseed_seed,
                "results_dir": str(interseed_cfg.results_root),
            },
        )
        append_manifest_line(
            manifest_path,
            {
                "event": "h2h_tier_trends_start",
                "seed": interseed_seed,
                "results_dir": str(interseed_cfg.results_root),
                "config_sha": interseed_cfg.config_sha,
            },
        )
        try:
            analysis.run_h2h_tier_trends(
                interseed_cfg,
                force=force,
                seed_s_tier_paths=seed_s_tier_paths,
            )
            append_manifest_line(
                manifest_path,
                {
                    "event": "h2h_tier_trends_complete",
                    "seed": interseed_seed,
                    "results_dir": str(interseed_cfg.results_root),
                    "config_sha": interseed_cfg.config_sha,
                },
            )
        except Exception as exc:  # noqa: BLE001
            stage_errors["h2h_tier_trends"] = f"{type(exc).__name__}: {exc}"

    interseed_resolved = _resolve_interseed_family_statuses(
        interseed_cfg=interseed_cfg,
        seed_pair=seed_pair,
        stage_statuses=stage_statuses,
        interseed_error=stage_errors.get("interseed_analysis"),
        h2h_error=stage_errors.get("h2h_tier_trends"),
    )
    for stage_name, resolved in interseed_resolved.items():
        stage_statuses[stage_name] = {
            "status": resolved.status,
            "depends_on": list(
                tuple(f"seed_{seed}.post_h2h" for seed in seed_pair)
                if stage_name == "interseed_analysis"
                else ("interseed_analysis", *(f"seed_{seed}.post_h2h" for seed in seed_pair))
            ),
            "required_outputs": [str(path) for path in resolved.required_outputs],
            "missing_required_outputs": [str(path) for path in resolved.missing_outputs],
            "diagnostics": resolved.diagnostics,
        }
        append_manifest_line(
            manifest_path,
            {
                "event": "stage-end",
                "stage": stage_name,
                "status": resolved.status,
                "diagnostics": resolved.diagnostics,
                "missing_required_outputs": [str(path) for path in resolved.missing_outputs],
                "config_sha": interseed_cfg.config_sha,
            },
        )

    sha_errors = _validate_required_config_sha_outputs(
        expected_sha=cfg.config_sha or "",
        manifest_path=manifest_path,
        seed_contexts=seed_contexts,
        interseed_cfg=interseed_cfg,
    )
    if sha_errors:
        stage_statuses["config_sha_validation"] = {
            "status": "failed",
            "diagnostics": sha_errors,
            "required_outputs": [
                str(manifest_path),
                *[str(ctx.active_config_path.with_suffix(".done.json")) for ctx in seed_contexts.values()],
            ],
            "missing_required_outputs": [],
        }

    if any(payload["status"] in {"failed", "missing"} for payload in stage_statuses.values()):
        overall_status = "failed_blocked"
    else:
        overall_status = "complete_success"

    first_blocking_failure = next(
        (
            {
                "stage": name,
                "status": payload["status"],
                **(
                    {"missing_required_outputs": payload.get("missing_required_outputs", [])}
                    if payload.get("missing_required_outputs")
                    else {}
                ),
                **(
                    {"error": payload["diagnostics"][0]}
                    if payload.get("diagnostics")
                    else {}
                ),
            }
            for name, payload in stage_statuses.items()
            if payload["status"] in {"failed", "missing"}
        ),
        None,
    )

    health_payload = {
        "seed_pair": list(seed_pair),
        "status": overall_status,
        "config_sha": cfg.config_sha,
        "stage_statuses": stage_statuses,
        "missing_required_outputs": {
            name: payload.get("missing_required_outputs", [])
            for name, payload in stage_statuses.items()
            if payload.get("missing_required_outputs")
        },
        "first_blocking_failure": first_blocking_failure,
    }
    _write_pipeline_health(pipeline_health_path, health_payload)

    append_manifest_line(
        manifest_path,
        {
            "event": "run_end",
            "status": overall_status,
            "health_artifact": str(pipeline_health_path),
            "config_sha": cfg.config_sha,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="farkle-two-seed-pipeline")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/fast_config.yaml"), help="Path to YAML config"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values",
    )
    parser.add_argument(
        "--seed-a",
        type=int,
        help="Override the first seed for dual-seed orchestration",
    )
    parser.add_argument(
        "--seed-b",
        type=int,
        help="Override the second seed for dual-seed orchestration",
    )
    parser.add_argument(
        "--seed-pair",
        type=int,
        nargs=2,
        metavar=("A", "B"),
        help="Override the dual-seed tuple (A B)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when completion markers exist",
    )
    parser.add_argument(
        "--parallel-seeds",
        action="store_true",
        help="Run per-seed simulation and analysis concurrently",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_info_logging()

    cfg = load_app_config(Path(args.config), seed_list_len=2)
    cfg = apply_dot_overrides(cfg, list(args.overrides or []))
    if args.parallel_seeds:
        cfg.orchestration.parallel_seeds = True

    seed_pair = _resolve_seed_pair(args, parser)
    if seed_pair is not None:
        cfg.sim.seed_list = list(seed_pair)
        cfg.sim.seed_pair = seed_pair
    resolved_seeds = cfg.sim.populate_seed_list(2)
    assign_config_sha(cfg)
    seed_pair = (resolved_seeds[0], resolved_seeds[1])

    run_pipeline(cfg, seed_pair=seed_pair, force=args.force)
    return 0


__all__ = ["main", "run_pipeline"]
