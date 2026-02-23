"""Two-seed simulation + analysis pipeline orchestrator."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from farkle import analysis
from farkle.analysis import combine, curate, game_stats, ingest, metrics
from farkle.analysis.stage_runner import StagePlanItem, StageRunContext, StageRunner
from farkle.config import AppConfig, apply_dot_overrides, load_app_config
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
class _PipelineStageContract:
    name: str
    required_outputs: tuple[Path, ...]
    depends_on: tuple[str, ...] = tuple()


def _write_pipeline_health(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
        },
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
    pair_root = seed_pair_root(cfg, seed_pair)
    meta_dir = _shared_meta_dir(cfg, pair_root, seed_pair)
    meta_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pair_root / "two_seed_pipeline_manifest.jsonl"
    pipeline_health_path = pair_root / "pipeline_health.json"
    stage_statuses: dict[str, dict[str, Any]] = {}
    first_blocking_failure: dict[str, Any] | None = None

    def _record_stage(
        stage_name: str,
        *,
        status: str,
        required_outputs: Sequence[Path] = tuple(),
        missing_outputs: Sequence[Path] = tuple(),
        depends_on: Sequence[str] = tuple(),
        error: str | None = None,
    ) -> None:
        nonlocal first_blocking_failure
        stage_statuses[stage_name] = {
            "status": status,
            "depends_on": list(depends_on),
            "required_outputs": [str(path) for path in required_outputs],
            "missing_required_outputs": [str(path) for path in missing_outputs],
            **({"error": error} if error else {}),
        }
        if first_blocking_failure is None and status in {"failed", "blocked"}:
            first_blocking_failure = {
                "stage": stage_name,
                "status": status,
                **({"error": error} if error else {}),
                **(
                    {"missing_required_outputs": [str(path) for path in missing_outputs]}
                    if missing_outputs
                    else {}
                ),
            }

    append_manifest_line(
        manifest_path,
        {
            "event": "run_start",
            "seed_pair": list(seed_pair),
            "results_dir": str(pair_root),
            "meta_analysis_dir": str(meta_dir),
        },
    )

    seed_contexts: dict[int, SeedRunContext] = {}
    seed_analysis_ok: dict[int, bool] = {}
    for seed in seed_pair:
        seed_cfg = prepare_seed_config(
            cfg,
            seed=seed,
            base_results_dir=seed_pair_seed_root(cfg, seed_pair, seed),
            meta_analysis_dir=meta_dir,
        )
        seed_context = SeedRunContext.from_config(seed_cfg)
        seed_contexts[seed] = seed_context
        active_config_path = seed_context.active_config_path

        append_manifest_line(
            manifest_path,
            {
                "event": "seed_start",
                "seed": seed,
                "results_dir": str(seed_cfg.results_root),
                "active_config": str(active_config_path),
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
            seed_analysis_ok[seed] = True
            _record_stage(f"seed_{seed}.analysis", status="success")
        except Exception as exc:  # noqa: BLE001
            seed_analysis_ok[seed] = False
            append_manifest_line(
                manifest_path,
                {
                    "event": "seed_analysis_failed",
                    "seed": seed,
                    "results_dir": str(seed_context.results_root),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            _record_stage(
                f"seed_{seed}.analysis",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

        seed_stage_contracts = [
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
        ]
        for contract in seed_stage_contracts:
            missing_outputs = [path for path in contract.required_outputs if not path.exists()]
            if not seed_analysis_ok[seed]:
                _record_stage(
                    contract.name,
                    status="blocked",
                    required_outputs=contract.required_outputs,
                    missing_outputs=missing_outputs,
                    depends_on=contract.depends_on,
                )
            elif missing_outputs:
                _record_stage(
                    contract.name,
                    status="failed",
                    required_outputs=contract.required_outputs,
                    missing_outputs=missing_outputs,
                    depends_on=contract.depends_on,
                )
            else:
                _record_stage(
                    contract.name,
                    status="success",
                    required_outputs=contract.required_outputs,
                    depends_on=contract.depends_on,
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
    interseed_should_run = all(stage_statuses.get(stage, {}).get("status") == "success" for stage in per_seed_post_h2h_stages)
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
            _record_stage("interseed_analysis", status="success", depends_on=per_seed_post_h2h_stages)
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
            _record_stage(
                "interseed_analysis",
                status="failed",
                depends_on=per_seed_post_h2h_stages,
                error=f"{type(exc).__name__}: {exc}",
            )
    else:
        _record_stage(
            "interseed_analysis",
            status="blocked",
            depends_on=per_seed_post_h2h_stages,
        )
    seed_s_tier_paths = [
        seed_contexts[seed].config.post_h2h_stage_dir / "h2h_s_tiers.json" for seed in seed_pair
    ]
    tier_contract = _PipelineStageContract(
        name="h2h_tier_trends",
        required_outputs=(interseed_cfg.stage_dir("h2h_tier_trends") / "s_tier_trends.parquet",),
        depends_on=("interseed_analysis", *per_seed_post_h2h_stages),
    )
    missing_tier_inputs = [path for path in seed_s_tier_paths if not path.exists()]
    tier_should_run = (
        stage_statuses.get("interseed_analysis", {}).get("status") == "success"
        and not missing_tier_inputs
    )
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
            },
        )
        try:
            analysis.run_h2h_tier_trends(
                interseed_cfg,
                force=force,
                seed_s_tier_paths=seed_s_tier_paths,
            )
            missing_tier_outputs = [
                path for path in tier_contract.required_outputs if not path.exists()
            ]
            if missing_tier_outputs:
                _record_stage(
                    tier_contract.name,
                    status="failed",
                    required_outputs=tier_contract.required_outputs,
                    missing_outputs=missing_tier_outputs,
                    depends_on=tier_contract.depends_on,
                )
            else:
                _record_stage(
                    tier_contract.name,
                    status="success",
                    required_outputs=tier_contract.required_outputs,
                    depends_on=tier_contract.depends_on,
                )
                append_manifest_line(
                    manifest_path,
                    {
                        "event": "h2h_tier_trends_complete",
                        "seed": interseed_seed,
                        "results_dir": str(interseed_cfg.results_root),
                    },
                )
        except Exception as exc:  # noqa: BLE001
            _record_stage(
                tier_contract.name,
                status="failed",
                required_outputs=tier_contract.required_outputs,
                depends_on=tier_contract.depends_on,
                error=f"{type(exc).__name__}: {exc}",
            )
    else:
        _record_stage(
            tier_contract.name,
            status="blocked",
            required_outputs=tier_contract.required_outputs,
            missing_outputs=missing_tier_inputs,
            depends_on=tier_contract.depends_on,
        )

    if any(payload["status"] in {"failed", "blocked"} for payload in stage_statuses.values()):
        overall_status = "failed_blocked"
    elif any(payload["status"] == "skipped" for payload in stage_statuses.values()):
        overall_status = "complete_with_skips"
    else:
        overall_status = "complete_success"

    health_payload = {
        "seed_pair": list(seed_pair),
        "status": overall_status,
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_info_logging()

    cfg = load_app_config(Path(args.config), seed_list_len=2)
    cfg = apply_dot_overrides(cfg, list(args.overrides or []))

    seed_pair = _resolve_seed_pair(args, parser)
    if seed_pair is not None:
        cfg.sim.seed_list = list(seed_pair)
        cfg.sim.seed_pair = seed_pair
    resolved_seeds = cfg.sim.populate_seed_list(2)
    seed_pair = (resolved_seeds[0], resolved_seeds[1])

    run_pipeline(cfg, seed_pair=seed_pair, force=args.force)
    return 0


__all__ = ["main", "run_pipeline"]
