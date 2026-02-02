# src/farkle/analysis/pipeline.py
"""CLI pipeline orchestrator for ingest, curation, and analysis.

Defines sequential stages for loading simulation outputs, combining metrics,
and generating reports, mirroring the repository's documented workflow.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Sequence, overload

import yaml  # type: ignore[import-untyped]

from farkle import analysis
from farkle.analysis import combine, curate, game_stats, ingest, metrics, rng_diagnostics
from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig, load_app_config
from farkle.analysis.stage_runner import StagePlanItem, StageRunContext, StageRunner
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


@overload
def _stringify_paths(obj: dict[str, Any]) -> dict[str, Any]:
    ...


@overload
def _stringify_paths(obj: list[Any]) -> list[Any]:
    ...


@overload
def _stringify_paths(obj: tuple[Any, ...]) -> tuple[Any, ...]:
    ...


@overload
def _stringify_paths(obj: Path) -> str:
    ...


@overload
def _stringify_paths(obj: Any) -> Any:
    ...


def _stringify_paths(obj: Any) -> Any:
    """Convert Paths nested inside mappings/sequences into plain strings."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_paths(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_stringify_paths(v) for v in obj)
    return obj


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for the analysis pipeline."""

    parser = argparse.ArgumentParser(prog="farkle-analyze")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/fast_config.yaml"), help="Path to YAML config"
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
        "--disable-trueskill",
        dest="disable_trueskill",
        action="store_true",
        default=None,
        help="Disable the trueskill analytics stage",
    )
    parser.add_argument(
        "--disable-head2head",
        dest="disable_head2head",
        action="store_true",
        default=None,
        help="Disable the head-to-head analytics stage",
    )
    parser.add_argument(
        "--disable-hgb",
        dest="disable_hgb",
        action="store_true",
        default=None,
        help="Disable the histogram-based gradient boosting analytics stage",
    )
    parser.add_argument(
        "--disable-tiering",
        dest="disable_tiering",
        action="store_true",
        default=None,
        help="Disable the tiering analytics stage",
    )
    parser.add_argument(
        "--disable-agreement",
        dest="disable_agreement",
        action="store_true",
        default=None,
        help="Disable the agreement analytics stage",
    )
    parser.add_argument(
        "--disable-game-stats",
        dest="disable_game_stats",
        action="store_true",
        default=None,
        help="Disable the game-stats analytics stage",
    )
    parser.add_argument(
        "--disable-rng-diagnostics",
        dest="disable_rng_diagnostics",
        action="store_true",
        default=None,
        help="Disable the RNG diagnostics analytics stage",
    )
    game_stats_group = parser.add_mutually_exclusive_group()
    game_stats_group.add_argument(
        "--game-stats",
        dest="run_game_stats",
        action="store_true",
        default=None,
        help="Run the game_stats stage after metrics (default: config)",
    )
    game_stats_group.add_argument(
        "--no-game-stats",
        dest="run_game_stats",
        action="store_false",
        default=None,
        help="Skip the game_stats stage after metrics",
    )
    rng_group = parser.add_mutually_exclusive_group()
    rng_group.add_argument(
        "--rng-diagnostics",
        dest="rng_diagnostics",
        action="store_true",
        default=None,
        help="Run the RNG diagnostics stage over curated rows (interseed only)",
    )
    rng_group.add_argument(
        "--no-rng-diagnostics",
        dest="rng_diagnostics",
        action="store_false",
        default=None,
        help="Skip the RNG diagnostics stage (interseed only)",
    )
    interseed_group = parser.add_mutually_exclusive_group()
    interseed_group.add_argument(
        "--interseed",
        dest="run_interseed",
        action="store_true",
        default=None,
        help="Include cross-seed analytics stages (default: config)",
    )
    interseed_group.add_argument(
        "--per-seed-only",
        dest="run_interseed",
        action="store_false",
        default=None,
        help="Skip cross-seed analytics stages like variance, meta, and pooled TrueSkill",
    )
    parser.add_argument(
        "--margin-thresholds",
        type=int,
        nargs="+",
        help="Override victory-margin thresholds used by game-stats outputs",
    )
    parser.add_argument(
        "--rare-event-target",
        type=int,
        help="Override the target score used to flag rare-event summaries",
    )
    parser.add_argument(
        "--rare-event-margin-quantile",
        type=float,
        help="Quantile for deriving the rare-event margin threshold",
    )
    parser.add_argument(
        "--rare-event-target-rate",
        type=float,
        help="Target rate for multi-target rare-event thresholds",
    )
    parser.add_argument(
        "--rng-lags",
        dest="rng_lags",
        type=int,
        action="append",
        help="Optional lags for RNG diagnostics (repeatable; defaults to 1)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "curate", "combine", "metrics", "analytics", "all"):
        sub.add_parser(name)

    parse = getattr(parser, "parse_intermixed_args", None)
    try:
        args = parse(argv) if parse else parser.parse_args(argv)
    except TypeError:
        args = parser.parse_args(argv)

    if args.seed_pair and (args.seed_a is not None or args.seed_b is not None):
        parser.error("Use --seed-pair or --seed-a/--seed-b, not both.")
    if (args.seed_a is None) ^ (args.seed_b is None):
        parser.error("--seed-a and --seed-b must be provided together.")
    seed_pair_override = None
    if args.seed_pair:
        seed_pair_override = (int(args.seed_pair[0]), int(args.seed_pair[1]))
    elif args.seed_a is not None and args.seed_b is not None:
        seed_pair_override = (int(args.seed_a), int(args.seed_b))

    LOGGER.info(
        "Analysis pipeline start",
        extra={"stage": "pipeline", "command": args.command, "config": str(args.config)},
    )
    app_cfg = load_app_config(Path(args.config))
    if seed_pair_override is not None:
        app_cfg.sim.seed_list = list(seed_pair_override)
        app_cfg.sim.seed_pair = seed_pair_override
    if args.margin_thresholds:
        app_cfg.analysis.game_stats_margin_thresholds = tuple(args.margin_thresholds)
    if args.rare_event_target is not None:
        app_cfg.analysis.rare_event_target_score = int(args.rare_event_target)
    if args.rare_event_margin_quantile is not None:
        app_cfg.analysis.rare_event_margin_quantile = float(args.rare_event_margin_quantile)
    if args.rare_event_target_rate is not None:
        app_cfg.analysis.rare_event_target_rate = float(args.rare_event_target_rate)
    rng_lags = (
        tuple(sorted({int(lag) for lag in args.rng_lags})) if args.rng_lags else None
    )

    deprecated_cli_flags = {
        "run_game_stats": args.run_game_stats is not None,
        "rng_diagnostics": args.rng_diagnostics is not None,
        "run_interseed": args.run_interseed is not None,
        "disable_trueskill": bool(args.disable_trueskill),
        "disable_head2head": bool(args.disable_head2head),
        "disable_hgb": bool(args.disable_hgb),
        "disable_tiering": bool(args.disable_tiering),
        "disable_agreement": bool(args.disable_agreement),
        "disable_game_stats": bool(args.disable_game_stats),
        "disable_rng_diagnostics": bool(args.disable_rng_diagnostics),
    }
    for flag, enabled in deprecated_cli_flags.items():
        if enabled:
            LOGGER.warning(
                "Deprecated CLI flag ignored; stages now run based on inputs",
                extra={"stage": "pipeline", "flag": flag},
            )

    layout = resolve_stage_layout(app_cfg)
    app_cfg.set_stage_layout(layout)
    for placement in layout.placements:
        app_cfg.stage_dir(placement.definition.key)
    analysis_dir = app_cfg.analysis_dir
    resolved = analysis_dir / "config.resolved.yaml"
    # Best-effort: write out the resolved (merged) config we actually used
    resolved_layout = layout.to_resolved_layout()
    stage_layout_snapshot = app_cfg._stage_layout
    app_cfg._stage_layout = None
    resolved_dict: dict[str, Any] = _stringify_paths(dataclasses.asdict(app_cfg))
    app_cfg._stage_layout = stage_layout_snapshot
    resolved_dict.pop("config_sha", None)
    resolved_dict.pop("_stage_layout", None)
    resolved_dict["stage_layout"] = resolved_layout
    resolved_yaml = yaml.safe_dump(resolved_dict, sort_keys=True)
    with atomic_path(str(resolved)) as tmp_path:
        Path(tmp_path).write_text(resolved_yaml)

    # NDJSON manifest (append-only)
    manifest_path = analysis_dir / app_cfg.manifest_name
    config_sha = hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()
    app_cfg.config_sha = config_sha  # allow downstream caching helpers to compare configs

    def _optional_stage(module: str, stage: str) -> Callable[[AppConfig], None]:
        def _runner(cfg: AppConfig) -> None:
            stage_log = analysis.stage_logger(stage, logger=LOGGER)
            mod = analysis._optional_import(module, stage_log=stage_log)
            if mod is None:
                return
            mod.run(cfg)

        return _runner

    def _head2head_only(cfg: AppConfig) -> None:
        stage_log = analysis.stage_logger("head2head", logger=LOGGER)
        head2head_mod = analysis._optional_import("farkle.analysis.head2head", stage_log=stage_log)
        if head2head_mod is not None:
            head2head_mod.run(cfg)

    def _post_h2h_only(cfg: AppConfig) -> None:
        post_log = analysis.stage_logger("post_h2h", logger=LOGGER)
        post_mod = analysis._optional_import("farkle.analysis.h2h_analysis", stage_log=post_log)
        if post_mod is not None:
            post_mod.run_post_h2h(cfg)

    def _interseed_summary(cfg: AppConfig) -> None:
        stage_log = analysis.stage_logger("interseed", logger=LOGGER)
        interseed_mod = analysis._optional_import(
            "farkle.analysis.interseed_analysis", stage_log=stage_log
        )
        if interseed_mod is None:
            return
        interseed_mod.run(cfg, run_stages=False)

    stage_map: dict[str, Callable[[AppConfig], None]] = {
        "ingest": ingest.run,
        "curate": curate.run,
        "combine": combine.run,
        "metrics": metrics.run,
        "game_stats": lambda cfg: game_stats.run(cfg),
        "rng_diagnostics": lambda cfg: rng_diagnostics.run(cfg, lags=rng_lags),
        "seed_summaries": analysis.run_seed_summaries,
        "variance": analysis.run_variance,
        "meta": analysis.run_meta,
        "trueskill": _optional_stage("farkle.analysis.trueskill", "trueskill"),
        "head2head": _head2head_only,
        "post_h2h": _post_h2h_only,
        "hgb": _optional_stage("farkle.analysis.hgb_feat", "hgb"),
        "tiering": _optional_stage("farkle.analysis.tiering_report", "tiering"),
        "agreement": _optional_stage("farkle.analysis.agreement", "agreement"),
        "interseed": _interseed_summary,
    }

    def _plan_steps(allowed_keys: set[str] | None) -> list[tuple[str, Callable[[AppConfig], None]]]:
        return [
            (placement.definition.key, stage_map[placement.definition.key])
            for placement in layout.placements
            if (allowed_keys is None or placement.definition.key in allowed_keys)
        ]

    if args.command == "all":
        steps = _plan_steps(None)
    elif args.command == "analytics":
        analytics_keys = {
            placement.definition.key
            for placement in layout.placements
            if placement.definition.group == "analytics"
        }
        steps = _plan_steps(analytics_keys)
    elif args.command == "metrics":
        metrics_keys = {"metrics", "game_stats"}
        metrics_keys.add("rng_diagnostics")
        steps = _plan_steps(metrics_keys)
    elif args.command == "combine":
        combine_keys = {"combine"}
        combine_keys.add("rng_diagnostics")
        steps = _plan_steps(combine_keys)
    elif args.command in stage_map:
        steps = _plan_steps({args.command})
    else:  # pragma: no cover - argparse enforces valid choices
        parser.error(f"Unknown command {args.command}")
    plan = [StagePlanItem(name=step_name, action=fn) for step_name, fn in steps]
    context = StageRunContext(
        config=app_cfg,
        manifest_path=manifest_path,
        run_label="pipeline",
        run_metadata={
            "command": args.command,
            "config_sha": config_sha,
            "resolved_config": str(resolved),
            "results_dir": str(app_cfg.results_root),
            "analysis_dir": str(analysis_dir),
        },
        run_end_metadata={"config_sha": config_sha},
        stage_start_event="step_start",
        stage_end_event="step_end",
        continue_on_error=True,
        use_progress=len(plan) > 1,
        progress_desc="pipeline",
        logger=LOGGER,
    )
    StageRunner.run(plan, context, raise_on_failure=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
