# src/farkle/cli/main.py
"""
Command line interface for the :mod:`farkle` package.
See ../../../cli_args.md for details.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import yaml  # type: ignore[import-untyped]

from farkle import analysis as analysis_pkg
from farkle.analysis import combine, curate, ingest, metrics
from farkle.config import (
    AppConfig,
    apply_dot_overrides,
    assign_config_sha,
    effective_config_dict,
    expected_seed_list_length,
    load_app_config,
)
from farkle.orchestration import two_seed_pipeline
from farkle.orchestration.seed_utils import seed_pair_root
from farkle.simulation import runner
from farkle.simulation.time_farkle import measure_sim_times
from farkle.simulation.watch_game import watch_game
from farkle.utils.logging import configure_logging, setup_info_logging
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser for simulation and analysis tasks."""
    parser = argparse.ArgumentParser(prog="farkle")
    parser.add_argument("--config", type=Path, help="Path to YAML configuration")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Root logging level",
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

    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_parser = sub.add_parser("run", help="Run a tournament")
    run_parser.add_argument(
        "--metrics",
        action="store_true",
        help="Collect per-strategy metrics in addition to win counts",
    )
    run_parser.add_argument(
        "--row-dir",
        type=Path,
        help="Write full per-game rows to this directory.  If None, rows will not be recorded",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when existing run artifacts are available",
    )

    # time (benchmark simulation throughput)
    time_parser = sub.add_parser("time", help="Benchmark simulation throughput")
    time_parser.add_argument("--players", type=int, default=5, help="Players per game (default: 5)")
    time_parser.add_argument(
        "--n-games",
        dest="n_games",
        type=int,
        default=1000,
        help="Number of games to run (default: 1000)",
    )
    time_parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs (default: 1)")
    time_parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")

    # watch
    watch_parser = sub.add_parser("watch", help="Interactively watch a game")
    watch_parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic play")

    # analyze
    analyze_parser = sub.add_parser("analyze", help="Data analysis helpers")
    analyze_sub = analyze_parser.add_subparsers(dest="an_cmd", required=True)
    analyze_sub.add_parser("ingest", help="Ingest raw CSV data")
    analyze_sub.add_parser("curate", help="Curate ingested data")
    analyze_sub.add_parser("combine", help="Combine curated data into a superset parquet")
    metrics_parser = analyze_sub.add_parser("metrics", help="Compute metrics")
    metrics_parser.add_argument(
        "--compute-game-stats",
        action="store_true",
        help="Also compute game-length statistics from curated rows",
    )
    metrics_parser.add_argument(
        "--rng-diagnostics",
        action="store_true",
        help="Compute RNG autocorrelation diagnostics from curated rows",
    )
    metrics_parser.add_argument(
        "--rng-lags",
        type=int,
        nargs="+",
        help="Positive lags (default: 1) for RNG diagnostics",
    )
    metrics_parser.add_argument(
        "--margin-thresholds",
        type=int,
        nargs="+",
        help="Victory-margin thresholds used for close-game summaries",
    )
    metrics_parser.add_argument(
        "--rare-event-target",
        type=int,
        help="Target score for multi-player reach flags (default: 10000)",
    )
    metrics_parser.add_argument(
        "--rare-event-margin-quantile",
        type=float,
        help="Quantile to derive the rare-event margin threshold (e.g., 0.001)",
    )
    metrics_parser.add_argument(
        "--rare-event-target-rate",
        type=float,
        help="Target rate for multi-target rare events (e.g., 1e-4)",
    )

    variance_parser = analyze_sub.add_parser(
        "variance", help="Compute cross-seed win-rate variance"
    )
    variance_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when the done-stamp appears fresh",
    )

    preprocess_parser = analyze_sub.add_parser(
        "preprocess", help="Run ingest, curate, combine, and metrics"
    )
    preprocess_parser.add_argument(
        "--compute-game-stats",
        action="store_true",
        help="Also compute game-length statistics from curated rows",
    )
    preprocess_parser.add_argument(
        "--rng-diagnostics",
        action="store_true",
        help="Also compute RNG autocorrelation diagnostics",
    )
    preprocess_parser.add_argument(
        "--rng-lags",
        type=int,
        nargs="+",
        help="Positive lags (default: 1) for RNG diagnostics",
    )
    preprocess_parser.add_argument(
        "--margin-thresholds",
        type=int,
        nargs="+",
        help="Victory-margin thresholds used for close-game summaries",
    )
    preprocess_parser.add_argument(
        "--rare-event-target",
        type=int,
        help="Target score for multi-player reach flags (default: 10000)",
    )
    preprocess_parser.add_argument(
        "--rare-event-margin-quantile",
        type=float,
        help="Quantile to derive the rare-event margin threshold (e.g., 0.001)",
    )
    preprocess_parser.add_argument(
        "--rare-event-target-rate",
        type=float,
        help="Target rate for multi-target rare events (e.g., 1e-4)",
    )

    pipeline_parser = analyze_sub.add_parser(
        "pipeline", help="Run ingest->curate->combine->metrics->analytics pipeline"
    )
    pipeline_parser.add_argument(
        "--compute-game-stats",
        action="store_true",
        help="Also compute game-length statistics from curated rows",
    )
    pipeline_parser.add_argument(
        "--rng-diagnostics",
        action="store_true",
        help="Also compute RNG autocorrelation diagnostics",
    )
    pipeline_parser.add_argument(
        "--rng-lags",
        type=int,
        nargs="+",
        help="Positive lags (default: 1) for RNG diagnostics",
    )
    pipeline_parser.add_argument(
        "--margin-thresholds",
        type=int,
        nargs="+",
        help="Victory-margin thresholds used for close-game summaries",
    )
    pipeline_parser.add_argument(
        "--rare-event-target",
        type=int,
        help="Target score for multi-player reach flags (default: 10000)",
    )
    pipeline_parser.add_argument(
        "--rare-event-margin-quantile",
        type=float,
        help="Quantile to derive the rare-event margin threshold (e.g., 0.001)",
    )
    pipeline_parser.add_argument(
        "--rare-event-target-rate",
        type=float,
        help="Target rate for multi-target rare events (e.g., 1e-4)",
    )
    pipeline_parser.add_argument(
        "--allow-missing-upstream",
        action="store_true",
        help="Allow analytics stages to skip when mandatory upstream artifacts are missing (manual debugging only)",
    )
    analytics_parser = analyze_sub.add_parser("analytics", help="Run analytics modules (TrueSkill, head-to-head, HGB)")
    analytics_parser.add_argument(
        "--allow-missing-upstream",
        action="store_true",
        help="Allow analytics stages to skip when mandatory upstream artifacts are missing (manual debugging only)",
    )
    two_seed_parser = analyze_sub.add_parser(
        "two-seed-pipeline",
        help=(
            "Deprecated: use the top-level `two-seed-pipeline` command for the "
            "two-seed simulation and analysis orchestration pipeline"
        ),
    )
    two_seed_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when completion markers exist",
    )

    # two-seed-pipeline (top-level)
    two_seed_top = sub.add_parser(
        "two-seed-pipeline",
        help="Run the two-seed simulation and analysis orchestration pipeline",
    )
    two_seed_top.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when completion markers exist",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_level(level: str | int) -> int:
    """Normalize a logging level string or integer to ``logging`` constants."""
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return int(level)


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


def _write_active_config(cfg: AppConfig, dest_dir: Path) -> None:
    """Persist the resolved configuration alongside simulation results."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved_dict = effective_config_dict(cfg)
    resolved_yaml = yaml.safe_dump(resolved_dict, sort_keys=True)
    target = dest_dir / "active_config.yaml"
    with atomic_path(str(target)) as tmp_path:
        Path(tmp_path).write_text(resolved_yaml, encoding="utf-8")

    done_payload = {"active_config": str(target), "config_sha": cfg.config_sha}
    done_path = target.with_suffix(".done.json")
    with atomic_path(str(done_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(done_payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_preprocess(
    cfg: AppConfig,
    *,
    compute_game_stats: bool = False,
) -> None:
    """Run ingest, curate, combine, metrics, and optional diagnostics."""
    ingest.run(cfg)
    curate.run(cfg)
    combine.run(cfg)
    metrics.run(cfg)
    if compute_game_stats:
        from farkle.analysis import game_stats

        game_stats.run(cfg)


def _resolve_log_file(args: argparse.Namespace, cfg: AppConfig | None) -> Path | None:
    """Resolve the CLI log destination for commands writing under ``data``."""
    if cfg is None:
        return None

    command = getattr(args, "command", None)
    an_cmd = getattr(args, "an_cmd", None)
    if command == "two-seed-pipeline" or (command == "analyze" and an_cmd == "two-seed-pipeline"):
        seed_pair = cfg.sim.require_seed_pair()
        return seed_pair_root(cfg, seed_pair) / "log.txt"
    if command in {"run", "analyze"}:
        return cfg.results_root / "log.txt"
    return None


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``farkle`` CLI dispatcher."""
    parser = build_parser()
    args, _ = parser.parse_known_args(argv)
    seed_pair_override = _resolve_seed_pair(args, parser)
    expected_seed_len = expected_seed_list_length(
        args.command, subcommand=getattr(args, "an_cmd", None)
    )

    setup_info_logging()
    log_level = _parse_level(args.log_level)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    LOGGER.info(
        "CLI arguments parsed",
        extra={
            "stage": "cli",
            "command": args.command,
            "config_path": str(args.config) if args.config is not None else None,
            "overrides": list(args.overrides or []),
            "log_level": logging.getLevelName(root_logger.level),
        },
    )

    cfg: AppConfig | None = None
    rng_lags: tuple[int, ...] | None = None
    compute_rng_diagnostics: bool | None = None
    if args.command in {"run", "analyze", "two-seed-pipeline"}:
        overlays: list[Path] = [args.config] if args.config is not None else []
        cfg = (
            load_app_config(*overlays, seed_list_len=expected_seed_len)
            if overlays
            else AppConfig()
        )
        cfg = apply_dot_overrides(cfg, list(args.overrides or []))
        if seed_pair_override is not None:
            cfg.sim.seed_list = list(seed_pair_override)
            cfg.sim.seed_pair = seed_pair_override
        if expected_seed_len is not None:
            cfg.sim.populate_seed_list(expected_seed_len)
        assign_config_sha(cfg)

        margin_thresholds = getattr(args, "margin_thresholds", None)
        if margin_thresholds:
            cfg.analysis.game_stats_margin_thresholds = tuple(margin_thresholds)
        rare_event_target = getattr(args, "rare_event_target", None)
        if rare_event_target is not None:
            cfg.analysis.rare_event_target_score = int(rare_event_target)
        rare_event_margin_quantile = getattr(args, "rare_event_margin_quantile", None)
        if rare_event_margin_quantile is not None:
            cfg.analysis.rare_event_margin_quantile = float(rare_event_margin_quantile)
        rare_event_target_rate = getattr(args, "rare_event_target_rate", None)
        if rare_event_target_rate is not None:
            cfg.analysis.rare_event_target_rate = float(rare_event_target_rate)
        rng_lags_arg = getattr(args, "rng_lags", None)
        if rng_lags_arg:
            rng_lags = tuple(sorted({int(lag) for lag in rng_lags_arg}))
        compute_rng_diagnostics = getattr(args, "rng_diagnostics", None)

        LOGGER.info(
            "Configuration prepared",
            extra={
                "stage": "cli",
                "command": args.command,
                "results_dir": str(cfg.results_root),
                "analysis_dir": str(cfg.analysis_dir),
                "n_players_list": list(cfg.sim.n_players_list),
                "expanded_metrics": cfg.sim.expanded_metrics,
            },
        )

    log_file = _resolve_log_file(args, cfg)
    if log_file is not None:
        configure_logging(level=log_level, log_file=log_file)
    else:
        root_logger.setLevel(log_level)

    if args.command == "run":
        if cfg is None:  # pragma: no cover - guarded by command gate above
            parser.error("Run command requires a loaded configuration")
        if args.metrics:
            cfg.sim.expanded_metrics = True
        if args.row_dir is not None:
            cfg.sim.row_dir = args.row_dir
        _write_active_config(cfg, cfg.results_root)
        resume_run = not args.force
        LOGGER.info(
            "Dispatching run command",
            extra={
                "stage": "cli",
                "command": "run",
                "seed": cfg.sim.seed,
                "n_players_list": cfg.sim.n_players_list,
                "expanded_metrics": cfg.sim.expanded_metrics,
                "results_dir": str(cfg.results_root),
                "row_dir": str(cfg.sim.row_dir) if cfg.sim.row_dir is not None else None,
                "force": args.force,
                "resume": resume_run,
            },
        )
        if len(cfg.sim.n_players_list) > 1:
            runner.run_multi(cfg, force=args.force)
        else:
            runner.run_single_n(cfg, cfg.sim.n_players_list[0], force=args.force)
        LOGGER.info("Run command completed", extra={"stage": "cli", "command": "run"})
    elif args.command == "time":
        LOGGER.info(
            "Dispatching measure_sim_times",
            extra={
                "stage": "cli",
                "command": "time",
                "players": args.players,
                "n_games": args.n_games,
                "jobs": args.jobs,
                "seed": args.seed,
            },
        )
        measure_sim_times(
            n_games=args.n_games, players=args.players, seed=args.seed, jobs=args.jobs
        )
    elif args.command == "watch":
        LOGGER.info(
            "Dispatching watch_game",
            extra={"stage": "cli", "command": "watch", "seed": args.seed},
        )
        watch_game(seed=args.seed)
    elif args.command == "analyze":
        if cfg is None:  # pragma: no cover - guarded by command gate above
            parser.error("Analyze command requires a loaded configuration")
        LOGGER.info(
            "Dispatching analysis command",
            extra={
                "stage": "cli",
                "command": f"analyze:{args.an_cmd}",
                "config_path": str(args.config) if args.config else None,
                "results_dir": str(cfg.results_root),
                "analysis_dir": str(cfg.analysis_dir),
            },
        )
        compute_game_stats = getattr(args, "compute_game_stats", False)
        if args.an_cmd == "ingest":
            ingest.run(cfg)
        elif args.an_cmd == "curate":
            curate.run(cfg)
        elif args.an_cmd == "combine":
            combine.run(cfg)
        elif args.an_cmd == "metrics":
            metrics.run(cfg)
        elif args.an_cmd == "preprocess":
            _run_preprocess(
                cfg,
                compute_game_stats=compute_game_stats,
            )
        elif args.an_cmd == "analytics":
            analysis_pkg.run_all(
                cfg,
                run_rng_diagnostics=compute_rng_diagnostics,
                rng_lags=rng_lags,
                allow_missing_upstream=getattr(args, "allow_missing_upstream", False),
            )
        elif args.an_cmd == "variance":
            analysis_pkg.run_variance(cfg, force=getattr(args, "force", False))
        elif args.an_cmd == "two-seed-pipeline":
            LOGGER.warning(
                "Analyze subcommand `two-seed-pipeline` is deprecated; "
                "use `farkle two-seed-pipeline` instead.",
                extra={"stage": "cli", "command": "analyze:two-seed-pipeline"},
            )
            seed_pair = cfg.sim.require_seed_pair()
            two_seed_pipeline.run_pipeline(
                cfg,
                seed_pair=seed_pair,
                force=getattr(args, "force", False),
            )
        elif args.an_cmd == "pipeline":
            _run_preprocess(
                cfg,
                compute_game_stats=compute_game_stats,
            )
            analysis_pkg.run_all(
                cfg,
                run_rng_diagnostics=compute_rng_diagnostics,
                rng_lags=rng_lags,
                allow_missing_upstream=getattr(args, "allow_missing_upstream", False),
            )
        if args.an_cmd == "metrics" and compute_game_stats:
            from farkle.analysis import game_stats

            game_stats.run(cfg)
        LOGGER.info(
            "Analysis command completed",
            extra={"stage": "cli", "command": f"analyze:{args.an_cmd}"},
        )
    elif args.command == "two-seed-pipeline":
        if cfg is None:  # pragma: no cover - guarded by command gate above
            parser.error("Two-seed pipeline command requires a loaded configuration")
        LOGGER.info(
            "Dispatching two-seed pipeline command",
            extra={
                "stage": "cli",
                "command": "two-seed-pipeline",
                "config_path": str(args.config) if args.config else None,
                "results_dir": str(cfg.results_root),
                "analysis_dir": str(cfg.analysis_dir),
            },
        )
        seed_pair = cfg.sim.require_seed_pair()
        two_seed_pipeline.run_pipeline(
            cfg,
            seed_pair=seed_pair,
            force=getattr(args, "force", False),
        )
        LOGGER.info(
            "Two-seed pipeline command completed",
            extra={"stage": "cli", "command": "two-seed-pipeline"},
        )
    else:  # pragma: no cover - argparse enforces valid choices
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
