"""Command line interface for the :mod:`farkle` package."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from farkle.analysis import combine, curate, ingest, metrics
from farkle.config import AppConfig, apply_dot_overrides, load_app_config
from farkle.simulation import runner
from farkle.simulation.time_farkle import measure_sim_times
from farkle.simulation.watch_game import watch_game
from farkle.utils.logging import setup_info_logging

LOGGER = logging.getLogger(__name__)


def _default_config_path() -> Path | None:
    candidate = Path("configs/farkle_mega_config.yaml")
    if candidate.exists():
        return candidate
    fallback = Path("farkle_mega_config.yaml")
    if fallback.exists():
        return fallback
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="farkle")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML configuration")
    parser.add_argument(
        "--overrides",
        "-O",
        action="append",
        default=[],
        metavar="section.option=value",
        help="Apply dotted overrides after loading config",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level for stdout logging",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run tournaments")
    run_parser.add_argument(
        "--metrics",
        action="store_true",
        help="Write expanded metrics parquet while running tournaments",
    )
    run_parser.add_argument(
        "--row-dir",
        type=Path,
        help="Write full per-game rows to this directory",
    )

    sub.add_parser("time", help="Benchmark simulation throughput", add_help=False)

    watch_parser = sub.add_parser("watch", help="Interactively watch a game")
    watch_parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic play")

    analyze_parser = sub.add_parser("analyze", help="Data analysis helpers")
    analyze_sub = analyze_parser.add_subparsers(dest="an_cmd", required=True)
    analyze_sub.add_parser("ingest", help="Ingest raw data")
    analyze_sub.add_parser("curate", help="Curate ingested data")
    analyze_sub.add_parser("combine", help="Combine curated data into a superset parquet")
    analyze_sub.add_parser("metrics", help="Compute metrics")
    analyze_sub.add_parser("pipeline", help="Run ingest+curate+combine+metrics sequentially")

    return parser


def _parse_level(level: str | int) -> int:
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return int(level)


def _build_config(config_path: Path | None, overrides: Sequence[str]) -> AppConfig:
    resolved = config_path or _default_config_path()
    cfg = load_app_config(resolved) if resolved else AppConfig()
    cfg = apply_dot_overrides(cfg, list(overrides))
    return cfg


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    setup_info_logging()
    root_logger = logging.getLogger()
    root_logger.setLevel(_parse_level(args.log_level))

    LOGGER.info(
        "CLI arguments parsed",
        extra={
            "stage": "cli",
            "command": args.command,
            "config_path": str(args.config) if args.config is not None else None,
            "overrides": list(args.overrides or []),
            "log_level": logging.getLevelName(root_logger.level),
            "remaining": list(remaining),
        },
    )

    if args.command == "run":
        cfg = _build_config(args.config, args.overrides)
        if args.metrics:
            cfg.sim.expanded_metrics = True
        if args.row_dir is not None:
            cfg.sim.row_dir = args.row_dir
        LOGGER.info(
            "Dispatching run command",
            extra={
                "stage": "cli",
                "command": "run",
                "seed": cfg.sim.seed,
                "n_players_list": list(cfg.sim.n_players_list),
                "expanded_metrics": cfg.sim.expanded_metrics,
            },
        )
        if len(cfg.sim.n_players_list) > 1:
            runner.run_multi(cfg)
        else:
            runner.run_single_n(cfg, cfg.sim.n_players_list[0])
        LOGGER.info("Run command completed", extra={"stage": "cli", "command": "run"})
    elif args.command == "time":
        LOGGER.info("Dispatching measure_sim_times", extra={"stage": "cli", "command": "time"})
        measure_sim_times()
    elif args.command == "watch":
        LOGGER.info(
            "Dispatching watch_game",
            extra={"stage": "cli", "command": "watch", "seed": args.seed},
        )
        watch_game(seed=args.seed)
    elif args.command == "analyze":
        cfg = _build_config(args.config, args.overrides)
        LOGGER.info(
            "Dispatching analysis command",
            extra={
                "stage": "cli",
                "command": f"analyze:{args.an_cmd}",
                "config_path": str(args.config) if args.config else str(_default_config_path()),
            },
        )
        if args.an_cmd == "ingest":
            ingest.run(cfg)
        elif args.an_cmd == "curate":
            curate.run(cfg)
        elif args.an_cmd == "combine":
            combine.run(cfg)
        elif args.an_cmd == "metrics":
            metrics.run(cfg)
        elif args.an_cmd == "pipeline":
            ingest.run(cfg)
            curate.run(cfg)
            combine.run(cfg)
            metrics.run(cfg)
        LOGGER.info(
            "Analysis command completed",
            extra={"stage": "cli", "command": f"analyze:{args.an_cmd}"},
        )
    else:  # pragma: no cover - argparse enforces valid choices
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
