"""Command line interface for the :mod:`farkle` package."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import yaml

from farkle.analysis import combine, curate, ingest, metrics
from farkle.simulation.run_tournament import run_tournament
from farkle.simulation.time_farkle import measure_sim_times
from farkle.simulation.watch_game import watch_game
from farkle.utils.logging import setup_info_logging

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _apply_override(cfg: dict[str, Any], expr: str) -> None:
    """Apply a ``key=value`` override to *cfg* (nested keys via dots)."""
    key, value = expr.split("=", 1)
    target = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = yaml.safe_load(value)


def load_config(path: str | Path | None, overrides: Sequence[str] | None = None) -> dict[str, Any]:
    """Load a YAML configuration file and apply overrides."""
    cfg: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise TypeError("Config root must be a mapping")
            cfg.update(loaded)
    for expr in overrides or []:
        _apply_override(cfg, expr)
    return cfg


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
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
        help="Write full per-game rows to this directory",
    )

    # time (delegates to measure_sim_times which parses its own args)
    sub.add_parser("time", help="Benchmark simulation throughput", add_help=False)

    # watch
    watch_parser = sub.add_parser("watch", help="Interactively watch a game")
    watch_parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic play")

    # analyze
    analyze_parser = sub.add_parser("analyze", help="Data analysis helpers")
    analyze_sub = analyze_parser.add_subparsers(dest="an_cmd", required=True)
    analyze_sub.add_parser("ingest", help="Ingest raw CSV data")
    analyze_sub.add_parser("curate", help="Curate ingested data")
    analyze_sub.add_parser("combine", help="Combine curated data into a superset parquet")
    analyze_sub.add_parser("metrics", help="Compute metrics")
    analyze_sub.add_parser("pipeline", help="Run ingest→curate→combine→metrics pipeline")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_level(level: str | int) -> int:
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return int(level)


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
        },
    )

    cfg = load_config(args.config, args.overrides)

    LOGGER.info(
        "Configuration loaded",
        extra={
            "stage": "cli",
            "command": args.command,
            "config_keys": sorted(cfg.keys()),
        },
    )

    if args.command == "run":
        if args.metrics:
            cfg["collect_metrics"] = True
        row_dir = args.row_dir
        if row_dir is not None:
            cfg["row_output_directory"] = row_dir
        elif "row_output_directory" in cfg and isinstance(
            cfg["row_output_directory"], str
        ):
            cfg["row_output_directory"] = Path(cfg["row_output_directory"])
        LOGGER.info(
            "Dispatching run_tournament",
            extra={
                "stage": "cli",
                "command": "run",
                "global_seed": cfg.get("global_seed"),
                "n_jobs": cfg.get("n_jobs"),
                "row_output_directory": (
                    str(cfg.get("row_output_directory"))
                    if cfg.get("row_output_directory") is not None
                    else None
                ),
                "collect_metrics": bool(cfg.get("collect_metrics", False)),
                "remaining_args": remaining,
            },
        )
        run_tournament(**cfg)
        LOGGER.info(
            "run_tournament completed",
            extra={"stage": "cli", "command": "run"},
        )
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
        from farkle.analysis.analysis_config import PipelineCfg

        pipeline_cfg = PipelineCfg(**cfg)
        LOGGER.info(
            "Dispatching analysis command",
            extra={
                "stage": "cli",
                "command": f"analyze:{args.an_cmd}",
                "config_path": str(args.config) if args.config is not None else None,
            },
        )
        if args.an_cmd == "ingest":
            ingest.run(pipeline_cfg)
        elif args.an_cmd == "curate":
            curate.run(pipeline_cfg)
        elif args.an_cmd == "combine":
            combine.run(pipeline_cfg)
        elif args.an_cmd == "metrics":
            metrics.run(pipeline_cfg)
        elif args.an_cmd == "pipeline":
            ingest.run(pipeline_cfg)
            curate.run(pipeline_cfg)
            combine.run(pipeline_cfg)
            metrics.run(pipeline_cfg)
        LOGGER.info(
            "Analysis command completed",
            extra={"stage": "cli", "command": f"analyze:{args.an_cmd}"},
        )
    else:  # pragma: no cover - argparse enforces valid choices
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
