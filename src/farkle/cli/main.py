"""Command line interface for the :mod:`farkle` package."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml

from farkle.analysis import curate, ingest, metrics
from farkle.simulation.run_tournament import run_tournament
from farkle.simulation.time_farkle import measure_sim_times
from farkle.simulation.watch_game import watch_game
from farkle.utils.logging import configure_logging

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
    analyze_sub.add_parser("metrics", help="Compute metrics")
    analyze_sub.add_parser("pipeline", help="Run ingest→curate→metrics pipeline")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    configure_logging(level=args.log_level)

    cfg = load_config(args.config, args.overrides)

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
        run_tournament(**cfg)
    elif args.command == "time":
        measure_sim_times()
    elif args.command == "watch":
        watch_game(seed=args.seed)
    elif args.command == "analyze":
        from farkle.analysis.analysis_config import PipelineCfg

        pipeline_cfg = PipelineCfg(**cfg)
        if args.an_cmd == "ingest":
            ingest.run(pipeline_cfg)
        elif args.an_cmd == "curate":
            curate.run(pipeline_cfg)
        elif args.an_cmd == "metrics":
            metrics.run(pipeline_cfg)
        elif args.an_cmd == "pipeline":
            ingest.run(pipeline_cfg)
            curate.run(pipeline_cfg)
            metrics.run(pipeline_cfg)
    else:  # pragma: no cover - argparse enforces valid choices
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
