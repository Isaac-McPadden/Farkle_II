# src/farkle/cli/main.py
"""Command line interface for the :mod:`farkle` package."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import yaml

from farkle.analysis import combine, curate, ingest, metrics
from farkle.config import AppConfig, apply_dot_overrides, load_app_config
from farkle.simulation import runner
from farkle.simulation.time_farkle import measure_sim_times
from farkle.simulation.watch_game import watch_game
from farkle.utils.logging import setup_info_logging
from farkle.utils.yaml_helpers import expand_dotted_keys

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


def normalize_cfg(raw: dict[str, Any], command: str) -> dict[str, Any]:  # noqa: ARG001
    """
    Accepts:
      - dotted keys (e.g. 'io.results_dir')
      - nested keys (e.g. {'io': {'results_dir': ...}})
      - flat legacy keys
    Produces:
      - flat keys needed by run_tournament(**cfg) for 'run'
      - keys matching PipelineCfg(**cfg) for 'analyze'
    """
    # 1) expand dotted -> nested
    nested = expand_dotted_keys(raw)

    # 2) lift nested into the flat names both sides expect
    out: dict[str, Any] = dict(nested)  # keep originals too (harmless for PipelineCfg)

    # --- IO / analysis roots ---
    io = nested.get("io", {})
    if isinstance(io, dict):
        if "results_dir" in io:
            out["results_dir"] = io["results_dir"]
        # prefer 'analysis_subdir' if provided in io (your YAML uses this)
        if "analysis_subdir" in io:
            out["analysis_subdir"] = io["analysis_subdir"]
        # fallback mapper in case someone used 'analysis_dir'
        if "analysis_dir" in io and "analysis_subdir" not in out:
            out["analysis_subdir"] = io["analysis_dir"]

    # --- analysis toggles & params (PipelineCfg) ---
    # These keys are read directly by PipelineCfg(**cfg)
    analysis = nested.get("analysis", {})
    if isinstance(analysis, dict):
        for k in ("run_trueskill", "run_head2head", "run_hgb", "n_jobs", "trueskill_beta"):
            if k in analysis:
                out[k] = analysis[k]

    # Optional sections (mapped to PipelineCfg fields where applicable)
    for sect, mapping in (
        ("ingest",   {"row_group_size": "row_group_size", "n_jobs": "n_jobs_ingest"}),
        ("combine",  {"max_players": "combine_max_players"}),  # only if consumed later
        ("metrics",  {"seat_range": "metrics_seat_range"}),
        ("trueskill",{"beta": "trueskill_beta"}),
        ("head2head",{}),
        ("hgb",      {"n_estimators": "hgb_max_iter"}),
    ):
        val = nested.get(sect, {})
        if isinstance(val, dict):
            for src, dst in mapping.items():
                if src in val:
                    out[dst] = val[src]

    # --- simulation (run_tournament) ---
    sim = nested.get("sim", {})
    if isinstance(sim, dict):
        # adjust if your run_tournament signature differs
        if "n_players" in sim:
            out["n_players"] = sim["n_players"]
        if "num_shuffles" in sim:
            # If you also track games_per_shuffle, multiply here.
            out["n_games"] = sim["num_shuffles"] * sim.get("games_per_shuffle", 1)
        if "seed" in sim:
            out["global_seed"] = sim["seed"]
        if "n_jobs" in sim:
            out["n_jobs"] = sim["n_jobs"]
        if "row_dir" in sim:
            out["row_output_directory"] = Path(sim["row_dir"]) if not isinstance(sim["row_dir"], Path) else sim["row_dir"]

    return out


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
    # accept dotted, nested, or flat YAML and normalize for run/analysis
    cfg = normalize_cfg(cfg, args.command)

    LOGGER.info(
        "Configuration loaded",
        extra={
            "stage": "cli",
            "command": args.command,
            "config_keys": sorted(cfg.keys()),
        },
    )

    if args.command == "run":
        cfg = load_app_config(args.config) if args.config is not None else AppConfig()
        cfg = apply_dot_overrides(cfg, args.overrides)
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
                "n_players_list": cfg.sim.n_players_list,
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
        cfg = load_app_config(args.config) if args.config is not None else AppConfig()
        cfg = apply_dot_overrides(cfg, args.overrides)
        LOGGER.info(
            "Dispatching analysis command",
            extra={
                "stage": "cli",
                "command": f"analyze:{args.an_cmd}",
                "config_path": str(args.config) if args.config else None,
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
        LOGGER.info("Analysis command completed", extra={"stage": "cli", "command": f"analyze:{args.an_cmd}"})
    else:  # pragma: no cover - argparse enforces valid choices
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
