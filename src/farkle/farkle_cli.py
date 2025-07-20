# src/farkle/cli.py
from __future__ import annotations

import argparse
from typing import Any, Mapping

import yaml

from farkle.farkle_io import simulate_many_games_stream
from farkle.simulation import generate_strategy_grid


def load_config(path: str) -> Mapping[str, Any]:
    """Load YAML configuration from *path*.

    Raises FileNotFoundError, yaml.YAMLError if the file cannot be read or parsed.
    """
    with open(path, encoding="utf-8") as fh:
        cfg: Mapping[str, Any] = yaml.safe_load(fh)

    missing = [k for k in ("strategy_grid", "sim") if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys: {', '.join(missing)}")

    return cfg


def main(argv: list[str] | None = None) -> None:
    """Console-script entry-point."""
    ap = argparse.ArgumentParser(prog="farkle")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a tournament from a YAML config")
    run.add_argument("config", help="Path to YAML configuration file")

    args = ap.parse_args(argv)

    if args.cmd == "run":
        cfg = load_config(args.config)
        strategies, _ = generate_strategy_grid(**cfg["strategy_grid"])
        simulate_many_games_stream(**cfg["sim"], strategies=strategies)

