# src/farkle/cli.py
from __future__ import annotations

import argparse
from typing import Any, Mapping

import yaml

from farkle.farkle_io import simulate_many_games_stream
from farkle.simulation import generate_strategy_grid


def main(argv: list[str] | None = None) -> None:
    """
    Console-script entry-point.
    Passing *argv* lets unit-tests inject fake arguments.
    
    Inputs
    ------
    argv : list[str] | None
        Command line argument list, or None for sys.argv.

    Returns
    -------
    None
    """
    ap = argparse.ArgumentParser(prog="farkle")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a tournament from a YAML config")
    run.add_argument("config", help="Path to YAML configuration file")

    args = ap.parse_args(argv)

    if args.cmd == "run":
        with open(args.config, encoding="utf-8") as fh:
            cfg: Mapping[str, Any] = yaml.safe_load(fh)

        strategies, _ = generate_strategy_grid(**cfg["strategy_grid"])
        simulate_many_games_stream(**cfg["sim"], strategies=strategies)