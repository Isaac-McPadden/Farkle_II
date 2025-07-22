# src/farkle/cli.py
"""CLI entry point for the Farkle package.

This module exposes the command line interface.  At present the only
available subcommand is ``run``.  It reads a YAML configuration file and
invokes :func:`simulate_many_games_stream` to execute simulations.
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping

import yaml

from farkle.farkle_io import simulate_many_games_stream
from farkle.simulation import generate_strategy_grid


def load_config(path: str) -> Mapping[str, Any]:
    """Load YAML configuration from *path*.

    Raises FileNotFoundError, yaml.YAMLError if the file cannot be read or
    parsed.
    """
    with open(path, encoding="utf-8") as fh:
        cfg: Mapping[str, Any] = yaml.safe_load(fh)

    missing = [k for k in ("strategy_grid", "sim") if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys: {', '.join(missing)}")

    return cfg


def main(argv: list[str] | None = None) -> None:
    """
    Console-script entry-point for farkle CLI.
    Passing *argv* lets unit-tests inject fake arguments.

    Inputs
    ------
    argv : list[str] | None
        Optional command line arguments. ``None`` uses ``sys.argv``.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If ``config`` cannot be opened.
    KeyError
        If required sections are missing from the YAML file.
    yaml.YAMLError
        If the file cannot be parsed.

    The configuration YAML must contain ``strategy_grid`` and ``sim`` keys. A
    minimal example looks like::

        strategy_grid:
          dice_threshold: [1, 2, 3]
        sim:
          n_games: 1000
    """
    parser = argparse.ArgumentParser(prog="farkle")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run a tournament from a YAML config"
    )
    run_parser.add_argument("config", help="Path to YAML configuration file")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        cfg = load_config(args.config)
        strategies, _ = generate_strategy_grid(**cfg["strategy_grid"])
        simulate_many_games_stream(**cfg["sim"], strategies=strategies)
