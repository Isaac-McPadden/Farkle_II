# src/farkle/cli.py
import argparse
from typing import Any, Mapping

import yaml
from farkle_io import simulate_many_games_stream
from simulation import generate_strategy_grid


def main() -> None:
    ap = argparse.ArgumentParser(prog="farkle")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run a tournament from a YAML config")
    run.add_argument("config")
    args = ap.parse_args()

    if args.cmd == "run":
        with open(args.config) as fh:
            cfg: Mapping[str, Any] = yaml.safe_load(fh)
        strategies, _ = generate_strategy_grid(**cfg["strategy_grid"])
        simulate_many_games_stream(**cfg["sim"], strategies=strategies)

if __name__ == "__main__":
    main()
