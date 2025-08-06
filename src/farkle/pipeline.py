from __future__ import annotations

import argparse
import sys
from typing import Callable, Sequence

from tqdm import tqdm

from farkle import analytics, curate, ingest, metrics
from farkle.analysis_config import PipelineCfg


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for the analysis pipeline."""
    cfg, cli_ns, remaining = PipelineCfg.parse_cli(argv)

    parser = argparse.ArgumentParser(prog="farkle-analyse")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "curate", "metrics", "analytics", "all"):
        sub.add_parser(name)

    args = parser.parse_args(remaining)
    verbose = getattr(cli_ns, "verbose", False)

    if args.command == "ingest":
        try:
            ingest.run(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"ingest step failed: {e}", file=sys.stderr)
            return 1
    elif args.command == "curate":
        try:
            curate.run(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"curate step failed: {e}", file=sys.stderr)
            return 1
    elif args.command == "metrics":
        try:
            metrics.run(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"metrics step failed: {e}", file=sys.stderr)
            return 1
    elif args.command == "analytics":
        try:
            analytics.run_all(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"analytics step failed: {e}", file=sys.stderr)
            return 1
    elif args.command == "all":
        steps: list[tuple[str, Callable[[PipelineCfg], None]]] = [
            ("ingest", ingest.run),
            ("curate", curate.run),
            ("metrics", metrics.run),
            ("analytics", analytics.run_all),
        ]
        for _name, fn in tqdm(steps, desc="pipeline", disable=not verbose):
            try:
                fn(cfg)
            except Exception as e:  # noqa: BLE001
                print(f"{_name} step failed: {e}", file=sys.stderr)
                return 1
    else:  # pragma: no cover - argparse enforces valid choices
        parser.error(f"Unknown command {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
