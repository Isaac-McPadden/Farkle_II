from __future__ import annotations

import argparse
import logging
import sys
from typing import Callable, Sequence

from tqdm import tqdm

from farkle import analytics, curate, ingest, metrics, aggregate
from farkle.analysis_config import PipelineCfg


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for the analysis pipeline."""
    cfg, cli_ns, remaining = PipelineCfg.parse_cli(argv)

    parser = argparse.ArgumentParser(prog="farkle-analyze")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "curate", "aggregate", "metrics", "analytics", "all"):
        sub.add_parser(name)

    args = parser.parse_args(remaining)
    verbose = getattr(cli_ns, "verbose", False)

    # Ensure DEBUG level is in effect before any sub-modules log
    effective_level = (
        logging.DEBUG if verbose else getattr(logging, cfg.log_level.upper(), logging.INFO)
    )
    log_kwargs = {
        "level": effective_level,
        "format": "%(message)s",
        "force": True,
    }
    if cfg.log_file is not None:
        log_kwargs["handlers"] = [
            logging.StreamHandler(),
            logging.FileHandler(cfg.log_file),
        ]
    logging.basicConfig(**log_kwargs)

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
    elif args.command == "aggregate":
        try:
            aggregate.run(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"aggregate step failed: {e}", file=sys.stderr)
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
            ("aggregate", aggregate.run),
            ("metrics", metrics.run),
            ("analytics", analytics.run_all),
        ]
        for _name, fn in tqdm(steps, desc="pipeline", disable=not verbose):
            try:
                fn(cfg)
            except Exception as e:  # noqa: BLE001
                # Propagate the failure so callers (and tests) can detect it.
                print(f"{_name} step failed: {e}", file=sys.stderr)
                raise
    else:  # pragma: no cover - argparse enforces valid choices
        parser.error(f"Unknown command {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
