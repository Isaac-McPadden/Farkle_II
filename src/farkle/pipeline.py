from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import yaml
from tqdm import tqdm

from farkle import aggregate, analytics, curate, ingest, metrics
from farkle.analysis_config import load_config

if TYPE_CHECKING:  # for type checkers without creating runtime deps
    from farkle.analysis_config import PipelineCfg


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for the analysis pipeline."""

    parser = argparse.ArgumentParser(prog="farkle-analyze")
    parser.add_argument(
        "--config", type=Path, default=Path("analysis_config.yaml"), help="Path to YAML config"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "curate", "aggregate", "metrics", "analytics", "all"):
        sub.add_parser(name)

    args = parser.parse_args(argv)

    cfg, cfg_sha = load_config(Path(args.config))
    pipeline_cfg = cfg.to_pipeline_cfg()

    analysis_dir = pipeline_cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    resolved = analysis_dir / "config.resolved.yaml"
    resolved.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=True))

    manifest_path = analysis_dir / pipeline_cfg.manifest_name
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:  # noqa: BLE001
            manifest = {}
    manifest["config_sha"] = cfg_sha
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "ingest":
        ingest.run(pipeline_cfg)
    elif args.command == "curate":
        curate.run(pipeline_cfg)
    elif args.command == "aggregate":
        aggregate.run(pipeline_cfg)
    elif args.command == "metrics":
        metrics.run(pipeline_cfg)
    elif args.command == "analytics":
        analytics.run_all(pipeline_cfg)
    elif args.command == "all":
        # Each step consumes a PipelineCfg; be precise for type-checkers
        steps: list[tuple[str, Callable[["PipelineCfg"], None]]] = [
            ("ingest", ingest.run),
            ("curate", curate.run),
            ("aggregate", aggregate.run),
            ("metrics", metrics.run),
            ("analytics", analytics.run_all),
        ]
        for _name, fn in tqdm(steps, desc="pipeline"):
            fn(pipeline_cfg)
    else:  # pragma: no cover - argparse enforces valid choices
        parser.error(f"Unknown command {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
