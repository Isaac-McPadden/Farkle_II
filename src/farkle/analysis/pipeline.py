from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import yaml
from tqdm import tqdm

from farkle import analysis
from farkle.analysis import combine, curate, ingest, metrics
from farkle.analysis.analysis_config import load_config
from farkle.app_config import AppConfig
from farkle.utils.writer import atomic_path

if TYPE_CHECKING:  # for type checkers without creating runtime deps
    from farkle.analysis.analysis_config import PipelineCfg  # noqa: F401


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for the analysis pipeline."""

    parser = argparse.ArgumentParser(prog="farkle-analyze")
    parser.add_argument(
        "--config", type=Path, default=Path("analysis_config.yaml"), help="Path to YAML config"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "curate", "combine", "metrics", "analytics", "all"):
        sub.add_parser(name)

    args = parser.parse_args(argv)

    cfg, cfg_sha = load_config(Path(args.config))
    app_cfg = AppConfig(analysis=cfg.to_pipeline_cfg())

    analysis_dir = app_cfg.analysis.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    resolved = analysis_dir / "config.resolved.yaml"
    with atomic_path(str(resolved)) as tmp_path:
        Path(tmp_path).write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=True))

    manifest_path = analysis_dir / app_cfg.analysis.manifest_name
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:  # noqa: BLE001
            manifest = {}
    manifest["config_sha"] = cfg_sha
    with atomic_path(str(manifest_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(manifest, indent=2))

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "ingest":
        ingest.run(app_cfg)
    elif args.command == "curate":
        curate.run(app_cfg)
    elif args.command == "combine":
        combine.run(app_cfg)
    elif args.command == "metrics":
        metrics.run(app_cfg)
    elif args.command == "analytics":
        analysis.run_all(app_cfg)
    elif args.command == "all":
        steps: list[tuple[str, Callable[[AppConfig], None]]] = [
            ("ingest", ingest.run),
            ("curate", curate.run),
            ("combine", combine.run),
            ("metrics", metrics.run),
            ("analytics", analysis.run_all),
        ]
        for _name, fn in tqdm(steps, desc="pipeline"):
            fn(app_cfg)
    else:  # pragma: no cover - argparse enforces valid choices
        parser.error(f"Unknown command {args.command}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
