from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

import yaml
from tqdm import tqdm

from farkle import analysis
from farkle.analysis import combine, curate, ingest, metrics
from farkle.config import AppConfig, apply_dot_overrides, load_app_config
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


def _default_config_path() -> Path | None:
    candidate = Path("configs/farkle_mega_config.yaml")
    if candidate.exists():
        return candidate
    fallback = Path("farkle_mega_config.yaml")
    if fallback.exists():
        return fallback
    return None


def _build_config(path: Path | None, overrides: Sequence[str]) -> AppConfig:
    resolved = path or _default_config_path()
    cfg = load_app_config(resolved) if resolved else AppConfig()
    cfg = apply_dot_overrides(cfg, list(overrides))
    return cfg


def _write_config_snapshot(cfg: AppConfig) -> None:
    analysis_dir = cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    resolved_path = analysis_dir / "config.resolved.yaml"
    payload = json.loads(json.dumps(asdict(cfg), default=str))
    with atomic_path(str(resolved_path)) as tmp_path:
        Path(tmp_path).write_text(yaml.safe_dump(payload, sort_keys=True))

    manifest_path = analysis_dir / cfg.analysis.manifest_name
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:  # pragma: no cover - corrupt manifest
            manifest = {}
    manifest["config_sha"] = cfg.config_sha
    with atomic_path(str(manifest_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="farkle-analyze")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config")
    parser.add_argument(
        "--overrides",
        "-O",
        action="append",
        default=[],
        metavar="section.option=value",
        help="Apply dotted overrides after loading config",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "curate", "combine", "metrics", "analytics", "all"):
        sub.add_parser(name)

    args = parser.parse_args(argv)

    LOGGER.info(
        "Analysis pipeline start",
        extra={
            "stage": "pipeline",
            "command": args.command,
            "config": str(args.config) if args.config else str(_default_config_path()),
        },
    )

    cfg = _build_config(args.config, args.overrides)
    _write_config_snapshot(cfg)

    commands: dict[str, Callable[[AppConfig], None]] = {
        "ingest": ingest.run,
        "curate": curate.run,
        "combine": combine.run,
        "metrics": metrics.run,
        "analytics": analysis.run_all,
    }

    if args.command in commands:
        commands[args.command](cfg)
    elif args.command == "all":
        steps: list[tuple[str, Callable[[AppConfig], None]]] = [
            ("ingest", ingest.run),
            ("curate", curate.run),
            ("combine", combine.run),
            ("metrics", metrics.run),
            ("analytics", analysis.run_all),
        ]
        for name, func in tqdm(steps, desc="pipeline"):
            LOGGER.info("Pipeline step", extra={"stage": "pipeline", "step": name})
            func(cfg)
    else:  # pragma: no cover - argparse enforces valid choices
        parser.error(f"Unknown command {args.command}")

    LOGGER.info("Analysis pipeline complete", extra={"stage": "pipeline"})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
