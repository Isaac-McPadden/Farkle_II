# src/farkle/analysis/pipeline.py
"""CLI pipeline orchestrator for ingest, curation, and analysis.

Defines sequential stages for loading simulation outputs, combining metrics,
and generating reports, mirroring the repository's documented workflow.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
from pathlib import Path
from typing import Callable, Sequence

import yaml  # type: ignore[import-untyped]
from tqdm import tqdm

from farkle import analysis
from farkle.analysis import combine, curate, game_stats, ingest, metrics, rng_diagnostics
from farkle.config import AppConfig, load_app_config
from farkle.utils.manifest import append_manifest_line
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


def _stringify_paths(obj: object) -> object:
    """Convert Paths nested inside mappings/sequences into plain strings."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_paths(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_stringify_paths(v) for v in obj)
    return obj


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for the analysis pipeline."""

    parser = argparse.ArgumentParser(prog="farkle-analyze")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/fast_config.yaml"), help="Path to YAML config"
    )
    parser.add_argument(
        "--compute-game-stats",
        action="store_true",
        help="Run the game-length/margin statistics step after combine/metrics",
    )
    parser.add_argument(
        "--rng-diagnostics",
        action="store_true",
        help="Run RNG diagnostics over curated rows after combine",
    )
    parser.add_argument(
        "--margin-thresholds",
        type=int,
        nargs="+",
        help="Override victory-margin thresholds used by game-stats outputs",
    )
    parser.add_argument(
        "--rare-event-target",
        type=int,
        help="Override the target score used to flag rare-event summaries",
    )
    parser.add_argument(
        "--rng-lags",
        dest="rng_lags",
        type=int,
        action="append",
        help="Optional lags for RNG diagnostics (repeatable; defaults to 1)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "curate", "combine", "metrics", "analytics", "all"):
        sub.add_parser(name)

    parse = getattr(parser, "parse_intermixed_args", None)
    try:
        args = parse(argv) if parse else parser.parse_args(argv)
    except TypeError:
        args = parser.parse_args(argv)

    LOGGER.info(
        "Analysis pipeline start",
        extra={"stage": "pipeline", "command": args.command, "config": str(args.config)},
    )
    app_cfg = load_app_config(Path(args.config))
    if args.margin_thresholds:
        app_cfg.analysis.game_stats_margin_thresholds = tuple(args.margin_thresholds)
    if args.rare_event_target is not None:
        app_cfg.analysis.rare_event_target_score = int(args.rare_event_target)
    rng_lags = tuple(int(lag) for lag in args.rng_lags) if args.rng_lags else None

    analysis_dir = app_cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for stage_dir in (
        app_cfg.ingest_stage_dir,
        app_cfg.combine_stage_dir,
        app_cfg.metrics_stage_dir,
        app_cfg.trueskill_stage_dir,
        app_cfg.head2head_stage_dir,
        app_cfg.tiering_stage_dir,
    ):
        stage_dir.mkdir(parents=True, exist_ok=True)
    resolved = analysis_dir / "config.resolved.yaml"
    # Best-effort: write out the resolved (merged) config we actually used
    resolved_dict = _stringify_paths(dataclasses.asdict(app_cfg))
    resolved_yaml = yaml.safe_dump(resolved_dict, sort_keys=True)
    with atomic_path(str(resolved)) as tmp_path:
        Path(tmp_path).write_text(resolved_yaml)

    # NDJSON manifest (append-only)
    manifest_path = analysis_dir / app_cfg.manifest_name
    config_sha = hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()
    append_manifest_line(
        manifest_path,
        {
            "event": "run_start",
            "command": args.command,
            "config_sha": config_sha,
            "resolved_config": str(resolved),
            "results_dir": str(app_cfg.results_dir),
            "analysis_dir": str(analysis_dir),
        },
    )

    # Build the step plan
    if args.command == "all":
        steps: list[tuple[str, Callable[[AppConfig], None]]] = [
            ("ingest", ingest.run),
            ("curate", curate.run),
            ("combine", combine.run),
            ("metrics", metrics.run),
            ("analytics", analysis.run_all),
        ]
    else:
        name_map: dict[str, Callable[[AppConfig], None]] = {
            "ingest": ingest.run,
            "curate": curate.run,
            "combine": combine.run,
            "metrics": metrics.run,
            "analytics": analysis.run_all,
        }
        if args.command not in name_map:  # pragma: no cover - argparse enforces valid choices
            parser.error(f"Unknown command {args.command}")
        steps = [(args.command, name_map[args.command])]

    def _insert_after(target: str, new: tuple[str, Callable[[AppConfig], None]]) -> None:
        for idx, (name, _fn) in enumerate(list(steps)):
            if name == target:
                steps.insert(idx + 1, new)
                return
        steps.append(new)

    if args.compute_game_stats:
        _insert_after("metrics", ("game_stats", lambda cfg: game_stats.run(cfg)))
    if args.rng_diagnostics:
        _insert_after(
            "combine", ("rng_diagnostics", lambda cfg: rng_diagnostics.run(cfg, lags=rng_lags))
        )
    # Execute with per-step manifest events
    iterator = tqdm(steps, desc="pipeline") if len(steps) > 1 else steps
    for _name, fn in iterator:
        LOGGER.info("Pipeline step", extra={"stage": "pipeline", "step": _name})
        append_manifest_line(manifest_path, {"event": "step_start", "step": _name})
        try:
            fn(app_cfg)
            append_manifest_line(manifest_path, {"event": "step_end", "step": _name, "ok": True})
        except Exception as e:  # noqa: BLE001
            append_manifest_line(
                manifest_path,
                {
                    "event": "step_end",
                    "step": _name,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                },
            )
            raise

    append_manifest_line(manifest_path, {"event": "run_end", "config_sha": config_sha})
    LOGGER.info("Analysis pipeline complete", extra={"stage": "pipeline"})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
