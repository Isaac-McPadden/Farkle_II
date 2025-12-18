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
from typing import Any, Callable, Sequence, overload

import yaml  # type: ignore[import-untyped]
from tqdm import tqdm

from farkle import analysis
from farkle.analysis import combine, curate, game_stats, ingest, metrics, rng_diagnostics
from farkle.config import AppConfig, load_app_config
from farkle.utils.manifest import append_manifest_line
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


@overload
def _stringify_paths(obj: dict[str, Any]) -> dict[str, Any]:
    ...


@overload
def _stringify_paths(obj: list[Any]) -> list[Any]:
    ...


@overload
def _stringify_paths(obj: tuple[Any, ...]) -> tuple[Any, ...]:
    ...


@overload
def _stringify_paths(obj: Path) -> str:
    ...


@overload
def _stringify_paths(obj: Any) -> Any:
    ...


def _stringify_paths(obj: Any) -> Any:
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
        help="Run the 04_game_stats stage after metrics",
    )
    parser.add_argument(
        "--rng-diagnostics",
        action="store_true",
        help="Run the 05_rng diagnostics stage over curated rows",
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

    for stage in (
        "00_ingest",
        "01_curate",
        "02_combine",
        "03_metrics",
        "04_game_stats",
        "05_rng",
        "05_seed_summaries",
        "06_variance",
        "07_meta",
        "08_agreement",
        "09_trueskill",
        "10_head2head",
        "11_hgb",
        "12_tiering",
    ):
        app_cfg.stage_subdir(stage)
    analysis_dir = app_cfg.analysis_dir
    resolved = analysis_dir / "config.resolved.yaml"
    # Best-effort: write out the resolved (merged) config we actually used
    resolved_dict: dict[str, Any] = _stringify_paths(dataclasses.asdict(app_cfg))
    resolved_dict.pop("config_sha", None)
    resolved_yaml = yaml.safe_dump(resolved_dict, sort_keys=True)
    with atomic_path(str(resolved)) as tmp_path:
        Path(tmp_path).write_text(resolved_yaml)

    # NDJSON manifest (append-only)
    manifest_path = analysis_dir / app_cfg.manifest_name
    config_sha = hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()
    app_cfg.config_sha = config_sha  # allow downstream caching helpers to compare configs
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

    def _maybe_add_game_stats(
        plan: list[tuple[str, Callable[[AppConfig], None]]],
    ) -> None:
        if args.compute_game_stats:
            plan.append(("game_stats", lambda cfg: game_stats.run(cfg)))

    def _maybe_add_rng(plan: list[tuple[str, Callable[[AppConfig], None]]]) -> None:
        if args.rng_diagnostics:
            plan.append(("rng_diagnostics", lambda cfg: rng_diagnostics.run(cfg, lags=rng_lags)))

    # Build the step plan
    if args.command == "all":
        steps: list[tuple[str, Callable[[AppConfig], None]]] = [
            ("ingest", ingest.run),
            ("curate", curate.run),
            ("combine", combine.run),
            ("metrics", metrics.run),
        ]
        _maybe_add_game_stats(steps)
        _maybe_add_rng(steps)
        steps.append(("analytics", analysis.run_all))
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
        if args.command == "metrics":
            _maybe_add_game_stats(steps)
        if args.command in {"combine", "metrics"}:
            _maybe_add_rng(steps)
    # Execute with per-step manifest events
    iterator = tqdm(steps, desc="pipeline") if len(steps) > 1 else steps
    failed_steps: list[str] = []
    first_failure: Exception | None = None
    for _name, fn in iterator:
        LOGGER.info("Pipeline step", extra={"stage": "pipeline", "step": _name})
        append_manifest_line(manifest_path, {"event": "step_start", "step": _name})
        try:
            fn(app_cfg)
            append_manifest_line(manifest_path, {"event": "step_end", "step": _name, "ok": True})
        except Exception as e:  # noqa: BLE001
            failed_steps.append(_name)
            first_failure = first_failure or e
            LOGGER.exception(
                "Pipeline step failed", extra={"stage": "pipeline", "step": _name, "error": e}
            )
            append_manifest_line(
                manifest_path,
                {
                    "event": "step_end",
                    "step": _name,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                },
            )

    run_end_payload: dict[str, Any] = {"event": "run_end", "config_sha": config_sha}
    if failed_steps:
        run_end_payload["failed_steps"] = failed_steps
    append_manifest_line(manifest_path, run_end_payload)
    if failed_steps:
        LOGGER.error(
            "Analysis pipeline completed with failures: %s",
            failed_steps,
            extra={"stage": "pipeline", "failed_steps": failed_steps},
        )
        if first_failure is not None:
            raise first_failure
        return 1

    LOGGER.info("Analysis pipeline complete", extra={"stage": "pipeline"})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
