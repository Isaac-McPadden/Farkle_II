"""Sequential two-seed simulation orchestrator."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from farkle.config import AppConfig, apply_dot_overrides, load_app_config
from farkle.orchestration.seed_utils import (
    base_results_dir,
    prepare_seed_config,
    seed_has_completion_markers,
    write_active_config,
)
from farkle.simulation import runner
from farkle.utils.logging import setup_info_logging

LOGGER = logging.getLogger(__name__)


def _resolve_seed_pair(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[int, int] | None:
    if args.seed_pair and (args.seed_a is not None or args.seed_b is not None):
        parser.error("Use --seed-pair or --seed-a/--seed-b, not both.")
    if (args.seed_a is None) ^ (args.seed_b is None):
        parser.error("--seed-a and --seed-b must be provided together.")
    if args.seed_pair:
        return (int(args.seed_pair[0]), int(args.seed_pair[1]))
    if args.seed_a is not None and args.seed_b is not None:
        return (int(args.seed_a), int(args.seed_b))
    return None


def run_seeds(
    cfg: AppConfig,
    *,
    seed_pair: tuple[int, int],
    force: bool = False,
) -> None:
    base_dir = base_results_dir(cfg)
    for seed in seed_pair:
        seed_cfg = prepare_seed_config(cfg, seed=seed, base_results_dir=base_dir)
        LOGGER.info(
            "Preparing seed run",
            extra={
                "stage": "orchestration",
                "seed": seed,
                "results_dir": str(seed_cfg.io.results_dir),
            },
        )
        if not force and seed_has_completion_markers(seed_cfg):
            LOGGER.info(
                "Skipping seed run (completion markers found)",
                extra={
                    "stage": "orchestration",
                    "seed": seed,
                    "results_dir": str(seed_cfg.io.results_dir),
                },
            )
            continue
        write_active_config(seed_cfg)
        runner.run_tournament(seed_cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="farkle-two-seed")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/fast_config.yaml"), help="Path to YAML config"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values",
    )
    parser.add_argument(
        "--seed-a",
        type=int,
        help="Override the first seed for dual-seed orchestration",
    )
    parser.add_argument(
        "--seed-b",
        type=int,
        help="Override the second seed for dual-seed orchestration",
    )
    parser.add_argument(
        "--seed-pair",
        type=int,
        nargs=2,
        metavar=("A", "B"),
        help="Override the dual-seed tuple (A B)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when completion markers exist",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_info_logging()

    cfg = load_app_config(Path(args.config))
    cfg = apply_dot_overrides(cfg, list(args.overrides or []))

    seed_pair = _resolve_seed_pair(args, parser)
    if seed_pair is None:
        seed_pair = cfg.sim.require_seed_pair()

    run_seeds(cfg, seed_pair=seed_pair, force=args.force)
    return 0


__all__ = ["main", "run_seeds"]
