"""Sequential two-seed simulation orchestrator."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Any, Sequence

import yaml  # type: ignore[import-untyped]

from farkle.config import AppConfig, apply_dot_overrides, load_app_config
from farkle.simulation import runner
from farkle.utils.logging import setup_info_logging
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


def _base_results_dir(cfg: AppConfig) -> Path:
    if not cfg.io.append_seed:
        return cfg.io.results_dir
    suffix = f"_seed_{cfg.sim.seed}"
    path_str = str(cfg.io.results_dir)
    if path_str.endswith(suffix):
        return Path(path_str[: -len(suffix)])
    return cfg.io.results_dir


def _resolve_results_dir(base: Path, seed: int, *, append_seed: bool) -> Path:
    if not append_seed:
        return base
    return Path(f"{base}_seed_{seed}")


def _seed_has_completion_markers(cfg: AppConfig) -> bool:
    for n in cfg.sim.n_players_list:
        n_dir = cfg.n_dir(n)
        row_dir = runner._resolve_row_output_dir(cfg, n)
        metric_chunk_dir = runner._resolve_metric_chunk_dir(cfg, n)
        ckpt_path = cfg.checkpoint_path(n)
        if not runner._has_existing_outputs(
            n_dir=n_dir,
            n_players=n,
            ckpt_path=ckpt_path,
            row_dir=row_dir,
            metric_chunk_dir=metric_chunk_dir,
        ):
            return False
    return True


def _stringify_paths(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_paths(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_stringify_paths(v) for v in obj)
    return obj


def _write_active_config(cfg: AppConfig) -> None:
    dest_dir = cfg.io.results_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved_dict = _stringify_paths(dataclasses.asdict(cfg))
    resolved_yaml = yaml.safe_dump(resolved_dict, sort_keys=True)
    target = dest_dir / "active_config.yaml"
    with atomic_path(str(target)) as tmp_path:
        Path(tmp_path).write_text(resolved_yaml, encoding="utf-8")


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


def _prepare_seed_config(
    base_cfg: AppConfig, *, seed: int, base_results_dir: Path
) -> AppConfig:
    io_cfg = dataclasses.replace(
        base_cfg.io,
        results_dir=_resolve_results_dir(base_results_dir, seed, append_seed=base_cfg.io.append_seed),
    )
    sim_cfg = dataclasses.replace(base_cfg.sim, seed=seed)
    return dataclasses.replace(base_cfg, io=io_cfg, sim=sim_cfg)


def run_seeds(
    cfg: AppConfig,
    *,
    seed_pair: tuple[int, int],
    force: bool = False,
) -> None:
    base_results_dir = _base_results_dir(cfg)
    for seed in seed_pair:
        seed_cfg = _prepare_seed_config(cfg, seed=seed, base_results_dir=base_results_dir)
        LOGGER.info(
            "Preparing seed run",
            extra={
                "stage": "orchestration",
                "seed": seed,
                "results_dir": str(seed_cfg.io.results_dir),
                "append_seed": seed_cfg.io.append_seed,
            },
        )
        if not force and _seed_has_completion_markers(seed_cfg):
            LOGGER.info(
                "Skipping seed run (completion markers found)",
                extra={
                    "stage": "orchestration",
                    "seed": seed,
                    "results_dir": str(seed_cfg.io.results_dir),
                },
            )
            continue
        _write_active_config(seed_cfg)
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
