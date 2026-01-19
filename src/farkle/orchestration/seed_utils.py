"""Shared helpers for seed-based orchestration."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from farkle.config import AppConfig
from farkle.simulation import runner
from farkle.utils.writer import atomic_path


def base_results_dir(cfg: AppConfig) -> Path:
    """Return the non-seed-suffixed results directory."""
    suffix = f"_seed_{cfg.sim.seed}"
    path_str = str(cfg.io.results_dir)
    if path_str.endswith(suffix):
        return Path(path_str[: -len(suffix)])
    return cfg.io.results_dir


def resolve_results_dir(base: Path, seed: int) -> Path:
    """Resolve the results dir for a given seed."""
    return Path(f"{base}_seed_{seed}")


def seed_has_completion_markers(cfg: AppConfig) -> bool:
    """Return True when all simulation outputs are present for this seed."""
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


def write_active_config(cfg: AppConfig, dest_dir: Path | None = None) -> None:
    """Persist the resolved configuration alongside results."""
    target_dir = dest_dir or cfg.io.results_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_dict = _stringify_paths(dataclasses.asdict(cfg))
    resolved_yaml = yaml.safe_dump(resolved_dict, sort_keys=True)
    target = target_dir / "active_config.yaml"
    with atomic_path(str(target)) as tmp_path:
        Path(tmp_path).write_text(resolved_yaml, encoding="utf-8")


def prepare_seed_config(
    base_cfg: AppConfig,
    *,
    seed: int,
    base_results_dir: Path,
    meta_analysis_dir: Path | None = None,
) -> AppConfig:
    """Return a config updated for a specific seed and results directory."""
    io_cfg = dataclasses.replace(
        base_cfg.io,
        results_dir=resolve_results_dir(base_results_dir, seed),
        meta_analysis_dir=(
            meta_analysis_dir if meta_analysis_dir is not None else base_cfg.io.meta_analysis_dir
        ),
    )
    sim_cfg = dataclasses.replace(base_cfg.sim, seed=seed)
    return dataclasses.replace(base_cfg, io=io_cfg, sim=sim_cfg)


__all__ = [
    "base_results_dir",
    "prepare_seed_config",
    "resolve_results_dir",
    "seed_has_completion_markers",
    "write_active_config",
]
