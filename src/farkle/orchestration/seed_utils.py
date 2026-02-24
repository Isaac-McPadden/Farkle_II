"""Shared helpers for seed-based orchestration."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from farkle.config import AppConfig, effective_config_dict
from farkle.simulation import runner
from farkle.utils.writer import atomic_path


def base_results_dir(cfg: AppConfig) -> Path:
    """Return the non-seed-suffixed results directory."""
    prefix = Path(cfg.io.results_dir_prefix)
    if not prefix.is_absolute():
        prefix = Path("data") / prefix
    match = re.match(r"^(?P<base>.+)_seed_\d+$", prefix.name)
    if match:
        prefix = prefix.with_name(match.group("base"))
    return prefix


def resolve_results_dir(base: Path, seed: int) -> Path:
    """Resolve the results dir for a given seed."""
    return Path(f"{base}_seed_{seed}")


def seed_pair_root(cfg: AppConfig, seed_pair: tuple[int, int]) -> Path:
    """Resolve the root directory for a seed pair run."""
    base_dir = base_results_dir(cfg)
    seed_a, seed_b = seed_pair
    return base_dir.with_name(f"{base_dir.name}_seed_pair_{seed_a}_{seed_b}")


def seed_pair_seed_root(cfg: AppConfig, seed_pair: tuple[int, int], seed: int) -> Path:
    """Resolve the per-seed results root under a seed-pair directory."""
    pair_root = seed_pair_root(cfg, seed_pair)
    base_name = base_results_dir(cfg).name
    return pair_root / f"{base_name}_seed_{seed}"


def seed_pair_meta_root(cfg: AppConfig, seed_pair: tuple[int, int]) -> Path | None:
    """Resolve the shared meta-analysis root for a seed pair."""
    if cfg.io.meta_analysis_dir is None:
        return None
    meta_path = Path(cfg.io.meta_analysis_dir)
    meta_name = meta_path.name if meta_path.is_absolute() else str(meta_path)
    seed_a, seed_b = seed_pair
    return seed_pair_root(cfg, seed_pair) / f"{meta_name}_{seed_a}_{seed_b}"


def split_seeded_results_dir(path: Path) -> tuple[Path, int | None]:
    """Split a seeded results directory into base path and seed (if present)."""
    match = re.match(r"^(?P<base>.+)_seed_(?P<seed>\d+)$", path.name)
    if match:
        return path.with_name(match.group("base")), int(match.group("seed"))
    return path, None


def seed_has_completion_markers(cfg: AppConfig) -> bool:
    """Return True when all simulation outputs are present for this seed."""
    return all(runner.simulation_is_complete(cfg, n) for n in cfg.sim.n_players_list)


def write_active_config(cfg: AppConfig, dest_dir: Path | None = None) -> None:
    """Persist the resolved configuration alongside results."""
    target_dir = dest_dir or cfg.results_root
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_dict = effective_config_dict(cfg)
    resolved_yaml = yaml.safe_dump(resolved_dict, sort_keys=True)
    target = target_dir / "active_config.yaml"
    with atomic_path(str(target)) as tmp_path:
        Path(tmp_path).write_text(resolved_yaml, encoding="utf-8")

    done_payload = {"active_config": str(target), "config_sha": cfg.config_sha}
    done_path = target.with_suffix(".done.json")
    with atomic_path(str(done_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(done_payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_seed_config(
    base_cfg: AppConfig,
    *,
    seed: int,
    base_results_dir: Path,
    meta_analysis_dir: Path | None = None,
) -> AppConfig:
    """Return a config updated for a specific seed and results directory."""
    base_dir = Path(base_results_dir)
    if not base_dir.is_absolute() and base_dir.parts and base_dir.parts[0] == "data":
        base_dir = Path(*base_dir.parts[1:])
    io_cfg = dataclasses.replace(
        base_cfg.io,
        results_dir_prefix=base_dir,
        meta_analysis_dir=(
            meta_analysis_dir if meta_analysis_dir is not None else base_cfg.io.meta_analysis_dir
        ),
    )
    sim_cfg = dataclasses.replace(base_cfg.sim, seed=seed)
    prepared = dataclasses.replace(base_cfg, io=io_cfg, sim=sim_cfg)
    prepared.config_sha = base_cfg.config_sha
    return prepared


__all__ = [
    "base_results_dir",
    "prepare_seed_config",
    "resolve_results_dir",
    "seed_pair_meta_root",
    "seed_pair_root",
    "seed_pair_seed_root",
    "seed_has_completion_markers",
    "split_seeded_results_dir",
    "write_active_config",
]
