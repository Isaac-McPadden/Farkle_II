"""Aggregate cross-seed game length and margin statistics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import norm, t

from farkle.analysis import stage_logger
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig
from farkle.orchestration.seed_utils import resolve_results_dir, split_seeded_results_dir
from farkle.utils.artifacts import write_parquet_atomic

LOGGER = logging.getLogger(__name__)

GAME_LENGTH_INPUTS = ("game_length_stats.parquet", "game_length.parquet")
MARGIN_INPUTS = ("margin_of_victory_stats.parquet", "margin_stats.parquet")
GAME_LENGTH_OUTPUT = "game_length_interseed.parquet"
MARGIN_OUTPUT = "margin_interseed.parquet"
NORMAL_975 = norm.ppf(0.975)
T_CRIT_N_SEEDS = 30


@dataclass(frozen=True)
class SeedInputs:
    seed: int
    analysis_dir: Path


def run(cfg: AppConfig, *, force: bool = False) -> None:
    """Aggregate per-seed game-length and margin stats into interseed summaries."""

    stage_log = stage_logger("interseed_game_stats", logger=LOGGER)
    stage_log.start()

    seeds = _seed_analysis_dirs(cfg)
    if not seeds:
        stage_log.missing_input("no seed analysis directories resolved")
        return

    game_length_paths = _seed_input_paths(
        seeds,
        cfg,
        candidates=GAME_LENGTH_INPUTS,
    )
    margin_paths = _seed_input_paths(
        seeds,
        cfg,
        candidates=MARGIN_INPUTS,
    )

    if len(game_length_paths) < 2 and len(margin_paths) < 2:
        stage_log.missing_input("fewer than two seeds with game stats")
        return

    output_dir = cfg.interseed_stage_dir
    game_length_output = output_dir / GAME_LENGTH_OUTPUT
    margin_output = output_dir / MARGIN_OUTPUT
    game_length_stamp = stage_done_path(output_dir, "interseed.game_length")
    margin_stamp = stage_done_path(output_dir, "interseed.margin")

    game_length_inputs = [path for _, path in game_length_paths]
    margin_inputs = [path for _, path in margin_paths]
    game_length_up_to_date = bool(game_length_inputs) and not force and stage_is_up_to_date(
        game_length_stamp,
        inputs=game_length_inputs,
        outputs=[game_length_output],
        config_sha=cfg.config_sha,
    )
    margin_up_to_date = bool(margin_inputs) and not force and stage_is_up_to_date(
        margin_stamp,
        inputs=margin_inputs,
        outputs=[margin_output],
        config_sha=cfg.config_sha,
    )

    if game_length_up_to_date and margin_up_to_date:
        LOGGER.info(
            "Interseed game stats up-to-date",
            extra={
                "stage": "interseed_game_stats",
                "game_length_output": str(game_length_output),
                "margin_output": str(margin_output),
            },
        )
        return

    if not game_length_up_to_date and game_length_inputs:
        game_length_frame = _load_seed_frames(game_length_paths)
        if game_length_frame.empty:
            stage_log.missing_input("no game-length stats available")
        else:
            aggregated = _aggregate_seed_stats(game_length_frame)
            table = pa.Table.from_pandas(aggregated, preserve_index=False)
            write_parquet_atomic(table, game_length_output, codec=cfg.parquet_codec)
            write_stage_done(
                game_length_stamp,
                inputs=game_length_inputs,
                outputs=[game_length_output],
                config_sha=cfg.config_sha,
            )

    if not margin_up_to_date and margin_inputs:
        margin_frame = _load_seed_frames(margin_paths)
        if margin_frame.empty:
            stage_log.missing_input("no margin stats available")
        else:
            aggregated = _aggregate_seed_stats(margin_frame)
            table = pa.Table.from_pandas(aggregated, preserve_index=False)
            write_parquet_atomic(table, margin_output, codec=cfg.parquet_codec)
            write_stage_done(
                margin_stamp,
                inputs=margin_inputs,
                outputs=[margin_output],
                config_sha=cfg.config_sha,
            )

    LOGGER.info(
        "Interseed game stats written",
        extra={
            "stage": "interseed_game_stats",
            "game_length_output": str(game_length_output),
            "margin_output": str(margin_output),
        },
    )


def _seed_analysis_dirs(cfg: AppConfig) -> list[SeedInputs]:
    seeds = cfg.sim.interseed_seed_list()
    analysis_subdir = cfg.io.analysis_subdir
    analysis_dirs: list[SeedInputs] = []

    base_results_dir = cfg.results_root
    if cfg.interseed_input_dir is not None:
        input_dir = Path(cfg.interseed_input_dir)
        if input_dir.name == analysis_subdir:
            base_results_dir = input_dir.parent
        else:
            analysis_dirs.append(SeedInputs(seed=int(cfg.sim.seed), analysis_dir=input_dir))

    base_root, _ = split_seeded_results_dir(base_results_dir)
    resolved_seeds = list(seeds) if seeds else []
    if not resolved_seeds and base_root.parent.exists():
        prefix = f\"{base_root.name}_seed_\"
        for entry in base_root.parent.iterdir():
            if not entry.is_dir() or not entry.name.startswith(prefix):
                continue
            parsed_base, parsed_seed = split_seeded_results_dir(entry)
            if parsed_seed is None or parsed_base != base_root:
                continue
            resolved_seeds.append(parsed_seed)
    if resolved_seeds:
        for seed in sorted(set(resolved_seeds)):
            results_dir = resolve_results_dir(base_root, seed)
            analysis_dir = results_dir / analysis_subdir
            analysis_dirs.append(SeedInputs(seed=seed, analysis_dir=analysis_dir))
    elif not analysis_dirs:
        analysis_dirs.append(SeedInputs(seed=int(cfg.sim.seed), analysis_dir=base_results_dir / analysis_subdir))

    unique: list[SeedInputs] = []
    seen: set[Path] = set()
    for entry in analysis_dirs:
        if entry.analysis_dir in seen:
            continue
        seen.add(entry.analysis_dir)
        if entry.analysis_dir.exists():
            unique.append(entry)
    return unique


def _seed_input_paths(
    seeds: Iterable[SeedInputs],
    cfg: AppConfig,
    *,
    candidates: Iterable[str],
) -> list[tuple[int, Path]]:
    stage_folder = cfg._interseed_input_folder("game_stats")
    if stage_folder is None:
        stage_folder = resolve_stage_layout(cfg).require_folder("game_stats")
    input_paths: list[tuple[int, Path]] = []
    for entry in seeds:
        stage_root = entry.analysis_dir / stage_folder / "pooled"
        for name in candidates:
            path = stage_root / name
            if path.exists():
                input_paths.append((entry.seed, path))
                break
    return input_paths


def _load_seed_frames(paths: Iterable[tuple[int, Path]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed, path in paths:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df = df.copy()
        if "players" in df.columns and "n_players" not in df.columns:
            df.rename(columns={"players": "n_players"}, inplace=True)
        if "n_players" in df.columns:
            df["n_players"] = pd.to_numeric(df["n_players"], errors="coerce").astype("Int64")
        df["seed"] = int(seed)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate_seed_stats(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    group_keys = [
        key for key in ("summary_level", "strategy", "n_players") if key in frame.columns
    ]
    if "n_players" not in group_keys:
        raise ValueError("game stats missing n_players column")

    numeric_cols = [
        col
        for col in frame.select_dtypes(include=[np.number]).columns
        if col != "seed"
    ]
    grouped = frame.groupby(group_keys, dropna=False, sort=False)
    n_seeds = grouped["seed"].nunique().rename("n_seeds")
    means = grouped[numeric_cols].mean()
    stds = grouped[numeric_cols].std(ddof=0)
    critical = _critical_values(n_seeds)

    result = means.copy()
    result.columns = [f"{col}_seed_mean" for col in means.columns]
    for col in numeric_cols:
        result[f"{col}_seed_std"] = stds[col]
        se = stds[col] / np.sqrt(n_seeds)
        ci_lo = means[col] - critical * se
        ci_hi = means[col] + critical * se
        result[f"{col}_seed_ci_lo"] = ci_lo.where(n_seeds > 1)
        result[f"{col}_seed_ci_hi"] = ci_hi.where(n_seeds > 1)

    result = result.join(n_seeds)
    result = result.reset_index()
    ordered_cols = group_keys + ["n_seeds"] + [
        col for col in result.columns if col not in set(group_keys) | {"n_seeds"}
    ]
    return result[ordered_cols]


def _critical_values(n_seeds: pd.Series) -> pd.Series:
    """Return two-sided 95% critical values, using t for small samples."""
    df = (n_seeds - 1).astype(float)
    crit = pd.Series(NORMAL_975, index=n_seeds.index, dtype=float)
    use_t = (n_seeds < T_CRIT_N_SEEDS) & (n_seeds > 1)
    if use_t.any():
        crit.loc[use_t] = t.ppf(0.975, df.loc[use_t])
    return crit


__all__ = ["run"]
