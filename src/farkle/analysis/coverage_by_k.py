# src/farkle/analysis/coverage_by_k.py
"""Summarize strategy coverage by player count (k) from metrics outputs."""
from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from farkle.analysis import stage_logger
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig
from farkle.simulation.simulation import generate_strategy_grid
from farkle.utils.artifacts import write_csv_atomic, write_parquet_atomic

LOGGER = logging.getLogger(__name__)

OUTPUT_PARQUET = "coverage_by_k.parquet"
OUTPUT_CSV = "coverage_by_k.csv"


def run(cfg: AppConfig, *, force: bool = False) -> None:
    """Compute coverage statistics per player count and seed."""
    stage_log = stage_logger("coverage_by_k", logger=LOGGER)
    stage_log.start()

    metrics_path = cfg.metrics_input_path()
    if not metrics_path.exists():
        stage_log.missing_input("metrics parquet missing", path=str(metrics_path))
        return

    stage_dir = cfg.stage_dir("coverage_by_k")
    output_parquet = stage_dir / OUTPUT_PARQUET
    output_csv = _optional_csv_path(cfg, stage_dir)

    done_path = stage_done_path(stage_dir, "coverage_by_k")
    coverage_inputs = _coverage_inputs(cfg, metrics_path)
    outputs = [output_parquet, *([output_csv] if output_csv else [])]

    if not force and stage_is_up_to_date(
        done_path,
        inputs=coverage_inputs,
        outputs=outputs,
        config_sha=cfg.config_sha,
    ):
        LOGGER.info(
            "Coverage-by-k outputs up-to-date",
            extra={"stage": "coverage_by_k", "path": str(done_path)},
        )
        return

    coverage = _build_coverage(cfg, metrics_path, coverage_inputs)
    if coverage.empty:
        stage_log.missing_input("metrics parquet empty", metrics_path=str(metrics_path))
        return

    table = pa.Table.from_pandas(coverage, preserve_index=False)
    write_parquet_atomic(table, output_parquet, codec=cfg.parquet_codec)
    if output_csv is not None:
        write_csv_atomic(coverage, output_csv)

    write_stage_done(
        done_path,
        inputs=coverage_inputs,
        outputs=outputs,
        config_sha=cfg.config_sha,
    )

    _log_imbalance_warnings(coverage)


def _coverage_inputs(cfg: AppConfig, metrics_path: Path) -> list[Path]:
    inputs = [metrics_path]
    player_counts = _player_counts_from_config(cfg)
    for k in player_counts:
        path = _resolve_isolated_metrics_path(cfg, k)
        if path is not None and path.exists():
            inputs.append(path)
    return inputs


def _player_counts_from_config(cfg: AppConfig) -> list[int]:
    counts = []
    for entry in cfg.sim.n_players_list:
        try:
            counts.append(int(entry))
        except (TypeError, ValueError):
            continue
    return sorted({k for k in counts if k > 0})


def _optional_csv_path(cfg: AppConfig, stage_dir: Path) -> Path | None:
    outputs = cfg.analysis.outputs or {}
    csv_setting = outputs.get("coverage_by_k_csv")
    if not csv_setting:
        return None
    if isinstance(csv_setting, str):
        target = Path(csv_setting)
        if not target.is_absolute():
            return stage_dir / target
        return target
    return stage_dir / OUTPUT_CSV


def _build_coverage(
    cfg: AppConfig, metrics_path: Path, coverage_inputs: Iterable[Path]
) -> pd.DataFrame:
    counts = _stream_metrics_counts(metrics_path, default_seed=int(cfg.sim.seed))
    k_grid = _player_counts_from_config(cfg)
    if not k_grid:
        k_grid = sorted(counts["k"].unique().tolist())

    if counts.empty:
        seed_grid = cfg.sim.seed_list or [cfg.sim.seed]
    else:
        seed_grid = sorted(counts["seed"].unique().tolist())
    seed_grid = [int(seed) for seed in seed_grid]

    grid = pd.MultiIndex.from_product([seed_grid, k_grid], names=["seed", "k"]).to_frame(
        index=False
    )
    counts = grid.merge(counts, on=["seed", "k"], how="left")
    counts["games"] = counts["games"].fillna(0).astype(int)
    counts["strategies"] = counts["strategies"].fillna(0).astype(int)
    if "missing_before_pad" not in counts.columns:
        counts["missing_before_pad"] = pd.NA
    counts["missing_before_pad"] = counts["missing_before_pad"].astype("Int64")

    expected_by_k = _expected_strategies_by_k(cfg, k_grid, coverage_inputs)
    counts["expected_strategies"] = counts["k"].map(expected_by_k).astype("Int64")
    inferred_missing = (
        counts["expected_strategies"].fillna(counts["strategies"]) - counts["strategies"]
    ).clip(lower=0)
    missing_before_pad = counts["missing_before_pad"]
    if missing_before_pad.notna().any():
        counts["missing_strategies"] = missing_before_pad.fillna(inferred_missing)
    else:
        counts["missing_strategies"] = inferred_missing
    counts["padded_strategies"] = counts["missing_strategies"]

    games_by_k = counts.groupby("k", sort=False)["games"].sum()
    strategies_by_k = counts.groupby("k", sort=False)["strategies"].max()
    seeds_by_k = counts.groupby("k", sort=False)["seed"].nunique()
    counts["games_per_k"] = counts["k"].map(games_by_k)
    counts["strategies_per_k"] = counts["k"].map(strategies_by_k)
    counts["seeds_present"] = counts["k"].map(seeds_by_k)

    ordered = [
        "k",
        "seed",
        "games",
        "games_per_k",
        "strategies",
        "strategies_per_k",
        "expected_strategies",
        "missing_before_pad",
        "missing_strategies",
        "padded_strategies",
        "seeds_present",
    ]
    remaining = [c for c in counts.columns if c not in ordered]
    return counts[ordered + remaining].sort_values(["k", "seed"]).reset_index(drop=True)


def _stream_metrics_counts(metrics_path: Path, *, default_seed: int) -> pd.DataFrame:
    dataset = ds.dataset(metrics_path, format="parquet")
    schema_names = set(dataset.schema.names)

    players_col = "n_players" if "n_players" in schema_names else "players"
    strategy_col = "strategy" if "strategy" in schema_names else "strategy_id"
    games_col = "games" if "games" in schema_names else "total_games_strat"
    seed_col = "seed" if "seed" in schema_names else None
    missing_col = "missing_before_pad" if "missing_before_pad" in schema_names else None

    required = {players_col, strategy_col, games_col}
    if not required.issubset(schema_names):
        missing = sorted(required - schema_names)
        raise ValueError(f"metrics parquet missing required columns: {missing}")

    columns = [players_col, strategy_col, games_col]
    if seed_col:
        columns.append(seed_col)
    if missing_col:
        columns.append(missing_col)

    counts: dict[tuple[int, int], dict[str, object]] = {}
    scanner = dataset.scanner(columns=columns, use_threads=True)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        df = batch.to_pandas()
        df = df.copy()
        df["k"] = pd.to_numeric(df[players_col], errors="coerce")
        df["strategy"] = pd.to_numeric(df[strategy_col], errors="coerce")
        df["games"] = pd.to_numeric(df[games_col], errors="coerce").fillna(0)
        if seed_col is None:
            df["seed"] = default_seed
        else:
            df["seed"] = pd.to_numeric(df[seed_col], errors="coerce").fillna(default_seed)
        if missing_col is not None:
            df["missing_before_pad"] = pd.to_numeric(df[missing_col], errors="coerce")

        df = df.dropna(subset=["k", "strategy", "seed"])
        if df.empty:
            continue
        df["k"] = df["k"].astype(int)
        df["strategy"] = df["strategy"].astype(int)
        df["seed"] = df["seed"].astype(int)
        df["games"] = df["games"].astype(int)

        for (seed, k), group in df.groupby(["seed", "k"], sort=False):
            entry = counts.setdefault(
                (seed, k),
                {"games": 0, "strategies": set(), "missing_before_pad": None},
            )
            entry["games"] = int(entry["games"]) + int(group["games"].sum())
            strategies = entry["strategies"]
            if isinstance(strategies, set):
                strategies.update(group["strategy"].unique().tolist())
            if missing_col is not None:
                missing_values = group["missing_before_pad"].dropna()
                if not missing_values.empty:
                    observed = int(missing_values.max())
                    current = entry.get("missing_before_pad")
                    if current is None or observed > int(current):
                        entry["missing_before_pad"] = observed

    rows = []
    for (seed, k), payload in sorted(counts.items()):
        strategies = payload.get("strategies", set())
        rows.append(
            {
                "seed": seed,
                "k": k,
                "games": int(payload.get("games", 0)),
                "strategies": len(strategies) if isinstance(strategies, set) else 0,
                "missing_before_pad": payload.get("missing_before_pad"),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["seed", "k", "games", "strategies", "missing_before_pad"],
    )


def _expected_strategies_by_k(
    cfg: AppConfig, player_counts: Iterable[int], inputs: Iterable[Path]
) -> dict[int, int]:
    expected: dict[int, int] = {}
    iso_paths = _map_isolated_paths(cfg, player_counts, inputs)
    fallback_count: int | None = None

    for k in player_counts:
        path = iso_paths.get(int(k))
        if path is not None and path.exists():
            try:
                expected[k] = int(pq.ParquetFile(path).metadata.num_rows)
                continue
            except (OSError, RuntimeError):
                LOGGER.warning(
                    "Coverage: failed to read isolated metrics metadata",
                    extra={"stage": "coverage_by_k", "path": str(path), "k": int(k)},
                )
        if fallback_count is None:
            _, meta = generate_strategy_grid(
                score_thresholds=cfg.sim.score_thresholds,
                dice_thresholds=cfg.sim.dice_thresholds,
                smart_five_opts=cfg.sim.smart_five_opts,
                smart_one_opts=cfg.sim.smart_one_opts,
                consider_score_opts=cfg.sim.consider_score_opts,
                consider_dice_opts=cfg.sim.consider_dice_opts,
                auto_hot_dice_opts=cfg.sim.auto_hot_dice_opts,
                run_up_score_opts=cfg.sim.run_up_score_opts,
            )
            fallback_count = int(meta["strategy_id"].shape[0])
        expected[k] = fallback_count
    return expected


def _map_isolated_paths(
    cfg: AppConfig, player_counts: Iterable[int], inputs: Iterable[Path]
) -> dict[int, Path]:
    lookup = {path.resolve(): path for path in inputs}
    mapping: dict[int, Path] = {}
    for k in player_counts:
        path = _resolve_isolated_metrics_path(cfg, int(k))
        if path is None:
            continue
        resolved = path.resolve()
        if resolved in lookup:
            mapping[int(k)] = lookup[resolved]
        else:
            mapping[int(k)] = path
    return mapping


def _resolve_isolated_metrics_path(cfg: AppConfig, k: int) -> Path | None:
    preferred = cfg.metrics_isolated_path(k)
    if preferred.exists():
        return preferred
    legacy = cfg.legacy_metrics_isolated_path(k)
    if legacy.exists():
        return legacy
    if preferred.parent.exists():
        return preferred
    return None


def _log_imbalance_warnings(coverage: pd.DataFrame) -> None:
    if coverage.empty:
        return
    for k, group in coverage.groupby("k", sort=True):
        strategy_min = int(group["strategies"].min())
        strategy_max = int(group["strategies"].max())
        games_min = int(group["games"].min())
        games_max = int(group["games"].max())
        missing = int(group["missing_strategies"].sum())

        if strategy_min != strategy_max:
            LOGGER.warning(
                "Coverage: strategy counts differ across seeds",
                extra={
                    "stage": "coverage_by_k",
                    "player_count": int(k),
                    "min_strategies": strategy_min,
                    "max_strategies": strategy_max,
                    "seeds": sorted(group["seed"].unique().tolist()),
                },
            )

        if games_min != games_max:
            LOGGER.warning(
                "Coverage: game counts differ across seeds",
                extra={
                    "stage": "coverage_by_k",
                    "player_count": int(k),
                    "min_games": games_min,
                    "max_games": games_max,
                    "seeds": sorted(group["seed"].unique().tolist()),
                },
            )

        if missing > 0:
            LOGGER.warning(
                "Coverage: missing strategies detected",
                extra={
                    "stage": "coverage_by_k",
                    "player_count": int(k),
                    "missing_strategies": missing,
                    "seeds": sorted(group["seed"].unique().tolist()),
                },
            )


__all__ = ["run"]
