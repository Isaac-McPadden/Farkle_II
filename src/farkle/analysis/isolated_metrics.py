# src/farkle/analysis/isolated_metrics.py
"""Compute per-strategy isolated metrics from tournament outputs.

Generates win-rate frames for individual strategies, handles padding across
player counts, and writes parquet artifacts consumed by downstream reporting
and ranking modules.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

from farkle.config import AppConfig
from farkle.simulation.run_tournament import METRIC_LABELS
from farkle.simulation.simulation import generate_strategy_grid
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.parallel import process_map

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricJob:
    """Single parquet load request."""

    seed: int
    player_count: int
    path: Path


@dataclass
class MetricsLocator:
    """Discover per-seed, per-player-count metric files.

    Parameters
    ----------
    data_root:
        Base directory containing the per-seed result folders.
    seeds:
        Seeds we expect to inspect.
    player_counts:
        Player-count levels (``k``) to harvest.
    override_roots:
        Optional mapping of ``seed -> custom directory``.  Paths may be absolute
        or relative to ``data_root``.
    results_template:
        Template used when ``override_roots`` does not supply a directory.
    subdir_template:
        Layout template for the per-player-count subdirectory.
    metrics_template:
        Filename template for the metrics parquet inside each ``k`` directory.
    """

    data_root: Path
    seeds: Sequence[int]
    player_counts: Sequence[int]
    override_roots: Mapping[int, str | Path] | None = None
    results_template: str = "results_seed_{seed}"
    subdir_template: str = "{n}_players"
    metrics_template: str = "{n}p_metrics.parquet"
    _seed_paths: dict[int, Path] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.seeds = tuple(int(s) for s in self.seeds)
        if not self.seeds:
            raise ValueError("MetricsLocator requires at least one seed")
        self.player_counts = tuple(int(k) for k in self.player_counts)
        if not self.player_counts:
            raise ValueError("MetricsLocator requires at least one player_count value")

        overrides = {int(k): Path(v) for k, v in (self.override_roots or {}).items()}
        seed_paths: dict[int, Path] = {}
        for seed in self.seeds:
            candidate = overrides.get(seed)
            if candidate is None:
                candidate = Path(self.results_template.format(seed=seed))
            if not candidate.is_absolute():
                candidate = self.data_root / candidate
            seed_paths[seed] = candidate
        self._seed_paths = seed_paths

    @property
    def seed_paths(self) -> dict[int, Path]:
        """Return a copy of the resolved seed root directories."""
        return dict(self._seed_paths)

    def path_for(self, seed: int, player_count: int) -> Path:
        """Return the expected metrics path for ``(seed, player_count)``."""
        base = self._seed_paths[int(seed)]
        rel = self.subdir_template.format(n=int(player_count))
        name = self.metrics_template.format(n=int(player_count))
        return base / rel / name

    def iter_jobs(self) -> Iterator[MetricJob]:
        """Yield :class:`MetricJob` entries for every seed x player_count pair."""
        for seed in self.seeds:
            for k in self.player_counts:
                yield MetricJob(seed=seed, player_count=k, path=self.path_for(seed, k))

    def as_mapping(self) -> dict[int, dict[int, Path]]:
        """Nested dictionary view ``{seed: {player_count: Path}}``."""
        return {
            seed: {k: self.path_for(seed, k) for k in self.player_counts} for seed in self.seeds
        }


@dataclass(frozen=True)
class MetricsSummary:
    """Diagnostics captured alongside the concatenated metrics DataFrame."""

    locator: MetricsLocator
    missing_jobs: list[MetricJob]
    row_counts: pd.DataFrame
    strategy_counts: pd.DataFrame
    warnings: list[str]
    expected_pairs: int
    loaded_pairs: int

    @property
    def has_missing(self) -> bool:
        """Indicate whether any expected metric files were absent."""
        return bool(self.missing_jobs)


def locator_from_config(
    cfg: AppConfig,
    *,
    seeds: Sequence[int],
    data_root: Path | str | None = None,
    player_counts: Sequence[int] | None = None,
    override_roots: Mapping[int, str | Path] | None = None,
    results_template: str = "results_seed_{seed}",
    subdir_template: str = "{n}_players",
    metrics_template: str = "{n}p_metrics.parquet",
) -> MetricsLocator:
    """Convenience factory that defaults to values from :class:`AppConfig`."""

    base_dir = Path(data_root) if data_root is not None else cfg.io.results_dir.parent
    players = player_counts or cfg.sim.n_players_list
    return MetricsLocator(
        data_root=base_dir,
        seeds=tuple(seeds),
        player_counts=tuple(players),
        override_roots=override_roots,
        results_template=results_template,
        subdir_template=subdir_template,
        metrics_template=metrics_template,
    )


def collect_isolated_metrics(
    locator: MetricsLocator,
    *,
    columns: Sequence[str] | None = None,
    n_jobs: int | None = None,
    strict: bool = False,
) -> tuple[pd.DataFrame, MetricsSummary]:
    """Load per-seed metrics, returning the concatenated frame and QA summary."""

    jobs: list[MetricJob] = []
    missing: list[MetricJob] = []
    for job in locator.iter_jobs():
        if job.path.exists():
            jobs.append(job)
        else:
            missing.append(job)
            LOGGER.warning(
                "Missing metrics parquet",
                extra={
                    "stage": "isolated_metrics",
                    "seed": job.seed,
                    "player_count": job.player_count,
                    "path": str(job.path),
                },
            )
    if strict and missing:
        raise FileNotFoundError(f"Missing metrics files: {[str(m.path) for m in missing]}")

    frames: list[pd.DataFrame] = []
    if jobs:
        loader = functools.partial(_load_job, columns=columns)
        frames = list(process_map(loader, jobs, n_jobs=n_jobs))

    if frames:
        frame = pd.concat(frames, ignore_index=True, sort=False)
    else:
        frame = pd.DataFrame(
            columns=[
                "strategy",
                "wins",
                "games",
                "win_rate",
                "winrate",
                "seed",
                "player_count",
                "k",
            ]
        )

    summary = _summarize(locator, frame, missing)
    return frame, summary


def _load_job(job: MetricJob, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load and normalize a single metrics parquet shard.

    Args:
        job: Descriptor containing seed, player count, and file path.
        columns: Optional subset of columns to read; required metric fields are
            always included.

    Returns:
        Dataframe with standardized columns for wins, games, winrate, and
        locator metadata.
    """
    needed = {"strategy", "wins", "total_games_strat", "games", "win_rate", "winrate"}
    read_cols: Sequence[str] | None
    read_cols = None if columns is None else sorted(set(columns).union(needed))

    df = pd.read_parquet(job.path, columns=read_cols)
    df = df.copy()

    if "total_games_strat" in df.columns and "games" not in df.columns:
        df.rename(columns={"total_games_strat": "games"}, inplace=True)

    if "wins" in df.columns:
        df["wins"] = pd.to_numeric(df["wins"], errors="coerce").astype("Int64")
    if "games" in df.columns:
        df["games"] = pd.to_numeric(df["games"], errors="coerce").astype("Int64")

    if "winrate" in df.columns:
        df["winrate"] = pd.to_numeric(df["winrate"], errors="coerce")
    elif "win_rate" in df.columns:
        df["winrate"] = pd.to_numeric(df["win_rate"], errors="coerce")
    elif {"wins", "games"}.issubset(df.columns):
        wins = df["wins"].astype(float)
        games = df["games"].replace({0: np.nan}).astype(float)
        df["winrate"] = (wins / games).fillna(0.0)
    else:
        df["winrate"] = np.nan

    df["seed"] = int(job.seed)
    df["player_count"] = int(job.player_count)
    df["k"] = int(job.player_count)
    df["metrics_path"] = str(job.path)

    return df


def _summarize(
    locator: MetricsLocator, frame: pd.DataFrame, missing: list[MetricJob]
) -> MetricsSummary:
    """Produce QA diagnostics for loaded metrics across seeds and player counts.

    Args:
        locator: Metrics locator used to determine expected seed/player pairs.
        frame: Concatenated metrics dataframe across available jobs.
        missing: Jobs that could not be loaded because files were absent.

    Returns:
        A :class:`MetricsSummary` containing per-seed counts and warning
        messages highlighting gaps or mismatches.
    """
    seeds = list(locator.seeds)
    ks = list(locator.player_counts)
    expected_pairs = len(seeds) * len(ks)

    if frame.empty:
        row_counts = pd.DataFrame(0, index=seeds, columns=ks)
        strat_counts = row_counts.copy()
    else:
        row_counts = (
            frame.groupby(["seed", "k"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=seeds, columns=ks, fill_value=0)
        )
        strat_counts = (
            frame.groupby(["seed", "k"])["strategy"]
            .nunique()
            .unstack(fill_value=0)
            .reindex(index=seeds, columns=ks, fill_value=0)
        )

    loaded_pairs = int((row_counts > 0).sum().sum())

    warnings: list[str] = []
    if not frame.empty:
        expected_per_k = strat_counts.max(axis=0)
        for k, expected in expected_per_k.items():
            if expected <= 0:
                continue
            for seed in seeds:
                actual = int(strat_counts.loc[seed, k])
                if actual != expected:
                    warnings.append(
                        f"Strategy count mismatch for seed {seed}, k={k}: {actual} != {expected}"
                    )

    for job in missing:
        warnings.append(
            f"Missing metrics parquet for seed {job.seed}, k={job.player_count}: {job.path}"
        )

    for seed in seeds:
        zero_ks = [k for k in ks if row_counts.loc[seed, k] == 0]
        if zero_ks:
            warnings.append(f"Seed {seed} missing data for player counts {sorted(zero_ks)}")

    for k in ks:
        zero_seeds = [seed for seed in seeds if row_counts.loc[seed, k] == 0]
        if zero_seeds:
            warnings.append(f"Player count {k} missing data for seeds {sorted(zero_seeds)}")

    if frame.empty and not missing:
        warnings.append("No metrics data loaded; verify locator paths.")

    summary = MetricsSummary(
        locator=locator,
        missing_jobs=missing,
        row_counts=row_counts,
        strategy_counts=strat_counts,
        warnings=warnings,
        expected_pairs=expected_pairs,
        loaded_pairs=loaded_pairs,
    )
    return summary


# ---------------------------------------------------------------------------
# Expanded metrics normalization
# ---------------------------------------------------------------------------


def build_isolated_metrics(cfg: AppConfig, player_count: int, *, force: bool = False) -> Path:
    """
    Normalize a per-k metrics parquet into ``analysis/data/<kp>/<kp>_isolated_metrics.parquet``.
    """

    src = cfg.results_dir / f"{player_count}_players" / f"{player_count}p_metrics.parquet"
    if not src.exists():
        raise FileNotFoundError(src)

    out_dir = cfg.analysis_dir / "data" / f"{player_count}p"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{player_count}p_isolated_metrics.parquet"
    if not force and dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst

    df = pd.read_parquet(src)
    processed = _prepare_metrics_dataframe(cfg, df, player_count)
    table = pa.Table.from_pandas(processed, preserve_index=False)
    write_parquet_atomic(table, dst)
    LOGGER.info(
        "Isolated metrics written",
        extra={
            "stage": "metrics",
            "player_count": player_count,
            "rows": table.num_rows,
            "path": str(dst),
        },
    )
    return dst


def _prepare_metrics_dataframe(cfg: AppConfig, df: pd.DataFrame, player_count: int) -> pd.DataFrame:
    """Clean metric columns and enforce consistent shapes for downstream use.

    Args:
        cfg: Application configuration providing strategy grid options.
        df: Raw metrics dataframe produced by simulation.
        player_count: Number of players associated with the metrics file.

    Returns:
        Normalized dataframe with padded strategies, corrected win metrics, and
        compressed metric columns.
    """
    df = df.copy()
    if "strategy" not in df.columns:
        df["strategy"] = []

    df["wins"] = df["wins"].fillna(0).astype(float)
    correction = df.get("sum_winner_hit_max_rounds", 0).fillna(0.0)
    df["false_wins_handled"] = correction
    df["_hit_flag"] = correction
    df["wins"] = df["wins"] - correction
    df["wins"] = df["wins"].clip(lower=0.0)

    needs_recalc = correction.gt(0) & df["wins"].gt(0)
    if needs_recalc.any():
        wins = df.loc[needs_recalc, "wins"].replace(0, np.nan)
        for label in METRIC_LABELS:
            sum_col = f"sum_{label}"
            sq_col = f"sq_sum_{label}"
            mean_col = f"mean_{label}"
            var_col = f"var_{label}"
            if {sum_col, sq_col, mean_col, var_col}.issubset(df.columns):
                sums = df.loc[needs_recalc, sum_col]
                sqs = df.loc[needs_recalc, sq_col]
                new_mean = (sums / wins).fillna(0.0)
                df.loc[needs_recalc, mean_col] = new_mean
                ex2 = (sqs / wins).fillna(0.0)
                new_var = np.maximum(ex2 - (new_mean**2), 0.0)
                df.loc[needs_recalc, var_col] = new_var

    games_col = "total_games_strat"
    if games_col not in df.columns:
        df[games_col] = df.get("games", 0)
    df[games_col] = df[games_col].fillna(df[games_col].max()).astype(float)
    df["games"] = df[games_col].astype(np.int64)
    safe_games = df[games_col].replace(0, np.nan)
    df["win_rate"] = (df["wins"] / safe_games).fillna(0.0)
    if "sum_winning_score" in df.columns:
        df["expected_score"] = (df["sum_winning_score"] / safe_games).fillna(0.0)

    df = df.set_index("strategy")
    df = _pad_strategies(cfg, df)

    games_mode = int(_games_mode(df["games"]))
    missing_mask = df["wins"].isna()
    if missing_mask.any():
        df.loc[missing_mask, "wins"] = 0
        df.loc[missing_mask, "games"] = games_mode
        df.loc[missing_mask, games_col] = games_mode
        df.loc[missing_mask, "win_rate"] = 0.0
        if "expected_score" in df.columns:
            df.loc[missing_mask, "expected_score"] = np.nan
        numeric_cols = [
            c
            for c in df.columns
            if c not in {"wins", games_col, "games", "win_rate", "expected_score", "_hit_flag"}
        ]
        for col in numeric_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df.loc[missing_mask, col] = 0.0

    penalty_mask = df["_hit_flag"].fillna(0) >= 1
    penalty_mask &= df["wins"].eq(0)
    if penalty_mask.any():
        df.loc[penalty_mask, "games"] = games_mode
        df.loc[penalty_mask, games_col] = games_mode
        df.loc[penalty_mask, "win_rate"] = 0.0
        if "expected_score" in df.columns:
            df.loc[penalty_mask, "expected_score"] = np.nan
        numeric_cols = [
            c
            for c in df.columns
            if c not in {"wins", games_col, "games", "win_rate", "expected_score", "_hit_flag"}
        ]
        for col in numeric_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df.loc[penalty_mask, col] = 0.0

    df["_hit_flag"] = df["_hit_flag"].fillna(0)

    df = _compress_metric_columns(df)
    df = df.drop(columns=[c for c in df.columns if "winner_hit_max_rounds" in c], errors="ignore")
    df.drop(columns=["_hit_flag"], inplace=True, errors="ignore")
    df["wins"] = df["wins"].round().astype(int)
    df["false_wins_handled"] = df["false_wins_handled"].round().astype(int)
    df["n_players"] = player_count
    df.reset_index(inplace=True)
    desired_order = ["strategy", "n_players", "games", "wins", "win_rate", "expected_score"]
    remaining = [c for c in df.columns if c not in desired_order]
    return df[desired_order + remaining]


_STRATEGY_CACHE: dict[int, list[str]] = {}


def _pad_strategies(cfg: AppConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Align metric rows to the full strategy grid defined in the config.

    Args:
        cfg: Application configuration containing simulation strategy options.
        df: Metrics dataframe indexed by strategy.

    Returns:
        Dataframe reindexed to include all possible strategies, introducing
        ``NaN`` rows where metrics are missing.
    """
    cache_key = id(cfg)
    if cache_key not in _STRATEGY_CACHE:
        strategies, _ = generate_strategy_grid(
            score_thresholds=cfg.sim.score_thresholds,
            dice_thresholds=cfg.sim.dice_thresholds,
            smart_five_opts=cfg.sim.smart_five_opts,
            smart_one_opts=cfg.sim.smart_one_opts,
            consider_score_opts=cfg.sim.consider_score_opts,
            consider_dice_opts=cfg.sim.consider_dice_opts,
            auto_hot_dice_opts=cfg.sim.auto_hot_dice_opts,
            run_up_score_opts=cfg.sim.run_up_score_opts,
        )
        _STRATEGY_CACHE[cache_key] = [str(s) for s in strategies]
    strategy_index = pd.Index(_STRATEGY_CACHE[cache_key], name="strategy")
    return df.reindex(strategy_index)


def _games_mode(series: pd.Series) -> float:
    """Return a sensible fill value for missing game counts."""
    mode = series.dropna().mode()
    if not mode.empty:
        return mode.iloc[0]
    valid = series.dropna()
    if not valid.empty:
        return valid.iloc[0]
    return 0.0


def _normalize_metric_name(name: str) -> str:
    """Remove redundant prefixes from metric names for presentation consistency."""
    if name.startswith("winning_"):
        name = name.removeprefix("winning_")
    if name.startswith("winner_"):
        name = name.removeprefix("winner_")
    return name


def _compress_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Condense sum/variance columns into mean and standard deviation pairs.

    Args:
        df: Metrics dataframe containing per-strategy aggregate columns.

    Returns:
        Dataframe with variance-derived standard deviation columns and
        simplified metric naming.
    """
    rename_map: dict[str, str] = {}
    var_cols = [c for c in df.columns if c.startswith("var_")]
    for var_col in var_cols:
        base = var_col.removeprefix("var_")
        if base == "winner_hit_max_rounds":
            drops = [var_col, f"sum_{base}", f"sq_sum_{base}", f"mean_{base}"]
            df.drop(columns=[c for c in drops if c in df.columns], inplace=True, errors="ignore")
            continue
        mean_col = f"mean_{base}"
        if mean_col not in df.columns:
            continue
        sd_col = f"sd_{base}"
        df[sd_col] = np.sqrt(df[var_col].clip(lower=0.0))
        drops = [var_col, f"sum_{base}", f"sq_sum_{base}"]
        df.drop(columns=[c for c in drops if c in df.columns], inplace=True, errors="ignore")
        normalized = _normalize_metric_name(base)
        rename_map[mean_col] = f"mean_{normalized}"
        rename_map[sd_col] = f"sd_{normalized}"

    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    return df


__all__ = [
    "MetricJob",
    "MetricsLocator",
    "MetricsSummary",
    "build_isolated_metrics",
    "collect_isolated_metrics",
    "locator_from_config",
]
