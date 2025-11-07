from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from farkle.config import AppConfig
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
            seed: {k: self.path_for(seed, k) for k in self.player_counts}
            for seed in self.seeds
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
            columns=["strategy", "wins", "games", "win_rate", "winrate", "seed", "player_count", "k"]
        )

    summary = _summarize(locator, frame, missing)
    return frame, summary


def _load_job(job: MetricJob, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    needed = {"strategy", "wins", "total_games_strat", "games", "win_rate", "winrate"}
    read_cols: Sequence[str] | None
    if columns is None:
        read_cols = None
    else:
        read_cols = sorted(set(columns).union(needed))

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


def _summarize(locator: MetricsLocator, frame: pd.DataFrame, missing: list[MetricJob]) -> MetricsSummary:
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
            warnings.append(
                f"Seed {seed} missing data for player counts {sorted(zero_ks)}"
            )

    for k in ks:
        zero_seeds = [seed for seed in seeds if row_counts.loc[seed, k] == 0]
        if zero_seeds:
            warnings.append(
                f"Player count {k} missing data for seeds {sorted(zero_seeds)}"
            )

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


__all__ = [
    "MetricJob",
    "MetricsLocator",
    "MetricsSummary",
    "collect_isolated_metrics",
    "locator_from_config",
]
