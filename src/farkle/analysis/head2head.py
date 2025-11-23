# src/farkle/analysis/head2head.py
"""Run head-to-head significance tests and derive tier candidates.

This module orchestrates game simulations, calculates Bonferroni-corrected
pairwise outcomes, and searches for tier configurations that satisfy specified
power and runtime targets.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from farkle.analysis import run_bonferroni_head2head as _h2h
from farkle.config import AppConfig
from farkle.simulation.simulation import simulate_many_games_from_seeds
from farkle.simulation.strategies import parse_strategy
from farkle.utils.random import spawn_seeds
from farkle.utils.stats import build_tiers, games_for_power
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)

_GFP_FIELDS = {
    "n_strategies",
    "k_players",
    "method",
    "power",
    "control",
    "detectable_lift",
    "baseline_rate",
    "tail",
    "full_pairwise",
    "use_BY",
    "min_games_floor",
    "max_games_cap",
    "bh_target_rank",
    "bh_target_frac",
    "endpoint",
}


@dataclass
class TierCandidate:
    """Estimated tier grouping plus runtime characteristics for head-to-head."""

    z: float
    runtime_hours: float
    tiers: dict[str, int]
    elite_count: int
    games_per_pair: int
    total_games: int
    total_pairs: int


def run(cfg: AppConfig) -> None:
    """Execute Bonferroni head-to-head analysis and manage tier artifacts.

    Args:
        cfg: Application configuration with analysis and file path settings.
    """
    # Always try to auto-tune tiers first so the human-facing tier file
    # reflects the Bonferroni H2H budget even if results are already present.
    design_kwargs = _build_design_kwargs(cfg)
    _maybe_autotune_tiers(cfg, design_kwargs)

    out = cfg.analysis_dir / "bonferroni_pairwise.parquet"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        LOGGER.info(
            "Head-to-head results up-to-date",
            extra={"stage": "head2head", "path": str(out)},
        )
        return

    LOGGER.info(
        "Head-to-head analysis running",
        extra={
            "stage": "head2head",
            "results_dir": str(cfg.results_dir),
            "n_jobs": cfg.analysis.n_jobs,
        },
    )
    try:
        _h2h.run_bonferroni_head2head(
            root=cfg.results_dir,
            n_jobs=cfg.analysis.n_jobs,
            seed=cfg.sim.seed,
            design=design_kwargs,
        )
    except Exception as e:  # noqa: BLE001
        LOGGER.warning(
            "Head-to-head skipped",
            extra={"stage": "head2head", "error": str(e)},
        )


def _build_design_kwargs(cfg: AppConfig) -> dict[str, Any]:
    """Construct sanitized design kwargs for Bonferroni power calculations.

    Args:
        cfg: Application configuration containing head-to-head options.

    Returns:
        Filtered and normalized kwargs compatible with ``games_for_power``.
    """
    design = dict(getattr(cfg.head2head, "bonferroni_design", {}) or {})
    filtered = {k: v for k, v in design.items() if k in _GFP_FIELDS}
    filtered.setdefault("k_players", 2)
    filtered["method"] = "bonferroni"
    filtered["full_pairwise"] = True
    filtered.setdefault("endpoint", "pairwise")
    tail = str(filtered.get("tail", "one_sided")).lower().replace("-", "_")
    if tail != "one_sided":
        raise ValueError(
            "Bonferroni head-to-head uses a one-sided exact test; set tail='one_sided'",
        )
    filtered["tail"] = "one_sided"
    return filtered


def _maybe_autotune_tiers(cfg: AppConfig, design_kwargs: dict[str, Any]) -> None:
    """Calibrate tier thresholds to hit a target runtime budget when possible.

    Args:
        cfg: Application configuration including throughput targets.
        design_kwargs: Prepared Bonferroni design parameters.
    """
    target_hours = cfg.analysis.head2head_target_hours
    games_per_sec = cfg.analysis.head2head_games_per_sec
    if not target_hours or target_hours <= 0:
        return

    ratings_path = cfg.analysis_dir / "ratings_pooled.parquet"
    tiers_path = cfg.analysis_dir / "tiers.json"
    if not ratings_path.exists():
        LOGGER.warning(
            "Tier auto-tune skipped: missing pooled ratings",
            extra={"stage": "head2head", "path": str(ratings_path)},
        )
        return

    df = pd.read_parquet(ratings_path, columns=["strategy", "mu", "sigma"])
    if df.empty:
        LOGGER.warning(
            "Tier auto-tune skipped: pooled ratings empty",
            extra={"stage": "head2head"},
        )
        return

    means = dict(zip(df["strategy"], df["mu"], strict=False))
    stdevs = dict(zip(df["strategy"], df["sigma"], strict=False))

    # Auto-calibrate throughput if not provided
    if not games_per_sec or games_per_sec <= 0:
        try:
            games_per_sec = _calibrate_h2h_games_per_sec(
                df, seed=cfg.sim.seed, n_jobs=cfg.analysis.n_jobs
            )
            LOGGER.info(
                "Measured head-to-head throughput",
                extra={
                    "stage": "head2head",
                    "games_per_sec": round(games_per_sec, 2),
                    "n_jobs": cfg.analysis.n_jobs,
                },
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Tier auto-tune skipped: failed to calibrate throughput",
                extra={"stage": "head2head", "error": str(exc)},
            )
            return

    tolerance_pct = max(0.0, cfg.analysis.head2head_tolerance_pct or 0.0)
    candidate = _search_candidate(
        means=means,
        stdevs=stdevs,
        target_hours=target_hours,
        tolerance_pct=tolerance_pct,
        games_per_sec=games_per_sec,
        design_kwargs=design_kwargs,
    )
    if candidate is None:
        LOGGER.warning(
            "Tier auto-tune failed to find a suitable threshold; retaining existing tiers",
            extra={"stage": "head2head"},
        )
        return

    if tiers_path.exists():
        backup = tiers_path.with_name("tiers_trueskill.json")
        if not backup.exists():
            shutil.copy2(tiers_path, backup)

    # Write the tuned tiers to the canonical path and a z-tagged sibling
    with atomic_path(str(tiers_path)) as tmp_path, Path(tmp_path).open("w") as fh:
        json.dump(candidate.tiers, fh, indent=2, sort_keys=True)

    z_tag = f"{candidate.z:.3f}"
    tiers_z_path = tiers_path.with_name(f"tiers_z{z_tag}.json")
    if not tiers_z_path.exists():
        with atomic_path(str(tiers_z_path)) as tmp_path, Path(tmp_path).open("w") as fh:
            json.dump(candidate.tiers, fh, indent=2, sort_keys=True)

    meta = {
        "z": candidate.z,
        "target_hours": target_hours,
        "tolerance_pct": tolerance_pct,
        "predicted_hours": candidate.runtime_hours,
        "elite_count": candidate.elite_count,
        "games_per_pair": candidate.games_per_pair,
        "total_pairs": candidate.total_pairs,
        "total_games": candidate.total_games,
        "games_per_sec": games_per_sec,
    }
    meta_path = cfg.analysis_dir / "tiers_autotune.json"
    with atomic_path(str(meta_path)) as tmp_path, Path(tmp_path).open("w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    LOGGER.info(
        "Tier auto-tune complete",
        extra={
            "stage": "head2head",
            "z": round(candidate.z, 4),
            "elite_count": candidate.elite_count,
            "predicted_hours": round(candidate.runtime_hours, 3),
            "games_per_pair": candidate.games_per_pair,
            "tiers_path": str(tiers_path),
            "tiers_z_path": str(tiers_z_path),
        },
    )


def _search_candidate(
    *,
    means: Dict[str, float],
    stdevs: Dict[str, float],
    target_hours: float,
    tolerance_pct: float,
    games_per_sec: float,
    design_kwargs: dict[str, Any],
) -> TierCandidate | None:
    """Search for a z-threshold that meets the runtime budget.

    Args:
        means: Strategy means from pooled ratings.
        stdevs: Strategy standard deviations.
        target_hours: Desired runtime budget for head-to-head matches.
        tolerance_pct: Allowed deviation from the runtime target.
        games_per_sec: Estimated throughput used for runtime predictions.
        design_kwargs: Additional parameters passed to power calculations.

    Returns:
        Candidate tier configuration closest to the target runtime, or ``None`` when
        no strategies are available.
    """
    if not means:
        return None

    target_low = target_hours * max(0.0, 1.0 - tolerance_pct / 100.0)
    target_high = target_hours * (1.0 + tolerance_pct / 100.0)

    def evaluate(z: float) -> TierCandidate:
        """Build a tier candidate and runtime estimate for a z-threshold.

        Args:
            z: Z-score threshold applied when grouping strategies into tiers.

        Returns:
            A populated :class:`TierCandidate` describing tier assignments and
            runtime predictions for the supplied threshold.
        """
        tiers = build_tiers(means, stdevs, z=z)
        top_tier = min(tiers.values()) if tiers else 1
        elite_count = sum(1 for val in tiers.values() if val == top_tier)
        runtime_hours, games_per_pair, total_games, total_pairs = _predict_runtime(
            elite_count, games_per_sec, design_kwargs
        )
        return TierCandidate(
            z=z,
            runtime_hours=runtime_hours,
            tiers=tiers,
            elite_count=elite_count,
            games_per_pair=games_per_pair,
            total_games=total_games,
            total_pairs=total_pairs,
        )

    min_z, max_z = 0.5, 6.0
    best_candidate = evaluate(max_z)
    low_candidate = evaluate(min_z)
    candidates = [low_candidate, best_candidate]

    if target_low <= low_candidate.runtime_hours <= target_high:
        return low_candidate
    if target_low <= best_candidate.runtime_hours <= target_high:
        return best_candidate

    z_low, z_high = min_z, max_z
    for _ in range(32):
        z_mid = 0.5 * (z_low + z_high)
        candidate = evaluate(z_mid)
        candidates.append(candidate)
        if target_low <= candidate.runtime_hours <= target_high:
            return candidate
        if candidate.runtime_hours > target_high and candidate.elite_count > 1:
            z_high = z_mid
        else:
            z_low = z_mid

    closest = min(candidates, key=lambda c: abs(c.runtime_hours - target_hours))
    return closest


def _predict_runtime(
    elite_count: int,
    games_per_sec: float,
    design_kwargs: dict[str, Any],
) -> tuple[float, int, int, int]:
    """Estimate runtime and workload counts for a given elite tier size.

    Args:
        elite_count: Number of strategies in the top tier.
        games_per_sec: Throughput used to convert games to hours.
        design_kwargs: Power-calculation options used for games estimation.

    Returns:
        Tuple of (runtime hours, games per pair, total games, total pairs).
    """
    if elite_count <= 1:
        return 0.0, 0, 0, 0

    kwargs = dict(design_kwargs)
    kwargs.update(
        {
            "n_strategies": elite_count,
            "method": "bonferroni",
            "full_pairwise": True,
            "endpoint": "pairwise",
        }
    )
    games_per_strategy = games_for_power(**kwargs)
    per_pair = max(0, elite_count - 1)
    k_players = int(kwargs.get("k_players", 2))
    games_per_pair = (
        0
        if per_pair == 0
        else int(math.ceil(games_per_strategy * max(1, k_players - 1) / per_pair))
    )
    pairs = elite_count * (elite_count - 1) // 2
    total_games = games_per_pair * pairs
    runtime_hours = total_games / (games_per_sec * 3600.0)
    return runtime_hours, games_per_pair, total_games, pairs


def _calibrate_h2h_games_per_sec(
    ratings_df: pd.DataFrame,
    *,
    seed: int,
    n_jobs: int | None,
    sample_games: int = 2000,
) -> float:
    """Measure head-to-head throughput using a small sample of games.

    Args:
        ratings_df: Pooled ratings including strategy and mu columns.
        seed: Seed used to create reproducible game seeds.
        n_jobs: Number of worker processes for simulation.
        sample_games: Number of games to simulate for calibration.

    Returns:
        Estimated games-per-second throughput.

    Raises:
        ValueError: If fewer than two strategies are available for calibration.
    """
    if ratings_df.shape[0] < 2:
        raise ValueError("Need at least two strategies to calibrate throughput")

    top2 = ratings_df.sort_values("mu", ascending=False)["strategy"].head(2).tolist()
    strats = [parse_strategy(top2[0]), parse_strategy(top2[1])]
    seeds = spawn_seeds(sample_games, seed=seed)
    t0 = time.perf_counter()
    simulate_many_games_from_seeds(seeds=seeds, strategies=strats, n_jobs=n_jobs or 1)
    elapsed = max(1e-9, time.perf_counter() - t0)
    return float(sample_games / elapsed)
