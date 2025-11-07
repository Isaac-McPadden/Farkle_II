from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from farkle.analysis import run_bonferroni_head2head as _h2h
from farkle.config import AppConfig
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
    z: float
    runtime_hours: float
    tiers: dict[str, int]
    elite_count: int
    games_per_pair: int
    total_games: int
    total_pairs: int


def run(cfg: AppConfig) -> None:
    out = cfg.analysis_dir / "bonferroni_pairwise.parquet"
    if out.exists() and out.stat().st_mtime >= cfg.curated_parquet.stat().st_mtime:
        LOGGER.info(
            "Head-to-head results up-to-date",
            extra={"stage": "head2head", "path": str(out)},
        )
        return

    design_kwargs = _build_design_kwargs(cfg)
    _maybe_autotune_tiers(cfg, design_kwargs)

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
    design = dict(getattr(cfg.head2head, "bonferroni_design", {}) or {})
    filtered = {k: v for k, v in design.items() if k in _GFP_FIELDS}
    filtered.setdefault("k_players", 2)
    filtered["method"] = "bonferroni"
    filtered["full_pairwise"] = True
    filtered.setdefault("endpoint", "pairwise")
    return filtered


def _maybe_autotune_tiers(cfg: AppConfig, design_kwargs: dict[str, Any]) -> None:
    target_hours = cfg.analysis.head2head_target_hours
    games_per_sec = cfg.analysis.head2head_games_per_sec
    if not target_hours or target_hours <= 0:
        return
    if not games_per_sec or games_per_sec <= 0:
        LOGGER.warning(
            "Tier auto-tune skipped: head2head_games_per_sec must be positive",
            extra={"stage": "head2head"},
        )
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

    means = dict(zip(df["strategy"], df["mu"]))
    stdevs = dict(zip(df["strategy"], df["sigma"]))

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

    with atomic_path(str(tiers_path)) as tmp_path, Path(tmp_path).open("w") as fh:
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
    if not means:
        return None

    target_low = target_hours * max(0.0, 1.0 - tolerance_pct / 100.0)
    target_high = target_hours * (1.0 + tolerance_pct / 100.0)

    def evaluate(z: float) -> TierCandidate:
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
    games_per_pair = games_for_power(**kwargs)
    pairs = elite_count * (elite_count - 1) // 2
    total_games = games_per_pair * pairs
    runtime_hours = total_games / (games_per_sec * 3600.0)
    return runtime_hours, games_per_pair, total_games, pairs
