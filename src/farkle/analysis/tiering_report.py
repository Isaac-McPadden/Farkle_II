# src/farkle/analysis/tiering_report.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from farkle.analysis.isolated_metrics import build_isolated_metrics
from farkle.config import AppConfig
from farkle.utils.mdd import tiering_ingredients_from_df
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


@dataclass
class TieringInputs:
    seeds: list[int]
    player_counts: list[int]
    weights_by_k: Mapping[int, float] | None
    z_star: float


def run(cfg: AppConfig) -> None:
    if not cfg.analysis.run_tiering_report:
        LOGGER.info("Tiering report disabled", extra={"stage": "tiering"})
        return

    inputs = _prepare_inputs(cfg)
    tier_file = cfg.analysis_dir / "tiers.json"
    if not tier_file.exists():
        LOGGER.warning(
            "Tiering report skipped: missing tiers.json",
            extra={"stage": "tiering", "path": str(tier_file)},
        )
        return

    df = _load_isolated_metrics(cfg, inputs)
    if df.empty:
        LOGGER.warning(
            "Tiering report skipped: no isolated metrics discovered",
            extra={"stage": "tiering"},
        )
        return

    tier_data = tiering_ingredients_from_df(
        df,
        strategy_col="strategy",
        k_col="n_players",
        seed_col="seed",
        wins_col="wins",
        games_col="games",
        winrate_col="win_rate",
        weights_by_k=inputs.weights_by_k,
        z_star=inputs.z_star,
    )

    frequentist_tiers = _build_frequentist_tiers(df, tier_data["mdd"])
    ts_tiers = json.loads(tier_file.read_text())
    report = _build_report(frequentist_tiers, ts_tiers)
    _write_outputs(cfg, report, tier_data)


def _prepare_inputs(cfg: AppConfig) -> TieringInputs:
    seeds = cfg.analysis.tiering_seeds or [cfg.sim.seed]
    player_counts = sorted({int(n) for n in cfg.sim.n_players_list})
    weights = cfg.analysis.tiering_weights_by_k
    if weights is not None:
        total = sum(weights.values())
        if total > 0:
            weights = {int(k): v / total for k, v in weights.items()}
        else:
            weights = None
    return TieringInputs(
        seeds=[int(s) for s in seeds],
        player_counts=player_counts,
        weights_by_k=weights,
        z_star=float(cfg.analysis.tiering_z_star or 2.0),
    )


def _results_dir_for_seed(cfg: AppConfig, seed: int) -> Path:
    current = cfg.io.results_dir
    name = current.name
    if str(seed) in name:
        return current
    parent = current.parent
    candidate = parent / f"results_seed_{seed}"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(candidate)


def _load_isolated_metrics(cfg: AppConfig, inputs: TieringInputs) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in inputs.seeds:
        try:
            root = _results_dir_for_seed(cfg, seed)
        except FileNotFoundError as exc:
            LOGGER.warning(
                "Skipping seed: results directory not found",
                extra={"stage": "tiering", "seed": seed, "error": str(exc)},
            )
            continue
        seed_cfg = replace(cfg, io=replace(cfg.io, results_dir=root))
        for k in inputs.player_counts:
            try:
                path = build_isolated_metrics(seed_cfg, k)
            except FileNotFoundError:
                LOGGER.warning(
                    "Missing isolated metrics",
                    extra={"stage": "tiering", "seed": seed, "player_count": k},
                )
                continue
            df = pd.read_parquet(path)
            df["seed"] = seed
            frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["strategy", "seed", "n_players", "games", "wins", "win_rate"]
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def _weighted_winrate(df: pd.DataFrame, weights_by_k: Mapping[int, float] | None) -> pd.Series:
    def _agg(group: pd.DataFrame) -> float:
        weights = group["games"].clip(lower=1).astype(float)
        if weights.sum() == 0:
            return group["win_rate"].mean()
        return float(np.average(group["win_rate"], weights=weights))

    per_k = df.groupby(["strategy", "n_players"]).apply(_agg).reset_index(name="rate")
    if weights_by_k:
        per_k["w"] = per_k["n_players"].map(weights_by_k).fillna(0.0)
        per_k["weighted"] = per_k["rate"] * per_k["w"]
        collapsed = per_k.groupby("strategy")["weighted"].sum()
    else:
        collapsed = per_k.groupby("strategy")["rate"].mean()
    return collapsed.sort_values(ascending=False)


def _build_frequentist_tiers(df: pd.DataFrame, mdd: float) -> pd.DataFrame:
    weights = None
    winrates = _weighted_winrate(df, weights)
    tiers: dict[str, int] = {}
    current_tier = 1
    threshold = None
    for strategy, rate in winrates.items():
        if threshold is None:
            threshold = rate - mdd
        elif rate < threshold:
            current_tier += 1
            threshold = rate - mdd
        tiers[strategy] = current_tier
    return pd.DataFrame(
        {
            "strategy": list(tiers.keys()),
            "win_rate": [winrates[s] for s in tiers],
            "mdd_tier": list(tiers.values()),
        }
    )


def _build_report(freq_df: pd.DataFrame, ts_tiers: Mapping[str, int]) -> pd.DataFrame:
    ts_series = pd.Series(ts_tiers, name="trueskill_tier")
    freq_df = freq_df.set_index("strategy")
    report = freq_df.join(ts_series, how="outer")
    report["trueskill_tier"] = report["trueskill_tier"].fillna(report["mdd_tier"].max() + 1).astype(int)
    report["delta_tier"] = report["mdd_tier"] - report["trueskill_tier"]
    min_mdd = report["mdd_tier"].min()
    min_ts = report["trueskill_tier"].min()
    report["in_mdd_top"] = report["mdd_tier"] == min_mdd
    report["in_ts_top"] = report["trueskill_tier"] == min_ts
    return report.reset_index()


def _write_outputs(cfg: AppConfig, report: pd.DataFrame, tier_data: dict) -> None:
    out_csv = cfg.analysis_dir / "tiering_report.csv"
    out_json = cfg.analysis_dir / "tiering_report.json"
    report.sort_values(["mdd_tier", "win_rate"], ascending=[True, False]).to_csv(out_csv, index=False)

    components = tier_data["components"]
    summary = {
        "mdd": tier_data["mdd"],
        "tau2_seed": components.tau2_seed,
        "tau2_sxk": tier_data.get("tau2_sxk", 0.0),
        "R": components.R,
        "K": components.K,
        "total_strategies": int(len(report)),
        "disagreements": int((report["delta_tier"] != 0).sum()),
        "overlap_top": int(((report["in_mdd_top"]) & (report["in_ts_top"])).sum()),
    }
    with atomic_path(str(out_json)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(summary, indent=2))

    LOGGER.info(
        "Tiering report written",
        extra={
            "stage": "tiering",
            "rows": len(report),
            "mdd": tier_data["mdd"],
            "disagreements": summary["disagreements"],
            "path": str(out_csv),
        },
    )
