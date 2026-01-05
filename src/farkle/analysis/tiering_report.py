# src/farkle/analysis/tiering_report.py
"""Generate tiering reports from isolated metrics and meta inputs.

Prepares tiering inputs, computes weighted win rates, and writes consolidated
outputs for both frequentist and TrueSkill-derived tiers across player counts.
"""
from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Hashable, Mapping, cast

import pandas as pd

try:  # pragma: no cover - pandas typing is optional at runtime
    from pandas._typing import Scalar
except ImportError:  # pragma: no cover - fallback for older pandas
    Scalar = Hashable  # type: ignore[assignment]

from farkle.analysis import stage_logger
from farkle.analysis.isolated_metrics import build_isolated_metrics
from farkle.config import AppConfig
from farkle.utils.mdd import tiering_ingredients_from_df
from farkle.utils.tiers import load_tier_payload, tier_mapping_from_payload, write_tier_payload
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


@dataclass
class TieringInputs:
    """Configuration bundle for generating frequentist tier reports."""

    seeds: list[int]
    player_counts: list[int]
    weights_by_k: Mapping[int, float] | None
    z_star: float
    min_gap: float | None


def _tiering_artifact(cfg: AppConfig, name: str) -> Path:
    """Return a tiering-stage path, migrating legacy outputs when present."""

    stage_dir = cfg.tiering_stage_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_path = stage_dir / name
    legacy_path = cfg.analysis_dir / name
    if legacy_path.exists() and not stage_path.exists():
        with contextlib.suppress(Exception):  # pragma: no cover - best-effort migration
            legacy_path.replace(stage_path)
    return stage_path if stage_path.exists() or not legacy_path.exists() else legacy_path


def run(cfg: AppConfig) -> None:
    """Generate a tier comparison report when frequentist analysis is enabled."""
    stage_log = stage_logger("tiering", logger=LOGGER)
    stage_log.start()

    if not getattr(cfg.analysis, "run_frequentist", False):
        stage_log.missing_input("run_frequentist disabled")
        return

    inputs = _prepare_inputs(cfg)
    tier_file = cfg.preferred_tiers_path()
    tier_payload = load_tier_payload(tier_file)
    ts_tiers = tier_mapping_from_payload(tier_payload, prefer="trueskill")
    if not ts_tiers:
        stage_log.missing_input("missing tiers.json", path=str(tier_file))
        return

    df = _load_isolated_metrics(cfg, inputs)
    if df.empty:
        stage_log.missing_input("no isolated metrics available")
        return

    tier_data = tiering_ingredients_from_df(
        df,
        strategy_col="strategy",
        k_col="n_players",
        seed_col="seed",
        wins_col="wins",
        games_col="games",
        winrate_col="win_rate",
        weights_by_k=dict(inputs.weights_by_k) if inputs.weights_by_k is not None else None,
        z_star=inputs.z_star,
    )

    winrates, winrates_by_players = _weighted_winrate(df, inputs.weights_by_k)
    frequentist_tiers = _build_frequentist_tiers(winrates, cast(float, tier_data["mdd"]))
    report = _build_report(frequentist_tiers, ts_tiers)
    _write_outputs(cfg, report, tier_data, inputs)
    _write_consolidated_tiers(
        cfg,
        ts_payload=tier_payload.get("trueskill", {}),
        freq_tiers=frequentist_tiers,
        mdd=cast(float, tier_data["mdd"]),
        weights_by_k=inputs.weights_by_k,
    )
    _write_frequentist_scores(cfg, frequentist_tiers, winrates, winrates_by_players)


def _prepare_inputs(cfg: AppConfig) -> TieringInputs:
    """Normalize tiering configuration values from :class:`AppConfig`."""
    seeds = cfg.analysis.tiering_seeds or [cfg.sim.seed]
    player_counts = sorted({int(n) for n in cfg.sim.n_players_list})
    weights = cfg.analysis.tiering_weights_by_k
    if weights is not None:
        total = sum(weights.values())
        weights = {int(k): v / total for k, v in weights.items()} if total > 0 else None
    return TieringInputs(
        seeds=[int(s) for s in seeds],
        player_counts=player_counts,
        weights_by_k=weights,
        z_star=float(cfg.analysis.tiering_z_star or 1.645),
        min_gap=cfg.analysis.tiering_min_gap,
    )


def _results_dir_for_seed(cfg: AppConfig, seed: int) -> Path:
    """Resolve the results directory for a given seed, validating existence."""
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
    """Load per-seed isolated metrics for the desired player counts."""
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
        return pd.DataFrame(columns=["strategy", "seed", "n_players", "games", "wins", "win_rate"])
    return pd.concat(frames, ignore_index=True, sort=False)


def _weighted_winrate(
    df: pd.DataFrame, weights_by_k: Mapping[int, float] | None
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute overall and per-k weighted win rates."""

    # weights = games clipped to >= 1
    w = df["games"].clip(lower=1).astype(float)

    tmp = df.assign(
        w=w,
        win_x_w=df["win_rate"] * w,
    )

    per_k = (
        tmp.groupby(["strategy", "n_players"], as_index=False)
           .agg(
               win_x_w=("win_x_w", "sum"),
               w=("w", "sum"),
           )
           .assign(win_rate=lambda x: x["win_x_w"] / x["w"])
           .drop(columns=["win_x_w", "w"])
    )

    if weights_by_k:
        per_k = per_k.assign(
            w_k=per_k["n_players"].map(weights_by_k).fillna(0.0),
            weighted=lambda x: x["win_rate"] * x["w_k"],
        )
        collapsed = per_k.groupby("strategy")["weighted"].sum()
    else:
        collapsed = per_k.groupby("strategy")["win_rate"].mean()

    return collapsed.sort_values(ascending=False), per_k



def _build_frequentist_tiers(winrates: pd.Series, mdd: float) -> pd.DataFrame:
    """Assign strategies to tiers using minimum detectable difference."""
    tiers: dict[Hashable, int] = {}
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
            "win_rate": [winrates[cast(Scalar, s)] for s in tiers],
            "mdd_tier": list(tiers.values()),
        }
    )


def _build_report(freq_df: pd.DataFrame, ts_tiers: Mapping[str, int]) -> pd.DataFrame:
    """Join frequentist and TrueSkill tiers, adding disagreement markers."""
    ts_series = pd.Series(ts_tiers, name="trueskill_tier")
    freq_df = freq_df.set_index("strategy")
    report = freq_df.join(ts_series, how="outer")
    report["trueskill_tier"] = (
        report["trueskill_tier"].fillna(report["mdd_tier"].max() + 1).astype(int)
    )
    report["delta_tier"] = report["mdd_tier"] - report["trueskill_tier"]
    min_mdd = report["mdd_tier"].min()
    min_ts = report["trueskill_tier"].min()
    report["in_mdd_top"] = report["mdd_tier"] == min_mdd
    report["in_ts_top"] = report["trueskill_tier"] == min_ts
    return report.reset_index()


def _write_frequentist_scores(
    cfg: AppConfig,
    frequentist_tiers: pd.DataFrame,
    winrates: pd.Series,
    winrates_by_players: pd.DataFrame,
) -> None:
    """Persist per-strategy tier assignments and win rates."""
    if winrates.empty:
        return

    tier_map = frequentist_tiers.set_index("strategy")["mdd_tier"]
    base = winrates_by_players.copy()
    base["players"] = base["n_players"].astype(int)
    base["strategy"] = base["strategy"].astype(str)
    base["strategy_id"] = base["strategy"]
    base["tier"] = base["strategy"].map(tier_map).astype(int)
    base["mdd_tier"] = base["tier"]
    base = base[
        [
            c
            for c in [
                "strategy",
                "strategy_id",
                "players",
                "n_players",
                "win_rate",
                "tier",
                "mdd_tier",
            ]
            if c in base.columns
        ]
    ]

    aggregated = pd.DataFrame(
        {
            "strategy": winrates.index.astype(str),
            "strategy_id": winrates.index.astype(str),
            "players": 0,
            "n_players": 0,
            "win_rate": winrates.values,
            "tier": tier_map.loc[winrates.index].astype(int).values,
            "mdd_tier": tier_map.loc[winrates.index].astype(int).values,
        }
    )

    scores = pd.concat([base, aggregated], ignore_index=True, sort=False)
    scores_path = _tiering_artifact(cfg, "frequentist_scores.parquet")
    with atomic_path(str(scores_path)) as tmp_path:
        scores.to_parquet(tmp_path, index=False)

    LOGGER.info(
        "Frequentist scores written",
        extra={
            "stage": "frequentist",
            "rows": len(scores),
            "path": str(scores_path),
        },
    )


def _write_outputs(
    cfg: AppConfig, report: pd.DataFrame, tier_data: dict, inputs: TieringInputs
) -> None:
    """Write tier comparison outputs to CSV and JSON files."""
    out_csv = _tiering_artifact(cfg, "tiering_report.csv")
    out_json = _tiering_artifact(cfg, "tiering_report.json")
    report.sort_values(["mdd_tier", "win_rate"], ascending=[True, False]).to_csv(
        out_csv, index=False
    )

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
        "trueskill_z_star": inputs.z_star,
        "trueskill_min_gap": inputs.min_gap,
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


def _write_consolidated_tiers(
    cfg: AppConfig,
    *,
    ts_payload: Mapping[str, object] | None,
    freq_tiers: pd.DataFrame,
    mdd: float,
    weights_by_k: Mapping[int, float] | None,
) -> None:
    """Persist unified TrueSkill and frequentist tiers."""

    tiers_path = _tiering_artifact(cfg, "tiers.json")
    freq_map = freq_tiers.set_index("strategy")["mdd_tier"].astype(int).to_dict()
    frequentist_payload: dict[str, object] = {"tiers": freq_map, "mdd": mdd}
    if weights_by_k:
        frequentist_payload["weights_by_k"] = dict(weights_by_k)

    write_tier_payload(
        tiers_path,
        trueskill=ts_payload or None,
        frequentist=frequentist_payload,
    )
