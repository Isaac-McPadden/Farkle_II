"""High level tournament runner using configuration objects.

The :func:`run_tournament` function acts as a thin wrapper around the
lower level helpers found in :mod:`farkle.simulation.run_tournament`.
It accepts an :class:`AppConfig` instance which mirrors the structure
used throughout the project.  Only a tiny subset of the original
configuration surface is supported – just enough for the unit tests –
but additional fields can be added in the future without touching the
public API.
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter
from typing import Mapping

import pyarrow as pa

import farkle.simulation.run_tournament as tournament_mod
from farkle.config import AppConfig
from farkle.simulation.run_tournament import METRIC_LABELS, TournamentConfig
from farkle.simulation.simulation import generate_strategy_grid
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.stats import games_for_power

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


def run_tournament(cfg: AppConfig) -> int:
    """Top-level dispatcher that runs single-N or multi-N based on the config.

    - If ``sim.n_players_list`` has one element, runs that N and returns total games (int).
    - If it has multiple elements, runs them all and returns the **sum** of total games.
    """
    ns = list(cfg.sim.n_players_list)
    if not ns:
        raise ValueError("sim.n_players_list must contain at least one player count")

    if len(ns) == 1:
        n = ns[0]
        LOGGER.info(
            "Running single-N tournament",
            extra={
                "stage": "simulation",
                "n_players": n,
                "num_shuffles_default": cfg.sim.num_shuffles,
                "seed": cfg.sim.seed,
                "n_jobs": cfg.sim.n_jobs,
                "expanded_metrics": cfg.sim.expanded_metrics,
            },
        )
        return run_single_n(cfg, n)

    LOGGER.info(
        "Running multi-N tournaments",
        extra={
            "stage": "simulation",
            "n_players_list": ns,
            "num_shuffles_default": cfg.sim.num_shuffles,
            "seed": cfg.sim.seed,
            "n_jobs": cfg.sim.n_jobs,
            "expanded_metrics": cfg.sim.expanded_metrics,
        },
    )
    totals = run_multi(cfg)
    return int(sum(totals.values()))

def run_single_n(cfg: AppConfig, n: int, grid_size: int | None = None) -> int:
    """Run a Farkle tournament for a single player count *n*."""
    if grid_size is None:
        strategies, _ = generate_strategy_grid(
            score_thresholds = cfg.sim.score_thresholds,
            dice_thresholds = cfg.sim.dice_thresholds,
            smart_five_opts = cfg.sim.smart_five_opts,
            smart_one_opts = cfg.sim.smart_one_opts,
            consider_score_opts = cfg.sim.consider_score_opts,
            consider_dice_opts = cfg.sim.consider_dice_opts,
            auto_hot_dice_opts = cfg.sim.auto_hot_dice_opts,
            run_up_score_opts = cfg.sim.run_up_score_opts,
        )
        grid_size = len(strategies)
    m_tests = grid_size    
    if cfg.sim.bonferroni_design.enabled and cfg.sim.bonferroni_design.recompute:
        n_shuffles = games_for_power(m_tests, method="bonferroni", full_pairwise=True)
    elif cfg.sim.bh_design.enabled and cfg.sim.bh_design.recompute:
        n_shuffles = games_for_power(m_tests, method="bh", full_pairwise=True)
    else:
        if n in cfg.sim.per_n and hasattr(cfg.sim.per_n[n], "num_shuffles"):
            n_shuffles = cfg.sim.per_n[n].num_shuffles
        else:
            n_shuffles = cfg.sim.num_shuffles
    # Update head-to-head games if needed
    if cfg.sim.bonferroni_design.enabled and cfg.sim.bonferroni_design.recompute:
        cfg.head2head.games_per_pair = games_for_power(m_tests, method="bonferroni", full_pairwise=True)
    elif cfg.sim.bh_design.enabled and cfg.sim.bh_design.recompute:
        cfg.head2head.games_per_pair = games_for_power(m_tests, method="bh", full_pairwise=True)
    # Prepare output paths
    results_dir = cfg.io.results_dir
    n_dir = results_dir / f"{n}_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = n_dir / f"{n}p_checkpoint.pkl"
    row_dir = None
    if cfg.sim.row_dir:
        row_dir = cfg.sim.row_dir
        if not row_dir.is_absolute():
            row_dir = n_dir / row_dir
    tourn_cfg = TournamentConfig(
    n_players = n,
    num_shuffles = n_shuffles,
    desired_sec_per_chunk = cfg.sim.desired_sec_per_chunk,
    ckpt_every_sec = cfg.sim.ckpt_every_sec
    )
    tournament_mod.run_tournament(
        n_players=n,
        global_seed=cfg.sim.seed,
        n_jobs=cfg.sim.n_jobs,
        checkpoint_path=ckpt_path,
        collect_metrics=cfg.sim.expanded_metrics,
        row_output_directory=row_dir,
        num_shuffles=n_shuffles,
        config=tourn_cfg
    )
    games_per_shuffle = grid_size // n
    total_games = n_shuffles * games_per_shuffle
    # ---- Final checkpoint handling: always create a parquet summary, and
    # when expanded_metrics=True also create a richer metrics parquet that includes sq-sums. ----
    payload = pickle.loads(ckpt_path.read_bytes())
    # win_totals may be a Counter, dict, or embedded under "win_totals"
    raw_counts = payload.get("win_totals", payload)
    if isinstance(raw_counts, Counter):
        win_totals = Counter(raw_counts)
    elif isinstance(raw_counts, Mapping):
        win_totals = Counter({str(k): int(v) for k, v in raw_counts.items()})
    else:
        raise TypeError(f"Unexpected win_totals payload type: {type(raw_counts)}")

    metric_sums: dict[str, dict[str, float]] = payload.get("metric_sums", {})
    metric_sq_sums: dict[str, dict[str, float]] = payload.get("metric_sq_sums", {})

    # (A) Canonical checkpoint summary parquet: strategy, wins, win_rate, and means if available.
    summary_rows: list[dict[str, float]] = []
    for strat, wins in win_totals.items():
        wins = int(wins)
        if wins < 0:
            continue
        total_games_strat = max(n_shuffles, 1)
        row = {
            "strategy": str(strat),
            "wins": wins,
            "win_rate": wins / total_games_strat,
        }
        if metric_sums:
            for label in METRIC_LABELS:
                s = metric_sums.get(label, {})
                sum_val = s.get(str(strat), s.get(strat, 0.0))
                row[f"mean_{label}"] = (sum_val / wins) if wins > 0 else 0.0
        summary_rows.append(row)

    if summary_rows:
        ckpt_parquet = n_dir / f"{n}p_checkpoint.parquet"
        write_parquet_atomic(pa.Table.from_pylist(summary_rows), ckpt_parquet)

    # (B) Expanded metrics parquet: include raw sums and square sums per label (plus means/vars).
    if cfg.sim.expanded_metrics:
        metrics_rows: list[dict[str, float]] = []
        for strat, wins in win_totals.items():
            wins = int(wins)
            if wins < 0:
                continue
            total_games_strat = max(n_shuffles, 1)
            base = {
                "strategy": str(strat),
                "wins": wins,
                "total_games_strat": total_games_strat,
                "win_rate": wins / total_games_strat,
            }
            # For each tracked label, record sum, sq_sum, mean, and variance (if we have both).
            for label in METRIC_LABELS:
                sums_for_label = metric_sums.get(label, {})
                sq_for_label = metric_sq_sums.get(label, {})
                sum_val = sums_for_label.get(str(strat), sums_for_label.get(strat, 0.0))
                sq_val = sq_for_label.get(str(strat), sq_for_label.get(strat, 0.0))
                base[f"sum_{label}"] = float(sum_val)
                base[f"sq_sum_{label}"] = float(sq_val)
                # Means are typically over wins when metrics are accumulated on winning rows
                mean_val = (sum_val / wins) if wins > 0 else 0.0
                base[f"mean_{label}"] = float(mean_val)
                # Population variance from raw second moment if available; guard negatives due to numeric error
                if wins > 0:
                    ex2 = sq_val / wins
                    var = max(ex2 - (mean_val ** 2), 0.0)
                else:
                    var = 0.0
                base[f"var_{label}"] = float(var)
            # Optional convenience: expected_score over *all* games if present in sums
            ws = metric_sums.get("winning_score", {}).get(str(strat),
                  metric_sums.get("winning_score", {}).get(strat, 0.0))
            base["expected_score"] = (ws / total_games_strat) if total_games_strat > 0 else 0.0
            metrics_rows.append(base)

        if metrics_rows:
            metrics_file = n_dir / f"{n}p_metrics.parquet"
            write_parquet_atomic(pa.Table.from_pylist(metrics_rows), metrics_file)
    return total_games

def run_multi(cfg: AppConfig) -> dict[int, int]:
    """Run tournaments for multiple player counts."""
    results: dict[int, int] = {}
    strategies, _ = generate_strategy_grid()
    grid_size = len(strategies)
    for n in cfg.sim.n_players_list:
        games = run_single_n(cfg, n, grid_size)
        results[n] = games
    return results

__all__ = ["run_tournament", "run_single_n", "run_multi"]
