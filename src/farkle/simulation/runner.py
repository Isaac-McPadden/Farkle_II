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
from farkle.simulation.power_helpers import games_for_power_from_design
from farkle.simulation.run_tournament import METRIC_LABELS, TournamentConfig
from farkle.simulation.simulation import generate_strategy_grid
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils.artifacts import write_parquet_atomic

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


def _resolve_strategies(
    cfg: AppConfig, 
    strategies: list[ThresholdStrategy] | None
    ) -> tuple[list[ThresholdStrategy], int, bool]:
    """
    Returns (strategies_list, grid_size, used_custom_grid: bool).
    """
    if strategies is None:
        strategies, _ = generate_strategy_grid(
            score_thresholds=cfg.sim.score_thresholds,
            dice_thresholds=cfg.sim.dice_thresholds,
            smart_five_opts=cfg.sim.smart_five_opts,
            smart_one_opts=cfg.sim.smart_one_opts,
            consider_score_opts=cfg.sim.consider_score_opts,
            consider_dice_opts=cfg.sim.consider_dice_opts,
            auto_hot_dice_opts=cfg.sim.auto_hot_dice_opts,
            run_up_score_opts=cfg.sim.run_up_score_opts,
            # prefer_score is and must be handled automatically
        )
        used_custom = any([
            cfg.sim.score_thresholds is not None,
            cfg.sim.dice_thresholds is not None,
            cfg.sim.smart_five_opts is not None,
            cfg.sim.smart_one_opts is not None,
            cfg.sim.consider_score_opts not in [(True, False), [True, False], None],
            cfg.sim.consider_dice_opts not in [(True, False), [True, False], None],
            cfg.sim.auto_hot_dice_opts not in [(False, True), [False, True], None],
            cfg.sim.run_up_score_opts not in [(False, True), [False, True], None],
        ])
    else:
        used_custom = True  # caller provided a custom grid explicitly

    grid_size = len(strategies)

    LOGGER.info(
        "Strategy grid prepared: %d strategies (%s grid)",
        grid_size, "custom" if used_custom else "default",
    )
    return strategies, grid_size, used_custom


def _compute_num_shuffles_from_config(
    cfg: AppConfig,
    n_strategies: int,
    n_players: int,
) -> int:
    """
    Precedence:
      1) per-n override
      2) recompute from power_method (if enabled)
      3) static sim.num_shuffles
    """
    # 1) per-n override
    if n_players in cfg.sim.per_n and hasattr(cfg.sim.per_n[n_players], "num_shuffles"):
        n_shuffles = cfg.sim.per_n[n_players].num_shuffles
        LOGGER.info("Using per-n override: n=%d -> num_shuffles=%d", n_players, n_shuffles)
        return n_shuffles

    # 2) recompute via selected method
    if cfg.sim.recompute_num_shuffles:
        method = cfg.sim.power_method  # "bh" | "bonferroni"
        design = cfg.sim.power_design

        n_games_per_strat = games_for_power_from_design(
            n_strategies=n_strategies,
            k_players=n_players,
            method=method,
            design=design,
        )

        endpoint = str(getattr(design, "endpoint", "top1")).lower().replace("-", "_").replace(" ", "_")
        if endpoint == "pairwise":
            m_tests = (n_strategies * (n_strategies - 1)) // 2 if design.full_pairwise else (n_strategies - 1)
            full_pairwise = design.full_pairwise
        else:
            m_tests = n_strategies
            full_pairwise = False
        n_shuffles = n_games_per_strat
        LOGGER.info(
            ("Power recompute: method=%s | endpoint=%s | n_strategies=%d | k_players=%d | m_tests=%d | "
            "power=%.3f | control=%.4g | tail=%s | full_pairwise=%s | use_BY=%s | "
            "detectable_lift=%.4f | baseline_rate=%s -> n_games_per_strat=%d -> num_shuffles=%d"),
            method, endpoint, n_strategies, n_players, m_tests,
            design.power, design.control, design.tail, full_pairwise,
            (bool(design.use_BY) if method == "bh" else False),
            design.detectable_lift, design.baseline_rate, n_games_per_strat, n_shuffles,
        )
        return n_shuffles

    # 3) fallback
    n_shuffles = cfg.sim.num_shuffles
    LOGGER.info("Using configured num_shuffles=%d", n_shuffles)
    return n_shuffles


def run_tournament(cfg: AppConfig) -> int:
    """Top-level dispatcher that runs single-N or multi-N based on the config.

    - If ``sim.n_players_list`` has one element, runs that N and returns total games (int).
    - If it has multiple elements, runs them all and returns the **sum** of total games.
    """
    n_vals = list(cfg.sim.n_players_list)
    if not n_vals:
        raise ValueError("sim.n_players_list must contain at least one player count")

    if len(n_vals) == 1:
        n = n_vals[0]
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
            "n_players_list": n_vals,
            "num_shuffles_default": cfg.sim.num_shuffles,
            "seed": cfg.sim.seed,
            "n_jobs": cfg.sim.n_jobs,
            "expanded_metrics": cfg.sim.expanded_metrics,
        },
    )
    totals = run_multi(cfg)
    return int(sum(totals.values()))


def run_single_n(cfg: AppConfig, n: int, strategies: list[ThresholdStrategy] | None = None) -> int:
    """Run a Farkle tournament for a single tournament with player count *n*."""
    # --- Grid & tests ---
    strategies, grid_size, _used_custom = _resolve_strategies(cfg, strategies)
    n_strategies = grid_size  # used for hypotheses count for power calcs
    LOGGER.info(f"{grid_size} total strategies, used custom state: {_used_custom}")
    # --- Tournament shuffles ---
    n_shuffles = _compute_num_shuffles_from_config(cfg, n_strategies=n_strategies, n_players=n)
    LOGGER.info(f"n_shuffles calculated to be {n_shuffles}")
    # --- Planned totals (log before executing) ---
    games_per_shuffle = grid_size // n
    total_games = n_shuffles * games_per_shuffle 
    LOGGER.info(
        "Planned: %dp games, %d strategies -> %d games/shuffle; %d shuffles; %d total games",
        n, grid_size, games_per_shuffle, n_shuffles, total_games
    )

    # --- Output paths ---
    results_dir = cfg.io.results_dir
    n_dir = results_dir / f"{n}_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = n_dir / f"{n}p_checkpoint.pkl"
    row_dir = None
    if cfg.sim.row_dir:
        row_dir = cfg.sim.row_dir
        if not row_dir.is_absolute():
            row_dir = n_dir / row_dir

    # --- Tournament run ---
    tourn_cfg = TournamentConfig(
        n_players=n,
        num_shuffles=n_shuffles,
        desired_sec_per_chunk=cfg.sim.desired_sec_per_chunk,
        ckpt_every_sec=cfg.sim.ckpt_every_sec,
        n_strategies=grid_size,
    )
    tournament_mod.run_tournament(
        n_players=n,
        global_seed=cfg.sim.seed,
        n_jobs=cfg.sim.n_jobs,
        checkpoint_path=ckpt_path,
        collect_metrics=cfg.sim.expanded_metrics,
        row_output_directory=row_dir,
        num_shuffles=n_shuffles,
        config=tourn_cfg,
        strategies=strategies,
    )

    # --- Final checkpoint post-processing ---
    payload = pickle.loads(ckpt_path.read_bytes())
    raw_counts = payload.get("win_totals", payload)
    if isinstance(raw_counts, Counter):
        win_totals = Counter(raw_counts)
    elif isinstance(raw_counts, Mapping):
        win_totals = Counter({str(k): int(v) for k, v in raw_counts.items()})
    else:
        raise TypeError(f"Unexpected win_totals payload type: {type(raw_counts)}")

    metric_sums: dict[str, dict[str, float]] = payload.get("metric_sums", {})
    metric_sq_sums: dict[str, dict[str, float]] = payload.get("metric_sq_sums", {})

    # (A) Summary parquet
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

    # (B) Expanded metrics parquet
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
            for label in METRIC_LABELS:
                sums_for_label = metric_sums.get(label, {})
                sq_for_label = metric_sq_sums.get(label, {})
                sum_val = sums_for_label.get(str(strat), sums_for_label.get(strat, 0.0))
                sq_val = sq_for_label.get(str(strat), sq_for_label.get(strat, 0.0))
                base[f"sum_{label}"] = float(sum_val)
                base[f"sq_sum_{label}"] = float(sq_val)
                mean_val = (sum_val / wins) if wins > 0 else 0.0
                base[f"mean_{label}"] = float(mean_val)
                if wins > 0:
                    ex2 = sq_val / wins
                    var = max(ex2 - (mean_val ** 2), 0.0)
                else:
                    var = 0.0
                base[f"var_{label}"] = float(var)
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
    # If you want the grid log here too, resolve + log once:
    strategies, grid_size, used_custom = _resolve_strategies(cfg, strategies)

    for n in cfg.sim.n_players_list:
        games = run_single_n(cfg, n, strategies=strategies)
        results[n] = games
    return results


__all__ = ["run_tournament", "run_single_n", "run_multi"]
