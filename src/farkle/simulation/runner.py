# src/farkle/simulation/runner.py
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

import hashlib
import logging
import pickle
import shutil
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import pyarrow as pa

import farkle.simulation.run_tournament as tournament_mod
from farkle.config import AppConfig
from farkle.simulation.power_helpers import games_for_power_from_design
from farkle.simulation.run_tournament import METRIC_LABELS, TournamentConfig
from farkle.simulation.simulation import experiment_size, generate_strategy_grid
from farkle.simulation.strategies import (
    STRATEGY_MANIFEST_NAME,
    ThresholdStrategy,
    build_strategy_manifest,
)
from farkle.utils import random as urandom
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.manifest import iter_manifest

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


def _resolve_strategies(
    cfg: AppConfig, strategies: list[ThresholdStrategy] | None
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
            include_stop_at=cfg.sim.include_stop_at,
            include_stop_at_heuristic=cfg.sim.include_stop_at_heuristic,
            # prefer_score is and must be handled automatically
        )
        used_custom = any(
            [
                cfg.sim.score_thresholds is not None,
                cfg.sim.dice_thresholds is not None,
                cfg.sim.smart_five_opts is not None,
                cfg.sim.smart_one_opts is not None,
                cfg.sim.consider_score_opts not in [(True, False), [True, False], None],
                cfg.sim.consider_dice_opts not in [(True, False), [True, False], None],
                cfg.sim.auto_hot_dice_opts not in [(False, True), [False, True], None],
                cfg.sim.run_up_score_opts not in [(False, True), [False, True], None],
                cfg.sim.include_stop_at,
                cfg.sim.include_stop_at_heuristic,
            ]
        )
    else:
        used_custom = True  # caller provided a custom grid explicitly

    grid_size = len(strategies)

    LOGGER.info(
        "Strategy grid prepared: %d strategies (%s grid)",
        grid_size,
        "custom" if used_custom else "default",
    )
    return strategies, grid_size, used_custom


def _smart_option_pairs_from_config(cfg: AppConfig) -> list[tuple[bool, bool]] | None:
    """Return allowed (smart_five, smart_one) combinations from the config."""
    smart_five_opts = cfg.sim.smart_five_opts
    smart_one_opts = cfg.sim.smart_one_opts

    if smart_five_opts is None and smart_one_opts is None:
        return None

    smart_five_values = list(smart_five_opts) if smart_five_opts is not None else [True, False]
    smart_one_values = list(smart_one_opts) if smart_one_opts is not None else [True, False]

    pairs = [
        (bool(sf), bool(so))
        for sf in smart_five_values
        for so in smart_one_values
        if bool(sf) or not bool(so)
    ]
    return pairs


def _grid_size_for_validation(
    cfg: AppConfig,
    strategies: Sequence[ThresholdStrategy] | None,
) -> tuple[int, str]:
    """Return the strategy count and its source label for validation logs."""
    if strategies is not None:
        return len(strategies), "len(strategies)"

    smart_pairs = _smart_option_pairs_from_config(cfg)
    size = experiment_size(
        score_thresholds=cfg.sim.score_thresholds,
        dice_thresholds=cfg.sim.dice_thresholds,
        smart_five_and_one_options=smart_pairs,
        consider_score_opts=cfg.sim.consider_score_opts,
        consider_dice_opts=cfg.sim.consider_dice_opts,
        auto_hot_dice_opts=cfg.sim.auto_hot_dice_opts,
        run_up_score_opts=cfg.sim.run_up_score_opts,
    )
    return size, "experiment_size"


def _filter_player_counts(
    cfg: AppConfig,
    player_counts: Sequence[int],
    strategies: Sequence[ThresholdStrategy] | None = None,
) -> tuple[list[int], list[int], int, str]:
    """Partition player counts into valid/invalid buckets and log the result."""
    grid_size, source = _grid_size_for_validation(cfg, strategies)
    valid: list[int] = []
    invalid: list[int] = []

    for n in player_counts:
        if n <= 0 or (grid_size % n) != 0:
            invalid.append(n)
        else:
            valid.append(n)

    log_extra = {
        "stage": "simulation",
        "grid_size": grid_size,
        "grid_source": source,
        "valid_player_counts": valid,
        "invalid_player_counts": invalid,
    }
    LOGGER.info("Validated player counts against %s=%d", source, grid_size, extra=log_extra)
    if invalid:
        LOGGER.warning("Dropping incompatible player counts: %s", invalid, extra=log_extra)

    return valid, invalid, grid_size, source


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

        endpoint = (
            str(getattr(design, "endpoint", "top1")).lower().replace("-", "_").replace(" ", "_")
        )
        if endpoint == "pairwise":
            m_tests = (
                (n_strategies * (n_strategies - 1)) // 2
                if design.full_pairwise
                else (n_strategies - 1)
            )
            full_pairwise = design.full_pairwise
        else:
            m_tests = n_strategies
            full_pairwise = False
        n_shuffles = n_games_per_strat
        LOGGER.info(
            (
                "Power recompute: method=%s | endpoint=%s | n_strategies=%d | k_players=%d | m_tests=%d | "
                "power=%.3f | control=%.4g | tail=%s | full_pairwise=%s | use_BY=%s | "
                "detectable_lift=%.4f | baseline_rate=%s -> n_games_per_strat=%d -> num_shuffles=%d"
            ),
            method,
            endpoint,
            n_strategies,
            n_players,
            m_tests,
            design.power,
            design.control,
            design.tail,
            full_pairwise,
            (bool(design.use_BY) if method == "bh" else False),
            design.detectable_lift,
            design.baseline_rate,
            n_games_per_strat,
            n_shuffles,
        )
        return n_shuffles

    # 3) fallback
    n_shuffles = cfg.sim.num_shuffles
    LOGGER.info("Using configured num_shuffles=%d", n_shuffles)
    return n_shuffles


def _resolve_row_output_dir(cfg: AppConfig, n_players: int) -> Path | None:
    """Return the per-N row output directory or ``None`` if rows are disabled."""
    return _resolve_per_n_output_dir(cfg, cfg.sim.row_dir, n_players)


def _resolve_metric_chunk_dir(cfg: AppConfig, n_players: int) -> Path | None:
    """Return the per-N metrics chunk directory or ``None`` if disabled."""
    return _resolve_per_n_output_dir(
        cfg,
        getattr(cfg.sim, "metric_chunk_dir", None),
        n_players,
    )


def _resolve_per_n_output_dir(
    cfg: AppConfig,
    raw_value: Path | None,
    n_players: int,
) -> Path | None:
    if not raw_value:
        return None

    raw_str = str(raw_value)
    placeholders = {
        "n": n_players,
        "n_players": n_players,
        "p": f"{n_players}p",
    }
    used_placeholders = False
    try:
        formatted_str = raw_str.format(**placeholders)
        used_placeholders = formatted_str != raw_str
    except KeyError:
        formatted_str = raw_str

    out_path = Path(formatted_str)
    if not used_placeholders:
        tail = out_path.name
        prefix = f"{n_players}p"
        if tail and not tail.startswith(prefix):
            out_path = out_path.parent / f"{prefix}_{tail}"
    if out_path.is_absolute():
        return out_path

    n_dir = cfg.results_root / f"{n_players}_players"
    return n_dir / out_path


def _strategy_manifest_digest(manifest: pd.DataFrame) -> str:
    payload = manifest.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_manifest_matches(manifest: pd.DataFrame, path: Path, *, label: str) -> None:
    if not path.exists():
        return

    existing = pd.read_parquet(path)
    if set(existing.columns) != set(manifest.columns):
        raise ValueError(
            f"{label} manifest schema mismatch at {path}; re-run with --force to regenerate."
        )
    existing = existing[manifest.columns]
    if not existing.equals(manifest):
        raise ValueError(
            f"{label} manifest mismatch at {path}; re-run with --force to regenerate."
        )


def _remove_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _purge_simulation_outputs(
    *,
    n_dir: Path,
    n_players: int,
    ckpt_path: Path,
    row_dir: Path | None,
    metric_chunk_dir: Path | None,
    strategy_manifest_path: Path,
) -> None:
    ckpt_parquet = n_dir / f"{n_players}p_checkpoint.parquet"
    metrics_parquet = n_dir / f"{n_players}p_metrics.parquet"
    _remove_paths([ckpt_path, ckpt_parquet, metrics_parquet, strategy_manifest_path])

    if row_dir is not None and row_dir.exists():
        if row_dir == n_dir:
            _remove_paths(list(row_dir.glob("rows_*.parquet")))
            _remove_paths([row_dir / "manifest.jsonl"])
        else:
            _remove_paths([row_dir])

    if metric_chunk_dir is not None and metric_chunk_dir.exists():
        if metric_chunk_dir == n_dir:
            _remove_paths(list(metric_chunk_dir.glob("metrics_*.parquet")))
            _remove_paths([metric_chunk_dir / "metrics_manifest.jsonl"])
        else:
            _remove_paths([metric_chunk_dir])


def _has_existing_outputs(
    *,
    n_dir: Path,
    n_players: int,
    ckpt_path: Path,
    row_dir: Path | None,
    metric_chunk_dir: Path | None,
    strategy_manifest_path: Path,
) -> bool:
    candidates = [
        ckpt_path,
        n_dir / f"{n_players}p_checkpoint.parquet",
        n_dir / f"{n_players}p_metrics.parquet",
        strategy_manifest_path,
    ]
    if any(path.exists() for path in candidates):
        return True
    if (n_dir / STRATEGY_MANIFEST_NAME).exists():
        return True
    if row_dir is not None and row_dir.exists():
        if (row_dir / "manifest.jsonl").exists():
            return True
        if next(row_dir.glob("rows_*.parquet"), None) is not None:
            return True
        if (row_dir / STRATEGY_MANIFEST_NAME).exists():
            return True
    if metric_chunk_dir is not None and metric_chunk_dir.exists():
        if (metric_chunk_dir / "metrics_manifest.jsonl").exists():
            return True
        if next(metric_chunk_dir.glob("metrics_*.parquet"), None) is not None:
            return True
    return False


def _validate_resume_outputs(
    *,
    cfg: AppConfig,
    n_players: int,
    n_shuffles: int,
    strategies_manifest: pd.DataFrame,
    ckpt_path: Path,
    row_dir: Path | None,
    metric_chunk_dir: Path | None,
) -> None:
    expected_meta = {
        "n_players": n_players,
        "num_shuffles": n_shuffles,
        "global_seed": cfg.sim.seed,
        "n_strategies": len(strategies_manifest),
        "strategy_manifest_sha": _strategy_manifest_digest(strategies_manifest),
    }
    root_manifest_path = cfg.strategy_manifest_root_path()
    legacy_manifest_paths = [ckpt_path.parent / STRATEGY_MANIFEST_NAME]
    if row_dir is not None:
        legacy_manifest_paths.append(row_dir / STRATEGY_MANIFEST_NAME)
    if root_manifest_path.exists():
        _validate_manifest_matches(
            strategies_manifest, root_manifest_path, label="Strategy"
        )
    else:
        legacy_match = None
        for legacy_path in legacy_manifest_paths:
            if legacy_path.exists():
                _validate_manifest_matches(
                    strategies_manifest, legacy_path, label="Strategy"
                )
                legacy_match = legacy_path
                break
        if legacy_match is not None:
            write_parquet_atomic(
                pa.Table.from_pandas(strategies_manifest, preserve_index=False),
                root_manifest_path,
            )

    if ckpt_path.exists():
        payload = pickle.loads(ckpt_path.read_bytes())
        meta = payload.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError(
                f"Checkpoint metadata missing at {ckpt_path}; rerun with --force."
            )
        for key, expected in expected_meta.items():
            if meta.get(key) != expected:
                raise ValueError(
                    f"Checkpoint metadata mismatch for {key} at {ckpt_path}; rerun with --force."
                )
    elif row_dir is None or not (row_dir / "manifest.jsonl").exists():
        if metric_chunk_dir is None or not (metric_chunk_dir / "metrics_manifest.jsonl").exists():
            raise ValueError(
                "Existing outputs found without a checkpoint manifest; rerun with --force."
            )

    expected_seeds = None
    if row_dir is not None:
        manifest_path = row_dir / "manifest.jsonl"
        if manifest_path.exists():
            expected_seeds = {int(s) for s in urandom.spawn_seeds(n_shuffles, seed=cfg.sim.seed)}
            seen: set[int] = set()
            duplicates = 0
            unexpected = 0
            wrong_n = 0
            for record in iter_manifest(manifest_path):
                seed_val = record.get("shuffle_seed")
                if seed_val is not None:
                    try:
                        seed_int = int(seed_val)
                    except (TypeError, ValueError):
                        continue
                    if seed_int in seen:
                        duplicates += 1
                    seen.add(seed_int)
                    if expected_seeds is not None and seed_int not in expected_seeds:
                        unexpected += 1
                n_val = record.get("n_players")
                if n_val is not None and int(n_val) != n_players:
                    wrong_n += 1
            if duplicates:
                raise ValueError(
                    f"Duplicate shuffle entries detected in {manifest_path}; rerun with --force."
                )
            if unexpected or wrong_n:
                raise ValueError(
                    f"Row manifest mismatch at {manifest_path}; rerun with --force."
                )

    if metric_chunk_dir is not None:
        metrics_manifest = metric_chunk_dir / "metrics_manifest.jsonl"
        if metrics_manifest.exists():
            wrong_n = 0
            seen_chunks: set[int] = set()
            duplicates = 0
            for record in iter_manifest(metrics_manifest):
                n_val = record.get("n_players")
                if n_val is not None and int(n_val) != n_players:
                    wrong_n += 1
                chunk_val = record.get("chunk_index")
                if chunk_val is not None:
                    try:
                        chunk_int = int(chunk_val)
                    except (TypeError, ValueError):
                        continue
                    if chunk_int in seen_chunks:
                        duplicates += 1
                    seen_chunks.add(chunk_int)
            if duplicates:
                raise ValueError(
                    f"Duplicate chunk entries detected in {metrics_manifest}; rerun with --force."
                )
            if wrong_n:
                raise ValueError(
                    f"Metrics manifest mismatch at {metrics_manifest}; rerun with --force."
                )


def run_tournament(cfg: AppConfig) -> int:
    """Top-level dispatcher that runs single-N or multi-N based on the config.

    - If ``sim.n_players_list`` has one element, runs that N and returns total games (int).
    - If it has multiple elements, runs them all and returns the **sum** of total games.
    """
    configured_n_vals = list(cfg.sim.n_players_list)
    if not configured_n_vals:
        raise ValueError("sim.n_players_list must contain at least one player count")

    n_vals, invalid_n_vals, grid_size_est, grid_source = _filter_player_counts(
        cfg, configured_n_vals
    )
    if not n_vals:
        raise ValueError(
            f"No valid player counts remain after validating against {grid_source}={grid_size_est}. "
            f"Invalid values: {invalid_n_vals}"
        )

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
    totals = run_multi(cfg, player_counts=n_vals)
    return int(sum(totals.values()))


def run_single_n(
    cfg: AppConfig,
    n: int,
    strategies: list[ThresholdStrategy] | None = None,
    *,
    force: bool = False,
) -> int:
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
        n,
        grid_size,
        games_per_shuffle,
        n_shuffles,
        total_games,
    )

    # --- Output paths ---
    results_dir = cfg.results_root
    n_dir = results_dir / f"{n}_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = n_dir / f"{n}p_checkpoint.pkl"
    resume = not force
    checkpoint_exists = ckpt_path.exists()
    LOGGER.info(
        "Preparing tournament outputs",
        extra={
            "stage": "simulation",
            "n_players": n,
            "checkpoint_path": str(ckpt_path),
            "force": force,
            "resume": resume,
            "checkpoint_exists": checkpoint_exists,
        },
    )
    row_dir = _resolve_row_output_dir(cfg, n)
    metric_chunk_dir = _resolve_metric_chunk_dir(cfg, n)
    manifest = build_strategy_manifest(strategies)
    if force:
        _purge_simulation_outputs(
            n_dir=n_dir,
            n_players=n,
            ckpt_path=ckpt_path,
            row_dir=row_dir,
            metric_chunk_dir=metric_chunk_dir,
            strategy_manifest_path=cfg.strategy_manifest_root_path(),
        )
    elif resume and _has_existing_outputs(
        n_dir=n_dir,
        n_players=n,
        ckpt_path=ckpt_path,
        row_dir=row_dir,
        metric_chunk_dir=metric_chunk_dir,
        strategy_manifest_path=cfg.strategy_manifest_root_path(),
    ):
        _validate_resume_outputs(
            cfg=cfg,
            n_players=n,
            n_shuffles=n_shuffles,
            strategies_manifest=manifest,
            ckpt_path=ckpt_path,
            row_dir=row_dir,
            metric_chunk_dir=metric_chunk_dir,
        )
    if not manifest.empty:
        manifest_path = cfg.strategy_manifest_root_path()
        if not manifest_path.exists():
            write_parquet_atomic(
                pa.Table.from_pandas(manifest, preserve_index=False), manifest_path
            )
        if row_dir is not None and row_dir != n_dir:
            row_dir.mkdir(parents=True, exist_ok=True)
        if LOGGER.isEnabledFor(logging.DEBUG):
            sample = manifest[["strategy_id", "strategy_str"]].head(5).to_dict("records")
            LOGGER.debug(
                "Strategy manifest written",
                extra={
                    "stage": "simulation",
                    "path": str(manifest_path),
                    "sample": sample,
                },
            )

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
        metric_chunk_directory=metric_chunk_dir,
        num_shuffles=n_shuffles,
        config=tourn_cfg,
        strategies=strategies,
        resume=resume,
        checkpoint_metadata={"strategy_manifest_sha": _strategy_manifest_digest(manifest)},
    )

    # --- Final checkpoint post-processing ---
    payload = pickle.loads(ckpt_path.read_bytes())
    raw_counts = payload.get("win_totals", payload)
    if isinstance(raw_counts, Counter):
        win_totals = Counter(raw_counts)
    elif isinstance(raw_counts, Mapping):
        win_totals = Counter(
            {int(k) if str(k).isdigit() else k: int(v) for k, v in raw_counts.items()}
        )
    else:
        raise TypeError(f"Unexpected win_totals payload type: {type(raw_counts)}")

    metric_sums: dict[str, dict[int | str, float]] = payload.get("metric_sums", {})
    metric_sq_sums: dict[str, dict[int | str, float]] = payload.get(
        "metric_square_sums",
        payload.get("metric_sq_sums", {}),
    )

    # (A) Summary parquet
    summary_rows: list[dict[str, float | str]] = []
    for strat, wins in win_totals.items():
        wins = int(wins)
        if wins < 0:
            continue
        total_games_strat = max(n_shuffles, 1)
        row: dict[str, float | str] = {
            "strategy": strat,
            "wins": float(wins),
            "win_rate": wins / total_games_strat,
        }
        if metric_sums:
            for label in METRIC_LABELS:
                s = metric_sums.get(label, {})
                sum_val = s.get(strat, s.get(str(strat), 0.0))
                row[f"mean_{label}"] = (sum_val / wins) if wins > 0 else 0.0
        summary_rows.append(row)

    if summary_rows:
        ckpt_parquet = n_dir / f"{n}p_checkpoint.parquet"
        write_parquet_atomic(pa.Table.from_pylist(summary_rows), ckpt_parquet)

    # (B) Expanded metrics parquet
    if cfg.sim.expanded_metrics:
        metrics_rows: list[dict[str, float | str]] = []
        for strat, wins in win_totals.items():
            wins = int(wins)
            if wins < 0:
                continue
            total_games_strat = max(n_shuffles, 1)
            base: dict[str, float | str] = {
                "strategy": strat,
                "wins": wins,
                "total_games_strat": total_games_strat,
                "win_rate": wins / total_games_strat,
            }
            for label in METRIC_LABELS:
                sums_for_label = metric_sums.get(label, {})
                sq_for_label = metric_sq_sums.get(label, {})
                sum_val = sums_for_label.get(strat, sums_for_label.get(str(strat), 0.0))
                sq_val = sq_for_label.get(strat, sq_for_label.get(str(strat), 0.0))
                base[f"sum_{label}"] = float(sum_val)
                base[f"sq_sum_{label}"] = float(sq_val)
                mean_val = (sum_val / wins) if wins > 0 else 0.0
                base[f"mean_{label}"] = float(mean_val)
                if wins > 0:
                    ex2 = sq_val / wins
                    var = max(ex2 - (mean_val**2), 0.0)
                else:
                    var = 0.0
                base[f"var_{label}"] = float(var)
            ws = metric_sums.get("winning_score", {}).get(
                strat, metric_sums.get("winning_score", {}).get(str(strat), 0.0)
            )
            base["expected_score"] = (ws / total_games_strat) if total_games_strat > 0 else 0.0
            metrics_rows.append(base)

        if metrics_rows:
            metrics_file = n_dir / f"{n}p_metrics.parquet"
            write_parquet_atomic(pa.Table.from_pylist(metrics_rows), metrics_file)

    return total_games


def run_multi(
    cfg: AppConfig,
    player_counts: Sequence[int] | None = None,
    *,
    force: bool = False,
) -> dict[int, int]:
    """Run tournaments for multiple player counts."""
    results: dict[int, int] = {}
    player_counts = (
        list(player_counts) if player_counts is not None else list(cfg.sim.n_players_list)
    )
    strategies, _ = generate_strategy_grid(
        score_thresholds=cfg.sim.score_thresholds,
        dice_thresholds=cfg.sim.dice_thresholds,
        smart_five_opts=cfg.sim.smart_five_opts,
        smart_one_opts=cfg.sim.smart_one_opts,
        consider_score_opts=cfg.sim.consider_score_opts,
        consider_dice_opts=cfg.sim.consider_dice_opts,
        auto_hot_dice_opts=cfg.sim.auto_hot_dice_opts,
        run_up_score_opts=cfg.sim.run_up_score_opts,
        include_stop_at=cfg.sim.include_stop_at,
        include_stop_at_heuristic=cfg.sim.include_stop_at_heuristic,
    )
    # If you want the grid log here too, resolve + log once:
    strategies, grid_size, used_custom = _resolve_strategies(cfg, strategies)

    valid_counts, invalid_counts, _, _ = _filter_player_counts(
        cfg, player_counts, strategies=strategies
    )
    if not valid_counts:
        LOGGER.warning(
            "No valid player counts remain after validating against len(strategies)=%d",
            grid_size,
            extra={
                "stage": "simulation",
                "grid_size": grid_size,
                "grid_source": "len(strategies)",
                "invalid_player_counts": invalid_counts,
                "valid_player_counts": valid_counts,
            },
        )
        return results

    for n in valid_counts:
        games = run_single_n(cfg, n, strategies=strategies, force=force)
        results[n] = games
    return results


__all__ = ["run_tournament", "run_single_n", "run_multi"]
