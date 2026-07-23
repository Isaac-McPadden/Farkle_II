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
import json
import logging
import os
import pickle
import shutil
import stat
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np
import pandas as pd
import pyarrow as pa

import farkle.simulation.run_tournament as tournament_mod
from farkle.config import AppConfig
from farkle.simulation.run_tournament import METRIC_LABELS, TournamentConfig
from farkle.simulation.simulation import experiment_size, generate_strategy_grid
from farkle.simulation.strategies import (
    STRATEGY_MANIFEST_NAME,
    ThresholdStrategy,
    build_strategy_manifest,
)
from farkle.simulation.workload_planner import (
    TournamentWorkloadPlan,
    WorkloadCapExceeded,
    plan_tournament_workload,
    write_workload_plan,
)
from farkle.utils import random as urandom
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.manifest import iter_manifest
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION, TOURNAMENT_METHOD_VERSION
from farkle.utils.writer import atomic_path

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


RemovePathCallable: TypeAlias = Callable[..., Any]
RemoveErrorInfo: TypeAlias = tuple[type[BaseException], BaseException, TracebackType]


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


def _plan_workload_from_config(
    cfg: AppConfig,
    n_strategies: int,
    n_players: int,
) -> TournamentWorkloadPlan:
    """Resolve the screening precision workload for one root/player-count cell."""

    return plan_tournament_workload(
        root_seed=cfg.sim.seed,
        k=n_players,
        strategy_count=n_strategies,
        resolution_delta=cfg.screening.resolution_delta,
        confidence=cfg.screening.interval_confidence,
        batch_count=cfg.batching.target_batches,
        min_shuffles_per_batch=cfg.batching.min_shuffles_per_batch,
        shuffle_cap=cfg.screening.max_shuffles_per_root_k,
        projected_games_per_second=cfg.screening.projected_games_per_second,
    )


def _workload_checkpoint_metadata(plan: TournamentWorkloadPlan) -> dict[str, object]:
    """Return precision fields that make replacement checkpoint contracts stale."""

    return {
        "workload_plan_version": plan.plan_version,
        "screening_resolution_delta": plan.resolution_delta,
        "screening_interval_confidence": plan.confidence,
        "batch_count": plan.batch_count,
        "shuffles_per_batch": plan.shuffles_per_batch,
        "batch_construction": plan.batch_construction,
        "coordinate_contract_version": 1,
        "deterministic_batch_size": plan.shuffles_per_batch,
    }


def _resolve_row_output_dir(cfg: AppConfig, n_players: int) -> Path | None:
    """Return the per-N row output directory or ``None`` if rows are disabled."""
    return cfg.simulation_row_dir(n_players)


def _resolve_metric_chunk_dir(cfg: AppConfig, n_players: int) -> Path | None:
    """Return the per-N metrics chunk directory or ``None`` if disabled."""
    return _resolve_per_n_output_dir(
        cfg,
        getattr(cfg.sim, "metric_chunk_dir", None),
        n_players,
    )


def simulation_done_path(cfg: AppConfig, n_players: int) -> Path:
    """Return the completion marker path for a per-N simulation run."""
    n_dir = cfg.results_root / f"{n_players}_players"
    return n_dir / "simulation.done.json"


def simulation_is_complete(cfg: AppConfig, n_players: int) -> bool:
    """Return whether the per-N marker has the current outcome/method identity."""

    done_path = simulation_done_path(cfg, n_players)
    if not done_path.is_file():
        return False
    try:
        payload = json.loads(done_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        payload.get("rng_scheme_version") == urandom.RNG_SCHEME_VERSION
        and payload.get("outcome_schema_version") == OUTCOME_SCHEMA_VERSION
        and payload.get("tournament_method_version") == TOURNAMENT_METHOD_VERSION
    )


def write_simulation_done(
    cfg: AppConfig,
    n_players: int,
    *,
    num_shuffles: int,
    shuffles_per_batch: int,
    n_strategies: int,
    outputs: Sequence[Path],
) -> Path:
    """Write a completion marker for a per-N simulation run."""
    done_path = simulation_done_path(cfg, n_players)
    payload = {
        "n_players": n_players,
        "seed": cfg.sim.seed,
        "root_seed": cfg.sim.seed,
        "k": n_players,
        "num_shuffles": num_shuffles,
        "shuffle_index_start": 0,
        "shuffle_index_end": num_shuffles - 1,
        "deterministic_batch_count": (
            (num_shuffles + shuffles_per_batch - 1) // shuffles_per_batch
        ),
        "shuffles_per_batch": shuffles_per_batch,
        "rng_scheme_version": urandom.RNG_SCHEME_VERSION,
        "rng_purpose_namespace": int(urandom.RandomPurpose.TOURNAMENT_SHUFFLE),
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        "tournament_method_version": TOURNAMENT_METHOD_VERSION,
        "n_strategies": n_strategies,
        "outputs": [str(p) for p in outputs],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    done_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(done_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return done_path


def _resolve_per_n_output_dir(
    cfg: AppConfig,
    raw_value: Path | None,
    n_players: int,
) -> Path | None:
    """Resolve a per-player-count output directory from config or placeholders.

    Args:
        cfg: Application config supplying the results root.
        raw_value: Optional configured directory template or relative path.
        n_players: Player count used to fill placeholders or default prefixes.

    Returns:
        Resolved output directory, or ``None`` when no override is configured.
    """
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
    """Compute a stable digest for the strategy manifest contents.

    Args:
        manifest: Strategy manifest frame to fingerprint.

    Returns:
        SHA-256 digest of the CSV-serialized manifest.
    """
    payload = manifest.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_manifest_matches(manifest: pd.DataFrame, path: Path, *, label: str) -> None:
    """Ensure an on-disk manifest matches the expected in-memory manifest.

    Args:
        manifest: Expected manifest frame for the current run.
        path: Existing manifest parquet path to validate.
        label: Human-readable manifest label for error messages.
    """
    if not path.exists():
        return

    existing = pd.read_parquet(path)
    if set(existing.columns) != set(manifest.columns):
        raise ValueError(
            f"{label} manifest schema mismatch at {path}; re-run with --force to regenerate."
        )
    existing = existing[manifest.columns]
    if not existing.equals(manifest):
        raise ValueError(f"{label} manifest mismatch at {path}; re-run with --force to regenerate.")


def _handle_remove_error(func: RemovePathCallable, path: str, exc_info: RemoveErrorInfo) -> None:
    """Retry a failed path removal after making the target writable.

    Args:
        func: Original removal function provided by ``shutil``.
        path: Path that failed to delete.
        exc_info: Exception tuple reported by the removal callback.
    """
    LOGGER.debug("Retrying removal after chmod for path=%s", path)
    try:
        os.chmod(path, stat.S_IWRITE)
    except OSError as chmod_err:
        LOGGER.exception("Failed to chmod path during removal retry: %s", path)
        raise exc_info[1] from chmod_err
    LOGGER.debug("Calling original remove function for path=%s", path)
    func(path)


def _remove_paths(paths: Sequence[Path]) -> None:
    """Remove files or directories, retrying read-only paths when needed.

    Args:
        paths: Paths to delete if they exist.
    """
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path, onerror=_handle_remove_error)
        else:
            try:
                path.unlink()
            except PermissionError:
                os.chmod(path, stat.S_IWRITE)
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
    """Remove simulation outputs that would conflict with a forced rerun.

    Args:
        n_dir: Base directory for the current player-count outputs.
        n_players: Player count for the output family.
        ckpt_path: Checkpoint path to remove.
        row_dir: Optional row-shard directory.
        metric_chunk_dir: Optional metric-chunk directory.
        strategy_manifest_path: Strategy manifest path to remove.
    """
    done_path = n_dir / "simulation.done.json"
    ckpt_parquet = n_dir / f"{n_players}p_checkpoint.parquet"
    metrics_parquet = n_dir / f"{n_players}p_metrics.parquet"
    _remove_paths([ckpt_path, ckpt_parquet, metrics_parquet, strategy_manifest_path, done_path])

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
) -> bool:
    """Check whether any outputs already exist for a player-count run.

    Args:
        n_dir: Base directory for the current player-count outputs.
        n_players: Player count for the output family.
        ckpt_path: Checkpoint path to inspect.
        row_dir: Optional row-shard directory.
        metric_chunk_dir: Optional metric-chunk directory.

    Returns:
        ``True`` when any checkpoint, manifest, or shard artifacts already exist.
    """
    candidates = [
        ckpt_path,
        n_dir / f"{n_players}p_checkpoint.parquet",
        n_dir / f"{n_players}p_metrics.parquet",
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
    """Validate resume-mode artifacts against the expected run metadata.

    Args:
        cfg: Application config driving the resumed run.
        n_players: Player count being resumed.
        n_shuffles: Expected number of shuffles for the run.
        strategies_manifest: Strategy manifest expected for the run.
        ckpt_path: Checkpoint path whose metadata should align with the manifest.
        row_dir: Optional row-shard directory to validate.
        metric_chunk_dir: Optional metric-chunk directory to validate.
    """
    workload_plan = _plan_workload_from_config(
        cfg,
        n_strategies=len(strategies_manifest),
        n_players=n_players,
    )
    expected_meta = {
        "n_players": n_players,
        "num_shuffles": n_shuffles,
        "global_seed": cfg.sim.seed,
        "n_strategies": len(strategies_manifest),
        "strategy_manifest_sha": _strategy_manifest_digest(strategies_manifest),
        "rng_scheme_version": urandom.RNG_SCHEME_VERSION,
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        "tournament_method_version": TOURNAMENT_METHOD_VERSION,
        "rng_bit_generator": "PCG64DXSM",
        **_workload_checkpoint_metadata(workload_plan),
    }
    root_manifest_path = cfg.strategy_manifest_root_path()
    if root_manifest_path.exists():
        _validate_manifest_matches(strategies_manifest, root_manifest_path, label="Strategy")
    else:
        raise ValueError(
            f"Canonical strategy manifest missing at {root_manifest_path}; rerun with --force."
        )

    has_row_manifest = row_dir is not None and (row_dir / "manifest.jsonl").exists()
    has_metrics_manifest = (
        metric_chunk_dir is not None and (metric_chunk_dir / "metrics_manifest.jsonl").exists()
    )
    if ckpt_path.exists():
        payload = pickle.loads(ckpt_path.read_bytes())
        meta = payload.get("meta")
        if not isinstance(meta, Mapping):
            raise ValueError(f"Checkpoint metadata missing at {ckpt_path}; rerun with --force.")
        for key, expected in expected_meta.items():
            if meta.get(key) != expected:
                raise ValueError(
                    f"Checkpoint metadata mismatch for {key} at {ckpt_path}; rerun with --force."
                )
    elif not has_row_manifest:
        if not has_metrics_manifest:
            if root_manifest_path.exists():
                return
            raise ValueError(
                "Existing outputs found without a checkpoint manifest; rerun with --force."
            )

    expected_seeds = None
    if row_dir is not None:
        manifest_path = row_dir / "manifest.jsonl"
        if manifest_path.exists():
            expected_seed_by_index = {
                index: urandom.coordinate_seed(
                    urandom.RandomPurpose.TOURNAMENT_SHUFFLE,
                    root_seed=cfg.sim.seed,
                    k=n_players,
                    shuffle_index=index,
                    dtype=np.uint32,
                )
                for index in range(n_shuffles)
            }
            expected_seeds = set(expected_seed_by_index.values())
            seen_indices: set[int] = set()
            duplicates = 0
            unexpected = 0
            coordinate_errors = 0
            for record in iter_manifest(manifest_path):
                seed_val = record.get("shuffle_seed")
                index_val = record.get("shuffle_index")
                if seed_val is None or index_val is None:
                    coordinate_errors += 1
                    continue
                try:
                    seed_int = int(seed_val)
                    index_int = int(index_val)
                except (TypeError, ValueError):
                    coordinate_errors += 1
                    continue
                if index_int in seen_indices:
                    duplicates += 1
                seen_indices.add(index_int)
                if (
                    seed_int not in expected_seeds
                    or expected_seed_by_index.get(index_int) != seed_int
                ):
                    unexpected += 1
                expected_batch = index_int // workload_plan.shuffles_per_batch
                if (
                    record.get("root_seed") != cfg.sim.seed
                    or record.get("n_players") != n_players
                    or record.get("deterministic_batch_id") != expected_batch
                    or record.get("rng_scheme_version") != urandom.RNG_SCHEME_VERSION
                    or record.get("rng_purpose_namespace")
                    != int(urandom.RandomPurpose.TOURNAMENT_SHUFFLE)
                    or record.get("outcome_schema_version") != OUTCOME_SCHEMA_VERSION
                    or record.get("tournament_method_version") != TOURNAMENT_METHOD_VERSION
                ):
                    coordinate_errors += 1
            if duplicates:
                raise ValueError(
                    f"Duplicate shuffle entries detected in {manifest_path}; rerun with --force."
                )
            if unexpected or coordinate_errors:
                raise ValueError(f"Row manifest mismatch at {manifest_path}; rerun with --force.")

    if metric_chunk_dir is not None:
        metrics_manifest = metric_chunk_dir / "metrics_manifest.jsonl"
        if metrics_manifest.exists():
            coordinate_errors = 0
            seen_batches: set[int] = set()
            seen_shuffle_indices: set[int] = set()
            duplicates = 0
            for record in iter_manifest(metrics_manifest):
                batch_value = record.get("deterministic_batch_id")
                block_value = record.get("process_block_index")
                start_value = record.get("shuffle_index_start")
                end_value = record.get("shuffle_index_end")
                count_value = record.get("shuffle_count")
                indices_value = record.get("shuffle_indices")
                seeds_value = record.get("shuffle_seeds")
                if (
                    batch_value is None
                    or block_value is None
                    or start_value is None
                    or end_value is None
                    or count_value is None
                    or not isinstance(indices_value, list)
                    or not isinstance(seeds_value, list)
                ):
                    coordinate_errors += 1
                    continue
                try:
                    batch_id = int(batch_value)
                    process_block_index = int(block_value)
                    start = int(start_value)
                    end = int(end_value)
                    count = int(count_value)
                    shuffle_indices = [int(value) for value in indices_value]
                    shuffle_seeds = [int(value) for value in seeds_value]
                except (TypeError, ValueError):
                    coordinate_errors += 1
                    continue
                if batch_id in seen_batches:
                    duplicates += 1
                seen_batches.add(batch_id)
                if seen_shuffle_indices.intersection(shuffle_indices):
                    duplicates += 1
                seen_shuffle_indices.update(shuffle_indices)
                expected_seeds_for_indices = [
                    urandom.coordinate_seed(
                        urandom.RandomPurpose.TOURNAMENT_SHUFFLE,
                        root_seed=cfg.sim.seed,
                        k=n_players,
                        shuffle_index=index,
                        dtype=np.uint32,
                    )
                    for index in shuffle_indices
                ]
                if (
                    record.get("root_seed") != cfg.sim.seed
                    or record.get("n_players") != n_players
                    or record.get("rng_scheme_version") != urandom.RNG_SCHEME_VERSION
                    or record.get("rng_purpose_namespace")
                    != int(urandom.RandomPurpose.TOURNAMENT_SHUFFLE)
                    or record.get("outcome_schema_version") != OUTCOME_SCHEMA_VERSION
                    or record.get("tournament_method_version") != TOURNAMENT_METHOD_VERSION
                    or process_block_index != batch_id + 1
                    or not shuffle_indices
                    or shuffle_indices != sorted(shuffle_indices)
                    or start != shuffle_indices[0]
                    or end != shuffle_indices[-1]
                    or count != len(shuffle_indices)
                    or any(index < 0 or index >= n_shuffles for index in shuffle_indices)
                    or any(
                        index // workload_plan.shuffles_per_batch != batch_id
                        for index in shuffle_indices
                    )
                    or shuffle_seeds != expected_seeds_for_indices
                ):
                    coordinate_errors += 1
            if duplicates:
                raise ValueError(
                    f"Duplicate process-block entries detected in {metrics_manifest}; "
                    "rerun with --force."
                )
            if coordinate_errors:
                raise ValueError(
                    f"Metrics manifest mismatch at {metrics_manifest}; rerun with --force."
                )


def run_tournament(cfg: AppConfig, *, force: bool = False) -> int:
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
                "resolution_delta": cfg.screening.resolution_delta,
                "batch_count": cfg.batching.target_batches,
                "seed": cfg.sim.seed,
                "n_jobs": cfg.sim.n_jobs,
                "expanded_metrics": cfg.sim.expanded_metrics,
            },
        )
        return run_single_n(cfg, n, force=force)

    LOGGER.info(
        "Running multi-N tournaments",
        extra={
            "stage": "simulation",
            "n_players_list": n_vals,
            "resolution_delta": cfg.screening.resolution_delta,
            "batch_count": cfg.batching.target_batches,
            "seed": cfg.sim.seed,
            "n_jobs": cfg.sim.n_jobs,
            "expanded_metrics": cfg.sim.expanded_metrics,
        },
    )
    totals = run_multi(cfg, player_counts=n_vals, force=force)
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
    LOGGER.info(f"{grid_size} total strategies, used custom state: {_used_custom}")
    workload_plan = _plan_workload_from_config(
        cfg,
        n_strategies=grid_size,
        n_players=n,
    )
    n_shuffles = workload_plan.required_shuffles
    total_games = workload_plan.required_games

    # --- Output paths and pre-scheduling plan publication ---
    results_dir = cfg.results_root
    n_dir = results_dir / f"{n}_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    workload_plan_path = n_dir / "simulation_workload_plan.json"
    write_workload_plan(workload_plan_path, workload_plan)
    LOGGER.info(
        "Tournament workload plan resolved",
        extra={"stage": "simulation", **workload_plan.to_dict()},
    )
    if workload_plan.cap_exceeded:
        LOGGER.error(
            "Tournament workload blocked by configured cap",
            extra={"stage": "simulation", **workload_plan.to_dict()},
        )
        raise WorkloadCapExceeded(workload_plan)

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
        progress_logging=cfg.sim.progress_logging,
        n_strategies=grid_size,
        mp_start_method=cfg.sim.mp_start_method,
        deterministic_batch_size=workload_plan.shuffles_per_batch,
    )
    tournament_mod.run_tournament(
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
        checkpoint_metadata={
            "strategy_manifest_sha": _strategy_manifest_digest(manifest),
            "rng_scheme_version": urandom.RNG_SCHEME_VERSION,
            "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
            "tournament_method_version": TOURNAMENT_METHOD_VERSION,
            "rng_bit_generator": "PCG64DXSM",
            **_workload_checkpoint_metadata(workload_plan),
        },
        workload_plan=workload_plan,
        workload_plan_path=workload_plan_path,
    )

    # --- Final checkpoint post-processing ---
    payload = pickle.loads(ckpt_path.read_bytes())
    raw_counts = payload.get("win_totals", payload)
    outcome_payload = payload.get("outcome_counts")
    win_totals = tournament_mod._coerce_counter(
        raw_counts, outcome_payload if isinstance(outcome_payload, Mapping) else None
    )

    metric_sums: dict[str, dict[int | str, float]] = payload.get("metric_sums", {})
    metric_sq_sums: dict[str, dict[int | str, float]] = payload.get(
        "metric_square_sums",
        payload.get("metric_sq_sums", {}),
    )

    # (A) Summary parquet
    summary_rows: list[dict[str, float | str]] = []
    for strat in sorted(win_totals.attempted_exposures, key=str):
        wins = int(win_totals.get(strat, 0))
        attempted = int(win_totals.attempted_exposures[strat])
        if attempted <= 0:
            continue
        completed = int(win_totals.completed_exposures[strat])
        safety_limit = int(win_totals.safety_limit_exposures[strat])
        row: dict[str, float | str] = {
            "strategy": strat,
            "wins": float(wins),
            "attempted_exposures": attempted,
            "completed_exposures": completed,
            "safety_limit_exposures": safety_limit,
            "losses": attempted - wins,
            "win_rate_per_attempt": wins / attempted,
            "win_rate": wins / attempted,
            "win_rate_given_completion": wins / completed if completed else float("nan"),
            "safety_limit_exposure_rate": safety_limit / attempted,
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
        for strat in sorted(win_totals.attempted_exposures, key=str):
            wins = int(win_totals.get(strat, 0))
            attempted = int(win_totals.attempted_exposures[strat])
            if attempted <= 0:
                continue
            completed = int(win_totals.completed_exposures[strat])
            safety_limit = int(win_totals.safety_limit_exposures[strat])
            base: dict[str, float | str] = {
                "strategy": strat,
                "wins": wins,
                "total_games_strat": attempted,
                "attempted_exposures": attempted,
                "completed_exposures": completed,
                "safety_limit_exposures": safety_limit,
                "losses": attempted - wins,
                "win_rate_per_attempt": wins / attempted,
                "win_rate": wins / attempted,
                "win_rate_given_completion": wins / completed if completed else float("nan"),
                "safety_limit_exposure_rate": safety_limit / attempted,
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
            base["expected_score"] = (ws / attempted) if attempted > 0 else 0.0
            metrics_rows.append(base)

        if metrics_rows:
            metrics_file = n_dir / f"{n}p_metrics.parquet"
            write_parquet_atomic(pa.Table.from_pylist(metrics_rows), metrics_file)

    outputs: list[Path] = [ckpt_path, workload_plan_path]
    ckpt_parquet = n_dir / f"{n}p_checkpoint.parquet"
    if ckpt_parquet.exists():
        outputs.append(ckpt_parquet)
    metrics_file = n_dir / f"{n}p_metrics.parquet"
    if metrics_file.exists():
        outputs.append(metrics_file)
    manifest_path = cfg.strategy_manifest_root_path()
    if manifest_path.exists():
        outputs.append(manifest_path)
    if row_dir is not None and row_dir.exists():
        outputs.append(row_dir)
    if metric_chunk_dir is not None and metric_chunk_dir.exists():
        outputs.append(metric_chunk_dir)
    write_simulation_done(
        cfg,
        n,
        num_shuffles=n_shuffles,
        shuffles_per_batch=workload_plan.shuffles_per_batch,
        n_strategies=grid_size,
        outputs=outputs,
    )

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


__all__ = [
    "run_tournament",
    "run_single_n",
    "run_multi",
    "simulation_done_path",
    "simulation_is_complete",
    "write_simulation_done",
]
