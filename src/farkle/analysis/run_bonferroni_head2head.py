# src/farkle/analysis/run_bonferroni_head2head.py
"""Bonferroni-corrected head-to-head comparison of top-tier strategies."""

from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import binomtest

from farkle.analysis.stage_state import stage_done_path, write_stage_done
from farkle.config import AppConfig
from farkle.orchestration.seed_utils import split_seeded_results_dir
from farkle.simulation.simulation import simulate_many_games_from_seeds
from farkle.simulation.strategies import (
    coerce_strategy_ids,
    parse_strategy_for_df,
    parse_strategy_identifier,
)
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.random import MAX_UINT32
from farkle.utils.stats import games_for_power
from farkle.utils.tiers import load_tier_payload, tier_mapping_from_payload
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


def _warn_legacy_stage_dirs(cfg: AppConfig, suffix: str) -> None:
    """Emit a warning when legacy stage directories are present."""

    expected = cfg.stage_layout.folder_for(suffix)
    if expected is None:
        return

    expected_dir = cfg.analysis_dir / expected
    for path in sorted(cfg.analysis_dir.glob(f"*_{suffix}")):
        if path == expected_dir:
            continue
        LOGGER.warning(
            "Legacy stage directory detected; prefer layout-aware helpers",
            extra={"stage": suffix, "legacy_path": str(path), "preferred": str(expected_dir)},
        )


def _tiers_path(cfg: AppConfig) -> Path:
    """Resolve a tiers.json path without requiring optional stages."""

    candidates: list[Path] = []
    for key in ("tiering", "trueskill"):
        folder = cfg.stage_layout.folder_for(key)
        if folder is not None:
            candidates.append(cfg.analysis_dir / folder / "tiers.json")
    candidates.append(cfg.analysis_dir / "tiers.json")

    for path in candidates:
        if path.exists():
            if path == cfg.analysis_dir / "tiers.json" and candidates and candidates[0] != path:
                LOGGER.warning(
                    "Legacy tiers path detected; prefer layout-aware locations",
                    extra={"stage": "head2head", "legacy_path": str(path), "preferred": str(candidates[0])},
                )
            return path

    return candidates[0]


def _load_top_strategies(
    *,
    ratings_path: Path,
    metrics_path: Path,
    ratings_limit: int = 100,
    metrics_limit: int = 100,
) -> tuple[list[str], dict[str, Any]]:
    """Collect top strategies from pooled ratings and metrics tables.

    Strategies are sorted by their respective performance indicators and trimmed to
    the provided limits before being combined into a de-duplicated list. Ordering is
    stable: ratings-first followed by metrics entries not already present.
    """

    def _sorted_from_parquet(path: Path, sort_col: str, limit: int, label: str) -> list[str]:
        if not path.exists():
            LOGGER.warning(
                "Fallback selection skipped: missing %s parquet",
                label,
                extra={"stage": "head2head", "path": str(path)},
            )
            return []

        try:
            df = pd.read_parquet(path, columns=["strategy", sort_col])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Fallback selection skipped: failed to read %s parquet",
                label,
                extra={"stage": "head2head", "path": str(path), "error": str(exc)},
            )
            return []

        if df.empty or sort_col not in df:
            LOGGER.warning(
                "Fallback selection skipped: %s parquet missing data",
                label,
                extra={"stage": "head2head", "path": str(path)},
            )
            return []

        df["strategy"] = coerce_strategy_ids(df["strategy"])
        return [
            str(value)
            for value in df.sort_values(sort_col, ascending=False)["strategy"]
            .head(limit)
            .dropna()
            .tolist()
        ]

    top_by_rating = _sorted_from_parquet(ratings_path, "mu", ratings_limit, "ratings")
    top_by_win_rate = _sorted_from_parquet(metrics_path, "win_rate", metrics_limit, "metrics")

    combined: list[str] = []
    seen: set[str] = set()
    for source in (top_by_rating, top_by_win_rate):
        for strategy in source:
            if strategy not in seen:
                combined.append(strategy)
                seen.add(strategy)

    LOGGER.info(
        "Top-strategy union prepared",
        extra={
            "stage": "head2head",
            "ratings_count": len(top_by_rating),
            "metrics_count": len(top_by_win_rate),
            "combined_count": len(combined),
            "ratings_path": str(ratings_path),
            "metrics_path": str(metrics_path),
        },
    )
    info = {
        "ratings_count": len(top_by_rating),
        "metrics_count": len(top_by_win_rate),
        "combined_count": len(combined),
        "ratings_path": str(ratings_path),
        "metrics_path": str(metrics_path),
    }
    return combined, info


def _count_pair_wins(
    df: pd.DataFrame, strategy_a: int | str, strategy_b: int | str
) -> tuple[int, int]:
    """Count how many times each strategy won within a simulated batch."""
    if "winner_strategy" in df.columns:
        winners = df["winner_strategy"]
        winners = coerce_strategy_ids(winners)
        counts = winners.value_counts()
        return int(counts.get(strategy_a, 0)), int(counts.get(strategy_b, 0))

    seat_column = None
    for candidate in ("winner", "winner_seat"):
        if candidate in df.columns:
            seat_column = candidate
            break
    if seat_column is None:
        raise KeyError("winner_strategy column missing and no winner identifier available")

    seat_numbers = (
        df[seat_column]
        .astype(str)
        .str.extract(r"P(?P<num>\d+)", expand=True)["num"]
        .astype("Int64")
    )
    seat_counts = seat_numbers.value_counts()
    wins_a = int(seat_counts.get(1, 0))
    wins_b = int(seat_counts.get(2, 0))
    return wins_a, wins_b


def _pairwise_schema() -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("players", pa.int64()),
        pa.field("seed", pa.int64()),
        pa.field("pair_id", pa.int64()),
        pa.field("a", pa.string()),
        pa.field("b", pa.string()),
        pa.field("games", pa.int64()),
        pa.field("wins_a", pa.int64()),
        pa.field("wins_b", pa.int64()),
        pa.field("win_rate_a", pa.float64()),
        pa.field("pval_one_sided", pa.float64()),
    ]
    return pa.schema(fields)


def _pairwise_ordered_schema() -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("players", pa.int64()),
        pa.field("seed", pa.int64()),
        pa.field("pair_id", pa.int64()),
        pa.field("a", pa.string()),
        pa.field("b", pa.string()),
        pa.field("ordering", pa.string()),
        pa.field("seat1_strategy", pa.string()),
        pa.field("seat2_strategy", pa.string()),
        pa.field("games", pa.int64()),
        pa.field("wins_a", pa.int64()),
        pa.field("wins_b", pa.int64()),
        pa.field("wins_seat1", pa.int64()),
        pa.field("wins_seat2", pa.int64()),
        pa.field("mean_farkles_seat1", pa.float64()),
        pa.field("mean_farkles_seat2", pa.float64()),
        pa.field("mean_score_seat1", pa.float64()),
        pa.field("mean_score_seat2", pa.float64()),
    ]
    return pa.schema(fields)


def _selfplay_schema() -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("players", pa.int64()),
        pa.field("seed", pa.int64()),
        pa.field("strategy", pa.string()),
        pa.field("games", pa.int64()),
        pa.field("wins_seat1", pa.int64()),
        pa.field("wins_seat2", pa.int64()),
        pa.field("seat1_win_rate", pa.float64()),
        pa.field("seat2_win_rate", pa.float64()),
        pa.field("seat_win_rate_diff", pa.float64()),
        pa.field("mean_farkles_seat1", pa.float64()),
        pa.field("mean_farkles_seat2", pa.float64()),
        pa.field("mean_score_seat1", pa.float64()),
        pa.field("mean_score_seat2", pa.float64()),
    ]
    return pa.schema(fields)


def run_bonferroni_head2head(
    *,
    seed: int = 0,
    root: Path | None = None,
    cfg: AppConfig | None = None,
    n_jobs: int = 1,
    design: Dict[str, Any] | None = None,
    completed_pair_ids: Iterable[int] | None = None,
    shard_dir: Path | None = None,
    shard_size: int = 25,
    progress_schedule: Sequence[float] | None = None,
) -> None:
    """Run pairwise games between top-tier strategies using Bonferroni tests.

    Parameters
    ----------
    seed : int, default ``0``
        Base seed for shuffling the schedule and deterministically assigning
        unique seeds to each simulated game.
    root : Path, optional
        Base results directory containing ``tiers.json``; overrides
        ``cfg.io.results_dir_prefix`` when provided.
    cfg : AppConfig, optional
        Application configuration used to resolve stage-aware directories. A fresh
        default will be created when omitted.
    n_jobs : int, default ``1``
        Number of worker processes; when greater than one, games are simulated in
        parallel and queued in batches so multiple pairs advance concurrently.

    The function reads ``data/tiers.json`` to find strategies in the highest
    tier. It derives a per-pair game budget from :func:`~farkle.utils.stats.games_for_power`
    using the configured design tail for power sizing and writes
    ``analysis/bonferroni_pairwise.parquet``. The output schema is::

        players: int64
        seed: int64
        pair_id: int64
        a: string
        b: string
        games: int64
        wins_a: int64
        wins_b: int64
        win_rate_a: float64
        pval_one_sided: float64

    Each row summarises a pairwise matchup with deterministic seeding; the
    p-value is produced by :func:`scipy.stats.binomtest` using the ``greater``
    alternative.
    """
    cfg = cfg or AppConfig()
    if root is not None:
        base_dir, seed_override = split_seeded_results_dir(Path(root))
        if not base_dir.is_absolute() and base_dir.parts and base_dir.parts[0] == "data":
            base_dir = Path(*base_dir.parts[1:])
        cfg.io.results_dir_prefix = base_dir
        if seed_override is not None:
            cfg.sim.seed = seed_override
    analysis_root = cfg.analysis_dir
    _warn_legacy_stage_dirs(cfg, "head2head")
    manifest = None
    if cfg.sim.n_players_list:
        manifest_path = cfg.strategy_manifest_root_path()
        if manifest_path.exists():
            manifest = pd.read_parquet(manifest_path)

    LOGGER.info(
        "Bonferroni head-to-head start",
        extra={
            "stage": "head2head",
            "root": str(cfg.results_root),
            "seed": seed,
            "n_jobs": n_jobs,
        },
    )

    sub_root = cfg.head2head_stage_dir
    tiers_path = _tiers_path(cfg)

    pairwise_parquet = cfg.head2head_path("bonferroni_pairwise.parquet")
    pairwise_ordered_parquet = cfg.head2head_path("bonferroni_pairwise_ordered.parquet")
    selfplay_parquet = cfg.head2head_path("bonferroni_selfplay_symmetry.parquet")
    default_shards = sub_root / "bonferroni_pairwise_shards"
    legacy_shards = analysis_root / "bonferroni_pairwise_shards"
    shard_dir = shard_dir or (default_shards if default_shards.exists() or not legacy_shards.exists() else legacy_shards)

    if not tiers_path.exists():
        raise RuntimeError(f"Tier file not found at {tiers_path}")

    payload = load_tier_payload(tiers_path)
    prefer_source = payload.get("active") if isinstance(payload, dict) else None
    tiers = tier_mapping_from_payload(payload, prefer=str(prefer_source) if prefer_source else "")
    if not tiers:
        raise RuntimeError(f"No tiers found in {tiers_path}")
    top_val = min(tiers.values())
    tier_elites = [str(s) for s, t in tiers.items() if t == top_val]
    ratings_path = cfg.trueskill_pooled_dir / "ratings_k_weighted.parquet"
    if not ratings_path.exists():
        fallback = cfg.trueskill_stage_dir / "ratings_k_weighted.parquet"
        if fallback.exists():
            LOGGER.warning(
                "Using legacy pooled ratings path; prefer stage-aware pooled directory",
                extra={"stage": "head2head", "legacy_path": str(fallback), "preferred": str(ratings_path)},
            )
            ratings_path = fallback
    metrics_path = cfg.metrics_input_path("metrics.parquet")
    union_strategies, union_info = _load_top_strategies(
        ratings_path=ratings_path,
        metrics_path=metrics_path,
    )
    union_candidates_path = cfg.head2head_stage_dir / "h2h_union_candidates.json"
    union_candidates_payload = {
        "candidates": union_strategies,
        "ratings_count": union_info["ratings_count"],
        "metrics_count": union_info["metrics_count"],
        "combined_count": union_info["combined_count"],
        "ratings_path": union_info["ratings_path"],
        "metrics_path": union_info["metrics_path"],
    }
    union_candidates_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(union_candidates_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(union_candidates_payload, indent=2, sort_keys=True))
    use_tier_elites = bool(getattr(cfg.head2head, "use_tier_elites", False))

    if use_tier_elites:
        elites = list(tier_elites)
        fallback_strategies = union_strategies
        selection_source = "tiers"
    else:
        elites = list(union_strategies)
        fallback_strategies = tier_elites
        selection_source = "union"

    if len(elites) < 2 and fallback_strategies:
        combined: list[str] = []
        seen: set[str] = set()
        for name in (*elites, *fallback_strategies):
            if name not in seen:
                combined.append(name)
                seen.add(name)
        elites = combined
        selection_source = f"{selection_source}+fallback"
        LOGGER.info(
            "Elite selection expanded with fallback strategies",
            extra={
                "stage": "head2head",
                "elite_count": len(elites),
                "tiers_path": str(tiers_path),
                "ratings_path": union_info["ratings_path"],
                "metrics_path": union_info["metrics_path"],
            },
        )
    LOGGER.info(
        "Loaded elite strategies",
        extra={
            "stage": "head2head",
            "selection_source": selection_source,
            "tiers_path": str(tiers_path),
            "tier_elite_count": len(tier_elites),
            "strategies": len(tiers),
            "elite_count": len(elites),
            "ratings_count": union_info["ratings_count"],
            "metrics_count": union_info["metrics_count"],
            "union_count": union_info["combined_count"],
            "ratings_path": union_info["ratings_path"],
            "metrics_path": union_info["metrics_path"],
        },
    )
    design_kwargs = dict(design or {})
    method = design_kwargs.pop("method", "bonferroni")
    full_pairwise = bool(design_kwargs.pop("full_pairwise", True))
    design_kwargs.setdefault("endpoint", "pairwise")
    design_kwargs.setdefault("k_players", 2)
    if method != "bonferroni":
        raise ValueError("Bonferroni head-to-head requires method='bonferroni'")
    if not full_pairwise:
        raise ValueError("Bonferroni head-to-head requires full_pairwise comparisons")
    tail = str(design_kwargs.get("tail", "one_sided")).lower().replace("-", "_")
    if tail not in {"one_sided", "two_sided"}:
        raise ValueError("tail must be 'one_sided' or 'two_sided'")
    design_kwargs["tail"] = tail
    k_players = int(design_kwargs.get("k_players", 2))
    if k_players != 2:
        raise ValueError("Bonferroni head-to-head only supports k_players=2")

    games_per_strategy = games_for_power(
        n_strategies=len(elites),
        method=method,
        full_pairwise=True,
        **design_kwargs,
    )
    safeguard = cfg.head2head.bonferroni_total_games_safeguard
    if safeguard is not None and safeguard > 0:
        estimated_total_games = len(elites) * games_per_strategy / 2
        if estimated_total_games > safeguard:
            reason = (
                "estimated_total_games exceeds bonferroni_total_games_safeguard "
                f"({len(elites)} * {games_per_strategy} / 2 = {estimated_total_games:.0f} > {safeguard})"
            )
            LOGGER.warning(
                "Bonferroni head-to-head skipped: safeguard exceeded",
                extra={
                    "stage": "head2head",
                    "elite_count": len(elites),
                    "games_per_strategy": games_per_strategy,
                    "safeguard": safeguard,
                    "reason": reason,
                },
            )
            inputs: list[Path] = [tiers_path]
            if ratings_path.exists():
                inputs.append(ratings_path)
            if metrics_path.exists():
                inputs.append(metrics_path)
            outputs = [
                pairwise_parquet,
                pairwise_ordered_parquet,
                selfplay_parquet,
                union_candidates_path,
            ]
            if not pairwise_parquet.exists():
                empty_table = pa.Table.from_pylist([], schema=_pairwise_schema())
                write_parquet_atomic(empty_table, pairwise_parquet)
                LOGGER.info(
                    "Bonferroni head-to-head: wrote empty pairwise parquet",
                    extra={"stage": "head2head", "path": str(pairwise_parquet)},
                )
            if not pairwise_ordered_parquet.exists():
                empty_ordered = pa.Table.from_pylist([], schema=_pairwise_ordered_schema())
                write_parquet_atomic(empty_ordered, pairwise_ordered_parquet)
            if not selfplay_parquet.exists():
                empty_selfplay = pa.Table.from_pylist([], schema=_selfplay_schema())
                write_parquet_atomic(empty_selfplay, selfplay_parquet)
            write_stage_done(
                stage_done_path(cfg.head2head_stage_dir, "bonferroni_head2head"),
                inputs=inputs,
                outputs=outputs,
                config_sha=cfg.config_sha,
                status="skipped",
                reason=reason,
            )
            return
    opponents = max(0, len(elites) - 1)
    games_per_pair = (
        0
        if opponents == 0
        else int(math.ceil(games_per_strategy * max(1, k_players - 1) / opponents))
    )
    LOGGER.info(
        "Bonferroni head-to-head sizing",
        extra={
            "stage": "head2head",
            "games_per_strategy": games_per_strategy,
            "games_per_pair": games_per_pair,
            "elite_count": len(elites),
            "k_players": k_players,
        },
    )

    if games_per_pair <= 0:
        LOGGER.info(
            "Bonferroni head-to-head: no games scheduled",
            extra={"stage": "head2head", "elite_count": len(elites)},
        )
        return

    sorted_elites = sorted(elites)
    pair_count = math.comb(len(sorted_elites), 2)
    total_games = games_per_pair * pair_count
    LOGGER.debug(
        "Head-to-head schedule prepared",
        extra={
            "stage": "head2head",
            "pairs": pair_count,
            "games": total_games,
            "games_per_pair": games_per_pair,
        },
    )

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    ordered_records: list[dict[str, Any]] = []
    pending_shard_records: list[dict[str, Any]] = []
    strategies_cache: Dict[int | str, Any] = {}
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_schema = _pairwise_schema()

    def _read_pair_ids_from_parquet(path: Path) -> set[int]:
        if not path.exists():
            return set()
        try:
            df = pd.read_parquet(path, columns=["pair_id"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Failed to load completed pair ids",
                extra={"stage": "head2head", "path": str(path), "error": str(exc)},
            )
            return set()
        if "pair_id" not in df:
            LOGGER.warning(
                "Existing parquet missing pair_id column",
                extra={"stage": "head2head", "path": str(path)},
            )
            return set()
        return {int(pid) for pid in df["pair_id"].dropna().astype(int).tolist()}

    shard_paths = sorted(shard_dir.glob("bonferroni_pairwise_shard_*.parquet"))
    shard_count = len(shard_paths)
    existing_shard_frames: list[pd.DataFrame] = []
    completed_pairs: set[int] = {int(pid) for pid in (completed_pair_ids or ())}
    for shard_path in shard_paths:
        completed_pairs.update(_read_pair_ids_from_parquet(shard_path))
        try:
            df = pd.read_parquet(shard_path)
            existing_shard_frames.append(df)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Failed to load shard data",
                extra={"stage": "head2head", "path": str(shard_path), "error": str(exc)},
            )
    existing_final_pairs = _read_pair_ids_from_parquet(pairwise_parquet)
    completed_pairs.update(existing_final_pairs)

    LOGGER.info(
        "Resuming head-to-head schedule",
        extra={
            "stage": "head2head",
            "precompleted_pairs": len(completed_pairs),
            "shards": shard_count,
        },
    )

    def flush_shard(records_to_flush: List[dict[str, Any]], shard_index: int) -> int:
        if not records_to_flush:
            return shard_index
        shard_path = shard_dir / f"bonferroni_pairwise_shard_{shard_index:04d}.parquet"
        shard_table = pa.Table.from_pylist(records_to_flush, schema=shard_schema)
        write_parquet_atomic(shard_table, shard_path)
        LOGGER.info(
            "Shard written",
            extra={
                "stage": "head2head",
                "rows": shard_table.num_rows,
                "path": str(shard_path),
            },
        )
        existing_shard_frames.append(shard_table.to_pandas())
        records_to_flush.clear()
        return shard_index + 1

    fast_interval = 30.0
    fast_phase_seconds = 600.0
    slow_interval = 600.0
    schedule: Sequence[float] = progress_schedule or [fast_interval, fast_phase_seconds, slow_interval]
    if len(schedule) != 3:
        raise ValueError(
            "progress_schedule must have three values: fast interval, fast phase seconds, slow interval",
        )
    fast_interval, fast_phase_seconds, slow_interval = (float(x) for x in schedule)
    start_time = time.monotonic()
    next_progress = start_time + fast_interval
    fast_phase_end = start_time + fast_phase_seconds

    def maybe_log_progress(completed: int) -> None:
        nonlocal next_progress
        now = time.monotonic()
        if now < next_progress:
            return
        LOGGER.info(
            "Head-to-head progress",
            extra={
                "stage": "head2head",
                "pairs_completed": completed,
                "pairs_total": pair_count,
                "elapsed_seconds": round(now - start_time, 1),
            },
        )
        if now < fast_phase_end:
            next_progress += fast_interval
        else:
            next_progress = max(next_progress, fast_phase_end) + slow_interval

    strategies_cache = {
        name: parse_strategy_identifier(
            name, manifest=manifest, parse_legacy=parse_strategy_for_df
        )
        for name in sorted_elites
    }

    scheduled_pairs: list[tuple[int, int | str, int | str, list[int]]] = []
    processed_pairs = len(completed_pairs)
    for pair_id, (a, b) in enumerate(combinations(sorted_elites, 2)):
        if pair_id in completed_pairs:
            LOGGER.debug(
                "Skipping completed pair",
                extra={"stage": "head2head", "pair_id": pair_id, "strategy_a": a, "strategy_b": b},
            )
            maybe_log_progress(processed_pairs)
            continue
        seeds = rng.integers(0, MAX_UINT32, size=games_per_pair, dtype=np.uint32).tolist()
        LOGGER.debug(
            "Queueing head-to-head batch",
            extra={
                "stage": "head2head",
                "strategy_a": a,
                "strategy_b": b,
                "games": len(seeds),
            },
        )
        if not seeds:
            continue
        scheduled_pairs.append((pair_id, a, b, seeds))

    pair_jobs = max(1, n_jobs)
    if not scheduled_pairs:
        shard_count = flush_shard(pending_shard_records, shard_count)
    else:
        effective_jobs = n_jobs if n_jobs > 0 else 1
        if effective_jobs <= 1:
            worker_count = 1
            pair_jobs = 1
        else:
            worker_count = min(len(scheduled_pairs), max(1, effective_jobs // 2))
            pair_jobs = max(1, effective_jobs // worker_count)

        def simulate_pair(job: tuple[int, int | str, int | str, list[int]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            pair_id, a, b, seeds = job
            split = len(seeds) // 2
            seeds_ab = seeds[:split]
            seeds_ba = seeds[split:]

            def seat_means(df: pd.DataFrame) -> tuple[float, float, float, float]:
                return (
                    float(df["P1_n_farkles"].mean()),
                    float(df["P2_n_farkles"].mean()),
                    float(df["P1_score"].mean()),
                    float(df["P2_score"].mean()),
                )

            df_ab = simulate_many_games_from_seeds(
                seeds=seeds_ab,
                strategies=[strategies_cache[a], strategies_cache[b]],
                n_jobs=pair_jobs,
            )

            df_ba = simulate_many_games_from_seeds(
                seeds=seeds_ba,
                strategies=[strategies_cache[b], strategies_cache[a]],
                n_jobs=pair_jobs,
            )

            (
                mean_farkles_ab_seat1,
                mean_farkles_ab_seat2,
                mean_score_ab_seat1,
                mean_score_ab_seat2,
            ) = seat_means(df_ab)
            (
                mean_farkles_ba_seat1,
                mean_farkles_ba_seat2,
                mean_score_ba_seat1,
                mean_score_ba_seat2,
            ) = seat_means(df_ba)

            wa_ab, wb_ab = _count_pair_wins(df_ab, a, b)
            wb_ba, wa_ba = _count_pair_wins(df_ba, b, a)
            wa = wa_ab + wa_ba
            wb = wb_ab + wb_ba
            games_played = len(seeds)
            if wa + wb != games_played:
                raise RuntimeError(
                    f"Tie or missing outcome detected for pair ({a}, {b}); wins_a={wa} wins_b={wb} games={games_played}",
                )
            pval = binomtest(wa, games_played, alternative="greater").pvalue
            record = {
                "players": k_players,
                "seed": seed,
                "pair_id": pair_id,
                "a": a,
                "b": b,
                "games": games_played,
                "wins_a": wa,
                "wins_b": wb,
                "win_rate_a": wa / games_played if games_played else math.nan,
                "pval_one_sided": pval,
            }
            ordered = [
                {
                    "players": k_players,
                    "seed": seed,
                    "pair_id": pair_id,
                    "a": a,
                    "b": b,
                    "ordering": "a_b",
                    "seat1_strategy": a,
                    "seat2_strategy": b,
                    "games": len(seeds_ab),
                    "wins_a": wa_ab,
                    "wins_b": wb_ab,
                    "wins_seat1": wa_ab,
                    "wins_seat2": wb_ab,
                    "mean_farkles_seat1": mean_farkles_ab_seat1,
                    "mean_farkles_seat2": mean_farkles_ab_seat2,
                    "mean_score_seat1": mean_score_ab_seat1,
                    "mean_score_seat2": mean_score_ab_seat2,
                },
                {
                    "players": k_players,
                    "seed": seed,
                    "pair_id": pair_id,
                    "a": a,
                    "b": b,
                    "ordering": "b_a",
                    "seat1_strategy": b,
                    "seat2_strategy": a,
                    "games": len(seeds_ba),
                    "wins_a": wa_ba,
                    "wins_b": wb_ba,
                    "wins_seat1": wb_ba,
                    "wins_seat2": wa_ba,
                    "mean_farkles_seat1": mean_farkles_ba_seat1,
                    "mean_farkles_seat2": mean_farkles_ba_seat2,
                    "mean_score_seat1": mean_score_ba_seat1,
                    "mean_score_seat2": mean_score_ba_seat2,
                },
            ]
            return record, ordered

        LOGGER.info(
            "Dispatching head-to-head batches",
            extra={
                "stage": "head2head",
                "pending_pairs": len(scheduled_pairs),
                "workers": worker_count,
                "pair_jobs": pair_jobs,
            },
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_job = {
                executor.submit(simulate_pair, job): job for job in scheduled_pairs
            }
            for future in as_completed(future_to_job):
                job_pair_id, job_a, job_b, _ = future_to_job[future]
                record, record_orderings = future.result()
                records.append(record)
                ordered_records.extend(record_orderings)
                pending_shard_records.append(record)
                if len(pending_shard_records) >= shard_size:
                    shard_count = flush_shard(pending_shard_records, shard_count)
                completed_pairs.add(job_pair_id)
                processed_pairs += 1
                maybe_log_progress(processed_pairs)
                LOGGER.debug(
                    "Completed head-to-head batch",
                    extra={
                        "stage": "head2head",
                        "strategy_a": job_a,
                        "strategy_b": job_b,
                        "wins_a": record["wins_a"],
                        "wins_b": record["wins_b"],
                        "pvalue": record["pval_one_sided"],
                    },
                )

    shard_count = flush_shard(pending_shard_records, shard_count)
    all_frames: list[pd.DataFrame] = []
    if records:
        all_frames.append(pd.DataFrame.from_records(records))
    all_frames.extend(existing_shard_frames)
    if pairwise_parquet.exists():
        try:
            all_frames.append(pd.read_parquet(pairwise_parquet))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Failed to load existing pairwise parquet",
                extra={"stage": "head2head", "path": str(pairwise_parquet), "error": str(exc)},
            )
    if not all_frames:
        LOGGER.info(
            "Bonferroni head-to-head: no results to write",
            extra={"stage": "head2head", "elite_count": len(elites)},
        )
        return
    combined_df = (
        pd.concat(all_frames, ignore_index=True)
        .sort_values("pair_id")
        .drop_duplicates(subset=["pair_id"], keep="last")
    )
    pairwise_table = pa.Table.from_pandas(combined_df, schema=shard_schema, preserve_index=False)
    write_parquet_atomic(pairwise_table, pairwise_parquet)

    ordered_schema = _pairwise_ordered_schema()
    ordered_df = pd.DataFrame.from_records(ordered_records)
    if ordered_df.empty:
        ordered_table = pa.Table.from_pylist([], schema=ordered_schema)
    else:
        ordered_table = pa.Table.from_pandas(ordered_df, schema=ordered_schema, preserve_index=False)
    write_parquet_atomic(ordered_table, pairwise_ordered_parquet)

    selfplay_records: list[dict[str, Any]] = []
    for strat in sorted_elites:
        selfplay_seeds = rng.integers(0, MAX_UINT32, size=games_per_pair, dtype=np.uint32).tolist()
        df_self = simulate_many_games_from_seeds(
            seeds=selfplay_seeds,
            strategies=[strategies_cache[strat], strategies_cache[strat]],
            n_jobs=pair_jobs,
        )
        wins_seat1, wins_seat2 = _count_pair_wins(df_self, strat, strat)
        games = len(selfplay_seeds)
        seat1_rate = wins_seat1 / games if games else math.nan
        seat2_rate = wins_seat2 / games if games else math.nan
        mean_farkles_seat1 = float(df_self["P1_n_farkles"].mean())
        mean_farkles_seat2 = float(df_self["P2_n_farkles"].mean())
        mean_score_seat1 = float(df_self["P1_score"].mean())
        mean_score_seat2 = float(df_self["P2_score"].mean())
        selfplay_records.append(
            {
                "players": k_players,
                "seed": seed,
                "strategy": strat,
                "games": games,
                "wins_seat1": wins_seat1,
                "wins_seat2": wins_seat2,
                "seat1_win_rate": seat1_rate,
                "seat2_win_rate": seat2_rate,
                "seat_win_rate_diff": seat1_rate - seat2_rate,
                "mean_farkles_seat1": mean_farkles_seat1,
                "mean_farkles_seat2": mean_farkles_seat2,
                "mean_score_seat1": mean_score_seat1,
                "mean_score_seat2": mean_score_seat2,
            }
        )

    selfplay_schema = _selfplay_schema()
    selfplay_df = pd.DataFrame.from_records(selfplay_records)
    if selfplay_df.empty:
        selfplay_table = pa.Table.from_pylist([], schema=selfplay_schema)
    else:
        selfplay_table = pa.Table.from_pandas(selfplay_df, schema=selfplay_schema, preserve_index=False)
    write_parquet_atomic(selfplay_table, selfplay_parquet)

    LOGGER.info(
        "Bonferroni head-to-head results written",
        extra={
            "stage": "head2head",
            "rows": pairwise_table.num_rows,
            "path": str(pairwise_parquet),
            "ordered_path": str(pairwise_ordered_parquet),
            "selfplay_rows": selfplay_table.num_rows,
            "selfplay_path": str(selfplay_parquet),
        },
    )
