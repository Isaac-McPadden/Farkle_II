# src/farkle/analysis/meta.py
"""Pool per-seed strategy summaries into per-player meta estimates.

This module aggregates the ``strategy_summary_{players}p_seed*.parquet`` files
into a pooled win-rate estimate for every strategy.  The meta-analysis uses
inverse-variance weights under a fixed-effects model and can switch to a single
DerSimonian–Laird random-effects variance component whenever the aggregated
I^2 exceeds the configured threshold.

Strategy presence rule
----------------------
A strategy contributes to the pooled estimate **only** if it appears in every
seed summary being combined for the corresponding player count.  The rule is
documented here and enforced before pooling: strategies missing from one or
more seeds are deterministically excluded, which guarantees resumable and
idempotent outputs.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import pandas as pd
import pandas.testing as pdt
import pyarrow as pa

from farkle.analysis import stage_logger
from farkle.analysis.stage_state import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.config import AppConfig
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.stats import wilson_ci
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)

SUMMARY_PATTERN = re.compile(r"strategy_summary_(\d+)p_seed(\d+)\.parquet$")
META_TEMPLATE = "strategy_summary_{players}p_meta.parquet"
META_JSON_TEMPLATE = "meta_{players}p.json"
Z_975 = 1.959963984540054
MIN_PROPORTION = 1e-9
MIN_VARIANCE = 1e-12
POOLED_COLUMNS = ["strategy_id", "players", "win_rate", "se", "ci_lo", "ci_hi", "n_seeds"]


@dataclass
class MetaResult:
    """Meta-analysis outputs paired with pooling diagnostics."""

    pooled: pd.DataFrame
    Q: float
    I2: float
    tau2: float
    method: Literal["fixed", "random"]


def _logit(p: float) -> float:
    """Compute the log-odds while guarding against boundary values."""
    clipped = min(max(p, MIN_PROPORTION), 1.0 - MIN_PROPORTION)
    return math.log(clipped / (1.0 - clipped))


def _inv_logit(x: float) -> float:
    """Convert a log-odds value back to probability space."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _wilson_logit_center(wins: float, games: float) -> float:
    """Return a logit-centered point estimate inside ``(0, 1)``."""

    lo, hi = wilson_ci(int(round(wins)), int(round(games)))
    lo = min(max(lo, MIN_PROPORTION), 1.0 - MIN_PROPORTION)
    hi = min(max(hi, MIN_PROPORTION), 1.0 - MIN_PROPORTION)
    center = 0.5 * (_logit(lo) + _logit(hi))
    return _inv_logit(center)


def _estimate_rate_and_variance(
    wins: float, games: float, win_rate: float | None
) -> tuple[float, float]:
    """Convert raw counts into a rate and variance suitable for pooling."""

    if games <= 0:
        return float("nan"), float("inf")
    if wins < 0 or wins > games:
        raise ValueError("wins must satisfy 0 <= wins <= games")

    rate = wins / games if win_rate is None or math.isnan(win_rate) else float(win_rate)
    rate = min(max(rate, 0.0), 1.0)
    if rate <= 0.0 or rate >= 1.0:
        rate = _wilson_logit_center(wins, games)

    variance = rate * (1.0 - rate) / games
    variance = max(variance, MIN_VARIANCE)
    return rate, variance


def _ensure_strategy_sets(seed_dfs: list[pd.DataFrame]) -> None:
    """Validate that every per-seed DataFrame holds identical strategy IDs."""

    if not seed_dfs:
        return
    strategy_sets = [set(df["strategy_id"].astype(str)) for df in seed_dfs]
    canonical = strategy_sets[0]
    for idx, values in enumerate(strategy_sets[1:], start=1):
        if values != canonical:
            raise ValueError(
                f"Strategy presence mismatch between seed indices 0 and {idx} "
                f"(sizes {len(canonical)} vs {len(values)})"
            )


def pool_winrates(seed_dfs: list[pd.DataFrame], use_random_if_I2_gt: float = 25.0) -> MetaResult:
    """Pool per-seed win rates into per-strategy meta estimates."""

    cleaned: list[pd.DataFrame] = []
    for df in seed_dfs:
        if df is None or df.empty:
            continue
        normalized = df.copy()
        if "strategy_id" in normalized:
            normalized["strategy_id"] = normalized["strategy_id"].astype(str)
        cleaned.append(normalized)
    if not cleaned:
        empty = pd.DataFrame(columns=POOLED_COLUMNS)
        return MetaResult(empty, Q=0.0, I2=0.0, tau2=0.0, method="fixed")

    _ensure_strategy_sets(cleaned)
    players_vals = {int(df["players"].iloc[0]) for df in cleaned}
    if len(players_vals) != 1:
        raise ValueError(f"Expected a single player count; saw {players_vals}")
    players = players_vals.pop()

    strategy_ids = sorted(cleaned[0]["strategy_id"].astype(str).unique())
    per_strategy_obs: dict[str, list[tuple[float, float]]] = {}
    per_strategy_meta: dict[str, dict[str, float]] = {}
    total_Q = 0.0
    total_df = 0
    total_w = 0.0
    total_w_sq = 0.0

    for strategy in strategy_ids:
        obs: list[tuple[float, float]] = []
        for df in cleaned:
            row = df.loc[df["strategy_id"] == strategy]
            if row.empty:
                continue
            wins = float(row["wins"].iloc[0])
            games = float(row["games"].iloc[0])
            win_rate = float(row["win_rate"].iloc[0]) if "win_rate" in row.columns else wins / games
            rate, variance = _estimate_rate_and_variance(wins, games, win_rate)
            obs.append((rate, variance))
        if not obs:
            continue

        weights = [1.0 / v for _, v in obs]
        sum_w = sum(weights)
        if sum_w <= 0.0:
            continue
        sum_w_sq = sum(w**2 for w in weights)
        fixed_mean = sum(w * y for w, (y, _) in zip(weights, obs, strict=False)) / sum_w
        Q = sum(w * (y - fixed_mean) ** 2 for w, (y, _) in zip(weights, obs, strict=False))
        df_count = max(0, len(obs) - 1)

        per_strategy_obs[strategy] = obs
        per_strategy_meta[strategy] = {
            "fixed_mean": fixed_mean,
            "sum_w": sum_w,
            "sum_w_sq": sum_w_sq,
            "n": len(obs),
        }
        total_Q += Q
        total_df += df_count
        total_w += sum_w
        total_w_sq += sum_w_sq

    if not per_strategy_obs:
        empty = pd.DataFrame(columns=POOLED_COLUMNS)
        return MetaResult(empty, Q=0.0, I2=0.0, tau2=0.0, method="fixed")

    tau2 = 0.0
    I2 = 0.0
    method: Literal["fixed", "random"] = "fixed"
    if total_df > 0 and total_Q > 0.0 and total_w > 0.0:
        c = total_w - (total_w_sq / total_w)
        if c > 0.0:
            tau2 = max(0.0, (total_Q - total_df) / c)
        if total_Q > total_df:
            I2 = max(0.0, (total_Q - total_df) / total_Q) * 100.0
        if tau2 > 0.0 and use_random_if_I2_gt < I2:
            method = "random"
        else:
            tau2 = 0.0

    pooled_rows: list[dict[str, float | int]] = []
    for strategy in strategy_ids:
        obs = per_strategy_obs.get(strategy, [])
        if not obs:
            continue
        adjusted_weights = [
            1.0 / (variance + tau2) if method == "random" else 1.0 / variance for _, variance in obs
        ]
        denom = sum(adjusted_weights)
        if denom <= 0.0:
            continue
        pooled_rate = (
            sum(w * rate for w, (rate, _) in zip(adjusted_weights, obs, strict=False)) / denom
        )
        se = math.sqrt(1.0 / denom)
        ci_lo = max(0.0, pooled_rate - Z_975 * se)
        ci_hi = min(1.0, pooled_rate + Z_975 * se)
        pooled_rows.append(
            {
                "strategy_id": strategy,
                "players": players,
                "win_rate": float(pooled_rate),
                "se": float(se),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "n_seeds": per_strategy_meta[strategy]["n"],
            }
        )

    pooled_df = pd.DataFrame(pooled_rows, columns=POOLED_COLUMNS)
    pooled_df.sort_values("strategy_id", inplace=True, kind="mergesort")
    pooled_df.reset_index(drop=True, inplace=True)
    return MetaResult(pooled_df, Q=float(total_Q), I2=float(I2), tau2=float(tau2), method=method)


def _parse_seed_file(path: Path) -> tuple[int, int] | None:
    """Extract player count and seed identifiers from a summary filename."""
    match = SUMMARY_PATTERN.match(path.name)
    if match is None:
        return None
    players = int(match.group(1))
    seed = int(match.group(2))
    return players, seed


def _apply_strategy_presence(
    frames: list[pd.DataFrame],
) -> tuple[list[pd.DataFrame], dict[str, list[int]]]:
    """Filter strategies to those present in every seed-specific summary.

    Args:
        frames: Per-seed dataframes containing ``strategy_id`` columns.

    Returns:
        Tuple of filtered frames and a mapping of missing strategies to the
        seeds where they were absent.
    """
    if not frames:
        return [], {}
    seeds = [int(df["seed"].iloc[0]) for df in frames]
    strategy_sets = [set(df["strategy_id"].astype(str)) for df in frames]
    if not strategy_sets:
        return frames, {}
    common = set.intersection(*strategy_sets) if strategy_sets else set()
    union = set.union(*strategy_sets) if strategy_sets else set()

    missing: dict[str, list[int]] = {}
    for strategy in sorted(union - common):
        missing_seeds = [
            seed for seed, sset in zip(seeds, strategy_sets, strict=False) if strategy not in sset
        ]
        missing[strategy] = missing_seeds

    filtered: list[pd.DataFrame] = []
    for df in frames:
        ids = df["strategy_id"].astype(str)
        mask = ids.isin(common)
        filtered_df = df.copy()
        filtered_df["strategy_id"] = ids
        filtered.append(filtered_df[mask].copy())
    return filtered, missing


def _normalize_meta_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column types and ordering for meta summary comparisons."""
    normalized = df.copy()
    if "strategy_id" in normalized:
        normalized["strategy_id"] = normalized["strategy_id"].astype(str)
    if "players" in normalized:
        normalized["players"] = normalized["players"].astype(int)
    for col in ("win_rate", "se", "ci_lo", "ci_hi"):
        if col in normalized:
            normalized[col] = normalized[col].astype(float)
    if "n_seeds" in normalized:
        normalized["n_seeds"] = normalized["n_seeds"].astype(int)
    normalized.sort_values("strategy_id", inplace=True, kind="mergesort")
    normalized.reset_index(drop=True, inplace=True)
    return normalized


def _parquet_matches(path: Path, new_df: pd.DataFrame) -> bool:
    """Check whether an existing parquet already matches the proposed content."""
    if not path.exists():
        return False
    try:
        existing = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return False
    new_norm = _normalize_meta_frame(new_df)
    existing_norm = _normalize_meta_frame(existing)
    if list(existing_norm.columns) != list(new_norm.columns):
        return False
    try:
        pdt.assert_frame_equal(existing_norm, new_norm, check_dtype=True, check_exact=True)
        return True
    except AssertionError:
        return False


def _write_json_atomic(payload: Mapping[str, float | str], path: Path) -> None:
    """Write a JSON file atomically to avoid partial outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def _collect_seed_summaries(cfg: AppConfig) -> dict[int, dict[int, Path]]:
    """Collect per-seed summary files keyed by player count and seed."""

    search_dirs: list[Path] = []
    stage_dir = cfg.stage_dir_if_active("seed_summaries")
    if stage_dir is not None:
        search_dirs.append(stage_dir)
    search_dirs.append(cfg.meta_analysis_dir)
    if cfg.analysis_dir not in search_dirs:
        search_dirs.append(cfg.analysis_dir)

    files_by_players: dict[int, dict[int, Path]] = {}
    for base in search_dirs:
        if not base.exists():
            continue
        for path in sorted(base.rglob("strategy_summary_*p_seed*.parquet")):
            parsed = _parse_seed_file(path)
            if parsed is None:
                continue
            players, seed = parsed
            files_by_players.setdefault(players, {})
            files_by_players[players].setdefault(seed, path)
    return files_by_players


def _select_seed_entries(
    entries: list[tuple[int, Path]],
    primary_seed: int,
    *,
    max_other_seeds: int | None,
    comparison_seed: int | None,
) -> list[tuple[int, Path]]:
    """Select which seed files participate in pooling.

    Always prefers the primary seed when present, then either a fixed comparison
    seed or a deterministic random subset of the remaining seeds.
    """

    by_seed = dict(entries)
    selected: list[tuple[int, Path]] = []

    primary_path = by_seed.pop(primary_seed, None)
    if primary_path is not None:
        selected.append((primary_seed, primary_path))

    if comparison_seed is not None:
        comparison_path = by_seed.get(comparison_seed)
        if comparison_path is not None:
            selected.append((comparison_seed, comparison_path))
        return sorted(selected)

    remaining = list(by_seed.items())
    if max_other_seeds is None:
        selected.extend(remaining)
    else:
        rng = random.Random(primary_seed)
        rng.shuffle(remaining)
        selected.extend(remaining[: max_other_seeds])

    return sorted(selected)


def run(cfg: AppConfig, *, force: bool = False, use_random_if_I2_gt: float | None = None) -> None:
    """Materialize per-player meta summaries."""

    stage_log = stage_logger("meta", logger=LOGGER)
    stage_log.start()

    threshold = use_random_if_I2_gt
    if threshold is None:
        threshold = getattr(cfg.analysis, "meta_random_if_I2_gt", 25.0)

    max_other_seeds = getattr(cfg.analysis, "meta_max_other_seeds", None)
    comparison_seed = getattr(cfg.analysis, "meta_comparison_seed", None)
    files_by_players = _collect_seed_summaries(cfg)

    if not files_by_players:
        stage_log.missing_input("no per-seed summaries found")
        return

    done = stage_done_path(cfg.meta_stage_dir, "meta")
    inputs = sorted({path for entries in files_by_players.values() for path in entries.values()})
    eligible_players = [players for players, entries in files_by_players.items() if len(entries) > 1]
    expected_outputs = [
        cfg.meta_output_path(players, META_TEMPLATE.format(players=players))
        for players in eligible_players
    ] + [cfg.meta_output_path(players, META_JSON_TEMPLATE.format(players=players)) for players in eligible_players]

    if not force and inputs and expected_outputs and stage_is_up_to_date(
        done, inputs=inputs, outputs=expected_outputs, config_sha=cfg.config_sha
    ):
        LOGGER.info("Meta outputs up-to-date", extra={"stage": "meta", "stamp": str(done)})
        return

    outputs: list[Path] = []

    for players, entries in sorted(files_by_players.items()):
        selected_entries = _select_seed_entries(
            sorted(entries.items()),
            cfg.sim.seed,
            max_other_seeds=max_other_seeds,
            comparison_seed=comparison_seed,
        )
        selected_seed_ids = [seed for seed, _path in selected_entries]
        if (
            max_other_seeds is not None
            or comparison_seed is not None
            or len(selected_seed_ids) != len(entries)
        ):
            LOGGER.info(
                "Meta seed selection",
                extra={
                    "stage": "meta",
                    "players": players,
                    "selected_seeds": selected_seed_ids,
                    "available_seeds": sorted(entries),
                },
            )
        if len(selected_entries) <= 1:
            LOGGER.info(
                "Meta pooling skipped: requires multiple seeds",
                extra={
                    "stage": "meta",
                    "players": players,
                    "available_seeds": sorted(entries),
                },
            )
            continue

        frames: list[pd.DataFrame] = []
        for _seed, path in selected_entries:
            df = pd.read_parquet(path)
            if df.empty:
                continue
            frames.append(df)
        if not frames:
            continue
        if LOGGER.isEnabledFor(logging.DEBUG):
            seed_sets: list[dict[str, object]] = []
            for df in frames:
                seed = int(df["seed"].iloc[0]) if "seed" in df.columns else None
                ids = sorted(df["strategy_id"].astype(str).unique().tolist())
                seed_sets.append(
                    {
                        "seed": seed,
                        "count": len(ids),
                        "sample": ids[:5],
                    }
                )
            LOGGER.debug(
                "Meta pooling strategy sets",
                extra={
                    "stage": "meta",
                    "players": players,
                    "sets": seed_sets,
                },
            )
        frames, missing = _apply_strategy_presence(frames)
        if not frames or frames[0].empty:
            LOGGER.info(
                "Meta pooling skipped: no common strategies",
                extra={"stage": "meta", "players": players},
            )
            continue
        if LOGGER.isEnabledFor(logging.DEBUG):
            common_ids = sorted(frames[0]["strategy_id"].astype(str).unique().tolist())
            LOGGER.debug(
                "Meta pooling common strategies",
                extra={
                    "stage": "meta",
                    "players": players,
                    "count": len(common_ids),
                    "sample": common_ids[:5],
                },
            )
        if missing:
            LOGGER.info(
                "Strategy presence pruning",
                extra={
                    "stage": "meta",
                    "players": players,
                    "dropped_strategies": len(missing),
                },
            )

        result = pool_winrates(frames, use_random_if_I2_gt=threshold)
        if result.pooled.empty:
            LOGGER.info(
                "Meta pooling skipped: pooled frame empty",
                extra={"stage": "meta", "players": players},
            )
            continue

        parquet_path = cfg.meta_output_path(players, META_TEMPLATE.format(players=players))
        outputs.append(parquet_path)
        if force or not _parquet_matches(parquet_path, result.pooled):
            table = pa.Table.from_pandas(result.pooled, preserve_index=False)
            write_parquet_atomic(table, parquet_path)
        else:
            LOGGER.info(
                "Meta parquet already up-to-date",
                extra={"stage": "meta", "players": players, "path": str(parquet_path)},
            )

        json_payload = {
            "Q": result.Q,
            "I2": result.I2,
            "tau2": result.tau2,
            "method": result.method,
        }
        json_path = cfg.meta_output_path(players, META_JSON_TEMPLATE.format(players=players))
        outputs.append(json_path)
        if force or not json_path.exists():
            _write_json_atomic(json_payload, json_path)
        else:
            try:
                existing = json.loads(json_path.read_text())
            except Exception:  # noqa: BLE001
                _write_json_atomic(json_payload, json_path)
            else:
                if existing != json_payload:
                    _write_json_atomic(json_payload, json_path)

        LOGGER.info(
            "Meta pooling complete",
            extra={
                "stage": "meta",
                "players": players,
                "strategies": len(result.pooled),
                "method": result.method,
            },
        )

    if outputs:
        write_stage_done(done, inputs=inputs, outputs=outputs, config_sha=cfg.config_sha)


__all__ = ["MetaResult", "pool_winrates", "run"]
