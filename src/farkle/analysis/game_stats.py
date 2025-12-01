# src/farkle/analysis/game_stats.py
"""Compute descriptive statistics for game lengths and victory margins.

Reads curated row-level parquet files (per-``n`` shards and the combined
superset) and aggregates the ``n_rounds`` column. Outputs both per-strategy
statistics and a small global summary grouped by ``n_players``.

The module also flags close margins and multi-target games, emitting
``analysis/rare_events.parquet`` with per-game records plus aggregated
frequencies per strategy and player-count cohort.

The module also derives per-game ``margin_of_victory`` from seat-level scores
and writes ``analysis/margin_stats.parquet`` with per-``(strategy, n_players)``
summaries. Margin schema:

``summary_level``
    Literal "strategy" for compatibility with ``game_length.parquet``.
``strategy``
    Strategy string taken from ``P#_strategy`` columns.
``n_players``
    Player count inferred from the shard path.
``observations``
    Number of games with at least two valid seat scores.
``mean_margin`` / ``median_margin`` / ``std_margin``
    Descriptive statistics over ``margin_of_victory``.
``prob_margin_le_500`` / ``prob_margin_le_1000``
    Close-game shares for the given thresholds.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from farkle.config import AppConfig
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.writer import atomic_path
from farkle.utils.schema_helpers import n_players_from_schema

LOGGER = logging.getLogger(__name__)


def run(cfg: AppConfig, *, force: bool = False) -> None:
    """Compute game statistics and write them to parquet outputs.

    Args:
        cfg: Application configuration used to resolve file locations.
        force: When True, recompute even if the outputs appear up-to-date.
    """

    analysis_dir = cfg.analysis_dir
    game_length_output = analysis_dir / "game_length.parquet"
    margin_output = analysis_dir / "margin_stats.parquet"
    rare_events_output = analysis_dir / "rare_events.parquet"
    stamp_path = analysis_dir / "game_length.done.json"

    per_n_inputs = _discover_per_n_inputs(cfg)
    combined_path = cfg.curated_parquet
    input_paths: list[Path] = [p for _, p in per_n_inputs]
    if combined_path.exists():
        input_paths.append(combined_path)

    if not input_paths:
        raise FileNotFoundError(
            "game-stats: no curated parquet files found under analysis/data"
        )

    outputs = [game_length_output, margin_output, rare_events_output]
    if not force and _is_up_to_date(stamp_path, inputs=input_paths, outputs=outputs):
        LOGGER.info(
            "Game-length stats up-to-date",
            extra={
                "stage": "game_stats",
                "game_length_output": str(game_length_output),
                "margin_output": str(margin_output),
                "stamp": str(stamp_path),
            },
        )
        return

    LOGGER.info(
        "Computing game-length stats",
        extra={
            "stage": "game_stats",
            "analysis_dir": str(analysis_dir),
            "game_length_output": str(game_length_output),
            "margin_output": str(margin_output),
            "force": force,
        },
    )

    strategy_stats = _per_strategy_stats(per_n_inputs)
    global_stats = _global_stats(combined_path) if combined_path.exists() else pd.DataFrame()

    combined = pd.concat([strategy_stats, global_stats], ignore_index=True)
    if combined.empty:
        raise RuntimeError("game-stats: no rows available to summarize")

    table = pa.Table.from_pandas(combined, preserve_index=False)
    write_parquet_atomic(table, game_length_output, codec=cfg.parquet_codec)

    margin_stats = _per_strategy_margin_stats(
        per_n_inputs, thresholds=cfg.game_stats_margin_thresholds
    )
    if margin_stats.empty:
        raise RuntimeError("game-stats: no margins available to summarize")

    margin_table = pa.Table.from_pandas(margin_stats, preserve_index=False)
    write_parquet_atomic(margin_table, margin_output, codec=cfg.parquet_codec)

    rare_events = _rare_event_flags(
        per_n_inputs,
        thresholds=cfg.game_stats_margin_thresholds,
        target_score=cfg.rare_event_target_score,
    )
    if rare_events.empty:
        raise RuntimeError("game-stats: no rare events available to summarize")

    rare_events_table = pa.Table.from_pandas(rare_events, preserve_index=False)
    write_parquet_atomic(rare_events_table, rare_events_output, codec=cfg.parquet_codec)
    _write_stamp(stamp_path, inputs=input_paths, outputs=outputs)

    LOGGER.info(
        "Game-length stats written",
        extra={
            "stage": "game_stats",
            "rows": len(combined),
            "game_length_output": str(game_length_output),
            "margin_output": str(margin_output),
        },
    )


def _discover_per_n_inputs(cfg: AppConfig) -> list[tuple[int, Path]]:
    """Return discovered per-``n`` curated parquets in ``analysis/data``."""

    inputs: list[tuple[int, Path]] = []
    for p_dir in sorted(cfg.data_dir.glob("*p")):
        if not p_dir.is_dir():
            continue
        try:
            n_players = int(p_dir.name.removesuffix("p"))
        except ValueError:
            continue

        preferred = p_dir / cfg.curated_rows_name
        legacy = p_dir / f"{n_players}p_ingested_rows.parquet"
        candidate = preferred if preferred.exists() else legacy
        if candidate.exists():
            inputs.append((n_players, candidate))
    return inputs


def _per_strategy_stats(per_n_inputs: Sequence[tuple[int, Path]]) -> pd.DataFrame:
    """Compute statistics grouped by strategy and player count."""

    rows: list[pd.Series] = []
    for n_players, path in per_n_inputs:
        ds_in = ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        if not strategy_cols:
            LOGGER.warning(
                "Per-N parquet missing strategy columns",
                extra={"stage": "game_stats", "path": str(path)},
            )
            continue

        columns = ["n_rounds", *strategy_cols]
        tbl = ds_in.to_table(columns=columns)
        df = tbl.to_pandas()
        melted = df.melt(id_vars=["n_rounds"], value_vars=strategy_cols, value_name="strategy")
        melted = melted.dropna(subset=["strategy"])
        melted["n_players"] = n_players

        grouped = melted.groupby(["strategy", "n_players"], sort=False)["n_rounds"]
        for (strategy, players), rounds in grouped:
            stats = _summarize_rounds(rounds)
            stats.update(
                {
                    "summary_level": "strategy",
                    "strategy": strategy,
                    "n_players": players,
                }
            )
            rows.append(pd.Series(stats))

    if not rows:
        return pd.DataFrame(
            columns=[
                "summary_level",
                "strategy",
                "n_players",
                "observations",
                "mean_rounds",
                "median_rounds",
                "std_rounds",
                "p10_rounds",
                "p50_rounds",
                "p90_rounds",
                "prob_rounds_le_5",
                "prob_rounds_le_10",
                "prob_rounds_ge_20",
            ]
        )

    return pd.DataFrame(rows)


def _per_strategy_margin_stats(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
) -> pd.DataFrame:
    """Compute victory-margin statistics grouped by strategy and player count."""

    rows: list[pd.Series] = []
    for n_players, path in per_n_inputs:
        ds_in = ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        score_cols = [name for name in ds_in.schema.names if name.startswith("P") and name.endswith("_score")]

        if not strategy_cols:
            LOGGER.warning(
                "Per-N parquet missing strategy columns",
                extra={"stage": "game_stats", "path": str(path)},
            )
            continue

        if not score_cols:
            LOGGER.warning(
                "Per-N parquet missing seat score columns; skipping margins",
                extra={"stage": "game_stats", "path": str(path)},
            )
            continue

        columns = [*score_cols, *strategy_cols]
        tbl = ds_in.to_table(columns=columns)
        df = tbl.to_pandas()
        df["margin_of_victory"] = _compute_margins(df, score_cols)

        melted = df.melt(
            id_vars=["margin_of_victory"],
            value_vars=strategy_cols,
            value_name="strategy",
        )
        melted = melted.dropna(subset=["strategy", "margin_of_victory"])
        melted["n_players"] = n_players

        grouped = melted.groupby(["strategy", "n_players"], sort=False)["margin_of_victory"]
        for (strategy, players), margins in grouped:
            stats = _summarize_margins(margins, thresholds)
            stats.update(
                {
                    "summary_level": "strategy",
                    "strategy": strategy,
                    "n_players": players,
                }
            )
            rows.append(pd.Series(stats))

    if not rows:
        columns = [
            "summary_level",
            "strategy",
            "n_players",
            "observations",
            "mean_margin",
            "median_margin",
            "std_margin",
            *[f"prob_margin_le_{thr}" for thr in thresholds],
        ]
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows)


def _rare_event_flags(
    per_n_inputs: Sequence[tuple[int, Path]],
    *,
    thresholds: Sequence[int],
    target_score: int,
) -> pd.DataFrame:
    """Compute per-game rare events and aggregate rates."""

    game_rows: list[pd.Series] = []
    for n_players, path in per_n_inputs:
        ds_in = ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        score_cols = [name for name in ds_in.schema.names if name.startswith("P") and name.endswith("_score")]

        if not strategy_cols:
            LOGGER.warning(
                "Per-N parquet missing strategy columns; skipping rare events",
                extra={"stage": "game_stats", "path": str(path)},
            )
            continue

        if not score_cols:
            LOGGER.warning(
                "Per-N parquet missing seat score columns; skipping rare events",
                extra={"stage": "game_stats", "path": str(path)},
            )
            continue

        columns = ["n_rounds", *strategy_cols, *score_cols]
        tbl = ds_in.to_table(columns=columns)
        df = tbl.to_pandas()

        margins = _compute_margins(df, score_cols)
        scores = df.loc[:, score_cols].apply(pd.to_numeric, errors="coerce")
        multi_target = (scores >= target_score).sum(axis=1) >= 2

        event_df = pd.DataFrame(
            {
                "summary_level": "game",
                "n_players": n_players,
                "margin_of_victory": margins,
                "multi_reached_target": multi_target,
            }
        )

        for thr in thresholds:
            event_df[f"margin_le_{thr}"] = event_df["margin_of_victory"] <= thr

        melted = event_df.join(df[strategy_cols])
        melted = melted.melt(
            id_vars=[
                "summary_level",
                "n_players",
                "margin_of_victory",
                "multi_reached_target",
                *[f"margin_le_{thr}" for thr in thresholds],
            ],
            value_vars=strategy_cols,
            value_name="strategy",
        )
        melted = melted.dropna(subset=["strategy"])
        melted["observations"] = 1
        melted = melted.drop(columns=["variable"])
        game_rows.extend(melted.to_dict(orient="records"))

    if not game_rows:
        columns = [
            "summary_level",
            "strategy",
            "n_players",
            "margin_of_victory",
            "multi_reached_target",
            "observations",
            *[f"margin_le_{thr}" for thr in thresholds],
        ]
        return pd.DataFrame(columns=columns)

    game_df = pd.DataFrame(game_rows)
    flag_cols = ["multi_reached_target", *[f"margin_le_{thr}" for thr in thresholds]]

    strategy_summary = (
        game_df.groupby(["strategy", "n_players"], sort=False)[flag_cols]
        .mean()
        .reset_index()
    )
    strategy_summary.insert(0, "summary_level", "strategy")
    strategy_summary["observations"] = (
        game_df.groupby(["strategy", "n_players"], sort=False)[flag_cols[0]].count().values
    )
    strategy_summary["margin_of_victory"] = pd.NA

    global_summary = game_df.groupby("n_players", sort=False)[flag_cols].mean().reset_index()
    global_summary.insert(0, "summary_level", "n_players")
    global_summary.insert(1, "strategy", pd.NA)
    global_summary["observations"] = (
        game_df.groupby("n_players", sort=False)[flag_cols[0]].count().values
    )
    global_summary["margin_of_victory"] = pd.NA

    combined = pd.concat([game_df, strategy_summary, global_summary], ignore_index=True)
    return combined[
        [
            "summary_level",
            "strategy",
            "n_players",
            "margin_of_victory",
            "multi_reached_target",
            "observations",
            *[f"margin_le_{thr}" for thr in thresholds],
        ]
    ]


def _global_stats(combined_path: Path) -> pd.DataFrame:
    """Aggregate ``n_rounds`` across all strategies, grouped by ``n_players``."""

    ds_in = ds.dataset(combined_path)
    schema = ds_in.schema
    n_players = n_players_from_schema(schema)
    columns = ["n_rounds", "seat_ranks"]
    tbl = ds_in.to_table(columns=columns)
    df = tbl.to_pandas()

    if "seat_ranks" not in df.columns:
        LOGGER.warning(
            "Combined parquet missing seat_ranks; skipping global game-length stats",
            extra={"stage": "game_stats", "path": str(combined_path)},
        )
        return pd.DataFrame()

    df["n_players"] = df["seat_ranks"].apply(
        lambda ranks: len(ranks) if isinstance(ranks, list) else n_players
    )
    grouped = df.groupby("n_players", sort=False)["n_rounds"]

    rows: list[pd.Series] = []
    for players, rounds in grouped:
        stats = _summarize_rounds(rounds)
        stats.update(
            {
                "summary_level": "n_players",
                "strategy": pd.NA,
                "n_players": int(players),
            }
        )
        rows.append(pd.Series(stats))

    return pd.DataFrame(rows)


def _compute_margins(df: pd.DataFrame, score_cols: Sequence[str]) -> pd.Series:
    """Derive per-game margin of victory from seat scores.

    For two-player games, this is ``|P1_score - P2_score|``. For more than two
    players, this is ``max(score) - min(score)`` based on available seat scores.
    Games with fewer than two valid scores return ``NaN`` margins.
    """

    scores = df.loc[:, score_cols].apply(pd.to_numeric, errors="coerce")
    valid_counts = scores.notna().sum(axis=1)
    margins = scores.max(axis=1, skipna=True) - scores.min(axis=1, skipna=True)
    margins[valid_counts < 2] = np.nan
    return margins.astype(float)


def _summarize_margins(
    values: Iterable[int | float | np.integer | np.floating],
    thresholds: Sequence[int],
) -> dict[str, float | int]:
    """Return descriptive statistics for per-game victory margins."""

    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    prob_keys = [f"prob_margin_le_{thr}" for thr in thresholds]
    if series.empty:
        base = {
            "observations": 0,
            "mean_margin": float("nan"),
            "median_margin": float("nan"),
            "std_margin": float("nan"),
        }
        base.update({key: float("nan") for key in prob_keys})
        return base

    stats: dict[str, float | int] = {
        "observations": int(series.size),
        "mean_margin": float(series.mean()),
        "median_margin": float(series.median()),
        "std_margin": float(series.std(ddof=0)),
    }
    stats.update({key: float((series <= thr).mean()) for key, thr in zip(prob_keys, thresholds, strict=True)})
    return stats


def _summarize_rounds(values: Iterable[int | float | np.integer | np.floating]) -> dict[str, float | int]:
    """Return descriptive statistics for the provided iterable of round counts."""

    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return {
            "observations": 0,
            "mean_rounds": float("nan"),
            "median_rounds": float("nan"),
            "std_rounds": float("nan"),
            "p10_rounds": float("nan"),
            "p50_rounds": float("nan"),
            "p90_rounds": float("nan"),
            "prob_rounds_le_5": float("nan"),
            "prob_rounds_le_10": float("nan"),
            "prob_rounds_ge_20": float("nan"),
        }

    count = int(series.size)
    return {
        "observations": count,
        "mean_rounds": float(series.mean()),
        "median_rounds": float(series.median()),
        "std_rounds": float(series.std(ddof=0)),
        "p10_rounds": float(series.quantile(0.10)),
        "p50_rounds": float(series.quantile(0.50)),
        "p90_rounds": float(series.quantile(0.90)),
        "prob_rounds_le_5": float((series <= 5).mean()),
        "prob_rounds_le_10": float((series <= 10).mean()),
        "prob_rounds_ge_20": float((series >= 20).mean()),
    }


def _stamp(path: Path) -> dict[str, float | int]:
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _write_stamp(stamp_path: Path, *, inputs: Iterable[Path], outputs: Iterable[Path]) -> None:
    payload = {
        "inputs": {str(p): _stamp(p) for p in inputs if p.exists()},
        "outputs": {str(p): _stamp(p) for p in outputs if p.exists()},
    }
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(stamp_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2))


def _is_up_to_date(stamp_path: Path, *, inputs: Iterable[Path], outputs: Iterable[Path]) -> bool:
    if not (stamp_path.exists() and all(p.exists() for p in outputs)):
        return False
    try:
        meta = json.loads(stamp_path.read_text())
    except Exception:  # noqa: BLE001
        return False

    in_meta = meta.get("inputs", {})
    out_meta = meta.get("outputs", {})

    def _matches(paths: Iterable[Path], recorded: dict[str, dict[str, float | int]]) -> bool:
        for p in paths:
            data = recorded.get(str(p))
            if data is None:
                return False
            stat = p.stat()
            if data.get("mtime") != stat.st_mtime or data.get("size") != stat.st_size:
                return False
        return True

    return _matches(inputs, in_meta) and _matches(outputs, out_meta)

