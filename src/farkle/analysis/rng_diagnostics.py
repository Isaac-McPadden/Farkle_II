"""RNG diagnostics for curated rows ordered by ``game_seed``.

Computes lagged autocorrelation for win indicators and game lengths at both
strategy and matchup-strategy levels. Outputs are written to
``00_rng/pooled/rng_diagnostics.parquet`` with approximate confidence intervals
and "expected ~0" reminders. When inputs or required columns are missing, the
module logs a skip instead of raising.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from itertools import chain
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from farkle.analysis import stage_logger
from farkle.analysis.stage_state import stage_done_path
from farkle.config import AppConfig
from farkle.utils.artifacts import write_parquet_atomic
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)

_EXPECTED_NOTE = "Expected ~0 under IID seeds"
_SEAT_STRATEGY_RE = re.compile(r"^P(\d+)_strategy$")


def run(cfg: AppConfig, *, lags: Sequence[int] | None = None, force: bool = False) -> None:
    """Compute lagged autocorrelation diagnostics for curated rows.

    Args:
        cfg: Application configuration for locating curated inputs and outputs.
        lags: Optional sequence of positive lags (defaults to ``(1,)``).
        force: Recompute even when the done-stamp matches inputs/outputs.
    """

    stage_log = stage_logger("rng_diagnostics", logger=LOGGER)
    stage_log.start()

    interseed_ready, interseed_reason = cfg.interseed_ready()
    if not interseed_ready:
        stage_log.missing_input(interseed_reason)
        return

    data_file = cfg.curated_parquet
    out_file = cfg.rng_output_path("rng_diagnostics.parquet")
    stamp_path = stage_done_path(cfg.rng_stage_dir, "rng_diagnostics")

    lags = _normalize_lags(lags)
    if not lags:
        stage_log.missing_input("no valid lags provided")
        return

    if not data_file.exists():
        stage_log.missing_input("missing curated parquet", path=str(data_file))
        return

    if not force and _is_up_to_date(
        stamp_path, inputs=[data_file], outputs=[out_file], lags=lags, config_sha=cfg.config_sha
    ):
        LOGGER.info(
            "rng-diagnostics: up-to-date",
            extra={"stage": "rng_diagnostics", "path": str(out_file), "stamp": str(stamp_path)},
        )
        return

    dataset = ds.dataset(data_file)
    schema_names = set(dataset.schema.names)
    strat_cols = _seat_strategy_columns(cfg, dataset.schema.names)
    winner_col = _winner_column(schema_names)

    required = {"game_seed", "n_rounds"}
    if not required.issubset(schema_names) or winner_col is None:
        stage_log.missing_input(
            "curated parquet missing required columns",
            path=str(data_file),
            required_cols=sorted(required | {"winner_strategy", "winner_seat"}),
        )
        return

    if not strat_cols:
        stage_log.missing_input(
            "curated parquet missing seat strategy columns",
            path=str(data_file),
            required_cols=["P1_strategy"],
        )
        return

    columns = ["game_seed", "n_rounds", winner_col, *strat_cols]
    df = dataset.to_table(columns=columns).to_pandas(categories=strat_cols)
    df = df.sort_values("game_seed")
    df["matchup"] = _build_matchup_labels(df, strat_cols)
    df["n_players"] = df[strat_cols].notna().sum(axis=1).astype(int)
    df["winner_strategy"] = _winner_strategies(df, winner_col, strat_cols)

    melted = _melt_strategies(df, strat_cols)
    if melted.empty:
        stage_log.missing_input("no per-strategy rows after melting", path=str(data_file))
        return

    diagnostics = _collect_diagnostics(melted, lags=lags)
    if diagnostics.empty:
        stage_log.missing_input("no diagnostics computed", path=str(data_file))
        return

    table_out = pa.Table.from_pandas(diagnostics, preserve_index=False)
    write_parquet_atomic(table_out, out_file, codec=cfg.parquet_codec)
    _write_stamp(
        stamp_path,
        inputs=[data_file],
        outputs=[out_file],
        lags=lags,
        config_sha=cfg.config_sha,
    )
    LOGGER.info(
        "rng-diagnostics: written",
        extra={"stage": "rng_diagnostics", "rows": len(diagnostics), "path": str(out_file)},
    )


def _normalize_lags(lags: Sequence[int] | None) -> tuple[int, ...]:
    if lags is None:
        return (1,)
    valid = sorted({int(lag) for lag in lags if int(lag) > 0})
    return tuple(valid)


def _winner_column(names: set[str]) -> str | None:
    if "winner_strategy" in names:
        return "winner_strategy"
    if "winner_seat" in names:
        return "winner_seat"
    return None


def _seat_strategy_columns(cfg: AppConfig, schema_names: Sequence[str]) -> list[str]:
    schema_set = set(schema_names)
    configured_candidates: list[str] = []
    if cfg.sim.n_players_list:
        max_players = max(cfg.sim.n_players_list)
        configured_candidates = [f"P{seat}_strategy" for seat in range(1, max_players + 1)]

    fallback_candidates = [
        name for name in schema_names if _SEAT_STRATEGY_RE.match(name) and name != "winner_strategy"
    ]
    merged_candidates = set(configured_candidates) | set(fallback_candidates)
    present = [name for name in merged_candidates if name in schema_set and name != "winner_strategy"]

    return sorted(present, key=lambda col: int(_SEAT_STRATEGY_RE.match(col).group(1)))


def _build_matchup_labels(df: pd.DataFrame, strat_cols: Sequence[str]) -> pd.Series:
    valid_seat_cols = [col for col in strat_cols if col in df.columns and _SEAT_STRATEGY_RE.match(col)]
    if not valid_seat_cols:
        return pd.Series(index=df.index, dtype="string")

    seat_values = df[valid_seat_cols].astype("string")
    stacked = seat_values.stack()
    stacked = stacked[stacked.notna()]
    if stacked.empty:
        return pd.Series(index=df.index, dtype="string")

    matchups = (
        stacked.groupby(level=0, sort=False)
        .agg(list)
        .map(lambda participants: " | ".join(sorted(participants)))
    )
    return matchups.reindex(df.index).astype("string")


def _winner_strategies(
    df: pd.DataFrame, winner_col: str, strat_cols: Sequence[str]
) -> pd.Series:
    if winner_col == "winner_strategy":
        return df[winner_col]

    valid_cols = [col for col in strat_cols if _SEAT_STRATEGY_RE.match(col)]
    seat_to_col_position = {
        int(cast(re.Match[str], _SEAT_STRATEGY_RE.match(col)).group(1)): idx
        for idx, col in enumerate(valid_cols)
    }
    winners = pd.Series(pd.NA, index=df.index, dtype="string")

    if not valid_cols:
        return winners

    seat_indices = pd.to_numeric(
        df[winner_col].astype("string").str.extract(r"^P(\d+)$", expand=False),
        errors="coerce",
    ).astype("Int64")
    resolved_positions = seat_indices.map(seat_to_col_position)
    valid_mask = resolved_positions.notna()
    if not valid_mask.any():
        return winners

    strategy_values = df[valid_cols].astype("string").to_numpy(dtype=object)
    winner_rows = np.flatnonzero(valid_mask.to_numpy())
    winner_cols = resolved_positions[valid_mask].astype(int).to_numpy()
    winners.iloc[winner_rows] = strategy_values[winner_rows, winner_cols]
    return winners


def _melt_strategies(df: pd.DataFrame, strat_cols: Sequence[str]) -> pd.DataFrame:
    id_vars = ["game_seed", "n_rounds", "matchup", "n_players", "winner_strategy"]
    melted = df[id_vars + list(strat_cols)].melt(
        id_vars=id_vars,
        value_vars=strat_cols,
        var_name="seat",
        value_name="strategy",
    )
    melted = melted.dropna(subset=["strategy"])
    melted["strategy"] = melted["strategy"].astype("category")
    melted["seat"] = melted["seat"].str.removesuffix("_strategy")
    melted["win_indicator"] = (melted["strategy"] == melted["winner_strategy"]).astype(int)
    return melted


def _collect_diagnostics(data: pd.DataFrame, *, lags: Iterable[int]) -> pd.DataFrame:
    rows: list[pd.Series] = []

    grouped_strategy = data.groupby(["strategy", "n_players"], observed=True, sort=False)
    strategy_diagnostics = (
        _group_diagnostics(
            group,
            lags=lags,
            summary_level="strategy",
            strategy=strategy_str,
            n_players=n_players_int,
        )
        for (strategy, n_players), group in grouped_strategy
        for strategy_str in (str(strategy),)
        for n_players_int in (int(cast(int, n_players)),)
    )
    # Each grouped call yields an iterable of pd.Series diagnostics.
    rows.extend(chain.from_iterable(strategy_diagnostics))

    grouped_matchup = data.groupby(
        ["matchup", "strategy", "n_players"], observed=True, sort=False
    )
    matchup_diagnostics = (
        _group_diagnostics(
            group,
            lags=lags,
            summary_level="matchup_strategy",
            strategy=strategy_str,
            matchup=matchup_str,
            n_players=n_players_int,
        )
        for (matchup, strategy, n_players), group in grouped_matchup
        for strategy_str in (str(strategy),)
        for n_players_int in (int(cast(int, n_players)),)
        for matchup_str in ((None if matchup is None else str(matchup)),)
    )
    rows.extend(chain.from_iterable(matchup_diagnostics))

    flattened = [row for row in rows if not row.empty]
    if not flattened:
        return pd.DataFrame()
    diagnostics = pd.DataFrame(flattened)
    diagnostics = diagnostics.sort_values(["summary_level", "strategy", "n_players", "lag", "metric"])
    return diagnostics


def _group_diagnostics(
    group: pd.DataFrame,
    *,
    lags: Iterable[int],
    summary_level: str,
    strategy: str,
    n_players: int,
    matchup: str | None = None,
) -> list[pd.Series]:
    rows: list[pd.Series] = []
    ordered = group.sort_values("game_seed")
    metrics = {
        "win_indicator": ordered["win_indicator"],
        "n_rounds": ordered["n_rounds"],
    }

    for lag in lags:
        for metric, series in metrics.items():
            cleaned = series.dropna()
            n_obs = len(cleaned)
            if n_obs <= lag:
                continue
            autocorr = cleaned.autocorr(lag=lag)
            if pd.isna(autocorr):
                continue
            stderr = 1.0 / n_obs**0.5
            ci_lower = autocorr - 1.96 * stderr
            ci_upper = autocorr + 1.96 * stderr
            rows.append(
                pd.Series(
                    {
                        "summary_level": summary_level,
                        "strategy": strategy,
                        "matchup": matchup,
                        "n_players": n_players,
                        "observations": n_obs,
                        "lag": lag,
                        "metric": metric,
                        "autocorr": autocorr,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "note": _EXPECTED_NOTE,
                    }
                )
            )
    return rows


def _stamp(path: Path) -> dict[str, float | int]:
    stat = path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _write_stamp(
    stamp_path: Path,
    *,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    lags: Sequence[int],
    config_sha: str | None,
) -> None:
    payload = {
        "inputs": {str(p): _stamp(p) for p in inputs if p.exists()},
        "outputs": {str(p): _stamp(p) for p in outputs if p.exists()},
        "params": {"lags": list(lags)},
        "config_sha": config_sha,
    }
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(stamp_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2))


def _is_up_to_date(
    stamp_path: Path,
    *,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    lags: Sequence[int],
    config_sha: str | None,
) -> bool:
    if not (stamp_path.exists() and all(p.exists() for p in outputs)):
        return False
    try:
        meta = json.loads(stamp_path.read_text())
    except Exception:  # noqa: BLE001
        return False

    recorded_sha = meta.get("config_sha")
    if config_sha is not None and recorded_sha not in (None, config_sha):
        return False

    if meta.get("params", {}).get("lags") != list(lags):
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


if __name__ == "__main__":  # pragma: no cover
    config = AppConfig()
    run(config)
