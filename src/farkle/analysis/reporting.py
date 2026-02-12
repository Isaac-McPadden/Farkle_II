# src/farkle/analysis/reporting.py
"""Generate per-player Markdown reports and Matplotlib figures.

This module reads previously generated analysis artifacts (TrueSkill ratings,
meta win-rate summaries, head-to-head decisions, etc.) and produces a small
collection of visualisations alongside a Markdown summary for each configured
player count.  All outputs are derived from existing files – no simulations or
heavy recomputation occurs here.

Each helper inspects timestamps so rerunning the report is idempotent unless a
``force`` flag is provided.  Outputs are written via :func:`atomic_path` to
avoid partially written files.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, SupportsFloat, cast

import matplotlib

# Force a non-interactive backend so the code works in headless environments.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from farkle.analysis.stage_registry import StageLayout
from farkle.config import AnalysisConfig, AppConfig
from farkle.utils.tiers import load_tier_payload, tier_mapping_from_payload
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)

LADDER_TOP_N = 25
FEATURE_TOP_N = 20
SEED_TOP_N = 10
PLOT_DPI = 150
TIER_LABELS = ("S", "A", "B", "C", "D", "F")
TIER_BUCKET_SIZES = (100, 200, 400, 800, 1600)


class ReportError(RuntimeError):
    """Raised when required artifacts are missing or malformed."""


@dataclass(slots=True)
class _ReportArtifacts:
    """Container for the small set of data frames used to build a report."""

    ratings: pd.DataFrame
    meta_summary: pd.DataFrame
    feature_importance: pd.DataFrame
    seed_summaries: pd.DataFrame
    tiers: dict[str, int]
    h2h_decisions: pd.DataFrame
    h2h_ranking: pd.DataFrame
    h2h_s_tiers: dict[str, str]
    heterogeneity: dict[str, float]
    run_metadata: dict[str, object]


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────


def _analysis_dir(cfg: AnalysisConfig | AppConfig) -> Path:
    """Return the analysis directory for the provided config."""

    analysis_dir = getattr(cfg, "analysis_dir", None)
    if analysis_dir is not None:
        return Path(str(analysis_dir))

    io_cfg = getattr(cfg, "io", None)
    results_root = getattr(cfg, "results_root", None)
    if results_root is not None:
        return Path(str(results_root)) / "analysis"
    if io_cfg is not None:
        prefix = getattr(io_cfg, "results_dir_prefix", None)
        if prefix is not None:
            base = Path(prefix)
            if not base.is_absolute():
                base = Path("data") / base
            seed = getattr(getattr(cfg, "sim", None), "seed", 0)
            suffix = f"_seed_{seed}"
            if not base.name.endswith(suffix):
                base = base.parent / f"{base.name}{suffix}"
            return base / "analysis"

    raise AttributeError("Configuration object does not expose an analysis_dir")


def _analysis_layout(cfg: AnalysisConfig | AppConfig | None) -> StageLayout | None:
    """Return a resolved layout when available from an :class:`AppConfig`."""

    layout = getattr(cfg, "stage_layout", None)
    return cast(StageLayout | None, layout)


def _meta_artifact_path(cfg: AnalysisConfig | AppConfig, players: int, filename: str) -> Path:
    """Resolve a meta-analysis artifact path with per-player preference."""

    if isinstance(cfg, AppConfig):
        return cfg.meta_input_path(players, filename)

    analysis_dir = _analysis_dir(cfg)
    layout = _analysis_layout(cfg)
    candidates = [
        *_stage_candidates(
            analysis_dir, "meta", layout=layout, filename=Path(f"{players}p") / filename
        ),
        *_stage_candidates(
            analysis_dir, "meta", layout=layout, filename=Path("pooled") / filename
        ),
        analysis_dir / filename,
    ]
    return _first_existing(candidates)


def _first_existing(candidates: list[Path]) -> Path:
    """Return the first existing path or the first candidate when none exist."""

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _tier_path(analysis_dir: Path, layout: StageLayout | None = None) -> Path:
    """Resolve ``tiers.json`` within stage-aware directories."""

    candidates = [
        *_stage_candidates(analysis_dir, "tiering", layout=layout, filename=Path("tiers.json")),
        *_stage_candidates(analysis_dir, "trueskill", layout=layout, filename=Path("tiers.json")),
        analysis_dir / "tiers.json",
    ]
    return _first_existing(candidates)


def _ratings_path(analysis_dir: Path, layout: StageLayout | None = None) -> Path:
    """Resolve pooled TrueSkill ratings with stage-aware fallbacks."""

    candidates = [
        *_stage_candidates(
            analysis_dir,
            "trueskill",
            layout=layout,
            filename=Path("pooled") / "ratings_pooled.parquet",
        ),
        *_stage_candidates(
            analysis_dir,
            "trueskill",
            layout=layout,
            filename=Path("ratings_pooled.parquet"),
        ),
        analysis_dir / "pooled" / "ratings_pooled.parquet",
        analysis_dir / "ratings_pooled.parquet",
    ]
    return _first_existing(candidates)


def _head2head_path(
    analysis_dir: Path, filename: str, layout: StageLayout | None = None
) -> Path:
    """Resolve a head-to-head artifact path with legacy fallback."""

    return _first_existing(
        [
            *_stage_candidates(
                analysis_dir, "head2head", layout=layout, filename=Path(filename)
            ),
            analysis_dir / filename,
        ]
    )


def _post_h2h_path(
    analysis_dir: Path, filename: str, layout: StageLayout | None = None
) -> Path:
    """Resolve a post head-to-head artifact path with legacy fallback."""

    return _first_existing(
        [
            *_stage_candidates(
                analysis_dir, "post_h2h", layout=layout, filename=Path(filename)
            ),
            *_stage_candidates(
                analysis_dir, "head2head", layout=layout, filename=Path(filename)
            ),
            analysis_dir / filename,
        ]
    )


def _stage_candidates(
    analysis_dir: Path,
    suffix: str,
    *,
    layout: StageLayout | None = None,
    filename: Path | None = None,
) -> list[Path]:
    """Return ordered stage-path candidates for a given suffix and optional filename."""

    preferred: list[Path] = []
    if layout is not None:
        folder = layout.folder_for(suffix)
        if folder is not None:
            preferred_dir = analysis_dir / folder
            preferred.append(preferred_dir if filename is None else preferred_dir / filename)

    stage_dirs = sorted(
        (p for p in analysis_dir.glob(f"*_{suffix}") if p.is_dir()), key=lambda p: p.name
    )
    for path in stage_dirs:
        if layout is not None and path.name == layout.folder_for(suffix):
            continue
        LOGGER.warning(
            "Legacy stage directory detected; prefer layout-aware helpers",
            extra={"stage": suffix, "legacy_path": str(path)},
        )

    candidates = preferred or stage_dirs
    if filename is None:
        return candidates
    return [stage_dir / filename for stage_dir in candidates]


def _sim_player_counts(
    cfg: AnalysisConfig | AppConfig, analysis_dir: Path, *, layout: StageLayout | None = None
) -> list[int]:
    """Determine the list of player counts that should be reported."""

    players: set[int] = set()
    # ``cfg`` may be an ``AppConfig`` or a bare ``AnalysisConfig``.
    analysis_cfg = getattr(cfg, "analysis", None)
    sim_cfg = getattr(cfg, "sim", None)

    if analysis_cfg is not None:
        analysis_players = getattr(analysis_cfg, "n_players_list", None) or []
        players.update(int(p) for p in analysis_players)

    if sim_cfg is not None:
        sim_players = getattr(sim_cfg, "n_players_list", None) or []
        players.update(int(p) for p in sim_players)

    ratings_path = _ratings_path(analysis_dir, layout=layout)
    if ratings_path.exists():
        try:
            df = pd.read_parquet(ratings_path, columns=["strategy", "mu", "sigma", "players"])
        except ValueError:
            df = pd.read_parquet(ratings_path)
        for column in ("players", "n_players", "player_count"):
            if column in df.columns:
                series = df[column].dropna().astype(int)
                players.update(series.unique().tolist())
                break

    if not players:
        raise ReportError("Unable to determine player counts for reporting")

    return sorted(players)


def _output_is_fresh(output: Path, inputs: Iterable[Path], *, force: bool) -> bool:
    """Return ``True`` if ``output`` is newer than every file in ``inputs``."""

    if force or not output.exists():
        return False
    out_mtime = output.stat().st_mtime
    for path in inputs:
        if not path.exists():
            return False
        if path.stat().st_mtime > out_mtime:
            return False
    return True


def _as_float(value: object) -> float:
    """Convert pandas scalars and numpy numbers to builtin ``float`` for typing."""

    return float(cast(SupportsFloat, value))


def _extract_scalar(value: object, *, label: str) -> object:
    """Return a scalar from pandas containers, validating singleton expectations."""

    if isinstance(value, pd.Series):
        if value.size != 1:
            raise ReportError(f"Expected {label} to be a scalar, got {value.size} entries")
        return value.iloc[0]
    return value


def _load_ratings(
    analysis_dir: Path, players: int, *, layout: StageLayout | None = None
) -> pd.DataFrame:
    """Load pooled TrueSkill ratings filtered for the requested player count.

    Args:
        analysis_dir: Root directory containing analysis outputs.
        players: Player count to select within the ratings parquet.

    Returns:
        Dataframe with ``strategy``, ``players``, ``mu``, and ``sigma`` columns
        sorted by rating.
    """
    path = _ratings_path(analysis_dir, layout=layout)
    if not path.exists():
        raise ReportError(f"Missing ratings parquet: {path}")

    df = pd.read_parquet(path)
    df = df.copy()
    if "strategy" not in df.columns or "mu" not in df.columns or "sigma" not in df.columns:
        raise ReportError("ratings_pooled.parquet missing required columns")

    player_column = None
    for column in ("players", "n_players", "player_count"):
        if column in df.columns:
            player_column = column
            break
    if player_column is None:
        raise ReportError("ratings parquet does not contain a player-count column")

    df["strategy"] = df["strategy"].astype(str)
    df[player_column] = df[player_column].astype(int)
    subset = df[df[player_column] == int(players)]
    if subset.empty:
        raise ReportError(f"No ratings found for {players}-player games")
    subset = subset.sort_values(["mu", "strategy"], ascending=[False, True], kind="mergesort")
    subset.reset_index(drop=True, inplace=True)
    subset.rename(columns={player_column: "players"}, inplace=True)
    return subset[["strategy", "players", "mu", "sigma"]]


def _load_meta_summary(cfg: AnalysisConfig | AppConfig, players: int) -> pd.DataFrame:
    """Load pooled win-rate meta summary if available for ``players``."""

    path = _meta_artifact_path(cfg, players, f"strategy_summary_{players}p_meta.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "strategy_id" in df.columns:
        df = df.rename(columns={"strategy_id": "strategy"})
    if "strategy" not in df.columns or "win_rate" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["strategy"] = df["strategy"].astype(str)
    if "players" in df.columns:
        df["players"] = df["players"].astype(int)
        df = df[df["players"] == int(players)]
    df.sort_values(
        ["win_rate", "strategy"], ascending=[False, True], inplace=True, kind="mergesort"
    )
    df.reset_index(drop=True, inplace=True)
    return df


def _load_feature_importance(analysis_dir: Path, players: int) -> pd.DataFrame:
    """Load model feature importances for the given player count."""
    path = analysis_dir / f"feature_importance_{players}p.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.copy()
    feature_column = None
    for column in ("feature", "name", "column"):
        if column in df.columns:
            feature_column = column
            break
    if feature_column is None:
        return pd.DataFrame()
    value_column = None
    for column in ("importance", "value", "gain", "weight"):
        if column in df.columns:
            value_column = column
            break
    if value_column is None:
        return pd.DataFrame()
    df[feature_column] = df[feature_column].astype(str)
    df[value_column] = df[value_column].astype(float)
    df = df[[feature_column, value_column]].rename(
        columns={feature_column: "feature", value_column: "importance"}
    )
    total = df["importance"].sum()
    if total > 0:
        df["importance"] = df["importance"] / total
    df.sort_values(
        ["importance", "feature"], ascending=[False, True], inplace=True, kind="mergesort"
    )
    df.reset_index(drop=True, inplace=True)
    return df


def _load_seed_summaries(analysis_dir: Path, players: int) -> pd.DataFrame:
    """Collect per-seed win-rate summaries across available parquet files."""
    frames: list[pd.DataFrame] = []
    pattern = f"strategy_summary_{players}p_seed"
    for path in sorted(analysis_dir.glob(f"{pattern}*.parquet")):
        df = pd.read_parquet(path)
        if df.empty:
            continue
        frame = df.copy()
        if "strategy" in frame.columns:
            frame.rename(columns={"strategy": "strategy_id"}, inplace=True)
        if "strategy_id" not in frame.columns or "win_rate" not in frame.columns:
            continue
        frame["strategy_id"] = frame["strategy_id"].astype(str)
        if "players" in frame.columns:
            frame["players"] = frame["players"].astype(int)
            frame = frame[frame["players"] == int(players)]
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["strategy_id", "seed", "win_rate"])
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "seed" in combined.columns:
        combined["seed"] = combined["seed"].astype(int)
    else:
        combined["seed"] = 0
    combined = combined[
        [c for c in combined.columns if c in {"strategy_id", "seed", "win_rate", "ci_lo", "ci_hi"}]
    ]
    combined = combined.sort_values(["strategy_id", "seed"], kind="mergesort").reset_index(
        drop=True
    )
    return combined


def _load_tiers(
    analysis_dir: Path, players: int, *, layout: StageLayout | None = None
) -> dict[str, int]:
    """Parse tier assignments from JSON, scoped to the desired player count."""
    path = _tier_path(analysis_dir, layout=layout)
    payload = load_tier_payload(path)
    return tier_mapping_from_payload(payload, prefer=str(players))


def _load_h2h_decisions(
    analysis_dir: Path, players: int, *, layout: StageLayout | None = None
) -> pd.DataFrame:
    """Load pairwise significance decisions for head-to-head comparisons."""
    path = _post_h2h_path(analysis_dir, "bonferroni_decisions.parquet", layout=layout)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()
    for column in ("a", "b"):
        if column not in df.columns:
            return pd.DataFrame()
        df[column] = df[column].astype(str)
    if "players" in df.columns:
        df["players"] = df["players"].astype(int)
        df = df[df["players"] == int(players)]
    return df


def _load_h2h_ranking(
    analysis_dir: Path, players: int, *, layout: StageLayout | None = None
) -> pd.DataFrame:
    """Load significant ranking output produced from head-to-head tests."""
    path = _post_h2h_path(analysis_dir, "h2h_significant_ranking.csv", layout=layout)
    if not path.exists():
        return pd.DataFrame(columns=["strategy", "rank"])
    df = pd.read_csv(path)
    if "strategy" not in df.columns:
        return pd.DataFrame(columns=["strategy", "rank"])
    df = df.copy()
    df["strategy"] = df["strategy"].astype(str)
    if "rank" not in df.columns:
        df["rank"] = np.arange(1, len(df) + 1)
    if "players" in df.columns:
        df["players"] = df["players"].astype(int)
        df = df[df["players"] == int(players)]
    df = df.sort_values(["rank", "strategy"], kind="mergesort").reset_index(drop=True)
    return df[["strategy", "rank"]]


def _load_h2h_s_tiers(
    analysis_dir: Path, *, layout: StageLayout | None = None
) -> dict[str, str]:
    """Load head-to-head S-tier buckets from post-H2H artifacts."""
    path = _post_h2h_path(analysis_dir, "h2h_s_tiers.json", layout=layout)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(value, str):
            continue
        result[str(key)] = value
    return result


def _load_meta_json(cfg: AnalysisConfig | AppConfig, players: int) -> dict[str, float]:
    """Load heterogeneity metrics from the meta-analysis JSON payload."""

    path = _meta_artifact_path(cfg, players, f"meta_{players}p.json")
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in payload.items():
        try:
            result[str(key)] = float(value)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            continue
    return result


def _load_run_metadata(analysis_dir: Path) -> dict[str, object]:
    """Load recorded run metadata such as config hash and git commit."""
    path = analysis_dir / "run_metadata.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _gather_artifacts(cfg: AnalysisConfig | AppConfig, players: int) -> _ReportArtifacts:
    """Load all report prerequisites for the specified player count."""
    analysis_dir = _analysis_dir(cfg)
    layout = _analysis_layout(cfg)
    ratings = _load_ratings(analysis_dir, players, layout=layout)
    meta_summary = _load_meta_summary(cfg, players)
    feature_importance = _load_feature_importance(analysis_dir, players)
    seed_summaries = _load_seed_summaries(analysis_dir, players)
    tiers = _load_tiers(analysis_dir, players, layout=layout)
    h2h_decisions = _load_h2h_decisions(analysis_dir, players, layout=layout)
    h2h_ranking = _load_h2h_ranking(analysis_dir, players, layout=layout)
    h2h_s_tiers = _load_h2h_s_tiers(analysis_dir, layout=layout)
    heterogeneity = _load_meta_json(cfg, players)
    run_metadata = _load_run_metadata(analysis_dir)
    return _ReportArtifacts(
        ratings=ratings,
        meta_summary=meta_summary,
        feature_importance=feature_importance,
        seed_summaries=seed_summaries,
        tiers=tiers,
        h2h_decisions=h2h_decisions,
        h2h_ranking=h2h_ranking,
        h2h_s_tiers=h2h_s_tiers,
        heterogeneity=heterogeneity,
        run_metadata=run_metadata,
    )


def _plot_output_path(cfg: AnalysisConfig | AppConfig, players: int, name: str) -> Path:
    """Create (and ensure) the plot output path for a given filename."""
    analysis_dir = _analysis_dir(cfg)
    plot_dir = analysis_dir / "plots" / f"{players}p"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir / name


def _prepare_axis(
    fig: Figure, count: int, *, base_height: float = 3.5, row_scale: float = 0.35
) -> Axes:
    """Set figure size based on row count and return a single subplot."""
    height = max(base_height, row_scale * max(1, count) + 1.5)
    fig.set_size_inches(8.5, height)
    ax = fig.add_subplot(1, 1, 1)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────


def plot_ladder_for_players(
    cfg: AnalysisConfig | AppConfig, players: int, *, force: bool = False
) -> Path:
    """Create a ladder plot of the top strategies by TrueSkill rating."""

    analysis_dir = _analysis_dir(cfg)
    layout = _analysis_layout(cfg)
    ratings = _load_ratings(analysis_dir, players, layout=layout)
    top = ratings.head(LADDER_TOP_N)
    output = _plot_output_path(cfg, players, f"ladder_{players}p.png")

    if _output_is_fresh(output, [_ratings_path(analysis_dir, layout=layout)], force=force):
        return output

    fig = plt.figure()
    ax = _prepare_axis(fig, len(top))
    y_positions = np.arange(len(top))[::-1]
    mu = top["mu"].to_numpy(dtype=float)
    sigma = top["sigma"].to_numpy(dtype=float)
    error = sigma * 2.0
    ax.errorbar(mu, y_positions, xerr=error, fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top["strategy"].tolist())
    ax.set_xlabel("TrueSkill μ (± 2σ)")
    ax.set_title(f"Top strategies for {players}-player games")
    ax.axvline(mu.mean(), linestyle="--", color="#aaaaaa", linewidth=1)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    with atomic_path(str(output)) as tmp_path:
        fig.savefig(tmp_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(
        "Ladder plot written",
        extra={"stage": "report", "players": players, "path": str(output)},
    )
    return output


def _determine_heatmap_order(artifacts: _ReportArtifacts, _players: int) -> list[str]:
    """Choose strategy ordering for heatmap plotting preferences."""
    if not artifacts.tiers and artifacts.h2h_ranking.empty:
        return artifacts.ratings.head(LADDER_TOP_N)["strategy"].tolist()

    if not artifacts.h2h_ranking.empty:
        return artifacts.h2h_ranking["strategy"].tolist()

    tiers = artifacts.tiers
    sorted_items = sorted(tiers.items(), key=lambda kv: (kv[1], kv[0]))
    return [strategy for strategy, _ in sorted_items]


def plot_h2h_heatmap_for_players(
    cfg: AnalysisConfig | AppConfig, players: int, *, force: bool = False
) -> Path | None:
    """Render a heatmap of head-to-head win rates for available strategies."""

    analysis_dir = _analysis_dir(cfg)
    layout = _analysis_layout(cfg)
    decisions = _load_h2h_decisions(analysis_dir, players, layout=layout)
    if decisions.empty:
        return None

    if (
        "wins_a" in decisions.columns
        and "wins_b" in decisions.columns
        and "games" in decisions.columns
    ):
        wins_a = decisions["wins_a"].astype(float)
        games = decisions["games"].replace(0, np.nan).astype(float)
        decisions = decisions.assign(win_rate=wins_a / games)
    elif "win_rate" not in decisions.columns:
        LOGGER.info(
            "Skipping head-to-head heatmap: win rates unavailable",
            extra={"stage": "report", "players": players},
        )
        return None

    output = _plot_output_path(cfg, players, f"h2h_heatmap_{players}p.png")
    inputs = [
        _post_h2h_path(_analysis_dir(cfg), "bonferroni_decisions.parquet", layout=layout),
        _tier_path(_analysis_dir(cfg), layout=layout),
        _post_h2h_path(_analysis_dir(cfg), "h2h_significant_ranking.csv", layout=layout),
    ]
    if _output_is_fresh(output, inputs, force=force):
        return output

    artifacts = _ReportArtifacts(
        ratings=_load_ratings(analysis_dir, players, layout=layout),
        meta_summary=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        seed_summaries=pd.DataFrame(),
        tiers=_load_tiers(analysis_dir, players, layout=layout),
        h2h_decisions=decisions,
        h2h_ranking=_load_h2h_ranking(analysis_dir, players, layout=layout),
        h2h_s_tiers=_load_h2h_s_tiers(analysis_dir, layout=layout),
        heterogeneity={},
        run_metadata={},
    )

    order = _determine_heatmap_order(artifacts, players)
    if not order:
        LOGGER.info(
            "Skipping head-to-head heatmap: no strategies available",
            extra={"stage": "report", "players": players},
        )
        return None

    matrix = pd.DataFrame(
        np.nan,
        index=pd.Index(order, name="row"),
        columns=pd.Index(order, name="col"),
    )

    for row in decisions.itertuples(index=False):
        if row.a in matrix.index and row.b in matrix.columns:
            win_rate = getattr(row, "win_rate", np.nan)
            matrix.loc[row.a, row.b] = _as_float(win_rate)
        if (
            hasattr(row, "wins_b")
            and hasattr(row, "games")
            and row.b in matrix.index
            and row.a in matrix.columns
        ):
            try:
                wins_b_value = _extract_scalar(row.wins_b, label="wins_b")
                games_value = _extract_scalar(row.games, label="games")
                wins_b_count = float(_as_float(wins_b_value))
                games_count = float(_as_float(games_value))
                win_rate_b = wins_b_count / games_count
                matrix.at[row.b, row.a] = win_rate_b
            except ZeroDivisionError:
                matrix.at[row.b, row.a] = _as_float(np.nan)
        elif hasattr(row, "win_rate") and row.b in matrix.index and row.a in matrix.columns:
            matrix.at[row.b, row.a] = 1.0 - _as_float(row.win_rate)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.set_size_inches(8 + max(0, len(order) - 10) * 0.3, 7 + max(0, len(order) - 10) * 0.3)
    im = ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_title(f"Head-to-head win rates for {players}-player games")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Win probability (row vs column)")
    fig.tight_layout()

    with atomic_path(str(output)) as tmp_path:
        fig.savefig(tmp_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(
        "Head-to-head heatmap written",
        extra={"stage": "report", "players": players, "path": str(output)},
    )
    return output


def plot_feature_importance_for_players(
    cfg: AnalysisConfig | AppConfig, players: int, *, force: bool = False
) -> Path | None:
    """Render a horizontal bar chart of feature importances."""

    analysis_dir = _analysis_dir(cfg)
    source = analysis_dir / f"feature_importance_{players}p.parquet"
    if not source.exists():
        return None

    df = _load_feature_importance(analysis_dir, players)
    if df.empty:
        return None
    top = df.head(FEATURE_TOP_N)
    output = _plot_output_path(cfg, players, f"feature_importance_{players}p.png")
    if _output_is_fresh(output, [source], force=force):
        return output

    fig = plt.figure()
    ax = _prepare_axis(fig, len(top))
    positions = np.arange(len(top))[::-1]
    ax.barh(positions, top["importance"], color="#2ca02c")
    ax.set_yticks(positions)
    ax.set_yticklabels(top["feature"].tolist())
    ax.set_xlabel("Relative importance")
    ax.set_title(f"Feature importance ({players}-player HGB)")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    with atomic_path(str(output)) as tmp_path:
        fig.savefig(tmp_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(
        "Feature importance plot written",
        extra={"stage": "report", "players": players, "path": str(output)},
    )
    return output


def plot_seed_variability_for_players(
    cfg: AnalysisConfig | AppConfig, players: int, *, force: bool = False
) -> Path | None:
    """Visualise seed-to-seed variability for the top strategies."""

    analysis_dir = _analysis_dir(cfg)
    layout = _analysis_layout(cfg)
    pattern = list(analysis_dir.glob(f"strategy_summary_{players}p_seed*.parquet"))
    if len(pattern) <= 1:
        return None

    seeds_df = _load_seed_summaries(analysis_dir, players)
    if seeds_df.empty:
        return None

    available = seeds_df["strategy_id"].unique().tolist()
    meta_df = _load_meta_summary(cfg, players)
    if not meta_df.empty:
        preferred = [s for s in meta_df["strategy"].tolist() if s in available]
    else:
        preferred = _load_ratings(analysis_dir, players, layout=layout)["strategy"].tolist()
    chosen = preferred[:SEED_TOP_N]
    if not chosen:
        return None

    filtered = seeds_df[seeds_df["strategy_id"].isin(chosen)].copy()
    grouped = filtered.groupby("strategy_id")
    summary = grouped["win_rate"].agg(["mean", "std", "count"]).reset_index()
    summary.rename(columns={"strategy_id": "strategy"}, inplace=True)
    summary["sem"] = summary.apply(
        lambda row: 0.0 if row["count"] <= 1 else row["std"] / math.sqrt(row["count"]),
        axis=1,
    )
    summary["ci"] = summary["sem"] * 1.96
    summary.sort_values(
        ["mean", "strategy"], ascending=[False, True], inplace=True, kind="mergesort"
    )

    output = _plot_output_path(cfg, players, f"seed_forest_{players}p.png")
    inputs = list(pattern)
    if _output_is_fresh(output, inputs, force=force):
        return output

    fig = plt.figure()
    ax = _prepare_axis(fig, len(summary), base_height=4.0)
    positions = np.arange(len(summary))[::-1]
    ax.errorbar(
        summary["mean"],
        positions,
        xerr=summary["ci"],
        fmt="o",
        color="#d62728",
        ecolor="#d62728",
        capsize=4,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(summary["strategy"].tolist())
    ax.set_xlabel("Win rate across seeds (mean ± 95% CI)")
    ax.set_xlim(0, 1)
    ax.set_title(f"Seed variability for top {len(summary)} strategies ({players}-player)")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    with atomic_path(str(output)) as tmp_path:
        fig.savefig(tmp_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(
        "Seed variability plot written",
        extra={"stage": "report", "players": players, "path": str(output)},
    )
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report generation
# ─────────────────────────────────────────────────────────────────────────────


def _format_rate(value: float) -> str:
    """Format a win rate with three decimals, preserving ``nan`` text."""
    return f"{value:.3f}" if not math.isnan(value) else "nan"


def _top_strategy_bullets(ratings: pd.DataFrame, meta_summary: pd.DataFrame) -> list[str]:
    """Summarize top-performing strategies for the executive summary."""
    items: list[str] = []
    top_ratings = ratings.head(3)
    for row in top_ratings.itertuples(index=False):
        sigma = _as_float(row.sigma)
        mu_value = _as_float(row.mu)
        spread = 2.0 * sigma
        items.append(f"{row.strategy}: μ={mu_value:.2f} ± {spread:.2f}")
    if not meta_summary.empty:
        top_meta = meta_summary.head(3)
        meta_items = [
            f"{row.strategy} ({row.win_rate:.1%})" for row in top_meta.itertuples(index=False)
        ]
        items.append("Win rates: " + ", ".join(meta_items))
    return items


def _top_feature_bullets(features: pd.DataFrame) -> list[str]:
    """Build bullet points highlighting key predictive features."""
    if features.empty:
        return ["Feature importances unavailable"]
    items = [
        f"{row.feature}: {row.importance:.1%}" for row in features.head(3).itertuples(index=False)
    ]
    return items


def _seed_stability_summary(seed_df: pd.DataFrame) -> str:
    """Describe seed-to-seed variability using a qualitative label."""
    seeds = seed_df["seed"].nunique()
    if seeds <= 1:
        return "Single seed available; stability not assessed."
    grouped = seed_df.groupby("strategy_id")["win_rate"].std(ddof=1)
    if grouped.empty or grouped.isna().all():
        return "Seed variability data unavailable."
    median_std = float(grouped.fillna(0).median())
    if median_std < 0.01:
        level = "small"
    elif median_std < 0.03:
        level = "moderate"
    else:
        level = "large"
    return f"Results appear {level} across {seeds} seeds (median σ≈{median_std:.3f})."


def _tier_buckets_from_ranked(ranked: Sequence[str]) -> dict[str, list[str]]:
    """Bucket ordered strategies into fixed-size tier labels."""
    if not ranked:
        return {}
    seen: set[str] = set()
    unique_ranked: list[str] = []
    for strategy in ranked:
        if strategy in seen:
            continue
        seen.add(strategy)
        unique_ranked.append(strategy)
    buckets: dict[str, list[str]] = {}
    start = 0
    for label, size in zip(TIER_LABELS[:-1], TIER_BUCKET_SIZES, strict=False):
        end = min(len(unique_ranked), start + size)
        buckets[label] = unique_ranked[start:end]
        start = end
    buckets[TIER_LABELS[-1]] = unique_ranked[start:]
    return {label: members for label, members in buckets.items() if members}


def _ranked_strategies_for_tiers(artifacts: _ReportArtifacts) -> list[str]:
    """Select the ranking source used for tier list rendering."""
    if not artifacts.h2h_ranking.empty:
        return artifacts.h2h_ranking["strategy"].tolist()
    return artifacts.ratings["strategy"].tolist()


def _tiers_section(tiers: Mapping[str, Sequence[str]]) -> str:
    """Render tier membership as Markdown bullet lines."""
    if not tiers:
        return "No tier information available."
    lines = []
    for label in TIER_LABELS:
        strategies = tiers.get(label, [])
        if not strategies:
            continue
        joined = ", ".join(strategies)
        lines.append(f"- {label}: {joined}")
    return "\n".join(lines)


def _s_tiers_section(s_tiers: Mapping[str, str]) -> list[str]:
    """Render the S+/S/S- breakdown for head-to-head candidates."""
    if not s_tiers:
        return []
    buckets: dict[str, list[str]] = {"S+": [], "S": [], "S-": []}
    for strategy, label in s_tiers.items():
        if label in buckets:
            buckets[label].append(strategy)
    lines = ["### Head-to-head S-tier breakdown"]
    for label in ("S+", "S", "S-"):
        strategies = sorted(buckets[label])
        if not strategies:
            continue
        lines.append(f"- {label}: {', '.join(strategies)}")
    return lines


def _build_report_body(
    players: int, artifacts: _ReportArtifacts, plot_paths: dict[str, Path | None]
) -> str:
    """Assemble the Markdown body for a per-player-count report."""
    ratings = artifacts.ratings
    meta_summary = artifacts.meta_summary
    features = artifacts.feature_importance
    seed_df = artifacts.seed_summaries
    heterogeneity = artifacts.heterogeneity
    run_meta = artifacts.run_metadata
    s_tiers = artifacts.h2h_s_tiers
    tier_ranking = _ranked_strategies_for_tiers(artifacts)
    tier_buckets = _tier_buckets_from_ranked(tier_ranking)

    n_strategies = len(ratings)
    n_seeds = (
        int(meta_summary["n_seeds"].max())
        if "n_seeds" in meta_summary.columns
        else seed_df["seed"].nunique()
    )
    n_seeds = int(n_seeds) if n_seeds else max(1, seed_df["seed"].nunique())
    top_tier = len(tier_buckets.get("S", [])) if tier_buckets else 0

    meta_line = [
        f"- config_hash: {run_meta.get('config_hash', 'unknown')}",
        f"- git_commit: {run_meta.get('git_commit', 'unknown')}",
        f"- run_timestamp: {run_meta.get('run_timestamp', 'unknown')}",
        f"- n_strategies: {n_strategies}",
        f"- n_seeds: {n_seeds}",
        f"- top_tier_size: {top_tier if top_tier else 'n/a'}",
    ]

    hetero_lines = []
    for key in sorted(heterogeneity):
        hetero_lines.append(f"    - {key}: {heterogeneity[key]:.3f}")

    exec_summary = []
    exec_summary.extend(_top_strategy_bullets(ratings, meta_summary))
    exec_summary.extend(_top_feature_bullets(features))
    exec_summary.append(_seed_stability_summary(seed_df))

    ladder_rows = []
    for row in ratings.head(5).itertuples(index=False):
        mu_value = _as_float(row.mu)
        sigma = _as_float(row.sigma)
        ci_lo = mu_value - 2 * sigma
        ci_hi = mu_value + 2 * sigma
        ladder_rows.append(f"| {row.strategy} | {mu_value:.2f} | [{ci_lo:.2f}, {ci_hi:.2f}] |")

    ladder_table = "\n".join(["| Strategy | μ | μ ± 2σ |", "| --- | --- | --- |"] + ladder_rows)

    feature_lines = []
    if features.empty:
        feature_lines.append("- Feature importances unavailable")
    else:
        for row in features.head(5).itertuples(index=False):
            feature_lines.append(f"- {row.feature}: {row.importance:.1%}")

    stability_line = _seed_stability_summary(seed_df)
    if plot_paths.get("seed") is None:
        stability_line += " (plot not generated)"

    h2h_section: str
    if plot_paths.get("h2h"):
        h2h_section = f"![Head-to-head](plots/{players}p/h2h_heatmap_{players}p.png)"
    else:
        h2h_section = "Head-to-head results not available for this player count."

    feature_img = (
        f"![Feature importance](plots/{players}p/feature_importance_{players}p.png)"
        if plot_paths.get("features")
        else "Feature importance data not available."
    )

    seed_img = (
        f"![Seed variability](plots/{players}p/seed_forest_{players}p.png)"
        if plot_paths.get("seed")
        else "Seed variability plot not generated."
    )

    ladder_img = f"![Ladder](plots/{players}p/ladder_{players}p.png)"

    report_lines = [
        f"# {players}-player Farkle strategy report",
        "",
        *meta_line,
        "",
        "## Executive summary",
        *[f"- {line}" for line in exec_summary],
        "",
        "## Ladder: top strategies",
        ladder_img,
        "",
        ladder_table,
        "",
        "## Head-to-head comparisons",
        h2h_section,
        "",
        "## Which features matter?",
        feature_img,
        "",
        *feature_lines,
        "",
        "## Stability across seeds",
        seed_img,
        "",
        stability_line,
        "",
        "## Tiers and rankings",
        _tiers_section(tier_buckets),
    ]
    s_tier_lines = _s_tiers_section(s_tiers)
    if s_tier_lines:
        report_lines.extend(["", *s_tier_lines])
    if hetero_lines:
        report_lines.extend(["", "### Meta-analysis metrics", *hetero_lines])
    return "\n".join(report_lines).rstrip() + "\n"


def generate_report_for_players(
    cfg: AnalysisConfig | AppConfig, players: int, *, force: bool = False
) -> Path:
    """Generate the Markdown report and associated figures for ``players``."""

    artifacts = _gather_artifacts(cfg, players)
    ladder_path = plot_ladder_for_players(cfg, players, force=force)
    h2h_path = plot_h2h_heatmap_for_players(cfg, players, force=force)
    feature_path = plot_feature_importance_for_players(cfg, players, force=force)
    seed_path = plot_seed_variability_for_players(cfg, players, force=force)

    plot_paths: dict[str, Path | None] = {
        "ladder": ladder_path,
        "h2h": h2h_path,
        "features": feature_path,
        "seed": seed_path,
    }

    report_body = _build_report_body(players, artifacts, plot_paths)
    analysis_dir = _analysis_dir(cfg)
    layout = _analysis_layout(cfg)
    report_path = analysis_dir / f"report_{players}p.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = [
        _ratings_path(analysis_dir, layout=layout),
        analysis_dir / f"strategy_summary_{players}p_meta.parquet",
    ]
    h2h_ranking_path = _post_h2h_path(
        analysis_dir, "h2h_significant_ranking.csv", layout=layout
    )
    if h2h_ranking_path.exists():
        inputs.append(h2h_ranking_path)
    h2h_s_tiers_path = _post_h2h_path(analysis_dir, "h2h_s_tiers.json", layout=layout)
    if h2h_s_tiers_path.exists():
        inputs.append(h2h_s_tiers_path)
    if _output_is_fresh(report_path, inputs, force=force):
        return report_path

    with atomic_path(str(report_path)) as tmp_path:
        Path(tmp_path).write_text(report_body)
    LOGGER.info(
        "Markdown report written",
        extra={"stage": "report", "players": players, "path": str(report_path)},
    )
    return report_path


def run_report(cfg: AnalysisConfig | AppConfig, *, force: bool = False) -> None:
    """Generate reports for every discovered player count."""

    analysis_dir = _analysis_dir(cfg)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    layout = _analysis_layout(cfg)
    players_list = _sim_player_counts(cfg, analysis_dir, layout=layout)
    LOGGER.info(
        "Report generation starting",
        extra={"stage": "report", "players": players_list},
    )
    for players in players_list:
        try:
            generate_report_for_players(cfg, players, force=force)
        except ReportError as exc:
            LOGGER.warning(
                "Skipping report due to missing artifacts",
                extra={"stage": "report", "players": players, "error": str(exc)},
            )
    LOGGER.info("Report generation complete", extra={"stage": "report"})
