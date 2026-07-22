"""Canonical within-k seat effects and explicitly secondary cross-k diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis.all_player_metrics import ATTEMPT_CONDITIONING
from farkle.config import AppConfig, ArtifactScope
from farkle.game.engine import TerminationStatus
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.streaming_loop import run_streaming_shard

_COUNT_SCHEMA: Final = pa.schema(
    [
        pa.field("root_seed", pa.int64(), nullable=False),
        pa.field("k", pa.int16(), nullable=False),
        pa.field("deterministic_batch_id", pa.int32(), nullable=False),
        pa.field("strategy", pa.int32(), nullable=False),
        pa.field("seat", pa.int16(), nullable=False),
        pa.field("raw_wins", pa.int64(), nullable=False),
        pa.field("raw_exposures", pa.int64(), nullable=False),
        pa.field("raw_completed_exposures", pa.int64(), nullable=False),
        pa.field("raw_safety_limit_exposures", pa.int64(), nullable=False),
    ]
)

_STANDARDIZED_COLUMNS: Final = [
    "root_seed",
    "effect_scope",
    "strategy",
    "seat",
    "common_k_support",
    "standardized_seat_effect",
]
_MIXTURE_COLUMNS: Final = [
    "root_seed",
    "effect_scope",
    "strategy",
    "seat",
    "common_k_support",
    "raw_wins",
    "raw_exposures",
    "raw_completed_exposures",
    "raw_safety_limit_exposures",
    "exposure_weighted_baseline",
    "exposure_weighted_seat_effect",
]
_SELFPLAY_COLUMNS: Final = [
    "root_seed",
    "k",
    "strategy",
    "p1_wins",
    "games_attempted",
    "games_completed",
    "games_safety_limit",
    "p1_win_rate_per_attempt",
    "p1_win_rate_given_completion",
    "p1_effect_vs_chance",
]
_MIRRORED_COLUMNS: Final = [
    "root_seed",
    "k",
    "strategy_a",
    "strategy_b",
    "paired_mirrored_games",
    "games_attempted",
    "games_completed",
    "games_safety_limit",
    "unpaired_forward_games",
    "unpaired_reverse_games",
    "mean_p1_win_difference",
]


@dataclass(frozen=True)
class SeatAnalysisArtifacts:
    """Paths written by canonical seat analysis."""

    batch_counts: tuple[Path, ...]
    by_k: tuple[Path, ...]
    population_by_k: tuple[Path, ...]
    standardized_across_k: Path
    exposure_mixture_diagnostic: Path
    selfplay_diagnostic: Path
    mirrored_diagnostic: Path

    @property
    def all_paths(self) -> tuple[Path, ...]:
        return (
            *self.batch_counts,
            *self.by_k,
            *self.population_by_k,
            self.standardized_across_k,
            self.exposure_mixture_diagnostic,
            self.selfplay_diagnostic,
            self.mirrored_diagnostic,
        )


def _source_columns(k: int) -> list[str]:
    return [
        "root_seed",
        "k",
        "deterministic_batch_id",
        "shuffle_index",
        "game_index",
        "winner_seat",
        "termination_status",
        *(f"P{seat}_strategy" for seat in range(1, k + 1)),
    ]


def _iter_seat_count_tables(source: Path, k: int) -> Iterator[pa.Table]:
    parquet_file = pq.ParquetFile(source)
    columns = _source_columns(k)
    missing = sorted(set(columns).difference(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"{source} lacks canonical seat-analysis columns: {missing}")
    coordinate: tuple[int, int, int] | None = None
    counts: defaultdict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0, 0, 0])

    def _flush() -> pa.Table | None:
        if coordinate is None or not counts:
            return None
        root_seed, row_k, batch_id = coordinate
        rows = [
            {
                "root_seed": root_seed,
                "k": row_k,
                "deterministic_batch_id": batch_id,
                "strategy": strategy,
                "seat": seat,
                "raw_wins": values[0],
                "raw_exposures": values[1],
                "raw_completed_exposures": values[2],
                "raw_safety_limit_exposures": values[3],
            }
            for (strategy, seat), values in sorted(counts.items())
        ]
        return pa.Table.from_pylist(rows, schema=_COUNT_SCHEMA)

    for batch in parquet_file.iter_batches(columns=columns):
        for row in batch.to_pylist():
            current = (
                int(row["root_seed"]),
                int(row["k"]),
                int(row["deterministic_batch_id"]),
            )
            if current[1] != k:
                raise ValueError(f"{source} contains k={current[1]} in canonical k={k} input")
            if coordinate is not None and current < coordinate:
                raise ValueError(f"{source} is not ordered by root, k, and deterministic batch")
            if coordinate is not None and current != coordinate:
                table = _flush()
                if table is not None:
                    yield table
                counts = defaultdict(lambda: [0, 0, 0, 0])
            coordinate = current
            try:
                status = TerminationStatus(row["termination_status"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{source} contains invalid termination status") from exc
            winner = row["winner_seat"]
            if status is TerminationStatus.SAFETY_LIMIT and winner is not None:
                raise ValueError(f"{source} credits a safety-limit winner")
            for seat in range(1, k + 1):
                strategy_value = row[f"P{seat}_strategy"]
                if strategy_value is None:
                    raise ValueError(f"{source} has a missing strategy exposure in seat {seat}")
                cell = counts[(int(strategy_value), seat)]
                cell[0] += int(winner == f"P{seat}")
                cell[1] += 1
                cell[2] += int(status is TerminationStatus.COMPLETED)
                cell[3] += int(status is TerminationStatus.SAFETY_LIMIT)
    table = _flush()
    if table is not None:
        yield table


def _write_batch_counts(cfg: AppConfig, k: int, source: Path, output: Path) -> None:
    manifest = output.with_suffix(".manifest.jsonl")
    manifest.unlink(missing_ok=True)
    sidecar = make_artifact_sidecar(
        cfg,
        output,
        producer="seat_analysis",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation="aggregate_seat_batch_exposures",
        baseline="chance_1_over_k",
        weighted_quantity="seat_win_indicator",
        support_count_role="raw_player_game_exposures",
        replication_unit="deterministic_shuffle_batch",
        conditioning=ATTEMPT_CONDITIONING,
        consistency_columns=_COUNT_SCHEMA.names,
        source_artifacts=[source],
        grouping_keys=["root_seed", "k", "deterministic_batch_id", "strategy", "seat"],
        player_counts=[k],
        required_player_counts=[k],
        missing_cell_policy="fail",
    )
    run_streaming_shard(
        out_path=str(output),
        manifest_path=str(manifest),
        schema=_COUNT_SCHEMA,
        batch_iter=_iter_seat_count_tables(source, k),
        row_group_size=cfg.row_group_size,
        compression=cfg.parquet_codec,
        manifest_extra={"root_seed": cfg.sim.seed, "k": k},
        sidecar=sidecar,
    )


def _validate_source(path: Path, k: int) -> int:
    """Validate a canonical by-k row partition and return its single root."""

    validate_artifact_sidecar(
        path,
        expected={
            "scope": ArtifactScope.BY_K.value,
            "source_scope": ArtifactScope.BY_K.value,
            "operation": "concatenate_rows_within_k",
            "player_counts": [k],
            "required_player_counts": [k],
            "missing_cell_policy": "fail",
        },
    )
    columns = _source_columns(k)
    schema = pq.read_schema(path)
    missing = sorted(set(columns).difference(schema.names))
    if missing:
        raise ValueError(f"{path} lacks canonical seat-analysis columns: {missing}")
    observed_k_set: set[int] = set()
    roots_set: set[int] = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=["root_seed", "k"]):
        for row in batch.to_pylist():
            if row["root_seed"] is not None:
                roots_set.add(int(row["root_seed"]))
            if row["k"] is not None:
                observed_k_set.add(int(row["k"]))
    observed_k = sorted(observed_k_set)
    if observed_k != [k]:
        raise ValueError(f"{path} has k support {observed_k}, expected [{k}]")
    roots = sorted(roots_set)
    if len(roots) != 1:
        raise ValueError(f"{path} must contain exactly one root, found {roots}")
    return roots[0]


def _within_k_frames(counts: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (
        counts["raw_exposures"]
        == counts["raw_completed_exposures"] + counts["raw_safety_limit_exposures"]
    ).all():
        raise ValueError("seat counts violate attempted exposure conservation")
    if (counts["raw_wins"] > counts["raw_completed_exposures"]).any():
        raise ValueError("seat counts credit a win outside completed exposure support")
    grouped = (
        counts.groupby(["root_seed", "k", "strategy", "seat"], as_index=False)
        .agg(
            raw_wins=("raw_wins", "sum"),
            raw_exposures=("raw_exposures", "sum"),
            raw_completed_exposures=("raw_completed_exposures", "sum"),
            raw_safety_limit_exposures=("raw_safety_limit_exposures", "sum"),
        )
        .sort_values(["strategy", "seat"])
    )
    grouped["chance_baseline"] = 1.0 / k
    grouped["win_rate"] = grouped["raw_wins"] / grouped["raw_exposures"]
    grouped["win_rate_per_attempt"] = grouped["win_rate"]
    grouped["win_rate_given_completion"] = grouped["raw_wins"].div(
        grouped["raw_completed_exposures"].replace(0, pd.NA)
    )
    grouped["safety_limit_exposure_rate"] = (
        grouped["raw_safety_limit_exposures"] / grouped["raw_exposures"]
    )
    grouped["raw_losses"] = grouped["raw_exposures"] - grouped["raw_wins"]
    grouped["seat_effect"] = grouped["win_rate"] - grouped["chance_baseline"]
    population = (
        counts.groupby(["root_seed", "k", "seat"], as_index=False)
        .agg(
            raw_wins=("raw_wins", "sum"),
            raw_exposures=("raw_exposures", "sum"),
            raw_completed_exposures=("raw_completed_exposures", "sum"),
            raw_safety_limit_exposures=("raw_safety_limit_exposures", "sum"),
        )
        .sort_values("seat")
    )
    population["chance_baseline"] = 1.0 / k
    population["win_rate"] = population["raw_wins"] / population["raw_exposures"]
    population["win_rate_per_attempt"] = population["win_rate"]
    population["win_rate_given_completion"] = population["raw_wins"].div(
        population["raw_completed_exposures"].replace(0, pd.NA)
    )
    population["safety_limit_exposure_rate"] = (
        population["raw_safety_limit_exposures"] / population["raw_exposures"]
    )
    population["raw_losses"] = population["raw_exposures"] - population["raw_wins"]
    population["seat_effect"] = population["win_rate"] - population["chance_baseline"]
    return grouped, population


def _declared_weights(cfg: AppConfig, ks: list[int]) -> tuple[dict[int, float], str, str]:
    if cfg.k_aggregation.method == "equal-k":
        weights = {k: 1.0 / len(ks) for k in ks}
        return weights, "equal_k_mean", "equal_k"
    configured = cfg.k_aggregation.k_weights or {}
    if set(configured) != set(ks):
        raise ValueError("declared seat standardization weights must cover every configured k")
    return (
        {int(k): float(weight) for k, weight in configured.items()},
        "declared_k_weighted_mean",
        "declared_mapping",
    )


def _standardized_frames(
    cfg: AppConfig,
    by_k: dict[int, pd.DataFrame],
    population_by_k: dict[int, pd.DataFrame],
    ks: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights, _, _ = _declared_weights(cfg, ks)
    common_seats = range(1, min(ks) + 1)
    common_strategies = set.intersection(
        *(set(frame["strategy"].astype(int)) for frame in by_k.values())
    )
    standardized: list[dict[str, Any]] = []
    mixture: list[dict[str, Any]] = []
    for strategy in sorted(common_strategies):
        for seat in common_seats:
            cells = [
                by_k[k].loc[(by_k[k]["strategy"] == strategy) & (by_k[k]["seat"] == seat)]
                for k in ks
            ]
            if any(cell.empty for cell in cells):
                continue
            effect = sum(
                float(cell.iloc[0]["seat_effect"]) * weights[k]
                for k, cell in zip(ks, cells, strict=True)
            )
            wins = sum(int(cell.iloc[0]["raw_wins"]) for cell in cells)
            exposures = sum(int(cell.iloc[0]["raw_exposures"]) for cell in cells)
            completed_exposures = sum(
                int(cell.iloc[0]["raw_completed_exposures"]) for cell in cells
            )
            safety_exposures = sum(
                int(cell.iloc[0]["raw_safety_limit_exposures"]) for cell in cells
            )
            baseline_mass = sum(
                int(cell.iloc[0]["raw_exposures"]) / k for k, cell in zip(ks, cells, strict=True)
            )
            standardized.append(
                {
                    "root_seed": int(cells[0].iloc[0]["root_seed"]),
                    "effect_scope": "strategy",
                    "strategy": strategy,
                    "seat": seat,
                    "common_k_support": ks,
                    "standardized_seat_effect": effect,
                }
            )
            mixture.append(
                {
                    "root_seed": int(cells[0].iloc[0]["root_seed"]),
                    "effect_scope": "strategy",
                    "strategy": strategy,
                    "seat": seat,
                    "common_k_support": ks,
                    "raw_wins": wins,
                    "raw_exposures": exposures,
                    "raw_completed_exposures": completed_exposures,
                    "raw_safety_limit_exposures": safety_exposures,
                    "exposure_weighted_baseline": baseline_mass / exposures,
                    "exposure_weighted_seat_effect": wins / exposures - baseline_mass / exposures,
                }
            )
    for seat in common_seats:
        cells = [population_by_k[k].loc[population_by_k[k]["seat"] == seat] for k in ks]
        if any(cell.empty for cell in cells):
            continue
        standardized.append(
            {
                "root_seed": int(cells[0].iloc[0]["root_seed"]),
                "effect_scope": "population",
                "strategy": None,
                "seat": seat,
                "common_k_support": ks,
                "standardized_seat_effect": sum(
                    float(cell.iloc[0]["seat_effect"]) * weights[k]
                    for k, cell in zip(ks, cells, strict=True)
                ),
            }
        )
        wins = sum(int(cell.iloc[0]["raw_wins"]) for cell in cells)
        exposures = sum(int(cell.iloc[0]["raw_exposures"]) for cell in cells)
        completed_exposures = sum(
            int(cell.iloc[0]["raw_completed_exposures"]) for cell in cells
        )
        safety_exposures = sum(
            int(cell.iloc[0]["raw_safety_limit_exposures"]) for cell in cells
        )
        baseline_mass = sum(
            int(cell.iloc[0]["raw_exposures"]) / k for k, cell in zip(ks, cells, strict=True)
        )
        mixture.append(
            {
                "root_seed": int(cells[0].iloc[0]["root_seed"]),
                "effect_scope": "population",
                "strategy": None,
                "seat": seat,
                "common_k_support": ks,
                "raw_wins": wins,
                "raw_exposures": exposures,
                "raw_completed_exposures": completed_exposures,
                "raw_safety_limit_exposures": safety_exposures,
                "exposure_weighted_baseline": baseline_mass / exposures,
                "exposure_weighted_seat_effect": wins / exposures - baseline_mass / exposures,
            }
        )
    standardized_frame = pd.DataFrame(standardized, columns=_STANDARDIZED_COLUMNS)
    mixture_frame = pd.DataFrame(mixture, columns=_MIXTURE_COLUMNS)
    standardized_frame["strategy"] = pd.array(
        standardized_frame["strategy"].tolist(), dtype="Int64"
    )
    mixture_frame["strategy"] = pd.array(mixture_frame["strategy"].tolist(), dtype="Int64")
    return standardized_frame, mixture_frame


def _game_diagnostics(sources: dict[int, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selfplay: defaultdict[tuple[int, int, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    mirrored: defaultdict[tuple[int, int, int], list[int]] = defaultdict(lambda: [0, 0])
    unmatched: defaultdict[tuple[int, int, int], list[int]] = defaultdict(lambda: [0, 0])
    safety: Counter[tuple[int, int, int]] = Counter()
    for k, source in sources.items():
        columns = _source_columns(k)
        pending: defaultdict[tuple[int, int, int, int], dict[int, deque[int]]] = defaultdict(
            lambda: {0: deque(), 1: deque()}
        )
        for batch in pq.ParquetFile(source).iter_batches(columns=columns):
            for row in batch.to_pylist():
                root = int(row["root_seed"])
                strategies = tuple(int(row[f"P{seat}_strategy"]) for seat in range(1, k + 1))
                status = TerminationStatus(row["termination_status"])
                if status is TerminationStatus.SAFETY_LIMIT and row["winner_seat"] is not None:
                    raise ValueError(f"{source} credits a safety-limit winner")
                p1_win = int(row["winner_seat"] == "P1")
                if len(set(strategies)) == 1:
                    cell = selfplay[(root, k, strategies[0])]
                    cell[0] += p1_win
                    cell[1] += 1
                    cell[2] += int(status is TerminationStatus.SAFETY_LIMIT)
                if k != 2 or strategies[0] == strategies[1]:
                    continue
                a, b = sorted(strategies)
                if status is TerminationStatus.SAFETY_LIMIT:
                    safety[(root, a, b)] += 1
                    continue
                orientation = int(strategies == (b, a))
                key = (root, int(row["deterministic_batch_id"]), a, b)
                opposite = 1 - orientation
                if pending[key][opposite]:
                    other = pending[key][opposite].popleft()
                    forward, reverse = (other, p1_win) if orientation == 1 else (p1_win, other)
                    pair = mirrored[(root, a, b)]
                    pair[0] += forward - reverse
                    pair[1] += 1
                else:
                    pending[key][orientation].append(p1_win)
        for (root, _, a, b), orientations in pending.items():
            cell = unmatched[(root, a, b)]
            cell[0] += len(orientations[0])
            cell[1] += len(orientations[1])
    selfplay_rows: list[dict[str, Any]] = [
        {
            "root_seed": root,
            "k": k,
            "strategy": strategy,
            "p1_wins": values[0],
            "games_attempted": values[1],
            "games_completed": values[1] - values[2],
            "games_safety_limit": values[2],
            "p1_win_rate_per_attempt": values[0] / values[1],
            "p1_win_rate_given_completion": (
                values[0] / (values[1] - values[2]) if values[1] > values[2] else None
            ),
            "p1_effect_vs_chance": values[0] / values[1] - 1.0 / k,
        }
        for (root, k, strategy), values in sorted(selfplay.items())
    ]
    mirrored_rows: list[dict[str, Any]] = [
        {
            "root_seed": root,
            "k": 2,
            "strategy_a": a,
            "strategy_b": b,
            "paired_mirrored_games": values[1],
            "games_attempted": (
                2 * values[1]
                + unmatched[(root, a, b)][0]
                + unmatched[(root, a, b)][1]
                + safety[(root, a, b)]
            ),
            "games_completed": (
                2 * values[1] + unmatched[(root, a, b)][0] + unmatched[(root, a, b)][1]
            ),
            "games_safety_limit": safety[(root, a, b)],
            "unpaired_forward_games": unmatched[(root, a, b)][0],
            "unpaired_reverse_games": unmatched[(root, a, b)][1],
            "mean_p1_win_difference": values[0] / values[1],
        }
        for (root, a, b), values in sorted(mirrored.items())
        if values[1]
    ]
    for (root, a, b), values in sorted(unmatched.items()):
        if (root, a, b) in mirrored:
            continue
        mirrored_rows.append(
            {
                "root_seed": root,
                "k": 2,
                "strategy_a": a,
                "strategy_b": b,
                "paired_mirrored_games": 0,
                "games_attempted": values[0] + values[1] + safety[(root, a, b)],
                "games_completed": values[0] + values[1],
                "games_safety_limit": safety[(root, a, b)],
                "unpaired_forward_games": values[0],
                "unpaired_reverse_games": values[1],
                "mean_p1_win_difference": None,
            }
        )
    for (root, a, b), safety_games in sorted(safety.items()):
        if (root, a, b) in mirrored or (root, a, b) in unmatched:
            continue
        mirrored_rows.append(
            {
                "root_seed": root,
                "k": 2,
                "strategy_a": a,
                "strategy_b": b,
                "paired_mirrored_games": 0,
                "games_attempted": safety_games,
                "games_completed": 0,
                "games_safety_limit": safety_games,
                "unpaired_forward_games": 0,
                "unpaired_reverse_games": 0,
                "mean_p1_win_difference": None,
            }
        )
    return (
        pd.DataFrame(selfplay_rows, columns=_SELFPLAY_COLUMNS),
        pd.DataFrame(mirrored_rows, columns=_MIRRORED_COLUMNS),
    )


def _write_frame(
    cfg: AppConfig,
    frame: pd.DataFrame,
    path: Path,
    *,
    scope: ArtifactScope,
    operation: str,
    sources: list[Path],
    ks: list[int],
    grouping_keys: list[str],
    k_method: str = "none",
    k_weights: dict[int, float] | None = None,
    missing_cell_policy: str = "fail",
    conditioning: str = ATTEMPT_CONDITIONING,
) -> None:
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="seat_analysis",
        scope=scope,
        source_scope=ArtifactScope.BY_K,
        operation=operation,
        baseline="chance_1_over_k",
        weighted_quantity="seat_win_indicator",
        k_aggregation_method=k_method,
        k_weights=k_weights,
        support_count_role="raw_player_game_exposures",
        uncertainty_method="descriptive",
        replication_unit="deterministic_shuffle_batch",
        conditioning=conditioning,
        consistency_columns=frame.columns.tolist(),
        source_artifacts=sources,
        grouping_keys=grouping_keys,
        player_counts=ks,
        required_player_counts=ks,
        missing_cell_policy=missing_cell_policy,
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        sidecar=sidecar,
        codec=cfg.parquet_codec,
    )


def build_canonical_seat_analysis(cfg: AppConfig, *, force: bool = False) -> SeatAnalysisArtifacts:
    """Build within-k seat effects and clearly labelled secondary diagnostics."""

    ks = sorted({int(k) for k in cfg.sim.n_players_list})
    sources = {k: cfg.combined_rows_by_k(k) for k in ks}
    missing = [path for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"canonical per-k seat inputs are missing: {missing}")
    roots = {_validate_source(path, k) for k, path in sources.items()}
    if roots != {int(cfg.sim.seed)}:
        raise ValueError(
            "seat analysis requires identical configured-root support; "
            f"expected [{cfg.sim.seed}], found {sorted(roots)}"
        )
    artifacts = SeatAnalysisArtifacts(
        batch_counts=tuple(cfg.seat_batch_counts_path(k) for k in ks),
        by_k=tuple(cfg.seat_effects_by_k_path(k) for k in ks),
        population_by_k=tuple(cfg.seat_population_by_k_path(k) for k in ks),
        standardized_across_k=cfg.seat_standardized_across_k_path(),
        exposure_mixture_diagnostic=cfg.seat_exposure_mixture_diagnostic_path(),
        selfplay_diagnostic=cfg.seat_selfplay_diagnostic_path(),
        mirrored_diagnostic=cfg.seat_mirrored_diagnostic_path(),
    )
    done = stage_done_path(cfg.metrics_stage_dir, "canonical_seat_analysis")
    if not force and stage_is_up_to_date(
        done,
        inputs=list(sources.values()),
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=list(artifacts.all_paths),
    ):
        return artifacts

    by_k: dict[int, pd.DataFrame] = {}
    population_by_k: dict[int, pd.DataFrame] = {}
    for k, count_path, effect_path, population_path in zip(
        ks, artifacts.batch_counts, artifacts.by_k, artifacts.population_by_k, strict=True
    ):
        _write_batch_counts(cfg, k, sources[k], count_path)
        counts = pq.read_table(count_path).to_pandas()
        effects, population = _within_k_frames(counts, k)
        by_k[k] = effects
        population_by_k[k] = population
        _write_frame(
            cfg,
            effects,
            effect_path,
            scope=ArtifactScope.BY_K,
            operation="calculate_strategy_seat_effects",
            sources=[count_path],
            ks=[k],
            grouping_keys=["root_seed", "k", "strategy", "seat"],
        )
        _write_frame(
            cfg,
            population,
            population_path,
            scope=ArtifactScope.BY_K,
            operation="calculate_population_seat_effects",
            sources=[count_path],
            ks=[k],
            grouping_keys=["root_seed", "k", "seat"],
        )

    standardized, mixture = _standardized_frames(cfg, by_k, population_by_k, ks)
    weights, operation, k_method = _declared_weights(cfg, ks)
    sidecar_weights = weights if k_method == "declared_mapping" else None
    _write_frame(
        cfg,
        standardized,
        artifacts.standardized_across_k,
        scope=ArtifactScope.ACROSS_K,
        operation=operation,
        sources=list(artifacts.by_k),
        ks=ks,
        grouping_keys=["root_seed", "effect_scope", "strategy", "seat"],
        k_method=k_method,
        k_weights=sidecar_weights,
        missing_cell_policy="declared_common_support",
    )
    _write_frame(
        cfg,
        mixture,
        artifacts.exposure_mixture_diagnostic,
        scope=ArtifactScope.DIAGNOSTICS,
        operation="within_k_exposure_combination",
        sources=list(artifacts.by_k),
        ks=ks,
        grouping_keys=["root_seed", "effect_scope", "strategy", "seat"],
        missing_cell_policy="declared_common_support",
    )
    selfplay, mirrored = _game_diagnostics(sources)
    _write_frame(
        cfg,
        selfplay,
        artifacts.selfplay_diagnostic,
        scope=ArtifactScope.DIAGNOSTICS,
        operation="calculate_self_play_diagnostics",
        sources=list(sources.values()),
        ks=ks,
        grouping_keys=["root_seed", "k", "strategy"],
    )
    _write_frame(
        cfg,
        mirrored,
        artifacts.mirrored_diagnostic,
        scope=ArtifactScope.DIAGNOSTICS,
        operation="calculate_mirrored_game_diagnostics",
        sources=list(sources.values()),
        ks=ks,
        grouping_keys=["root_seed", "strategy_a", "strategy_b"],
        conditioning='termination_status == "completed"',
    )
    write_stage_done(
        done,
        inputs=list(sources.values()),
        outputs=list(artifacts.all_paths),
        cfg=cfg,
        stage="metrics",
        sidecar_artifacts=list(artifacts.all_paths),
    )
    return artifacts


__all__ = ["SeatAnalysisArtifacts", "build_canonical_seat_analysis"]
