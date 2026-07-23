"""Descriptive TrueSkill screening contribution and model diagnostics."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import trueskill

from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    ensure_artifact_sidecar_atomic,
    make_artifact_sidecar,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.parallel import normalize_n_jobs, resolve_mp_context
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done

_HOLDOUT_FRACTION: Final = 0.2
TRUESKILL_METHOD_VERSION: Final = 3
TRUESKILL_CONDITIONING: Final = (
    "Descriptive TrueSkill screening conditional on games that completed under "
    "the configured safety-round limit; safety-limit attempts are excluded from "
    "rating updates and are reported separately. This is not the canonical "
    "per-attempt tournament win-rate estimand."
)
_EVIDENCE_BACKED: Final = "evidence_backed_completed_games"
_PRIOR_ONLY: Final = "prior_only_unrated"
_RATING_COLUMNS: Final = {
    "strategy",
    "mu",
    "sigma",
    "strategy_attempted_exposures",
    "strategy_completed_exposures",
    "strategy_excluded_safety_limit_exposures",
    "strategy_performed_updates",
    "rating_status",
    "cell_games_attempted",
    "cell_games_completed",
    "cell_games_excluded_safety_limit",
    "cell_performed_updates",
}


def trueskill_method_contract(procedure: str) -> dict[str, Any]:
    """Return the versioned completed-game TrueSkill method identity."""

    return {
        "kind": "trueskill",
        "procedure": procedure,
        "parameters": {
            "method_version": TRUESKILL_METHOD_VERSION,
            "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
            "conditioning": "termination_status == completed",
            "safety_limit_policy": "excluded_without_update_or_rank_imputation",
        },
    }


@dataclass(frozen=True)
class ClassifiedTrueSkillGame:
    """One canonical attempted game classified before any TrueSkill update."""

    termination_status: str
    players: list[str]
    ranks: list[int] | None


def classify_trueskill_row(row: Mapping[str, object], k: int) -> ClassifiedTrueSkillGame:
    """Validate one outcome-schema-v2 row for completed-only TrueSkill use."""

    if row.get("outcome_schema_version") != OUTCOME_SCHEMA_VERSION:
        raise ValueError(f"TrueSkill requires outcome_schema_version={OUTCOME_SCHEMA_VERSION}")
    status = row.get("termination_status")
    if status not in {"completed", "safety_limit"}:
        raise ValueError(f"unsupported TrueSkill termination_status: {status!r}")
    normalized_status = str(status)
    players: list[str] = []
    raw_ranks: list[object] = []
    for seat in range(1, k + 1):
        strategy = row.get(f"P{seat}_strategy")
        if strategy is None:
            raise ValueError(f"TrueSkill row lacks P{seat}_strategy")
        players.append(str(strategy))
        raw_ranks.append(row.get(f"P{seat}_rank"))

    winner = row.get("winner_seat")
    if status == "safety_limit":
        if winner is not None or any(rank is not None for rank in raw_ranks):
            raise ValueError("safety-limit TrueSkill rows must have null winner and null ranks")
        return ClassifiedTrueSkillGame(normalized_status, players, None)

    if winner not in {f"P{seat}" for seat in range(1, k + 1)}:
        raise ValueError("completed TrueSkill row must have a valid winner_seat")
    if any(isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) for rank in raw_ranks):
        raise ValueError("completed TrueSkill row must have integer ranks")
    one_based = [int(cast(int | np.integer, rank)) for rank in raw_ranks]
    if sorted(one_based) != list(range(1, k + 1)):
        raise ValueError("completed TrueSkill ranks must be the permutation 1..k")
    winner_index = int(str(winner)[1:]) - 1
    if one_based[winner_index] != 1:
        raise ValueError("completed TrueSkill winner_seat must identify rank 1")
    return ClassifiedTrueSkillGame(
        normalized_status,
        players,
        [rank - 1 for rank in one_based],
    )


def _validate_rating_support(table: pa.Table, k: int) -> None:
    """Validate cell and strategy support conservation in a rating artifact."""

    frame = table.to_pandas()
    if frame.empty:
        raise ArtifactContractError("TrueSkill rating artifact must contain strategies")
    cell_columns = [
        "cell_games_attempted",
        "cell_games_completed",
        "cell_games_excluded_safety_limit",
        "cell_performed_updates",
    ]
    cell_values: dict[str, int] = {}
    for column in cell_columns:
        values = frame[column].drop_duplicates()
        if len(values) != 1:
            raise ArtifactContractError(f"TrueSkill rating rows disagree on {column}")
        cell_values[column] = int(values.iloc[0])
    attempted = cell_values["cell_games_attempted"]
    completed = cell_values["cell_games_completed"]
    excluded = cell_values["cell_games_excluded_safety_limit"]
    updates = cell_values["cell_performed_updates"]
    if min(cell_values.values()) < 0 or attempted != completed + excluded:
        raise ArtifactContractError("TrueSkill cell support conservation failed")
    if updates > completed:
        raise ArtifactContractError("TrueSkill cell updates exceed completed games")

    strategy_attempted = frame["strategy_attempted_exposures"].to_numpy(dtype=np.int64)
    strategy_completed = frame["strategy_completed_exposures"].to_numpy(dtype=np.int64)
    strategy_excluded = frame["strategy_excluded_safety_limit_exposures"].to_numpy(dtype=np.int64)
    strategy_updates = frame["strategy_performed_updates"].to_numpy(dtype=np.int64)
    if (
        np.any(strategy_attempted < 0)
        or np.any(strategy_completed < 0)
        or np.any(strategy_excluded < 0)
        or np.any(strategy_updates < 0)
        or np.any(strategy_attempted != strategy_completed + strategy_excluded)
        or np.any(strategy_updates > strategy_completed)
    ):
        raise ArtifactContractError("TrueSkill strategy support conservation failed")
    if (
        strategy_attempted.sum() > k * attempted
        or strategy_completed.sum() > k * completed
        or strategy_excluded.sum() > k * excluded
        or strategy_updates.sum() > k * updates
        or (updates and strategy_updates.sum() < 2 * updates)
    ):
        raise ArtifactContractError("TrueSkill exposure support exceeds cell support")
    expected_status = np.where(strategy_updates > 0, _EVIDENCE_BACKED, _PRIOR_ONLY)
    if not np.array_equal(frame["rating_status"].astype(str).to_numpy(), expected_status):
        raise ArtifactContractError("TrueSkill rating status disagrees with performed updates")


@dataclass(frozen=True)
class ScreeningRatingCell:
    """One canonical root/k ratings artifact and its game-row source."""

    root_seed: int
    k: int
    ratings_path: Path
    game_rows_path: Path | None = None


def publish_rating_cell_contract(
    cfg: AppConfig,
    cell: ScreeningRatingCell,
    *,
    completed_artifact_sha256: str | None = None,
    expected_sidecar_sha256: str | None = None,
    code_revision: str = "unknown",
) -> None:
    """Finalize a rating sidecar only from an independent cell completion.

    The completed artifact digest is mandatory when the sidecar is absent, so
    callers cannot authenticate arbitrary existing bytes by constructing new
    metadata for them.  An expected sidecar digest additionally gates the
    missing-sidecar recovery path.
    """

    missing_columns = sorted(_RATING_COLUMNS.difference(pq.read_schema(cell.ratings_path).names))
    if missing_columns:
        raise ArtifactContractError(
            f"TrueSkill method-v{TRUESKILL_METHOD_VERSION} rating support is missing: "
            f"{missing_columns}"
        )
    table = pq.read_table(cell.ratings_path)
    _validate_rating_support(table, cell.k)
    if sidecar_path(cell.ratings_path).exists():
        validate_artifact_sidecar(
            cell.ratings_path,
            expected={
                "scope": ArtifactScope.BY_K.value,
                "operation": "sequential_rating",
                "player_counts": [cell.k],
                "seed_scope": "single_root",
                "conditioning": TRUESKILL_CONDITIONING,
                "method_contract": trueskill_method_contract("sequential_rating"),
            },
        )
        if (
            expected_sidecar_sha256 is not None
            and sha256_file(sidecar_path(cell.ratings_path)) != expected_sidecar_sha256
        ):
            raise ArtifactContractError("rating sidecar differs from cell completion identity")
        return
    if completed_artifact_sha256 is None:
        raise ArtifactContractError(
            "missing rating sidecar requires an independent cell completion identity"
        )
    if sha256_file(cell.ratings_path) != completed_artifact_sha256:
        raise ArtifactContractError("rating bytes differ from cell completion identity")
    sidecar = make_artifact_sidecar(
        cfg,
        cell.ratings_path,
        producer="trueskill",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation="sequential_rating",
        weighted_quantity="trueskill_mu",
        support_count_role="attempted_completed_excluded_and_performed_updates",
        uncertainty_method="trueskill_model_sigma_screening_only",
        replication_unit="ordered_game",
        conditioning=TRUESKILL_CONDITIONING,
        consistency_columns=table.schema.names,
        source_artifacts=[cell.game_rows_path] if cell.game_rows_path is not None else [],
        grouping_keys=["strategy"],
        player_counts=[cell.k],
        required_player_counts=[cell.k],
        missing_cell_policy="fail",
        seed_scope="single_root",
        code_revision=code_revision,
        method_contract=cast(Any, trueskill_method_contract("sequential_rating")),
    )
    ensure_artifact_sidecar_atomic(
        cell.ratings_path,
        sidecar,
        expected={
            "scope": ArtifactScope.BY_K.value,
            "operation": "sequential_rating",
            "player_counts": [cell.k],
            "seed_scope": "single_root",
            "conditioning": TRUESKILL_CONDITIONING,
            "method_contract": trueskill_method_contract("sequential_rating"),
        },
    )
    if (
        expected_sidecar_sha256 is not None
        and sha256_file(sidecar_path(cell.ratings_path)) != expected_sidecar_sha256
    ):
        sidecar_path(cell.ratings_path).unlink(missing_ok=True)
        raise ArtifactContractError("recovered rating sidecar differs from completion identity")


def _load_rating_frame(cell: ScreeningRatingCell) -> pd.DataFrame:
    schema = pq.read_schema(cell.ratings_path)
    required = set(_RATING_COLUMNS)
    missing = sorted(required.difference(schema.names))
    if missing:
        raise ValueError(f"{cell.ratings_path} lacks canonical rating columns: {missing}")
    frame = pq.read_table(
        cell.ratings_path,
        columns=sorted(required),
    ).to_pandas()
    if frame["strategy"].duplicated().any():
        raise ValueError(f"{cell.ratings_path} contains duplicate strategies")
    frame["strategy"] = frame["strategy"].astype(str)
    frame["evidence_backed"] = frame["rating_status"].eq(_EVIDENCE_BACKED)
    frame["root_seed"] = cell.root_seed
    frame["k"] = cell.k
    frame["percentile_rank"] = np.nan
    evidence = frame["evidence_backed"]
    frame.loc[evidence, "percentile_rank"] = frame.loc[evidence, "mu"].rank(
        method="average",
        pct=True,
    )
    return frame


def build_percentile_contribution(
    cfg: AppConfig,
    cells: Sequence[ScreeningRatingCell],
    *,
    force: bool = False,
) -> Path:
    """Average within-cell percentile ranks over complete root/k support."""

    if not cells:
        raise ValueError("TrueSkill candidate contribution requires rating cells")
    coordinates = [(cell.root_seed, cell.k) for cell in cells]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("TrueSkill rating cells contain duplicate root/k coordinates")
    for cell in cells:
        validate_artifact_sidecar(
            cell.ratings_path,
            expected={
                "scope": ArtifactScope.BY_K.value,
                "operation": "sequential_rating",
                "player_counts": [cell.k],
                "seed_scope": "single_root",
                "conditioning": TRUESKILL_CONDITIONING,
                "method_contract": trueskill_method_contract("sequential_rating"),
            },
        )
    output = cfg.trueskill_candidate_contribution_path()
    done = stage_done_path(cfg.trueskill_stage_dir, "trueskill_percentile_contribution")
    inputs = [cell.ratings_path for cell in cells]
    if not force and stage_is_up_to_date(
        done,
        inputs=inputs,
        outputs=[output],
        cfg=cfg,
        stage="trueskill",
        sidecar_artifacts=[output],
    ):
        return output
    frames = [_load_rating_frame(cell) for cell in cells]
    long = pd.concat(frames, ignore_index=True)
    required_cells = len(cells)
    contribution = (
        long.groupby("strategy", as_index=False)
        .agg(
            mean_percentile_rank=("percentile_rank", "mean"),
            rating_cells_present=("percentile_rank", "count"),
            minimum_percentile_rank=("percentile_rank", "min"),
            attempted_exposures_total=("strategy_attempted_exposures", "sum"),
            completed_exposures_total=("strategy_completed_exposures", "sum"),
            excluded_safety_limit_exposures_total=(
                "strategy_excluded_safety_limit_exposures",
                "sum",
            ),
            performed_updates_total=("strategy_performed_updates", "sum"),
        )
        .sort_values(["mean_percentile_rank", "strategy"], ascending=[False, True])
    )
    contribution["rating_cells_required"] = required_cells
    contribution["complete_support"] = (
        contribution["rating_cells_present"] == contribution["rating_cells_required"]
    )
    contribution = contribution.loc[contribution["complete_support"]].copy()
    contribution["candidate_contribution_rank"] = range(1, len(contribution) + 1)
    contribution.reset_index(drop=True, inplace=True)

    table = pa.Table.from_pandas(contribution, preserve_index=False)
    roots = sorted({cell.root_seed for cell in cells})
    ks = sorted({cell.k for cell in cells})
    sidecar = make_artifact_sidecar(
        cfg,
        output,
        producer="trueskill_screening",
        scope=ArtifactScope.ACROSS_K,
        source_scope=ArtifactScope.BY_K,
        operation="equal_root_k_percentile_mean",
        weighted_quantity="within_root_k_mu_percentile_rank",
        k_aggregation_method="equal_k",
        support_count_role="canonical_root_k_rating_cells",
        uncertainty_method="descriptive_screening_only",
        replication_unit="root_k_rating_cell",
        conditioning=TRUESKILL_CONDITIONING,
        consistency_columns=table.schema.names,
        source_artifacts=[cell.ratings_path for cell in cells],
        grouping_keys=["strategy"],
        player_counts=ks,
        required_player_counts=ks,
        missing_cell_policy="fail",
        seed_scope="both_roots_combined" if len(roots) == 2 else "single_root",
        method_contract=cast(
            Any,
            trueskill_method_contract("equal_root_k_percentile_mean"),
        ),
    )
    write_parquet_artifact_atomic(table, output, sidecar=sidecar, codec=cfg.parquet_codec)
    write_stage_done(
        done,
        inputs=inputs,
        outputs=[output],
        cfg=cfg,
        stage="trueskill",
        sidecar_artifacts=[output],
    )
    return output


def _game_columns(k: int) -> list[str]:
    return [
        "termination_status",
        "outcome_schema_version",
        "winner_seat",
        *(f"P{seat}_strategy" for seat in range(1, k + 1)),
        *(f"P{seat}_rank" for seat in range(1, k + 1)),
    ]


def _games(path: Path, k: int, *, reverse: bool = False) -> Iterator[tuple[list[str], list[int]]]:
    parquet_file = pq.ParquetFile(path)
    columns = _game_columns(k)
    missing = sorted(set(columns).difference(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"{path} lacks TrueSkill diagnostic columns: {missing}")
    row_groups = (
        range(parquet_file.num_row_groups - 1, -1, -1)
        if reverse
        else range(parquet_file.num_row_groups)
    )
    for row_group in row_groups:
        rows = parquet_file.read_row_group(row_group, columns=columns).to_pylist()
        iterable = reversed(rows) if reverse else iter(rows)
        for row in iterable:
            game = classify_trueskill_row(row, k)
            if game.ranks is not None:
                yield game.players, game.ranks


def _support_counts(path: Path, k: int) -> dict[str, int]:
    """Count attempted/completed/excluded games with exact conservation."""

    parquet_file = pq.ParquetFile(path)
    columns = _game_columns(k)
    missing = sorted(set(columns).difference(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"{path} lacks TrueSkill support columns: {missing}")
    attempted = 0
    completed = 0
    excluded = 0
    for batch in parquet_file.iter_batches(columns=columns):
        for row in batch.to_pylist():
            game = classify_trueskill_row(row, k)
            attempted += 1
            if game.termination_status == "completed":
                completed += 1
            else:
                excluded += 1
    if attempted != completed + excluded:
        raise ValueError("TrueSkill support conservation failed")
    return {
        "attempted_games": attempted,
        "completed_games": completed,
        "excluded_safety_limit_games": excluded,
    }


def _fit(
    games: Iterator[tuple[list[str], list[int]]],
    *,
    beta: float,
    tau: float,
    draw_probability: float,
    limit: int | None = None,
) -> tuple[dict[str, tuple[float, float]], int]:
    env = trueskill.TrueSkill(beta=beta, tau=tau, draw_probability=draw_probability)
    ratings: dict[str, trueskill.Rating] = {}
    game_count = 0
    for players, ranks in games:
        groups = [(ratings.setdefault(player, env.create_rating()),) for player in players]
        updated = env.rate(groups, ranks=ranks)
        for player, group in zip(players, updated, strict=True):
            ratings[player] = group[0]
        game_count += 1
        if limit is not None and game_count >= limit:
            break
    return {key: (rating.mu, rating.sigma) for key, rating in ratings.items()}, game_count


def _rank_correlation(
    baseline: Mapping[str, tuple[float, float]],
    alternative: Mapping[str, tuple[float, float]],
) -> float | None:
    common = sorted(set(baseline).intersection(alternative))
    if len(common) < 2:
        return None
    baseline_order = sorted(common, key=lambda key: (-baseline[key][0], key))
    alternative_order = sorted(common, key=lambda key: (-alternative[key][0], key))
    baseline_rank = {key: rank for rank, key in enumerate(baseline_order, 1)}
    alternative_rank = {key: rank for rank, key in enumerate(alternative_order, 1)}
    left = np.array([baseline_rank[key] for key in common], dtype=float)
    right = np.array([alternative_rank[key] for key in common], dtype=float)
    if left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _max_mu_shift(
    baseline: Mapping[str, tuple[float, float]],
    alternative: Mapping[str, tuple[float, float]],
) -> float | None:
    common = set(baseline).intersection(alternative)
    if not common:
        return None
    return max(abs(baseline[key][0] - alternative[key][0]) for key in common)


def _heldout_scores(
    path: Path,
    k: int,
    *,
    beta: float,
    tau: float,
    draw_probability: float,
) -> dict[str, float | int | None]:
    game_total = _support_counts(path, k)["completed_games"]
    train_games = max(1, math.floor(game_total * (1.0 - _HOLDOUT_FRACTION))) if game_total else 0
    fitted, observed_train = _fit(
        _games(path, k),
        beta=beta,
        tau=tau,
        draw_probability=draw_probability,
        limit=train_games,
    )
    log_losses: list[float] = []
    brier_scores: list[float] = []
    confidences: list[float] = []
    correct: list[float] = []
    for index, (players, ranks) in enumerate(_games(path, k)):
        if index < observed_train:
            continue
        mus = np.array([fitted.get(player, (25.0, 25.0 / 3.0))[0] for player in players])
        logits = (mus - mus.max()) / max(beta, 1e-12)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        winner_positions = np.flatnonzero(np.asarray(ranks) == min(ranks))
        target: np.ndarray = np.zeros(k, dtype=float)
        target[winner_positions] = 1.0 / len(winner_positions)
        log_losses.append(float(-np.sum(target * np.log(np.maximum(probabilities, 1e-15)))))
        brier_scores.append(float(np.sum((probabilities - target) ** 2)))
        predicted = int(np.argmax(probabilities))
        confidences.append(float(probabilities[predicted]))
        correct.append(float(predicted in winner_positions))
    holdout_games = len(log_losses)
    return {
        "training_games": observed_train,
        "holdout_games": holdout_games,
        "heldout_log_loss": float(np.mean(log_losses)) if log_losses else None,
        "uniform_log_loss": math.log(k) if holdout_games else None,
        "heldout_brier_score": float(np.mean(brier_scores)) if brier_scores else None,
        "uniform_brier_score": 1.0 - 1.0 / k if holdout_games else None,
        "mean_top_probability": float(np.mean(confidences)) if confidences else None,
        "top_prediction_accuracy": float(np.mean(correct)) if correct else None,
        "top_probability_calibration_gap": (
            float(np.mean(confidences) - np.mean(correct)) if confidences else None
        ),
    }


def diagnose_rating_cell(
    cell: ScreeningRatingCell,
    *,
    beta: float,
    tau: float,
    draw_probability: float,
) -> dict[str, object]:
    """Replay one root/k stream for tau, order, and held-out diagnostics."""

    if cell.game_rows_path is None:
        raise ValueError("TrueSkill diagnostics require canonical game rows")
    baseline_frame = _load_rating_frame(cell)
    strategies = baseline_frame["strategy"].astype(str).tolist()
    mus = baseline_frame["mu"].to_numpy(dtype=float)
    sigmas = baseline_frame["sigma"].to_numpy(dtype=float)
    baseline = {
        strategy: (float(mu), float(sigma))
        for strategy, mu, sigma in zip(strategies, mus, sigmas, strict=True)
    }
    tau_zero, tau_games = _fit(
        _games(cell.game_rows_path, cell.k),
        beta=beta,
        tau=0.0,
        draw_probability=draw_probability,
    )
    reversed_order, reversed_games = _fit(
        _games(cell.game_rows_path, cell.k, reverse=True),
        beta=beta,
        tau=tau,
        draw_probability=draw_probability,
    )
    support = _support_counts(cell.game_rows_path, cell.k)
    if "cell_performed_updates" in baseline_frame:
        update_counts = baseline_frame["cell_performed_updates"].drop_duplicates()
        if len(update_counts) != 1:
            raise ValueError("TrueSkill rating rows disagree on cell performed updates")
        performed_updates = int(update_counts.iloc[0])
    else:
        performed_updates = tau_games
    if performed_updates > support["completed_games"]:
        raise ValueError("TrueSkill performed updates exceed completed support")
    support["performed_update_games"] = performed_updates
    prior_only = (
        baseline_frame.loc[
            baseline_frame["rating_status"].eq(_PRIOR_ONLY),
            "strategy",
        ]
        .astype(str)
        .tolist()
    )
    if tau_games != reversed_games:
        raise ValueError("TrueSkill diagnostic replay orders disagree on completed support")
    return {
        "root_seed": cell.root_seed,
        "k": cell.k,
        **support,
        "rating_status": (
            "prior_only_unrated" if len(prior_only) == len(baseline_frame) else "evidence_backed"
        ),
        "prior_only_strategy_count": len(prior_only),
        "prior_only_strategies": ",".join(sorted(prior_only)),
        "tau_zero_games": tau_games,
        "tau_zero_rank_correlation": _rank_correlation(baseline, tau_zero),
        "tau_zero_max_abs_mu_shift": _max_mu_shift(baseline, tau_zero),
        "reversed_order_games": reversed_games,
        "reversed_order_rank_correlation": _rank_correlation(baseline, reversed_order),
        "reversed_order_max_abs_mu_shift": _max_mu_shift(baseline, reversed_order),
        **_heldout_scores(
            cell.game_rows_path,
            cell.k,
            beta=beta,
            tau=tau,
            draw_probability=draw_probability,
        ),
    }


def build_screening_diagnostics(
    cfg: AppConfig,
    cells: Sequence[ScreeningRatingCell],
    *,
    force: bool = False,
) -> Path | None:
    """Write replay diagnostics for cells with available canonical game rows."""

    eligible = [
        cell for cell in cells if cell.game_rows_path is not None and cell.game_rows_path.exists()
    ]
    if not eligible:
        return None
    output = cfg.trueskill_screening_diagnostics_path()
    done = stage_done_path(cfg.trueskill_stage_dir, "trueskill_screening_diagnostics")
    inputs = [
        path
        for cell in eligible
        for path in (cell.ratings_path, cell.game_rows_path)
        if path is not None
    ]
    if not force and stage_is_up_to_date(
        done,
        inputs=inputs,
        outputs=[output],
        cfg=cfg,
        stage="trueskill",
        sidecar_artifacts=[output],
    ):
        return output
    worker_count = min(normalize_n_jobs(cfg.analysis.n_jobs), len(eligible))
    if worker_count > 1:
        context = resolve_mp_context(cfg.analysis.mp_start_method or cfg.sim.mp_start_method)
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = [
                executor.submit(
                    diagnose_rating_cell,
                    cell,
                    beta=cfg.trueskill.beta,
                    tau=cfg.trueskill.tau,
                    draw_probability=cfg.trueskill.draw_probability,
                )
                for cell in eligible
            ]
            rows = [future.result() for future in futures]
    else:
        rows = [
            diagnose_rating_cell(
                cell,
                beta=cfg.trueskill.beta,
                tau=cfg.trueskill.tau,
                draw_probability=cfg.trueskill.draw_probability,
            )
            for cell in eligible
        ]
    frame = pd.DataFrame(rows).sort_values(["root_seed", "k"])
    table = pa.Table.from_pandas(frame, preserve_index=False)
    roots = sorted({cell.root_seed for cell in eligible})
    ks = sorted({cell.k for cell in eligible})
    sidecar = make_artifact_sidecar(
        cfg,
        output,
        producer="trueskill_screening",
        scope=ArtifactScope.DIAGNOSTICS,
        source_scope=ArtifactScope.BY_K,
        operation="aggregate_trueskill_screening_diagnostics",
        weighted_quantity="trueskill_screening_sensitivity",
        support_count_role="ordered_games",
        uncertainty_method="descriptive_replay_and_heldout_prediction",
        replication_unit="game",
        conditioning=TRUESKILL_CONDITIONING,
        consistency_columns=table.schema.names,
        source_artifacts=[
            path
            for cell in eligible
            for path in (cell.ratings_path, cell.game_rows_path)
            if path is not None
        ],
        grouping_keys=["root_seed", "k"],
        player_counts=ks,
        required_player_counts=ks,
        missing_cell_policy="fail",
        seed_scope="both_roots_combined" if len(roots) == 2 else "single_root",
        method_contract=cast(
            Any,
            trueskill_method_contract("aggregate_trueskill_screening_diagnostics"),
        ),
    )
    write_parquet_artifact_atomic(table, output, sidecar=sidecar, codec=cfg.parquet_codec)
    write_stage_done(
        done,
        inputs=inputs,
        outputs=[output],
        cfg=cfg,
        stage="trueskill",
        sidecar_artifacts=[output],
    )
    return output


__all__ = [
    "ScreeningRatingCell",
    "TRUESKILL_CONDITIONING",
    "TRUESKILL_METHOD_VERSION",
    "build_percentile_contribution",
    "build_screening_diagnostics",
    "classify_trueskill_row",
    "diagnose_rating_cell",
    "publish_rating_cell_contract",
    "trueskill_method_contract",
]
