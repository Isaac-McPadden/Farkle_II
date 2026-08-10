"""Descriptive TrueSkill screening contribution and model diagnostics."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import trueskill

from farkle.config import AppConfig, ArtifactScope
from farkle.utils.arrow_batches import iter_parquet_tables_by_bytes
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    TrueSkillMethodContract,
    ensure_artifact_sidecar_atomic,
    make_artifact_sidecar,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
    write_artifact_with_sidecar_atomic,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.parallel import (
    ProcessTreeMemoryGuard,
    StageParallelPolicy,
    apply_native_thread_limits,
    process_map,
    resolve_mp_context,
    resolve_stage_parallel_policy,
)
from farkle.utils.release_identity import is_v3_config
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done
from farkle.utils.strategy_ids import (
    canonical_strategy_id,
    canonical_strategy_ids,
    require_strategy_id_field,
)

_HOLDOUT_FRACTION: Final = 0.2
TRUESKILL_METHOD_VERSION: Final = 3
TRUESKILL_DIAGNOSTIC_METHOD_VERSION: Final = 2
MU_SOFTMAX_HEURISTIC: Final = "mu_softmax_heuristic"
MU_SOFTMAX_HEURISTIC_OPERATION: Final = (
    "aggregate_trueskill_screening_diagnostics_mu_softmax_heuristic"
)
MU_SOFTMAX_HEURISTIC_CLAIM: Final = (
    "Held-out descriptive scores use mu_softmax_heuristic probabilities computed "
    "as softmax(mu / beta). TrueSkill sigma is ignored; these are heuristic "
    "probabilities, not TrueSkill predictive probabilities."
)
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
_DIAGNOSTIC_CELL_OPERATION: Final = "trueskill_screening_diagnostic_cell"
_DEFAULT_DIAGNOSTIC_BATCH_BYTES: Final = 16 * 1024 * 1024
_DEFAULT_DIAGNOSTIC_BATCH_ROWS: Final = 100_000


def trueskill_method_contract(procedure: str) -> TrueSkillMethodContract:
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


def trueskill_diagnostic_method_contract() -> TrueSkillMethodContract:
    """Return the versioned replay and mu-softmax diagnostic identity."""

    contract = trueskill_method_contract(MU_SOFTMAX_HEURISTIC_OPERATION)
    parameters = contract.get("parameters")
    assert parameters is not None
    parameters.update(
        {
            "diagnostic_method_version": TRUESKILL_DIAGNOSTIC_METHOD_VERSION,
            "heldout_probability_method": MU_SOFTMAX_HEURISTIC,
            "heldout_probability_formula": "softmax(mu / beta)",
            "heldout_probability_sigma_policy": "ignored",
            "heldout_fraction": _HOLDOUT_FRACTION,
            "heldout_rating_policy": "freeze_after_chronological_training_prefix",
            "heldout_target": "unique_rank_1_completed_game_winner",
            "interpretation": MU_SOFTMAX_HEURISTIC_CLAIM,
        }
    )
    return contract


def _trueskill_diagnostic_freshness_key(cfg: AppConfig) -> dict[str, Any]:
    """Bind diagnostic-only method identity without staling percentile ratings."""

    return {
        **cfg.freshness_key(),
        "trueskill_diagnostic_method_version": TRUESKILL_DIAGNOSTIC_METHOD_VERSION,
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
        players.append(
            str(
                canonical_strategy_id(
                    strategy,
                    context=f"TrueSkill row P{seat}_strategy",
                )
            )
        )
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
    if is_v3_config(cfg):
        if expected_sidecar_sha256 is not None:
            raise ArtifactContractError(
                "authenticated v3 cannot reconstruct a missing sidecar from cached bytes"
            )
        content = cell.ratings_path.read_bytes()

        def _write_rating(staged: Path) -> None:
            staged.write_bytes(content)

        write_artifact_with_sidecar_atomic(
            cell.ratings_path,
            sidecar,
            _write_rating,
        )
    else:
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
    require_strategy_id_field(schema, "strategy", context=str(cell.ratings_path))
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
    frame["strategy"] = canonical_strategy_ids(
        frame["strategy"],
        context=f"{cell.ratings_path} strategy",
    )
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
    contribution["strategy"] = canonical_strategy_ids(
        contribution["strategy"],
        context="TrueSkill percentile contribution strategy",
    )

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


def _iter_classified_batch(batch: pa.Table, k: int) -> Iterator[ClassifiedTrueSkillGame]:
    """Classify a bounded projected Arrow table without a wide Python row list."""

    columns = {name: batch.column(name).combine_chunks() for name in _game_columns(k)}
    for row_index in range(batch.num_rows):
        yield classify_trueskill_row(
            {name: column[row_index].as_py() for name, column in columns.items()},
            k,
        )


def _iter_classified_batch_reverse(batch: pa.Table, k: int) -> Iterator[ClassifiedTrueSkillGame]:
    """Classify a bounded projected Arrow table in reverse record order."""

    columns = {name: batch.column(name).combine_chunks() for name in _game_columns(k)}
    for row_index in range(batch.num_rows - 1, -1, -1):
        yield classify_trueskill_row(
            {name: column[row_index].as_py() for name, column in columns.items()},
            k,
        )


def _update_ratings(
    env: trueskill.TrueSkill,
    ratings: dict[str, trueskill.Rating],
    players: Sequence[str],
    ranks: Sequence[int],
) -> None:
    """Apply one deterministic completed-game TrueSkill update in seat order."""

    groups = [(ratings.setdefault(player, env.create_rating()),) for player in players]
    updated = env.rate(groups, ranks=list(ranks))
    for player, group in zip(players, updated, strict=True):
        ratings[player] = group[0]


def _rating_pairs(ratings: Mapping[str, trueskill.Rating]) -> dict[str, tuple[float, float]]:
    return {strategy: (rating.mu, rating.sigma) for strategy, rating in ratings.items()}


def _write_reverse_spool(batch: pa.Table, directory: Path, batch_index: int) -> Path:
    """Persist one bounded projected batch so reverse replay never widens a row group."""

    path = directory / f"{batch_index:012d}.arrow"
    with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, batch.schema) as writer:
        writer.write_table(batch)
    return path


def _iter_reverse_spool(
    directory: Path, batch_count: int, k: int
) -> Iterator[ClassifiedTrueSkillGame]:
    """Yield the exact reverse record order from bounded on-disk projected batches."""

    for batch_index in range(batch_count - 1, -1, -1):
        path = directory / f"{batch_index:012d}.arrow"
        with pa.memory_map(str(path), "r") as source:
            table = pa.ipc.open_file(source).read_all()
        yield from _iter_classified_batch_reverse(table, k)


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


def mu_softmax_heuristic_probabilities(
    ratings: Sequence[tuple[float, float]],
    *,
    beta: float,
) -> np.ndarray:
    """Return descriptive softmax(mu / beta) probabilities.

    Sigma is accepted and validated to make the ignored model state explicit.
    This heuristic is not a TrueSkill predictive-probability calculation.
    """

    if not ratings:
        raise ValueError("mu_softmax_heuristic requires at least one rating")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("mu_softmax_heuristic requires finite beta > 0")
    rating_array = np.asarray(ratings, dtype=float)
    if rating_array.shape != (len(ratings), 2) or not np.all(np.isfinite(rating_array)):
        raise ValueError("mu_softmax_heuristic requires finite (mu, sigma) ratings")
    if np.any(rating_array[:, 1] <= 0):
        raise ValueError("mu_softmax_heuristic requires sigma > 0")
    logits = (rating_array[:, 0] - np.max(rating_array[:, 0])) / beta
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum()


def diagnose_rating_cell(
    cell: ScreeningRatingCell,
    *,
    beta: float,
    tau: float,
    draw_probability: float,
    max_batch_bytes: int = _DEFAULT_DIAGNOSTIC_BATCH_BYTES,
    batch_rows: int = _DEFAULT_DIAGNOSTIC_BATCH_ROWS,
) -> dict[str, object]:
    """Replay one root/k stream with one forward and one bounded reverse pass.

    Source rows are decoded once into a byte-bounded temporary projection.  It
    supplies exact completed-game support for the chronological held-out split,
    then supports forward training/scoring and reverse replay without rescanning
    the Parquet source or widening a row group.
    """

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
    parquet_file = pq.ParquetFile(cell.game_rows_path)
    columns = _game_columns(cell.k)
    missing = sorted(set(columns).difference(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"{cell.game_rows_path} lacks TrueSkill diagnostic columns: {missing}")
    tau_zero_env = trueskill.TrueSkill(beta=beta, tau=0.0, draw_probability=draw_probability)
    tau_zero_ratings: dict[str, trueskill.Rating] = {}
    completed_support = baseline_frame["cell_games_completed"].drop_duplicates()
    if len(completed_support) != 1:
        raise ValueError("TrueSkill rating rows disagree on completed-game support")
    expected_completed = int(completed_support.iloc[0])
    training_games = (
        max(1, math.floor(expected_completed * (1.0 - _HOLDOUT_FRACTION)))
        if expected_completed
        else 0
    )
    train_env = trueskill.TrueSkill(beta=beta, tau=tau, draw_probability=draw_probability)
    train_ratings: dict[str, trueskill.Rating] = {}
    observed_train = 0
    log_loss_sum = brier_sum = confidence_sum = correct_sum = 0.0
    holdout_games = 0
    attempted = completed = excluded = tau_games = 0
    with tempfile.TemporaryDirectory(
        prefix=".trueskill_diag_", dir=cell.game_rows_path.parent
    ) as tmp:
        spool_dir = Path(tmp)
        spool_batch_count = 0
        for batch_index, (_rg, _bi, batch) in enumerate(
            iter_parquet_tables_by_bytes(
                cell.game_rows_path,
                columns=columns,
                max_batch_bytes=max_batch_bytes,
                max_batch_rows=batch_rows,
                use_threads=False,
            )
        ):
            _write_reverse_spool(batch, spool_dir, batch_index)
            spool_batch_count += 1
            for game in _iter_classified_batch(batch, cell.k):
                attempted += 1
                if game.ranks is None:
                    excluded += 1
                    continue
                completed += 1
                _update_ratings(tau_zero_env, tau_zero_ratings, game.players, game.ranks)
                tau_games += 1
                if observed_train < training_games:
                    _update_ratings(train_env, train_ratings, game.players, game.ranks)
                    observed_train += 1
                    continue
                ratings = [
                    (
                        (rating.mu, rating.sigma)
                        if (rating := train_ratings.get(player)) is not None
                        else (25.0, 25.0 / 3.0)
                    )
                    for player in game.players
                ]
                probabilities = mu_softmax_heuristic_probabilities(ratings, beta=beta)
                winner_positions = np.flatnonzero(np.asarray(game.ranks) == 0)
                target = np.zeros(cell.k, dtype=float)
                target[winner_positions] = 1.0 / len(winner_positions)
                log_loss_sum += float(-np.sum(target * np.log(np.maximum(probabilities, 1e-15))))
                brier_sum += float(np.sum((probabilities - target) ** 2))
                predicted = int(np.argmax(probabilities))
                confidence_sum += float(probabilities[predicted])
                correct_sum += float(predicted in winner_positions)
                holdout_games += 1

        if attempted != completed + excluded:
            raise ValueError("TrueSkill support conservation failed")
        if completed != expected_completed:
            raise ValueError(
                "TrueSkill diagnostic source support differs from the authenticated rating cell"
            )

        reverse_env = trueskill.TrueSkill(beta=beta, tau=tau, draw_probability=draw_probability)
        reverse_ratings: dict[str, trueskill.Rating] = {}
        reversed_games = 0
        for game in _iter_reverse_spool(spool_dir, spool_batch_count, cell.k):
            if game.ranks is not None:
                _update_ratings(reverse_env, reverse_ratings, game.players, game.ranks)
                reversed_games += 1

    if "cell_performed_updates" in baseline_frame:
        update_counts = baseline_frame["cell_performed_updates"].drop_duplicates()
        if len(update_counts) != 1:
            raise ValueError("TrueSkill rating rows disagree on cell performed updates")
        performed_updates = int(update_counts.iloc[0])
    else:
        performed_updates = tau_games
    if performed_updates > completed:
        raise ValueError("TrueSkill performed updates exceed completed support")
    prior_only = (
        baseline_frame.loc[
            baseline_frame["rating_status"].eq(_PRIOR_ONLY),
            "strategy",
        ]
        .astype(str)
        .tolist()
    )
    if tau_games != reversed_games or tau_games != completed:
        raise ValueError("TrueSkill diagnostic replay orders disagree on completed support")
    return {
        "root_seed": cell.root_seed,
        "k": cell.k,
        "attempted_games": attempted,
        "completed_games": completed,
        "excluded_safety_limit_games": excluded,
        "performed_update_games": performed_updates,
        "rating_status": (
            "prior_only_unrated" if len(prior_only) == len(baseline_frame) else "evidence_backed"
        ),
        "prior_only_strategy_count": len(prior_only),
        "prior_only_strategies": ",".join(sorted(prior_only)),
        "tau_zero_games": tau_games,
        "tau_zero_rank_correlation": _rank_correlation(baseline, _rating_pairs(tau_zero_ratings)),
        "tau_zero_max_abs_mu_shift": _max_mu_shift(baseline, _rating_pairs(tau_zero_ratings)),
        "reversed_order_games": reversed_games,
        "reversed_order_rank_correlation": _rank_correlation(
            baseline, _rating_pairs(reverse_ratings)
        ),
        "reversed_order_max_abs_mu_shift": _max_mu_shift(baseline, _rating_pairs(reverse_ratings)),
        "mu_softmax_heuristic_claim": MU_SOFTMAX_HEURISTIC_CLAIM,
        "mu_softmax_heuristic_training_games": observed_train,
        "mu_softmax_heuristic_holdout_games": holdout_games,
        "mu_softmax_heuristic_heldout_log_loss": (
            log_loss_sum / holdout_games if holdout_games else None
        ),
        "mu_softmax_heuristic_uniform_reference_log_loss": (
            math.log(cell.k) if holdout_games else None
        ),
        "mu_softmax_heuristic_heldout_brier_score": (
            brier_sum / holdout_games if holdout_games else None
        ),
        "mu_softmax_heuristic_uniform_reference_brier_score": (
            1.0 - 1.0 / cell.k if holdout_games else None
        ),
        "mu_softmax_heuristic_mean_top_probability": (
            confidence_sum / holdout_games if holdout_games else None
        ),
        "mu_softmax_heuristic_top_prediction_accuracy": (
            correct_sum / holdout_games if holdout_games else None
        ),
        "mu_softmax_heuristic_mean_top_probability_minus_accuracy": (
            (confidence_sum - correct_sum) / holdout_games if holdout_games else None
        ),
    }


def _diagnostic_cell_freshness_key(cfg: AppConfig, cell: ScreeningRatingCell) -> dict[str, Any]:
    """Bind a resumable diagnostic cell to its logical root/k coordinate."""

    return {
        **_trueskill_diagnostic_freshness_key(cfg),
        "root_seed": cell.root_seed,
        "player_count": cell.k,
        "reverse_replay_source": "bounded_forward_projection_spool",
    }


def _diagnostic_cell_done_path(cfg: AppConfig, cell: ScreeningRatingCell) -> Path:
    return stage_done_path(
        cfg.by_k_dir("trueskill", cell.k),
        f"screening_diagnostics_{cell.k}_seed{cell.root_seed}",
    )


def _build_diagnostic_cell(
    cfg: AppConfig,
    cell: ScreeningRatingCell,
    *,
    force: bool,
    max_batch_bytes: int,
) -> Path:
    """Atomically publish one root/k diagnostic after exactly one forward replay."""

    if cell.game_rows_path is None or not cell.game_rows_path.is_file():
        raise FileNotFoundError(
            f"TrueSkill diagnostic game rows missing for {cell.root_seed}/{cell.k}"
        )
    output = cfg.trueskill_screening_diagnostic_cell_path(cell.k, root_seed=cell.root_seed)
    done = _diagnostic_cell_done_path(cfg, cell)
    inputs = [cell.ratings_path, cell.game_rows_path]
    freshness = _diagnostic_cell_freshness_key(cfg, cell)
    if not force and stage_is_up_to_date(
        done,
        inputs=inputs,
        outputs=[output],
        cfg=cfg,
        stage="trueskill",
        freshness_key=freshness,
        sidecar_artifacts=[output],
    ):
        return output
    row = diagnose_rating_cell(
        cell,
        beta=cfg.trueskill.beta,
        tau=cfg.trueskill.tau,
        draw_probability=cfg.trueskill.draw_probability,
        max_batch_bytes=max_batch_bytes,
        batch_rows=_DEFAULT_DIAGNOSTIC_BATCH_ROWS,
    )
    table = pa.Table.from_pylist([row])
    sidecar = make_artifact_sidecar(
        cfg,
        output,
        producer="trueskill_screening",
        scope=ArtifactScope.BY_K,
        source_scope=ArtifactScope.BY_K,
        operation=_DIAGNOSTIC_CELL_OPERATION,
        weighted_quantity="trueskill_screening_sensitivity_and_mu_softmax_heuristic_scores",
        support_count_role="ordered_attempted_and_completed_games",
        uncertainty_method="descriptive_replay_and_mu_softmax_heuristic_scoring",
        replication_unit="ordered_game",
        conditioning=TRUESKILL_CONDITIONING,
        consistency_columns=table.schema.names,
        source_artifacts=inputs,
        grouping_keys=["root_seed", "k"],
        player_counts=[cell.k],
        required_player_counts=[cell.k],
        missing_cell_policy="fail",
        seed_scope="single_root",
        method_contract=cast(Any, trueskill_diagnostic_method_contract()),
    )
    write_parquet_artifact_atomic(table, output, sidecar=sidecar, codec=cfg.parquet_codec)
    write_stage_done(
        done,
        inputs=inputs,
        outputs=[output],
        cfg=cfg,
        stage="trueskill",
        freshness_key=freshness,
        sidecar_artifacts=[output],
    )
    return output


def _initialize_diagnostic_worker(policy: StageParallelPolicy) -> None:
    apply_native_thread_limits(policy)


def _build_diagnostic_cell_task(
    args: tuple[AppConfig, ScreeningRatingCell, bool, int],
) -> Path:
    cfg, cell, force, max_batch_bytes = args
    return _build_diagnostic_cell(
        cfg,
        cell,
        force=force,
        max_batch_bytes=max_batch_bytes,
    )


def build_screening_diagnostics(
    cfg: AppConfig,
    cells: Sequence[ScreeningRatingCell],
    *,
    force: bool = False,
) -> Path | None:
    """Publish validated root/k diagnostics, then deterministically aggregate them."""

    if not cells:
        return None
    coordinates = [(cell.root_seed, cell.k) for cell in cells]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("TrueSkill diagnostic cells contain duplicate root/k coordinates")
    if any(cell.game_rows_path is None or not cell.game_rows_path.exists() for cell in cells):
        raise FileNotFoundError(
            "TrueSkill diagnostics require canonical game rows for every root/k cell"
        )
    eligible = sorted(cells, key=lambda cell: (cell.root_seed, cell.k))
    output = cfg.trueskill_screening_diagnostics_path()
    done = stage_done_path(cfg.trueskill_stage_dir, "trueskill_screening_diagnostics")
    cell_outputs = [
        cfg.trueskill_screening_diagnostic_cell_path(cell.k, root_seed=cell.root_seed)
        for cell in eligible
    ]
    cell_complete = all(
        stage_is_up_to_date(
            _diagnostic_cell_done_path(cfg, cell),
            inputs=[cell.ratings_path, cast(Path, cell.game_rows_path)],
            outputs=[cell_output],
            cfg=cfg,
            stage="trueskill",
            freshness_key=_diagnostic_cell_freshness_key(cfg, cell),
            sidecar_artifacts=[cell_output],
        )
        for cell, cell_output in zip(eligible, cell_outputs, strict=True)
    )
    if (
        not force
        and cell_complete
        and stage_is_up_to_date(
            done,
            inputs=cell_outputs,
            outputs=[output],
            cfg=cfg,
            stage="trueskill",
            freshness_key=_trueskill_diagnostic_freshness_key(cfg),
            sidecar_artifacts=[output],
        )
    ):
        return output

    policy = resolve_stage_parallel_policy(
        "trueskill",
        cfg.analysis,
        n_jobs_override=cfg.analysis.n_jobs,
        resources=cfg.resources,
    )
    apply_native_thread_limits(policy)
    memory_guard = ProcessTreeMemoryGuard(
        cfg.resources.hard_memory_limit_mb,
        rss_warning_mb=cfg.resources.target_memory_mb,
        sample_interval_seconds=cfg.resources.rss_sample_interval_seconds,
    )
    memory_guard.check_before_schedule(force=True)
    max_batch_bytes = cfg.resources.stage_batch_bytes["trueskill"]
    tasks = [(cfg, cell, force, max_batch_bytes) for cell in eligible]
    context = resolve_mp_context(cfg.analysis.mp_start_method or cfg.sim.mp_start_method)
    built = list(
        process_map(
            _build_diagnostic_cell_task,
            tasks,
            n_jobs=min(policy.process_workers, len(tasks)),
            initializer=_initialize_diagnostic_worker,
            initargs=(policy,),
            window=cfg.resources.max_in_flight_per_worker * max(1, policy.process_workers),
            mp_context=context,
            memory_guard=memory_guard,
        )
    )
    if sorted(built, key=lambda path: path.as_posix()) != sorted(
        cell_outputs, key=lambda path: path.as_posix()
    ):
        raise RuntimeError(
            "TrueSkill diagnostic cells did not publish in deterministic coordinate order"
        )
    memory_guard.check_before_schedule(force=True)
    rows: list[dict[str, object]] = []
    for cell, cell_output in zip(eligible, cell_outputs, strict=True):
        cell_done = _diagnostic_cell_done_path(cfg, cell)
        if not stage_is_up_to_date(
            cell_done,
            inputs=[cell.ratings_path, cast(Path, cell.game_rows_path)],
            outputs=[cell_output],
            cfg=cfg,
            stage="trueskill",
            freshness_key=_diagnostic_cell_freshness_key(cfg, cell),
            sidecar_artifacts=[cell_output],
        ):
            raise ArtifactContractError(
                f"incomplete TrueSkill diagnostic cell: {cell.root_seed}/{cell.k}"
            )
        cell_table = pq.read_table(cell_output)
        if cell_table.num_rows != 1:
            raise ArtifactContractError(f"invalid TrueSkill diagnostic cell payload: {cell_output}")
        data = {
            name: cell_table.column(name).combine_chunks()[0].as_py()
            for name in cell_table.column_names
        }
        if data.get("root_seed") != cell.root_seed or data.get("k") != cell.k:
            raise ArtifactContractError(f"invalid TrueSkill diagnostic cell payload: {cell_output}")
        rows.append(data)
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
        operation=MU_SOFTMAX_HEURISTIC_OPERATION,
        weighted_quantity="trueskill_screening_sensitivity_and_mu_softmax_heuristic_scores",
        support_count_role="ordered_games",
        uncertainty_method="descriptive_replay_and_mu_softmax_heuristic_scoring",
        replication_unit="game",
        conditioning=TRUESKILL_CONDITIONING,
        consistency_columns=table.schema.names,
        source_artifacts=cell_outputs,
        grouping_keys=["root_seed", "k"],
        player_counts=ks,
        required_player_counts=ks,
        missing_cell_policy="fail",
        seed_scope="both_roots_combined" if len(roots) == 2 else "single_root",
        method_contract=cast(
            Any,
            trueskill_diagnostic_method_contract(),
        ),
    )
    write_parquet_artifact_atomic(table, output, sidecar=sidecar, codec=cfg.parquet_codec)
    memory_guard.check_before_schedule(force=True)
    write_stage_done(
        done,
        inputs=cell_outputs,
        outputs=[output],
        cfg=cfg,
        stage="trueskill",
        freshness_key=_trueskill_diagnostic_freshness_key(cfg),
        sidecar_artifacts=[output],
    )
    return output


__all__ = [
    "MU_SOFTMAX_HEURISTIC",
    "MU_SOFTMAX_HEURISTIC_CLAIM",
    "MU_SOFTMAX_HEURISTIC_OPERATION",
    "ScreeningRatingCell",
    "TRUESKILL_CONDITIONING",
    "TRUESKILL_DIAGNOSTIC_METHOD_VERSION",
    "TRUESKILL_METHOD_VERSION",
    "build_percentile_contribution",
    "build_screening_diagnostics",
    "classify_trueskill_row",
    "diagnose_rating_cell",
    "mu_softmax_heuristic_probabilities",
    "publish_rating_cell_contract",
    "trueskill_diagnostic_method_contract",
    "trueskill_method_contract",
]
