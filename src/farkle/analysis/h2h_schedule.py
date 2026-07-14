"""Power-plan and execute immutable H2H pair/root/order blocks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import norm

from farkle.analysis.stage_state import (
    stage_done_path,
    stage_is_up_to_date,
    write_stage_done,
)
from farkle.config import AppConfig, ArtifactScope
from farkle.simulation.simulation import simulate_many_games_from_seeds
from farkle.simulation.strategies import parse_strategy_for_df, parse_strategy_identifier
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import (
    write_json_artifact_atomic,
    write_parquet_artifact_atomic,
)
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose, coordinate_seed

SCORE_TEST_ID: Final = "independent_two_proportion_score_v1"
POWER_METHOD_ID: Final = "normal_power_for_independent_two_proportion_score_v1"


@dataclass(frozen=True)
class H2HScheduleArtifacts:
    """Power plan and optional ready-to-execute block manifest."""

    power_plan: Path
    block_manifest: Path | None
    schedule_state: str


@dataclass(frozen=True)
class H2HExecutionArtifacts:
    """Completed block checkpoints and their row-preserving union."""

    order_counts: Path
    block_paths: tuple[Path, ...]


def independent_score_planning_power(
    games_per_order: int,
    q_ab: float,
    q_ba: float,
    alpha: float,
) -> float:
    """Return deterministic large-sample power for the two-sided score procedure."""

    if games_per_order < 1:
        raise ValueError("games_per_order must be positive")
    if not 0.0 < q_ab < 1.0 or not 0.0 < q_ba < 1.0:
        raise ValueError("planning probabilities must be strictly between zero and one")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    common = 0.5 * (q_ab + q_ba)
    null_sd = math.sqrt(2.0 * common * (1.0 - common) / games_per_order)
    alternative_sd = math.sqrt((q_ab * (1.0 - q_ab) + q_ba * (1.0 - q_ba)) / games_per_order)
    critical_difference = float(norm.ppf(1.0 - alpha / 2.0)) * null_sd
    difference = q_ab - q_ba
    upper = float(norm.sf((critical_difference - difference) / alternative_sd))
    lower = float(norm.cdf((-critical_difference - difference) / alternative_sd))
    return min(1.0, max(0.0, upper + lower))


def _scenario_probabilities(effect: float, seat1_advantage: float) -> tuple[float, float]:
    """Map the reported half-difference and common seat effect to order probabilities."""

    q_ab = 0.5 + seat1_advantage + effect
    q_ba = 0.5 + seat1_advantage - effect
    if not 0.0 < q_ba < q_ab < 1.0:
        raise ValueError(
            "head-to-head effect and seat-advantage scenario produce invalid probabilities: "
            f"q_ab={q_ab}, q_ba={q_ba}"
        )
    return q_ab, q_ba


def _worst_scenario_power(
    *,
    games_per_root_order_block: int,
    root_count: int,
    effect: float,
    scenarios: tuple[float, ...],
    alpha_per_pair: float,
) -> float:
    games_per_order = games_per_root_order_block * root_count
    return min(
        independent_score_planning_power(
            games_per_order,
            *_scenario_probabilities(effect, advantage),
            alpha_per_pair,
        )
        for advantage in scenarios
    )


def _minimum_block_games(
    *,
    root_count: int,
    effect: float,
    scenarios: tuple[float, ...],
    alpha_per_pair: float,
    target_power: float,
) -> int:
    """Find the smallest equal root/order block size satisfying worst-case power."""

    def sufficient(block_games: int) -> bool:
        return (
            _worst_scenario_power(
                games_per_root_order_block=block_games,
                root_count=root_count,
                effect=effect,
                scenarios=scenarios,
                alpha_per_pair=alpha_per_pair,
            )
            >= target_power
        )

    upper = 1
    while not sufficient(upper):
        upper *= 2
        if upper > 2**50:
            raise RuntimeError("H2H power search failed to find a finite allocation")
    lower = 0
    while lower + 1 < upper:
        midpoint = (lower + upper) // 2
        if sufficient(midpoint):
            upper = midpoint
        else:
            lower = midpoint
    return upper


def _load_frozen_family(cfg: AppConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = cfg.h2h_candidate_family_manifest_path()
    membership_path = cfg.h2h_candidate_family_path()
    for path in (manifest_path, membership_path):
        validate_artifact_sidecar(
            path,
            expected={
                "scope": ArtifactScope.H2H_2P.value,
                "operation": "candidate_family_freeze",
            },
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    membership = pq.read_table(membership_path).to_pandas()
    candidates = manifest.get("candidates")
    family_hash = manifest.get("family_hash")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("frozen candidate manifest must contain at least two candidates")
    if not isinstance(family_hash, str) or len(family_hash) != 64:
        raise ValueError("frozen candidate manifest has an invalid family hash")
    selected = sorted(
        membership.loc[membership["final_family"].astype(bool), "strategy"].astype(str).tolist()
    )
    if selected != sorted(str(candidate) for candidate in candidates):
        raise ValueError("candidate manifest and membership artifact disagree")
    hashes = set(membership["family_hash"].astype(str))
    if hashes != {family_hash}:
        raise ValueError("candidate membership rows disagree with the frozen family hash")
    return manifest, membership


def _roots_from_manifest(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    raw = manifest.get("root_seeds")
    if not isinstance(raw, list) or len(raw) not in {1, 2}:
        raise ValueError("candidate manifest must declare one or two root seeds")
    roots = tuple(int(value) for value in raw)
    if len(set(roots)) != len(roots):
        raise ValueError("candidate manifest root seeds must be distinct")
    return roots


def _configured_roots(cfg: AppConfig) -> tuple[int, ...]:
    if cfg.sim.seed_list is not None:
        return tuple(int(value) for value in cfg.sim.seed_list)
    if cfg.sim.seed_pair is not None:
        return tuple(int(value) for value in cfg.sim.seed_pair)
    return (int(cfg.sim.seed),)


def _power_grid(
    cfg: AppConfig,
    *,
    block_games: int,
    roots: tuple[int, ...],
    alpha_per_pair: float,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    games_per_order = block_games * len(roots)
    for effect in cfg.head2head.sensitivity_deltas:
        for advantage in cfg.head2head.seat1_advantage_scenarios:
            q_ab, q_ba = _scenario_probabilities(float(effect), float(advantage))
            rows.append(
                {
                    "reported_effect": float(effect),
                    "seat1_advantage": float(advantage),
                    "q_ab": q_ab,
                    "q_ba": q_ba,
                    "games_per_order": games_per_order,
                    "achieved_power": independent_score_planning_power(
                        games_per_order,
                        q_ab,
                        q_ba,
                        alpha_per_pair,
                    ),
                }
            )
    return rows


def _block_id(family_hash: str, pair_id: int, root_seed: int, order: int) -> str:
    value = f"{family_hash}:{pair_id}:{root_seed}:{order}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:24]


def _schedule_frame(
    *,
    candidates: list[str],
    family_hash: str,
    roots: tuple[int, ...],
    block_games: int,
    alpha_per_pair: float,
    target_power: float,
    worst_power: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_id, (strategy_a, strategy_b) in enumerate(combinations(sorted(candidates), 2)):
        for root_index, root_seed in enumerate(roots):
            for order in (0, 1):
                seat1 = strategy_a if order == 0 else strategy_b
                seat2 = strategy_b if order == 0 else strategy_a
                rows.append(
                    {
                        "family_hash": family_hash,
                        "pair_id": pair_id,
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "root_seed": root_seed,
                        "root_index": root_index,
                        "order": order,
                        "order_label": "a_b" if order == 0 else "b_a",
                        "seat1_strategy": seat1,
                        "seat2_strategy": seat2,
                        "games_required": block_games,
                        "game_index_start": 0,
                        "game_index_stop": block_games,
                        "rng_scheme_version": RNG_SCHEME_VERSION,
                        "rng_purpose_namespace": int(RandomPurpose.H2H_GAME),
                        "score_test_id": SCORE_TEST_ID,
                        "alpha_per_pair": alpha_per_pair,
                        "target_power": target_power,
                        "worst_scenario_achieved_power": worst_power,
                        "block_id": _block_id(family_hash, pair_id, root_seed, order),
                    }
                )
    return pd.DataFrame(rows)


def _planning_sidecar_common(
    *,
    sources: list[Path],
    roots: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "producer": "h2h_schedule",
        "scope": ArtifactScope.H2H_2P,
        "source_scope": ArtifactScope.H2H_2P,
        "baseline": "equal_seat_order_rates",
        "weighted_quantity": "seat_adjusted_h2h_effect",
        "support_count_role": "independent_games_per_root_order_block",
        "uncertainty_method": SCORE_TEST_ID,
        "replication_unit": "independent_h2h_game",
        "conditioning": "frozen_finite_grid_candidate_family",
        "source_artifacts": sources,
        "player_counts": [2],
        "required_player_counts": [2],
        "missing_cell_policy": "fail",
        "seed_scope": "both_roots_combined" if len(roots) == 2 else "single_root",
    }


def plan_h2h_schedule(cfg: AppConfig, *, force: bool = False) -> H2HScheduleArtifacts:
    """Power-size and freeze equal pair/root/order simulation blocks."""

    family, _membership = _load_frozen_family(cfg)
    candidates = [str(value) for value in family["candidates"]]
    family_hash = str(family["family_hash"])
    roots = _roots_from_manifest(family)
    if roots != _configured_roots(cfg):
        raise ValueError(
            "frozen candidate-family roots do not match the active configuration: "
            f"{roots} != {_configured_roots(cfg)}"
        )
    if len(roots) == 1 and not cfg.head2head.allow_single_root:
        raise ValueError("single-root H2H is disabled by head2head.allow_single_root")
    pair_count = len(candidates) * (len(candidates) - 1) // 2
    alpha_per_pair = cfg.head2head.family_alpha / pair_count
    scenarios = tuple(float(value) for value in cfg.head2head.seat1_advantage_scenarios)
    block_games = _minimum_block_games(
        root_count=len(roots),
        effect=cfg.head2head.practical_delta,
        scenarios=scenarios,
        alpha_per_pair=alpha_per_pair,
        target_power=cfg.head2head.target_power,
    )
    worst_power = _worst_scenario_power(
        games_per_root_order_block=block_games,
        root_count=len(roots),
        effect=cfg.head2head.practical_delta,
        scenarios=scenarios,
        alpha_per_pair=alpha_per_pair,
    )
    previous_power = (
        _worst_scenario_power(
            games_per_root_order_block=block_games - 1,
            root_count=len(roots),
            effect=cfg.head2head.practical_delta,
            scenarios=scenarios,
            alpha_per_pair=alpha_per_pair,
        )
        if block_games > 1
        else 0.0
    )
    games_per_pair = block_games * len(roots) * 2
    total_games = games_per_pair * pair_count
    cap = cfg.head2head.total_game_cap
    blocked = cap is not None and total_games > cap
    state = "blocked_by_cap" if blocked else "ready"
    power_grid = _power_grid(
        cfg,
        block_games=block_games,
        roots=roots,
        alpha_per_pair=alpha_per_pair,
    )
    plan: dict[str, Any] = {
        "family_hash": family_hash,
        "candidate_count": len(candidates),
        "unordered_pair_count": pair_count,
        "root_seeds": list(roots),
        "single_root_execution": len(roots) == 1,
        "score_test_id": SCORE_TEST_ID,
        "power_method_id": POWER_METHOD_ID,
        "test_direction": "two_sided",
        "final_multiplicity_method": "holm",
        "planning_multiplicity_method": "bonferroni",
        "family_alpha": cfg.head2head.family_alpha,
        "alpha_per_pair": alpha_per_pair,
        "target_power": cfg.head2head.target_power,
        "target_effect": cfg.head2head.practical_delta,
        "games_per_root_order_block": block_games,
        "games_per_order_across_roots": block_games * len(roots),
        "games_per_pair": games_per_pair,
        "projected_total_games": total_games,
        "total_game_cap": cap,
        "schedule_state": state,
        "worst_scenario_achieved_power": worst_power,
        "previous_equal_block_size_worst_power": previous_power,
        "seat_order_rng_contract": "independent_coordinate_streams",
        "power_validation": power_grid,
        "cap_guidance": (
            None
            if not blocked
            else "increase head2head.total_game_cap to at least projected_total_games"
        ),
    }
    power_path = cfg.h2h_power_plan_path()
    schedule_path = cfg.h2h_block_manifest_path()
    family_sources = [
        cfg.h2h_candidate_family_manifest_path(),
        cfg.h2h_candidate_family_path(),
    ]
    done = stage_done_path(cfg.h2h_2p_dir(), "h2h_schedule_plan")
    expected_outputs = [power_path] if blocked else [power_path, schedule_path]
    if not force and stage_is_up_to_date(
        done,
        inputs=family_sources,
        outputs=expected_outputs,
        cfg=cfg,
        stage="head2head",
        sidecar_artifacts=expected_outputs,
    ):
        return H2HScheduleArtifacts(
            power_plan=power_path,
            block_manifest=None if blocked else schedule_path,
            schedule_state=state,
        )

    common = _planning_sidecar_common(sources=family_sources, roots=roots)
    power_sidecar = make_artifact_sidecar(
        cfg,
        power_path,
        operation="score_test_power_plan",
        consistency_columns=list(plan),
        grouping_keys=["family_hash"],
        **common,
    )
    write_json_artifact_atomic(plan, power_path, sidecar=power_sidecar)
    if blocked:
        write_stage_done(
            done,
            inputs=family_sources,
            outputs=[power_path],
            cfg=cfg,
            stage="head2head",
            status="blocked_by_cap",
            reason=(f"projected H2H games {total_games} exceed " f"head2head.total_game_cap={cap}"),
            sidecar_artifacts=[power_path],
        )
        return H2HScheduleArtifacts(
            power_plan=power_path,
            block_manifest=None,
            schedule_state=state,
        )

    schedule = _schedule_frame(
        candidates=candidates,
        family_hash=family_hash,
        roots=roots,
        block_games=block_games,
        alpha_per_pair=alpha_per_pair,
        target_power=cfg.head2head.target_power,
        worst_power=worst_power,
    )
    schedule_sidecar = make_artifact_sidecar(
        cfg,
        schedule_path,
        operation="construct_pair_root_order_blocks",
        consistency_columns=schedule.columns.tolist(),
        grouping_keys=["pair_id", "root_seed", "order"],
        **common,
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(schedule, preserve_index=False),
        schedule_path,
        sidecar=schedule_sidecar,
        codec=cfg.parquet_codec,
    )
    write_stage_done(
        done,
        inputs=family_sources,
        outputs=[power_path, schedule_path],
        cfg=cfg,
        stage="head2head",
        sidecar_artifacts=[power_path, schedule_path],
    )
    return H2HScheduleArtifacts(
        power_plan=power_path,
        block_manifest=schedule_path,
        schedule_state=state,
    )


def _winner_seat_counts(frame: pd.DataFrame) -> tuple[int, int]:
    winner_column = next(
        (column for column in ("winner_seat", "winner") if column in frame),
        None,
    )
    if winner_column is None:
        raise ValueError("H2H simulation rows lack winner_seat/winner")
    winners = frame[winner_column].astype(str).str.strip().str.lower()
    seat1 = winners.isin({"1", "p1", "seat1", "seat_1"})
    seat2 = winners.isin({"2", "p2", "seat2", "seat_2"})
    if not (seat1 | seat2).all():
        invalid = sorted(winners.loc[~(seat1 | seat2)].unique().tolist())
        raise ValueError(f"H2H simulation contains invalid winner seats: {invalid}")
    return int(seat1.sum()), int(seat2.sum())


def _simulate_block(
    block: dict[str, Any],
    strategy_manifest_path: Path,
    chunk_games: int,
) -> dict[str, Any]:
    manifest = pd.read_parquet(strategy_manifest_path)
    strategy1 = parse_strategy_identifier(
        block["seat1_strategy"],
        manifest=manifest,
        parse_legacy=parse_strategy_for_df,
    )
    strategy2 = parse_strategy_identifier(
        block["seat2_strategy"],
        manifest=manifest,
        parse_legacy=parse_strategy_for_df,
    )
    games_required = int(block["games_required"])
    wins_seat1 = 0
    wins_seat2 = 0
    for start in range(0, games_required, chunk_games):
        stop = min(games_required, start + chunk_games)
        seeds = [
            coordinate_seed(
                RandomPurpose.H2H_GAME,
                root_seed=int(block["root_seed"]),
                k=2,
                pair_index=int(block["pair_id"]),
                order=int(block["order"]),
                game_index=game_index,
            )
            for game_index in range(start, stop)
        ]
        frame = simulate_many_games_from_seeds(
            seeds=seeds,
            strategies=[strategy1, strategy2],
            n_jobs=1,
            root_seed=int(block["root_seed"]),
        )
        first, second = _winner_seat_counts(frame)
        wins_seat1 += first
        wins_seat2 += second
    if wins_seat1 + wins_seat2 != games_required:
        raise RuntimeError("H2H block win counts do not equal scheduled games")
    return {
        **block,
        "games_completed": games_required,
        "wins_seat1": wins_seat1,
        "wins_seat2": wins_seat2,
    }


def _block_path(cfg: AppConfig, block: Mapping[str, Any]) -> Path:
    return cfg.h2h_block_result_path(
        int(block["pair_id"]),
        int(block["root_seed"]),
        int(block["order"]),
    )


def _valid_existing_block(path: Path, block: Mapping[str, Any]) -> bool:
    if not path.exists():
        return False
    validate_artifact_sidecar(
        path,
        expected={
            "scope": ArtifactScope.H2H_2P.value,
            "operation": "simulate_root_order_block",
        },
    )
    frame = pq.read_table(
        path,
        columns=["block_id", "family_hash", "games_completed"],
    ).to_pandas()
    if len(frame) != 1:
        raise ValueError(f"H2H block checkpoint must contain one row: {path}")
    row = frame.iloc[0]
    if (
        str(row["block_id"]) != str(block["block_id"])
        or str(row["family_hash"]) != str(block["family_hash"])
        or int(cast(int, row["games_completed"])) != int(block["games_required"])
    ):
        raise ValueError(f"immutable H2H block checkpoint conflicts with schedule: {path}")
    return True


def _write_block(
    cfg: AppConfig,
    result: dict[str, Any],
    *,
    schedule_path: Path,
    roots: tuple[int, ...],
) -> Path:
    path = _block_path(cfg, result)
    frame = pd.DataFrame([result])
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="h2h_schedule",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="simulate_root_order_block",
        baseline="equal_seat_order_rates",
        weighted_quantity="seat1_win_count",
        support_count_role="independent_games_per_root_order_block",
        uncertainty_method=SCORE_TEST_ID,
        replication_unit="independent_h2h_game",
        conditioning="frozen_finite_grid_candidate_family",
        consistency_columns=frame.columns.tolist(),
        source_artifacts=[schedule_path],
        grouping_keys=["pair_id", "root_seed", "order"],
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined" if len(roots) == 2 else "single_root",
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        sidecar=sidecar,
        codec=cfg.parquet_codec,
    )
    return path


BlockRunner = Callable[[dict[str, Any], Path, int], dict[str, Any]]


def execute_h2h_schedule(
    cfg: AppConfig,
    *,
    n_jobs: int | None = None,
    chunk_games: int = 1_000,
    block_runner: BlockRunner = _simulate_block,
) -> H2HExecutionArtifacts:
    """Execute missing immutable blocks and publish their row-preserving union."""

    if chunk_games < 1:
        raise ValueError("chunk_games must be positive")
    plan_path = cfg.h2h_power_plan_path()
    schedule_path = cfg.h2h_block_manifest_path()
    validate_artifact_sidecar(
        plan_path,
        expected={"scope": ArtifactScope.H2H_2P.value, "operation": "score_test_power_plan"},
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schedule_state") != "ready":
        raise RuntimeError(
            "H2H execution is blocked; raise head2head.total_game_cap to the planned workload"
        )
    validate_artifact_sidecar(
        schedule_path,
        expected={
            "scope": ArtifactScope.H2H_2P.value,
            "operation": "construct_pair_root_order_blocks",
        },
    )
    schedule = (
        pq.read_table(schedule_path)
        .to_pandas()
        .sort_values(["pair_id", "root_index", "order"], kind="mergesort")
    )
    roots = tuple(int(value) for value in plan["root_seeds"])
    records = cast(list[dict[str, Any]], schedule.to_dict(orient="records"))
    block_paths = tuple(_block_path(cfg, record) for record in records)
    pending = [
        record
        for record, path in zip(records, block_paths, strict=True)
        if not _valid_existing_block(path, record)
    ]
    manifest_path = cfg.strategy_manifest_root_path()
    if pending and not manifest_path.exists():
        raise FileNotFoundError(f"strategy manifest is required for H2H execution: {manifest_path}")

    configured_jobs = cfg.head2head.n_jobs if n_jobs is None else n_jobs
    worker_count = (os.cpu_count() or 1) if configured_jobs == 0 else int(configured_jobs)
    if worker_count < 1:
        raise ValueError("H2H process executor worker count must be positive or zero for auto")
    if block_runner is not _simulate_block and worker_count != 1:
        raise ValueError("custom H2H block runners require n_jobs=1")
    if worker_count == 1:
        for block in pending:
            _write_block(
                cfg,
                block_runner(block, manifest_path, chunk_games),
                schedule_path=schedule_path,
                roots=roots,
            )
    elif pending:
        with ProcessPoolExecutor(max_workers=min(worker_count, len(pending))) as executor:
            iterator = iter(pending)
            active: dict[Future[dict[str, Any]], dict[str, Any]] = {}

            def submit_one() -> bool:
                try:
                    block = next(iterator)
                except StopIteration:
                    return False
                future = executor.submit(_simulate_block, block, manifest_path, chunk_games)
                active[future] = block
                return True

            for _ in range(min(len(pending), worker_count * 2)):
                submit_one()
            while active:
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    active.pop(future)
                    _write_block(
                        cfg,
                        future.result(),
                        schedule_path=schedule_path,
                        roots=roots,
                    )
                    submit_one()

    completed_records: list[dict[str, Any]] = []
    for record, path in zip(records, block_paths, strict=True):
        if not _valid_existing_block(path, record):
            raise RuntimeError(f"scheduled H2H block did not complete: {path}")
        block_rows = pq.read_table(path).to_pylist()
        if len(block_rows) != 1:
            raise ValueError(f"H2H block checkpoint must contain one row: {path}")
        completed_records.append(block_rows[0])
    combined = pd.DataFrame(completed_records).sort_values(
        ["pair_id", "root_index", "order"], kind="mergesort"
    )
    output = cfg.h2h_order_counts_path()
    sidecar = make_artifact_sidecar(
        cfg,
        output,
        producer="h2h_schedule",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="concatenate_root_order_blocks",
        baseline="equal_seat_order_rates",
        weighted_quantity="seat1_win_count",
        support_count_role="independent_games_per_root_order_block",
        uncertainty_method=SCORE_TEST_ID,
        replication_unit="independent_h2h_game",
        conditioning="frozen_finite_grid_candidate_family",
        consistency_columns=combined.columns.tolist(),
        source_artifacts=list(block_paths),
        grouping_keys=["pair_id", "root_seed", "order"],
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined" if len(roots) == 2 else "single_root",
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(combined, preserve_index=False),
        output,
        sidecar=sidecar,
        codec=cfg.parquet_codec,
    )
    write_stage_done(
        stage_done_path(cfg.h2h_2p_dir(), "h2h_block_execution"),
        inputs=[schedule_path],
        outputs=[output],
        cfg=cfg,
        stage="head2head",
        sidecar_artifacts=[output],
    )
    return H2HExecutionArtifacts(order_counts=output, block_paths=block_paths)


__all__ = [
    "H2HExecutionArtifacts",
    "H2HScheduleArtifacts",
    "POWER_METHOD_ID",
    "SCORE_TEST_ID",
    "execute_h2h_schedule",
    "independent_score_planning_power",
    "plan_h2h_schedule",
]
