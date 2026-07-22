"""Hand-built mixed-outcome oracles for root tournament analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis import game_stats
from farkle.analysis.all_player_metrics import build_all_player_batch_metrics
from farkle.analysis.performance import build_canonical_performance
from farkle.analysis.seat_analysis import build_canonical_seat_analysis
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.simulation.run_tournament import OutcomeCounter
from farkle.utils.artifact_contract import make_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.schema_helpers import expected_schema_for


def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=41, n_players_list=[2, 3]),
    )
    cfg.analysis.n_jobs = 1
    cfg.screening.bootstrap_replicates = 20
    cfg.screening.candidate_contribution_size = 1
    cfg.screening.controls = [10]
    cfg.screening.practical_delta_by_k = {2: 0.05, 3: 0.05}
    cfg.screening.delta_across_k = 0.05
    return cfg


def _row(
    *,
    k: int,
    game: int,
    batch: int,
    strategies: list[int],
    winner_seat: int | None,
) -> dict[str, object]:
    completed = winner_seat is not None
    scores = [60 + seat for seat in range(k)]
    if completed:
        scores[winner_seat - 1] = 120
        ordered_seats = [winner_seat, *[seat for seat in range(1, k + 1) if seat != winner_seat]]
        ranks = {seat: rank for rank, seat in enumerate(ordered_seats, start=1)}
    else:
        ordered_seats = []
        ranks = {}
    row: dict[str, object] = {
        "root_seed": 41,
        "k": k,
        "shuffle_index": game,
        "game_index": game,
        "deterministic_batch_id": batch,
        "shuffle_seed": 1_000 + game,
        "termination_status": "completed" if completed else "safety_limit",
        "hit_safety_limit": not completed,
        "outcome_schema_version": 2,
        "winner_seat": f"P{winner_seat}" if completed else None,
        "winner_strategy": strategies[winner_seat - 1] if completed else None,
        "game_seed": 2_000 + game,
        "rng_scheme_version": 1,
        "rng_purpose_namespace": 102,
        "seat_ranks": [f"P{seat}" for seat in ordered_seats] if completed else [None] * k,
        "winning_score": 120 if completed else None,
        "victory_margin": 59 if completed else None,
        "n_rounds": 4 if completed else 7,
    }
    for seat, strategy in enumerate(strategies, start=1):
        rank = ranks.get(seat)
        row.update(
            {
                f"P{seat}_score": scores[seat - 1],
                f"P{seat}_farkles": seat,
                f"P{seat}_rolls": 2 * (4 if completed else 7),
                f"P{seat}_highest_turn": scores[seat - 1],
                f"P{seat}_strategy": strategy,
                f"P{seat}_rank": rank,
                f"P{seat}_loss_margin": 0 if rank == 1 else (59 if completed else None),
                f"P{seat}_smart_five_uses": 0,
                f"P{seat}_n_smart_five_dice": 0,
                f"P{seat}_smart_one_uses": 0,
                f"P{seat}_n_smart_one_dice": 0,
                f"P{seat}_hot_dice": 0,
                f"P{seat}_n_turns": 4 if completed else 7,
                f"P{seat}_hit_max_rounds": not completed,
            }
        )
    return row


def _mixed_rows(k: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for batch in (0, 1):
        offset = batch * 10
        if k == 2:
            rows.extend(
                [
                    _row(
                        k=2,
                        game=offset,
                        batch=batch,
                        strategies=[10, 20],
                        winner_seat=2,
                    ),
                    _row(
                        k=2,
                        game=offset + 1,
                        batch=batch,
                        strategies=[20, 30],
                        winner_seat=None,
                    ),
                    _row(
                        k=2,
                        game=offset + 2,
                        batch=batch,
                        strategies=[30, 10],
                        winner_seat=2,
                    ),
                    _row(
                        k=2,
                        game=offset + 3,
                        batch=batch,
                        strategies=[10, 10],
                        winner_seat=None,
                    ),
                ]
            )
        else:
            rows.extend(
                [
                    _row(
                        k=3,
                        game=offset,
                        batch=batch,
                        strategies=[10, 20, 30],
                        winner_seat=3,
                    ),
                    _row(
                        k=3,
                        game=offset + 1,
                        batch=batch,
                        strategies=[10, 20, 30],
                        winner_seat=None,
                    ),
                ]
            )
    return rows


def _write_inputs(cfg: AppConfig) -> dict[int, list[dict[str, object]]]:
    rows_by_k = {k: _mixed_rows(k) for k in (2, 3)}
    combined_frames: list[pd.DataFrame] = []
    for k, rows in rows_by_k.items():
        table = pa.Table.from_pylist(rows, schema=expected_schema_for(k))
        ingested = cfg.ingested_rows_curated(k)
        ingested.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, ingested)
        combined = cfg.combined_rows_by_k(k)
        sidecar = make_artifact_sidecar(
            cfg,
            combined,
            producer="test",
            scope=ArtifactScope.BY_K,
            source_scope=ArtifactScope.BY_K,
            operation="concatenate_rows_within_k",
            consistency_columns=table.schema.names,
            player_counts=[k],
            required_player_counts=[k],
            missing_cell_policy="fail",
        )
        write_parquet_artifact_atomic(table, combined, sidecar=sidecar)
        combined_frames.append(table.to_pandas())
    cfg.curated_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(combined_frames, ignore_index=True, sort=False).to_parquet(cfg.curated_parquet)
    return rows_by_k


def test_mixed_outcomes_propagate_without_fabricated_winner(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    rows_by_k = _write_inputs(cfg)

    counter = OutcomeCounter()
    for row in rows_by_k[2]:
        status = counter.record_row(row, k=2, source="hand fixture")
        if status.value == "completed":
            counter[row["winner_strategy"]] += 1
    assert counter.games_attempted == 8
    assert counter.games_completed == 4
    assert counter.games_safety_limit == 4
    assert sum(counter.values()) == counter.games_completed
    assert all(row["winner_strategy"] is None for row in rows_by_k[2] if row["hit_safety_limit"])

    for k in (2, 3):
        build_all_player_batch_metrics(cfg, k)
    metrics_2 = pq.read_table(cfg.metrics_all_player_batch_path(2)).to_pandas()
    totals_2 = metrics_2.groupby("strategy", as_index=True).sum(numeric_only=True)
    assert totals_2.loc[10, "raw_player_game_exposures"] == 8
    assert totals_2.loc[10, "raw_completed_player_game_exposures"] == 4
    assert totals_2.loc[10, "raw_safety_limit_player_game_exposures"] == 4
    assert totals_2.loc[10, "raw_wins"] == 2
    assert totals_2.loc[10, "raw_losses"] == 6
    assert totals_2.loc[10, "raw_n_turns_sum"] == 44

    performance = build_canonical_performance(cfg)
    by_k_2 = pq.read_table(performance.by_k[0]).to_pandas().set_index("strategy")
    assert by_k_2.loc[10, "win_rate_per_attempt"] == pytest.approx(0.25)
    assert by_k_2.loc[10, "chance_delta"] == pytest.approx(-0.25)
    assert by_k_2.loc[10, "win_rate_given_completion"] == pytest.approx(0.5)
    assert by_k_2.loc[10, "safety_limit_exposure_rate"] == pytest.approx(0.5)
    assert by_k_2.loc[10, "batch_mcse"] == pytest.approx(0.0)
    across = pq.read_table(performance.across_k).to_pandas().set_index("strategy")
    assert across.loc[10, "raw_attempted_exposures"] == 12
    assert across.loc[10, "raw_completed_exposures"] == 6
    assert across.loc[10, "raw_safety_limit_exposures"] == 6

    seat = build_canonical_seat_analysis(cfg)
    counts_2 = pq.read_table(seat.batch_counts[0]).to_pandas()
    assert counts_2["raw_exposures"].sum() == 16
    assert counts_2["raw_completed_exposures"].sum() == 8
    assert counts_2["raw_safety_limit_exposures"].sum() == 8
    assert counts_2["raw_wins"].sum() == 4
    selfplay = pq.read_table(seat.selfplay_diagnostic).to_pandas()
    selfplay_10 = selfplay.loc[(selfplay["k"] == 2) & (selfplay["strategy"] == 10)].iloc[0]
    assert selfplay_10["games_attempted"] == 2
    assert selfplay_10["games_completed"] == 0
    assert selfplay_10["games_safety_limit"] == 2
    assert selfplay_10["p1_wins"] == 0
    assert selfplay_10["p1_win_rate_per_attempt"] == pytest.approx(0.0)

    game_stats.run(cfg, force=True)
    game_length = pd.read_parquet(cfg.game_stats_concat_path("game_length.parquet"))
    strategy_rows = game_length.loc[game_length["summary_level"].eq("strategy")]
    assert set(strategy_rows["strategy"].dropna().astype(int)) == {10, 20, 30}
    for k, attempted_games in ((2, 8), (3, 4)):
        cell = strategy_rows.loc[strategy_rows["n_players"].eq(k)]
        assert cell["observations"].sum() == attempted_games * k
        assert cell["completed_observations"].sum() == (4 if k == 2 else 2) * k
        assert cell["safety_limit_observations"].sum() == (4 if k == 2 else 2) * k
    assert game_stats._seat_strategy_columns(
        ["P2_strategy", "winner_strategy", "P1_strategy", "not_a_strategy"]
    ) == ["P1_strategy", "P2_strategy"]

    rare = pd.read_parquet(cfg.game_stats_output_path("rare_events.parquet"))
    rare_strategy = rare.loc[rare["summary_level"].eq("strategy")]
    for k, attempted_games in ((2, 8), (3, 4)):
        cell = rare_strategy.loc[rare_strategy["n_players"].eq(k)]
        assert cell["observations"].sum() == attempted_games * k
        assert cell["completed_observations"].sum() == (4 if k == 2 else 2) * k
        assert cell["safety_limit_observations"].sum() == (4 if k == 2 else 2) * k
