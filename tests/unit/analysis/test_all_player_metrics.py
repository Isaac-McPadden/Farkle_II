from __future__ import annotations

from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.helpers.artifact_sidecars import (
    make_authenticated_v3_config,
    publish_v3_parquet,
)

from farkle.analysis.all_player_metrics import (
    _execution_batch_limits,
    all_player_batch_schema,
    build_all_player_batch_metrics,
    validate_unconditional_all_player_schema,
)
from farkle.config import AppConfig
from farkle.utils.artifact_contract import validate_artifact_sidecar
from farkle.utils.schema_helpers import expected_schema_for


def _cfg(tmp_path: Path) -> AppConfig:
    return make_authenticated_v3_config(tmp_path, name="all_player_metrics", root_seed=7)


def _exposure_values(
    strategy: int,
    score: int,
    turns: int,
    rank: int,
    *,
    hit_max_rounds: bool = False,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "score": score,
        "n_turns": turns,
        "hit_max_rounds": hit_max_rounds,
        "rank": rank,
        "loss_margin": 0 if rank == 1 else 50,
        "rolls": turns + 1,
        "farkles": 1,
        "highest_turn": score,
        "hot_dice": 0,
        "smart_five_uses": 1,
        "n_smart_five_dice": 5,
        "smart_one_uses": 0,
        "n_smart_one_dice": 0,
    }


def _game_row(
    *,
    shuffle_index: int,
    winner_seat: str,
    n_rounds: int,
    p1: dict[str, object],
    p2: dict[str, object],
) -> dict[str, object]:
    row: dict[str, object] = {
        "root_seed": 7,
        "k": 2,
        "shuffle_index": shuffle_index,
        "game_index": 0,
        "deterministic_batch_id": 0,
        "shuffle_seed": 100 + shuffle_index,
        "winner_seat": winner_seat,
        "winner_strategy": p1["strategy"] if winner_seat == "P1" else p2["strategy"],
        "termination_status": "completed",
        "outcome_schema_version": 2,
        "game_seed": 200 + shuffle_index,
        "rng_scheme_version": 2,
        "rng_purpose_namespace": 102,
        "seat_ranks": [winner_seat, "P2" if winner_seat == "P1" else "P1"],
        "winning_score": max(cast(int, p1["score"]), cast(int, p2["score"])),
        "n_rounds": n_rounds,
    }
    for seat, exposure in ((1, p1), (2, p2)):
        row.update({f"P{seat}_{name}": value for name, value in exposure.items()})
    return row


def _write_source(cfg: AppConfig, rows: list[dict[str, object]]) -> Path:
    path = cfg.ingested_rows_curated(2)
    return publish_v3_parquet(
        cfg,
        path,
        pa.Table.from_pylist(rows, schema=expected_schema_for(2)),
        stage_key="curate",
        producer="curate",
        operation="curate_game_rows",
        source_scope="by_k",
    )


def test_all_player_turn_returns_include_zero_score_turns(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_source(
        cfg,
        [
            _game_row(
                shuffle_index=0,
                winner_seat="P1",
                n_rounds=2,
                p1=_exposure_values(10, 100, 2, 1),
                p2=_exposure_values(20, 50, 3, 2),
            ),
            _game_row(
                shuffle_index=1,
                winner_seat="P1",
                n_rounds=4,
                p1=_exposure_values(20, 80, 4, 1),
                p2=_exposure_values(10, 0, 4, 2),
            ),
        ],
    )

    output = build_all_player_batch_metrics(cfg, 2)
    rows = {row["strategy"]: row for row in pq.read_table(output).to_pylist()}

    strategy_10 = rows[10]
    assert strategy_10["raw_player_game_exposures"] == 2
    assert strategy_10["raw_wins"] == 1
    assert strategy_10["raw_final_score_sum"] == pytest.approx(100)
    assert strategy_10["raw_n_turns_sum"] == pytest.approx(6)
    assert strategy_10["turn_return_turn_weighted"] == pytest.approx(100 / 6)
    assert strategy_10["turn_return_game_weighted_exact"] == pytest.approx(25)
    assert strategy_10["turn_return_round_proxy"] == pytest.approx(25)
    assert strategy_10["turn_round_mismatch_prevalence"] == pytest.approx(0)

    strategy_20 = rows[20]
    assert strategy_20["turn_return_turn_weighted"] == pytest.approx(130 / 7)
    assert strategy_20["turn_return_game_weighted_exact"] == pytest.approx((50 / 3 + 20) / 2)
    assert strategy_20["turn_return_round_proxy"] == pytest.approx(22.5)
    assert strategy_20["round_proxy_gap"] == pytest.approx(22.5 - (50 / 3 + 20) / 2)
    assert strategy_20["turn_round_mismatch_prevalence"] == pytest.approx(0.5)
    assert strategy_20["raw_rank_observations"] == 2
    assert strategy_20["raw_rank_sum"] == pytest.approx(3)
    assert strategy_20["raw_max_round_abort_exposures"] == 0
    assert strategy_10["raw_max_round_abort_exposures"] == 0

    validate_artifact_sidecar(
        output,
        expected={
            "scope": "by_k",
            "operation": "aggregate_player_batch_statistics",
            "conditioning": "all_attempted_player_game_exposures_safety_limit_is_loss",
            "method_contract": {
                "kind": "turn_metrics",
                "procedure": "aggregate_player_batch_statistics",
                "parameters": {
                    "exposure_denominator": "player_game_exposure",
                    "completed_diagnostic_denominator": "completed_player_game_exposure",
                    "safety_limit_numerator": "safety_limit_player_game_exposure",
                    "outcome_schema_version": 2,
                    "tournament_method_version": 2,
                },
            },
        },
    )


def test_unconditional_schema_rejects_conditional_fields() -> None:
    schema = all_player_batch_schema().append(pa.field("win_conditioned_score", pa.float64()))
    with pytest.raises(ValueError, match="winner-conditioned"):
        validate_unconditional_all_player_schema(schema)


def test_all_player_metrics_reject_missing_exact_turn_counter(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    row = _game_row(
        shuffle_index=0,
        winner_seat="P1",
        n_rounds=2,
        p1=_exposure_values(10, 100, 2, 1),
        p2=_exposure_values(20, 50, 3, 2),
    )
    row["P2_n_turns"] = None
    _write_source(cfg, [row])

    with pytest.raises(ValueError, match="coordinate-and-turn row contract"):
        build_all_player_batch_metrics(cfg, 2)


def test_all_player_metrics_accept_zero_turn_safety_limit_as_attempted_loss(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    row = _game_row(
        shuffle_index=0,
        winner_seat="P1",
        n_rounds=0,
        p1=_exposure_values(10, 0, 0, 1, hit_max_rounds=True),
        p2=_exposure_values(20, 0, 0, 2, hit_max_rounds=True),
    )
    row.update(
        {
            "winner_seat": None,
            "winner_strategy": None,
            "termination_status": "safety_limit",
            "hit_safety_limit": True,
            "winning_score": None,
            "seat_ranks": [None, None],
            "P1_rank": None,
            "P2_rank": None,
            "P1_loss_margin": None,
            "P2_loss_margin": None,
        }
    )
    _write_source(cfg, [row])

    output = build_all_player_batch_metrics(cfg, 2)
    rows = pq.read_table(output).to_pylist()

    assert len(rows) == 2
    for metric in rows:
        assert metric["raw_player_game_exposures"] == 1
        assert metric["raw_completed_player_game_exposures"] == 0
        assert metric["raw_safety_limit_player_game_exposures"] == 1
        assert metric["raw_wins"] == 0
        assert metric["raw_losses"] == 1
        assert metric["raw_n_turns_sum"] == 0
        assert metric["turn_return_turn_weighted"] is None
        assert metric["turn_return_game_weighted_exact"] == 0
        assert metric["turn_return_round_proxy"] == 0


def test_arrow_execution_batches_are_byte_bounded_and_logically_invariant(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    rows = [
        _game_row(
            shuffle_index=index,
            winner_seat="P1" if index % 2 == 0 else "P2",
            n_rounds=3 + index % 3,
            p1=_exposure_values(10 if index % 2 else 20, 101 + index, 3 + index % 3, 1),
            p2=_exposure_values(20 if index % 2 else 10, 47 + index, 4 + index % 3, 2),
        )
        for index in range(40)
    ]
    source = _write_source(cfg, rows)
    cfg.resources.stage_batch_bytes["all_player_metrics"] = 512
    small_bytes, small_rows = _execution_batch_limits(cfg, source, 2)
    small_output = build_all_player_batch_metrics(cfg, 2, force=True)
    small = pq.read_table(small_output)

    cfg.resources.stage_batch_bytes["all_player_metrics"] = 1024 * 1024
    large_bytes, large_rows = _execution_batch_limits(cfg, source, 2)
    large_output = build_all_player_batch_metrics(cfg, 2, force=True)
    large = pq.read_table(large_output)

    assert small_bytes < large_bytes
    assert small_rows < large_rows
    assert small.equals(large)


def test_all_player_root_k_checkpoint_reuses_and_repairs_corruption(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_source(
        cfg,
        [
            _game_row(
                shuffle_index=0,
                winner_seat="P1",
                n_rounds=2,
                p1=_exposure_values(10, 100, 2, 1),
                p2=_exposure_values(20, 50, 3, 2),
            )
        ],
    )
    output = build_all_player_batch_metrics(cfg, 2)
    original = output.read_bytes()
    original_mtime = output.stat().st_mtime_ns

    assert build_all_player_batch_metrics(cfg, 2) == output
    assert output.stat().st_mtime_ns == original_mtime

    output.write_bytes(b"corrupt")
    repaired = build_all_player_batch_metrics(cfg, 2)
    assert repaired.read_bytes() == original
    assert pq.read_table(repaired).num_rows == 2
