from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.helpers.artifact_sidecars import make_authenticated_v3_config, publish_v3_parquet

from farkle.analysis import outcome_distributions as outcomes
from farkle.utils.artifact_contract import ArtifactContractError, sidecar_path
from farkle.utils.schema_helpers import expected_schema_for


def _row(scores, strategies, *, shuffle=0, game=0, stopped=False, width=2, root=11):
    k = len(scores)
    order = sorted(range(k), key=lambda i: (-scores[i], i))
    winner = order[0]
    ranks = {seat: rank for rank, seat in enumerate(order, 1)}
    row = {
        "root_seed": root,
        "k": k,
        "shuffle_index": shuffle,
        "game_index": game,
        "deterministic_batch_id": shuffle // width,
        "outcome_schema_version": 2,
        "termination_status": "safety_limit" if stopped else "completed",
        "hit_safety_limit": stopped,
        "winner_seat": None if stopped else f"P{winner + 1}",
        "winner_strategy": None if stopped else strategies[winner],
        "winning_score": None if stopped else scores[winner],
        "victory_margin": None if stopped else scores[winner] - scores[order[1]],
        "seat_ranks": [None] * k if stopped else [f"P{i + 1}" for i in order],
    }
    for seat, (score, strategy) in enumerate(zip(scores, strategies, strict=True), 1):
        row.update(
            {
                f"P{seat}_score": score,
                f"P{seat}_strategy": strategy,
                f"P{seat}_rank": None if stopped else ranks[seat - 1],
                f"P{seat}_loss_margin": None if stopped else scores[winner] - score,
                f"P{seat}_hit_max_rounds": stopped,
            }
        )
    return row


def _table(rows, k=4):
    return pa.Table.from_pylist(rows, schema=expected_schema_for(k))


def _accumulate(rows, k=4):
    return outcomes.accumulate_outcome_slice(
        _table(rows, k), root_seed=11, k=k, shuffles_per_batch=2
    )


def _frequency_map(table):
    return {
        (r["strategy"], r["seat"], r["distribution"], r["value"]): r["count"]
        for r in table.to_pylist()
    }


def test_tied_scores_have_shared_ranks_but_one_official_winner() -> None:
    table = _table([_row([10000, 8000, 8000, 5000], [10, 20, 20, 30])])
    before = table.to_pylist()
    result = outcomes.accumulate_outcome_slice(table, root_seed=11, k=4, shuffles_per_batch=2)
    assert table.to_pylist() == before  # No mutation of engine ranks or official winner.
    bins = _frequency_map(result.frequency_views())
    for seat, strategy, rank, midrank in (
        (1, 10, 1, 2),
        (2, 20, 2, 5),
        (3, 20, 2, 5),
        (4, 30, 4, 8),
    ):
        assert bins[strategy, seat, "competition_rank", rank] == 1
        assert bins[strategy, seat, "midrank", midrank] == 1
    assert bins[20, 0, "completed_score", 8000] == 2
    assert bins[20, 0, "losing_margin", 2000] == 2
    assert bins[10, 0, "winning_margin", 2000] == 1
    assert result.batch_stats["wins"].to_pylist() == [1, 0, 0, 0]
    assert {
        r["value_scale"]
        for r in result.batch_frequencies.to_pylist()
        if r["distribution"] == "midrank"
    } == {2}


def test_tied_first_official_loser_keeps_zero_margin_and_repeated_winner_strategy() -> None:
    result = _accumulate([_row([10000, 10000, 7000, 0], [20, 10, 30, 20])])
    bins = _frequency_map(result.frequency_views())
    assert bins[10, 2, "competition_rank", 1] == 1
    assert bins[10, 2, "midrank", 3] == 1
    assert bins[10, 2, "losing_margin", 0] == 1
    assert bins[20, 1, "winning_margin", 0] == 1
    assert bins[20, 4, "losing_margin", 10000] == 1
    assert not any(key[:3] == (10, 2, "winning_margin") for key in bins)
    support = {r["strategy"]: r for r in result.support_views().to_pylist() if r["seat"] == 0}
    assert support[20]["wins"] == support[20]["losses_completed"] == 1
    assert support[10]["wins"] == 0 and support[10]["losses_completed"] == 1
    assert support[10]["deficit_sum"] == 0  # observed zero with positive conditional support


def test_stopped_scores_and_undefined_conditionals_have_explicit_support() -> None:
    result = _accumulate([_row([100, 100, 0, 50], [10, 20, 30, 10], stopped=True)])
    assert set(result.batch_frequencies["distribution"].to_pylist()) == {"stopped_score"}
    assert sum(map(int, result.batch_frequencies["count"].to_numpy())) == 4
    for row in result.support_views().to_pylist():
        assert row["attempted"] == row["safety"] > 0
        assert row["completed"] == row["wins"] == row["losses_completed"] == 0
        assert row["support_status"] == "observed"
    # Positive completed support, but no losses for strategy 10, is distinct.
    completed = _accumulate([_row([10000, 8000, 8000, 5000], [10, 20, 20, 30])])
    first = next(
        r for r in completed.support_views().to_pylist() if r["strategy"] == 10 and r["seat"] == 0
    )
    assert first["completed"] == 1 and first["losses_completed"] == first["deficit_sum"] == 0
    assert not any(
        r["strategy"] == 10 and r["distribution"] == "losing_margin"
        for r in completed.frequency_views().to_pylist()
    )


def test_empty_partial_schemas_and_sparse_support_do_not_certify_zeros() -> None:
    empty = _accumulate([])
    for table, schema in (
        (empty.batch_frequencies, outcomes.outcome_frequency_schema(batch=True)),
        (empty.batch_stats, outcomes.outcome_support_schema(batch=True)),
        (empty.frequency_views(), outcomes.outcome_frequency_schema()),
        (empty.support_views(), outcomes.outcome_support_schema()),
        (empty.coordinates, outcomes.outcome_coordinate_schema()),
    ):
        assert table.num_rows == 0 and table.schema == schema
        assert all(not field.nullable and not pa.types.is_null(field.type) for field in schema)
    result = _accumulate([_row([20, 10, 0, 0], [10, 20, 20, 20])])
    assert not any(
        r["strategy"] == 10 and r["seat"] == 2 for r in result.support_views().to_pylist()
    )
    assert set(result.batch_stats["support_status"].to_pylist()) == {"observed"}


def _oracle(rows):
    """Independent row/seat Counter oracle, including overall bins and batch vectors."""
    freq = Counter()
    stats = defaultdict(lambda: [0] * 7)
    for row in rows:
        k = row["k"]
        scores = [row[f"P{s}_score"] for s in range(1, k + 1)]
        ordered = sorted(scores, reverse=True)
        completed = row["termination_status"] == "completed"
        for seat, score in enumerate(scores, 1):
            strategy = row[f"P{seat}_strategy"]
            win = completed and row["winner_seat"] == f"P{seat}"
            loss = completed and not win
            deficit = row["winning_score"] - score if loss else 0
            victory = row["victory_margin"] if win else 0
            key = (row["root_seed"], k, row["deterministic_batch_id"], strategy, seat)
            for i, value in enumerate(
                [1, int(completed), int(not completed), int(win), int(loss), deficit, victory]
            ):
                stats[key][i] += value
            if completed:
                positions = [i + 1 for i, value in enumerate(ordered) if value == score]
                bins = [
                    ("completed_score", score),
                    ("competition_rank", min(positions)),
                    ("midrank", min(positions) + max(positions)),
                    ("winning_margin", victory) if win else ("losing_margin", deficit),
                ]
            else:
                bins = [("stopped_score", score)]
            for distribution, value in bins:
                freq[(*key, distribution, value)] += 1
    return freq, dict(stats)


@pytest.mark.parametrize("k", [2, 4, 12])
@pytest.mark.parametrize("slice_games", [1, 7, 43])
def test_exact_counts_conservation_and_slice_invariance(k: int, slice_games: int) -> None:
    rows = [
        _row(
            [((g + seat * 3) % 7) * 50 for seat in range(k)],
            [10 + (g + seat) % 5 for seat in range(k)],
            shuffle=g // 3,
            game=g % 3,
            stopped=g % 6 == 0,
        )
        for g in range(43)
    ]
    expected_f, expected_b = _oracle(rows)
    actual_f = Counter()
    actual_b = defaultdict(lambda: [0] * 7)
    for start in range(0, len(rows), slice_games):
        result = _accumulate(rows[start : start + slice_games], k)
        for r in result.batch_frequencies.to_pylist():
            key = tuple(
                r[name]
                for name in (
                    "root_seed",
                    "k",
                    "deterministic_batch_id",
                    "strategy",
                    "seat",
                    "distribution",
                    "value",
                )
            )
            actual_f[key] += r["count"]
        for r in result.batch_stats.to_pylist():
            key = tuple(
                r[name] for name in ("root_seed", "k", "deterministic_batch_id", "strategy", "seat")
            )
            for i, field in enumerate(outcomes._COUNTS):
                actual_b[key][i] += r[field]
        view = result.frequency_views().to_pylist()
        overall = Counter(
            {
                (r["strategy"], r["distribution"], r["value"]): r["count"]
                for r in view
                if r["seat"] == 0
            }
        )
        actual_seats = Counter()
        for r in view:
            if r["seat"]:
                actual_seats[r["strategy"], r["distribution"], r["value"]] += r["count"]
        assert actual_seats == overall
        for r in result.support_views().to_pylist():
            assert r["attempted"] == r["completed"] + r["safety"]
            assert r["completed"] == r["wins"] + r["losses_completed"]
    assert actual_f == expected_f
    assert dict(actual_b) == expected_b
    completed = sum(row["termination_status"] == "completed" for row in rows)
    assert sum(v[0] for v in actual_b.values()) == k * len(rows)
    assert sum(v[1] for v in actual_b.values()) == k * completed
    assert sum(v[3] for v in actual_b.values()) == completed


@pytest.mark.parametrize(
    "field,value",
    [
        ("root_seed", None),
        ("k", 2),
        ("game_index", None),
        ("shuffle_index", -1),
        ("deterministic_batch_id", 1),
        ("outcome_schema_version", 1),
        ("P1_strategy", None),
        ("P4_score", None),
        ("P3_score", -1),
        ("winner_seat", None),
        ("winner_strategy", 99),
        ("winning_score", 9999),
        ("victory_margin", 1),
        ("P2_rank", 1),
        ("P2_loss_margin", 1),
        ("seat_ranks", ["P2", "P1", "P3", "P4"]),
        ("termination_status", None),
        ("hit_safety_limit", True),
        ("P1_hit_max_rounds", None),
    ],
)
def test_malformed_required_outcomes_fail(field, value) -> None:
    row = _row([10000, 8000, 8000, 5000], [10, 20, 20, 30])
    row[field] = value
    with pytest.raises(ValueError):
        _accumulate([row])


@pytest.mark.parametrize(
    "field,value",
    [("winner_seat", "P1"), ("winning_score", 100), ("P2_rank", 2), ("P3_loss_margin", 0)],
)
def test_interrupted_games_cannot_claim_completed_outcomes(field, value) -> None:
    row = _row([100, 100, 0, 50], [10, 20, 30, 10], stopped=True)
    row[field] = value
    with pytest.raises(ValueError):
        _accumulate([row])


def test_missing_columns_wrong_types_duplicates_and_budget_fail() -> None:
    row = _row([10, 0, 0, 0], [1, 2, 3, 4])
    table = _table([row])
    options = {"root_seed": 11, "k": 4, "shuffles_per_batch": 2}
    with pytest.raises(ValueError, match="P4_score"):
        outcomes.accumulate_outcome_slice(table.drop(["P4_score"]), **options)
    with pytest.raises(ValueError, match="incompatible type"):
        outcomes.accumulate_outcome_slice(
            table.set_column(
                table.schema.get_field_index("P1_score"), "P1_score", pa.array([10.0])
            ),
            **options,
        )
    with pytest.raises(ValueError, match="duplicate"):
        _accumulate([row, row])
    with pytest.raises(MemoryError, match="one game"):
        outcomes.accumulate_outcome_slice(table, max_bytes=10, **options)
    with pytest.raises(MemoryError, match="split"):
        outcomes.accumulate_outcome_slice(
            pa.concat_tables([table, table]), max_bytes=4 * 4096 * 4, **options
        )


def test_padding_is_not_an_exposure_and_counts_are_checked_int64() -> None:
    row = _row([100, 50], [1, 2])
    padded = pa.Table.from_pylist([row], schema=expected_schema_for(12))
    result = outcomes.accumulate_outcome_slice(padded, root_seed=11, k=2, shuffles_per_batch=2)
    assert sum(map(int, result.batch_stats["attempted"].to_numpy())) == 2
    assert result.batch_frequencies.schema.field("count").type == pa.int64()
    schema = outcomes.outcome_frequency_schema()
    rows = [
        {
            "root_seed": 11,
            "k": 2,
            "strategy": 1,
            "seat": 0,
            "distribution": "completed_score",
            "value": 10,
            "value_scale": 1,
            "count": n,
        }
        for n in (np.iinfo(np.int64).max, 1)
    ]
    with pytest.raises(OverflowError):
        outcomes._checked_group(pa.Table.from_pylist(rows, schema=schema), schema, ("count",))
    rows[0]["count"], rows[1]["count"] = 2**53 + 1, 2
    exact = outcomes._checked_group(pa.Table.from_pylist(rows, schema=schema), schema, ("count",))
    assert exact["count"].to_pylist() == [2**53 + 3]


def test_all_tied_selfplay_retains_one_win_and_three_zero_margin_losses() -> None:
    result = _accumulate([_row([100] * 4, [10] * 4)])
    bins = _frequency_map(result.frequency_views())
    assert bins[10, 0, "competition_rank", 1] == 4
    assert bins[10, 0, "midrank", 5] == 4
    assert bins[10, 0, "losing_margin", 0] == 3
    assert bins[10, 0, "winning_margin", 0] == 1
    support = next(r for r in result.support_views().to_pylist() if r["seat"] == 0)
    assert support["completed"] == 4 and support["wins"] == 1
    assert support["losses_completed"] == 3 and support["deficit_sum"] == 0


@pytest.mark.parametrize("seat_ranks", [None, [], [None] * 100])
def test_malformed_variable_length_rank_lists_fail_before_accumulation(seat_ranks) -> None:
    row = _row([100, 50, 0, 0], [1, 2, 3, 4])
    row["seat_ranks"] = seat_ranks
    with pytest.raises(ValueError, match="exactly k entries"):
        _accumulate([row])


def _publish(tmp_path, rows, k=4):
    cfg = make_authenticated_v3_config(tmp_path, player_counts=(k,))
    path = cfg.ingested_rows_curated(k)
    publish_v3_parquet(
        cfg,
        path,
        _table(rows, k),
        stage_key="curate",
        producer="curate",
        operation="curate_game_rows",
    )
    return cfg, path


def test_curated_reader_uses_bounded_slices_and_authenticates(tmp_path: Path, monkeypatch) -> None:
    rows = [_row([100, 50, 50, 0], [1, 2, 2, 3], shuffle=i) for i in range(5)]
    cfg, path = _publish(tmp_path, rows)
    before = path.read_bytes()
    monkeypatch.setattr(pq, "read_table", lambda *a, **kw: pytest.fail("whole-file read"))
    slices = list(
        outcomes.iter_curated_outcome_slices(cfg, 4, shuffles_per_batch=2, max_batch_rows=1)
    )
    assert len(slices) == 5
    assert [p.coordinates["deterministic_batch_id"][0].as_py() for p in slices] == [0, 0, 1, 1, 2]
    assert path.read_bytes() == before
    sidecar_path(path).unlink()
    with pytest.raises((ArtifactContractError, FileNotFoundError)):
        list(outcomes.iter_curated_outcome_slices(cfg, 4, shuffles_per_batch=2))


def test_curated_reader_rejects_duplicate_across_io_slices_and_empty_source(tmp_path: Path) -> None:
    row = _row([100, 50, 0, 0], [1, 2, 3, 4])
    cfg, _ = _publish(tmp_path / "duplicate", [row, row])
    with pytest.raises(ValueError, match="across source slices"):
        list(outcomes.iter_curated_outcome_slices(cfg, 4, shuffles_per_batch=2, max_batch_rows=1))
    cfg, path = _publish(tmp_path / "empty", [])
    with pytest.raises(ValueError, match="no observed game support"):
        list(outcomes.iter_curated_outcome_slices(cfg, 4, shuffles_per_batch=2))
    path.unlink()
    with pytest.raises(FileNotFoundError):
        list(outcomes.iter_curated_outcome_slices(cfg, 4, shuffles_per_batch=2))
