from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest
from tests.unit.analysis.test_outcome_publication import _curated

from farkle.analysis import seat_behavior as sb
from farkle.analysis.all_player_metrics import build_all_player_batch_metrics
from farkle.analysis.outcome_publication import build_outcome_distributions
from farkle.analysis.release_audit import audit_sidecar_completeness
from farkle.utils.artifact_contract import sha256_file, sidecar_path, validate_artifact_sidecar
from farkle.utils.authenticated_contract import ArtifactMismatchError
from farkle.utils.partitioned_stage import PartitionedUnit


def _game(*, stopped=False):
    row = {
        "root_seed": 11,
        "k": 2,
        "deterministic_batch_id": 0,
        "n_rounds": 4,
        "termination_status": "safety_limit" if stopped else "completed",
    }
    for s, score, turns in ((1, 100, 2), (2, 60, 4)):
        row.update({f"P{s}_strategy": 7, f"P{s}_score": score, f"P{s}_n_turns": turns})
        row.update({f"P{s}_{n}": 2 for n in sb.BEHAVIORS})
    return row


def test_repeated_strategy_is_two_exposures_and_one_game():
    rows = list(sb.behavior_records(_game(), root=11, k=2, batch=0, ordinal=0))
    assert [r["strategy"] for r in rows] == [7, 7, -1]
    assert rows[2]["score_sum"] == 2
    assert rows[2]["turns_sum"] == 6
    total = sb._zero()
    for row in rows[:2]:
        sb._add(total, row)
    values = {r["metric"]: r for r in sb.summary_rows(11, 2, 7, 0, "completed", total)}
    assert values["points_per_turn"]["estimate"] == pytest.approx(160 / 6)
    assert values["mean_game_points_per_turn"]["estimate"] == 32.5
    assert values["mean_game_points_per_round"]["estimate"] == 20
    assert values["round_proxy_gap"]["estimate"] == 12.5
    assert values["turn_round_mismatch_rate"]["estimate"] == 0.5
    assert values["mean_highest_turn"]["estimate"] == 2
    assert "highest_turn_per_turn" not in values
    assert values["smart_one_dice_per_use"]["estimate"] == 1


@pytest.mark.parametrize(
    "fault", ["null", "positive_zero", "completed_zero", "negative", "zero_round_positive"]
)
def test_invalid_actual_seat_values_fail(fault):
    row = _game(stopped=fault != "completed_zero")
    if fault == "null":
        row["P1_rolls"] = None
    elif fault == "negative":
        row["P1_n_turns"] = -1
    elif fault == "zero_round_positive":
        row["n_rounds"] = 0
    else:
        row["P1_n_turns"] = 0
    with pytest.raises((ValueError, OverflowError)):
        list(sb.behavior_records(row, root=11, k=2, batch=0, ordinal=0))


def test_zero_safety_and_missing_denominators_are_explicit():
    row = _game(stopped=True)
    row.update(P1_score=0, P1_n_turns=0, P1_smart_one_uses=0, P1_n_smart_one_dice=0)
    values = list(sb.behavior_records(row, root=11, k=2, batch=0, ordinal=0))
    assert values[0]["score_per_turn_sum"] == 0
    assert values[0]["zero_turn_exposures"] == 1
    summary = {r["metric"]: r for r in sb.summary_rows(11, 2, 7, 1, "safety_limit", values[0])}
    assert summary["points_per_turn"]["estimate"] is None
    assert summary["mean_game_points_per_turn"]["estimate"] == 0
    assert summary["smart_one_dice_per_use"]["status"] == "no_denominator"
    total = sb._zero()
    total["exposures"] = 2**63 - 1
    with pytest.raises(OverflowError):
        sb._add(total, values[0])


def test_authenticated_behavior_resume_corruption_and_audit(tmp_path):
    cfg = _curated(tmp_path)
    build_outcome_distributions(cfg, 2)
    build_all_player_batch_metrics(cfg, 2)
    paths = sb.build_seat_behavior(cfg, 2)
    sb.build_equal_k_opportunity(cfg)
    assert sb.behavior_is_up_to_date(cfg, 2)
    assert audit_sidecar_completeness(cfg.results_root) == []
    for checkpoint in cfg.seat_behavior_units_dir(2).rglob("*.parquet"):
        assert (
            validate_artifact_sidecar(checkpoint).consistency_columns
            == pq.read_schema(checkpoint).names
        )
    for name in ("plan.json", "publication.json"):
        checkpoint = cfg.seat_behavior_units_dir(2) / name
        assert set(validate_artifact_sidecar(checkpoint).consistency_columns) == set(
            json.loads(checkpoint.read_text())
        )
    before = {p: (sha256_file(p), p.stat().st_mtime_ns) for p in paths}
    sb.build_seat_behavior(cfg, 2)
    assert before == {p: (sha256_file(p), p.stat().st_mtime_ns) for p in paths}
    rows = pq.read_table(cfg.seat_behavior_batch_path(2)).to_pylist()
    assert len(rows) == 8
    assert sum(r["exposures"] for r in rows) == 2
    assert sum(r["exposures"] == 0 for r in rows) == 6
    inventory = json.loads(paths[-1].read_text())
    source_stage = next(s for s in inventory["stages"] if s["name"] == "source")
    source = (
        cfg.seat_behavior_units_dir(2)
        / "source"
        / "units"
        / source_stage["units"][0]["relative_output"]
    )
    source.unlink()
    assert not sb.behavior_is_up_to_date(cfg, 2)
    assert any(
        "seat behavior checkpoint" in x for x in audit_sidecar_completeness(cfg.results_root)
    )
    sb.build_seat_behavior(cfg, 2)
    assert sb.behavior_is_up_to_date(cfg, 2)
    assert {p: sha256_file(p) for p in paths} == {p: v[0] for p, v in before.items()}
    sidecar_path(source).unlink()
    sb.build_seat_behavior(cfg, 2)
    assert audit_sidecar_completeness(cfg.results_root) == []


def test_canonical_source_order_survives_slice_merge(tmp_path):
    from tests.unit.analysis.test_outcome_distributions import _row, _table

    from farkle.analysis.outcome_publication import _Design, _source_units

    rows = []
    for i in range(5):
        row = _row([100, 80, 60, 40], [1, 2, 3, 4], shuffle=i)
        row["n_rounds"] = 7
        for seat in range(1, 5):
            row[f"P{seat}_n_turns"] = seat + i + 1
            row.update({f"P{seat}_{n}": seat for n in sb.BEHAVIORS})
        rows.append(row)
    source = tmp_path / "source.parquet"
    pq.write_table(_table(rows), source, row_group_size=2)
    d = _Design(
        11, 4, 1, 5, 1, ((1, 1), (2, 1), (3, 1), (4, 1)), str(source), (), "a" * 64, "b" * 64
    )
    # Fixture batch coordinates match the fixed batch width, independently of I/O slices.
    table = pq.read_table(source)
    import pyarrow as pa

    table = table.set_column(
        table.schema.get_field_index("deterministic_batch_id"),
        table.schema.field("deterministic_batch_id"),
        pa.array([0] * 5, type=pa.int32()),
    )
    pq.write_table(table, source, row_group_size=2)
    products = []
    for width in (1, 3, 5):
        paths = []
        for i, unit in enumerate(_source_units(d, width)):
            path = tmp_path / f"w{width}-{i}.parquet"
            sb._SourceWriter(d, 4 * 1024**2)(unit, path)
            paths.append(path)
        merged = tmp_path / f"merged{width}.parquet"
        sb._MergeWriter(4 * 1024**2)(
            PartitionedUnit((0, 0, *map(str, paths)), "unused.parquet"), merged
        )
        products.append(pq.read_table(merged).to_pylist())
    assert products[0] == products[1] == products[2]


def test_summary_interruption_reuses_authenticated_slices(tmp_path, monkeypatch):
    cfg = _curated(tmp_path)
    build_outcome_distributions(cfg, 2)
    build_all_player_batch_metrics(cfg, 2)
    original = sb._SummaryWriter.__call__

    def interrupted(*args):
        raise InterruptedError("summary publication interruption")

    monkeypatch.setattr(sb._SummaryWriter, "__call__", interrupted)
    with pytest.raises(InterruptedError):
        sb.build_seat_behavior(cfg, 2)
    assert not sb.behavior_is_up_to_date(cfg, 2)
    source = (
        cfg.seat_behavior_units_dir(2)
        / "source"
        / "units"
        / "batch-00000000"
        / "slice-00000000.parquet"
    )
    before = (sha256_file(source), source.stat().st_mtime_ns)
    monkeypatch.setattr(sb._SummaryWriter, "__call__", original)
    sb.build_seat_behavior(cfg, 2)
    assert (sha256_file(source), source.stat().st_mtime_ns) == before
    assert sb.behavior_is_up_to_date(cfg, 2)


def test_equal_k_zero_completed_support_and_missing_k(tmp_path):
    import pyarrow as pa

    from farkle.config import AppConfig, IOConfig, SimConfig
    from farkle.utils.artifact_contract import make_artifact_sidecar
    from farkle.utils.artifacts import write_parquet_artifact_atomic

    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "opportunity"),
        sim=SimConfig(seed=11, n_players_list=[2, 4]),
    )
    for k, completed, mean in ((2, 1, 2.0), (4, 0, None)):
        path = cfg.turn_opportunity_path(k)
        metadata = make_artifact_sidecar(
            cfg,
            path,
            producer="test",
            scope="by_k",
            source_scope="by_k",
            operation="seat_behavior_v1",
            consistency_columns=sb.OPPORTUNITY_SCHEMA.names,
            player_counts=[k],
            required_player_counts=[k],
            missing_cell_policy="fail",
        )
        rows = [
            {
                "root_seed": 11,
                "k": k,
                "seat": seat,
                "completed_games": completed,
                "turns_sum": k if seat == 0 else 1,
                "mean_turns": None,
                "turn_share": None,
                "mean_turn_range": mean if seat == 0 else None,
                "turn_share_imbalance": None,
                "status": "ok" if completed else "no_completed_support",
            }
            for seat in range(k + 1)
        ]
        write_parquet_artifact_atomic(
            pa.Table.from_pylist(rows, schema=sb.OPPORTUNITY_SCHEMA), path, sidecar=metadata
        )
    output = sb.build_equal_k_opportunity(cfg)
    result = pq.read_table(output).to_pylist()[0]
    assert result["required_ks"] == [2, 4]
    assert result["weights"] == [0.5, 0.5]
    assert result["mean_turn_range_equal_k"] is None
    assert result["status"] == "no_completed_support"
    cfg.turn_opportunity_path(4).unlink()
    with pytest.raises((FileNotFoundError, ValueError, ArtifactMismatchError)):
        sb.build_equal_k_opportunity(cfg)
