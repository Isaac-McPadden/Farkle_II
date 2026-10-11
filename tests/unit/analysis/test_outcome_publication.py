from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from tests.helpers.artifact_sidecars import (
    make_authenticated_v3_config,
    publish_v3_simulation_run,
)
from tests.helpers.outcome_fixture import publish_outcome_source
from tests.unit.analysis.test_outcome_distributions import _row, _table

from farkle.analysis import curate, ingest
from farkle.analysis import outcome_publication as pub
from farkle.analysis.release_audit import audit_sidecar_completeness
from farkle.config import assign_config_sha
from farkle.simulation.simulation import _play_game, simulation_rows_to_table
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils.artifact_contract import sha256_file, sidecar_path, validate_artifact_sidecar
from farkle.utils.authenticated_graph import _capture_graph_inventory
from farkle.utils.partitioned_stage import PartitionedUnit
from farkle.utils.random import RandomPurpose


def _curated(tmp_path):
    cfg = make_authenticated_v3_config(tmp_path)
    cfg.sim.row_dir = Path("rows")
    cfg.ingest.n_jobs = cfg.analysis.n_jobs = 1
    assign_config_sha(cfg)
    strategies = tuple(
        ThresholdStrategy(score_threshold=0, dice_threshold=6, strategy_id=i) for i in (11, 12)
    )
    row = _play_game(
        123,
        list(strategies),
        target_score=200,
        provenance={
            "root_seed": 11,
            "k": 2,
            "shuffle_index": 0,
            "game_index": 0,
            "deterministic_batch_id": 0,
            "shuffle_seed": 456,
            "game_seed": 123,
            "rng_scheme_version": 2,
            "rng_purpose_namespace": int(RandomPurpose.TOURNAMENT_GAME),
        },
    )
    publish_v3_simulation_run(cfg, simulation_rows_to_table([row], 2), strategies=strategies)
    ingest.run(cfg)
    curate.run(cfg)
    return cfg


def test_authenticated_publication_resume_and_missing_unit(tmp_path):
    cfg = _curated(tmp_path)
    outputs = pub.build_outcome_distributions(cfg, 2)
    assert pub.outcome_is_up_to_date(cfg, 2)
    assert audit_sidecar_completeness(cfg.results_root) == []
    for path in outputs:
        validate_artifact_sidecar(path)
    before = {p: (p.stat().st_mtime_ns, sha256_file(p)) for p in outputs}
    pub.build_outcome_distributions(cfg, 2)
    assert before == {p: (p.stat().st_mtime_ns, sha256_file(p)) for p in outputs}
    support = pq.read_table(cfg.outcome_support_path(2)).to_pylist()
    assert len(support) == 6
    assert sum(r["attempted"] for r in support if r["seat"] == 0) == 2
    assert sum(r["support_status"] == "verified_zero" for r in support) == 2
    inventory = pq.read_table(cfg.outcome_batch_inventory_path(2)).to_pylist()
    assert inventory[0]["expected_games"] == inventory[0]["observed_games"] == 1
    assert inventory[0]["expected_support_rows"] == inventory[0]["emitted_support_rows"] == 4
    source = next((cfg.outcome_units_dir(2) / "source").rglob("*.parquet"))
    assert any(
        item.location.relative_path.endswith(source.name)
        for item in _capture_graph_inventory(cfg.results_root)
    )
    normalized = next((cfg.outcome_units_dir(2) / "normalized").rglob("*.parquet"))
    preserved = normalized.stat().st_mtime_ns
    source.unlink()
    assert not pub.outcome_is_up_to_date(cfg, 2)
    assert any(
        "outcome checkpoint" in failure for failure in audit_sidecar_completeness(cfg.results_root)
    )
    pub.build_outcome_distributions(cfg, 2)
    assert pub.outcome_is_up_to_date(cfg, 2)
    assert normalized.stat().st_mtime_ns == preserved
    sidecar_path(source).unlink()
    assert not pub.outcome_is_up_to_date(cfg, 2)
    pub.build_outcome_distributions(cfg, 2)
    assert pub.outcome_is_up_to_date(cfg, 2)
    assert audit_sidecar_completeness(cfg.results_root) == []
    assert all(sha256_file(p) == digest for p, (_, digest) in before.items())
    cfg.outcome_summaries_path(2).write_bytes(b"corrupt")
    assert not pub.outcome_is_up_to_date(cfg, 2)
    pub.build_outcome_distributions(cfg, 2)
    assert pub.outcome_is_up_to_date(cfg, 2)
    assert normalized.stat().st_mtime_ns == preserved


def _slice_fixture(
    tmp_path,
    rows,
    *,
    roster=((10, 1), (20, 1), (30, 1), (40, 1)),
    width=2,
    batches=2,
    rows_per_unit=1,
    row_group_size=3,
):
    source = tmp_path / "curated.parquet"
    pq.write_table(_table(rows), source, row_group_size=row_group_size)
    design = pub._Design(
        11, 4, batches, width, 1, roster, str(source), (), sha256_file(source), "a" * 64
    )
    outputs = []
    for unit in pub._source_units(design, rows_per_unit):
        output = tmp_path / f"slice_{len(outputs)}.parquet"
        pub._SliceWriter(design, 4 * 1024**2)(unit, output)
        outputs.append((unit, output))
    return design, outputs


def _rows():
    return [
        _row([100, 80, 80, 50], [10, 20, 30, 40], shuffle=0),
        _row([100, 100, 30, 20], [10, 20, 30, 40], shuffle=1),
        _row([20, 0, 40, 10], [10, 20, 30, 40], shuffle=2, stopped=True),
        _row([10, 60, 60, 40], [10, 20, 30, 40], shuffle=3),
    ]


def test_certified_batches_retain_zero_loss_and_zero_exposure_vectors(tmp_path):
    d, outputs = _slice_fixture(tmp_path, _rows())
    certificates = []
    for batch in range(2):
        paths = [p for u, p in outputs if u.key[0] == batch]
        merged = tmp_path / f"merged_{batch}.parquet"
        pub._MergeWriter(4 * 1024**2)(
            PartitionedUnit((batch, 0, *map(str, paths)), "unused.parquet"), merged
        )
        certificate = tmp_path / f"certificate_{batch}.parquet"
        pub._CertifyWriter(d, 4 * 1024**2)(
            PartitionedUnit((batch, 0, str(merged)), "unused.parquet"), certificate
        )
        certificates.extend(pq.read_table(certificate).to_pylist())
        footer = pq.ParquetFile(certificate).metadata.metadata
        assert footer is not None
        inventory = json.loads(footer[b"outcome_inventory"])
        assert inventory["observed_games"] == inventory["expected_games"] == 2
        assert inventory["coordinate_digest"] == pub._coordinate_digest(d, batch)
    assert len(certificates) == 2 * 4 * 4
    assert sum(r["support_status"] == "verified_zero" for r in certificates) == 24
    # Reconstruct the selected Task 3 ratio inputs from retained B alone.
    ten = [r for r in certificates if r["strategy"] == 10]
    c = sum(r["completed"] for r in ten)
    losses = sum(r["losses_completed"] for r in ten)
    deficit = sum(r["deficit_sum"] for r in ten)
    assert (c, losses, deficit) == (3, 1, 50)
    assert deficit / c == pytest.approx((losses / c) * (deficit / losses))
    assert sum(r["losses_completed"] for r in ten if r["deterministic_batch_id"] == 0) == 0


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_roster"])
def test_required_support_faults_are_not_zero_filled(tmp_path, fault):
    rows = _rows()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[2] = rows[1]
    else:
        rows[3] = _row([10, 60, 60, 40], [10, 20, 30, 30], shuffle=3)
    with pytest.raises(ValueError, match="coordinate|source|roster|coverage|multiplicity"):
        d, outputs = _slice_fixture(tmp_path, rows)
        merged = tmp_path / "merged.parquet"
        pub._MergeWriter(4 * 1024**2)(
            PartitionedUnit(
                (1, 0, *(str(p) for u, p in outputs if u.key[0] == 1)), "unused.parquet"
            ),
            merged,
        )
        pub._CertifyWriter(d, 4 * 1024**2)(
            PartitionedUnit((1, 0, str(merged)), "unused.parquet"), tmp_path / "certificate.parquet"
        )


def test_slice_boundaries_and_row_groups_preserve_exact_counts(tmp_path):
    logical = []
    for size in (1, 2):
        folder = tmp_path / str(size)
        folder.mkdir()
        _, outputs = _slice_fixture(folder, _rows(), rows_per_unit=size, row_group_size=3)
        logical.append(list(pub._merged([p for _, p in outputs], 4 * 1024**2)))
    assert logical[0] == logical[1]


def test_int64_overflow_fails_explicitly():
    with pytest.raises(OverflowError):
        pub._checked(2**63)


def test_source_footer_is_admitted_before_arrow_decoding(tmp_path):
    cfg = _curated(tmp_path)
    pub._admit_source_footers(cfg)
    cfg.resources.stage_batch_bytes["analysis"] = 1
    with pytest.raises(MemoryError, match="source Parquet footer"):
        pub._admit_source_footers(cfg)


def test_published_frequencies_equal_seat_sum_and_summary_quantiles(tmp_path):
    cfg = _curated(tmp_path)
    pub.build_outcome_distributions(cfg, 2)
    rows = pq.read_table(cfg.outcome_frequencies_path(2)).to_pylist()
    overall, seats = Counter(), Counter()
    for row in rows:
        key = row["strategy"], row["distribution"], row["value"]
        (overall if row["seat"] == 0 else seats)[key] += row["count"]
    assert overall == seats
    summaries = pq.read_table(cfg.outcome_summaries_path(2)).to_pylist()
    for row in summaries:
        if row["denominator"] == 0:
            assert row["estimate"] is None and row["status"] != "ok"
        if row["quantity"] == "midrank" and row["denominator"]:
            assert row["estimate"] in (1, 2)
    assert all(
        pq.read_schema(p).equals(s)
        for p, s in (
            (cfg.outcome_frequencies_path(2), pub.outcome_frequency_schema()),
            (cfg.outcome_batch_stats_path(2), pub.outcome_support_schema(batch=True)),
            (cfg.outcome_batch_inventory_path(2), pub.outcome_inventory_schema()),
            (cfg.outcome_support_path(2), pub.outcome_support_schema()),
            (cfg.outcome_summaries_path(2), pub.outcome_summary_schema()),
        )
    )


def test_hand_calculated_summaries_and_aligned_retention(tmp_path):
    cfg = make_authenticated_v3_config(tmp_path, player_counts=(4,))
    cfg.sim.row_dir = Path("rows")
    cfg.ingest.n_jobs = cfg.analysis.n_jobs = 1
    assign_config_sha(cfg)
    publish_outcome_source(cfg, _rows(), roster=(10, 20, 30, 40), width=2)
    pub.build_outcome_distributions(cfg, 4)
    table = pq.read_table(cfg.outcome_summaries_path(4)).to_pylist()
    keys = [
        (
            r["root_seed"],
            r["k"],
            r["strategy"],
            r["seat"],
            r["quantity"],
            r["statistic"],
            r["conditioning"],
        )
        for r in table
    ]
    assert keys == sorted(set(keys))
    summary = {(r["strategy"], r["seat"], r["quantity"], r["statistic"]): r for r in table}

    def value(strategy, quantity, statistic="estimate"):
        return summary[strategy, 0, quantity, statistic]["estimate"]

    assert value(10, "completed_win_rate") == 2 / 3
    assert value(10, "attempted_loss_rate") == 1 / 2
    assert value(10, "safety_rate") == 1 / 4
    assert value(10, "mean_deficit_completed") == 50 / 3
    assert value(10, "mean_losing_margin") == 50
    assert value(10, "mean_winning_margin") == 10
    assert value(10, "completed_score", "q05") == 10
    assert value(10, "completed_score", "q50") == 100
    assert value(20, "completed_score", "q50") == 80
    assert value(20, "completed_score", "mean") == 80
    assert value(20, "midrank", "q50") == 1.5
    assert value(40, "mean_winning_margin") is None
    assert summary[40, 0, "mean_winning_margin", "estimate"]["status"] == "no_conditional_support"
    assert summary[10, 2, "completed_score", "mean"]["status"] == "verified_zero"
    retained = pq.read_table(cfg.outcome_batch_stats_path(4)).to_pylist()
    for strategy in (10, 20, 30, 40):
        counts = {
            field: sum(r[field] for r in retained if r["strategy"] == strategy)
            for field in pub._COUNTS
        }
        assert (
            value(strategy, "mean_deficit_completed") == counts["deficit_sum"] / counts["completed"]
        )
        if counts["losses_completed"]:
            assert (
                value(strategy, "mean_losing_margin")
                == counts["deficit_sum"] / counts["losses_completed"]
            )


def test_interrupted_parallel_source_units_resume_without_replaying_valid_work(
    tmp_path, monkeypatch
):
    cfg = make_authenticated_v3_config(tmp_path, player_counts=(4,))
    cfg.sim.row_dir = Path("rows")
    cfg.ingest.n_jobs = cfg.analysis.n_jobs = 1
    assign_config_sha(cfg)
    publish_outcome_source(cfg, _rows(), roster=(10, 20, 30, 40), width=2)
    original = pub._SliceWriter.__call__

    def interrupt(self, unit, output):
        if unit.key[0] == 1:
            raise InterruptedError("test interruption after first durable batch")
        return original(self, unit, output)

    monkeypatch.setattr(pub._SliceWriter, "__call__", interrupt)
    with pytest.raises(InterruptedError, match="test interruption"):
        pub.build_outcome_distributions(cfg, 4)
    assert not pub.outcome_is_up_to_date(cfg, 4)
    first = next((cfg.outcome_units_dir(4) / "source").rglob("*.parquet"))
    stamp = first.stat().st_mtime_ns
    monkeypatch.setattr(pub._SliceWriter, "__call__", original)
    cfg.analysis.n_jobs = 2
    pub.build_outcome_distributions(cfg, 4)
    assert first.stat().st_mtime_ns == stamp
    assert pub.outcome_is_up_to_date(cfg, 4)
    tables = [pq.read_table(p) for p in pub.outcome_publication_paths(cfg, 4)[:5]]
    cfg.analysis.n_jobs = 1
    cfg.resources.stage_batch_bytes["analysis"] *= 2
    pub.build_outcome_distributions(cfg, 4, force=True)
    assert all(
        before.equals(pq.read_table(path))
        for before, path in zip(tables, pub.outcome_publication_paths(cfg, 4)[:5], strict=True)
    )
    assert audit_sidecar_completeness(cfg.results_root) == []


def test_all_safety_support_keeps_completed_summaries_undefined(tmp_path):
    cfg = make_authenticated_v3_config(tmp_path, player_counts=(4,))
    cfg.sim.row_dir = Path("rows")
    cfg.ingest.n_jobs = cfg.analysis.n_jobs = 1
    assign_config_sha(cfg)
    rows = [
        _row([10, 20, 30, 40], [10, 20, 30, 40], shuffle=i, stopped=True, width=1) for i in range(2)
    ]
    publish_outcome_source(cfg, rows, roster=(10, 20, 30, 40), width=1)
    pub.build_outcome_distributions(cfg, 4)
    assert set(pq.read_table(cfg.outcome_frequencies_path(4))["distribution"].to_pylist()) == {
        "stopped_score"
    }
    summaries = pq.read_table(cfg.outcome_summaries_path(4)).to_pylist()
    completed = [
        r for r in summaries if r["seat"] == 0 and r["conditioning"].startswith("completed")
    ]
    assert completed and all(
        r["estimate"] is None and r["status"] == "no_completed_support" for r in completed
    )
    rates = [r for r in summaries if r["quantity"] == "attempted_loss_rate" and r["seat"] == 0]
    assert all(r["estimate"] == 1 and r["status"] == "ok" for r in rates)


def test_explicit_roster_multiplicity_certifies_repeated_strategy_exposures(tmp_path):
    rows = [_row([100, 100, 30, 20], [10, 20, 30, 10], shuffle=i) for i in range(2)]
    d, outputs = _slice_fixture(tmp_path, rows, roster=((10, 2), (20, 1), (30, 1)), batches=1)
    merged = tmp_path / "merged.parquet"
    pub._MergeWriter(4 * 1024**2)(
        PartitionedUnit((0, 0, *(str(p) for _, p in outputs)), "unused.parquet"), merged
    )
    result = tmp_path / "certified.parquet"
    pub._CertifyWriter(d, 4 * 1024**2)(
        PartitionedUnit((0, 0, str(merged)), "unused.parquet"), result
    )
    support = [r for r in pq.read_table(result).to_pylist() if r["strategy"] == 10]
    assert sum(r["attempted"] for r in support) == 4
    assert sum(r["wins"] for r in support) == sum(r["losses_completed"] for r in support) == 2
