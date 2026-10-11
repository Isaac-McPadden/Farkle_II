from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.helpers.artifact_sidecars import make_authenticated_v3_config
from tests.helpers.outcome_fixture import publish_outcome_source
from tests.unit.analysis.test_outcome_distributions import _row

from farkle.analysis import outcome_precision as precision
from farkle.analysis import outcome_publication as publication
from farkle.analysis.outcome_distributions import _COUNTS, outcome_support_schema
from farkle.analysis.release_audit import audit_sidecar_completeness
from farkle.config import assign_config_sha
from farkle.utils.artifact_contract import sha256_file, sidecar_path, validate_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.authenticated_graph import _capture_graph_inventory
from farkle.utils.partitioned_stage import PartitionedUnit
from farkle.utils.random import PURPOSE_NAMESPACES, RandomPurpose, coordinate_rng


def _vectors() -> np.ndarray:
    # Unequal denominators, ten zero-loss batches and thirty positive-loss
    # batches. Paired actual-seat views deliberately have different weights.
    b = np.arange(40)
    c = 3 + b % 7
    losses = np.where(b < 10, 0, 1 + b % 2)
    w = c - losses
    s = b % 3
    seat1 = np.column_stack([c + s, c, s, w, losses, losses * (b + 10), w * (b + 2)])
    seat2 = seat1.copy()
    seat2[:, [0, 1, 3, 6]] *= 2
    seat2[:, 0] = seat2[:, 1] + seat2[:, 2]
    seat2[:, 4] = seat2[:, 1] - seat2[:, 3]
    seat2[:, 5] *= 2
    return np.stack([seat1 + seat2, seat1, seat2])


def _write_vectors(path: Path, vectors: np.ndarray) -> None:
    rows = []
    for seat, values in enumerate(vectors):
        for batch, counts in enumerate(values):
            rows.append(
                {
                    "root_seed": 11,
                    "k": 2,
                    "strategy": 10,
                    "seat": seat,
                    "deterministic_batch_id": batch,
                    **dict(zip(_COUNTS, map(int, counts), strict=True)),
                    "support_status": "observed" if counts[0] else "verified_zero",
                }
            )
    pq.write_table(pa.Table.from_pylist(rows, schema=outcome_support_schema(batch=True)), path)


def _fixture(tmp_path: Path, *, replicates=20, workers=1, scratch=False):
    cfg = make_authenticated_v3_config(tmp_path)
    cfg.sim.row_dir = Path("rows")
    cfg.ingest.n_jobs = 1
    cfg.analysis.n_jobs = workers
    cfg.screening.bootstrap_replicates = replicates
    cfg.resources.stage_batch_bytes["analysis"] = 2 * 1024 * 1024
    if scratch:
        cfg.io.temp_dir = tmp_path / "scratch"
        cfg.io.temp_dir.mkdir()
    assign_config_sha(cfg)
    rows = [_row([1000 + b, 900], [10, 20], shuffle=b, width=1, stopped=b == 2) for b in range(3)]
    publish_outcome_source(cfg, rows, roster=(10, 20), width=1)
    publication.build_outcome_distributions(cfg, 2)
    return cfg


def test_rng_namespace_and_coordinate_isolation():
    assert RandomPurpose.OUTCOME_BOOTSTRAP == 402
    assert len(PURPOSE_NAMESPACES) == len(set(PURPOSE_NAMESPACES.values()))
    expected = coordinate_rng(
        RandomPurpose.OUTCOME_BOOTSTRAP, root_seed=11, k=2, replicate_index=8
    ).integers(0, 40, size=40)
    actual = precision.batch_multiplicities(11, 2, 8, 40)
    np.testing.assert_array_equal(actual, np.bincount(expected, minlength=40))
    assert actual.sum() == 40
    for root, k, replicate in ((12, 2, 8), (11, 3, 8), (11, 2, 9)):
        assert not np.array_equal(actual, precision.batch_multiplicities(root, k, replicate, 40))


def test_shared_draw_ratios_ranges_and_reduction_match_direct_oracle(tmp_path):
    cfg = make_authenticated_v3_config(tmp_path)
    cfg.screening.bootstrap_replicates = 2000
    vectors = _vectors()
    path = tmp_path / "vectors.parquet"
    _write_vectors(path, vectors)
    writer = precision._RangeWriter(cfg, 2, 40, 4 * 1024 * 1024)
    shards = []
    actual = np.empty((3, 4, 2000))
    for start in range(0, 2000, 100):
        shard = tmp_path / f"range-{start}.parquet"
        writer(PartitionedUnit((start, start + 100, 0, str(path)), shard.name), shard)
        shards.append(shard)
        for i, row in enumerate(pq.read_table(shard).to_pylist()):
            seat, metric = divmod(i, 4)
            actual[seat, metric, start : start + 100] = row["estimates"]
    expected = np.empty_like(actual)
    for j in range(2000):
        # Independent implementation expands sampled batch indices, preserving
        # zero-event batches and the same draw across all seat/metric cells.
        draw = coordinate_rng(
            RandomPurpose.OUTCOME_BOOTSTRAP, root_seed=11, k=2, replicate_index=j
        ).integers(0, 40, size=40)
        totals = vectors[:, draw, :].sum(axis=1)
        expected[:, :, j] = totals[:, [3, 5, 5, 6]] / totals[:, [1, 1, 4, 3]]
    np.testing.assert_array_equal(actual, expected)
    assert not np.isclose(expected[0, 2, 0], np.mean(vectors[0, 10:, 5] / vectors[0, 10:, 4]))
    # Splitting execution ranges cannot alter draw ownership.
    short = tmp_path / "short.parquet"
    writer(PartitionedUnit((13, 47, 0, str(path)), short.name), short)
    for i, row in enumerate(pq.read_table(short).to_pylist()):
        np.testing.assert_array_equal(row["estimates"], actual[i // 4, i % 4, 13:47])
    out = tmp_path / "summary.parquet"
    precision._SummaryWriter(cfg, 40, 4 * 1024 * 1024)(
        PartitionedUnit((0, str(path), *map(str, shards)), out.name), out
    )
    rows = {(row["seat"], row["summary_id"]): row for row in pq.read_table(out).to_pylist()}
    for seat in range(3):
        for m, summary in enumerate(precision.SUMMARY_IDS):
            row = rows[seat, summary]
            assert row["status"] == "ok"
            lo, hi = np.quantile(expected[seat, m], [0.025, 0.975], method="linear")
            assert row["ci_low"] == pytest.approx(lo)
            assert row["ci_high"] == pytest.approx(hi)
            assert row["mcse"] == pytest.approx(np.std(expected[seat, m], ddof=1))
            assert row["replicates_valid"] == 2000 and row["replicates_invalid"] == 0
        win, loss = rows[seat, "completed_win_rate"], rows[seat, "completed_loss_rate"]
        assert loss["ci_low"] == 1 - win["ci_high"]
        assert loss["ci_high"] == 1 - win["ci_low"]
        assert loss["mcse"] == win["mcse"] and loss["interval_width"] == win["interval_width"]
        assert rows[seat, "mean_deficit_completed"]["estimate"] == pytest.approx(
            loss["estimate"] * rows[seat, "mean_losing_margin"]["estimate"]
        )
    # Loss interval is explicitly completed-conditioned, with safety excluded.
    total = vectors[0].sum(axis=0)
    assert rows[0, "completed_win_rate"]["estimate"] == total[3] / total[1]
    assert rows[0, "completed_win_rate"]["estimate"] != total[3] / total[0]


@pytest.mark.parametrize(
    "case,status",
    [
        ("zero", "verified_zero"),
        ("safety", "no_completed_support"),
        ("no_losses", "no_conditional_support"),
        ("few_batches", "insufficient_batches"),
        ("few_contributing", "insufficient_batches"),
        ("few_replicates", "insufficient_replicates"),
        ("undefined", "undefined_resamples"),
        ("degenerate", "degenerate_resamples"),
    ],
)
def test_withholding_policy_and_precedence(case, status):
    values = _vectors()[0].copy()
    r, metric = 2000, 2
    samples = np.array([])
    if case == "zero":
        values[:] = 0
    elif case == "safety":
        values[:] = 0
        values[:, 0] = values[:, 2] = 1
    elif case == "no_losses":
        values[:, 3] = values[:, 1]
        values[:, 4:6] = 0
    elif case == "few_batches":
        values = values[:29]
    elif case == "few_contributing":
        values[10, 3] += values[10, 4]
        values[10, 4:6] = 0
    elif case == "few_replicates":
        r = 1999
    elif case == "undefined":
        samples = np.ones(r)
        samples[0] = np.nan
    else:
        samples = np.zeros(r)
    row = precision.precision_row(values, metric, samples, confidence=0.95, replicates=r)
    assert row["status"] == status
    assert all(row[n] is None for n in ("ci_low", "ci_high", "mcse", "interval_width"))
    assert row["replicates_requested"] == r
    if case in ("undefined", "degenerate"):
        assert row["replicates_valid"] + row["replicates_invalid"] == r
        assert row["replicate_min"] == row["replicate_max"] == (1 if case == "undefined" else 0)
    else:
        assert row["replicates_valid"] == row["replicates_invalid"] == 0
    if case in ("zero", "safety", "no_losses"):
        assert row["estimate"] is None


def test_zero_deficit_is_not_undefined_losing_severity():
    values = np.tile([2, 2, 0, 2, 0, 0, 500], (30, 1))
    deficit = precision.precision_row(values, 1, np.zeros(2000), confidence=0.95, replicates=2000)
    severity = precision.precision_row(values, 2, np.array([]), confidence=0.95, replicates=2000)
    assert deficit["estimate"] == 0 and deficit["status"] == "degenerate_resamples"
    assert severity["estimate"] is None and severity["status"] == "no_conditional_support"


def test_missing_replicates_fail_instead_of_partial_interval():
    with pytest.raises(ValueError, match="incomplete"):
        precision.precision_row(_vectors()[0], 0, np.ones(1999), confidence=0.95, replicates=2000)


def test_zero_denominator_draw_is_retained_without_redrawing(tmp_path, monkeypatch):
    cfg = make_authenticated_v3_config(tmp_path)
    cfg.screening.bootstrap_replicates = 2000
    vector_path = tmp_path / "vectors.parquet"
    _write_vectors(vector_path, _vectors())
    draws = []

    def zero_loss_draw(root, k, replicate, batches):
        draws.append((root, k, replicate))
        weights = np.zeros(batches, dtype=np.int64)
        weights[0] = batches  # This batch has no losses, but remains in the draw.
        return weights

    monkeypatch.setattr(precision, "batch_multiplicities", zero_loss_draw)
    output = tmp_path / "range.parquet"
    precision._RangeWriter(cfg, 2, 40, 2 * 1024 * 1024)(
        PartitionedUnit((0, 1, 0, str(vector_path)), output.name), output
    )
    rows = pq.read_table(output).to_pylist()
    assert draws == [(11, 2, 0)]
    assert all(r["estimates"] == [None] for r in rows if r["summary_id"] == "mean_losing_margin")
    assert all(r["estimates"] == [1.0] for r in rows if r["summary_id"] == "completed_win_rate")


def test_authenticated_publication_resume_and_corrupt_range(tmp_path):
    cfg = _fixture(tmp_path)
    sources = {p: sha256_file(p) for p in precision._inputs(cfg, 2)}
    outputs = precision.build_outcome_precision(cfg, 2)
    assert precision.precision_is_up_to_date(cfg, 2)
    assert audit_sidecar_completeness(cfg.results_root) == []
    rows = pq.read_table(cfg.outcome_precision_path(2)).to_pylist()
    assert len(rows) == 2 * 3 * 5
    assert {r["status"] for r in rows} == {
        "insufficient_batches",
        "verified_zero",
        "no_conditional_support",
    }
    assert all(r["ci_low"] is None and r["replicates_valid"] == 0 for r in rows)
    before = {p: (p.stat().st_mtime_ns, sha256_file(p)) for p in outputs}
    precision.build_outcome_precision(cfg, 2)
    assert before == {p: (p.stat().st_mtime_ns, sha256_file(p)) for p in outputs}
    root = cfg.outcome_precision_ranges_dir(2)
    payload = json.loads(cfg.outcome_precision_inventory_path(2).read_text())
    shard = root / "units" / payload["stages"][1]["units"][0]["relative_output"]
    vector = root / "vectors" / "units" / "tile-00000000.parquet"
    preserved = vector.stat().st_mtime_ns
    assert any(
        item.location.relative_path.endswith(shard.name)
        and "_outcome_precision_ranges" in item.location.relative_path
        for item in _capture_graph_inventory(cfg.results_root)
    )
    shard.write_bytes(b"corrupt")
    assert not precision.precision_is_up_to_date(cfg, 2)
    assert any(
        "precision checkpoint" in failure
        for failure in audit_sidecar_completeness(cfg.results_root)
    )
    cfg.analysis.n_jobs = 2  # Actual Windows-spawn repair, same statistical identity.
    precision.build_outcome_precision(cfg, 2)
    assert precision.precision_is_up_to_date(cfg, 2)
    assert vector.stat().st_mtime_ns == preserved
    assert all(sha256_file(p) == digest for p, (_, digest) in before.items())
    sidecar_path(shard).unlink()
    precision.build_outcome_precision(cfg, 2)
    assert precision.precision_is_up_to_date(cfg, 2)
    assert audit_sidecar_completeness(cfg.results_root) == []
    assert sources == {p: sha256_file(p) for p in sources}
    for p in outputs:
        metadata = validate_artifact_sidecar(p)
        if p.suffix == ".json":
            assert set(metadata.consistency_columns) == set(json.loads(p.read_text()))


def test_missing_batch_row_fails_and_never_zero_fills(tmp_path):
    cfg = _fixture(tmp_path)
    path = cfg.outcome_batch_stats_path(2)
    table = pq.read_table(path)
    pq.write_table(table.slice(1), path)
    with pytest.raises(ValueError, match="authenticated complete"):
        precision.build_outcome_precision(cfg, 2)
    assert not cfg.outcome_precision_path(2).exists()


def test_complete_grid_validation_rejects_resigned_missing_duplicate_and_bad_inventory(tmp_path):
    cfg = _fixture(tmp_path)
    design = publication._resolve_design(cfg, 2)
    budget = cfg.resources.stage_batch_bytes["analysis"]
    path = cfg.outcome_batch_stats_path(2)
    original = pq.read_table(path)
    for corrupt in (
        original.slice(1),
        pa.concat_tables([original.slice(0, 1), original.slice(0, original.num_rows - 1)]),
    ):
        write_parquet_artifact_atomic(
            corrupt,
            path,
            sidecar=publication._metadata(cfg, 2, path, "outcome_batch_statistics", []),
        )
        with pytest.raises(ValueError, match="missing|duplicate"):
            precision._validate_support(cfg, design, budget)
    write_parquet_artifact_atomic(
        original, path, sidecar=publication._metadata(cfg, 2, path, "outcome_batch_statistics", [])
    )
    ipath = cfg.outcome_batch_inventory_path(2)
    records = pq.read_table(ipath).to_pylist()
    records[0]["completed_games"] += 1
    write_parquet_artifact_atomic(
        pa.Table.from_pylist(records, schema=publication.outcome_inventory_schema()),
        ipath,
        sidecar=publication._metadata(cfg, 2, ipath, "outcome_batch_inventory", []),
    )
    with pytest.raises(ValueError, match="disagree"):
        precision._validate_support(cfg, design, budget)


def test_interruption_reuses_vectors_and_repartition_preserves_logical_results(
    tmp_path, monkeypatch
):
    cfg = _fixture(tmp_path, scratch=True)
    writer = precision._RangeWriter.__call__

    def interrupt(*args):
        raise InterruptedError("range interrupted")

    monkeypatch.setattr(precision._RangeWriter, "__call__", interrupt)
    with pytest.raises(Exception, match="range interrupted"):
        precision.build_outcome_precision(cfg, 2)
    assert not cfg.outcome_precision_path(2).exists()
    vector = cfg.outcome_precision_ranges_dir(2) / "vectors" / "units" / "tile-00000000.parquet"
    preserved = vector.stat().st_mtime_ns
    monkeypatch.setattr(precision._RangeWriter, "__call__", writer)
    precision.build_outcome_precision(cfg, 2)
    assert vector.stat().st_mtime_ns == preserved
    expected = pq.read_table(cfg.outcome_precision_path(2))
    # Smaller resource allocation can reuse a complete logical publication.
    cfg.resources.stage_batch_bytes["analysis"] = 192 * 1024
    precision.build_outcome_precision(cfg, 2)
    assert vector.stat().st_mtime_ns == preserved
    # Forced repartition changes physical tiles, never the statistical batches.
    precision.build_outcome_precision(cfg, 2, force=True)
    assert (
        json.loads((cfg.outcome_precision_ranges_dir(2) / "plan.json").read_text())["tile_size"]
        == 1
    )
    assert pq.read_table(cfg.outcome_precision_path(2)).equals(expected)
    assert precision.precision_is_up_to_date(cfg, 2)
    assert cfg.io.temp_dir is not None
    assert not list(cfg.io.temp_dir.iterdir())


def test_paths_are_nonmutating_and_schema_is_explicit(tmp_path):
    cfg = make_authenticated_v3_config(tmp_path)
    paths = precision.precision_publication_paths(cfg, 2)
    assert not any(p.parent.exists() for p in paths)
    table = pa.Table.from_pylist([], schema=precision.outcome_precision_schema())
    assert table.schema.field("strategy").type == pa.int32()
    assert table.schema.field("estimate").nullable
    assert not table.schema.field("status").nullable
