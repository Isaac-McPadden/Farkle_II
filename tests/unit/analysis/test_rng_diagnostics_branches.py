from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest
from tests.helpers.artifact_sidecars import make_authenticated_v3_config, publish_v3_parquet

from farkle.analysis import combine, rng_diagnostics
from farkle.analysis.stage_registry import resolve_stage_definition
from farkle.config import ArtifactScope, assign_config_sha
from farkle.utils.artifact_contract import sidecar_path
from farkle.utils.authenticated_contract import validate_authenticated_artifact_unbound
from farkle.utils.random import RNG_SCHEME_VERSION, RandomPurpose
from farkle.utils.release_identity import _typed_method
from farkle.utils.schema_helpers import expected_schema_for
from farkle.utils.stage_completion import CompletionState, resolve_stage_state
from farkle.utils.telemetry import SupervisorHeartbeatRecorder, use_supervisor_recorder


def _curated_table(
    *,
    root_seed: int,
    matchups: list[tuple[int, int]],
    order: list[int] | None = None,
) -> pa.Table:
    if order is None:
        order = list(range(len(matchups)))
    p1 = [matchups[index][0] for index in order]
    p2 = [matchups[index][1] for index in order]
    n = len(order)
    winners = [p1[index] if index % 3 else p2[index] for index in range(n)]
    rounds = [2 + (index * 7) % 13 for index in range(n)]
    return pa.Table.from_pylist(
        [
            {
                "root_seed": root_seed,
                "k": 2,
                "shuffle_index": order[index],
                "game_index": 0,
                "rng_scheme_version": RNG_SCHEME_VERSION,
                "rng_purpose_namespace": int(RandomPurpose.TOURNAMENT_GAME),
                "n_rounds": rounds[index],
                "winner_strategy": winners[index],
                "P1_strategy": p1[index],
                "P2_strategy": p2[index],
            }
            for index in range(n)
        ],
        schema=expected_schema_for(2),
    )


def _config_with_input(
    tmp_path: Path,
    *,
    name: str,
    table: pa.Table,
    root_seed: int = 9,
    partitions: int = 2,
    cap: int = 100,
    workers: int = 1,
    lags: tuple[int, ...] = (1, 2),
):
    cfg = make_authenticated_v3_config(tmp_path, name=name, root_seed=root_seed, player_counts=(2,))
    cfg.analysis.rng_diagnostic_partitions = partitions
    cfg.analysis.rng_max_matchup_groups = cap
    cfg.analysis.rng_diagnostic_lags = lags
    cfg.analysis.n_jobs = workers
    assign_config_sha(cfg)
    publish_v3_parquet(
        cfg,
        cfg.ingested_rows_curated(2),
        table,
        stage_key="curate",
        producer="curate",
        operation="curate_game_rows",
        source_scope="by_k",
    )
    combine.run(cfg)
    return cfg


def _read_results(cfg) -> pd.DataFrame:
    return pq.read_table(cfg.rng_output_path("rng_diagnostics.parquet")).to_pandas()


def _sorted_results(cfg) -> pd.DataFrame:
    frame = _read_results(cfg)
    return frame.sort_values(
        ["summary_level", "strategy", "matchup_id", "lag", "metric"],
        kind="mergesort",
        na_position="first",
    ).reset_index(drop=True)


def test_compact_group_records_and_shared_ring_accumulator() -> None:
    table = _curated_table(root_seed=9, matchups=[(1, 2), (1, 3), (1, 2)])
    arrays = rng_diagnostics._extract_batch_arrays(
        table.to_batches()[0],
        winner_col="winner_strategy",
        strat_cols=("P1_strategy", "P2_strategy"),
        expected_root_seed=9,
    )
    counts = rng_diagnostics._count_records(arrays)

    strategy_one = counts[
        (counts["group_type"] == rng_diagnostics._GROUP_STRATEGY) & (counts["group_id"] == 1)
    ]
    matchup_rows = counts[counts["group_type"] == rng_diagnostics._GROUP_MATCHUP]
    assert strategy_one["count"].tolist() == [3]
    assert sorted(matchup_rows["count"].tolist()) == [1, 2]
    assert counts.dtype.hasobject is False

    metric = rng_diagnostics._OnlineMetric((1, 2, 4))
    assert metric.ring.size == 4
    for value in (1.0, 0.0, 1.0, 1.0):
        metric.push(value)
    assert metric.result(0)[0] == pytest.approx(-0.5)
    assert metric.result(2) == (None, "insufficient_pairs")


def test_seed_46_matchup_is_canonicalized_and_repeated_observations_aggregate() -> None:
    matchups = [(13, 31)] * 26 + [(31, 13)] * 28
    table = _curated_table(root_seed=46, matchups=matchups)
    for seat in range(3, 13):
        table = table.append_column(f"P{seat}_strategy", pa.nulls(len(matchups), pa.int32()))
    strat_cols = tuple(f"P{seat}_strategy" for seat in range(1, 13))
    arrays = rng_diagnostics._extract_batch_arrays(
        table.to_batches()[0],
        winner_col="winner_strategy",
        strat_cols=strat_cols,
        expected_root_seed=46,
    )

    counts = rng_diagnostics._count_records(arrays)
    matchup = counts[counts["group_type"] == rng_diagnostics._GROUP_MATCHUP]
    assert matchup.size == 1
    assert int(matchup["group_id"][0]) == 110_276_747_336_793_579
    assert int(matchup["count"][0]) == 54
    assert tuple(int(matchup[f"p{index}"][0]) for index in range(12)) == (
        13,
        31,
        *([-1] * 10),
    )

    observations = rng_diagnostics._observation_records(arrays)
    matchup_observations = observations[
        observations["group_type"] == rng_diagnostics._GROUP_MATCHUP
    ]
    assert set(zip(matchup_observations["p0"], matchup_observations["p1"], strict=True)) == {
        (13, 31)
    }


def test_full_canonical_key_distinguishes_an_injected_digest_collision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    collision_id = np.uint64(0x0123456789ABCDEF)

    def collide(k: np.ndarray, sorted_seats: np.ndarray) -> np.ndarray:
        del sorted_seats
        return np.full(k.size, collision_id, dtype=np.uint64)

    monkeypatch.setattr(rng_diagnostics, "_matchup_ids", collide)
    table = _curated_table(root_seed=9, matchups=[(1, 2)] * 3 + [(1, 3)] * 3)
    arrays = rng_diagnostics._extract_batch_arrays(
        table.to_batches()[0],
        winner_col="winner_strategy",
        strat_cols=("P1_strategy", "P2_strategy"),
        expected_root_seed=9,
    )
    counts = rng_diagnostics._count_records(arrays)
    matchups = counts[counts["group_type"] == rng_diagnostics._GROUP_MATCHUP]
    assert matchups.size == 2
    assert {int(value) for value in matchups["group_id"]} == {int(collision_id)}

    merged = tmp_path / "collision.bin"
    output = tmp_path / "eligibility.parquet"
    counts.tofile(merged)
    rng_diagnostics._write_eligibility_partition(
        merged,
        output,
        dtype=counts.dtype,
        partition=0,
        minimum_observations=3,
        root_seed=9,
        batch_rows=16,
    )
    eligible = pq.read_table(output).to_pandas()
    eligible = eligible.loc[eligible["group_type"] == rng_diagnostics._GROUP_MATCHUP]
    assert {tuple(row) for row in eligible[["p0", "p1"]].to_numpy()} == {(1, 2), (1, 3)}

    membership_dtype = rng_diagnostics._selection_key_dtype(2)
    membership = np.empty(1, dtype=membership_dtype)
    selected = matchups[matchups["p1"] == 2][0]
    for name in membership_dtype.names or ():
        membership[name] = selected[name]
    membership.sort(order=list(membership_dtype.names or ()), kind="stable")
    observations = rng_diagnostics._observation_records(arrays)
    matchup_observations = observations[
        observations["group_type"] == rng_diagnostics._GROUP_MATCHUP
    ]
    kept = matchup_observations[rng_diagnostics._membership_mask(matchup_observations, membership)]
    assert kept.size == 3
    assert set(zip(kept["p0"], kept["p1"], strict=True)) == {(1, 2)}


def test_eligibility_error_preserves_primary_exception_and_removes_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    route_root = tmp_path / "route"
    units = route_root / "units"
    units.mkdir(parents=True)
    dtype = rng_diagnostics._count_dtype(2)
    record = np.zeros(1, dtype=dtype)
    record["group_type"] = rng_diagnostics._GROUP_MATCHUP
    record["k"] = 2
    record["p0"] = 31
    record["p1"] = 13
    record["group_id"] = rng_diagnostics._matchup_ids(
        np.array([2], dtype=np.int16), np.array([[13, 31]], dtype=np.int32)
    )
    record["count"] = 1
    schema = rng_diagnostics._count_arrow_schema(2)
    route_path = units / "row-group-00000.arrow"
    with route_path.open("wb") as handle, ipc.new_file(handle, schema) as writer:
        writer.write_batch(rng_diagnostics._count_records_to_batch(record, schema))

    created: list[Path] = []
    real_temporary_directory = TemporaryDirectory

    class TrackingTemporaryDirectory(TemporaryDirectory[str]):
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            super().__init__(*args, **kwargs)
            created.append(Path(self.name))

    monkeypatch.setattr(rng_diagnostics, "TemporaryDirectory", TrackingTemporaryDirectory)
    writer = rng_diagnostics._EligibilityWriter(str(route_root), 1, 1, 2, 3, 9, 16)
    failed_output = tmp_path / "failed-output.parquet"
    with pytest.raises(ValueError, match="not canonically ordered"):
        writer(rng_diagnostics.PartitionedUnit((0,), "part-000.parquet"), failed_output)
    failed_output.unlink()
    assert created and all(not path.exists() for path in created)

    record["p0"] = 13
    record["p1"] = 31
    with route_path.open("wb") as handle, ipc.new_file(handle, schema) as route_writer:
        route_writer.write_batch(rng_diagnostics._count_records_to_batch(record, schema))
    success_output = tmp_path / "success-output.parquet"
    writer(rng_diagnostics.PartitionedUnit((0,), "part-000.parquet"), success_output)
    assert pq.read_metadata(success_output).num_rows == 1
    success_output.unlink()
    assert all(not path.exists() for path in created)

    class CleanupFailsAfterRemoval(real_temporary_directory[str]):
        def cleanup(self) -> None:
            super().cleanup()
            raise PermissionError("synthetic cleanup failure")

    monkeypatch.setattr(rng_diagnostics, "TemporaryDirectory", CleanupFailsAfterRemoval)
    with (
        pytest.raises(ValueError, match="primary eligibility error"),
        rng_diagnostics._temporary_workspace(prefix="rng-primary-error-"),
    ):
        raise ValueError("primary eligibility error")


def test_small_fixture_matches_legacy_statistics_where_semantics_are_retained(
    tmp_path: Path,
) -> None:
    matchups = [(1, 2)] * 8
    table = _curated_table(root_seed=9, matchups=matchups, order=[7, 1, 4, 0, 6, 2, 5, 3])
    cfg = _config_with_input(tmp_path, name="equivalence", table=table)

    rng_diagnostics.run(cfg)
    actual = _read_results(cfg)
    ordered = table.to_pandas().sort_values(
        ["root_seed", "k", "shuffle_index", "game_index"], kind="mergesort"
    )
    for strategy in (1, 2):
        values = (ordered["winner_strategy"] == strategy).astype(float)
        rounds = ordered["n_rounds"].astype(float)
        for lag in (1, 2):
            win_row = actual.loc[
                (actual["summary_level"] == "strategy")
                & (actual["strategy"] == strategy)
                & (actual["metric"] == "win_indicator")
                & (actual["lag"] == lag)
            ].iloc[0]
            round_row = actual.loc[
                (actual["summary_level"] == "strategy")
                & (actual["strategy"] == strategy)
                & (actual["metric"] == "n_rounds")
                & (actual["lag"] == lag)
            ].iloc[0]
            matchup_row = actual.loc[
                (actual["summary_level"] == "matchup")
                & (actual["metric"] == "n_rounds")
                & (actual["lag"] == lag)
            ].iloc[0]
            assert win_row["autocorr"] == pytest.approx(values.autocorr(lag=lag), abs=1e-15)
            assert round_row["autocorr"] == pytest.approx(rounds.autocorr(lag=lag), abs=1e-15)
            assert matchup_row["autocorr"] == pytest.approx(rounds.autocorr(lag=lag), abs=1e-15)
    matchup = actual.loc[actual["summary_level"] == "matchup"]
    assert matchup["strategy"].isna().all()
    assert set(matchup["metric"]) == {"n_rounds"}
    assert len(matchup) == 2
    sidecar = json.loads(
        sidecar_path(cfg.rng_output_path("rng_diagnostics.parquet")).read_text(encoding="utf-8")
    )
    assert "participant_strategy_ids" in sidecar["method_contract"]["grouping_keys"]


def test_worker_and_partition_count_do_not_change_logical_results(tmp_path: Path) -> None:
    table = _curated_table(root_seed=9, matchups=[(1, 2)] * 9 + [(1, 3)] * 9)
    serial = _config_with_input(tmp_path, name="serial", table=table, partitions=4, workers=1)
    parallel = _config_with_input(tmp_path, name="parallel", table=table, partitions=4, workers=2)

    rng_diagnostics.run(serial)
    recorder = SupervisorHeartbeatRecorder(
        logging.getLogger("tests.rng.specialized_progress"),
        run="root_9",
        interval_seconds=45.0,
    )
    scope = recorder.begin_scope(
        "rng_scope",
        run="root_9",
        stage="rng_diagnostics",
        phase="action",
    )
    with use_supervisor_recorder(recorder, scope):
        rng_diagnostics.run(parallel)

    pd.testing.assert_frame_equal(
        _sorted_results(serial),
        _sorted_results(parallel),
        check_exact=False,
        rtol=1e-15,
        atol=1e-15,
    )
    assert (
        serial.rng_output_path("rng_diagnostics.parquet").read_bytes()
        == parallel.rng_output_path("rng_diagnostics.parquet").read_bytes()
    )
    assert (
        serial.rng_output_path("rng_group_selection.parquet").read_bytes()
        == parallel.rng_output_path("rng_group_selection.parquet").read_bytes()
    )
    completed = cast(dict[str, dict[str, object]], recorder.summary()["completed_progress"])
    rng_summary = completed["rng_scope:rng_diagnostics"]
    assert rng_summary["row_groups"] == 1
    assert rng_summary["partitions"] == 4
    assert rng_summary["reconciled_from"] == (
        "authenticated_rng_outputs_and_partition_manifests"
    )
    scope.finish(status="success")
    recorder.close()


def test_merge_memmaps_are_closed_after_success_and_failure(tmp_path: Path) -> None:
    count_dtype = rng_diagnostics._count_dtype(2)
    count_input = tmp_path / "count-input.bin"
    count_output = tmp_path / "count-output.bin"
    counts = np.zeros(2, dtype=count_dtype)
    counts["group_type"] = rng_diagnostics._GROUP_STRATEGY
    counts["k"] = 2
    counts["group_id"] = [1, 2]
    counts["p0"] = -1
    counts["p1"] = -1
    counts["count"] = 1
    counts.tofile(count_input)
    rng_diagnostics._merge_count_files([count_input], count_output, count_dtype)
    count_input.unlink()
    count_output.unlink()

    observation_dtype = rng_diagnostics._observation_dtype(2)
    observation_input_a = tmp_path / "observation-a.bin"
    observation_input_b = tmp_path / "observation-b.bin"
    observation_output = tmp_path / "observation-output.bin"
    observation = np.zeros(1, dtype=observation_dtype)
    observation["group_type"] = rng_diagnostics._GROUP_STRATEGY
    observation["k"] = 2
    observation["group_id"] = 1
    observation["p0"] = -1
    observation["p1"] = -1
    observation.tofile(observation_input_a)
    observation.tofile(observation_input_b)
    with pytest.raises(ValueError, match="duplicate RNG diagnostic semantic observation"):
        rng_diagnostics._merge_observation_files(
            [observation_input_a, observation_input_b], observation_output, observation_dtype
        )
    observation_input_a.unlink()
    observation_input_b.unlink()
    observation_output.unlink()


def test_deterministic_cap_is_encounter_order_invariant_and_blocks_completion(
    tmp_path: Path,
) -> None:
    matchups = [pair for opponent in range(2, 14) for pair in [(1, opponent)] * 3]
    forward_table = _curated_table(root_seed=9, matchups=matchups)
    reverse_order = list(reversed(range(len(matchups))))
    reverse_table = _curated_table(root_seed=9, matchups=matchups, order=reverse_order)
    forward = _config_with_input(
        tmp_path, name="cap_forward", table=forward_table, partitions=4, cap=5
    )
    reverse = _config_with_input(
        tmp_path, name="cap_reverse", table=reverse_table, partitions=4, cap=5
    )

    rng_diagnostics.run(forward)
    rng_diagnostics.run(reverse)

    selected_forward = pq.read_table(
        forward.rng_output_path("rng_group_selection.parquet"),
        columns=["group_type", "k", "group_id"],
    ).to_pandas()
    selected_reverse = pq.read_table(
        reverse.rng_output_path("rng_group_selection.parquet"),
        columns=["group_type", "k", "group_id"],
    ).to_pandas()
    selected_forward = selected_forward.loc[selected_forward["group_type"] == 1]
    selected_reverse = selected_reverse.loc[selected_reverse["group_type"] == 1]
    assert sorted(selected_forward["group_id"].tolist()) == sorted(
        selected_reverse["group_id"].tolist()
    )
    summary = json.loads(
        forward.rng_output_path("rng_diagnostics_summary.json").read_text(encoding="utf-8")
    )
    assert summary["deterministically_capped_group_count"] == 7
    assert summary["completeness_status"] == "blocked_by_cap"
    assert (
        resolve_stage_state(
            forward.rng_stage_dir / "rng_diagnostics.done.json",
            [forward.combined_manifest_path()],
            [
                forward.rng_output_path("rng_diagnostics.parquet"),
                forward.rng_output_path("rng_diagnostics_summary.json"),
            ],
            cfg=forward,
            stage="rng_diagnostics",
        )
        is CompletionState.BLOCKED_BY_CAP
    )


def test_insufficient_observations_publish_explicit_not_estimable_summary(
    tmp_path: Path,
) -> None:
    table = _curated_table(root_seed=9, matchups=[(1, 2), (1, 3)])
    cfg = _config_with_input(tmp_path, name="not_estimable", table=table, lags=(2,))

    rng_diagnostics.run(cfg)

    assert _read_results(cfg).empty
    summary = json.loads(
        cfg.rng_output_path("rng_diagnostics_summary.json").read_text(encoding="utf-8")
    )
    assert summary["completeness_status"] == "not_estimable"
    assert summary["below_minimum_group_count"] == summary["total_candidate_group_count"]
    assert summary["exclusion_reasons"]["below_minimum_usable_observations"] > 0


def test_corrupt_partition_is_quarantined_and_other_partitions_are_reused(
    tmp_path: Path,
) -> None:
    table = _curated_table(root_seed=9, matchups=[(1, 2)] * 8 + [(1, 3)] * 8)
    cfg = _config_with_input(tmp_path, name="corruption", table=table, partitions=4)
    rng_diagnostics.run(cfg)
    checkpoint_root = next((cfg.rng_stage_dir / "checkpoints").iterdir()) / "04_stats"
    units = sorted((checkpoint_root / "units").glob("part-*.parquet"))
    untouched = {path.name: path.read_bytes() for path in units[1:]}
    units[0].write_bytes(b"corrupt")
    for path in (
        cfg.rng_output_path("rng_diagnostics.parquet"),
        cfg.rng_output_path("rng_diagnostics_summary.json"),
    ):
        path.unlink()
        sidecar_path(path).unlink()
    (cfg.rng_stage_dir / "rng_diagnostics.done.json").unlink()

    rng_diagnostics.run(cfg)

    assert all(path.read_bytes() == untouched[path.name] for path in units[1:])
    assert any((checkpoint_root / "quarantine").iterdir())
    assert validate_authenticated_artifact_unbound(
        cfg.rng_output_path("rng_diagnostics.parquet"), validate_provenance=False
    )


def test_sparse_high_cardinality_stays_bounded_and_resumes_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mostly singleton matchups exercise the phase that previously retained every
    # rejected key and accumulator.  The source is intentionally moderate for CI;
    # its cardinality-to-row ratio is the stress property under test.
    matchups = [(index * 2 + 1, index * 2 + 2) for index in range(12_000)]
    table = _curated_table(root_seed=9, matchups=matchups)
    cfg = _config_with_input(
        tmp_path, name="sparse", table=table, partitions=16, cap=100, lags=(1,)
    )
    original = rng_diagnostics._EligibilityWriter.__call__
    interrupted = {"raised": False}

    def interrupt_once(self, unit, path):
        if unit.key == (3,) and not interrupted["raised"]:
            interrupted["raised"] = True
            raise RuntimeError("synthetic kill")
        return original(self, unit, path)

    monkeypatch.setattr(rng_diagnostics._EligibilityWriter, "__call__", interrupt_once)
    with pytest.raises(RuntimeError, match="synthetic kill"):
        rng_diagnostics.run(cfg)
    checkpoint_root = next((cfg.rng_stage_dir / "checkpoints").iterdir()) / "02_eligibility"
    completed_before_resume = list((checkpoint_root / "units").glob("*.unit.done.json"))
    assert completed_before_resume
    assert not (checkpoint_root / "units" / "part-003.parquet.unit.done.json").exists()
    completed_mtimes = {
        stamp.name.removesuffix(".unit.done.json"): (
            checkpoint_root / "units" / stamp.name.removesuffix(".unit.done.json")
        )
        .stat()
        .st_mtime_ns
        for stamp in completed_before_resume
    }
    assert not (checkpoint_root / "partition_manifest.jsonl").exists()

    monkeypatch.setattr(rng_diagnostics._EligibilityWriter, "__call__", original)
    rng_diagnostics.run(cfg)
    summary = json.loads(
        cfg.rng_output_path("rng_diagnostics_summary.json").read_text(encoding="utf-8")
    )
    assert summary["total_candidate_group_count"] == 36_000
    assert summary["selected_group_count"] == 0
    assert all(
        (checkpoint_root / "units" / name).stat().st_mtime_ns == modified
        for name, modified in completed_mtimes.items()
    )
    assert (
        summary["peak_sampled_process_tree_rss_mb"] < cfg.resources.aggregate_memory_hard_limit_mb
    )
    assert summary["aggregate_memory_hard_limit_mb"] == 2_304
    assert summary["aggregate_memory_hard_limit_mb"] == 2304
    assert (checkpoint_root / "partition_manifest.jsonl").exists()


def test_rng_config_freshness_and_typed_metadata_migrate_to_method_v4() -> None:
    cfg = make_authenticated_v3_config(Path("."), name="typed_rng", root_seed=9)
    baseline = cfg.stage_config_sha("rng_diagnostics")
    cfg.analysis.rng_diagnostic_partitions = 64
    assert cfg.stage_config_sha("rng_diagnostics") != baseline
    assert resolve_stage_definition("rng_diagnostics").cache_key_version == 6

    capacity = rng_diagnostics.RNGDiagnosticCapacityMetadata(
        effective_matchup_group_cap=100,
        normalized_lags=(1, 2),
        partition_count=4,
        minimum_usable_observations=3,
        total_candidate_group_count=12,
        candidate_strategy_group_count=4,
        candidate_matchup_group_count=8,
        eligible_group_count=9,
        eligible_strategy_group_count=4,
        eligible_matchup_group_count=5,
        selected_group_count=9,
        selected_strategy_group_count=4,
        selected_matchup_group_count=5,
        below_minimum_group_count=3,
        deterministically_capped_group_count=0,
        observation_count_distribution=(),
        usable_groups_per_lag=(),
        completeness_status="complete",
    )
    parameters = rng_diagnostics._rng_method_parameters(capacity, strat_cols=("P1_strategy",))
    assert parameters["rng_diagnostic_method_version"] == 4
    assert parameters["tracked_matchup_group_count"] == 5
    assert parameters["skipped_matchup_group_count"] == 3

    # Exercise the release adapter with the migrated generic aliases retained.
    artifact = Path("rng_diagnostics.parquet")
    metadata = rng_diagnostics.make_artifact_sidecar(
        cfg,
        artifact,
        producer="rng_diagnostics",
        scope=ArtifactScope.DIAGNOSTICS,
        source_scope=ArtifactScope.CONCAT_KS,
        operation="semantic_coordinate_lag_correlation",
        method_contract={
            "kind": "diagnostic_band",
            "procedure": "semantic_coordinate_lag_correlation",
            "parameters": parameters,
        },
    )
    typed = _typed_method(cfg, metadata, method_version=4)
    assert typed.rng_diagnostic_lags == (1, 2)
    assert typed.rng_tracked_matchup_group_count == 5
