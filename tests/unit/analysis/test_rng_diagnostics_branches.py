from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
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
from farkle.utils.stage_completion import CompletionState, resolve_stage_state


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
    return pa.table(
        {
            "root_seed": pa.array([root_seed] * n, type=pa.int64()),
            "k": pa.array([2] * n, type=pa.int16()),
            "shuffle_index": pa.array(order, type=pa.int64()),
            "game_index": pa.array([0] * n, type=pa.int64()),
            "rng_scheme_version": pa.array([RNG_SCHEME_VERSION] * n, type=pa.int16()),
            "rng_purpose_namespace": pa.array(
                [int(RandomPurpose.TOURNAMENT_GAME)] * n, type=pa.int16()
            ),
            "n_rounds": pa.array(rounds, type=pa.int32()),
            "winner_strategy": pa.array(winners, type=pa.int32()),
            "P1_strategy": pa.array(p1, type=pa.int32()),
            "P2_strategy": pa.array(p2, type=pa.int32()),
        }
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


def test_worker_and_partition_count_do_not_change_logical_results(tmp_path: Path) -> None:
    table = _curated_table(root_seed=9, matchups=[(1, 2)] * 9 + [(1, 3)] * 9)
    serial = _config_with_input(tmp_path, name="serial", table=table, partitions=2, workers=1)
    parallel = _config_with_input(tmp_path, name="parallel", table=table, partitions=4, workers=2)

    rng_diagnostics.run(serial)
    rng_diagnostics.run(parallel)

    pd.testing.assert_frame_equal(
        _sorted_results(serial),
        _sorted_results(parallel),
        check_exact=False,
        rtol=1e-15,
        atol=1e-15,
    )


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
    assert not (checkpoint_root / "partition_manifest.jsonl").exists()

    monkeypatch.setattr(rng_diagnostics._EligibilityWriter, "__call__", original)
    rng_diagnostics.run(cfg)
    summary = json.loads(
        cfg.rng_output_path("rng_diagnostics_summary.json").read_text(encoding="utf-8")
    )
    assert summary["total_candidate_group_count"] == 36_000
    assert summary["selected_group_count"] == 0
    assert summary["peak_sampled_process_tree_rss_mb"] < cfg.resources.rss_abort_mb
    assert summary["hard_memory_ceiling_mb"] == 1_024
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
