from __future__ import annotations

import json
import logging
import pickle
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from tests.helpers.artifact_sidecars import (
    make_authenticated_v3_config,
    publish_v3_strategy_manifest,
)

import farkle.simulation.runner as runner
import farkle.utils.authenticated_contract as authenticated_contract
from farkle.analysis.release_audit import audit_sidecar_completeness
from farkle.config import AppConfig, assign_config_sha
from farkle.simulation.strategies import ThresholdStrategy
from farkle.simulation.workload_planner import TournamentWorkloadPlan
from farkle.utils.artifact_contract import (
    sha256_file,
    sidecar_path,
    write_artifact_with_sidecar_atomic,
)
from farkle.utils.authenticated_contract import (
    canonical_json_bytes,
    load_immutable_manifest_sidecar,
)
from farkle.utils.random import RNG_SCHEME_VERSION
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION, TOURNAMENT_METHOD_VERSION
from farkle.utils.streaming_loop import run_streaming_shard
from farkle.utils.telemetry import SupervisorHeartbeatRecorder, use_supervisor_recorder


def _plan(cfg: AppConfig, count: int) -> TournamentWorkloadPlan:
    return TournamentWorkloadPlan(
        root_seed=int(cfg.sim.seed),
        k=2,
        strategy_count=2,
        confidence=0.95,
        resolution_delta=0.9,
        required_shuffles_unrounded=count,
        required_shuffles=count,
        batch_count=count,
        shuffles_per_batch=1,
        batch_construction="equal_contiguous",
        games_per_shuffle=1,
        required_games=count,
        achieved_resolution=0.9,
        shuffle_cap=None,
        cap_exceeded=False,
        achieved_resolution_at_cap=None,
    )


def _prepare(cfg: AppConfig, count: int) -> tuple[Path, Path, Path]:
    cfg.sim.row_dir = Path("rows")
    assign_config_sha(cfg)
    strategy = publish_v3_strategy_manifest(
        cfg,
        (
            ThresholdStrategy(300, 3, strategy_id=1),
            ThresholdStrategy(500, 2, strategy_id=2),
        ),
    )
    workload = cfg.n_dir(2) / "simulation_workload_plan.json"
    runner._write_workload_plan_simulation_output(
        cfg,
        workload,
        _plan(cfg, count),
        n_players=2,
        strategy_manifest=strategy,
    )
    row_dir = cfg.simulation_row_dir(2)
    assert row_dir is not None
    row_dir.mkdir(parents=True, exist_ok=True)
    return strategy, workload, row_dir


def _write_shard(
    cfg: AppConfig,
    *,
    strategy: Path,
    workload: Path,
    row_dir: Path,
    index: int,
) -> Path:
    shard = row_dir / f"rows_{cfg.sim.seed}_2p_{index:012d}.parquet"
    table = pa.table({"value": pa.array([index], type=pa.int64())})
    run_streaming_shard(
        out_path=str(shard),
        manifest_path=str(row_dir / "manifest.jsonl"),
        schema=table.schema,
        batch_iter=(table,),
        sidecar=runner._simulation_output_sidecar(
            cfg,
            shard,
            n_players=2,
            operation="publish_simulation_row_shard",
            sources=[strategy, workload],
        ),
        manifest_extra={
            "path": shard.name,
            "root_seed": cfg.sim.seed,
            "n_players": 2,
            "shuffle_index": index,
            "shuffle_seed": index + 100,
            "deterministic_batch_id": index,
            "rng_scheme_version": RNG_SCHEME_VERSION,
            "rng_purpose_namespace": 101,
            "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
            "tournament_method_version": TOURNAMENT_METHOD_VERSION,
        },
    )
    return shard


def _finish(cfg: AppConfig, strategy: Path, workload: Path, row_dir: Path, count: int) -> Path:
    return runner.write_simulation_done(
        cfg,
        2,
        num_shuffles=count,
        shuffles_per_batch=1,
        n_strategies=2,
        outputs=[strategy, workload, row_dir],
    )


def _publish_many(
    tmp_path: Path,
    *,
    name: str,
    count: int,
    order: list[int] | None = None,
) -> tuple[AppConfig, list[Path], Path, Path]:
    cfg = make_authenticated_v3_config(tmp_path, name=name, root_seed=42)
    strategy, workload, row_dir = _prepare(cfg, count)
    shards = [
        _write_shard(
            cfg,
            strategy=strategy,
            workload=workload,
            row_dir=row_dir,
            index=index,
        )
        for index in (order if order is not None else list(range(count)))
    ]
    completion = _finish(cfg, strategy, workload, row_dir, count)
    return cfg, sorted(shards), row_dir / "manifest.jsonl", completion


def test_large_shard_publication_never_uses_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_authenticated_v3_config(tmp_path, name="bounded", root_seed=42)
    strategy, workload, row_dir = _prepare(cfg, 1)
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        if path.name.startswith("rows_") and path.suffix == ".parquet":
            raise AssertionError("large shard passed through unbounded Path.read_bytes")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    _write_shard(
        cfg,
        strategy=strategy,
        workload=workload,
        row_dir=row_dir,
        index=0,
    )
    _finish(cfg, strategy, workload, row_dir, 1)


def test_many_shard_publication_and_resume_operation_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    count = 12
    cfg = make_authenticated_v3_config(tmp_path, name="counts", root_seed=42)
    strategy, workload, row_dir = _prepare(cfg, count)
    operations = {"data_hashes": 0, "data_writes": 0, "metadata_validations": 0}
    real_hash = authenticated_contract.sha256_file
    real_install = authenticated_contract._install_once
    real_metadata = runner.validate_authenticated_artifact_metadata

    def counted_hash(path: Path | str, **kwargs: Any) -> str:
        candidate = Path(path)
        if candidate.parent == row_dir and candidate.name.startswith("._tmp_"):
            operations["data_hashes"] += 1
        return real_hash(path, **kwargs)

    def counted_install(source: Path | str, destination: Path | str) -> None:
        candidate = Path(destination)
        if candidate.name.startswith("rows_") and candidate.suffix == ".parquet":
            operations["data_writes"] += 1
        real_install(source, destination)

    def counted_metadata(path: Path | str, **kwargs: Any):  # noqa: ANN202
        candidate = Path(path)
        if candidate.name.startswith("rows_") and candidate.suffix == ".parquet":
            operations["metadata_validations"] += 1
        return real_metadata(path, **kwargs)

    monkeypatch.setattr(authenticated_contract, "sha256_file", counted_hash)
    monkeypatch.setattr(authenticated_contract, "_install_once", counted_install)
    monkeypatch.setattr(runner, "validate_authenticated_artifact_metadata", counted_metadata)
    for index in range(count):
        _write_shard(
            cfg,
            strategy=strategy,
            workload=workload,
            row_dir=row_dir,
            index=index,
        )
    _finish(cfg, strategy, workload, row_dir, count)
    publication = dict(operations)
    assert publication == {
        "data_hashes": count,
        "data_writes": count,
        "metadata_validations": 2 * count,
    }

    assert runner.simulation_is_complete(cfg, 2)
    resume_delta = {key: operations[key] - publication[key] for key in operations}
    assert resume_delta == {
        "data_hashes": 0,
        "data_writes": 0,
        "metadata_validations": count,
    }
    # The replaced adapter path performed 2N data writes and 6N publication
    # hashes, then 2N more hashes per routine resume.  The measured producer
    # path is N/N/0 respectively; metadata validation remains linear in N.


def test_manifest_and_completion_identity_are_order_invariant(tmp_path: Path) -> None:
    first = _publish_many(tmp_path, name="workers_1", count=4, order=[0, 1, 2, 3])
    second = _publish_many(tmp_path, name="workers_4", count=4, order=[3, 1, 0, 2])
    _cfg_a, shards_a, manifest_a, completion_a = first
    _cfg_b, shards_b, manifest_b, completion_b = second
    assert [path.read_bytes() for path in shards_a] == [path.read_bytes() for path in shards_b]
    assert [sidecar_path(path).read_bytes() for path in shards_a] == [
        sidecar_path(path).read_bytes() for path in shards_b
    ]
    assert manifest_a.read_bytes() == manifest_b.read_bytes()
    assert sidecar_path(manifest_a).read_bytes() == sidecar_path(manifest_b).read_bytes()
    assert completion_a.read_bytes() == completion_b.read_bytes()


def test_interruption_boundaries_never_create_false_completion(tmp_path: Path) -> None:
    cfg = make_authenticated_v3_config(tmp_path, name="interrupt", root_seed=42)
    strategy, workload, row_dir = _prepare(cfg, 1)
    assert not runner.simulation_is_complete(cfg, 2)

    raw = row_dir / "rows_42_2p_000000000000.parquet"
    with raw.open("wb") as handle:
        handle.write(b"staged-data-without-sidecar")
    (row_dir / "manifest.jsonl").write_text(
        json.dumps({"path": raw.name, "shuffle_index": 0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sidecar|identity|promote"):
        runner.write_simulation_done(
            cfg,
            2,
            num_shuffles=1,
            shuffles_per_batch=1,
            n_strategies=2,
            outputs=[strategy, workload, row_dir],
            allow_unsealed_v3_outputs=True,
            rewritten_outputs=[raw],
        )
    assert not runner.simulation_done_path(cfg, 2).exists()

    raw.unlink()
    (row_dir / "manifest.jsonl").unlink()
    _write_shard(
        cfg,
        strategy=strategy,
        workload=workload,
        row_dir=row_dir,
        index=0,
    )
    with pytest.raises(ValueError, match="entry count mismatch"):
        _finish(cfg, strategy, workload, row_dir, 2)
    assert not runner.simulation_done_path(cfg, 2).exists()
    runner._publish_simulation_outputs_v3(
        cfg,
        n_players=2,
        outputs=[strategy, workload, row_dir],
        done_path=runner.simulation_done_path(cfg, 2),
        allow_unsealed_outputs=False,
        num_shuffles=1,
        shuffles_per_batch=1,
    )
    assert not runner.simulation_is_complete(cfg, 2)
    _finish(cfg, strategy, workload, row_dir, 1)
    assert runner.simulation_is_complete(cfg, 2)


def test_interrupted_atomic_staging_names_are_excluded_without_deletion(
    tmp_path: Path,
) -> None:
    cfg = make_authenticated_v3_config(tmp_path, name="atomic_resume", root_seed=42)
    _strategy, _workload, row_dir = _prepare(cfg, 1)
    staging = [
        row_dir / "._tmp_worker",
        row_dir / "._artifact_v3_checkpoint",
        row_dir / "._sidecar_v3_metadata",
    ]
    for path in staging:
        path.write_bytes(b"incomplete")
    unrelated_hidden = row_dir / ".keep"
    unrelated_hidden.write_bytes(b"preserve")

    expanded = runner._completion_output_files(
        [row_dir],
        runner.simulation_done_path(cfg, 2),
    )
    assert not any(path in expanded for path in staging)
    assert unrelated_hidden in expanded

    assert all(path.read_bytes() == b"incomplete" for path in staging)
    assert unrelated_hidden.read_bytes() == b"preserve"


def test_force_preserves_sealed_shards_and_neighbor_files(tmp_path: Path) -> None:
    cfg, shards, manifest, done = _publish_many(tmp_path, name="sealed_force", count=2)
    sentinel = manifest.parent / "keep.txt"
    sentinel.write_bytes(b"keep")
    fake = manifest.parent / "metrics_backup.parquet"
    fake.write_bytes(b"unrelated")
    saved = {
        p: p.read_bytes()
        for p in [*shards, *map(sidecar_path, shards), manifest, done, sentinel, fake]
    }
    strategies = [
        ThresholdStrategy(300, 3, strategy_id=1),
        ThresholdStrategy(500, 2, strategy_id=2),
    ]
    assert runner.run_single_n(cfg, 2, strategies, force=True) > 0
    assert {p: p.read_bytes() for p in saved} == saved


def test_force_inventory_authenticates_incomplete_sealed_cell(tmp_path: Path) -> None:
    cfg = make_authenticated_v3_config(tmp_path, name="incomplete_force", root_seed=42)
    strategy, workload, row_dir = _prepare(cfg, 2)
    shard = _write_shard(cfg, strategy=strategy, workload=workload, row_dir=row_dir, index=0)
    sentinel = row_dir / "keep.txt"
    sentinel.write_bytes(b"keep")
    assert runner._validate_force_inventory(cfg, _plan(cfg, 2), row_dir, row_dir) == (True, {})
    with shard.open("ab") as file:
        file.write(b"corrupt")
    prior = {path: path.read_bytes() for path in (shard, sidecar_path(shard), workload, sentinel)}
    with pytest.raises(authenticated_contract.ArtifactMismatchError):
        runner._validate_force_inventory(cfg, _plan(cfg, 2), row_dir, row_dir)
    assert {path: path.read_bytes() for path in prior} == prior


@pytest.mark.parametrize("mutation", ["data", "missing_sidecar", "sidecar", "manifest"])
def test_routine_resume_rejects_mutated_publication(tmp_path: Path, mutation: str) -> None:
    cfg, shards, manifest, _completion = _publish_many(tmp_path, name=f"mutate_{mutation}", count=2)
    if mutation == "data":
        with shards[0].open("ab") as handle:
            handle.write(b"changed-length")
    elif mutation == "missing_sidecar":
        sidecar_path(shards[0]).unlink()
    elif mutation == "sidecar":
        with sidecar_path(shards[0]).open("ab") as handle:
            handle.write(b" ")
    else:
        lines = manifest.read_bytes().splitlines(keepends=True)
        manifest.write_bytes(b"".join(lines[:-1]))
    assert not runner.simulation_is_complete(cfg, 2)


def test_stale_code_and_config_identity_are_rejected(tmp_path: Path) -> None:
    cfg, _shards, _manifest, _completion = _publish_many(tmp_path, name="stale", count=1)
    code_identity = cfg._code_identity
    assert code_identity is not None
    cfg._code_identity = replace(code_identity, commit="b" * 40)
    assert not runner.simulation_is_complete(cfg, 2)

    cfg._code_identity = code_identity
    cfg.sim.seed = 43
    assign_config_sha(cfg)
    assert not runner.simulation_is_complete(cfg, 2)


def test_deep_release_audit_detects_same_length_data_corruption(tmp_path: Path) -> None:
    cfg, shards, _manifest, _completion = _publish_many(tmp_path, name="deep", count=2)
    assert audit_sidecar_completeness(cfg.results_root) == []
    with shards[0].open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 0x01]))
    failures = audit_sidecar_completeness(cfg.results_root)
    assert any("incompatible sidecar" in failure for failure in failures)


def test_native_manifest_records_are_canonical_identity_records(tmp_path: Path) -> None:
    _cfg, _shards, manifest, _completion = _publish_many(
        tmp_path, name="manifest", count=3, order=[2, 0, 1]
    )
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [record["shuffle_index"] for record in records] == [0, 1, 2]
    assert all("pid" not in record and "ts" not in record for record in records)
    assert all(
        {
            "data_sha256",
            "byte_length",
            "sidecar_sha256",
            "schema_fingerprint_sha256",
        }.issubset(record)
        for record in records
    )
    assert load_immutable_manifest_sidecar(manifest).manifest_sha256 == sha256_file(manifest)
    assert canonical_json_bytes(records[0]) + b"\n" in manifest.read_bytes()


def test_real_two_worker_tournament_uses_producer_owned_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process_map = runner.tournament_mod.parallel.process_map
    observed: list[tuple[int, int, int]] = []

    def measure(fn, items, **kwargs):
        events: list[Any] = []
        callback = kwargs["progress_callback"]

        def progress(event):
            events.append(event)
            callback(event)

        yield from process_map(fn, items, **{**kwargs, "progress_callback": progress})
        observed.append(
            (
                kwargs["n_jobs"],
                kwargs["window"],
                max(event["pending"] for event in events if event["event"] == "submitted"),
            )
        )

    monkeypatch.setattr(runner.tournament_mod.parallel, "process_map", measure)

    def _run(name: str, workers: int, multiplier: int) -> AppConfig:
        cfg = make_authenticated_v3_config(tmp_path, name=name, root_seed=42)
        cfg.sim.row_dir = Path("rows")
        cfg.sim.metric_chunk_dir = Path("metric_chunks")
        cfg.sim.expanded_metrics = True
        cfg.sim.n_jobs = workers
        cfg.resources.max_in_flight_per_worker = multiplier
        # The preflight subprocess runs this probe with two fully attested workers.
        cfg.resources.process_tree_warning_threshold_mb = 1536
        cfg.screening.resolution_delta = 0.9
        cfg.batching.target_batches = 2
        cfg.batching.min_shuffles_per_batch = 1
        assign_config_sha(cfg)
        games = runner.run_single_n(
            cfg,
            2,
            strategies=[
                ThresholdStrategy(300, 3, strategy_id=1),
                ThresholdStrategy(500, 2, strategy_id=2),
            ],
        )
        assert games > 0
        assert runner.simulation_is_complete(cfg, 2)
        return cfg

    sequential = _run("real_workers_1", 1, 3)
    parallel = _run("real_workers_2", 2, 1)
    assert observed == [(1, 3, 1), (2, 2, 2)]

    def _published_bytes(cfg: AppConfig) -> dict[str, bytes]:
        return {
            path.relative_to(cfg.results_root).as_posix(): path.read_bytes()
            for path in sorted(cfg.results_root.rglob("*"))
            if path.is_file() and not path.name.startswith("._")
        }

    assert _published_bytes(sequential) == _published_bytes(parallel)


@pytest.fixture
def aggregate_run(tmp_path: Path) -> tuple[AppConfig, list[ThresholdStrategy]]:
    cfg = make_authenticated_v3_config(tmp_path, name="aggregate", root_seed=42)
    cfg.sim.row_dir = Path("rows")
    cfg.sim.metric_chunk_dir = Path("chunks")
    cfg.sim.expanded_metrics = True
    cfg.sim.n_jobs = 1
    cfg.sim.ckpt_every_sec = 0
    cfg.screening.resolution_delta = 0.9
    cfg.batching.target_batches = 2
    cfg.batching.min_shuffles_per_batch = 1
    assign_config_sha(cfg)
    return cfg, [
        ThresholdStrategy(300, 3, strategy_id=1),
        ThresholdStrategy(500, 2, strategy_id=2),
    ]


@pytest.mark.parametrize("storage", ["rows_and_chunks", "rows_only", "chunks_only"])
def test_complete_aggregate_restores_without_replaying_or_scheduling(
    aggregate_run: tuple[AppConfig, list[ThresholdStrategy]],
    monkeypatch: pytest.MonkeyPatch,
    storage: str,
    recovery_recorder,
) -> None:
    cfg, strategies = aggregate_run
    if storage == "chunks_only":
        cfg.sim.row_dir = None
    if storage == "rows_only":
        cfg.sim.metric_chunk_dir = None
    assign_config_sha(cfg)
    games = runner.run_single_n(cfg, 2, strategies=strategies)
    done = runner.simulation_done_path(cfg, 2)
    paths = [
        cfg.n_dir(2) / name
        for name in ("2p_checkpoint.pkl", "2p_checkpoint.parquet", "2p_metrics.parquet")
    ]
    original = {path: path.read_bytes() for path in [done, *paths]}
    checkpoint = pickle.loads(original[paths[0]])
    assert checkpoint["meta"]["complete_aggregate"]["version"] == 1
    assert checkpoint["outcome_counts"]["games_attempted"] == games
    done.unlink()

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("complete aggregate attempted a replay or executor submission")

    tournament = runner.tournament_mod
    monkeypatch.setattr(tournament, "_load_row_manifest_aggregates", unexpected)
    monkeypatch.setattr(tournament, "_load_metric_chunk_aggregates", unexpected)
    monkeypatch.setattr(tournament.parallel, "process_map", unexpected)
    with recovery_recorder() as recorder:
        assert runner.run_single_n(cfg, 2, strategies=strategies) == games
    summary = recorder.summary()["completed_progress"]["simulation:simulation:k=2"]
    assert summary["newly_simulated_games"] == 0
    assert summary["reused_attempted_games"] == games
    assert summary["recovery_phases"]["simulation_recovery_authentication"]["files_checked"] > 0
    assert (
        summary["recovery_phases"]["simulation_recovery_restoration"]["aggregate_shuffles_restored"]
        == 2
    )
    assert {path: path.read_bytes() for path in original} == original
    assert runner.simulation_is_complete(cfg, 2)


def test_complete_aggregate_refuses_corrupt_shard_bytes(
    aggregate_run: tuple[AppConfig, list[ThresholdStrategy]],
) -> None:
    cfg, strategies = aggregate_run
    runner.run_single_n(cfg, 2, strategies=strategies)
    done = runner.simulation_done_path(cfg, 2)
    done.unlink()
    row_dir = cfg.simulation_row_dir(2)
    assert row_dir is not None
    shard = next(row_dir.glob("rows_*.parquet"))
    with shard.open("r+b") as handle:
        handle.write(b"FAIL")
    with pytest.raises(ValueError, match="aggregate checkpoint shard authentication failed"):
        runner.run_single_n(cfg, 2, strategies=strategies)
    assert not done.exists()


def test_complete_aggregate_cannot_cover_missing_manifest_member(
    aggregate_run: tuple[AppConfig, list[ThresholdStrategy]], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, strategies = aggregate_run
    runner.run_single_n(cfg, 2, strategies=strategies)
    runner.simulation_done_path(cfg, 2).unlink()
    row_dir = cfg.simulation_row_dir(2)
    assert row_dir is not None
    manifest = row_dir / "manifest.jsonl"
    manifest.write_bytes(manifest.read_bytes().splitlines(keepends=True)[0])

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("damaged sealed inventory was used to schedule a shuffle")

    monkeypatch.setattr(runner.tournament_mod.parallel, "process_map", unexpected)
    with pytest.raises(ValueError, match="sealed simulation manifest bytes differ"):
        runner.run_single_n(cfg, 2, strategies=strategies)


@pytest.mark.parametrize("case", ["missing", "corrupt", "unsealed", "legacy", "incompatible"])
def test_unusable_aggregate_checkpoint_replays_existing_rows(
    aggregate_run: tuple[AppConfig, list[ThresholdStrategy]],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    cfg, strategies = aggregate_run
    games = runner.run_single_n(cfg, 2, strategies=strategies)
    runner.simulation_done_path(cfg, 2).unlink()
    checkpoint = cfg.n_dir(2) / "2p_checkpoint.pkl"
    if case == "missing":
        checkpoint.unlink()
        sidecar_path(checkpoint).unlink()
    elif case == "corrupt":
        # Change contents without changing the byte length or producer sidecar.
        with checkpoint.open("r+b") as handle:
            handle.write(b"broken")
        corrupt = checkpoint.read_bytes()
        loads = pickle.loads

        def authenticated_only(contents: bytes) -> Any:
            assert contents != corrupt
            return loads(contents)

        monkeypatch.setattr(pickle, "loads", authenticated_only)
    elif case == "unsealed":
        sidecar_path(checkpoint).unlink()
    else:
        payload = pickle.loads(checkpoint.read_bytes())
        if case == "legacy":
            del payload["meta"]["complete_aggregate"]
            del payload["meta"]["simulation_stage_config_sha"]
        else:
            payload["meta"]["complete_aggregate"]["reduction"] = "unknown"
        sidecar = runner._simulation_output_sidecar(
            cfg,
            checkpoint,
            n_players=2,
            operation="publish_simulation_checkpoint",
            sources=[
                cfg.strategy_manifest_root_path(),
                cfg.n_dir(2) / "simulation_workload_plan.json",
            ],
        )

        def write(staged: Path) -> None:
            staged.write_bytes(pickle.dumps(payload))

        write_artifact_with_sidecar_atomic(checkpoint, sidecar, write)

    replayed: list[int] = []
    tournament = runner.tournament_mod
    replay = tournament._load_row_manifest_aggregates

    def count_replay(path: Path, **kwargs: Any) -> Any:
        aggregate = replay(path, **kwargs)
        replayed.append(aggregate[0].games_attempted)
        return aggregate

    def unexpected_game(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("durable shuffle was simulated again")

    monkeypatch.setattr(tournament, "_load_row_manifest_aggregates", count_replay)
    monkeypatch.setattr(tournament, "_play_one_shuffle", unexpected_game)
    assert runner.run_single_n(cfg, 2, strategies=strategies) == games
    assert replayed == [games]
    assert "complete_aggregate" not in pickle.loads(checkpoint.read_bytes())["meta"]


def test_periodic_checkpoint_contains_its_independently_reduced_coverage(
    aggregate_run: tuple[AppConfig, list[ThresholdStrategy]], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, strategies = aggregate_run
    tournament = runner.tournament_mod

    def interrupt(*_args: Any, **_kwargs: Any) -> Any:
        raise InterruptedError("before final reduction")

    with monkeypatch.context() as patch:
        patch.setattr(tournament, "_load_metric_chunk_aggregates", interrupt)
        with pytest.raises(InterruptedError, match="before final reduction"):
            runner.run_single_n(cfg, 2, strategies=strategies)
    checkpoint = cfg.n_dir(2) / "2p_checkpoint.pkl"
    payload = pickle.loads(checkpoint.read_bytes())
    assert payload["meta"]["completed_shuffle_indices"] == list(
        range(payload["meta"]["num_shuffles"])
    )
    assert "complete_aggregate" not in payload["meta"]
    assert (
        payload["meta"]["partial_aggregate"]["shuffle_indices"]
        == payload["meta"]["completed_shuffle_indices"]
    )
    assert payload["metric_sums"]["winning_score"]
    assert not runner.simulation_done_path(cfg, 2).exists()
    assert (
        runner.run_single_n(cfg, 2, strategies=strategies)
        == payload["outcome_counts"]["games_attempted"]
    )
    assert "complete_aggregate" in pickle.loads(checkpoint.read_bytes())["meta"]


def test_interrupted_final_checkpoint_restores_prior_sealed_checkpoint(
    aggregate_run: tuple[AppConfig, list[ThresholdStrategy]],
    monkeypatch: pytest.MonkeyPatch,
    recovery_recorder,
) -> None:
    cfg, strategies = aggregate_run
    tournament = runner.tournament_mod
    checkpoint = cfg.n_dir(2) / "2p_checkpoint.pkl"
    replace_file = authenticated_contract.replace_file_atomic
    prior = {}

    def interrupt(source: Path | str, destination: Path | str) -> None:
        if Path(source) == sidecar_path(checkpoint):
            prior.update({p: p.read_bytes() for p in (checkpoint, sidecar_path(checkpoint))})
        if (
            Path(destination) == sidecar_path(checkpoint)
            and b"complete_aggregate" in checkpoint.read_bytes()
        ):
            raise InterruptedError("before sealing final checkpoint")
        replace_file(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(authenticated_contract, "replace_file_atomic", interrupt)
        with pytest.raises(InterruptedError, match="before sealing final checkpoint"):
            runner.run_single_n(cfg, 2, strategies=strategies)
    assert prior == {p: p.read_bytes() for p in (checkpoint, sidecar_path(checkpoint))}
    assert "partial_aggregate" in pickle.loads(checkpoint.read_bytes())["meta"]
    assert not runner.simulation_done_path(cfg, 2).exists()

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("a durable game was simulated again")

    monkeypatch.setattr(tournament, "_play_one_shuffle", unexpected)
    with recovery_recorder() as recorder:
        runner.run_single_n(cfg, 2, strategies=strategies)
    summary = recorder.summary()["completed_progress"]["simulation:simulation:k=2"]
    assert summary["reused_attempted_games"] == 2
    assert summary["newly_simulated_games"] == 0
    assert summary["recovery_phases"]["simulation_recovery_reconstruction"]["rows_replayed"] == 0
    assert "complete_aggregate" in pickle.loads(checkpoint.read_bytes())["meta"]


@pytest.mark.parametrize(
    ("storage", "incompatible"),
    [("rows", False), ("chunks", False), ("both", False), ("rows", True)],
)
def test_partial_aggregate_reduces_only_the_durable_remainder(
    aggregate_run: tuple[AppConfig, list[ThresholdStrategy]],
    monkeypatch: pytest.MonkeyPatch,
    storage: str,
    incompatible: bool,
    recovery_recorder,
) -> None:
    cfg, strategies = aggregate_run
    if storage == "rows":
        cfg.sim.metric_chunk_dir = None
    if storage == "chunks":
        cfg.sim.row_dir = None
    cfg.batching.target_batches = 4
    assign_config_sha(cfg)
    tournament = runner.tournament_mod
    checkpoint = cfg.n_dir(2) / "2p_checkpoint.pkl"
    saved = {}
    save = tournament._save_checkpoint

    def capture(*args: Any, **kwargs: Any) -> None:
        save(*args, **kwargs)
        meta = kwargs["meta"]
        if meta.get("completed_shuffle_indices") == [0, 1]:
            assert meta["partial_aggregate"]["shuffle_indices"] == [0, 1]
            saved.update(
                {path: path.read_bytes() for path in (checkpoint, sidecar_path(checkpoint))}
            )

    with monkeypatch.context() as patch:
        patch.setattr(tournament, "_save_checkpoint", capture)
        runner.run_single_n(cfg, 2, strategies=strategies)
    original = pickle.loads(checkpoint.read_bytes())
    summaries = {path: path.read_bytes() for path in cfg.n_dir(2).glob("*.parquet")}
    for path, contents in saved.items():
        path.write_bytes(contents)
    if incompatible:
        payload = pickle.loads(checkpoint.read_bytes())
        payload["meta"]["partial_aggregate"]["version"] = 999
        sidecar = runner._simulation_output_sidecar(
            cfg,
            checkpoint,
            n_players=2,
            operation="publish_simulation_checkpoint",
            sources=[
                cfg.strategy_manifest_root_path(),
                cfg.n_dir(2) / "simulation_workload_plan.json",
            ],
        )

        def write(staged: Path) -> None:
            staged.write_bytes(pickle.dumps(payload))

        write_artifact_with_sidecar_atomic(checkpoint, sidecar, write)
    runner.simulation_done_path(cfg, 2).unlink()
    reducer_name = (
        "_load_metric_chunk_aggregates" if storage == "chunks" else "_load_row_manifest_aggregates"
    )
    replay = getattr(tournament, reducer_name)
    observed: list[int] = []

    def observe(path: Path, *, records: list[dict[str, Any]] | None = None, **kwargs: Any) -> Any:
        assert (records is None) == incompatible
        if records is None:
            records = [json.loads(line) for line in path.read_text().splitlines()]
        index = "process_block_index" if storage == "chunks" else "shuffle_index"
        observed.extend(item[index] for item in records)
        return replay(path, records=records, **kwargs)

    def no_game(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("durable shuffle was simulated again")

    monkeypatch.setattr(tournament, reducer_name, observe)
    monkeypatch.setattr(tournament, "_play_one_shuffle", no_game)
    cfg.sim.n_jobs = 2
    with recovery_recorder() as recorder:
        runner.run_single_n(cfg, 2, strategies=strategies)
    summary = recorder.summary()["completed_progress"]["simulation:simulation:k=2"]
    assert summary["reused_attempted_games"] == 4
    assert summary["newly_simulated_games"] == 0
    replay_counts = summary["recovery_phases"]["simulation_recovery_reconstruction"]
    assert replay_counts["files_checked"] == len(observed)
    assert replay_counts["rows_replayed"] >= len(observed)
    assert observed == ([0, 1, 2, 3] if incompatible else [3, 4] if storage == "chunks" else [2, 3])
    restored = pickle.loads(checkpoint.read_bytes())
    for key in ("win_totals", "outcome_counts", "metric_sums", "metric_square_sums"):
        assert restored[key] == original[key]
    assert {path: path.read_bytes() for path in summaries} == summaries
    assert ("complete_aggregate" in restored["meta"]) is (not incompatible)


@pytest.fixture
def recovery_recorder(monkeypatch: pytest.MonkeyPatch):
    from farkle.utils import scientific_runtime

    monkeypatch.setattr(
        scientific_runtime,
        "numerical_environment",
        lambda names, **_: ("1" * 64, dict.fromkeys(names, "2" * 64)),
    )

    @contextmanager
    def record():
        recorder = SupervisorHeartbeatRecorder(
            logging.getLogger("tests.recovery"), run="test", interval_seconds=0
        )
        scope = recorder.begin_scope("simulation", run="test", stage="simulation", phase="action")
        with use_supervisor_recorder(recorder, scope):
            yield recorder
        scope.finish(status="success")
        recorder.close()

    return record


@pytest.mark.parametrize("lost_shard", [False, True])
def test_checkpoint_ahead_of_append_manifest(
    aggregate_run: tuple[AppConfig, list[ThresholdStrategy]],
    monkeypatch: pytest.MonkeyPatch,
    lost_shard: bool,
) -> None:
    cfg, strategies = aggregate_run
    cfg.sim.metric_chunk_dir = None
    cfg.batching.target_batches = 3
    assign_config_sha(cfg)
    tournament = runner.tournament_mod
    save = tournament._save_checkpoint
    play = tournament._play_one_shuffle

    def stop(*args: Any, **kwargs: Any) -> None:
        save(*args, **kwargs)
        if kwargs["meta"]["partial_aggregate"]["shuffle_indices"] == [0, 1]:
            raise InterruptedError("after incorporating two shuffles")

    with monkeypatch.context() as patch:
        patch.setattr(tournament, "_save_checkpoint", stop)
        with pytest.raises(InterruptedError):
            runner.run_single_n(cfg, 2, strategies=strategies)
    row_dir = cfg.simulation_row_dir(2)
    assert row_dir is not None
    (row_dir / "manifest.jsonl").unlink()
    first = min(row_dir.glob("rows_*.parquet"))
    if lost_shard:
        first.unlink()
        sidecar_path(first).unlink()
    scheduled = []

    def observe(task: Any, **kwargs: Any) -> Any:
        scheduled.append(task.shuffle_index)
        return play(task, **kwargs)

    monkeypatch.setattr(tournament, "_play_one_shuffle", observe)
    runner.run_single_n(cfg, 2, strategies=strategies)
    assert scheduled == ([0, 2] if lost_shard else [2])
    assert runner.simulation_is_complete(cfg, 2)


def test_partial_row_block_rebuilds_its_metric_shard(
    aggregate_run: tuple[AppConfig, list[ThresholdStrategy]],
    monkeypatch: pytest.MonkeyPatch,
    recovery_recorder,
) -> None:
    cfg, strategies = aggregate_run
    cfg.batching.min_shuffles_per_batch = 2
    assign_config_sha(cfg)
    play = runner.tournament_mod._play_one_shuffle

    def interrupt(task: Any, **kwargs: Any) -> Any:
        if task.shuffle_index == 3:
            raise InterruptedError("after the first row of a metric block")
        return play(task, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(runner.tournament_mod, "_play_one_shuffle", interrupt)
        with pytest.raises(InterruptedError):
            runner.run_single_n(cfg, 2, strategies=strategies)
    scheduled = []

    def observe(task: Any, **kwargs: Any) -> Any:
        scheduled.append(task.shuffle_index)
        return play(task, **kwargs)

    monkeypatch.setattr(runner.tournament_mod, "_play_one_shuffle", observe)
    with recovery_recorder() as recorder:
        runner.run_single_n(cfg, 2, strategies=strategies)
    assert scheduled == [3]
    summary = recorder.summary()["completed_progress"]["simulation:simulation:k=2"]
    assert summary["reused_attempted_games"] == 3
    assert summary["newly_simulated_games"] == 1
    metric_dir = runner._resolve_metric_chunk_dir(cfg, 2)
    assert metric_dir is not None
    records = [
        json.loads(line)
        for line in (metric_dir / "metrics_manifest.jsonl").read_text().splitlines()
    ]
    assert [record["shuffle_indices"] for record in records] == [[0, 1], [2, 3]]
    assert runner.simulation_is_complete(cfg, 2)


def test_out_of_order_partial_coverage_resumes_with_different_resources(
    aggregate_run: tuple[AppConfig, list[ThresholdStrategy]], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, strategies = aggregate_run
    cfg.sim.metric_chunk_dir = None
    cfg.batching.target_batches = 4
    cfg.resources.logical_cpu_budget = 1
    cfg.resources.scheduler_memory_budget_mb = 512
    assign_config_sha(cfg)
    tournament = runner.tournament_mod
    process_map = tournament.parallel.process_map
    save = tournament._save_checkpoint

    def reversed_blocks(function: Any, items: Any, **kwargs: Any) -> Any:
        return process_map(function, reversed(list(items)), **kwargs)

    def stop(*args: Any, **kwargs: Any) -> None:
        save(*args, **kwargs)
        assert kwargs["meta"]["partial_aggregate"]["shuffle_indices"] == [3]
        raise InterruptedError("after an out-of-order block")

    with monkeypatch.context() as patch:
        patch.setattr(tournament.parallel, "process_map", reversed_blocks)
        patch.setattr(tournament, "_save_checkpoint", stop)
        with pytest.raises(InterruptedError):
            runner.run_single_n(cfg, 2, strategies=strategies)

    submitted = []
    workers = []

    def observe(function: Any, items: Any, **kwargs: Any) -> Any:
        pending = list(items)
        submitted.extend(task.shuffle_index for _, tasks in pending for task in tasks)
        workers.append(kwargs["n_jobs"])
        return process_map(function, pending, **kwargs)

    monkeypatch.setattr(tournament.parallel, "process_map", observe)
    cfg.sim.n_jobs = 2
    cfg.resources.logical_cpu_budget = 2
    cfg.resources.scheduler_memory_budget_mb = 768
    assign_config_sha(cfg)
    runner.run_single_n(cfg, 2, strategies=strategies)
    assert submitted == [0, 1, 2]
    assert workers == [2]
    assert runner.simulation_is_complete(cfg, 2)


@pytest.mark.parametrize("damage", ["duplicate", "overlap", "corrupt"])
def test_partial_recovery_rejects_invalid_durable_evidence(
    aggregate_run: tuple[AppConfig, list[ThresholdStrategy]],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    cfg, strategies = aggregate_run
    tournament = runner.tournament_mod
    save = tournament._save_checkpoint

    def stop(*args: Any, **kwargs: Any) -> None:
        save(*args, **kwargs)
        raise InterruptedError

    with monkeypatch.context() as patch:
        patch.setattr(tournament, "_save_checkpoint", stop)
        with pytest.raises(InterruptedError):
            runner.run_single_n(cfg, 2, strategies=strategies)
    row_dir = cfg.simulation_row_dir(2)
    metric_dir = runner._resolve_metric_chunk_dir(cfg, 2)
    assert row_dir is not None and metric_dir is not None
    manifest = row_dir / "manifest.jsonl"
    if damage == "duplicate":
        with manifest.open("ab") as handle:
            handle.write(manifest.read_bytes())
        message = "Duplicate shard path"
    elif damage == "overlap":
        manifest = metric_dir / "metrics_manifest.jsonl"
        record = json.loads(manifest.read_text())
        record["shuffle_indices"] *= 2
        record["shuffle_seeds"] *= 2
        record["shuffle_count"] = 2
        manifest.write_text(json.dumps(record) + "\n")
        message = "Metrics manifest mismatch"
    else:
        row = next(row_dir.glob("rows_*.parquet"))
        with row.open("r+b") as handle:
            handle.write(b"FAIL")
        message = "aggregate checkpoint shard authentication failed"
    with pytest.raises(ValueError, match=message):
        runner.run_single_n(cfg, 2, strategies=strategies)
    assert not runner.simulation_done_path(cfg, 2).exists()
