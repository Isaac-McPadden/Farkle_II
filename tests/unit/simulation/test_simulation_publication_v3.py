from __future__ import annotations

import json
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
from farkle.utils.artifact_contract import sha256_file, sidecar_path
from farkle.utils.authenticated_contract import (
    canonical_json_bytes,
    load_immutable_manifest_sidecar,
)
from farkle.utils.random import RNG_SCHEME_VERSION
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION, TOURNAMENT_METHOD_VERSION
from farkle.utils.streaming_loop import run_streaming_shard


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
    real_replace = authenticated_contract.replace_file_atomic
    real_metadata = runner.validate_authenticated_artifact_metadata

    def counted_hash(path: Path | str, **kwargs: Any) -> str:
        candidate = Path(path)
        if candidate.parent == row_dir and candidate.name.startswith("._tmp_"):
            operations["data_hashes"] += 1
        return real_hash(path, **kwargs)

    def counted_replace(source: Path | str, destination: Path | str) -> None:
        candidate = Path(destination)
        if candidate.name.startswith("rows_") and candidate.suffix == ".parquet":
            operations["data_writes"] += 1
        real_replace(source, destination)

    def counted_metadata(path: Path | str, **kwargs: Any):  # noqa: ANN202
        candidate = Path(path)
        if candidate.name.startswith("rows_") and candidate.suffix == ".parquet":
            operations["metadata_validations"] += 1
        return real_metadata(path, **kwargs)

    monkeypatch.setattr(authenticated_contract, "sha256_file", counted_hash)
    monkeypatch.setattr(authenticated_contract, "replace_file_atomic", counted_replace)
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


def test_interrupted_atomic_staging_files_are_removed_and_never_published(
    tmp_path: Path,
) -> None:
    cfg = make_authenticated_v3_config(tmp_path, name="atomic_resume", root_seed=42)
    strategy, _workload, row_dir = _prepare(cfg, 1)
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

    removed = runner._cleanup_interrupted_simulation_staging_files(
        n_dir=cfg.n_dir(2),
        row_dir=row_dir,
        metric_chunk_dir=None,
        strategy_manifest_path=strategy,
    )
    assert removed == sorted(staging, key=lambda path: path.as_posix())
    assert not any(path.exists() for path in staging)
    assert unrelated_hidden.read_bytes() == b"preserve"


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


def test_real_two_worker_tournament_uses_producer_owned_publication(tmp_path: Path) -> None:
    def _run(name: str, workers: int) -> AppConfig:
        cfg = make_authenticated_v3_config(tmp_path, name=name, root_seed=42)
        cfg.sim.row_dir = Path("rows")
        cfg.sim.metric_chunk_dir = Path("metric_chunks")
        cfg.sim.expanded_metrics = True
        cfg.sim.n_jobs = workers
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

    sequential = _run("real_workers_1", 1)
    parallel = _run("real_workers_2", 2)

    def _published_bytes(cfg: AppConfig) -> dict[str, bytes]:
        return {
            path.relative_to(cfg.results_root).as_posix(): path.read_bytes()
            for path in sorted(cfg.results_root.rglob("*"))
            if path.is_file() and not path.name.startswith("._")
        }

    assert _published_bytes(sequential) == _published_bytes(parallel)
