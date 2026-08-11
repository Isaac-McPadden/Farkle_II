from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from farkle.config import ResourcesConfig
from farkle.utils.authenticated_contract import CodeIdentity, CodeIdentityPolicy
from farkle.utils.parallel import ResourceSafetyError
from farkle.utils.partitioned_stage import (
    PartitionedStageError,
    PartitionedStageIdentity,
    PartitionedUnit,
    resolved_code_identity_sha256,
    run_partitioned_stage,
    validate_final_manifest,
)
from farkle.utils.random import RandomPurpose, coordinate_rng


def _identity() -> PartitionedStageIdentity:
    return PartitionedStageIdentity(
        stage_name="partitioned_stage",
        root_seed=41,
        input_identities=(("source", "a" * 64),),
        statistical_config_sha256="b" * 64,
        code_identity_sha256="c" * 64,
        schema_version=3,
        method_version=2,
    )


def _resources() -> ResourcesConfig:
    return ResourcesConfig(
        scheduler_memory_budget_mb=768,
        parent_process_memory_mb=192,
        logical_cpu_budget=4,
        native_threads_per_worker=1,
        estimated_worker_memory_mb={"partitioned_stage": 64},
        stage_batch_bytes={"partitioned_stage": 4096},
    )


def _unit_source():
    return (PartitionedUnit((index,), f"part-{index:03d}.bin") for index in range(12))


def _deterministic_writer(unit: PartitionedUnit, path: Path) -> None:
    coordinate = int(unit.key[0])
    rng = coordinate_rng(
        RandomPurpose.BOOTSTRAP,
        root_seed=41,
        replicate_index=coordinate,
    )
    value = int(rng.integers(0, 2**31))
    path.write_bytes(f"root=41;coordinate={coordinate};value={value}\n".encode())


def _interrupting_writer(unit: PartitionedUnit, path: Path) -> None:
    if unit.key == (2,):
        raise RuntimeError("simulated interruption")
    _deterministic_writer(unit, path)


def _outputs(root: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted((root / "units").glob("*.bin"), key=lambda item: item.name)
    }


def test_worker_count_invariance_and_deterministic_manifest_order(tmp_path: Path) -> None:
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"
    serial = run_partitioned_stage(
        root=serial_root,
        identity=_identity(),
        unit_source=_unit_source,
        writer=_deterministic_writer,
        resources=_resources(),
        requested_workers=1,
    )
    parallel = run_partitioned_stage(
        root=parallel_root,
        identity=_identity(),
        unit_source=_unit_source,
        writer=_deterministic_writer,
        resources=_resources(),
        requested_workers=2,
        mp_start_method="spawn",
    )

    assert _outputs(serial_root) == _outputs(parallel_root)
    assert serial.manifest_sha256 == parallel.manifest_sha256
    lines = [json.loads(line) for line in serial.manifest_path.read_text().splitlines()]
    assert [line["unit_key"] for line in lines[1:-1]] == [[index] for index in range(12)]


def test_interruption_resume_reuses_only_authenticated_units(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_partitioned_stage(
            root=root,
            identity=_identity(),
            unit_source=_unit_source,
            writer=_interrupting_writer,
            resources=_resources(),
            requested_workers=1,
        )
    assert not (root / "partition_manifest.jsonl").exists()
    assert len(_outputs(root)) == 2
    orphan = root / "units" / "._partition_orphan"
    orphan.write_bytes(b"partial")

    resumed = run_partitioned_stage(
        root=root,
        identity=_identity(),
        unit_source=_unit_source,
        writer=_deterministic_writer,
        resources=_resources(),
        requested_workers=2,
        mp_start_method="spawn",
    )
    assert resumed.reused_units == 2
    assert resumed.completed_units == 10
    assert resumed.required_units == 12
    assert not orphan.exists()
    assert any(path.read_bytes() == b"partial" for path in (root / "quarantine").iterdir())


def test_corrupt_and_missing_units_are_quarantined_and_rebuilt(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    run_partitioned_stage(
        root=root,
        identity=_identity(),
        unit_source=_unit_source,
        writer=_deterministic_writer,
        resources=_resources(),
        requested_workers=1,
    )
    (root / "units" / "part-001.bin").write_bytes(b"corrupt")
    (root / "units" / "part-002.bin.unit.done.json").unlink()

    repaired = run_partitioned_stage(
        root=root,
        identity=_identity(),
        unit_source=_unit_source,
        writer=_deterministic_writer,
        resources=_resources(),
        requested_workers=1,
    )
    assert repaired.reused_units == 10
    assert repaired.completed_units == 2
    assert validate_final_manifest(
        repaired.manifest_path,
        root=root,
        identity=_identity(),
        unit_source=_unit_source,
    ) == (repaired.manifest_sha256, 12)
    assert any((root / "quarantine").iterdir())


def test_identity_change_invalidates_every_unit(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    run_partitioned_stage(
        root=root,
        identity=_identity(),
        unit_source=_unit_source,
        writer=_deterministic_writer,
        resources=_resources(),
        requested_workers=1,
    )
    changed = replace(_identity(), statistical_config_sha256="d" * 64)
    result = run_partitioned_stage(
        root=root,
        identity=changed,
        unit_source=_unit_source,
        writer=_deterministic_writer,
        resources=_resources(),
        requested_workers=1,
    )
    assert result.reused_units == 0
    assert result.completed_units == 12


def test_per_unit_input_identity_change_rewrites_only_that_unit(tmp_path: Path) -> None:
    root = tmp_path / "stage"

    def units(digest: str):
        return (
            PartitionedUnit((0,), "part-000.bin", (("source", "a" * 64),)),
            PartitionedUnit((1,), "part-001.bin", (("source", digest),)),
        )

    run_partitioned_stage(
        root=root,
        identity=_identity(),
        unit_source=lambda: iter(units("b" * 64)),
        writer=_deterministic_writer,
        resources=_resources(),
        requested_workers=1,
    )
    result = run_partitioned_stage(
        root=root,
        identity=_identity(),
        unit_source=lambda: iter(units("c" * 64)),
        writer=_deterministic_writer,
        resources=_resources(),
        requested_workers=1,
    )

    assert result.reused_units == 1
    assert result.completed_units == 1


def test_manifest_authenticates_validator_metadata(tmp_path: Path) -> None:
    root = tmp_path / "stage"

    def validator(unit: PartitionedUnit, output: Path) -> dict[str, int]:
        return {"coordinate": int(unit.key[0]), "bytes": output.stat().st_size}

    result = run_partitioned_stage(
        root=root,
        identity=_identity(),
        unit_source=_unit_source,
        writer=_deterministic_writer,
        resources=_resources(),
        requested_workers=1,
        validator=validator,
    )
    records = [json.loads(line) for line in result.manifest_path.read_text().splitlines()]

    assert records[1]["unit_metadata"]["coordinate"] == 0
    assert records[-2]["unit_metadata"]["coordinate"] == 11


def test_resumable_identity_binds_full_dirty_code_fingerprint() -> None:
    first = CodeIdentity(
        commit="a" * 40,
        policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY.value,
        state="development_dirty",
        dirty_fingerprint_sha256="b" * 64,
    )
    second = replace(first, dirty_fingerprint_sha256="c" * 64)

    assert resolved_code_identity_sha256(SimpleNamespace(_code_identity=first)) != (
        resolved_code_identity_sha256(SimpleNamespace(_code_identity=second))
    )


def test_unsorted_unit_enumeration_fails_closed(tmp_path: Path) -> None:
    def unsorted_units():
        return iter((PartitionedUnit((2,), "b.bin"), PartitionedUnit((1,), "a.bin")))

    with pytest.raises(PartitionedStageError, match="strictly increasing"):
        run_partitioned_stage(
            root=tmp_path / "stage",
            identity=_identity(),
            unit_source=unsorted_units,
            writer=_deterministic_writer,
            resources=_resources(),
            requested_workers=1,
        )


class _AbortingGuard:
    peak_rss_bytes = 949 * 1024 * 1024

    def __init__(self) -> None:
        self.calls = 0

    def check_before_schedule(self, *, force: bool = False) -> int:
        self.calls += 1
        if not force and self.calls >= 2:
            raise ResourceSafetyError("synthetic RSS abort")
        return self.peak_rss_bytes


def test_rss_abort_stops_submission_and_never_publishes_manifest(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    with pytest.raises(ResourceSafetyError, match="synthetic RSS abort"):
        run_partitioned_stage(
            root=root,
            identity=_identity(),
            unit_source=_unit_source,
            writer=_deterministic_writer,
            resources=_resources(),
            requested_workers=1,
            memory_guard=_AbortingGuard(),  # type: ignore[arg-type]
        )
    assert not (root / "partition_manifest.jsonl").exists()


class _LateAbortingGuard:
    peak_rss_bytes = 951 * 1024 * 1024

    def __init__(self) -> None:
        self.forced_calls = 0

    def check_before_schedule(self, *, force: bool = False) -> int:
        if force:
            self.forced_calls += 1
            if self.forced_calls == 3:
                raise ResourceSafetyError("synthetic late RSS abort")
        return self.peak_rss_bytes


def test_late_rss_abort_quarantines_published_manifest(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    with pytest.raises(ResourceSafetyError, match="synthetic late RSS abort"):
        run_partitioned_stage(
            root=root,
            identity=_identity(),
            unit_source=_unit_source,
            writer=_deterministic_writer,
            resources=_resources(),
            requested_workers=1,
            memory_guard=_LateAbortingGuard(),  # type: ignore[arg-type]
        )
    assert not (root / "partition_manifest.jsonl").exists()
