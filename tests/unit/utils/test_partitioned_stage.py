from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from farkle.config import ResourcesConfig
from farkle.utils.parallel import ResourceSafetyError
from farkle.utils.partitioned_stage import (
    PartitionedStageError,
    PartitionedStageIdentity,
    PartitionedUnit,
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
        target_memory_mb=768,
        rss_abort_mb=950,
        parent_reserve_mb=192,
        logical_cpu_workers=4,
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
