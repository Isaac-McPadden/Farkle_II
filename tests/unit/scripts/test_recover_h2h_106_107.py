"""Guard the copy boundary for the one-off historical H2H recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.recover_h2h_106_107 import Recovery


def test_recovery_requires_new_disjoint_destination(tmp_path: Path) -> None:
    source = tmp_path / "original"
    source.mkdir()
    for dest in (source, source / "new", tmp_path):
        with pytest.raises(ValueError, match="disjoint"):
            Recovery(source, dest)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        Recovery(source, existing)


@pytest.mark.parametrize("reconstruct", [False, True])
def test_recovery_checks_witness_before_publication(tmp_path: Path, reconstruct: bool) -> None:
    source = tmp_path / "original"
    source.mkdir()
    artifact = source / "family"
    artifact.write_bytes(b"superseded" if reconstruct else b"historical")
    original = artifact.read_bytes(), artifact.stat().st_mtime_ns
    expected = hashlib.sha256(b"historical").hexdigest()
    witness = source / "historical-consumer"
    witness.write_text(expected, encoding="ascii")
    run = Recovery(source, tmp_path / "recovered")
    try:
        if reconstruct:
            run.reconstructions[artifact] = b"historical"
        with pytest.raises(ValueError, match="expected .* observed"):
            run.copy(artifact, "0" * 64, witness)
        assert not (run.dest / artifact.name).exists()
        assert not run.files
        target = run.copy(artifact, witness.read_text(encoding="ascii"), witness)
        assert target.read_bytes() == b"historical"
        assert target != artifact
        with pytest.raises(ValueError, match="expected .* observed"):
            run.copy(artifact, "1" * 64, witness)
    finally:
        run.inventory.close()
    assert (artifact.read_bytes(), artifact.stat().st_mtime_ns) == original
    entry = json.loads((run.evidence / "verified_files.jsonl").read_text(encoding="utf-8"))
    assert entry == {
        "path": artifact.name,
        "sha256": expected,
        "witness": witness.name,
        "mode": "reconstructed" if reconstruct else "copied",
    }
