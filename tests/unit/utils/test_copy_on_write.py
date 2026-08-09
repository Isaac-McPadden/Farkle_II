from __future__ import annotations

import errno
import shutil
from pathlib import Path

from farkle.utils import copy_on_write


def test_reflink_capability_is_reported_without_physical_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"stable bytes" * 128)

    def clone(src: Path, dst: Path) -> str:
        # Emulate successful platform cloning; semantic independence is what the
        # lifecycle requires, not inode sharing.
        shutil.copyfile(src, dst)
        return "fixture_reflink"

    monkeypatch.setattr(copy_on_write, "try_reflink", clone)
    provenance = copy_on_write.clone_or_copy_bounded(
        source,
        destination,
        copy_buffer_bytes=64,
    )

    assert destination.read_bytes() == source.read_bytes()
    assert provenance.representation == "reflink"
    assert provenance.backend == "fixture_reflink"
    assert provenance.fallback_reason is None


def test_unsupported_reflink_uses_bounded_physical_copy(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(bytes(range(256)) * 32)

    def unsupported(_src: Path, _dst: Path) -> str:
        raise OSError(errno.EOPNOTSUPP, "fixture filesystem has no clone support")

    monkeypatch.setattr(copy_on_write, "try_reflink", unsupported)
    provenance = copy_on_write.clone_or_copy_bounded(
        source,
        destination,
        copy_buffer_bytes=127,
    )

    assert destination.read_bytes() == source.read_bytes()
    assert provenance.representation == "physical_copy"
    assert provenance.backend == "bounded_copyfileobj"
    assert provenance.fallback_reason is not None
    assert provenance.fallback_reason.endswith(":OSError")
    assert provenance.copy_buffer_bytes == 127


def test_clone_destination_is_independent_after_creation(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"before")
    monkeypatch.setattr(
        copy_on_write,
        "try_reflink",
        lambda src, dst: (shutil.copyfile(src, dst), "fixture_reflink")[1],
    )

    copy_on_write.clone_or_copy_bounded(source, destination, copy_buffer_bytes=8)
    source.write_bytes(b"after!")

    assert destination.read_bytes() == b"before"
