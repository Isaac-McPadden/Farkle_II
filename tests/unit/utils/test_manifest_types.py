import importlib
import sys
import types

import pytest

from farkle.utils.manifest import append_manifest_line, append_manifest_many, iter_manifest
from farkle.utils.types import normalize_compression


def test_normalize_compression_invalid_value_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported parquet compression"):
        normalize_compression("zip")


def test_append_manifest_ensure_dir_false_on_missing_parent_raises(tmp_path) -> None:
    manifest_path = tmp_path / "missing" / "manifest.ndjson"

    with pytest.raises(FileNotFoundError):
        append_manifest_line(
            manifest_path,
            {"path": "row.parquet"},
            add_timestamp=False,
            ensure_dir=False,
        )


def test_iter_manifest_skips_blank_lines_between_records(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.ndjson"
    manifest_path.write_text('{"path":"a"}\n\n   \n{"path":"b"}\n', encoding="utf-8")

    rows = list(iter_manifest(manifest_path))

    assert [row["path"] for row in rows] == ["a", "b"]



def test_append_manifest_many_empty_records_no_file_created(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.ndjson"

    append_manifest_many(manifest_path, [], add_timestamp=False)

    assert not manifest_path.exists()


def test_open_append_fd_adds_binary_flag_on_windows(monkeypatch, tmp_path) -> None:
    import farkle.utils.manifest as manifest_mod

    observed: dict[str, int] = {}

    monkeypatch.setattr(manifest_mod.os, "name", "nt", raising=False)
    monkeypatch.setattr(manifest_mod.os, "O_BINARY", 0x8000, raising=False)

    def fake_open(path: str, flags: int, mode: int) -> int:
        observed["flags"] = flags
        observed["mode"] = mode
        return 123

    monkeypatch.setattr(manifest_mod.os, "open", fake_open)

    fd = manifest_mod._open_append_fd(tmp_path / "manifest.ndjson")

    assert fd == 123
    assert observed["flags"] & manifest_mod.os.O_BINARY



def test_append_manifest_many_without_ensure_dir_writes_when_parent_exists(tmp_path) -> None:
    manifest_path = tmp_path / "existing" / "manifest.ndjson"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    append_manifest_many(
        manifest_path,
        [{"path": "a"}],
        add_timestamp=False,
        ensure_dir=False,
    )

    rows = list(iter_manifest(manifest_path))
    assert rows == [{"path": "a"}]


def test_windows_lock_helpers_retry_and_unlock(monkeypatch, tmp_path) -> None:
    original_platform = sys.platform
    import farkle.utils.manifest as manifest_mod

    fake_msvcrt = types.SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2)
    attempts: list[int] = []

    def fake_locking(fd: int, mode: int, _size: int) -> None:
        attempts.append(mode)
        if mode == fake_msvcrt.LK_NBLCK and len(attempts) == 1:
            raise OSError("busy")

    fake_msvcrt.locking = fake_locking
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(sys, "platform", "win32")

    reloaded = importlib.reload(manifest_mod)
    monkeypatch.setattr(reloaded.os, "lseek", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(reloaded.time, "sleep", lambda *_args, **_kwargs: None)

    reloaded._lock_fd(11)
    reloaded._unlock_fd(11)

    assert attempts == [fake_msvcrt.LK_NBLCK, fake_msvcrt.LK_NBLCK, fake_msvcrt.LK_UNLCK]

    monkeypatch.setattr(sys, "platform", original_platform)
    importlib.reload(reloaded)
