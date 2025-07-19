import importlib
import importlib.metadata
import pathlib

import farkle


def _reload(monkeypatch):
    # Restore original unlink so reload grabs it
    monkeypatch.setattr(pathlib.Path, "unlink", farkle._orig_unlink)
    return importlib.reload(farkle)


def test_version_patch(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda _pkg: "9.9.9")
    mod = _reload(monkeypatch)
    assert mod.__version__ == "9.9.9"


def test_version_fallback(monkeypatch):
    def raise_pkg(_pkg):
        raise importlib.metadata.PackageNotFoundError
    monkeypatch.setattr(importlib.metadata, "version", raise_pkg)
    mod = _reload(monkeypatch)
    assert mod.__version__ == mod._read_version_from_toml()


def test_safe_unlink_permissionerror(tmp_path, monkeypatch):
    target = tmp_path / "file.txt"
    target.write_text("x")

    def raise_perm(_path: pathlib.Path, *, missing_ok: bool = False):
        _ = missing_ok
        raise PermissionError

    monkeypatch.setattr(farkle, "_orig_unlink", raise_perm)

    farkle._safe_unlink(target)
    assert target.exists()