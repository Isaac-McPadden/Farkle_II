from __future__ import annotations

from pathlib import Path

import pytest
from scripts import check_structure_release


def test_artifact_root_is_required() -> None:
    with pytest.raises(SystemExit) as excinfo:
        check_structure_release._parser().parse_args([])

    assert excinfo.value.code == 2


def test_explicit_artifact_roots_are_forwarded(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    captured: dict[str, object] = {}

    def fake_run_release_audits(repository_root, *, config_paths, artifact_roots):
        captured["repository_root"] = repository_root
        captured["config_paths"] = tuple(config_paths)
        captured["artifact_roots"] = tuple(artifact_roots)
        return []

    monkeypatch.setattr(check_structure_release, "run_release_audits", fake_run_release_audits)

    result = check_structure_release.main(
        ["--artifact-root", str(first), "--artifact-root", str(second)]
    )

    assert result == 0
    assert captured["repository_root"] == check_structure_release.ROOT
    assert captured["config_paths"] == check_structure_release.RUNNABLE_CONFIGS
    assert captured["artifact_roots"] == (first, second)
