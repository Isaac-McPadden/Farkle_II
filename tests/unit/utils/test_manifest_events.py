from __future__ import annotations

import importlib
import json
import sys
import types

import pytest

import farkle.utils.manifest as manifest


def test_append_manifest_event_writes_v2_payload(tmp_path) -> None:
    manifest_path = tmp_path / "analysis_manifest.jsonl"

    manifest.append_manifest_event(
        manifest_path,
        {"event": manifest.EVENT_STAGE_START, "stage": "report"},
        run_id="run-7",
        config_sha="sha-123",
        add_timestamp=False,
    )

    rows = list(manifest.iter_manifest(manifest_path))
    assert rows == [
        {
            "schema_version": manifest.MANIFEST_SCHEMA_VERSION,
            "run_id": "run-7",
            "event": manifest.EVENT_STAGE_START,
            "config_sha": "sha-123",
            "stage": "report",
        }
    ]


def test_append_manifest_event_validation_errors() -> None:
    with pytest.raises(ValueError, match="missing 'event'"):
        manifest.append_manifest_event(
            "unused.jsonl",
            {},
            run_id="run-1",
            config_sha=None,
        )

    with pytest.raises(ValueError, match="legacy manifest event"):
        manifest.append_manifest_event(
            "unused.jsonl",
            {"event": "step_end"},
            run_id="run-1",
            config_sha=None,
        )


def test_manifest_lines_are_v2_and_rotation_cover_edge_cases(tmp_path) -> None:
    valid = tmp_path / "valid.jsonl"
    valid.write_text(
        "\n"
        + json.dumps(
            {
                "schema_version": manifest.MANIFEST_SCHEMA_VERSION,
                "event": manifest.EVENT_RUN_START,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert manifest._manifest_lines_are_v2(valid) is True

    invalid_json = tmp_path / "invalid.jsonl"
    invalid_json.write_text("{bad json}\n", encoding="utf-8")
    assert manifest._manifest_lines_are_v2(invalid_json) is False

    invalid_shape = tmp_path / "invalid_shape.jsonl"
    invalid_shape.write_text("[1, 2, 3]\n", encoding="utf-8")
    assert manifest._manifest_lines_are_v2(invalid_shape) is False

    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        json.dumps({"schema_version": manifest.MANIFEST_SCHEMA_VERSION, "event": "stage-end"})
        + "\n",
        encoding="utf-8",
    )
    assert manifest._manifest_lines_are_v2(legacy) is False

    wrong_version = tmp_path / "wrong_version.jsonl"
    wrong_version.write_text(
        json.dumps({"schema_version": 1, "event": manifest.EVENT_RUN_START}) + "\n",
        encoding="utf-8",
    )
    assert manifest._manifest_lines_are_v2(wrong_version) is False

    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    assert manifest._manifest_lines_are_v2(unreadable) is False

    rotating = tmp_path / "manifest.jsonl"
    rotating.write_text(json.dumps({"event": "step_end"}) + "\n", encoding="utf-8")
    rotated = tmp_path / "manifest.pre_v2.jsonl"
    rotated.write_text("stale\n", encoding="utf-8")

    result = manifest.ensure_manifest_v2(rotating)

    assert result == rotated
    assert rotated.read_text(encoding="utf-8") == json.dumps({"event": "step_end"}) + "\n"
    assert not rotating.exists()

    v2_manifest = tmp_path / "v2.jsonl"
    v2_manifest.write_text(
        json.dumps(
            {
                "schema_version": manifest.MANIFEST_SCHEMA_VERSION,
                "event": manifest.EVENT_RUN_END,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert manifest.ensure_manifest_v2(v2_manifest) is None
    assert manifest.ensure_manifest_v2(tmp_path / "missing.jsonl") is None


def test_make_run_id_sanitizes_label(monkeypatch) -> None:
    monkeypatch.setattr(manifest.time, "time_ns", lambda: 123456789)

    run_id = manifest.make_run_id("Alpha Beta/Delta")

    assert run_id == "Alpha_Beta_Delta_123456789"


def test_write_all_handles_partial_writes(monkeypatch) -> None:
    calls: list[bytes] = []

    def fake_write(_fd: int, data: memoryview) -> int:
        blob = bytes(data)
        calls.append(blob)
        return 2 if len(blob) > 2 else len(blob)

    monkeypatch.setattr(manifest.os, "write", fake_write)

    manifest._write_all(5, b"abcde")

    assert calls == [b"abcde", b"cde", b"e"]


def test_posix_lock_helpers_use_fcntl(monkeypatch) -> None:
    import farkle.utils.manifest as manifest_mod

    original_platform = sys.platform
    fake_fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_UN=2)
    calls: list[tuple[int, int]] = []

    fake_fcntl.flock = lambda fd, mode: calls.append((fd, mode))
    monkeypatch.setitem(sys.modules, "fcntl", fake_fcntl)
    monkeypatch.setattr(sys, "platform", "linux")

    reloaded = importlib.reload(manifest_mod)
    reloaded._lock_fd(11)
    reloaded._unlock_fd(11)

    assert calls == [(11, fake_fcntl.LOCK_EX), (11, fake_fcntl.LOCK_UN)]

    monkeypatch.setattr(sys, "platform", original_platform)
    importlib.reload(reloaded)
