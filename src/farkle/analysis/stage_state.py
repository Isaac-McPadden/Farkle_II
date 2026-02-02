"""Shared helpers for tracking stage completion and resumability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from farkle.utils.writer import atomic_path

__all__ = ["read_stage_done", "stage_done_path", "stage_is_up_to_date", "write_stage_done"]

_DEFAULT_STATUS = "success"


def stage_done_path(stage_dir: Path, name: str) -> Path:
    """Return a ``.done.json`` marker for a named stage directory."""

    return stage_dir / f"{name}.done.json"


def _coerce_paths(paths: Iterable[Path]) -> list[Path]:
    return [Path(p) for p in paths]


def read_stage_done(done_path: Path) -> dict[str, object]:
    """Return stage metadata from a ``.done.json`` stamp.

    Missing fields are populated with defaults for backwards compatibility.
    If the marker is missing or malformed, ``status`` and ``reason`` describe
    the issue without requiring callers to probe the filesystem directly.
    """

    payload: dict[str, object] = {
        "config_sha": None,
        "inputs": [],
        "outputs": [],
        "status": "missing",
        "reason": None,
    }
    if not done_path.exists():
        return payload
    try:
        meta = json.loads(done_path.read_text())
    except Exception:
        payload["status"] = "invalid"
        payload["reason"] = "invalid json"
        return payload
    payload["config_sha"] = meta.get("config_sha")
    payload["inputs"] = list(meta.get("inputs") or [])
    payload["outputs"] = list(meta.get("outputs") or [])
    payload["status"] = meta.get("status", _DEFAULT_STATUS)
    payload["reason"] = meta.get("reason")
    return payload


def stage_is_up_to_date(
    done_path: Path,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    *,
    config_sha: str | None = None,
) -> bool:
    """Return ``True`` when *done_path* is newer than *inputs* and outputs exist."""

    if not done_path.exists():
        return False
    meta = read_stage_done(done_path)
    status = meta.get("status", _DEFAULT_STATUS)
    if status not in {"success", "skipped"}:
        return False
    recorded_sha = meta.get("config_sha")
    if config_sha is not None and recorded_sha not in (None, config_sha):
        return False

    done_mtime = done_path.stat().st_mtime
    for inp in _coerce_paths(inputs):
        if not inp.exists() or inp.stat().st_mtime > done_mtime:
            return False
    return all(Path(out).exists() for out in outputs)


def write_stage_done(
    done_path: Path,
    *,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    config_sha: str | None = None,
    status: str = _DEFAULT_STATUS,
    reason: str | None = None,
) -> None:
    """Persist a minimal completion stamp for a stage."""

    payload = {
        "config_sha": config_sha,
        "inputs": [str(p) for p in _coerce_paths(inputs)],
        "outputs": [str(p) for p in _coerce_paths(outputs)],
        "status": status,
        "reason": reason,
    }
    done_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(done_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
