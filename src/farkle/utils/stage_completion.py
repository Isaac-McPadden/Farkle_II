# src/farkle/utils/stage_completion.py
"""Shared helpers for tracking stage completion and resumability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from farkle.utils.writer import atomic_path

__all__ = ["read_stage_done", "stage_done_path", "stage_is_up_to_date", "write_stage_done"]

_DEFAULT_STATUS = "success"
_ALLOWED_STATUSES = {"success", "failed", "skipped"}
_SCHEMA_VERSION = 2
_DEFAULT_CACHE_KEY_VERSION = 2


def stage_done_path(stage_dir: Path, name: str) -> Path:
    """Return a ``.done.json`` marker for a named stage directory."""

    return stage_dir / f"{name}.done.json"


def _coerce_paths(paths: Iterable[Path]) -> list[Path]:
    return [Path(p) for p in paths]


def read_stage_done(done_path: Path) -> dict[str, object]:
    """Return stage metadata from a ``.done.json`` stamp."""

    payload: dict[str, object] = {
        "schema_version": None,
        "stage": None,
        "config_sha": None,
        "stage_config_sha": None,
        "cache_key_version": None,
        "inputs": [],
        "outputs": [],
        "status": "missing",
        "reason": None,
        "blocking_dependency": None,
        "upstream_stage": None,
    }
    if not done_path.exists():
        return payload
    try:
        meta = json.loads(done_path.read_text())
    except Exception:
        payload["status"] = "invalid"
        payload["reason"] = "invalid json"
        return payload
    payload["schema_version"] = meta.get("schema_version")
    payload["stage"] = meta.get("stage")
    payload["config_sha"] = meta.get("config_sha")
    payload["stage_config_sha"] = meta.get("stage_config_sha")
    payload["cache_key_version"] = meta.get("cache_key_version")
    payload["inputs"] = list(meta.get("inputs") or [])
    payload["outputs"] = list(meta.get("outputs") or [])
    payload["status"] = meta.get("status", _DEFAULT_STATUS)
    payload["reason"] = meta.get("reason")
    payload["blocking_dependency"] = meta.get("blocking_dependency")
    payload["upstream_stage"] = meta.get("upstream_stage")
    return payload


def stage_is_up_to_date(
    done_path: Path,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    *,
    config_sha: str | None = None,
    stage_config_sha: str | None = None,
    cache_key_version: int = _DEFAULT_CACHE_KEY_VERSION,
    stage: str | None = None,
    cfg: Any | None = None,
) -> bool:
    """Return ``True`` when *done_path* is newer than *inputs* and outputs exist."""

    if not done_path.exists():
        return False
    meta = read_stage_done(done_path)
    status = meta.get("status", _DEFAULT_STATUS)
    if status != "success":
        return False
    if meta.get("schema_version") != _SCHEMA_VERSION:
        return False
    if stage is not None and meta.get("stage") != stage:
        return False
    if cfg is not None and stage is not None:
        if config_sha is None:
            config_sha = getattr(cfg, "config_sha", None)
        if stage_config_sha is None:
            stage_config_sha = cfg.stage_config_sha(stage)
        cache_key_version = int(cfg.stage_cache_key_version(stage))
    recorded_sha = meta.get("config_sha")
    if config_sha is not None and recorded_sha not in (None, config_sha):
        return False
    expected_stage_sha = stage_config_sha if stage_config_sha is not None else config_sha
    recorded_stage_sha = meta.get("stage_config_sha")
    if expected_stage_sha is None or recorded_stage_sha != expected_stage_sha:
        return False
    recorded_cache_key_version = meta.get("cache_key_version")
    if recorded_cache_key_version != cache_key_version:
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
    stage_config_sha: str | None = None,
    cache_key_version: int = _DEFAULT_CACHE_KEY_VERSION,
    stage: str | None = None,
    cfg: Any | None = None,
    status: str = _DEFAULT_STATUS,
    reason: str | None = None,
    blocking_dependency: str | None = None,
    upstream_stage: str | None = None,
) -> None:
    """Persist a minimal completion stamp for a stage."""

    if status not in _ALLOWED_STATUSES:
        raise ValueError(
            f"Unsupported stage status {status!r}; expected one of {_ALLOWED_STATUSES}"
        )
    if status in {"failed", "skipped"} and (blocking_dependency is None or upstream_stage is None):
        raise ValueError(
            "blocking_dependency and upstream_stage are required when status is failed/skipped"
        )
    if cfg is not None and stage is not None:
        if config_sha is None:
            config_sha = getattr(cfg, "config_sha", None)
        if stage_config_sha is None:
            stage_config_sha = cfg.stage_config_sha(stage)
        cache_key_version = int(cfg.stage_cache_key_version(stage))

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "stage": stage,
        "config_sha": config_sha,
        "stage_config_sha": stage_config_sha if stage_config_sha is not None else config_sha,
        "cache_key_version": cache_key_version,
        "inputs": [str(p) for p in _coerce_paths(inputs)],
        "outputs": [str(p) for p in _coerce_paths(outputs)],
        "status": status,
        "reason": reason,
        "blocking_dependency": blocking_dependency,
        "upstream_stage": upstream_stage,
    }
    done_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(done_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
