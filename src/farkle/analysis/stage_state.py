"""Shared helpers for tracking stage completion and resumability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from farkle.utils.writer import atomic_path

__all__ = ["stage_done_path", "stage_is_up_to_date", "write_stage_done"]


def stage_done_path(stage_dir: Path, name: str) -> Path:
    """Return a ``.done.json`` marker for a named stage directory."""

    return stage_dir / f"{name}.done.json"


def _coerce_paths(paths: Iterable[Path]) -> list[Path]:
    return [Path(p) for p in paths]


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
    try:
        meta = json.loads(done_path.read_text())
    except Exception:
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
) -> None:
    """Persist a minimal completion stamp for a stage."""

    payload = {
        "config_sha": config_sha,
        "inputs": [str(p) for p in _coerce_paths(inputs)],
        "outputs": [str(p) for p in _coerce_paths(outputs)],
    }
    done_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(done_path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
