"""Stage I/O helpers for worker resolution and artifact path selection."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, NamedTuple

from farkle.utils.parallel import normalize_n_jobs


class ArtifactSelection(NamedTuple):
    """Preferred/legacy artifact resolution result."""

    path: Path
    used_legacy: bool


def resolve_worker_count(*requested_jobs: int | None, item_count: int) -> int:
    """Return bounded worker count using one or more ``n_jobs`` inputs."""

    if item_count <= 0:
        return 1
    resolved = [normalize_n_jobs(v) for v in requested_jobs if v is not None]
    baseline = max(resolved) if resolved else 1
    return max(1, min(item_count, baseline))


def select_preferred_or_legacy(
    preferred: Path, legacy: Path | None = None
) -> ArtifactSelection | None:
    """Choose preferred path when present, otherwise an existing legacy path."""

    if preferred.exists():
        return ArtifactSelection(preferred, used_legacy=False)
    if legacy is not None and legacy.exists():
        return ArtifactSelection(legacy, used_legacy=True)
    return None


def discover_per_k_artifacts(
    k_values: Iterable[int],
    *,
    preferred_path: Callable[[int], Path],
    legacy_path: Callable[[int], Path],
) -> list[tuple[int, ArtifactSelection]]:
    """Resolve per-k artifacts in deterministic ``k`` order."""

    selected: list[tuple[int, ArtifactSelection]] = []
    for k in k_values:
        choice = select_preferred_or_legacy(preferred_path(k), legacy_path(k))
        if choice is not None:
            selected.append((k, choice))
    return selected


__all__ = [
    "ArtifactSelection",
    "discover_per_k_artifacts",
    "resolve_worker_count",
    "select_preferred_or_legacy",
]
