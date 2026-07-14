"""Read-only release audits for contracts, migration, and artifact sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterable

from farkle.config import ArtifactScope, load_app_config
from farkle.utils.artifact_contract import SIDECAR_SUFFIX, sidecar_path, validate_artifact_sidecar

_DERIVED_SUFFIXES: Final = {".json", ".md", ".parquet", ".png"}
_STATE_SUFFIXES: Final = (".checkpoint.json", ".done.json")
_RETIRED_ENTRY_POINTS: Final = (
    "src/farkle/analysis/agreement.py",
    "src/farkle/analysis/coverage_by_k.py",
    "src/farkle/analysis/interseed_analysis.py",
    "src/farkle/analysis/meta.py",
    "src/farkle/analysis/reporting.py",
    "src/farkle/analysis/variance.py",
    "src/farkle/orchestration/pipeline.py",
    "src/farkle/utils/stage_io.py",
    "src/farkle/utils/tiers.py",
    "src/pipeline.py",
)
_CANONICAL_SCOPE_PARTS: Final = frozenset(scope.value for scope in ArtifactScope)


def audit_runnable_configs(config_paths: Iterable[Path]) -> list[str]:
    """Return failures from loading and validating runnable config files."""

    failures: list[str] = []
    for path in sorted(Path(item) for item in config_paths):
        try:
            cfg = load_app_config(path)
            cfg.validate_statistical_contract(require_two_roots=False)
        except Exception as exc:  # noqa: BLE001 - audit reports every contract failure
            failures.append(f"{path}: {type(exc).__name__}: {exc}")
    return failures


def audit_retired_entry_points(repository_root: Path) -> list[str]:
    """Return retired source entry points that still exist after migration."""

    root = repository_root.resolve()
    return [relative for relative in _RETIRED_ENTRY_POINTS if (root / relative).exists()]


def _is_canonical_derived_artifact(path: Path, audit_root: Path) -> bool:
    relative = path.relative_to(audit_root)
    if not _CANONICAL_SCOPE_PARTS.intersection(relative.parts):
        return False
    if path.name.endswith(SIDECAR_SUFFIX) or path.name.endswith(_STATE_SUFFIXES):
        return False
    return path.suffix.lower() in _DERIVED_SUFFIXES


def audit_sidecar_completeness(audit_root: Path) -> list[str]:
    """Return missing, orphaned, duplicate, or incompatible sidecar failures."""

    root = audit_root.resolve()
    if not root.exists():
        return []
    failures: list[str] = []
    data_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and _is_canonical_derived_artifact(path, root)
    )
    expected_sidecars = {sidecar_path(path) for path in data_paths}
    observed_sidecars = set(root.rglob(f"*{SIDECAR_SUFFIX}"))
    for path in data_paths:
        metadata_path = sidecar_path(path)
        if not metadata_path.is_file():
            failures.append(f"missing sidecar: {path}")
            continue
        try:
            validate_artifact_sidecar(path)
        except Exception as exc:  # noqa: BLE001 - audit reports every contract failure
            failures.append(f"incompatible sidecar: {path}: {exc}")
    for metadata_path in sorted(observed_sidecars.difference(expected_sidecars)):
        failures.append(f"orphan sidecar: {metadata_path}")
    return failures


def run_release_audits(
    repository_root: Path,
    *,
    config_paths: Iterable[Path],
    artifact_roots: Iterable[Path] = (),
) -> list[str]:
    """Return every release-audit failure in deterministic order."""

    failures = audit_runnable_configs(config_paths)
    failures.extend(audit_retired_entry_points(repository_root))
    for artifact_root in artifact_roots:
        failures.extend(audit_sidecar_completeness(artifact_root))
    return sorted(failures)


__all__ = [
    "audit_retired_entry_points",
    "audit_runnable_configs",
    "audit_sidecar_completeness",
    "run_release_audits",
]
