"""Process-local counters for authenticated graph work.

The active collector is operational instrumentation only.  It never carries a
trust decision, artifact identity, or reusable filesystem result.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Iterator, Literal

OpenClass = Literal[
    "artifact",
    "sidecar",
    "manifest",
    "completion",
    "run_context",
    "active_config",
    "other",
]


@dataclass(slots=True)
class AuthenticationTelemetry:
    """Mutable operational counters owned by one orchestration invocation."""

    directory_traversals: int = 0
    artifact_opens: int = 0
    sidecar_opens: int = 0
    manifest_opens: int = 0
    completion_opens: int = 0
    run_context_opens: int = 0
    active_config_opens: int = 0
    other_opens: int = 0
    sha256_calls: int = 0
    artifact_bytes_hashed: int = 0
    sidecar_bytes_hashed: int = 0
    manifest_bytes_hashed: int = 0
    completion_bytes_hashed: int = 0
    run_context_bytes_hashed: int = 0
    other_bytes_hashed: int = 0
    schema_validations: int = 0
    metadata_validations: int = 0
    stage_state_resolutions: int = 0
    graph_audit_invocations: int = 0
    graph_root_traversals: int = 0
    snapshot_builds: int = 0
    snapshot_hits: int = 0
    snapshot_rejected_uses: int = 0
    snapshot_invalidations: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def bump(self, field_name: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self, field_name, int(getattr(self, field_name)) + int(amount))

    def as_metadata(self) -> dict[str, int]:
        with self._lock:
            values = {
                item.name: int(getattr(self, item.name))
                for item in fields(self)
                if item.name != "_lock"
            }
        values["total_opens"] = sum(
            int(values[f"{name}_opens"])
            for name in (
                "artifact",
                "sidecar",
                "manifest",
                "completion",
                "run_context",
                "active_config",
                "other",
            )
        )
        values["total_bytes_hashed"] = sum(
            int(values[f"{name}_bytes_hashed"])
            for name in (
                "artifact",
                "sidecar",
                "manifest",
                "completion",
                "run_context",
                "other",
            )
        )
        return values


_ACTIVE: contextvars.ContextVar[AuthenticationTelemetry | None] = contextvars.ContextVar(
    "farkle_authentication_telemetry",
    default=None,
)


@contextlib.contextmanager
def use_authentication_telemetry(
    telemetry: AuthenticationTelemetry,
) -> Iterator[AuthenticationTelemetry]:
    """Record authentication operations in *telemetry* for this context."""

    token = _ACTIVE.set(telemetry)
    try:
        yield telemetry
    finally:
        _ACTIVE.reset(token)


def active_authentication_telemetry() -> AuthenticationTelemetry | None:
    """Return the current operational collector, if one was explicitly installed."""

    return _ACTIVE.get()


def classify_authentication_path(path: Path | str) -> OpenClass:
    candidate = Path(path)
    name = candidate.name.lower()
    if name == "run_context.json":
        return "run_context"
    if name.endswith(".done.json"):
        return "completion"
    if name.endswith(".sidecar.json"):
        return "sidecar"
    if "manifest" in name:
        return "manifest"
    if name in {"active_config.yaml", "active_config.yml"}:
        return "active_config"
    if candidate.suffix.lower() in {
        ".arrow",
        ".json",
        ".jsonl",
        ".md",
        ".npy",
        ".parquet",
        ".pkl",
        ".png",
        ".txt",
        ".yaml",
        ".yml",
    }:
        return "artifact"
    return "other"


def record_authentication_open(path: Path | str) -> None:
    telemetry = _ACTIVE.get()
    if telemetry is None:
        return
    kind = classify_authentication_path(path)
    telemetry.bump(f"{kind}_opens")


def record_authentication_hash(path: Path | str, byte_count: int) -> None:
    telemetry = _ACTIVE.get()
    if telemetry is None:
        return
    kind = classify_authentication_path(path)
    if kind == "active_config":
        kind = "other"
    telemetry.bump("sha256_calls")
    telemetry.bump(f"{kind}_bytes_hashed", byte_count)


def record_schema_validation() -> None:
    telemetry = _ACTIVE.get()
    if telemetry is not None:
        telemetry.bump("schema_validations")


def record_metadata_validation() -> None:
    telemetry = _ACTIVE.get()
    if telemetry is not None:
        telemetry.bump("metadata_validations")


def record_stage_state_resolution() -> None:
    telemetry = _ACTIVE.get()
    if telemetry is not None:
        telemetry.bump("stage_state_resolutions")


__all__ = [
    "AuthenticationTelemetry",
    "active_authentication_telemetry",
    "classify_authentication_path",
    "record_authentication_hash",
    "record_authentication_open",
    "record_metadata_validation",
    "record_schema_validation",
    "record_stage_state_resolution",
    "use_authentication_telemetry",
]
