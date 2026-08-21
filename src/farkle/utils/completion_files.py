"""Semantic classification for completion-like files.

Canonical stage completions and durable partition checkpoints deliberately use
different contracts even though both end in ``.done.json``. This module owns
the path-level namespace boundary used by snapshot construction and release
auditing; callers must never infer canonical membership from the suffix alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Iterable, Mapping

_DONE_SUFFIX: Final = ".done.json"
_UNIT_DONE_SUFFIX: Final = ".unit.done.json"
_STATE_SUFFIXES: Final = (".checkpoint.json", ".state.json")
_UNIT_STAMP_FIELDS: Final = frozenset(
    {
        "unit_stamp_schema_version",
        "completion_state",
        "stage_name",
        "stage_identity_sha256",
        "root_seed",
        "input_identities",
        "statistical_config_sha256",
        "code_identity_sha256",
        "output_schema_version",
        "method_version",
        "unit_key",
        "relative_output",
        "unit_input_identities",
        "output_size_bytes",
        "output_sha256",
        "unit_metadata",
        "stamp_sha256",
    }
)
_STAGE_COMPLETION_FIELDS: Final = frozenset(
    {"lifecycle_contract_version", "stage_identity_sha256", "state", "outputs"}
)


def _json_mapping(path: Path) -> Mapping[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _is_partition_unit_contract(path: Path, output: Path) -> bool:
    payload = _json_mapping(path)
    relative_output = payload.get("relative_output") if payload is not None else None
    return bool(
        payload is not None
        and set(payload) == _UNIT_STAMP_FIELDS
        and payload.get("unit_stamp_schema_version") == 1
        and payload.get("completion_state") == "complete_valid"
        and isinstance(relative_output, str)
        and Path(relative_output).name == output.name
        and isinstance(payload.get("output_size_bytes"), int)
        and isinstance(payload.get("output_sha256"), str)
        and len(str(payload.get("output_sha256"))) == 64
        and isinstance(payload.get("stamp_sha256"), str)
        and len(str(payload.get("stamp_sha256"))) == 64
    )


def _is_operational_stage_completion(path: Path) -> bool:
    payload = _json_mapping(path)
    return bool(
        payload is not None
        and set(payload) == _STAGE_COMPLETION_FIELDS
        and payload.get("lifecycle_contract_version") == 1
        and payload.get("state")
        in {
            "complete_valid",
            "partial_resumable",
            "blocked_by_cap",
            "complete_stale",
        }
    )


class CompletionFileKind(StrEnum):
    """Typed namespaces for files that can resemble completion state."""

    CANONICAL_STAGE = "canonical_stage_completion"
    PARTITION_UNIT = "partition_unit_resumability"
    ACTIVE_CONFIGURATION = "active_configuration_state"
    OPERATIONAL_STATE = "checkpoint_or_operational_state"
    UNRELATED_DONE_LIKE = "unrelated_or_malformed_done_like"
    NOT_COMPLETION_LIKE = "not_completion_like"


@dataclass(frozen=True, slots=True)
class CompletionNamespace:
    """Structural contract for one root or pair completion namespace."""

    graph_root: Path
    analysis_root: Path
    canonical_paths: frozenset[Path]

    @classmethod
    def build(
        cls,
        *,
        graph_root: Path,
        analysis_root: Path,
        canonical_paths: Iterable[Path] = (),
    ) -> CompletionNamespace:
        return cls(
            graph_root=Path(graph_root).resolve(),
            analysis_root=Path(analysis_root).resolve(),
            canonical_paths=frozenset(Path(path).resolve() for path in canonical_paths),
        )

    @property
    def canonical_names(self) -> frozenset[str]:
        return frozenset(path.name for path in self.canonical_paths)

    @property
    def canonical_stage_directories(self) -> frozenset[Path]:
        return frozenset(
            path.parent for path in self.canonical_paths if path.parent.parent == self.analysis_root
        )

    def classify(self, path: Path) -> CompletionFileKind:
        """Classify *path* without authenticating or mutating its bytes."""

        candidate = Path(path).resolve()
        name = candidate.name
        if name == "active_config.done.json":
            return CompletionFileKind.ACTIVE_CONFIGURATION
        if name.endswith(_STATE_SUFFIXES):
            return CompletionFileKind.OPERATIONAL_STATE
        if not name.endswith(_DONE_SUFFIX):
            return CompletionFileKind.NOT_COMPLETION_LIKE
        if candidate in self.canonical_paths or name in self.canonical_names:
            return CompletionFileKind.CANONICAL_STAGE

        try:
            relative = candidate.relative_to(self.graph_root)
        except ValueError:
            return CompletionFileKind.UNRELATED_DONE_LIKE

        if name.endswith(_UNIT_DONE_SUFFIX):
            output = candidate.with_name(name[: -len(_UNIT_DONE_SUFFIX)])
            # PartitionedStage writes its stamp adjacent to its output below
            # its configurable output prefix. The adjacent output and typed
            # unit-stamp envelope are both required; the suffix by itself never
            # proves partition-unit membership.
            if output.is_file() and _is_partition_unit_contract(candidate, output):
                return CompletionFileKind.PARTITION_UNIT
            return CompletionFileKind.UNRELATED_DONE_LIKE

        # Simulation lifecycle stamps occupy exactly
        # <graph-root>/<k>_players/simulation.done.json.
        if (
            name == "simulation.done.json"
            and candidate.parent.parent == self.graph_root
            and candidate.parent.name.endswith("_players")
            and candidate.parent.name[: -len("_players")].isdigit()
        ):
            return CompletionFileKind.CANONICAL_STAGE

        # Analysis lifecycle stamps occupy one stage directory immediately
        # below the configured analysis root. Existing canonical stage
        # directories may also own substage stamps; those are operational
        # unless present in the executable canonical inventory.
        if candidate.parent.parent == self.analysis_root:
            if candidate.parent not in self.canonical_stage_directories:
                return CompletionFileKind.CANONICAL_STAGE
            return CompletionFileKind.OPERATIONAL_STATE

        if (
            "checkpoints" in relative.parts
            or any(part.startswith("_") for part in relative.parts[:-1])
            or "by_k" in relative.parts
            or "diagnostics" in relative.parts
        ):
            return CompletionFileKind.OPERATIONAL_STATE
        if _is_operational_stage_completion(candidate):
            return CompletionFileKind.OPERATIONAL_STATE
        return CompletionFileKind.UNRELATED_DONE_LIKE


__all__ = ["CompletionFileKind", "CompletionNamespace"]
