"""Authenticated artifact and lifecycle primitives for contract version 3.

This module is deliberately additive.  Existing stage producers still publish
version-2 sidecars through :mod:`farkle.utils.artifact_contract`; later
migrations opt into these stricter primitives one stage at a time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, closing, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, TypeVar, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from farkle.config import AppConfig, ArtifactScope, effective_config_dict
from farkle.protocols import ARTIFACT_CONTRACT_VERSION as ARTIFACT_CONTRACT_VERSION
from farkle.protocols import DERIVED_SCHEMA_VERSION
from farkle.utils.artifact_contract import (
    make_staged_path,
    read_json_file_with_retry,
    read_json_file_with_sha256,
    replace_file_atomic,
    retry_transient_io,
    sha256_file,
    sidecar_path,
)
from farkle.utils.artifact_contract import (
    publish_staged_file_atomic as _publish_staged_file_atomic,
)
from farkle.utils.stage_completion import CompletionState
from farkle.utils.writer import atomic_path

LOGGER = logging.getLogger(__name__)


LIFECYCLE_CONTRACT_VERSION: Final = 1
MANIFEST_CONTRACT_VERSION: Final = 1
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_BY_K_RE: Final = re.compile(r"^([1-9][0-9]*)p$")
_SIMULATION_SCOPE: Final = ArtifactScope.DIAGNOSTICS.value
_T = TypeVar("_T")


def publish_staged_file_atomic(staged_path: Path | str, final_path: Path | str) -> None:
    """Use the central publisher while retaining the local replacement test seam."""

    _publish_staged_file_atomic(staged_path, final_path, replace=replace_file_atomic)


class AuthenticatedContractError(RuntimeError):
    """Base class for fail-closed authenticated-contract failures."""


class MissingSidecarError(AuthenticatedContractError):
    """An artifact exists without its required adjacent sidecar."""


class CorruptSidecarError(AuthenticatedContractError):
    """A present sidecar is malformed or internally inconsistent."""


class ArtifactMismatchError(AuthenticatedContractError):
    """Artifact bytes, schema, scope, or declared provenance do not match."""


class CodeIdentityError(AuthenticatedContractError):
    """A reliable code identity could not be established under policy."""


def _require_sha256(value: str, *, label: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON mapping keys must be strings")
            normalized[key] = _jsonable(item)
        return normalized
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON does not permit NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a contract value with one deterministic JSON representation."""

    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def identity_sha256(value: Any) -> str:
    """Hash a value's canonical JSON representation."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalArtifactLocation:
    """Path-independent identity of one artifact in a canonical stage scope."""

    stage_key: str
    scope: str
    relative_path: str
    player_count: int | None = None

    def __post_init__(self) -> None:
        if self.stage_key == "simulation":
            if self.scope != _SIMULATION_SCOPE or self.player_count is not None:
                raise ValueError("simulation locations require diagnostics scope")
            scope = ArtifactScope.DIAGNOSTICS
        else:
            scope = ArtifactScope(self.scope)
        relative = Path(self.relative_path)
        if not self.stage_key.strip():
            raise ValueError("stage_key must be non-blank")
        if not self.relative_path or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("relative_path must remain within its canonical scope")
        if scope is not None and scope.requires_player_count:
            if isinstance(self.player_count, bool) or self.player_count is None:
                raise ValueError("by_k locations require player_count")
            if self.player_count < 1:
                raise ValueError("player_count must be positive")
        elif scope is not None and self.player_count is not None:
            raise ValueError(f"{scope.value} locations do not accept player_count")

    def path(self, cfg: AppConfig) -> Path:
        """Resolve this logical identity through the application's path API."""

        if self.stage_key == "simulation":
            if self.scope != _SIMULATION_SCOPE or self.player_count is not None:
                raise ValueError("simulation artifacts use the diagnostics scope")
            return cfg.results_root / self.relative_path
        return cfg.scope_path(
            self.stage_key,
            ArtifactScope(self.scope),
            self.relative_path,
            k=self.player_count,
        )

    def require_path(self, cfg: AppConfig, path: Path | str) -> Path:
        """Reject a physical path that does not exactly realize this identity."""

        actual = Path(path).resolve()
        expected = self.path(cfg).resolve()
        if actual != expected:
            raise ArtifactMismatchError(
                f"artifact path {actual} does not match canonical path {expected}"
            )
        return Path(path)


def derive_canonical_location(
    cfg: AppConfig, path: Path | str, *, stage_key: str
) -> CanonicalArtifactLocation:
    """Derive scope and relative name from a canonical physical path."""

    artifact = Path(path).resolve()
    if stage_key == "simulation":
        stage_root = cfg.results_root.resolve()
        try:
            relative = artifact.relative_to(stage_root)
        except ValueError as exc:
            raise ArtifactMismatchError(
                f"artifact {artifact} is outside the canonical simulation root"
            ) from exc
        location = CanonicalArtifactLocation(
            stage_key=stage_key,
            scope=_SIMULATION_SCOPE,
            relative_path=relative.as_posix(),
        )
        location.require_path(cfg, artifact)
        return location
    stage_root = cfg.stage_dir(stage_key, create=False).resolve()
    try:
        relative = artifact.relative_to(stage_root)
    except ValueError as exc:
        raise ArtifactMismatchError(
            f"artifact {artifact} is outside canonical stage {stage_key!r}"
        ) from exc
    if len(relative.parts) < 2:
        raise ArtifactMismatchError("artifact is not inside one of the six canonical scopes")
    try:
        scope = ArtifactScope(relative.parts[0])
    except ValueError as exc:
        raise ArtifactMismatchError(
            f"unknown canonical scope directory {relative.parts[0]!r}"
        ) from exc
    if scope.requires_player_count:
        if len(relative.parts) < 3 or (match := _BY_K_RE.fullmatch(relative.parts[1])) is None:
            raise ArtifactMismatchError(
                "by_k artifact path must include a positive '<k>p' directory"
            )
        player_count = int(match.group(1))
        artifact_relative = Path(*relative.parts[2:]).as_posix()
    else:
        player_count = None
        artifact_relative = Path(*relative.parts[1:]).as_posix()
    location = CanonicalArtifactLocation(
        stage_key=stage_key,
        scope=scope.value,
        player_count=player_count,
        relative_path=artifact_relative,
    )
    location.require_path(cfg, artifact)
    return location


@dataclass(frozen=True, slots=True)
class ArrowFieldIdentity:
    """Recursive Arrow field identity, including type and nullability."""

    name: str
    type: str
    nullable: bool
    children: tuple["ArrowFieldIdentity", ...] = ()


@dataclass(frozen=True, slots=True)
class ArrowSchemaIdentity:
    """Exact ordered Arrow schema declaration and versioned fingerprint."""

    schema_version: int
    fields: tuple[ArrowFieldIdentity, ...]
    fingerprint_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        _require_sha256(self.fingerprint_sha256, label="fingerprint_sha256")
        expected = identity_sha256({"schema_version": self.schema_version, "fields": self.fields})
        if self.fingerprint_sha256 != expected:
            raise ValueError("Arrow schema fingerprint does not match its declaration")


def _arrow_field_identity(field: pa.Field) -> ArrowFieldIdentity:
    dtype = field.type
    children: tuple[ArrowFieldIdentity, ...] = ()
    if pa.types.is_struct(dtype) or pa.types.is_union(dtype):
        children = tuple(
            _arrow_field_identity(dtype.field(index)) for index in range(dtype.num_fields)
        )
    elif (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_fixed_size_list(dtype)
    ):
        children = (_arrow_field_identity(dtype.value_field),)
    elif pa.types.is_map(dtype):
        children = (
            _arrow_field_identity(dtype.key_field),
            _arrow_field_identity(dtype.item_field),
        )
    return ArrowFieldIdentity(
        name=field.name,
        type=str(dtype),
        nullable=field.nullable,
        children=children,
    )


def arrow_schema_identity(schema: pa.Schema, *, schema_version: int) -> ArrowSchemaIdentity:
    """Build a fingerprint from the actual ordered Arrow fields."""

    fields = tuple(_arrow_field_identity(field) for field in schema)
    fingerprint = identity_sha256({"schema_version": schema_version, "fields": fields})
    return ArrowSchemaIdentity(
        schema_version=schema_version,
        fields=fields,
        fingerprint_sha256=fingerprint,
    )


def parquet_schema_identity(path: Path | str, *, schema_version: int) -> ArrowSchemaIdentity:
    """Read only Parquet metadata and fingerprint its actual Arrow schema."""

    from farkle.utils.authentication_telemetry import record_schema_validation

    record_schema_validation()
    schema = retry_transient_io(lambda: pq.read_schema(path))
    return arrow_schema_identity(schema, schema_version=schema_version)


def _extract_required_path(payload: Mapping[str, Any], dotted_path: str) -> Any:
    cursor: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(cursor, Mapping):
            raise ValueError(f"stage config field {dotted_path!r} is absent")
        if part in cursor:
            cursor = cursor[part]
            continue
        numeric_part = int(part) if part.isdecimal() else None
        if numeric_part is None or numeric_part not in cursor:
            raise ValueError(f"stage config field {dotted_path!r} is absent")
        cursor = cursor[numeric_part]
    return cursor


@dataclass(frozen=True, slots=True)
class StageConfigIdentity:
    """Hash of an explicit allowlist of public, stage-affecting configuration."""

    stage_key: str
    field_paths: tuple[str, ...]
    selected_config: Mapping[str, Any]
    sha256: str

    def __post_init__(self) -> None:
        if not self.stage_key.strip() or not self.field_paths:
            raise ValueError("stage config identity requires a stage and non-empty field allowlist")
        if tuple(sorted(set(self.field_paths))) != self.field_paths:
            raise ValueError("stage config field_paths must be sorted and unique")
        if any(part.startswith("_") for path in self.field_paths for part in path.split(".")):
            raise ValueError("private configuration fields are not public stage identity")
        expected = identity_sha256(
            {
                "stage_key": self.stage_key,
                "field_paths": self.field_paths,
                "selected_config": self.selected_config,
            }
        )
        if self.sha256 != expected:
            raise ValueError("stage configuration digest does not match its selected fields")


def stage_config_identity(
    cfg: AppConfig, *, stage_key: str, field_paths: Sequence[str]
) -> StageConfigIdentity:
    """Create a stage identity from an explicit public-config field allowlist."""

    paths = tuple(sorted(set(field_paths)))
    if any(part.startswith("_") for path in paths for part in path.split(".")):
        raise ValueError("private configuration fields are not public stage identity")
    public = effective_config_dict(cfg)
    selected = {path: _extract_required_path(public, path) for path in paths}
    payload = {"stage_key": stage_key, "field_paths": paths, "selected_config": selected}
    return StageConfigIdentity(
        stage_key=stage_key,
        field_paths=paths,
        selected_config=selected,
        sha256=identity_sha256(payload),
    )


class CodeIdentityPolicy(str, Enum):
    """Permitted Git identity policies."""

    RELEASE_CLEAN = "release_clean"
    DEVELOPMENT_DIRTY = "development_dirty"


@dataclass(frozen=True, slots=True)
class CodeIdentity:
    """Git commit plus explicit clean/dirty identity policy."""

    commit: str
    policy: str
    state: str
    dirty_fingerprint_sha256: str | None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.commit):
            raise ValueError("code commit must be a full lowercase 40-character Git SHA")
        CodeIdentityPolicy(self.policy)
        if self.state not in {"clean", "development_dirty"}:
            raise ValueError("unsupported code identity state")
        if self.state == "clean" and self.dirty_fingerprint_sha256 is not None:
            raise ValueError("clean code identity cannot have a dirty fingerprint")
        if self.state == "development_dirty":
            if self.dirty_fingerprint_sha256 is None:
                raise ValueError("dirty code identity requires a deterministic fingerprint")
            _require_sha256(self.dirty_fingerprint_sha256, label="dirty_fingerprint_sha256")


def _git(repo_root: Path, *args: str, text: bool = False) -> bytes | str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CodeIdentityError(f"unable to determine Git code identity: {exc}") from exc
    return result.stdout


def resolve_code_identity(
    repo_root: Path | str,
    *,
    policy: CodeIdentityPolicy,
    untracked_inventory: Sequence[Path | str] = ("src", "tests", "configs", "pyproject.toml"),
) -> CodeIdentity:
    """Resolve a clean identity or warn and fingerprint a dirty worktree."""

    requested_root = Path(repo_root).resolve()
    top_level_raw = cast(str, _git(requested_root, "rev-parse", "--show-toplevel", text=True))
    top_level = Path(top_level_raw.strip()).resolve()
    commit = cast(str, _git(top_level, "rev-parse", "HEAD", text=True)).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise CodeIdentityError("Git did not return a full 40-character commit SHA")
    status = cast(bytes, _git(top_level, "status", "--porcelain=v1", "-z", "--untracked-files=all"))
    if not status:
        return CodeIdentity(
            commit=commit,
            policy=policy.value,
            state="clean",
            dirty_fingerprint_sha256=None,
        )
    if policy is CodeIdentityPolicy.RELEASE_CLEAN:
        LOGGER.warning(
            "release-preferred execution is using a dirty Git worktree; outputs are not release eligible",
            extra={"stage": "code_identity", "policy": policy.value},
        )

    digest = hashlib.sha256()
    digest.update(b"tracked-index\0")
    digest.update(cast(bytes, _git(top_level, "diff", "--cached", "--binary", "--no-ext-diff")))
    digest.update(b"tracked-worktree\0")
    digest.update(cast(bytes, _git(top_level, "diff", "--binary", "--no-ext-diff")))
    inventory_roots = tuple((top_level / Path(item)).resolve() for item in untracked_inventory)
    untracked_raw = cast(
        bytes,
        _git(top_level, "ls-files", "--others", "--exclude-standard", "-z"),
    )
    untracked = sorted(path for path in untracked_raw.split(b"\0") if path)
    for relative_raw in untracked:
        relative = Path(os.fsdecode(relative_raw))
        absolute = (top_level / relative).resolve()
        if not any(absolute == root or root in absolute.parents for root in inventory_roots):
            continue
        digest.update(b"untracked\0")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(absolute).encode("ascii"))
    return CodeIdentity(
        commit=commit,
        policy=policy.value,
        state="development_dirty",
        dirty_fingerprint_sha256=digest.hexdigest(),
    )


def producer_code_identity(cfg: AppConfig) -> CodeIdentity:
    """Use the invocation's Git guard, including for a standalone producer."""
    return cfg._code_identity or resolve_code_identity(
        Path(__file__).resolve().parents[3], policy=CodeIdentityPolicy.DEVELOPMENT_DIRTY
    )


@dataclass(frozen=True, slots=True)
class VersionIdentity:
    """All version dimensions that can make scientific artifacts incompatible."""

    artifact_contract_version: int
    lifecycle_contract_version: int
    rng_scheme_version: int
    outcome_schema_version: int
    schema_version: int
    estimand_version: int
    conditioning_version: int
    method_versions: Mapping[str, int]

    def __post_init__(self) -> None:
        versions = {
            "artifact_contract_version": self.artifact_contract_version,
            "lifecycle_contract_version": self.lifecycle_contract_version,
            "rng_scheme_version": self.rng_scheme_version,
            "outcome_schema_version": self.outcome_schema_version,
            "schema_version": self.schema_version,
            "estimand_version": self.estimand_version,
            "conditioning_version": self.conditioning_version,
            **dict(self.method_versions),
        }
        if any(isinstance(value, bool) or value < 1 for value in versions.values()):
            raise ValueError("all provenance versions must be positive integers")
        if self.artifact_contract_version != ARTIFACT_CONTRACT_VERSION:
            raise ValueError(f"artifact contract must be version {ARTIFACT_CONTRACT_VERSION}")
        if self.lifecycle_contract_version != LIFECYCLE_CONTRACT_VERSION:
            raise ValueError(f"lifecycle contract must be version {LIFECYCLE_CONTRACT_VERSION}")
        if not self.method_versions:
            raise ValueError("at least one named method version is required")


@dataclass(frozen=True, slots=True)
class MethodContract:
    """Typed statistical/method identity; no untyped parameter bag is accepted."""

    procedure: str
    method_version: int
    baseline: str
    replication_unit: str
    k_weights: tuple[tuple[int, float], ...] | None = None
    multiplicity: str | None = None
    family_hash: str | None = None
    schedule_hash: str | None = None
    practical_margin: float | None = None
    equivalence_margin: float | None = None
    ordinary_alpha: float | None = None
    simultaneous_alpha: float | None = None
    conditioning: str = "unconditional"
    source_scope: str | None = None
    root_seeds: tuple[int, ...] = ()
    player_counts: tuple[int, ...] = ()
    required_player_counts: tuple[int, ...] = ()
    weighted_quantity: str = "none"
    support_count_role: str = "raw_support_provenance"
    uncertainty_method: str = "none"
    k_aggregation_method: str = "none"
    missing_cell_policy: str = "not_applicable"
    seed_scope: str = "single_root"
    consistency_columns: tuple[str, ...] = ()
    grouping_keys: tuple[str, ...] = ()
    semantic_contract_sha256: str | None = None
    rng_effective_matchup_group_cap: int | None = None
    rng_diagnostic_lags: tuple[int, ...] = ()
    rng_tracked_matchup_group_count: int | None = None
    rng_skipped_matchup_group_count: int | None = None
    rng_skipped_matchup_row_count: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "procedure",
            "baseline",
            "replication_unit",
            "conditioning",
            "weighted_quantity",
            "support_count_role",
            "uncertainty_method",
            "k_aggregation_method",
            "missing_cell_policy",
            "seed_scope",
        ):
            if not cast(str, getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-blank")
        if self.method_version < 1:
            raise ValueError("method_version must be positive")
        for name in ("family_hash", "schedule_hash"):
            value = cast(str | None, getattr(self, name))
            if value is not None:
                _require_sha256(value, label=name)
        if self.k_weights is not None:
            keys = [key for key, _ in self.k_weights]
            values = [value for _, value in self.k_weights]
            if keys != sorted(set(keys)) or any(key < 1 for key in keys):
                raise ValueError("k_weights keys must be sorted unique positive player counts")
            if any(value <= 0.0 or not math.isfinite(value) for value in values):
                raise ValueError("k_weights values must be finite and positive")
            if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("k_weights must sum to one")
        for name in ("practical_margin", "equivalence_margin"):
            margin_value = cast(float | None, getattr(self, name))
            if margin_value is not None and (not math.isfinite(margin_value) or margin_value < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("ordinary_alpha", "simultaneous_alpha"):
            alpha_value = cast(float | None, getattr(self, name))
            if alpha_value is not None and (
                not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0
            ):
                raise ValueError(f"{name} must be strictly between zero and one")
        for name in ("root_seeds", "player_counts", "required_player_counts"):
            coordinate_values = cast(tuple[int, ...], getattr(self, name))
            if coordinate_values != tuple(sorted(set(coordinate_values))):
                raise ValueError(f"{name} must be sorted and unique")
            if name != "root_seeds" and any(value < 1 for value in coordinate_values):
                raise ValueError(f"{name} must contain positive values")
        if not set(self.required_player_counts).issubset(self.player_counts):
            raise ValueError("required_player_counts must be present in player_counts")
        if self.semantic_contract_sha256 is not None:
            _require_sha256(
                self.semantic_contract_sha256,
                label="semantic_contract_sha256",
            )
        rng_counts = (
            self.rng_tracked_matchup_group_count,
            self.rng_skipped_matchup_group_count,
            self.rng_skipped_matchup_row_count,
        )
        if any(value is not None and value < 0 for value in rng_counts):
            raise ValueError("RNG diagnostic capacity counts must be nonnegative")
        if self.rng_effective_matchup_group_cap is not None and (
            self.rng_effective_matchup_group_cap < 1
        ):
            raise ValueError("RNG diagnostic effective matchup-group cap must be positive")
        if self.rng_diagnostic_lags and (
            tuple(sorted(set(self.rng_diagnostic_lags))) != self.rng_diagnostic_lags
            or any(lag < 1 for lag in self.rng_diagnostic_lags)
        ):
            raise ValueError("RNG diagnostic lags must be unique increasing positive integers")
        if self.procedure == "semantic_coordinate_lag_correlation" and (
            self.rng_effective_matchup_group_cap is None
            or not self.rng_diagnostic_lags
            or any(value is None for value in rng_counts)
        ):
            raise ValueError("RNG diagnostic method metadata requires cap, lags, and group counts")

    @property
    def sha256(self) -> str:
        return identity_sha256(self)


@dataclass(frozen=True, slots=True)
class StageIdentity:
    """Canonical stage freshness identity from semantic dependencies only."""

    stage_key: str
    stage_cache_key_version: int
    stage_config: StageConfigIdentity
    versions: VersionIdentity
    code: CodeIdentity
    method_contract_sha256: str
    upstream_identity_sha256: tuple[str, ...]
    immutable_design_identities: Mapping[str, str]
    sha256: str

    def __post_init__(self) -> None:
        if self.stage_key != self.stage_config.stage_key:
            raise ValueError("stage identity and config identity keys differ")
        if self.stage_cache_key_version < 1:
            raise ValueError("stage_cache_key_version must be positive")
        _require_sha256(self.method_contract_sha256, label="method_contract_sha256")
        for value in self.upstream_identity_sha256:
            _require_sha256(value, label="upstream identity")
        for name, value in self.immutable_design_identities.items():
            if not name.strip():
                raise ValueError("immutable design identity names must be non-blank")
            _require_sha256(value, label=f"immutable design identity {name}")
        expected = identity_sha256(self._payload_without_digest())
        if self.sha256 != expected:
            raise ValueError("stage identity digest does not match its contract")

    def _payload_without_digest(self) -> Mapping[str, Any]:
        return {
            "lifecycle_contract_version": self.versions.lifecycle_contract_version,
            "stage_key": self.stage_key,
            "stage_cache_key_version": self.stage_cache_key_version,
            "stage_config_identity": self.stage_config,
            "versions": self.versions,
            "code_identity": self.code,
            "method_contract_sha256": self.method_contract_sha256,
            "upstream_identities": self.upstream_identity_sha256,
            "immutable_design_identities": self.immutable_design_identities,
        }


def make_stage_identity(
    *,
    stage_key: str,
    stage_cache_key_version: int,
    stage_config: StageConfigIdentity,
    versions: VersionIdentity,
    code: CodeIdentity,
    method_contract: MethodContract,
    upstream_identity_sha256: Sequence[str],
    immutable_design_identities: Mapping[str, str],
) -> StageIdentity:
    """Build the contract's canonical stage identity."""

    if method_contract.method_version not in versions.method_versions.values():
        raise ValueError("method contract version is absent from version identity")
    for parameter_name in ("family_hash", "schedule_hash"):
        parameter_value = cast(str | None, getattr(method_contract, parameter_name))
        if (
            parameter_value is not None
            and immutable_design_identities.get(parameter_name) != parameter_value
        ):
            raise ValueError(f"method {parameter_name} must equal the immutable design identity")
    provisional = {
        "lifecycle_contract_version": versions.lifecycle_contract_version,
        "stage_key": stage_key,
        "stage_cache_key_version": stage_cache_key_version,
        "stage_config_identity": stage_config,
        "versions": versions,
        "code_identity": code,
        "method_contract_sha256": method_contract.sha256,
        "upstream_identities": tuple(upstream_identity_sha256),
        "immutable_design_identities": dict(immutable_design_identities),
    }
    return StageIdentity(
        stage_key=stage_key,
        stage_cache_key_version=stage_cache_key_version,
        stage_config=stage_config,
        versions=versions,
        code=code,
        method_contract_sha256=method_contract.sha256,
        upstream_identity_sha256=tuple(upstream_identity_sha256),
        immutable_design_identities=dict(immutable_design_identities),
        sha256=identity_sha256(provisional),
    )


@dataclass(frozen=True, slots=True)
class ArtifactFormatIdentity:
    """Exact non-Parquet format identity without a fictitious Arrow schema."""

    media_type: str
    format_version: int
    structural_schema_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.media_type.strip():
            raise ValueError("artifact media_type must be non-blank")
        if self.format_version < 1:
            raise ValueError("artifact format_version must be positive")
        if self.structural_schema_sha256 is not None:
            _require_sha256(
                self.structural_schema_sha256,
                label="structural_schema_sha256",
            )


def _json_structural_shape(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_structural_shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_json_structural_shape(value[0])] if value else []
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    raise ArtifactMismatchError(f"unsupported JSON structural value: {type(value).__name__}")


def file_format_identity(
    path: Path | str,
    *,
    media_type: str,
    format_version: int = 1,
) -> ArtifactFormatIdentity:
    """Inspect non-Parquet bytes and return their actual typed format identity."""

    artifact = Path(path)
    structural: str | None = None
    try:
        if media_type == "application/json":
            structural = identity_sha256(
                _json_structural_shape(json.loads(artifact.read_text(encoding="utf-8")))
            )
        elif media_type == "application/x-ndjson":
            text = artifact.read_text(encoding="utf-8")
            try:
                shapes = {
                    json.dumps(
                        _json_structural_shape(json.loads(text)),
                        sort_keys=True,
                    )
                }
            except json.JSONDecodeError:
                shapes = {
                    json.dumps(
                        _json_structural_shape(json.loads(line)),
                        sort_keys=True,
                    )
                    for line in text.splitlines()
                    if line.strip()
                }
            structural = identity_sha256(sorted(shapes))
        elif media_type.startswith("text/") or media_type == "application/yaml":
            artifact.read_text(encoding="utf-8")
        elif media_type == "image/png" and not artifact.read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            raise ArtifactMismatchError(f"invalid PNG artifact: {artifact}")
        elif media_type == "application/x-npy":
            array = np.load(artifact, mmap_mode="r", allow_pickle=False)
            structural = identity_sha256({"dtype": array.dtype.descr, "shape": list(array.shape)})
            del array
    except (OSError, TypeError, UnicodeError, ValueError) as exc:
        raise ArtifactMismatchError(
            f"artifact does not satisfy declared format {media_type}: {artifact}"
        ) from exc
    return ArtifactFormatIdentity(
        media_type=media_type,
        format_version=format_version,
        structural_schema_sha256=structural,
    )


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Exact physical, byte, schema, and logical-operation identity."""

    location: CanonicalArtifactLocation
    byte_length: int
    content_sha256: str
    arrow_schema: ArrowSchemaIdentity | None
    logical_operation: str
    format_identity: "ArtifactFormatIdentity | None" = None

    def __post_init__(self) -> None:
        if self.byte_length <= 0:
            raise ValueError("artifact byte_length must be positive")
        _require_sha256(self.content_sha256, label="content_sha256")
        if not self.logical_operation.strip():
            raise ValueError("logical_operation must be non-blank")
        if (self.arrow_schema is None) == (self.format_identity is None):
            raise ValueError(
                "artifact identity requires exactly one Arrow schema or non-Parquet format identity"
            )

    @property
    def sha256(self) -> str:
        return identity_sha256(self)


@dataclass(frozen=True, slots=True)
class SourceArtifactIdentity:
    """Ordinary upstream artifact bound to exact data and sidecar bytes."""

    logical_role: str
    artifact: ArtifactIdentity
    sidecar_sha256: str
    sidecar_contract_sha256: str

    def __post_init__(self) -> None:
        if not self.logical_role.strip():
            raise ValueError("source logical_role must be non-blank")
        _require_sha256(self.sidecar_sha256, label="source sidecar_sha256")
        _require_sha256(self.sidecar_contract_sha256, label="source sidecar_contract_sha256")

    @property
    def sha256(self) -> str:
        return identity_sha256(self)


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One coordinate-sorted entry in an immutable sharded-input manifest."""

    coordinate: tuple[int | str, ...]
    canonical_relative_path: str
    data_sha256: str
    sidecar_sha256: str
    schema_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if not self.coordinate:
            raise ValueError("manifest coordinate must be non-empty")
        if any(isinstance(value, bool) for value in self.coordinate):
            raise ValueError("manifest coordinates do not accept booleans")
        relative = Path(self.canonical_relative_path)
        if not self.canonical_relative_path or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest path must be canonical and relative")
        for name in ("data_sha256", "sidecar_sha256", "schema_fingerprint_sha256"):
            _require_sha256(cast(str, getattr(self, name)), label=name)


@dataclass(frozen=True, slots=True)
class ManifestRootSummary:
    """Streaming root over a canonical, coordinate-sorted shard manifest."""

    root_sha256: str
    coordinate_support_sha256: str
    entry_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.root_sha256, label="root_sha256")
        _require_sha256(self.coordinate_support_sha256, label="coordinate_support_sha256")
        if self.entry_count < 0:
            raise ValueError("manifest entry_count must be nonnegative")


def compute_manifest_root(entries: Iterable[ManifestEntry]) -> ManifestRootSummary:
    """Hash canonical entries once without retaining the sharded inventory in RAM."""

    root = hashlib.sha256()
    support = hashlib.sha256()
    previous_order_key: tuple[tuple[int, int | str], ...] | None = None
    count = 0
    for entry in entries:
        coordinate_key = canonical_json_bytes(entry.coordinate)
        order_key = _manifest_coordinate_order_key(entry.coordinate)
        if previous_order_key is not None and order_key <= previous_order_key:
            raise ValueError("manifest entries must have strictly increasing coordinates")
        encoded = canonical_json_bytes(entry)
        root.update(len(encoded).to_bytes(8, "big"))
        root.update(encoded)
        support.update(len(coordinate_key).to_bytes(8, "big"))
        support.update(coordinate_key)
        previous_order_key = order_key
        count += 1
    return ManifestRootSummary(
        root_sha256=root.hexdigest(),
        coordinate_support_sha256=support.hexdigest(),
        entry_count=count,
    )


def _manifest_coordinate_order_key(
    coordinate: tuple[int | str, ...],
) -> tuple[tuple[int, int | str], ...]:
    """Use numeric integer order and lexical string order without lossy coercion."""

    return tuple((0, value) if isinstance(value, int) else (1, value) for value in coordinate)


@dataclass(frozen=True, slots=True)
class ManifestRootIdentity:
    """Authenticated small manifest root used instead of rehashing every shard."""

    logical_role: str
    location: CanonicalArtifactLocation
    manifest_sha256: str
    sidecar_sha256: str
    sidecar_contract_sha256: str
    summary: ManifestRootSummary

    def __post_init__(self) -> None:
        if not self.logical_role.strip():
            raise ValueError("manifest identity requires a logical role")
        for name in ("manifest_sha256", "sidecar_sha256", "sidecar_contract_sha256"):
            _require_sha256(cast(str, getattr(self, name)), label=name)

    @property
    def sha256(self) -> str:
        return identity_sha256(self)


@dataclass(frozen=True, slots=True)
class ImmutableManifestSidecar:
    """Small authenticated sidecar for a canonical immutable shard manifest."""

    artifact_contract_version: int
    manifest_contract_version: int
    location: CanonicalArtifactLocation
    manifest_sha256: str
    summary: ManifestRootSummary
    stage_identity: StageIdentity
    sidecar_contract_sha256: str

    def __post_init__(self) -> None:
        if self.artifact_contract_version != ARTIFACT_CONTRACT_VERSION:
            raise ValueError("unsupported artifact contract version")
        if self.manifest_contract_version != MANIFEST_CONTRACT_VERSION:
            raise ValueError("unsupported manifest contract version")
        _require_sha256(self.manifest_sha256, label="manifest_sha256")
        expected = identity_sha256(self._payload_without_digest())
        if self.sidecar_contract_sha256 != expected:
            raise ValueError("manifest sidecar digest does not match its payload")

    def _payload_without_digest(self) -> Mapping[str, Any]:
        return {
            "artifact_contract_version": self.artifact_contract_version,
            "manifest_contract_version": self.manifest_contract_version,
            "location": self.location,
            "manifest_sha256": self.manifest_sha256,
            "summary": self.summary,
            "stage_identity": self.stage_identity,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self) + b"\n"


@dataclass(frozen=True, slots=True)
class AuthenticatedSidecar:
    """Version-3 sidecar binding output, stage, method, code, and inputs."""

    artifact_contract_version: int
    artifact: ArtifactIdentity
    stage_identity: StageIdentity
    method_contract: MethodContract
    versions: VersionIdentity
    source_artifacts: tuple[SourceArtifactIdentity, ...]
    manifest_roots: tuple[ManifestRootIdentity, ...]
    sidecar_contract_sha256: str

    def __post_init__(self) -> None:
        if self.artifact_contract_version != ARTIFACT_CONTRACT_VERSION:
            raise ValueError(f"sidecar contract must be version {ARTIFACT_CONTRACT_VERSION}")
        if self.versions != self.stage_identity.versions:
            raise ValueError("sidecar and stage version identities differ")
        if self.method_contract.sha256 != self.stage_identity.method_contract_sha256:
            raise ValueError("sidecar method contract is not bound by stage identity")
        roles = [source.logical_role for source in self.source_artifacts]
        roles.extend(manifest.logical_role for manifest in self.manifest_roots)
        if len(roles) != len(set(roles)):
            raise ValueError("source and manifest logical roles must be unique")
        if roles != sorted(roles):
            raise ValueError("source and manifest logical roles must be in canonical order")
        expected_upstream = tuple(
            [source.sha256 for source in self.source_artifacts]
            + [manifest.sha256 for manifest in self.manifest_roots]
        )
        if expected_upstream != self.stage_identity.upstream_identity_sha256:
            raise ValueError("stage upstream identities do not match sidecar sources")
        expected = identity_sha256(self._payload_without_digest())
        if self.sidecar_contract_sha256 != expected:
            raise ValueError("sidecar contract digest does not match its payload")

    def _payload_without_digest(self) -> Mapping[str, Any]:
        return {
            "artifact_contract_version": self.artifact_contract_version,
            "artifact": self.artifact,
            "stage_identity": self.stage_identity,
            "method_contract": self.method_contract,
            "versions": self.versions,
            "source_artifacts": self.source_artifacts,
            "manifest_roots": self.manifest_roots,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self) + b"\n"


def make_authenticated_sidecar(
    *,
    artifact: ArtifactIdentity,
    stage_identity: StageIdentity,
    method_contract: MethodContract,
    sources: Sequence[SourceArtifactIdentity] = (),
    manifest_roots: Sequence[ManifestRootIdentity] = (),
) -> AuthenticatedSidecar:
    """Construct a fully bound version-3 sidecar."""

    payload = {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "artifact": artifact,
        "stage_identity": stage_identity,
        "method_contract": method_contract,
        "versions": stage_identity.versions,
        "source_artifacts": tuple(sources),
        "manifest_roots": tuple(manifest_roots),
    }
    return AuthenticatedSidecar(
        artifact_contract_version=ARTIFACT_CONTRACT_VERSION,
        artifact=artifact,
        stage_identity=stage_identity,
        method_contract=method_contract,
        versions=stage_identity.versions,
        source_artifacts=tuple(sources),
        manifest_roots=tuple(manifest_roots),
        sidecar_contract_sha256=identity_sha256(payload),
    )


def _construct(cls: type[_T], payload: Mapping[str, Any], **nested: Any) -> _T:
    names = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    if set(payload) != names:
        missing = sorted(names.difference(payload))
        extra = sorted(set(payload).difference(names))
        raise ValueError(f"invalid {cls.__name__} fields; missing={missing}, extra={extra}")
    values = dict(payload)
    values.update(nested)
    return cls(**values)


def _parse_location(payload: Mapping[str, Any]) -> CanonicalArtifactLocation:
    return _construct(CanonicalArtifactLocation, payload)


def _parse_field(payload: Mapping[str, Any]) -> ArrowFieldIdentity:
    children = tuple(_parse_field(item) for item in payload["children"])
    return _construct(ArrowFieldIdentity, payload, children=children)


def _parse_schema(payload: Mapping[str, Any]) -> ArrowSchemaIdentity:
    fields = tuple(_parse_field(item) for item in payload["fields"])
    return _construct(ArrowSchemaIdentity, payload, fields=fields)


def _parse_artifact(payload: Mapping[str, Any]) -> ArtifactIdentity:
    arrow_payload = payload["arrow_schema"]
    format_payload = payload.get("format_identity")
    return _construct(
        ArtifactIdentity,
        payload,
        location=_parse_location(payload["location"]),
        arrow_schema=None if arrow_payload is None else _parse_schema(arrow_payload),
        format_identity=(
            None if format_payload is None else _construct(ArtifactFormatIdentity, format_payload)
        ),
    )


def _parse_stage_config(payload: Mapping[str, Any]) -> StageConfigIdentity:
    return _construct(
        StageConfigIdentity,
        payload,
        field_paths=tuple(payload["field_paths"]),
    )


def _parse_code(payload: Mapping[str, Any]) -> CodeIdentity:
    return _construct(CodeIdentity, payload)


def _parse_versions(payload: Mapping[str, Any]) -> VersionIdentity:
    return _construct(VersionIdentity, payload)


def _parse_method(payload: Mapping[str, Any]) -> MethodContract:
    weights = payload["k_weights"]
    return _construct(
        MethodContract,
        payload,
        k_weights=None if weights is None else tuple((int(k), float(v)) for k, v in weights),
        root_seeds=tuple(int(value) for value in payload["root_seeds"]),
        player_counts=tuple(int(value) for value in payload["player_counts"]),
        required_player_counts=tuple(int(value) for value in payload["required_player_counts"]),
        consistency_columns=tuple(str(value) for value in payload["consistency_columns"]),
        grouping_keys=tuple(str(value) for value in payload["grouping_keys"]),
        rng_diagnostic_lags=tuple(int(value) for value in payload.get("rng_diagnostic_lags", ())),
    )


def _parse_stage(payload: Mapping[str, Any]) -> StageIdentity:
    return _construct(
        StageIdentity,
        payload,
        stage_config=_parse_stage_config(payload["stage_config"]),
        versions=_parse_versions(payload["versions"]),
        code=_parse_code(payload["code"]),
        upstream_identity_sha256=tuple(payload["upstream_identity_sha256"]),
    )


def _parse_source(payload: Mapping[str, Any]) -> SourceArtifactIdentity:
    return _construct(
        SourceArtifactIdentity,
        payload,
        artifact=_parse_artifact(payload["artifact"]),
    )


def _parse_manifest(payload: Mapping[str, Any]) -> ManifestRootIdentity:
    return _construct(
        ManifestRootIdentity,
        payload,
        location=_parse_location(payload["location"]),
        summary=_construct(ManifestRootSummary, payload["summary"]),
    )


def _parse_immutable_manifest_sidecar(
    payload: Mapping[str, Any],
) -> ImmutableManifestSidecar:
    return _construct(
        ImmutableManifestSidecar,
        payload,
        location=_parse_location(payload["location"]),
        summary=_construct(ManifestRootSummary, payload["summary"]),
        stage_identity=_parse_stage(payload["stage_identity"]),
    )


def _parse_sidecar(payload: Mapping[str, Any]) -> AuthenticatedSidecar:
    return _construct(
        AuthenticatedSidecar,
        payload,
        artifact=_parse_artifact(payload["artifact"]),
        stage_identity=_parse_stage(payload["stage_identity"]),
        method_contract=_parse_method(payload["method_contract"]),
        versions=_parse_versions(payload["versions"]),
        source_artifacts=tuple(_parse_source(item) for item in payload["source_artifacts"]),
        manifest_roots=tuple(_parse_manifest(item) for item in payload["manifest_roots"]),
    )


def load_authenticated_sidecar(path: Path | str) -> AuthenticatedSidecar:
    """Load a sidecar, distinguishing absence from present corruption."""

    artifact = Path(path)
    metadata_path = sidecar_path(artifact)
    if not metadata_path.exists():
        raise MissingSidecarError(f"missing sidecar for {artifact}")
    try:
        from farkle.utils.authentication_telemetry import record_metadata_validation

        payload = read_json_file_with_retry(metadata_path)
        record_metadata_validation()
        if not isinstance(payload, Mapping):
            raise TypeError("sidecar root must be an object")
        return _parse_sidecar(payload)
    except MissingSidecarError:
        raise
    except Exception as exc:
        raise CorruptSidecarError(f"corrupt sidecar {metadata_path}: {exc}") from exc


def load_immutable_manifest_sidecar(path: Path | str) -> ImmutableManifestSidecar:
    """Load and internally validate an immutable-manifest sidecar."""

    manifest_path = Path(path)
    metadata_path = sidecar_path(manifest_path)
    if not metadata_path.exists():
        raise MissingSidecarError(f"missing manifest sidecar for {manifest_path}")
    try:
        from farkle.utils.authentication_telemetry import record_metadata_validation

        payload = read_json_file_with_retry(metadata_path)
        record_metadata_validation()
        if not isinstance(payload, Mapping):
            raise TypeError("manifest sidecar root must be an object")
        return _parse_immutable_manifest_sidecar(payload)
    except Exception as exc:
        raise CorruptSidecarError(f"corrupt manifest sidecar {metadata_path}: {exc}") from exc


def _current_artifact_identity(
    path: Path,
    *,
    location: CanonicalArtifactLocation,
    schema_version: int,
    logical_operation: str,
    format_identity: ArtifactFormatIdentity | None = None,
) -> ArtifactIdentity:
    try:
        byte_length, content_hash = retry_transient_io(
            lambda: (path.stat().st_size, sha256_file(path))
        )
        if format_identity is None:
            schema = parquet_schema_identity(path, schema_version=schema_version)
            current_format = None
        else:
            schema = None
            current_format = file_format_identity(
                path,
                media_type=format_identity.media_type,
                format_version=format_identity.format_version,
            )
    except (OSError, pa.ArrowException) as exc:
        raise ArtifactMismatchError(f"artifact cannot be authenticated: {path}: {exc}") from exc
    return ArtifactIdentity(
        location=location,
        byte_length=byte_length,
        content_sha256=content_hash,
        arrow_schema=schema,
        logical_operation=logical_operation,
        format_identity=current_format,
    )


def validate_authenticated_artifact(
    path: Path | str,
    *,
    cfg: AppConfig,
    expected_location: CanonicalArtifactLocation,
    expected_stage_identity: StageIdentity | None = None,
    expected_method_contract: MethodContract | None = None,
    expected_versions: VersionIdentity | None = None,
    expected_sidecar_sha256: str | None = None,
    source_paths: Mapping[str, Path] | None = None,
    source_configs: Mapping[str, AppConfig] | None = None,
    manifest_paths: Mapping[str, tuple[Path, Path]] | None = None,
    manifest_configs: Mapping[str, AppConfig] | None = None,
) -> AuthenticatedSidecar:
    """Fail closed unless path, bytes, schema, provenance, and inputs all match."""

    artifact_path = expected_location.require_path(cfg, path)
    metadata = load_authenticated_sidecar(artifact_path)
    metadata_path = sidecar_path(artifact_path)
    if expected_sidecar_sha256 is not None:
        _require_sha256(expected_sidecar_sha256, label="expected_sidecar_sha256")
        if sha256_file(metadata_path) != expected_sidecar_sha256:
            raise ArtifactMismatchError("exact sidecar hash does not match completion identity")
    if metadata.artifact.location != expected_location:
        raise ArtifactMismatchError("sidecar declares a different canonical artifact location")
    current = _current_artifact_identity(
        artifact_path,
        location=expected_location,
        schema_version=(
            metadata.versions.schema_version
            if metadata.artifact.arrow_schema is None
            else metadata.artifact.arrow_schema.schema_version
        ),
        logical_operation=metadata.artifact.logical_operation,
        format_identity=metadata.artifact.format_identity,
    )
    if current != metadata.artifact:
        raise ArtifactMismatchError("artifact bytes or actual Arrow schema do not match sidecar")
    if (
        expected_stage_identity is not None
        and metadata.stage_identity.sha256 != expected_stage_identity.sha256
    ):
        raise ArtifactMismatchError("stage identity does not match")
    if (
        expected_method_contract is not None
        and metadata.method_contract != expected_method_contract
    ):
        raise ArtifactMismatchError("method contract does not match")
    if expected_versions is not None and metadata.versions != expected_versions:
        raise ArtifactMismatchError("method/RNG/outcome/schema version identity does not match")

    if metadata.source_artifacts:
        source_roles = {source.logical_role for source in metadata.source_artifacts}
        if (
            source_paths is None
            or source_configs is None
            or set(source_paths) != source_roles
            or set(source_configs) != source_roles
        ):
            raise ArtifactMismatchError(
                "all declared source artifact roles require exact paths and owning configs"
            )
        for source in metadata.source_artifacts:
            source_path = source_paths[source.logical_role]
            source_cfg = source_configs[source.logical_role]
            source.artifact.location.require_path(source_cfg, source_path)
            try:
                current_source = _current_artifact_identity(
                    source_path,
                    location=source.artifact.location,
                    schema_version=(
                        DERIVED_SCHEMA_VERSION
                        if source.artifact.arrow_schema is None
                        else source.artifact.arrow_schema.schema_version
                    ),
                    logical_operation=source.artifact.logical_operation,
                    format_identity=source.artifact.format_identity,
                )
            except ArtifactMismatchError as exc:
                raise ArtifactMismatchError(
                    f"source artifact bytes/schema changed: {source.logical_role}"
                ) from exc
            if current_source != source.artifact:
                raise ArtifactMismatchError(
                    f"source artifact bytes/schema changed: {source.logical_role}"
                )
            source_sidecar = sidecar_path(source_path)
            if not source_sidecar.exists():
                raise MissingSidecarError(f"missing source sidecar: {source.logical_role}")
            if sha256_file(source_sidecar) != source.sidecar_sha256:
                raise ArtifactMismatchError(f"source sidecar changed: {source.logical_role}")
            loaded_source = load_authenticated_sidecar(source_path)
            if loaded_source.sidecar_contract_sha256 != source.sidecar_contract_sha256:
                raise ArtifactMismatchError(
                    f"source sidecar contract changed: {source.logical_role}"
                )

    if metadata.manifest_roots:
        manifest_roles = {manifest.logical_role for manifest in metadata.manifest_roots}
        if (
            manifest_paths is None
            or manifest_configs is None
            or set(manifest_paths) != manifest_roles
            or set(manifest_configs) != manifest_roles
        ):
            raise ArtifactMismatchError(
                "all declared immutable manifest roles require exact paths and owning configs"
            )
        for manifest in metadata.manifest_roots:
            manifest_path, manifest_sidecar = manifest_paths[manifest.logical_role]
            manifest.location.require_path(manifest_configs[manifest.logical_role], manifest_path)
            if manifest_sidecar.resolve() != sidecar_path(manifest_path).resolve():
                raise ArtifactMismatchError(
                    f"manifest sidecar is not adjacent: {manifest.logical_role}"
                )
            if sha256_file(manifest_path) != manifest.manifest_sha256:
                raise ArtifactMismatchError(f"manifest bytes changed: {manifest.logical_role}")
            if sha256_file(manifest_sidecar) != manifest.sidecar_sha256:
                raise ArtifactMismatchError(f"manifest sidecar changed: {manifest.logical_role}")
            loaded_manifest = load_immutable_manifest_sidecar(manifest_path)
            if (
                loaded_manifest.sidecar_contract_sha256 != manifest.sidecar_contract_sha256
                or loaded_manifest.summary != manifest.summary
                or loaded_manifest.location != manifest.location
            ):
                raise ArtifactMismatchError(
                    f"manifest root contract changed: {manifest.logical_role}"
                )
    return metadata


def capture_source_artifact(
    path: Path | str,
    *,
    cfg: AppConfig,
    expected_location: CanonicalArtifactLocation,
    logical_role: str,
    source_paths: Mapping[str, Path] | None = None,
    source_configs: Mapping[str, AppConfig] | None = None,
    manifest_paths: Mapping[str, tuple[Path, Path]] | None = None,
    manifest_configs: Mapping[str, AppConfig] | None = None,
) -> SourceArtifactIdentity:
    """Capture a validated source's exact data, sidecar, and contract identities."""

    metadata = validate_authenticated_artifact(
        path,
        cfg=cfg,
        expected_location=expected_location,
        source_paths=source_paths,
        source_configs=source_configs,
        manifest_paths=manifest_paths,
        manifest_configs=manifest_configs,
    )
    return SourceArtifactIdentity(
        logical_role=logical_role,
        artifact=metadata.artifact,
        sidecar_sha256=sha256_file(sidecar_path(path)),
        sidecar_contract_sha256=metadata.sidecar_contract_sha256,
    )


def _require_unbound_canonical_suffix(
    path: Path,
    location: CanonicalArtifactLocation,
) -> None:
    """Validate a canonical location when the owning config is an upstream run."""

    relative_parts = Path(location.relative_path).parts
    if location.stage_key == "simulation":
        expected_suffix = relative_parts
    elif location.scope == ArtifactScope.BY_K.value:
        assert location.player_count is not None
        expected_suffix = (
            location.scope,
            f"{location.player_count}p",
            *relative_parts,
        )
    else:
        expected_suffix = (location.scope, *relative_parts)
    actual_parts = path.resolve().parts
    if tuple(actual_parts[-len(expected_suffix) :]) != tuple(expected_suffix):
        raise ArtifactMismatchError(
            f"artifact path {path} does not realize canonical location {location}"
        )


def _verify_artifact_bytes(path: Path, metadata: AuthenticatedSidecar) -> None:
    _require_unbound_canonical_suffix(path, metadata.artifact.location)
    _verify_artifact_content(path, metadata)


def _verify_artifact_content(path: Path, metadata: AuthenticatedSidecar) -> None:
    """Check bytes and schema, including a prior generation in its rollback slot."""
    current = _current_artifact_identity(
        path,
        location=metadata.artifact.location,
        schema_version=(
            metadata.versions.schema_version
            if metadata.artifact.arrow_schema is None
            else metadata.artifact.arrow_schema.schema_version
        ),
        logical_operation=metadata.artifact.logical_operation,
        format_identity=metadata.artifact.format_identity,
    )
    if current != metadata.artifact:
        raise ArtifactMismatchError("artifact bytes or actual schema do not match sidecar")


_VerifyBytes = Callable[[Path, AuthenticatedSidecar], None]
_STAGE_PARQUET_CHECK: ContextVar[_VerifyBytes | None] = ContextVar(
    "stage_parquet_check", default=None
)
_MAX_RETAINED_PARQUET_BYTES = 64 * 1024


@contextmanager
def reuse_stage_parquet_facts() -> Iterator[None]:
    """Share pure byte computations during one parent's stage resolutions.

    Every lookup includes a complete fresh read. Publication, replacement or
    rebinding can never carry a path verdict forward. Only immutable Python
    bytes, their digest and their versioned Arrow schema live across checks.
    """
    from farkle.utils.authentication_telemetry import (
        record_authentication_hash,
        record_authentication_open,
        record_schema_validation,
    )

    owner_pid = os.getpid()
    owner_thread = threading.current_thread()
    active = True

    @lru_cache(maxsize=32)
    def facts(path: Path, contents: bytes, version: int) -> tuple[str, ArrowSchemaIdentity]:
        digest = hashlib.sha256(contents).hexdigest()
        record_authentication_hash(path, len(contents))
        record_schema_validation()
        schema = pq.read_schema(pa.BufferReader(contents))
        return digest, arrow_schema_identity(schema, schema_version=version)

    def verify(path: Path, metadata: AuthenticatedSidecar) -> None:
        artifact = metadata.artifact
        if (
            not active
            or owner_pid != os.getpid()
            or threading.current_thread() is not owner_thread
            or artifact.arrow_schema is None
            or artifact.byte_length > _MAX_RETAINED_PARQUET_BYTES
        ):
            _verify_artifact_bytes(path, metadata)
            return
        _require_unbound_canonical_suffix(path, artifact.location)

        def read() -> bytes:
            record_authentication_open(path)
            with path.open("rb") as handle:
                return handle.read(artifact.byte_length + 1)

        try:
            contents = retry_transient_io(read)
            if len(contents) != artifact.byte_length:
                raise ArtifactMismatchError(f"artifact byte length differs: {path}")
            digest, schema = facts(path, contents, artifact.arrow_schema.schema_version)
        except (OSError, pa.ArrowException) as exc:
            raise ArtifactMismatchError(f"artifact cannot be authenticated: {path}: {exc}") from exc
        if digest != artifact.content_sha256 or schema != artifact.arrow_schema:
            raise ArtifactMismatchError(f"artifact bytes or actual schema do not match: {path}")

    token = _STAGE_PARQUET_CHECK.set(verify)
    try:
        yield
    finally:
        active = False
        facts.cache_clear()
        _STAGE_PARQUET_CHECK.reset(token)


def validate_authenticated_artifact_unbound(
    path: Path | str,
    *,
    expected_location: CanonicalArtifactLocation | None = None,
    expected_method_contract: MethodContract | None = None,
    expected_versions: VersionIdentity | None = None,
    source_paths: Mapping[str, Path] | None = None,
    manifest_paths: Mapping[str, tuple[Path, Path]] | None = None,
    validate_provenance: bool = True,
) -> AuthenticatedSidecar:
    """Validate exact v3 bytes/schema and logical path without an owning config.

    Pair stages use this only for artifacts owned by separately authenticated
    root run contexts.  Publication still records path-independent canonical
    locations; this helper never infers a new identity from a physical path.
    """

    artifact_path = Path(path)
    metadata = load_authenticated_sidecar(artifact_path)
    location = expected_location or metadata.artifact.location
    if metadata.artifact.location != location:
        raise ArtifactMismatchError("sidecar declares a different canonical artifact location")
    _verify_artifact_bytes(artifact_path, metadata)
    if (
        expected_method_contract is not None
        and metadata.method_contract != expected_method_contract
    ):
        raise ArtifactMismatchError("method contract does not match")
    if expected_versions is not None and metadata.versions != expected_versions:
        raise ArtifactMismatchError("method/RNG/outcome/schema version identity does not match")
    if validate_provenance:
        _validate_unbound_dependencies(metadata, source_paths, manifest_paths)
    return metadata


def _validate_unbound_dependencies(
    metadata: AuthenticatedSidecar,
    source_paths: Mapping[str, Path] | None,
    manifest_paths: Mapping[str, tuple[Path, Path]] | None,
    verification: _CompletionVerification | None = None,
) -> None:
    source_roles = {source.logical_role for source in metadata.source_artifacts}
    if source_roles:
        if source_paths is None or set(source_paths) != source_roles:
            raise ArtifactMismatchError("all declared source artifact roles require exact paths")
        capture_source = (
            capture_source_artifact_unbound if verification is None else verification.source
        )
        for source in metadata.source_artifacts:
            current_source = capture_source(
                source_paths[source.logical_role],
                logical_role=source.logical_role,
            )
            if current_source != source:
                raise ArtifactMismatchError(
                    f"source artifact bytes or sidecar changed: {source.logical_role}"
                )
    manifest_roles = {manifest.logical_role for manifest in metadata.manifest_roots}
    if manifest_roles:
        if manifest_paths is None or set(manifest_paths) != manifest_roles:
            raise ArtifactMismatchError("all declared immutable manifest roles require exact paths")
        for manifest in metadata.manifest_roots:
            manifest_path, manifest_sidecar_path = manifest_paths[manifest.logical_role]
            _require_unbound_canonical_suffix(manifest_path, manifest.location)
            if manifest_sidecar_path.resolve() != sidecar_path(manifest_path).resolve():
                raise ArtifactMismatchError(
                    f"manifest sidecar is not adjacent: {manifest.logical_role}"
                )
            if verification is None:
                loaded_manifest: AuthenticatedSidecar | ImmutableManifestSidecar | None = (
                    load_immutable_manifest_sidecar(manifest_path)
                )
                sidecar_sha = None
                manifest_sha = sha256_file(manifest_path)
            else:
                loaded_manifest, sidecar_sha = verification.sidecar(manifest_path)
                manifest_sha = verification.manifest_sha256(manifest_path)
            if (
                not isinstance(loaded_manifest, ImmutableManifestSidecar)
                or manifest_sha != manifest.manifest_sha256
                or (sidecar_sha or sha256_file(manifest_sidecar_path)) != manifest.sidecar_sha256
                or loaded_manifest.sidecar_contract_sha256 != manifest.sidecar_contract_sha256
                or loaded_manifest.summary != manifest.summary
                or loaded_manifest.location != manifest.location
                or loaded_manifest.manifest_sha256 != manifest.manifest_sha256
            ):
                raise ArtifactMismatchError(f"manifest root changed: {manifest.logical_role}")


def validate_authenticated_artifact_metadata(
    path: Path | str,
    *,
    cfg: AppConfig | None = None,
    expected_location: CanonicalArtifactLocation | None = None,
    expected_stage_identity: StageIdentity | None = None,
    expected_versions: VersionIdentity | None = None,
    expected_sidecar_sha256: str | None = None,
) -> AuthenticatedSidecar:
    """Validate immutable publication metadata without rereading artifact bytes.

    This is the routine-resume seam for producer-owned immutable outputs.  The
    sidecar remains the authoritative exact-byte identity; this check verifies
    its canonical location, exact sidecar bytes, declared byte length, and
    current code/config identity.  A deep audit must use
    :func:`validate_authenticated_artifact_unbound`, which additionally
    rehashes the complete artifact and reopens its schema.
    """

    artifact_path = Path(path)
    verification = _CompletionVerification()
    try:
        verification.watch(sidecar_path(artifact_path))
    except FileNotFoundError as exc:
        raise MissingSidecarError(f"missing sidecar for {artifact_path}") from exc
    metadata = load_authenticated_sidecar(artifact_path)
    location = metadata.artifact.location
    if expected_location is not None and location != expected_location:
        raise ArtifactMismatchError("sidecar declares a different canonical artifact location")
    if cfg is not None:
        location.require_path(cfg, artifact_path)
    else:
        _require_unbound_canonical_suffix(artifact_path, location)
    try:
        byte_length = verification.watch(artifact_path).st_size
    except OSError as exc:
        raise ArtifactMismatchError(f"artifact cannot be inspected: {artifact_path}") from exc
    if byte_length != metadata.artifact.byte_length:
        raise ArtifactMismatchError("artifact byte length does not match sidecar")
    if expected_sidecar_sha256 is not None:
        _require_sha256(expected_sidecar_sha256, label="expected_sidecar_sha256")
        if sha256_file(sidecar_path(artifact_path)) != expected_sidecar_sha256:
            raise ArtifactMismatchError("exact sidecar hash does not match completion identity")
    if (
        expected_stage_identity is not None
        and metadata.stage_identity.sha256 != expected_stage_identity.sha256
    ):
        raise ArtifactMismatchError("stage identity does not match")
    if expected_versions is not None and metadata.versions != expected_versions:
        raise ArtifactMismatchError("method/RNG/outcome/schema version identity does not match")
    verification.require_unchanged()
    return metadata


def capture_source_artifact_unbound(
    path: Path | str,
    *,
    logical_role: str,
) -> SourceArtifactIdentity:
    """Capture an exact source owned by another authenticated run context."""

    metadata = validate_authenticated_artifact_unbound(path, validate_provenance=False)
    return SourceArtifactIdentity(
        logical_role=logical_role,
        artifact=metadata.artifact,
        sidecar_sha256=sha256_file(sidecar_path(path)),
        sidecar_contract_sha256=metadata.sidecar_contract_sha256,
    )


def capture_manifest_root(
    *,
    logical_role: str,
    manifest_path: Path,
    cfg: AppConfig,
    expected_location: CanonicalArtifactLocation,
    expected_stage_identity: StageIdentity | None = None,
) -> ManifestRootIdentity:
    """Bind a small authenticated manifest root without rehashing its shards."""

    expected_location.require_path(cfg, manifest_path)
    metadata = load_immutable_manifest_sidecar(manifest_path)
    if metadata.location != expected_location:
        raise ArtifactMismatchError("manifest sidecar declares a different canonical location")
    if (
        expected_stage_identity is not None
        and metadata.stage_identity.sha256 != expected_stage_identity.sha256
    ):
        raise ArtifactMismatchError("manifest stage identity does not match")
    if sha256_file(manifest_path) != metadata.manifest_sha256:
        raise ArtifactMismatchError("manifest bytes do not match its sidecar")
    manifest_sidecar_path = sidecar_path(manifest_path)
    return ManifestRootIdentity(
        logical_role=logical_role,
        location=expected_location,
        manifest_sha256=metadata.manifest_sha256,
        sidecar_sha256=sha256_file(manifest_sidecar_path),
        sidecar_contract_sha256=metadata.sidecar_contract_sha256,
        summary=metadata.summary,
    )


@contextmanager
def _destination_candidate(staged: Path, destination: Path) -> Iterator[Path]:
    """Validate the actual installable file, including the copy from external scratch."""
    if staged.stat().st_dev == destination.parent.stat().st_dev:
        yield staged
        return

    fd, local = make_staged_path(destination, prefix="._publish_")
    try:
        with (
            os.fdopen(fd, "wb") as target,
            _protected_publication_files([staged]) as check_source,
            staged.open("rb") as source,
        ):
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
            check_source()
        yield local
        # Preserve the staged-publisher API: a successful install consumes the
        # caller's stage. Frozen reuse leaves it for the caller's normal cleanup.
        if not local.exists():
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Could not remove consumed staging file %s", staged, exc_info=True)
    finally:
        local.unlink(missing_ok=True)


def _install_manifest(
    staged_data: Path,
    staged_sidecar: Path,
    final_path: Path,
    metadata: ImmutableManifestSidecar,
    sources: Sequence[SourceArtifactIdentity],
    paths: Mapping[str, Path],
    controls: Mapping[str, Path],
) -> ImmutableManifestSidecar:
    from farkle.utils.frozen_h2h import same_production
    from farkle.utils.manifest import publication_lock
    from farkle.utils.scientific_invocation import publication_transaction
    from farkle.utils.scientific_reuse import immutable_input_error

    if (
        set(paths) != {source.logical_role for source in sources}
        or tuple(source.sha256 for source in sources)
        != metadata.stage_identity.upstream_identity_sha256
    ):
        raise ArtifactMismatchError("manifest stage upstream identities differ from its inputs")
    inputs = [p for path in paths.values() for p in (path, sidecar_path(path))]
    seal = sidecar_path(final_path)
    simulation = metadata.location.stage_key == "simulation"
    with (
        _destination_candidate(staged_data, final_path) as local_data,
        _protected_publication_files([*inputs, *controls.values()]) as check_inputs,
        publication_lock(final_path),
    ):
        _recover_previous_pair(final_path)
        with _protected_publication_files([local_data, staged_sidecar]) as check_stages:
            staged_metadata = _parse_immutable_manifest_sidecar(
                read_json_file_with_retry(staged_sidecar)
            )
            if staged_metadata.sidecar_contract_sha256 != metadata.sidecar_contract_sha256:
                raise ArtifactMismatchError("staged manifest sidecar changed")
            if sha256_file(local_data) != metadata.manifest_sha256:
                raise ArtifactMismatchError("staged manifest data changed")
            _validate_control_inputs(metadata.stage_identity, controls)
            for source in sources:
                if (
                    capture_source_artifact_unbound(
                        paths[source.logical_role], logical_role=source.logical_role
                    )
                    != source
                ):
                    raise ArtifactMismatchError(f"manifest source changed: {source.logical_role}")
            previous = (
                load_immutable_manifest_sidecar(final_path)
                if simulation and seal.exists()
                else None
            )
            if previous is not None and (
                previous.location != metadata.location
                or previous.summary != metadata.summary
                or previous.manifest_sha256 != metadata.manifest_sha256
                or sha256_file(final_path) != previous.manifest_sha256
                or not same_production(previous.stage_identity, metadata.stage_identity)
            ):
                raise immutable_input_error(final_path)
            check_stages()
            check_inputs()
        with publication_transaction():
            if previous is not None:
                return previous
            if simulation:
                publish_staged_file_atomic(local_data, final_path)
                check_inputs()
                _publish_staged_file_atomic(staged_sidecar, seal, replace=_install_once)
            else:
                _replace_pair(local_data, staged_sidecar, final_path, check_inputs=check_inputs)
    return metadata


def publish_immutable_manifest_atomic(
    path: Path | str,
    *,
    cfg: AppConfig,
    location: CanonicalArtifactLocation,
    stage_identity: StageIdentity,
    entries: Iterable[ManifestEntry],
    sources: Sequence[SourceArtifactIdentity] = (),
    source_paths: Mapping[str, Path] | None = None,
    control_paths: Mapping[str, Path] | None = None,
) -> ImmutableManifestSidecar:
    """Stream and atomically publish an immutable canonical manifest and root."""

    final_path = location.require_path(cfg, path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    data_fd, staged_data = make_staged_path(
        final_path, prefix="._manifest_v3_", temp_dir=cfg.io.temp_dir
    )
    os.close(data_fd)
    sidecar_fd, staged_sidecar = make_staged_path(
        final_path,
        prefix="._manifest_sidecar_v3_",
    )
    os.close(sidecar_fd)
    try:
        root = hashlib.sha256()
        support = hashlib.sha256()
        previous_order_key: tuple[tuple[int, int | str], ...] | None = None
        count = 0
        with staged_data.open("wb") as handle:
            for entry in entries:
                coordinate_key = canonical_json_bytes(entry.coordinate)
                order_key = _manifest_coordinate_order_key(entry.coordinate)
                if previous_order_key is not None and order_key <= previous_order_key:
                    raise ValueError("manifest entries must have strictly increasing coordinates")
                encoded = canonical_json_bytes(entry)
                handle.write(encoded + b"\n")
                root.update(len(encoded).to_bytes(8, "big"))
                root.update(encoded)
                support.update(len(coordinate_key).to_bytes(8, "big"))
                support.update(coordinate_key)
                previous_order_key = order_key
                count += 1
        if count == 0:
            raise ValueError("immutable manifest must contain at least one entry")
        summary = ManifestRootSummary(
            root_sha256=root.hexdigest(),
            coordinate_support_sha256=support.hexdigest(),
            entry_count=count,
        )
        payload = {
            "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            "manifest_contract_version": MANIFEST_CONTRACT_VERSION,
            "location": location,
            "manifest_sha256": sha256_file(staged_data),
            "summary": summary,
            "stage_identity": stage_identity,
        }
        metadata = ImmutableManifestSidecar(
            artifact_contract_version=ARTIFACT_CONTRACT_VERSION,
            manifest_contract_version=MANIFEST_CONTRACT_VERSION,
            location=location,
            manifest_sha256=cast(str, payload["manifest_sha256"]),
            summary=summary,
            stage_identity=stage_identity,
            sidecar_contract_sha256=identity_sha256(payload),
        )
        staged_sidecar.write_bytes(metadata.canonical_bytes())
        return _install_manifest(
            staged_data,
            staged_sidecar,
            final_path,
            metadata,
            sources,
            source_paths or {},
            control_paths or {},
        )
    finally:
        staged_data.unlink(missing_ok=True)
        staged_sidecar.unlink(missing_ok=True)


def publish_immutable_manifest_bytes_atomic(
    path: Path | str,
    *,
    cfg: AppConfig,
    location: CanonicalArtifactLocation,
    stage_identity: StageIdentity,
    entries: Iterable[ManifestEntry],
    write_data: Callable[[Path], None],
    sources: Sequence[SourceArtifactIdentity] = (),
    source_paths: Mapping[str, Path] | None = None,
    control_paths: Mapping[str, Path] | None = None,
) -> ImmutableManifestSidecar:
    """Publish an immutable native manifest plus a typed coordinate root.

    The native bytes remain readable by the stage-specific streaming consumer;
    the adjacent v3 sidecar binds their exact hash and a separately canonical
    coordinate-sorted shard root.
    """

    final_path = location.require_path(cfg, path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    data_fd, staged_data = make_staged_path(
        final_path, prefix="._manifest_v3_", temp_dir=cfg.io.temp_dir
    )
    os.close(data_fd)
    sidecar_fd, staged_sidecar = make_staged_path(
        final_path,
        prefix="._manifest_sidecar_v3_",
    )
    os.close(sidecar_fd)
    try:
        write_data(staged_data)
        if not staged_data.is_file() or staged_data.stat().st_size <= 0:
            raise ArtifactMismatchError("manifest writer produced no bytes")
        summary = compute_manifest_root(entries)
        if summary.entry_count == 0:
            raise ValueError("immutable manifest must contain at least one entry")
        payload = {
            "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            "manifest_contract_version": MANIFEST_CONTRACT_VERSION,
            "location": location,
            "manifest_sha256": sha256_file(staged_data),
            "summary": summary,
            "stage_identity": stage_identity,
        }
        metadata = ImmutableManifestSidecar(
            artifact_contract_version=ARTIFACT_CONTRACT_VERSION,
            manifest_contract_version=MANIFEST_CONTRACT_VERSION,
            location=location,
            manifest_sha256=cast(str, payload["manifest_sha256"]),
            summary=summary,
            stage_identity=stage_identity,
            sidecar_contract_sha256=identity_sha256(payload),
        )
        staged_sidecar.write_bytes(metadata.canonical_bytes())
        return _install_manifest(
            staged_data,
            staged_sidecar,
            final_path,
            metadata,
            sources,
            source_paths or {},
            control_paths or {},
        )
    finally:
        staged_data.unlink(missing_ok=True)
        staged_sidecar.unlink(missing_ok=True)


def publish_authenticated_parquet_atomic(
    path: Path | str,
    *,
    cfg: AppConfig,
    location: CanonicalArtifactLocation,
    stage_identity: StageIdentity,
    method_contract: MethodContract,
    write_data: Callable[[Path], None],
    sources: Sequence[SourceArtifactIdentity] = (),
    manifest_roots: Sequence[ManifestRootIdentity] = (),
    source_paths: Mapping[str, Path] | None = None,
    source_configs: Mapping[str, AppConfig] | None = None,
    manifest_paths: Mapping[str, tuple[Path, Path]] | None = None,
    manifest_configs: Mapping[str, AppConfig] | None = None,
) -> AuthenticatedSidecar:
    """Publish Parquet then its v3 sidecar atomically with bounded I/O retries."""

    final_path = location.require_path(cfg, path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    data_fd, staged_data = make_staged_path(
        final_path, prefix="._artifact_v3_", temp_dir=cfg.io.temp_dir
    )
    os.close(data_fd)
    try:
        write_data(staged_data)
        return publish_staged_authenticated_artifact_atomic(
            staged_data,
            final_path,
            cfg=cfg,
            location=location,
            stage_identity=stage_identity,
            method_contract=method_contract,
            sources=sources,
            manifest_roots=manifest_roots,
            source_paths=source_paths,
            source_configs=source_configs,
            manifest_paths=manifest_paths,
            manifest_configs=manifest_configs,
        )
    finally:
        staged_data.unlink(missing_ok=True)


def publish_authenticated_file_atomic(
    path: Path | str,
    *,
    cfg: AppConfig,
    location: CanonicalArtifactLocation,
    stage_identity: StageIdentity,
    method_contract: MethodContract,
    format_identity: ArtifactFormatIdentity,
    write_data: Callable[[Path], None],
    sources: Sequence[SourceArtifactIdentity] = (),
    manifest_roots: Sequence[ManifestRootIdentity] = (),
    source_paths: Mapping[str, Path] | None = None,
    source_configs: Mapping[str, AppConfig] | None = None,
    manifest_paths: Mapping[str, tuple[Path, Path]] | None = None,
    manifest_configs: Mapping[str, AppConfig] | None = None,
) -> AuthenticatedSidecar:
    """Publish a typed JSON/text/PNG/binary artifact and its v3 sidecar."""

    final_path = location.require_path(cfg, path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    data_fd, staged_data = make_staged_path(
        final_path, prefix="._artifact_v3_", temp_dir=cfg.io.temp_dir
    )
    os.close(data_fd)
    try:
        write_data(staged_data)
        return publish_staged_authenticated_artifact_atomic(
            staged_data,
            final_path,
            cfg=cfg,
            location=location,
            stage_identity=stage_identity,
            method_contract=method_contract,
            format_identity=format_identity,
            sources=sources,
            manifest_roots=manifest_roots,
            source_paths=source_paths,
            source_configs=source_configs,
            manifest_paths=manifest_paths,
            manifest_configs=manifest_configs,
        )
    finally:
        staged_data.unlink(missing_ok=True)


def _install_once(source: Path | str, destination: Path | str) -> None:
    """Atomically install a new file without replacing a racing publisher."""
    if os.name == "nt":
        os.rename(source, destination)
    else:
        os.link(source, destination)
        Path(source).unlink()


@contextmanager
def _protected_publication_files(paths: Iterable[Path]) -> Iterator[Callable[[], None]]:
    """Hold OS-backed protection for direct inputs until their dependent commit."""
    from farkle.utils.scientific_files import MeasuredFiles

    unique = set(paths)
    if not unique:
        yield lambda: None
        return
    with closing(MeasuredFiles()) as files:
        for path in sorted(unique):
            files.protect(path)
        # A Linux lease pins the inode. Rename-over can rebind the pathname
        # without opening the leased inode for write.
        bindings = {p: p.stat() for p in unique} if os.name != "nt" else {}

        def check() -> None:
            files.boundary()
            for path, before in bindings.items():
                if not os.path.samestat(before, path.stat()):
                    raise ArtifactMismatchError(f"Protected publication input was replaced: {path}")

        check()
        yield check


def _validate_control_inputs(stage: StageIdentity, paths: Mapping[str, Path]) -> None:
    expected = {
        k: v for k, v in stage.immutable_design_identities.items() if k.startswith("control:")
    }
    if set(paths) != set(expected):
        raise ArtifactMismatchError("all declared control roles require exact paths")
    for role, path in paths.items():
        if sha256_file(path) != expected[role]:
            raise ArtifactMismatchError(f"control identity changed: {role}")


@contextmanager
def _verified_publication_inputs(
    metadata: AuthenticatedSidecar,
    source_paths: Mapping[str, Path],
    manifest_paths: Mapping[str, tuple[Path, Path]],
    source_configs: Mapping[str, AppConfig] | None,
    manifest_configs: Mapping[str, AppConfig] | None,
    controls: Mapping[str, Path],
) -> Iterator[Callable[[], None]]:
    """Authenticate every declared role under the same protection as the commit."""
    sources = metadata.source_artifacts
    manifests = metadata.manifest_roots
    if set(source_paths) != {source.logical_role for source in sources} or (
        source_configs is not None and set(source_configs) != set(source_paths)
    ):
        raise ArtifactMismatchError("all declared source artifact roles require exact paths")
    if set(manifest_paths) != {manifest.logical_role for manifest in manifests} or (
        manifest_configs is not None and set(manifest_configs) != set(manifest_paths)
    ):
        raise ArtifactMismatchError("all declared immutable manifest roles require exact paths")
    paths = [p for path in source_paths.values() for p in (path, sidecar_path(path))]
    paths.extend(p for pair in manifest_paths.values() for p in pair)
    with _protected_publication_files([*paths, *controls.values()]) as check:
        if source_configs is not None:
            for source in sources:
                role = source.logical_role
                source.artifact.location.require_path(source_configs[role], source_paths[role])
        if manifest_configs is not None:
            for manifest in manifests:
                role = manifest.logical_role
                manifest.location.require_path(manifest_configs[role], manifest_paths[role][0])
        _validate_control_inputs(metadata.stage_identity, controls)
        _validate_unbound_dependencies(metadata, source_paths, manifest_paths)
        yield check


def _previous_data(path: Path) -> Path:
    return path.parent / "_execution" / ("._previous_" + path.name)


def _previous_unpaired_sidecar(path: Path) -> Path:
    return path.parent / "_execution" / ("._unpaired_" + sidecar_path(path).name)


def pending_publication_recovery(path: Path) -> bool:
    """Read-only probe of the A1 writer's reserved rollback slots."""
    prior = _previous_data(path)
    return any(
        os.path.lexists(p) for p in (prior, sidecar_path(prior), _previous_unpaired_sidecar(path))
    )


def restore_verified_publication(
    path: Path,
    *,
    check_path: Callable[[Path], object],
    check_owner: Callable[[AuthenticatedSidecar], None],
    sources: Sequence[Path],
) -> None:
    """Authenticate a withdrawn pair under the A1 writer lock before rollback."""
    from farkle.utils.manifest import publication_lock
    from farkle.utils.scientific_invocation import publication_transaction

    prior = _previous_data(path)
    prior_seal = sidecar_path(prior)
    seal = sidecar_path(path)
    unpaired = _previous_unpaired_sidecar(path)
    check_path(prior.parent / "._publication_.v3-artifacts.lock")
    with publication_lock(path.parent / ".v3-artifacts"):
        if not pending_publication_recovery(path):
            return
        inputs = [p for source in sources for p in (source, sidecar_path(source))]
        for member in (prior, prior_seal, seal, path, unpaired, *inputs):
            check_path(member)
        if unpaired.exists() or not prior_seal.is_file():
            raise ArtifactMismatchError(f"Ambiguous publication backup for {path}")
        with _protected_publication_files(inputs) as check_inputs:
            members = [p for p in (prior, prior_seal, path, seal) if p.exists()]
            with _protected_publication_files(members) as check_files:
                previous = load_authenticated_sidecar(prior)
                check_owner(previous)
                old_data = prior if prior.exists() else path
                _verify_artifact_content(old_data, previous)
                expected = previous
                committed = seal.exists()
                if committed:
                    current = validate_authenticated_artifact_unbound(
                        path, validate_provenance=False
                    )
                    check_owner(current)
                    expected = current
                elif (
                    prior.exists()
                    and path.exists()
                    and (digest := sha256_file(path)) != previous.artifact.content_sha256
                ):
                    # A1 leaves the prevalidated candidate seal beside the destination
                    # when the writer exits between renames. It can identify the
                    # displaced bytes, but must never be installed to complete them.
                    for stage in path.parent.glob("._sidecar_v3_*"):
                        check_path(stage)
                        try:
                            candidate = _parse_sidecar(read_json_file_with_retry(stage))
                        except (ValueError, TypeError, KeyError, AuthenticatedContractError):
                            continue
                        if (
                            candidate.artifact.content_sha256 == digest
                            and candidate.artifact.location == previous.artifact.location
                        ):
                            check_owner(candidate)
                            _verify_artifact_content(path, candidate)
                            break
                    else:
                        raise ArtifactMismatchError(
                            f"Unrecognized interrupted publication data at {path}"
                        )
                check_inputs()
                check_files()
            with publication_transaction():
                _recover_previous_pair(path)
            restored = validate_authenticated_artifact_unbound(path, validate_provenance=False)
            if restored != expected:
                raise ArtifactMismatchError(f"Recovered publication identity differs at {path}")


def recover_interrupted_publication(path: Path, *, manifest: bool = False) -> None:
    """Restore a withdrawn pair before a retry attempts to read scientific inputs."""
    from farkle.utils.manifest import publication_lock

    if sidecar_path(_previous_data(path)).exists() or _previous_unpaired_sidecar(path).exists():
        with publication_lock(path if manifest else path.parent / ".v3-artifacts"):
            _recover_previous_pair(path)


def _recover_previous_pair(path: Path) -> None:
    """Recover a process interruption under the writer lock. The sidecar commits last."""
    previous = _previous_data(path)
    seal = sidecar_path(path)
    previous_seal = sidecar_path(previous)
    unpaired_seal = _previous_unpaired_sidecar(path)
    if unpaired_seal.exists():
        if seal.exists():
            _discard_committed_backup(unpaired_seal)
        else:
            # The previous publication was already missing its data file.
            # Restore that absence before putting its sidecar back.
            path.unlink(missing_ok=True)
            replace_file_atomic(unpaired_seal, seal)
        return
    if not previous_seal.exists():
        return
    if not seal.exists():
        # The previous seal was withdrawn before moving its data. Either move
        # may have been interrupted, including a previous attempt to recover.
        if previous.exists():
            replace_file_atomic(previous, path)
        replace_file_atomic(previous_seal, seal)
    else:
        _discard_committed_backup(previous_seal, previous)


def _discard_committed_backup(seal: Path, data: Path | None = None) -> None:
    # Cleanup is repeatable. A sharing violation cannot reject a committed pair.
    try:
        if data is not None:
            data.unlink(missing_ok=True)
        seal.unlink(missing_ok=True)
    except OSError:
        LOGGER.warning("Retained publication backup at %s", seal, exc_info=True)


def _install_immutable_pair(
    staged: Path,
    staged_seal: Path,
    path: Path,
    metadata: AuthenticatedSidecar,
    check_inputs: Callable[[], None],
) -> None:
    from farkle.utils.scientific_reuse import immutable_input_error

    try:
        _publish_staged_file_atomic(staged, path, replace=_install_once)
    except FileExistsError:
        # The current producer must independently reproduce a data-only orphan.
        if (
            metadata.method_contract.procedure == "publish_strategy_manifest"
            or sha256_file(path) != metadata.artifact.content_sha256
        ):
            raise immutable_input_error(path) from None
    check_inputs()
    _publish_staged_file_atomic(staged_seal, sidecar_path(path), replace=_install_once)


def _replace_pair(
    staged: Path, staged_seal: Path, path: Path, *, check_inputs: Callable[[], None]
) -> None:
    """Replace a mutable publication under its lock, retaining the prior generation."""
    seal = sidecar_path(path)
    previous = _previous_data(path)
    previous_seal = sidecar_path(previous)
    unpaired_seal = _previous_unpaired_sidecar(path)
    if previous.exists() or previous_seal.exists() or unpaired_seal.exists():
        raise ArtifactMismatchError(f"Unresolved publication backup for {path}")
    had_seal = seal.exists()
    had_data = path.exists()
    withdrawn = previous_seal if had_data else unpaired_seal
    try:
        if had_seal:
            replace_file_atomic(seal, withdrawn)
            if had_data:
                replace_file_atomic(path, previous)
        publish_staged_file_atomic(staged, path)
        check_inputs()
        publish_staged_file_atomic(staged_seal, seal)
    except BaseException:
        if withdrawn.exists() or not had_seal:
            seal.unlink(missing_ok=True)
        _recover_previous_pair(path)
        raise
    if had_seal:
        _discard_committed_backup(withdrawn, previous if had_data else None)


def publish_staged_authenticated_artifact_atomic(
    staged_data: Path | str,
    path: Path | str,
    *,
    cfg: AppConfig,
    location: CanonicalArtifactLocation,
    stage_identity: StageIdentity,
    method_contract: MethodContract,
    format_identity: ArtifactFormatIdentity | None = None,
    sources: Sequence[SourceArtifactIdentity] = (),
    manifest_roots: Sequence[ManifestRootIdentity] = (),
    source_paths: Mapping[str, Path] | None = None,
    source_configs: Mapping[str, AppConfig] | None = None,
    manifest_paths: Mapping[str, tuple[Path, Path]] | None = None,
    manifest_configs: Mapping[str, AppConfig] | None = None,
    control_paths: Mapping[str, Path] | None = None,
    validate_unbound_sources: bool = False,
) -> AuthenticatedSidecar:
    """Publish an already staged v3 artifact without permitting sidecar backfill."""

    from farkle.utils.frozen_h2h import reuse_frozen_input
    from farkle.utils.manifest import publication_lock
    from farkle.utils.scientific_invocation import publication_transaction
    from farkle.utils.scientific_reuse import ROOT_INPUT_OPERATIONS

    # Retained for callers of the prior API. Dependency verification is unconditional.
    _ = validate_unbound_sources
    staged_path = Path(staged_data)
    final_path = location.require_path(cfg, path)
    recover_interrupted_publication(final_path)
    final_sidecar = sidecar_path(final_path)
    if staged_path.resolve() in {final_path.resolve(), final_sidecar.resolve()}:
        raise ArtifactMismatchError("publication requires separately staged data")
    if not staged_path.is_file() or staged_path.stat().st_size <= 0:
        raise ArtifactMismatchError(f"artifact writer did not create {staged_path}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    # Stage the small commit record on the destination volume. Its rename is
    # the last fallible installation, even when the data use external scratch.
    sidecar_fd, staged_sidecar = make_staged_path(final_path, prefix="._sidecar_v3_")
    os.close(sidecar_fd)
    try:
        with (
            _destination_candidate(staged_path, final_path) as local_data,
            ExitStack() as staged_guards,
        ):
            check_data = staged_guards.enter_context(_protected_publication_files([local_data]))
            artifact = _current_artifact_identity(
                local_data,
                location=location,
                schema_version=stage_identity.versions.schema_version,
                logical_operation=method_contract.procedure,
                format_identity=format_identity,
            )
            metadata = make_authenticated_sidecar(
                artifact=artifact,
                stage_identity=stage_identity,
                method_contract=method_contract,
                sources=sources,
                manifest_roots=manifest_roots,
            )
            staged_sidecar.write_bytes(metadata.canonical_bytes())
            check_seal = staged_guards.enter_context(_protected_publication_files([staged_sidecar]))
            staged_metadata = _parse_sidecar(read_json_file_with_retry(staged_sidecar))
            if staged_metadata.sidecar_contract_sha256 != metadata.sidecar_contract_sha256:
                raise ArtifactMismatchError("staged artifact sidecar changed")
            resolved_sources = dict(source_paths or {})
            resolved_manifests = dict(manifest_paths or {})
            # Share a directory lock so simulation shards do not each require a lock file.
            with (
                _verified_publication_inputs(
                    metadata,
                    resolved_sources,
                    resolved_manifests,
                    source_configs,
                    manifest_configs,
                    control_paths or {},
                ) as check_inputs,
                publication_lock(final_path.parent / ".v3-artifacts"),
            ):
                _recover_previous_pair(final_path)
                frozen = reuse_frozen_input(cfg, metadata)
                if frozen is not None:
                    validate_authenticated_artifact_unbound(
                        final_path,
                        expected_location=location,
                        source_paths=resolved_sources,
                        manifest_paths=resolved_manifests,
                    )
                check_inputs()
                check_data()
                check_seal()
                with publication_transaction():
                    staged_guards.close()  # Release only the private stages for renaming.
                    if frozen is not None:
                        return frozen
                    if method_contract.procedure in ROOT_INPUT_OPERATIONS:
                        _install_immutable_pair(
                            local_data, staged_sidecar, final_path, metadata, check_inputs
                        )
                    else:
                        _replace_pair(
                            local_data, staged_sidecar, final_path, check_inputs=check_inputs
                        )
        return metadata
    finally:
        staged_sidecar.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class CompletionOutputIdentity:
    """Completion-stamp binding for one exact output and sidecar."""

    artifact: ArtifactIdentity | None
    sidecar_sha256: str
    manifest: ManifestRootIdentity | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.sidecar_sha256, label="completion sidecar_sha256")
        if (self.artifact is None) == (self.manifest is None):
            raise ValueError("completion output requires exactly one artifact or manifest identity")

    @property
    def location(self) -> CanonicalArtifactLocation:
        if self.artifact is not None:
            return self.artifact.location
        assert self.manifest is not None
        return self.manifest.location


@dataclass(frozen=True, slots=True)
class AuthenticatedCompletion:
    """Minimal immutable completion identity for lifecycle classification."""

    lifecycle_contract_version: int
    stage_identity_sha256: str
    state: str
    outputs: tuple[CompletionOutputIdentity, ...]
    completion_producer_v1: CodeIdentity | None = None

    def __post_init__(self) -> None:
        if self.lifecycle_contract_version != LIFECYCLE_CONTRACT_VERSION:
            raise ValueError("unsupported lifecycle contract version")
        _require_sha256(self.stage_identity_sha256, label="stage_identity_sha256")
        if self.state not in {
            CompletionState.COMPLETE_VALID.value,
            CompletionState.BLOCKED_BY_CAP.value,
        }:
            raise ValueError("completion state must be complete_valid or blocked_by_cap")
        if self.state == CompletionState.COMPLETE_VALID.value and not self.outputs:
            raise ValueError("complete_valid requires at least one output")
        locations = [output.location for output in self.outputs]
        if len(locations) != len(set(locations)):
            raise ValueError("completion outputs must have unique canonical locations")
        if locations != sorted(locations, key=canonical_json_bytes):
            raise ValueError("completion outputs must be in canonical location order")


def write_authenticated_completion_atomic(
    path: Path,
    completion: AuthenticatedCompletion,
    *,
    temp_dir: Path | None = None,
) -> None:
    """Atomically publish an authenticated completion identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path), temp_dir=temp_dir) as temporary_path:
        payload = asdict(completion)
        if completion.completion_producer_v1 is None:
            payload.pop("completion_producer_v1")
        Path(temporary_path).write_bytes(canonical_json_bytes(payload) + b"\n")


def _load_completion(path: Path) -> AuthenticatedCompletion:
    payload = read_json_file_with_retry(path)
    if not isinstance(payload, Mapping):
        raise TypeError("completion root must be an object")
    outputs = tuple(
        _construct(
            CompletionOutputIdentity,
            item,
            artifact=(None if item["artifact"] is None else _parse_artifact(item["artifact"])),
            manifest=(None if item.get("manifest") is None else _parse_manifest(item["manifest"])),
        )
        for item in payload["outputs"]
    )
    code = (
        _parse_code(payload["completion_producer_v1"])
        if "completion_producer_v1" in payload
        else None
    )
    return _construct(
        AuthenticatedCompletion, {**payload, "completion_producer_v1": code}, outputs=outputs
    )


def load_authenticated_completion(path: Path | str) -> AuthenticatedCompletion:
    """Load and structurally validate one authenticated completion stamp."""

    return _load_completion(Path(path))


class _CompletionVerification:
    """Physical reads owned by one completion check. Binding checks stay per use.

    File metadata detects changes during the operation. Each first read still
    establishes the byte or metadata identity required by the caller.
    """

    def __init__(self, *, stage_resolution: bool = False) -> None:
        self._verify_bytes = (
            _STAGE_PARQUET_CHECK.get() if stage_resolution else None
        ) or _verify_artifact_bytes
        self._observed: dict[Path, os.stat_result] = {}
        self._sidecars: dict[
            Path, tuple[AuthenticatedSidecar | ImmutableManifestSidecar | None, str]
        ] = {}
        self._artifact_depth: dict[Path, bool] = {}
        self._manifest_hashes: dict[Path, str] = {}

    def watch(self, path: Path) -> os.stat_result:
        path = path.absolute()
        if path not in self._observed:
            self._observed[path] = retry_transient_io(path.stat)
        return self._observed[path]

    def require_unchanged(self) -> None:
        # Ignore access time, which can advance as a result of the reads.
        def version(stat: os.stat_result) -> tuple[int, ...]:
            return (
                stat.st_dev,
                stat.st_ino,
                stat.st_mode,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )

        for path, before in self._observed.items():
            after = retry_transient_io(path.stat)
            if version(before) != version(after):
                raise ArtifactMismatchError(f"file changed during completion validation: {path}")

    def sidecar(
        self, path: Path
    ) -> tuple[AuthenticatedSidecar | ImmutableManifestSidecar | None, str]:
        from farkle.utils.authentication_telemetry import record_metadata_validation

        adjacent = sidecar_path(path).absolute()
        if adjacent not in self._sidecars:
            try:
                self.watch(adjacent)
                payload, digest = read_json_file_with_sha256(adjacent)
                metadata: AuthenticatedSidecar | ImmutableManifestSidecar | None = None
                if isinstance(payload, dict) and payload.get("artifact_contract_version") == 3:
                    record_metadata_validation()
                    metadata = (
                        _parse_immutable_manifest_sidecar(payload)
                        if "manifest_contract_version" in payload
                        else _parse_sidecar(payload)
                    )
                self._sidecars[adjacent] = metadata, digest
            except FileNotFoundError as exc:
                raise MissingSidecarError(f"missing sidecar for {path}") from exc
            except Exception as exc:
                from farkle.utils.evidence_reader import EvidenceReadDeferred

                if isinstance(exc, EvidenceReadDeferred):
                    raise
                raise CorruptSidecarError(f"corrupt sidecar {adjacent}: {exc}") from exc
        return self._sidecars[adjacent]

    def artifact(self, path: Path, *, deep: bool) -> AuthenticatedSidecar:
        path = path.absolute()
        metadata, _ = self.sidecar(path)
        if not isinstance(metadata, AuthenticatedSidecar):
            raise ArtifactMismatchError(f"expected an authenticated artifact sidecar: {path}")
        previous_deep = self._artifact_depth.get(path)
        if previous_deep is None or (deep and not previous_deep):
            stat = self.watch(path)
            if deep:
                self._verify_bytes(path, metadata)
            else:
                _require_unbound_canonical_suffix(path, metadata.artifact.location)
                if stat.st_size != metadata.artifact.byte_length:
                    raise ArtifactMismatchError("artifact byte length does not match sidecar")
            self._artifact_depth[path] = deep
        return metadata

    def source(self, path: Path | str, *, logical_role: str) -> SourceArtifactIdentity:
        artifact_path = Path(path)
        metadata = self.artifact(artifact_path, deep=True)
        _, sidecar_sha = self.sidecar(artifact_path)
        return SourceArtifactIdentity(
            logical_role=logical_role,
            artifact=metadata.artifact,
            sidecar_sha256=sidecar_sha,
            sidecar_contract_sha256=metadata.sidecar_contract_sha256,
        )

    def manifest_sha256(self, path: Path) -> str:
        path = path.absolute()
        if path not in self._manifest_hashes:
            self.watch(path)
            self._manifest_hashes[path] = sha256_file(path)
        return self._manifest_hashes[path]


def classify_authenticated_lifecycle(
    completion_path: Path,
    *,
    cfg: AppConfig,
    expected_stage_identity: StageIdentity,
    required_locations: Sequence[CanonicalArtifactLocation],
    partial_paths: Sequence[Path] = (),
    source_paths: Mapping[str, Path] | None = None,
    source_configs: Mapping[str, AppConfig] | None = None,
    manifest_paths: Mapping[str, tuple[Path, Path]] | None = None,
    manifest_configs: Mapping[str, AppConfig] | None = None,
    deep_verify_artifacts: bool = True,
) -> CompletionState:
    """Classify authenticated work into exactly one canonical lifecycle state."""

    # Retained for API compatibility with callers that own same-run configs;
    # v3 source validation below is path-independent for cross-root consumers.
    _ = source_configs, manifest_configs
    materialized = any(location.path(cfg).exists() for location in required_locations) or any(
        path.exists() for path in partial_paths
    )
    if not completion_path.exists():
        return CompletionState.PARTIAL_RESUMABLE if materialized else CompletionState.NOT_STARTED
    verification = _CompletionVerification()
    try:
        verification.watch(completion_path)
        completion = _load_completion(completion_path)
    except Exception:
        return CompletionState.COMPLETE_STALE
    return _classify_loaded_completion(
        completion,
        cfg=cfg,
        expected_stage_identity=expected_stage_identity,
        required_locations=required_locations,
        source_paths=source_paths,
        manifest_paths=manifest_paths,
        deep_verify_artifacts=deep_verify_artifacts,
        verification=verification,
    )


def require_completion_output_metadata(
    output: CompletionOutputIdentity,
    metadata: AuthenticatedSidecar | ImmutableManifestSidecar | None,
    sidecar_sha256: str,
    *,
    stage_config: StageConfigIdentity,
) -> None:
    """Check the exact inventory binding shared by preview and lifecycle validation."""
    name = output.location.relative_path
    if metadata is None or sidecar_sha256 != output.sidecar_sha256:
        raise ArtifactMismatchError(f"completion sidecar identity differs for {name}")
    if metadata.stage_identity.stage_config.sha256 != stage_config.sha256:
        raise ArtifactMismatchError(f"scoped scientific configuration differs for {name}")
    if output.manifest is not None:
        if (
            not isinstance(metadata, ImmutableManifestSidecar)
            or metadata.location != output.location
            or output.manifest.manifest_sha256 != metadata.manifest_sha256
            or output.manifest.summary != metadata.summary
        ):
            raise ArtifactMismatchError(f"completion manifest identity differs for {name}")
    elif not isinstance(metadata, AuthenticatedSidecar) or metadata.artifact != output.artifact:
        raise ArtifactMismatchError(f"completion artifact identity differs for {name}")


def _classify_loaded_completion(
    completion: AuthenticatedCompletion,
    *,
    cfg: AppConfig,
    expected_stage_identity: StageIdentity,
    required_locations: Sequence[CanonicalArtifactLocation],
    source_paths: Mapping[str, Path] | None,
    manifest_paths: Mapping[str, tuple[Path, Path]] | None,
    deep_verify_artifacts: bool,
    verification: _CompletionVerification,
) -> CompletionState:
    from farkle.utils.scientific_reuse import compatible_producer, launch_matches

    current_code = expected_stage_identity.code
    original_code = completion.completion_producer_v1
    expected_digest = expected_stage_identity.sha256
    foreign = original_code is not None and original_code != current_code
    if foreign:
        if not completion.outputs:
            return CompletionState.COMPLETE_STALE
        # Reconstruct the original stamp preimage, including exact output sidecars.
        # Each output must independently establish its full scientific compatibility.
        expected_digest = identity_sha256(
            {**expected_stage_identity._payload_without_digest(), "code_identity": original_code}
        )
    if completion.stage_identity_sha256 != expected_digest:
        return CompletionState.COMPLETE_STALE
    try:
        if completion.state == CompletionState.BLOCKED_BY_CAP.value:
            verification.require_unchanged()
            return CompletionState.BLOCKED_BY_CAP
        expected_paths = {location.path(cfg).resolve(): location for location in required_locations}
        recorded = {output.location: output for output in completion.outputs}
        if set(recorded) != set(required_locations):
            return CompletionState.COMPLETE_STALE
        for path, location in expected_paths.items():
            output = recorded[location]
            metadata, sidecar_sha = verification.sidecar(path)
            require_completion_output_metadata(
                output, metadata, sidecar_sha, stage_config=expected_stage_identity.stage_config
            )
            if foreign and (metadata is None or not launch_matches(metadata.stage_identity)):
                return CompletionState.COMPLETE_STALE
            dependency = expected_stage_identity.immutable_design_identities.get(
                "stage_dependencies_v1"
            )
            if dependency is not None:
                from farkle.utils.scientific_identity import compatible_dependency_digests

                if metadata is None or not compatible_dependency_digests(
                    metadata.stage_identity.immutable_design_identities.get(
                        "stage_dependencies_v1"
                    ),
                    dependency,
                ):
                    return CompletionState.COMPLETE_STALE
            if output.manifest is not None:
                if (
                    not isinstance(metadata, ImmutableManifestSidecar)
                    or metadata.stage_identity.versions != expected_stage_identity.versions
                    or not compatible_producer(metadata.stage_identity, current_code)
                    or verification.manifest_sha256(path) != metadata.manifest_sha256
                ):
                    return CompletionState.COMPLETE_STALE
                continue
            assert output.artifact is not None
            location.require_path(cfg, path)
            if (
                not isinstance(metadata, AuthenticatedSidecar)
                or metadata.artifact.location != location
                or metadata.versions != expected_stage_identity.versions
                or not compatible_producer(metadata.stage_identity, current_code)
            ):
                return CompletionState.COMPLETE_STALE
            verification.artifact(path, deep=deep_verify_artifacts)
            if deep_verify_artifacts:
                _validate_unbound_dependencies(
                    metadata,
                    {
                        item.logical_role: source_paths[item.logical_role]
                        for item in metadata.source_artifacts
                        if source_paths is not None and item.logical_role in source_paths
                    },
                    {
                        item.logical_role: manifest_paths[item.logical_role]
                        for item in metadata.manifest_roots
                        if manifest_paths is not None and item.logical_role in manifest_paths
                    },
                    verification,
                )
        verification.require_unchanged()
    except (AuthenticatedContractError, OSError, TypeError, ValueError):
        return CompletionState.COMPLETE_STALE
    return CompletionState.COMPLETE_VALID


def finalize_missing_sidecar_atomic(
    path: Path | str,
    *,
    cfg: AppConfig,
    expected_sidecar: AuthenticatedSidecar,
    completion_output: CompletionOutputIdentity,
    source_paths: Mapping[str, Path] | None = None,
    source_configs: Mapping[str, AppConfig] | None = None,
    manifest_paths: Mapping[str, tuple[Path, Path]] | None = None,
    manifest_configs: Mapping[str, AppConfig] | None = None,
) -> AuthenticatedSidecar:
    """Finalize only a genuinely missing sidecar already bound by completion.

    A present sidecar is always validated and never replaced.  Missing metadata
    can be reconstructed only when both the exact artifact identity and the
    exact canonical sidecar bytes are already named by an independent
    completion output identity.
    """

    if completion_output.artifact is None:
        raise ArtifactMismatchError("artifact sidecar recovery requires an artifact completion")
    artifact_path = expected_sidecar.artifact.location.require_path(cfg, path)
    metadata_path = sidecar_path(artifact_path)
    if metadata_path.exists():
        return validate_authenticated_artifact(
            artifact_path,
            cfg=cfg,
            expected_location=expected_sidecar.artifact.location,
            expected_stage_identity=expected_sidecar.stage_identity,
            expected_method_contract=expected_sidecar.method_contract,
            expected_versions=expected_sidecar.versions,
            expected_sidecar_sha256=completion_output.sidecar_sha256,
            source_paths=source_paths,
            source_configs=source_configs,
            manifest_paths=manifest_paths,
            manifest_configs=manifest_configs,
        )
    current = _current_artifact_identity(
        artifact_path,
        location=expected_sidecar.artifact.location,
        schema_version=expected_sidecar.versions.schema_version,
        logical_operation=expected_sidecar.method_contract.procedure,
        format_identity=expected_sidecar.artifact.format_identity,
    )
    if current != expected_sidecar.artifact or completion_output.artifact != current:
        raise ArtifactMismatchError(
            "completion identity does not bind current artifact bytes/schema"
        )
    sidecar_bytes = expected_sidecar.canonical_bytes()
    if hashlib.sha256(sidecar_bytes).hexdigest() != completion_output.sidecar_sha256:
        raise ArtifactMismatchError("completion identity does not bind expected sidecar bytes")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_fd, staged = make_staged_path(
        metadata_path,
        prefix="._sidecar_v3_",
        temp_dir=cfg.io.temp_dir,
    )
    os.close(sidecar_fd)
    try:
        staged.write_bytes(sidecar_bytes)
        publish_staged_file_atomic(staged, metadata_path)
    finally:
        staged.unlink(missing_ok=True)
    return validate_authenticated_artifact(
        artifact_path,
        cfg=cfg,
        expected_location=expected_sidecar.artifact.location,
        expected_stage_identity=expected_sidecar.stage_identity,
        expected_method_contract=expected_sidecar.method_contract,
        expected_versions=expected_sidecar.versions,
        expected_sidecar_sha256=completion_output.sidecar_sha256,
        source_paths=source_paths,
        source_configs=source_configs,
        manifest_paths=manifest_paths,
        manifest_configs=manifest_configs,
    )


__all__ = [
    "ARTIFACT_CONTRACT_VERSION",
    "LIFECYCLE_CONTRACT_VERSION",
    "ArtifactFormatIdentity",
    "ArtifactIdentity",
    "ArtifactMismatchError",
    "AuthenticatedCompletion",
    "AuthenticatedContractError",
    "AuthenticatedSidecar",
    "CanonicalArtifactLocation",
    "CodeIdentity",
    "CodeIdentityError",
    "CodeIdentityPolicy",
    "CompletionOutputIdentity",
    "CorruptSidecarError",
    "ImmutableManifestSidecar",
    "ManifestEntry",
    "ManifestRootIdentity",
    "ManifestRootSummary",
    "MethodContract",
    "MissingSidecarError",
    "SourceArtifactIdentity",
    "StageConfigIdentity",
    "StageIdentity",
    "VersionIdentity",
    "arrow_schema_identity",
    "canonical_json_bytes",
    "capture_manifest_root",
    "capture_source_artifact",
    "capture_source_artifact_unbound",
    "classify_authenticated_lifecycle",
    "compute_manifest_root",
    "derive_canonical_location",
    "finalize_missing_sidecar_atomic",
    "file_format_identity",
    "identity_sha256",
    "load_authenticated_sidecar",
    "load_authenticated_completion",
    "load_immutable_manifest_sidecar",
    "make_authenticated_sidecar",
    "make_stage_identity",
    "parquet_schema_identity",
    "publish_authenticated_parquet_atomic",
    "publish_authenticated_file_atomic",
    "publish_staged_authenticated_artifact_atomic",
    "publish_immutable_manifest_atomic",
    "publish_immutable_manifest_bytes_atomic",
    "resolve_code_identity",
    "reuse_stage_parquet_facts",
    "stage_config_identity",
    "validate_authenticated_artifact",
    "validate_authenticated_artifact_metadata",
    "validate_authenticated_artifact_unbound",
    "write_authenticated_completion_atomic",
]
