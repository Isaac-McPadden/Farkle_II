"""Authenticated artifact and lifecycle primitives for contract version 3.

This module is deliberately additive.  Existing stage producers still publish
version-2 sidecars through :mod:`farkle.utils.artifact_contract`; later
migrations opt into these stricter primitives one stage at a time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, TypeVar, cast

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.config import AppConfig, ArtifactScope, effective_config_dict
from farkle.utils.artifact_contract import (
    read_json_file_with_retry,
    replace_file_atomic,
    retry_transient_io,
    sha256_file,
    sidecar_path,
)
from farkle.utils.stage_completion import CompletionState
from farkle.utils.writer import atomic_path

ARTIFACT_CONTRACT_VERSION: Final = 3
LIFECYCLE_CONTRACT_VERSION: Final = 1
MANIFEST_CONTRACT_VERSION: Final = 1
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_BY_K_RE: Final = re.compile(r"^([1-9][0-9]*)p$")
_T = TypeVar("_T")


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
        scope = ArtifactScope(self.scope)
        relative = Path(self.relative_path)
        if not self.stage_key.strip():
            raise ValueError("stage_key must be non-blank")
        if not self.relative_path or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("relative_path must remain within its canonical scope")
        if scope.requires_player_count:
            if isinstance(self.player_count, bool) or self.player_count is None:
                raise ValueError("by_k locations require player_count")
            if self.player_count < 1:
                raise ValueError("player_count must be positive")
        elif self.player_count is not None:
            raise ValueError(f"{scope.value} locations do not accept player_count")

    def path(self, cfg: AppConfig) -> Path:
        """Resolve this logical identity through the application's path API."""

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

    schema = retry_transient_io(lambda: pq.read_schema(path))
    return arrow_schema_identity(schema, schema_version=schema_version)


def _extract_required_path(payload: Mapping[str, Any], dotted_path: str) -> Any:
    cursor: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise ValueError(f"stage config field {dotted_path!r} is absent")
        cursor = cursor[part]
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
            if self.policy != CodeIdentityPolicy.DEVELOPMENT_DIRTY.value:
                raise ValueError("dirty code identity requires explicit development policy")
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
    """Resolve a release-clean identity or a deterministic dirty development one."""

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
        raise CodeIdentityError("release mode requires a clean Git worktree")

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

    def __post_init__(self) -> None:
        for name in ("procedure", "baseline", "replication_unit"):
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
class ArtifactIdentity:
    """Exact physical, byte, schema, and logical-operation identity."""

    location: CanonicalArtifactLocation
    byte_length: int
    content_sha256: str
    arrow_schema: ArrowSchemaIdentity
    logical_operation: str

    def __post_init__(self) -> None:
        if self.byte_length <= 0:
            raise ValueError("artifact byte_length must be positive")
        _require_sha256(self.content_sha256, label="content_sha256")
        if not self.logical_operation.strip():
            raise ValueError("logical_operation must be non-blank")

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
    return _construct(
        ArtifactIdentity,
        payload,
        location=_parse_location(payload["location"]),
        arrow_schema=_parse_schema(payload["arrow_schema"]),
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
        payload = read_json_file_with_retry(metadata_path)
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
        payload = read_json_file_with_retry(metadata_path)
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
) -> ArtifactIdentity:
    try:
        byte_length, content_hash = retry_transient_io(
            lambda: (path.stat().st_size, sha256_file(path))
        )
        schema = parquet_schema_identity(path, schema_version=schema_version)
    except (OSError, pa.ArrowException) as exc:
        raise ArtifactMismatchError(f"artifact cannot be authenticated: {path}: {exc}") from exc
    return ArtifactIdentity(
        location=location,
        byte_length=byte_length,
        content_sha256=content_hash,
        arrow_schema=schema,
        logical_operation=logical_operation,
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
        schema_version=metadata.artifact.arrow_schema.schema_version,
        logical_operation=metadata.artifact.logical_operation,
    )
    if current != metadata.artifact:
        raise ArtifactMismatchError("artifact bytes or actual Arrow schema do not match sidecar")
    if expected_stage_identity is not None and metadata.stage_identity != expected_stage_identity:
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
                    schema_version=source.artifact.arrow_schema.schema_version,
                    logical_operation=source.artifact.logical_operation,
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
    if expected_stage_identity is not None and metadata.stage_identity != expected_stage_identity:
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


def publish_immutable_manifest_atomic(
    path: Path | str,
    *,
    cfg: AppConfig,
    location: CanonicalArtifactLocation,
    stage_identity: StageIdentity,
    entries: Iterable[ManifestEntry],
) -> ImmutableManifestSidecar:
    """Stream and atomically publish an immutable canonical manifest and root."""

    final_path = location.require_path(cfg, path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    data_fd, data_name = tempfile.mkstemp(prefix="._manifest_v3_", dir=final_path.parent)
    os.close(data_fd)
    sidecar_fd, sidecar_name = tempfile.mkstemp(
        prefix="._manifest_sidecar_v3_", dir=final_path.parent
    )
    os.close(sidecar_fd)
    staged_data = Path(data_name)
    staged_sidecar = Path(sidecar_name)
    final_sidecar = sidecar_path(final_path)
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
        final_sidecar.unlink(missing_ok=True)
        replace_file_atomic(staged_data, final_path)
        replace_file_atomic(staged_sidecar, final_sidecar)
        capture_manifest_root(
            logical_role="publication_check",
            manifest_path=final_path,
            cfg=cfg,
            expected_location=location,
            expected_stage_identity=stage_identity,
        )
        return metadata
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
    data_fd, data_name = tempfile.mkstemp(prefix="._artifact_v3_", dir=final_path.parent)
    os.close(data_fd)
    sidecar_fd, sidecar_name = tempfile.mkstemp(prefix="._sidecar_v3_", dir=final_path.parent)
    os.close(sidecar_fd)
    staged_data = Path(data_name)
    staged_sidecar = Path(sidecar_name)
    final_sidecar = sidecar_path(final_path)
    try:
        write_data(staged_data)
        artifact = _current_artifact_identity(
            staged_data,
            location=location,
            schema_version=stage_identity.versions.schema_version,
            logical_operation=method_contract.procedure,
        )
        metadata = make_authenticated_sidecar(
            artifact=artifact,
            stage_identity=stage_identity,
            method_contract=method_contract,
            sources=sources,
            manifest_roots=manifest_roots,
        )
        staged_sidecar.write_bytes(metadata.canonical_bytes())
        final_sidecar.unlink(missing_ok=True)
        replace_file_atomic(staged_data, final_path)
        replace_file_atomic(staged_sidecar, final_sidecar)
        validate_authenticated_artifact(
            final_path,
            cfg=cfg,
            expected_location=location,
            expected_stage_identity=stage_identity,
            expected_method_contract=method_contract,
            expected_versions=stage_identity.versions,
            source_paths=source_paths,
            source_configs=source_configs,
            manifest_paths=manifest_paths,
            manifest_configs=manifest_configs,
        )
        return metadata
    finally:
        staged_data.unlink(missing_ok=True)
        staged_sidecar.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class CompletionOutputIdentity:
    """Completion-stamp binding for one exact output and sidecar."""

    artifact: ArtifactIdentity
    sidecar_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.sidecar_sha256, label="completion sidecar_sha256")


@dataclass(frozen=True, slots=True)
class AuthenticatedCompletion:
    """Minimal immutable completion identity for lifecycle classification."""

    lifecycle_contract_version: int
    stage_identity_sha256: str
    state: str
    outputs: tuple[CompletionOutputIdentity, ...]

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
        locations = [output.artifact.location for output in self.outputs]
        if len(locations) != len(set(locations)):
            raise ValueError("completion outputs must have unique canonical locations")
        if locations != sorted(locations, key=canonical_json_bytes):
            raise ValueError("completion outputs must be in canonical location order")


def write_authenticated_completion_atomic(
    path: Path,
    completion: AuthenticatedCompletion,
) -> None:
    """Atomically publish an authenticated completion identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as temporary_path:
        Path(temporary_path).write_bytes(canonical_json_bytes(completion) + b"\n")


def _load_completion(path: Path) -> AuthenticatedCompletion:
    payload = read_json_file_with_retry(path)
    if not isinstance(payload, Mapping):
        raise TypeError("completion root must be an object")
    outputs = tuple(
        _construct(
            CompletionOutputIdentity,
            item,
            artifact=_parse_artifact(item["artifact"]),
        )
        for item in payload["outputs"]
    )
    return _construct(AuthenticatedCompletion, payload, outputs=outputs)


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
) -> CompletionState:
    """Classify authenticated work into exactly one canonical lifecycle state."""

    materialized = any(location.path(cfg).exists() for location in required_locations) or any(
        path.exists() for path in partial_paths
    )
    if not completion_path.exists():
        return CompletionState.PARTIAL_RESUMABLE if materialized else CompletionState.NOT_STARTED
    try:
        completion = _load_completion(completion_path)
    except Exception:
        return CompletionState.COMPLETE_STALE
    if completion.stage_identity_sha256 != expected_stage_identity.sha256:
        return CompletionState.COMPLETE_STALE
    if completion.state == CompletionState.BLOCKED_BY_CAP.value:
        return CompletionState.BLOCKED_BY_CAP
    expected_paths = {location.path(cfg).resolve(): location for location in required_locations}
    recorded = {output.artifact.location: output for output in completion.outputs}
    if set(recorded) != set(required_locations):
        return CompletionState.COMPLETE_STALE
    try:
        for path, location in expected_paths.items():
            output = recorded[location]
            metadata = validate_authenticated_artifact(
                path,
                cfg=cfg,
                expected_location=location,
                expected_stage_identity=expected_stage_identity,
                expected_sidecar_sha256=output.sidecar_sha256,
                source_paths=source_paths,
                source_configs=source_configs,
                manifest_paths=manifest_paths,
                manifest_configs=manifest_configs,
            )
            if metadata.artifact != output.artifact:
                return CompletionState.COMPLETE_STALE
    except AuthenticatedContractError:
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
    )
    if current != expected_sidecar.artifact or completion_output.artifact != current:
        raise ArtifactMismatchError(
            "completion identity does not bind current artifact bytes/schema"
        )
    sidecar_bytes = expected_sidecar.canonical_bytes()
    if hashlib.sha256(sidecar_bytes).hexdigest() != completion_output.sidecar_sha256:
        raise ArtifactMismatchError("completion identity does not bind expected sidecar bytes")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_fd, sidecar_name = tempfile.mkstemp(prefix="._sidecar_v3_", dir=metadata_path.parent)
    os.close(sidecar_fd)
    staged = Path(sidecar_name)
    try:
        staged.write_bytes(sidecar_bytes)
        replace_file_atomic(staged, metadata_path)
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
    "classify_authenticated_lifecycle",
    "compute_manifest_root",
    "derive_canonical_location",
    "finalize_missing_sidecar_atomic",
    "identity_sha256",
    "load_authenticated_sidecar",
    "load_immutable_manifest_sidecar",
    "make_authenticated_sidecar",
    "make_stage_identity",
    "parquet_schema_identity",
    "publish_authenticated_parquet_atomic",
    "publish_immutable_manifest_atomic",
    "resolve_code_identity",
    "stage_config_identity",
    "validate_authenticated_artifact",
    "write_authenticated_completion_atomic",
]
