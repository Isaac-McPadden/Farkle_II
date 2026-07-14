"""Versioned JSON sidecars for derived analysis artifacts.

The data file and its adjacent sidecar form one validated artifact.  Writers
stage both files, invalidate any previous sidecar, replace the data, and then
publish the new sidecar.  A crash can therefore leave a missing sidecar, but
can never leave a new data file that validates against stale metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, NotRequired, TypeAlias, TypedDict, cast

from farkle.config import ArtifactScope, compute_config_sha

if TYPE_CHECKING:
    from farkle.config import AppConfig

SIDECAR_SUFFIX: Final = ".sidecar.json"
ARTIFACT_CONTRACT_VERSION: Final = 2
_SHA256_LENGTH: Final = 64
_K_AGGREGATION_METHODS: Final = {"equal_k", "declared_mapping", "none"}
_MISSING_CELL_POLICIES: Final = {"fail", "declared_common_support", "not_applicable"}
_SEED_SCOPES: Final = {
    "single_root",
    "both_roots_combined",
    "root_pair_stability",
    "not_applicable",
}


class ArtifactContractError(RuntimeError):
    """Raised when an artifact and its sidecar do not satisfy the contract."""


class OperationMethodContract(TypedDict):
    """Contract for a precisely named transformation without extra semantics."""

    kind: Literal["operation"]
    procedure: str
    parameters: NotRequired[dict[str, Any]]


class H2HMethodContract(TypedDict):
    """Contract for H2H planning, execution, inference, or digestion."""

    kind: Literal["h2h"]
    procedure: str
    parameters: NotRequired[dict[str, Any]]


class TrueSkillMethodContract(TypedDict):
    """Contract for finite-grid TrueSkill screening calculations."""

    kind: Literal["trueskill"]
    procedure: str
    parameters: NotRequired[dict[str, Any]]


class DiagnosticBandMethodContract(TypedDict):
    """Contract for dependence-aware descriptive diagnostic bands."""

    kind: Literal["diagnostic_band"]
    procedure: str
    parameters: NotRequired[dict[str, Any]]


class ConditionalMetricsMethodContract(TypedDict):
    """Contract for outputs whose denominator is conditionally selected."""

    kind: Literal["conditional_metrics"]
    procedure: str
    parameters: NotRequired[dict[str, Any]]


class TurnMetricsMethodContract(TypedDict):
    """Contract for exact turn-denominated player metrics."""

    kind: Literal["turn_metrics"]
    procedure: str
    parameters: NotRequired[dict[str, Any]]


class RootCombinationMethodContract(TypedDict):
    """Contract for raw-count root combination and stability diagnostics."""

    kind: Literal["root_combination"]
    procedure: str
    parameters: NotRequired[dict[str, Any]]


MethodContract: TypeAlias = (
    OperationMethodContract
    | H2HMethodContract
    | TrueSkillMethodContract
    | DiagnosticBandMethodContract
    | ConditionalMetricsMethodContract
    | TurnMetricsMethodContract
    | RootCombinationMethodContract
)

_METHOD_KINDS: Final = {
    "operation",
    "h2h",
    "trueskill",
    "diagnostic_band",
    "conditional_metrics",
    "turn_metrics",
    "root_combination",
}


@dataclass(frozen=True)
class ArtifactSidecar:
    """Minimum metadata required beside every derived analysis artifact."""

    artifact_contract_version: int
    estimand_version: int
    schema_version: int
    artifact_name: str
    producer: str
    scope: str
    source_scope: str
    operation: str
    method_contract: MethodContract
    baseline: str
    weighted_quantity: str
    k_aggregation_method: str
    k_weights: dict[str, float] | None
    support_count_role: str
    uncertainty_method: str
    replication_unit: str
    conditioning: str
    consistency_columns: list[str]
    source_artifacts: list[str]
    grouping_keys: list[str]
    player_counts: list[int]
    required_player_counts: list[int]
    missing_cell_policy: str
    seed_scope: str
    rng_scheme_version: int
    config_hash: str
    input_manifest_hashes: list[str]
    code_revision: str
    artifact_sha256: str = ""
    artifact_size_bytes: int = 0

    def with_artifact_identity(self, path: Path) -> "ArtifactSidecar":
        """Return metadata bound to the exact bytes at *path*."""

        return replace(
            self,
            artifact_name=path.name,
            artifact_sha256=sha256_file(path),
            artifact_size_bytes=path.stat().st_size,
        )


def sidecar_path(artifact_path: Path | str) -> Path:
    """Return the deterministic adjacent sidecar path for an artifact."""

    path = Path(artifact_path)
    return path.with_name(f"{path.name}{SIDECAR_SUFFIX}")


def sha256_file(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_hashes(paths: Sequence[Path | str]) -> list[str]:
    """Return deterministic content hashes for input manifests."""

    return [sha256_file(path) for path in sorted((Path(p) for p in paths), key=str)]


def make_artifact_sidecar(
    cfg: AppConfig,
    artifact_path: Path | str,
    *,
    producer: str,
    scope: ArtifactScope | str,
    source_scope: ArtifactScope | str,
    operation: str,
    baseline: str = "none",
    weighted_quantity: str = "none",
    k_aggregation_method: str = "none",
    k_weights: Mapping[int, float] | None = None,
    support_count_role: str = "raw_support_provenance",
    uncertainty_method: str = "none",
    replication_unit: str = "none",
    conditioning: str = "unconditional",
    consistency_columns: Sequence[str] = (),
    source_artifacts: Sequence[Path | str] = (),
    grouping_keys: Sequence[str] = (),
    player_counts: Sequence[int] = (),
    required_player_counts: Sequence[int] = (),
    missing_cell_policy: str = "not_applicable",
    seed_scope: str = "single_root",
    input_manifests: Sequence[Path | str] = (),
    code_revision: str = "unknown",
    method_contract: MethodContract | None = None,
) -> ArtifactSidecar:
    """Build minimum contract metadata from the active application config."""

    contract = cfg.artifact_contract
    normalized_scope = scope.value if isinstance(scope, ArtifactScope) else str(scope)
    normalized_source_scope = (
        source_scope.value if isinstance(source_scope, ArtifactScope) else str(source_scope)
    )
    resolved_method_contract = method_contract or _default_method_contract(
        producer=producer,
        operation=operation,
        conditioning=conditioning,
    )
    return ArtifactSidecar(
        artifact_contract_version=contract.artifact_contract_version,
        estimand_version=contract.estimand_version,
        schema_version=contract.schema_version,
        artifact_name=Path(artifact_path).name,
        producer=producer,
        scope=normalized_scope,
        source_scope=normalized_source_scope,
        operation=operation,
        method_contract=resolved_method_contract,
        baseline=baseline,
        weighted_quantity=weighted_quantity,
        k_aggregation_method=k_aggregation_method,
        k_weights=(
            None if k_weights is None else {str(k): float(v) for k, v in sorted(k_weights.items())}
        ),
        support_count_role=support_count_role,
        uncertainty_method=uncertainty_method,
        replication_unit=replication_unit,
        conditioning=conditioning,
        consistency_columns=list(consistency_columns),
        source_artifacts=[str(Path(path)) for path in source_artifacts],
        grouping_keys=list(grouping_keys),
        player_counts=sorted({int(k) for k in player_counts}),
        required_player_counts=sorted({int(k) for k in required_player_counts}),
        missing_cell_policy=missing_cell_policy,
        seed_scope=seed_scope,
        rng_scheme_version=cfg.rng.scheme_version,
        config_hash=cfg.config_sha or compute_config_sha(cfg),
        input_manifest_hashes=manifest_hashes(input_manifests),
        code_revision=code_revision,
    )


def _default_method_contract(*, producer: str, operation: str, conditioning: str) -> MethodContract:
    """Return the narrow tagged method contract implied by canonical metadata."""

    normalized_producer = producer.lower()
    normalized_operation = operation.lower()
    if "h2h" in normalized_producer or normalized_producer in {"candidate_family", "dominance"}:
        kind = "h2h"
    elif "trueskill" in normalized_producer:
        kind = "trueskill"
    elif normalized_operation == "diagnostic_band":
        kind = "diagnostic_band"
    elif normalized_producer == "all_player_metrics":
        kind = "turn_metrics"
    elif normalized_producer == "root_stability" or "combine_roots" in normalized_operation:
        kind = "root_combination"
    elif conditioning != "unconditional":
        kind = "conditional_metrics"
    else:
        kind = "operation"
    return cast(MethodContract, {"kind": kind, "procedure": operation})


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(char in "0123456789abcdef" for char in value)


def _validate_sidecar_fields(metadata: ArtifactSidecar) -> None:
    """Validate metadata semantics that do not require reading the artifact."""

    positive_versions = {
        "artifact_contract_version": metadata.artifact_contract_version,
        "estimand_version": metadata.estimand_version,
        "schema_version": metadata.schema_version,
        "rng_scheme_version": metadata.rng_scheme_version,
    }
    invalid_versions = [name for name, value in positive_versions.items() if value < 1]
    if invalid_versions:
        raise ArtifactContractError(f"sidecar versions must be positive: {invalid_versions}")
    if metadata.artifact_contract_version != ARTIFACT_CONTRACT_VERSION:
        raise ArtifactContractError(
            "sidecar artifact contract is stale or unsupported: "
            f"{metadata.artifact_contract_version}; expected {ARTIFACT_CONTRACT_VERSION}"
        )

    required_text = {
        "artifact_name": metadata.artifact_name,
        "producer": metadata.producer,
        "source_scope": metadata.source_scope,
        "operation": metadata.operation,
        "baseline": metadata.baseline,
        "weighted_quantity": metadata.weighted_quantity,
        "support_count_role": metadata.support_count_role,
        "uncertainty_method": metadata.uncertainty_method,
        "replication_unit": metadata.replication_unit,
        "conditioning": metadata.conditioning,
        "config_hash": metadata.config_hash,
        "code_revision": metadata.code_revision,
    }
    blank = [name for name, value in required_text.items() if not value.strip()]
    if blank:
        raise ArtifactContractError(f"sidecar fields must not be blank: {blank}")

    try:
        ArtifactScope(metadata.scope)
    except ValueError as exc:
        raise ArtifactContractError(f"unsupported artifact scope: {metadata.scope!r}") from exc
    if metadata.k_aggregation_method not in _K_AGGREGATION_METHODS:
        raise ArtifactContractError(
            f"unsupported k_aggregation_method: {metadata.k_aggregation_method!r}"
        )
    if metadata.missing_cell_policy not in _MISSING_CELL_POLICIES:
        raise ArtifactContractError(
            f"unsupported missing_cell_policy: {metadata.missing_cell_policy!r}"
        )
    if metadata.seed_scope not in _SEED_SCOPES:
        raise ArtifactContractError(f"unsupported seed_scope: {metadata.seed_scope!r}")
    method_contract = metadata.method_contract
    if not isinstance(method_contract, dict):
        raise ArtifactContractError("method_contract must be a tagged object")
    if method_contract.get("kind") not in _METHOD_KINDS:
        raise ArtifactContractError(
            f"unsupported method_contract kind: {method_contract.get('kind')!r}"
        )
    procedure = method_contract.get("procedure")
    if not isinstance(procedure, str) or not procedure.strip():
        raise ArtifactContractError("method_contract procedure must be non-blank")
    if procedure != metadata.operation:
        raise ArtifactContractError(
            "method_contract procedure must equal the sidecar operation identifier"
        )
    parameters = method_contract.get("parameters")
    if parameters is not None and not isinstance(parameters, dict):
        raise ArtifactContractError("method_contract parameters must be an object when present")

    if metadata.k_aggregation_method == "declared_mapping":
        if not metadata.k_weights:
            raise ArtifactContractError("declared_mapping requires non-empty k_weights")
        weight_sum = sum(metadata.k_weights.values())
        if (
            any(weight <= 0 for weight in metadata.k_weights.values())
            or abs(weight_sum - 1.0) > 1e-12
        ):
            raise ArtifactContractError("declared k_weights must be positive and sum to one")
    elif metadata.k_weights is not None:
        raise ArtifactContractError(
            f"{metadata.k_aggregation_method} requires k_weights to be null"
        )

    player_counts = metadata.player_counts
    required_counts = metadata.required_player_counts
    if player_counts != sorted(set(player_counts)) or any(k < 1 for k in player_counts):
        raise ArtifactContractError("player_counts must contain sorted unique positive values")
    if required_counts != sorted(set(required_counts)) or any(k < 1 for k in required_counts):
        raise ArtifactContractError(
            "required_player_counts must contain sorted unique positive values"
        )
    if metadata.missing_cell_policy == "fail" and not set(required_counts).issubset(player_counts):
        raise ArtifactContractError("required player-count support is incomplete under fail policy")

    invalid_hashes = [value for value in metadata.input_manifest_hashes if not _is_sha256(value)]
    if invalid_hashes:
        raise ArtifactContractError("input_manifest_hashes must contain lowercase SHA-256 digests")
    if metadata.artifact_sha256 and not _is_sha256(metadata.artifact_sha256):
        raise ArtifactContractError("artifact_sha256 must be a lowercase SHA-256 digest")
    if metadata.artifact_size_bytes < 0:
        raise ArtifactContractError("artifact_size_bytes must not be negative")


def _canonical_json(metadata: ArtifactSidecar) -> str:
    return json.dumps(asdict(metadata), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_artifact_with_sidecar_atomic(
    artifact_path: Path | str,
    metadata: ArtifactSidecar,
    write_data: Callable[[Path], None],
) -> ArtifactSidecar:
    """Atomically publish data plus hash-bound metadata.

    ``write_data`` receives a temporary path in the destination directory.  A
    previous sidecar is invalidated before the data replacement, and the new
    sidecar is published only after the new data is in place.
    """

    final_path = Path(artifact_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    data_fd, data_name = tempfile.mkstemp(prefix="._artifact_", dir=final_path.parent)
    os.close(data_fd)
    staged_data = Path(data_name)
    try:
        write_data(staged_data)
        return publish_staged_artifact_with_sidecar(staged_data, final_path, metadata)
    finally:
        staged_data.unlink(missing_ok=True)


def publish_staged_artifact_with_sidecar(
    staged_data: Path | str,
    artifact_path: Path | str,
    metadata: ArtifactSidecar,
) -> ArtifactSidecar:
    """Publish an already-written temporary data file with its sidecar."""

    staged_path = Path(staged_data)
    final_path = Path(artifact_path)
    final_sidecar = sidecar_path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if not staged_path.is_file() or staged_path.stat().st_size == 0:
        raise ArtifactContractError(f"artifact writer did not create {staged_path}")
    if staged_path.parent.resolve() != final_path.parent.resolve():
        raise ArtifactContractError("staged artifact must be on the destination filesystem")

    sidecar_fd, sidecar_name = tempfile.mkstemp(prefix="._sidecar_", dir=final_path.parent)
    os.close(sidecar_fd)
    staged_sidecar = Path(sidecar_name)
    try:
        bound = replace(metadata.with_artifact_identity(staged_path), artifact_name=final_path.name)
        _validate_sidecar_fields(bound)
        staged_sidecar.write_text(_canonical_json(bound), encoding="utf-8")

        final_sidecar.unlink(missing_ok=True)
        os.replace(staged_path, final_path)
        os.replace(staged_sidecar, final_sidecar)
        validate_artifact_sidecar(final_path)
        return bound
    finally:
        staged_sidecar.unlink(missing_ok=True)


def load_artifact_sidecar(artifact_path: Path | str) -> ArtifactSidecar:
    """Load and structurally validate an artifact sidecar."""

    path = Path(artifact_path)
    metadata_path = sidecar_path(path)
    if not metadata_path.is_file():
        raise ArtifactContractError(
            f"missing sidecar for {path}; expected adjacent {metadata_path.name}"
        )
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = ArtifactSidecar(**payload)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ArtifactContractError(f"invalid sidecar {metadata_path}: {exc}") from exc
    _validate_sidecar_fields(metadata)
    return metadata


def validate_artifact_sidecar(
    artifact_path: Path | str,
    *,
    expected: Mapping[str, Any] | None = None,
) -> ArtifactSidecar:
    """Validate the sidecar, byte identity, and optional consumer expectations."""

    path = Path(artifact_path)
    if not path.is_file():
        raise ArtifactContractError(f"artifact does not exist: {path}")
    metadata = load_artifact_sidecar(path)
    if metadata.artifact_name != path.name:
        raise ArtifactContractError(
            f"sidecar artifact_name {metadata.artifact_name!r} does not match {path.name!r}"
        )
    if metadata.artifact_size_bytes != path.stat().st_size:
        raise ArtifactContractError(f"artifact size does not match sidecar: {path}")
    digest = sha256_file(path)
    if metadata.artifact_sha256 != digest:
        raise ArtifactContractError(f"artifact content hash does not match sidecar: {path}")

    for key, wanted in (expected or {}).items():
        if not hasattr(metadata, key):
            raise ArtifactContractError(f"unknown sidecar expectation: {key}")
        actual = getattr(metadata, key)
        if actual != wanted:
            raise ArtifactContractError(
                f"incompatible sidecar for {path}: {key}={actual!r}, expected {wanted!r}"
            )
    return metadata


__all__ = [
    "ArtifactContractError",
    "ARTIFACT_CONTRACT_VERSION",
    "ArtifactSidecar",
    "MethodContract",
    "SIDECAR_SUFFIX",
    "load_artifact_sidecar",
    "make_artifact_sidecar",
    "manifest_hashes",
    "publish_staged_artifact_with_sidecar",
    "sha256_file",
    "sidecar_path",
    "validate_artifact_sidecar",
    "write_artifact_with_sidecar_atomic",
]
