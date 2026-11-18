"""Protect the exact publications consumed by an already issued H2H design."""

from __future__ import annotations

from pathlib import Path

from farkle.config import AppConfig
from farkle.utils.artifact_contract import sha256_file, sidecar_path
from farkle.utils.authenticated_contract import (
    ArtifactMismatchError,
    AuthenticatedContractError,
    AuthenticatedSidecar,
    SourceArtifactIdentity,
    derive_canonical_location,
    load_authenticated_completion,
    validate_authenticated_artifact_unbound,
)
from farkle.utils.stage_completion import stage_done_path


def frozen_family_inputs(cfg: AppConfig) -> dict[Path, AuthenticatedSidecar]:
    """Follow the frozen planning sidecars to the family and its exact source edges."""

    power = cfg.h2h_power_plan_path()
    schedule = cfg.h2h_block_manifest_path()
    done = stage_done_path(cfg.stage_dir("h2h_power"), "h2h_power")
    if schedule.exists() and not power.is_file():
        raise ArtifactMismatchError(
            f"Frozen H2H block manifest has no power plan at {power}; preserve this root for recovery"
        )
    if not power.is_file() or not schedule.is_file():
        evidence = [done]
        if cfg.stage_layout.folder_for("h2h_execute") is not None:
            authorization = cfg.h2h_cap_authorization_path()
            evidence.extend([authorization, sidecar_path(authorization)])
        if any(path.exists() for path in evidence):
            raise ArtifactMismatchError(
                f"Frozen H2H plan is incomplete at {power.parent}; preserve this root for recovery"
            )

    bound: dict[Path, AuthenticatedSidecar] = {}

    def resolve(reference: SourceArtifactIdentity) -> AuthenticatedSidecar:
        location = reference.artifact.location
        path = location.path(cfg).resolve()
        metadata = bound.get(path)
        if metadata is None:
            metadata = validate_authenticated_artifact_unbound(
                path, expected_location=location, validate_provenance=False
            )
            bound[path] = metadata
        if (
            metadata.artifact != reference.artifact
            or metadata.sidecar_contract_sha256 != reference.sidecar_contract_sha256
            or sha256_file(sidecar_path(path)) != reference.sidecar_sha256
        ):
            raise ArtifactMismatchError(f"frozen source bytes or sidecar differ at {path}")
        return metadata

    for anchor, operation in (
        (power, "score_test_power_plan"),
        (schedule, "construct_pair_root_order_blocks"),
    ):
        if not anchor.exists() and not sidecar_path(anchor).exists():
            continue
        try:
            plan = validate_authenticated_artifact_unbound(
                anchor,
                expected_location=derive_canonical_location(cfg, anchor, stage_key="h2h_power"),
                validate_provenance=False,
            )
            bound[anchor.resolve()] = plan
            family_paths = {source.artifact.location.path(cfg) for source in plan.source_artifacts}
            if plan.method_contract.procedure != operation or family_paths != {
                cfg.h2h_candidate_family_path(),
                cfg.h2h_candidate_family_manifest_path(),
            }:
                raise ArtifactMismatchError(
                    "unexpected planning operation or candidate family references"
                )
            for source in plan.source_artifacts:
                family = resolve(source)
                for contribution in family.source_artifacts:
                    resolve(contribution)
        except (AuthenticatedContractError, OSError, ValueError, KeyError) as exc:
            raise ArtifactMismatchError(
                f"Cannot authenticate frozen H2H references from {anchor}: {exc}; "
                "preserve this root for recovery or use a new output root"
            ) from exc
    if done.exists():
        try:
            completion = load_authenticated_completion(done)
            pinned = {item.location.path(cfg): item for item in completion.outputs}
        except (AuthenticatedContractError, OSError, KeyError, ValueError, TypeError) as exc:
            raise ArtifactMismatchError(
                f"Cannot authenticate frozen H2H completion {done}: {exc}; "
                "preserve this root for recovery"
            ) from exc
        if set(pinned) != {power, schedule}:
            raise ArtifactMismatchError(
                f"Frozen H2H completion inventory differs at {done}; preserve this root for recovery"
            )
        for path, item in pinned.items():
            if item.artifact != bound[
                path.resolve()
            ].artifact or item.sidecar_sha256 != sha256_file(sidecar_path(path)):
                raise ArtifactMismatchError(
                    f"Frozen H2H completion no longer authenticates {path}; preserve this root for recovery"
                )
    return bound


def reuse_frozen_input(
    cfg: AppConfig, proposed: AuthenticatedSidecar
) -> AuthenticatedSidecar | None:
    """Return the original authenticated publication or reject before either file is replaced."""

    location = proposed.artifact.location
    # Candidate freeze can consume explicit contribution paths in either of
    # these scopes. The actual pinned paths come from the planning sidecars.
    if location.scope not in {"cross_seed", "across_k"} and location.stage_key not in {
        "candidate_freeze",
        "h2h_power",
    }:
        return None
    if cfg.stage_layout.folder_for("h2h_power") is None:
        return None
    original = frozen_family_inputs(cfg).get(location.path(cfg).resolve())
    if original is None or original.sidecar_contract_sha256 == proposed.sidecar_contract_sha256:
        return original

    # A changed pair lineage can reflect regenerated lifecycle records. Its
    # digest alone cannot establish that the difference is purely operational.
    stage = proposed.stage_identity
    path = location.path(cfg)
    previous = original.stage_identity
    differences = [
        name
        for name, old, new in (
            ("artifact content or schema", original.artifact, proposed.artifact),
            ("method", previous.method_contract_sha256, stage.method_contract_sha256),
            ("code", previous.code, stage.code),
            ("configuration", previous.stage_config.sha256, stage.stage_config.sha256),
            ("upstream inputs", previous.upstream_identity_sha256, stage.upstream_identity_sha256),
            ("versions", previous.versions, stage.versions),
            (
                "design or lineage",
                previous.immutable_design_identities,
                stage.immutable_design_identities,
            ),
        )
        if old != new
    ]
    raise ArtifactMismatchError(
        f"Frozen H2H input {path} differs in {', '.join(differences) or 'sidecar identity'} "
        f"(content {original.artifact.content_sha256} -> {proposed.artifact.content_sha256}, "
        f"stage {original.stage_identity.sha256} -> {stage.sha256}); "
        "preserve the frozen plan and use a new output root for the changed design"
    )
