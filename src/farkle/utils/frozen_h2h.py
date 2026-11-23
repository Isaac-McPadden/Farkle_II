"""Protect the exact publications consumed by an already issued H2H design."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from farkle.config import AppConfig
from farkle.simulation.strategies import STRATEGY_TUPLE_FIELDS
from farkle.utils.artifact_contract import sha256_file, sidecar_path
from farkle.utils.authenticated_contract import (
    ArtifactMismatchError,
    AuthenticatedContractError,
    AuthenticatedSidecar,
    SourceArtifactIdentity,
    StageIdentity,
    derive_canonical_location,
    load_authenticated_completion,
    producer_code_identity,
    validate_authenticated_artifact_unbound,
)
from farkle.utils.scientific_reuse import (
    ROOT_INPUT_OPERATIONS,
    compatible_producer,
    immutable_input_error,
    sealed_root_input,
)
from farkle.utils.stage_completion import stage_done_path
from farkle.utils.strategy_ids import canonical_strategy_ids

BUILTIN_ENGINE = "farkle_builtin_v1"
SYNTHETIC_ENGINE = "test_only_synthetic_v1"
FROZEN_DESIGN_VERSION = 2


def contribution_science(path: Path, publication: AuthenticatedSidecar) -> dict[str, str]:
    science = publication.stage_identity.immutable_design_identities.get("publication_science_v1")
    if science is None:
        raise ValueError(f"Missing scientific identity for candidate contribution {path}")
    return {
        "artifact_name": path.name,
        "sha256": publication.artifact.content_sha256,
        "publication_science_v1": science,
        "method_contract_sha256": publication.method_contract.sha256,
        "conditioning": publication.method_contract.conditioning,
    }


def canonical_strategy_mapping(
    frame: pd.DataFrame, candidates: Sequence[int]
) -> list[dict[str, Any]]:
    """Select the exact strategy attributes consumed by the game, in canonical ID order."""
    columns = ["strategy_id", *STRATEGY_TUPLE_FIELDS]
    if not frame.columns.is_unique or set(columns) - set(frame.columns):
        raise ValueError("strategy manifest is missing unique canonical strategy columns")
    ids = canonical_strategy_ids(frame["strategy_id"], context="frozen strategy mapping")
    if ids.duplicated().any():
        raise ValueError("duplicate strategy ID in strategy manifest")
    selected = frame.loc[ids.isin(candidates), columns].sort_values("strategy_id")
    if sorted(candidates) != selected["strategy_id"].tolist():
        raise ValueError("strategy manifest does not cover every frozen candidate exactly once")
    rows = selected.to_dict(orient="records")
    for row in rows:
        for key in columns:
            value = row[key]
            if key == "favor_dice_or_score":
                valid = value in ("score", "dice")
            else:
                expected = (
                    int if key in {"strategy_id", "score_threshold", "dice_threshold"} else bool
                )
                valid = type(value) is expected
            if not valid:
                raise ValueError(
                    f"invalid strategy manifest field {key} for candidate {row['strategy_id']}"
                )
    return [{str(k): v for k, v in row.items()} for row in rows]


def read_authenticated_mapping(
    path: Path, digest: str, candidates: Sequence[int], *, temp_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Decode exactly the authenticated bytes, including on replaceable/synced filesystems."""
    observed = hashlib.sha256()
    with tempfile.TemporaryFile(dir=temp_dir) as snapshot:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                observed.update(chunk)
                snapshot.write(chunk)
        if observed.hexdigest() != digest:
            raise ValueError("strategy manifest bytes differ from the authenticated publication")
        snapshot.seek(0)
        rows: dict[int, dict[str, Any]] = {}
        wanted = set(candidates)
        columns = ["strategy_id", *STRATEGY_TUPLE_FIELDS]
        for batch in pq.ParquetFile(snapshot).iter_batches(batch_size=8192, columns=columns):
            frame = batch.to_pandas()
            selected = frame.loc[frame["strategy_id"].isin(wanted)]
            if not selected.empty:
                for row in canonical_strategy_mapping(selected, selected["strategy_id"].tolist()):
                    sid = row["strategy_id"]
                    if sid in rows:
                        raise ValueError("duplicate strategy ID in strategy manifest")
                    rows[sid] = row
        if len(wanted) != len(candidates) or rows.keys() != wanted:
            raise ValueError("strategy manifest does not cover every frozen candidate exactly once")
        return [rows[sid] for sid in sorted(rows)]


def validated_root_strategy_manifest(cfg: AppConfig) -> AuthenticatedSidecar:
    """Authenticate the root-owned publication before any resume, force, or freeze."""
    path = cfg.strategy_manifest_root_path()
    metadata = validate_authenticated_artifact_unbound(
        path,
        expected_location=derive_canonical_location(cfg, path, stage_key="simulation"),
        validate_provenance=False,
    )
    if metadata.method_contract.procedure != "publish_strategy_manifest":
        raise ArtifactMismatchError(f"Unexpected strategy manifest operation at {path}")
    if not compatible_producer(metadata.stage_identity, producer_code_identity(cfg)):
        raise ArtifactMismatchError(
            f"Strategy manifest at {path} belongs to different code; use a new output root"
        )
    return metadata


def authenticated_strategy_design(
    cfg: AppConfig, candidates: Sequence[int], *, engine: str
) -> dict[str, Any]:
    """Bind canonical game inputs to the existing authenticated simulation publication."""
    if engine not in (BUILTIN_ENGINE, SYNTHETIC_ENGINE):
        raise ArtifactMismatchError(f"Unclassified H2H engine {engine!r}")
    path = cfg.strategy_manifest_root_path()
    try:
        metadata = validated_root_strategy_manifest(cfg)
        science = metadata.stage_identity.immutable_design_identities.get("publication_science_v1")
        if science is None:
            raise ValueError("strategy manifest lacks versioned scientific producer evidence")
        mapping = read_authenticated_mapping(
            path, metadata.artifact.content_sha256, candidates, temp_dir=cfg.io.temp_dir
        )
        manifest = cfg.h2h_candidate_family_manifest_path()
        family = {}
        contribution_edges: tuple[SourceArtifactIdentity, ...] | None = None
        for role, member in (
            ("frozen_membership", cfg.h2h_candidate_family_path()),
            ("frozen_family_manifest", cfg.h2h_candidate_family_manifest_path()),
        ):
            source = validate_authenticated_artifact_unbound(
                member,
                expected_location=derive_canonical_location(
                    cfg, member, stage_key="candidate_freeze"
                ),
                validate_provenance=False,
            )
            origin = source.stage_identity.immutable_design_identities.get("publication_science_v1")
            if origin is None or source.method_contract.procedure != "candidate_family_freeze":
                raise ValueError(f"unclassified frozen family input at {member}")
            if not compatible_producer(source.stage_identity, producer_code_identity(cfg)):
                raise ValueError("strategy mapping and frozen family belong to different code")
            if len(source.source_artifacts) != 2 or (
                contribution_edges is not None and contribution_edges != source.source_artifacts
            ):
                raise ValueError("frozen family must bind the same two scientific contributions")
            contribution_edges = source.source_artifacts
            # The family hash in the schedule binds scientific membership and both
            # contributions. Exact JSON may also carry scratch paths and estimates.
            family[role] = origin
        from farkle.utils.scientific_identity import require_input_roles

        require_input_roles(cfg, "h2h_power", {*family, "strategy_manifest"})
        payload = json.loads(manifest.read_bytes())
        if payload.get("source_identity_version") != 2 or sorted(payload["candidates"]) != sorted(
            candidates
        ):
            raise ValueError("missing or incompatible frozen family scientific identity")
        recorded = dict(payload["source_identity"])
        if set(recorded) != {"win_rate", "trueskill"}:
            raise ValueError("frozen family lacks the complete scientific contribution inventory")
        for edge in contribution_edges or ():
            contribution = edge.artifact.location.path(cfg)
            item = validate_authenticated_artifact_unbound(contribution, validate_provenance=False)
            role = (
                "trueskill"
                if item.method_contract.procedure == "equal_root_k_percentile_mean"
                else "win_rate"
            )
            if (
                edge.artifact != item.artifact
                or edge.sidecar_contract_sha256 != item.sidecar_contract_sha256
                or edge.sidecar_sha256 != sha256_file(sidecar_path(contribution))
                or recorded.pop(role, None) != contribution_science(contribution, item)
                or not compatible_producer(item.stage_identity, producer_code_identity(cfg))
            ):
                raise ValueError(f"frozen family {role} scientific contribution differs")
        return {
            "version": FROZEN_DESIGN_VERSION,
            "engine": engine,
            "family_publications": family,
            "strategy_mapping": mapping,
            "strategy_artifact_sha256": metadata.artifact.content_sha256,
            "strategy_publication_science_v1": science,
            "strategy_method_sha256": metadata.method_contract.sha256,
            "strategy_config_sha256": metadata.stage_identity.stage_config.sha256,
            "planning_dependencies_sha256": cfg.stage_config_sha("h2h_power"),
            "execution_dependencies_sha256": cfg.stage_config_sha("h2h_execute"),
            "block_dependencies_sha256": cfg.partition_dependency_sha("h2h_execute", "block"),
        }
    except (OSError, ValueError, KeyError, AuthenticatedContractError) as exc:
        raise ArtifactMismatchError(
            f"H2H requires an authenticated strategy manifest at {path}: {exc}. "
            "Publish the simulation strategy manifest for this root before planning or authorization."
        ) from exc


def verify_frozen_strategy_design(cfg: AppConfig, plan: Mapping[str, Any]) -> None:
    design = plan.get("frozen_design")
    if not isinstance(design, dict) or design.get("version") != FROZEN_DESIGN_VERSION:
        raise ArtifactMismatchError(
            "Historical H2H plan has no authenticated strategy mapping; preserve this plan "
            "and publish a new design in a new output root before authorizing execution"
        )
    frozen_family_inputs(cfg)
    if json.loads(cfg.h2h_power_plan_path().read_bytes()) != plan:
        raise ArtifactMismatchError("Requested authorization differs from the published H2H plan")
    candidates = [row["strategy_id"] for row in design["strategy_mapping"]]
    engine = design.get("engine")
    if (
        not isinstance(engine, str)
        or authenticated_strategy_design(cfg, candidates, engine=engine) != design
    ):
        raise ArtifactMismatchError("Loaded scientific design differs from the authorized H2H plan")


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
            expected_paths = {
                cfg.h2h_candidate_family_path(),
                cfg.h2h_candidate_family_manifest_path(),
            }
            if "frozen_design" in json.loads(power.read_bytes()):
                expected_paths.add(cfg.strategy_manifest_root_path())
            if plan.method_contract.procedure != operation or family_paths != expected_paths:
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


def same_publication(original: AuthenticatedSidecar, proposed: AuthenticatedSidecar) -> bool:
    old, new = original.stage_identity, proposed.stage_identity
    return (
        original.artifact == proposed.artifact
        and original.source_artifacts == proposed.source_artifacts
        and original.manifest_roots == proposed.manifest_roots
        and original.method_contract == proposed.method_contract
        and same_production(old, new)
    )


def same_production(old: StageIdentity, new: StageIdentity) -> bool:
    def design(stage: StageIdentity) -> dict[str, str]:
        return {
            k: v for k, v in stage.immutable_design_identities.items() if k != "run_lineage_sha256"
        }

    return (
        old.stage_key == new.stage_key
        and old.stage_cache_key_version == new.stage_cache_key_version
        and old.stage_config.sha256 == new.stage_config.sha256
        and old.versions == new.versions
        and old.method_contract_sha256 == new.method_contract_sha256
        and old.upstream_identity_sha256 == new.upstream_identity_sha256
        and design(old) == design(new)
        and compatible_producer(old, new.code)
    )


def reuse_frozen_input(
    cfg: AppConfig, proposed: AuthenticatedSidecar
) -> AuthenticatedSidecar | None:
    """Return the original authenticated publication or reject before either file is replaced."""

    location = proposed.artifact.location
    path = location.path(cfg)
    if path == cfg.strategy_manifest_root_path():
        if not path.exists() and not sidecar_path(path).exists():
            return None
        previous_strategy = validated_root_strategy_manifest(cfg)
        science = proposed.stage_identity.immutable_design_identities.get("publication_science_v1")
        if (
            science is None
            or previous_strategy.artifact != proposed.artifact
            or (
                previous_strategy.stage_identity.immutable_design_identities.get(
                    "publication_science_v1"
                )
                != science
            )
        ):
            raise ArtifactMismatchError(
                f"Immutable root strategy manifest differs at {path}; use a new output root"
            )
        return previous_strategy
    if proposed.method_contract.procedure in ROOT_INPUT_OPERATIONS or (
        location.stage_key in {"simulation", "ingest", "curate", "metrics", "trueskill"}
        and sealed_root_input(path)
    ):
        if not sidecar_path(path).exists():
            return None
        from farkle.utils.authenticated_contract import load_authenticated_sidecar

        previous_input = load_authenticated_sidecar(path)
        if not same_publication(previous_input, proposed):
            raise immutable_input_error(path)
        return previous_input
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
    if same_publication(original, proposed):
        return original
    raise ArtifactMismatchError(
        f"Frozen H2H input {path} differs in {', '.join(differences) or 'sidecar identity'} "
        f"(content {original.artifact.content_sha256} -> {proposed.artifact.content_sha256}, "
        f"stage {original.stage_identity.sha256} -> {stage.sha256}); "
        "preserve the frozen plan and use a new output root for the changed design"
    )
