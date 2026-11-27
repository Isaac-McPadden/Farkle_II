# src/farkle/simulation/runner.py
"""High level tournament runner using configuration objects.

The :func:`run_tournament` function acts as a thin wrapper around the
lower level helpers found in :mod:`farkle.simulation.run_tournament`.
It accepts an :class:`AppConfig` instance which mirrors the structure
used throughout the project.  Only a tiny subset of the original
configuration surface is supported – just enough for the unit tests –
but additional fields can be added in the future without touching the
public API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import stat
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa

import farkle.simulation.run_tournament as tournament_mod
from farkle.config import AppConfig, ArtifactScope
from farkle.protocols import SCREENING_INTERVAL_CONFIDENCE
from farkle.simulation.game_profile import GameProfile
from farkle.simulation.run_tournament import METRIC_LABELS, TournamentConfig
from farkle.simulation.simulation import (
    _prepare_public_helper_strategies,
    generate_strategy_grid,
)
from farkle.simulation.strategies import (
    STRATEGY_MANIFEST_NAME,
    ThresholdStrategy,
    build_strategy_manifest,
)
from farkle.simulation.workload_planner import (
    TournamentWorkloadPlan,
    WorkloadCapExceeded,
    plan_tournament_workload,
    write_workload_plan,
)
from farkle.utils import random as urandom
from farkle.utils.artifact_contract import (
    ArtifactSidecar,
    make_artifact_sidecar,
    sha256_file,
    sidecar_path,
    write_artifact_with_sidecar_atomic,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic, write_parquet_atomic
from farkle.utils.authenticated_contract import (
    ArtifactMismatchError,
    AuthenticatedContractError,
    AuthenticatedSidecar,
    ManifestEntry,
    canonical_json_bytes,
    compute_manifest_root,
    derive_canonical_location,
    load_immutable_manifest_sidecar,
    pending_publication_recovery,
    producer_code_identity,
    validate_authenticated_artifact_metadata,
    validate_authenticated_artifact_unbound,
)
from farkle.utils.manifest import iter_manifest
from farkle.utils.parallel import (
    apply_native_thread_limits,
    is_process_pool_worker,
    make_process_tree_memory_guard,
    normalize_n_jobs,
    resolve_stage_parallel_policy,
)
from farkle.utils.release_identity import (
    capture_v3_inputs,
    exact_direct_sources,
    is_v3_config,
    publish_native_manifest_v3,
    recover_simulation_output,
    scientific_artifact_binding,
)
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION, TOURNAMENT_METHOD_VERSION
from farkle.utils.scientific_identity import producer_scope
from farkle.utils.scientific_invocation import stage_action
from farkle.utils.scientific_runtime import numerical_observation
from farkle.utils.stage_completion import (
    CompletionState,
    freshness_sha256,
    resolve_stage_state,
    write_stage_done,
)
from farkle.utils.telemetry import (
    current_supervisor_recorder,
    current_supervisor_scope,
    observe_recovery,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


def _resolve_strategies(
    cfg: AppConfig, strategies: list[ThresholdStrategy] | None
) -> tuple[list[ThresholdStrategy], int, bool]:
    """
    Returns (strategies_list, grid_size, used_custom_grid: bool).
    """
    if strategies is None:
        strategies, _ = generate_strategy_grid(
            score_thresholds=cfg.sim.score_thresholds,
            dice_thresholds=cfg.sim.dice_thresholds,
            smart_five_opts=cfg.sim.smart_five_opts,
            smart_one_opts=cfg.sim.smart_one_opts,
            consider_score_opts=cfg.sim.consider_score_opts,
            consider_dice_opts=cfg.sim.consider_dice_opts,
            auto_hot_dice_opts=cfg.sim.auto_hot_dice_opts,
            run_up_score_opts=cfg.sim.run_up_score_opts,
            include_stop_at=cfg.sim.include_stop_at,
            include_stop_at_heuristic=cfg.sim.include_stop_at_heuristic,
            # prefer_score is and must be handled automatically
        )
        used_custom = any(
            [
                cfg.sim.score_thresholds is not None,
                cfg.sim.dice_thresholds is not None,
                cfg.sim.smart_five_opts is not None,
                cfg.sim.smart_one_opts is not None,
                cfg.sim.consider_score_opts not in [(True, False), [True, False], None],
                cfg.sim.consider_dice_opts not in [(True, False), [True, False], None],
                cfg.sim.auto_hot_dice_opts not in [(False, True), [False, True], None],
                cfg.sim.run_up_score_opts not in [(False, True), [False, True], None],
                cfg.sim.include_stop_at,
                cfg.sim.include_stop_at_heuristic,
            ]
        )
    else:
        used_custom = True  # caller provided a custom grid explicitly

    strategies = _prepare_public_helper_strategies(strategies)
    grid_size = len(strategies)

    LOGGER.info(
        "Strategy grid prepared: %d strategies (%s grid)",
        grid_size,
        "custom" if used_custom else "default",
    )
    return strategies, grid_size, used_custom


def validate_workload(
    cfg: AppConfig, *, player_counts: Sequence[int] | None = None, grid_size: int | None = None
) -> None:
    """Reject the entire requested support before starting any simulation cell."""
    counts = cfg.sim.n_players_list if player_counts is None else player_counts
    if isinstance(counts, str) or not isinstance(counts, Sequence):
        raise ValueError("sim.n_players_list must be a list of player counts.")
    if not counts:
        raise ValueError("sim.n_players_list is empty. Specify at least one player count.")
    if grid_size is None:
        _, grid_size, _ = _resolve_strategies(cfg, None)
    invalid = [
        n
        for n in counts
        if isinstance(n, bool) or not isinstance(n, int) or n < 2 or n > grid_size or grid_size % n
    ]
    if invalid:
        raise ValueError(
            f"Invalid sim.n_players_list entries {invalid!r} for the configured roster of "
            f"{grid_size:,} strategies. Choose integer player counts of at least 2 that "
            "divide the positive roster size, or adjust the strategy grid."
        )
    if len(set(counts)) != len(counts):
        raise ValueError("sim.n_players_list contains duplicates. List each player count once.")


def _plan_workload_from_config(
    cfg: AppConfig,
    n_strategies: int,
    n_players: int,
) -> TournamentWorkloadPlan:
    """Resolve the screening precision workload for one root/player-count cell."""

    return plan_tournament_workload(
        root_seed=cfg.sim.seed,
        k=n_players,
        strategy_count=n_strategies,
        resolution_delta=cfg.screening.resolution_delta,
        confidence=SCREENING_INTERVAL_CONFIDENCE,
        batch_count=cfg.batching.target_batches,
        min_shuffles_per_batch=cfg.batching.min_shuffles_per_batch,
        shuffle_cap=cfg.screening.max_shuffles_per_root_k,
        projected_games_per_second=cfg.screening.projected_games_per_second,
    )


def _workload_checkpoint_metadata(plan: TournamentWorkloadPlan) -> dict[str, object]:
    """Return precision fields that make replacement checkpoint contracts stale."""

    return {
        "workload_plan_version": plan.plan_version,
        "screening_resolution_delta": plan.resolution_delta,
        "screening_interval_confidence": plan.confidence,
        "batch_count": plan.batch_count,
        "shuffles_per_batch": plan.shuffles_per_batch,
        "batch_construction": plan.batch_construction,
        "coordinate_contract_version": 1,
        "deterministic_batch_size": plan.shuffles_per_batch,
    }


def _resolve_row_output_dir(cfg: AppConfig, n_players: int) -> Path | None:
    """Return the per-N row output directory or ``None`` if rows are disabled."""
    return cfg.simulation_row_dir(n_players)


def _resolve_metric_chunk_dir(cfg: AppConfig, n_players: int) -> Path | None:
    """Return the per-N metrics chunk directory or ``None`` if disabled."""
    return _resolve_per_n_output_dir(
        cfg,
        getattr(cfg.sim, "metric_chunk_dir", None),
        n_players,
    )


def simulation_done_path(cfg: AppConfig, n_players: int) -> Path:
    """Return the completion marker path for a per-N simulation run."""
    n_dir = cfg.results_root / f"{n_players}_players"
    return n_dir / "simulation.done.json"


@numerical_observation()
def simulation_is_complete(
    cfg: AppConfig, n_players: int, *, report_recovery: bool = False
) -> bool:
    """Return whether the per-N completion authenticates current work exactly."""

    if _pending_simulation_publications(cfg, n_players):
        return False
    done_path = simulation_done_path(cfg, n_players)
    scope = current_supervisor_scope() if report_recovery else None
    if scope is not None:
        scope.reset_recovery()
    recovery = (
        observe_recovery("authentication", "Check simulation completion")
        if scope is not None and done_path.is_file()
        else None
    )
    if recovery is not None:
        recovery.publish(root_seed=cfg.sim.seed, current_player_count=n_players)
    outputs: list[Path] = []
    inputs: list[Path] = []
    if is_v3_config(cfg):
        try:
            payload = json.loads(done_path.read_text(encoding="utf-8"))
            for item in payload["outputs"]:
                declared = item.get("artifact") or item.get("manifest")
                location = declared["location"]
                outputs.append(cfg.results_root / str(location["relative_path"]))
            inputs = [
                cfg.strategy_manifest_root_path(),
                done_path.parent / "simulation_workload_plan.json",
            ]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            return False
    state = resolve_stage_state(
        done_path,
        inputs=inputs,
        outputs=outputs,
        cfg=cfg,
        stage="simulation",
        stage_config_sha=_simulation_stage_config_sha(cfg, n_players),
        cache_key_version=cfg.stage_cache_key_version("simulation"),
    )
    if state is not CompletionState.COMPLETE_VALID:
        return False
    if is_v3_config(cfg):
        try:
            for path in outputs:
                if path.name in {"manifest.jsonl", "metrics_manifest.jsonl"}:
                    _publish_simulation_manifest_v3(
                        cfg,
                        path=path,
                        n_players=n_players,
                        report_recovery=report_recovery,
                    )
            return any(
                path.resolve().is_relative_to(cfg.n_dir(n_players).resolve()) for path in outputs
            )
        except (OSError, ValueError, AuthenticatedContractError):
            return False
    try:
        payload = json.loads(done_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return int(payload.get("n_players", -1)) == n_players


def _simulation_stage_config_sha(cfg: AppConfig, n_players: int) -> str:
    """Bind the shared simulation scope to one concrete root/k cell."""

    identity: dict[str, object] = {
        "player_count_dependencies_v1": cfg.partition_dependency_sha("simulation", "player_count"),
        "shuffle_dependencies_v1": cfg.partition_dependency_sha("simulation", "shuffle"),
        "root_seed": int(cfg.sim.seed),
        "n_players": int(n_players),
    }
    if cfg._game_profile_sha256 is not None:
        identity["game_profile_sha256"] = cfg._game_profile_sha256
    return freshness_sha256(identity)


def _simulation_output_sidecar(
    cfg: AppConfig,
    path: Path,
    *,
    n_players: int,
    operation: str,
    sources: Sequence[Path] = (),
    support_counts: Sequence[int] | None = None,
    ordered_roster: str | None = None,
) -> ArtifactSidecar:
    """Build the v3 producer contract used at the original simulation write."""

    counts = list(support_counts if support_counts is not None else [n_players])
    return make_artifact_sidecar(
        cfg,
        path,
        producer="simulation",
        scope=ArtifactScope.DIAGNOSTICS,
        source_scope=ArtifactScope.DIAGNOSTICS,
        operation=operation,
        baseline="tournament_design",
        weighted_quantity="raw_simulation_evidence",
        support_count_role="root_k_simulation",
        uncertainty_method="deterministic_monte_carlo",
        replication_unit="shuffle",
        conditioning="all_attempted_games",
        source_artifacts=sources,
        player_counts=counts,
        required_player_counts=counts,
        missing_cell_policy="fail",
        seed_scope="single_root",
        method_contract={
            "kind": "operation",
            "procedure": operation,
            "parameters": {
                "tournament_method_version": TOURNAMENT_METHOD_VERSION,
                "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
                **({"ordered_builtin_roster_v1": ordered_roster} if ordered_roster else {}),
            },
        },
    )


def _write_parquet_simulation_output(
    cfg: AppConfig,
    table: pa.Table,
    path: Path,
    *,
    n_players: int,
    operation: str,
    sources: Sequence[Path],
) -> None:
    sidecar = _simulation_output_sidecar(
        cfg,
        path,
        n_players=n_players,
        operation=operation,
        sources=sources,
    )
    write_parquet_artifact_atomic(table, path, sidecar=sidecar)


def _write_workload_plan_simulation_output(
    cfg: AppConfig,
    path: Path,
    plan: TournamentWorkloadPlan,
    *,
    n_players: int,
    strategy_manifest: Path,
) -> None:
    """Publish the immutable workload plan once, or validate an exact resume."""

    expected = json.dumps(plan.to_dict(), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if path.exists():
        from farkle.utils.scientific_reuse import compatible_producer

        metadata = validate_authenticated_artifact_metadata(path, cfg=cfg)
        stored = path.read_bytes()
        if (
            hashlib.sha256(stored).hexdigest() != metadata.artifact.content_sha256
            or metadata.method_contract.procedure != "publish_simulation_workload_plan"
            or not compatible_producer(metadata.stage_identity, producer_code_identity(cfg))
            or metadata.stage_identity.immutable_design_identities.get("stage_dependencies_v1")
            != cfg.stage_config_sha("simulation")
            or not exact_direct_sources(path, [strategy_manifest])
        ):
            raise ValueError(f"simulation workload plan identity is stale: {path}")
        if not workload_plan_matches(json.loads(stored), plan):
            raise ValueError(
                f"simulation workload plan does not match current configuration: {path}"
            )
        return
    sidecar = _simulation_output_sidecar(
        cfg,
        path,
        n_players=n_players,
        operation="publish_simulation_workload_plan",
        sources=[strategy_manifest],
    )

    def _write(staged: Path) -> None:
        staged.write_bytes(expected)

    write_artifact_with_sidecar_atomic(path, sidecar, _write)


def workload_plan_matches(original: Mapping[str, Any], plan: TournamentWorkloadPlan) -> bool:
    """Retained performance projections are independent of the planned workload."""
    current = plan.to_dict()
    projections = {"projected_games_per_second", "projected_runtime_seconds"}
    return original.keys() == current.keys() and all(
        original[key] == value for key, value in current.items() if key not in projections
    )


def _completion_output_files(paths: Sequence[Path], done_path: Path) -> list[Path]:
    """Expand directories so a stamp never recursively authenticates itself."""

    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            immutable_manifests = [
                candidate
                for candidate in (
                    path / "manifest.jsonl",
                    path / "metrics_manifest.jsonl",
                )
                if candidate.is_file() and sidecar_path(candidate).is_file()
            ]
            if immutable_manifests:
                files.extend(immutable_manifests)
                continue
            files.extend(
                child
                for child in sorted(path.rglob("*"), key=lambda item: item.as_posix())
                if child.is_file()
                and child.resolve() != done_path.resolve()
                and not child.name.endswith(".sidecar.json")
                and not child.name.startswith("._publication_")
                and not _is_atomic_staging_file(child)
            )
        else:
            files.append(path)
    return list(dict.fromkeys(files))


_ATOMIC_STAGING_PREFIXES = (
    "._tmp_",
    "._artifact_v3_",
    "._sidecar_v3_",
    "._manifest_v3_",
    "._manifest_sidecar_v3_",
)


def _is_atomic_staging_file(path: Path) -> bool:
    """Return whether *path* is a non-durable atomic-write staging file."""

    return path.name.startswith(_ATOMIC_STAGING_PREFIXES)


def _publish_simulation_manifest_v3(
    cfg: AppConfig,
    *,
    path: Path,
    n_players: int,
    expected_entry_count: int | None = None,
    report_recovery: bool = False,
) -> None:
    recovery = observe_recovery("discovery", f"Read inventory {path}") if report_recovery else None
    records = list(iter_manifest(path))
    if recovery is not None:
        recovery.finish()
        recovery = observe_recovery(
            "authentication", f"Check shard metadata in {path}", total_files=len(records)
        )
    entries: list[ManifestEntry] = []
    canonical_records: list[tuple[tuple[int, ...], dict[str, Any]]] = []
    member_science: dict[tuple[int | str, ...], str] = {}
    code = producer_code_identity(cfg)
    from farkle.utils.scientific_reuse import closure, compatible_producer, launch_matches

    members_portable = True
    member_dependencies: dict[str, str] = {}
    reference_metadata = None
    for index, record in enumerate(records):
        raw_path = record.get("path")
        if not isinstance(raw_path, str):
            raise ValueError(f"simulation manifest entry is missing a path: {path}")
        source = Path(raw_path)
        if not source.is_absolute():
            source = path.parent / source
        metadata = validate_authenticated_artifact_metadata(source, cfg=cfg)
        member = closure(metadata.stage_identity)
        members_portable &= launch_matches(metadata.stage_identity)
        if any(
            key in member_dependencies and member_dependencies[key] != value
            for key, value in member.items()
        ):
            raise ValueError(f"Inconsistent scientific dependencies among shards: {source}")
        member_dependencies.update(member)
        if reference_metadata is None:
            reference_metadata = metadata
        schema = metadata.artifact.arrow_schema
        if schema is None:
            raise ValueError(f"simulation shard is not Parquet: {source}")
        coordinate_value = record.get(
            "shuffle_index",
            record.get("process_block_index", record.get("deterministic_batch_id", index)),
        )
        coordinate = (int(coordinate_value),)
        adjacent = sidecar_path(source)
        current_sidecar_sha256 = sha256_file(adjacent)
        declared_identity = {
            "data_sha256": metadata.artifact.content_sha256,
            "byte_length": metadata.artifact.byte_length,
            "sidecar_sha256": current_sidecar_sha256,
            "schema_fingerprint_sha256": schema.fingerprint_sha256,
        }
        if any(record.get(key) != value for key, value in declared_identity.items()):
            raise ValueError(f"simulation manifest identity mismatch for {source}")
        entries.append(
            ManifestEntry(
                coordinate=coordinate,
                canonical_relative_path=source.resolve()
                .relative_to(cfg.results_root.resolve())
                .as_posix(),
                data_sha256=metadata.artifact.content_sha256,
                sidecar_sha256=current_sidecar_sha256,
                schema_fingerprint_sha256=schema.fingerprint_sha256,
            )
        )
        canonical_record = {key: value for key, value in record.items() if key not in {"pid", "ts"}}
        canonical_record.update(declared_identity)
        member_science[coordinate] = freshness_sha256(
            {
                "shard": scientific_artifact_binding(metadata, code),
                "coordinates_and_counts": {
                    key: value
                    for key, value in canonical_record.items()
                    if key not in {"path", "sidecar_sha256"}
                },
            }
        )
        canonical_records.append((coordinate, canonical_record))
        if recovery is not None:
            recovery.advance(files=1)
    if recovery is not None:
        recovery.finish()
    paired = sorted(
        zip(entries, canonical_records, strict=True), key=lambda item: item[0].coordinate
    )
    entries = [item[0] for item in paired]
    ordered_records = [item[1][1] for item in paired]
    inventory_science = freshness_sha256(
        {"ordered_members_v1": [member_science[e.coordinate] for e in entries]}
    )
    if not entries:
        raise ValueError(f"simulation manifest has no authenticated entries: {path}")
    if expected_entry_count is not None and len(entries) != expected_entry_count:
        raise ValueError(
            f"simulation manifest entry count mismatch at {path}: "
            f"expected {expected_entry_count}, found {len(entries)}"
        )
    expected_summary = compute_manifest_root(entries)
    expected_native_sha = hashlib.sha256()
    for record in ordered_records:
        expected_native_sha.update(canonical_json_bytes(record) + b"\n")
    if sidecar_path(path).is_file():
        existing = load_immutable_manifest_sidecar(path)
        if (
            existing.summary == expected_summary
            and existing.stage_identity.immutable_design_identities.get("manifest_members_v1")
            == inventory_science
            and existing.manifest_sha256 == expected_native_sha.hexdigest()
            and sha256_file(path) == existing.manifest_sha256
            and reference_metadata is not None
            and compatible_producer(existing.stage_identity, code)
            and existing.stage_identity.stage_config
            == reference_metadata.stage_identity.stage_config
        ):
            return
        raise ValueError(f"existing simulation manifest identity is stale: {path}")
    publish_native_manifest_v3(
        path,
        cfg=cfg,
        stage_key="simulation",
        entries=entries,
        source_paths=[],
        operation="publish_simulation_shard_manifest",
        player_counts=[n_players],
        native_records=ordered_records,
        member_science=inventory_science,
        member_dependencies=member_dependencies if members_portable else None,
    )


def _publish_simulation_outputs_v3(
    cfg: AppConfig,
    *,
    n_players: int,
    outputs: Sequence[Path],
    done_path: Path,
    allow_unsealed_outputs: bool,
    rewritten_outputs: Sequence[Path] = (),
    num_shuffles: int | None = None,
    shuffles_per_batch: int | None = None,
) -> None:
    if not is_v3_config(cfg):
        return
    files = _completion_output_files(outputs, done_path)
    _ = allow_unsealed_outputs, rewritten_outputs
    manifests = [
        path
        for path in files
        if path.suffix.lower() == ".jsonl"
        and path.name in {"manifest.jsonl", "metrics_manifest.jsonl"}
    ]
    ordinary = [path for path in files if path not in manifests]
    for path in sorted(ordinary, key=lambda item: item.as_posix()):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"simulation output is missing or empty: {path}")
        adjacent = sidecar_path(path)
        if not adjacent.is_file():
            raise ValueError(
                "refusing to promote pre-existing simulation bytes: simulation "
                f"producer did not publish an authenticated v3 sidecar for {path}"
            )
        try:
            validate_authenticated_artifact_metadata(path, cfg=cfg)
        except ArtifactMismatchError as exc:
            raise ValueError(f"simulation output metadata is invalid: {path}") from exc
    for manifest in manifests:
        expected_entry_count = None
        if num_shuffles is not None:
            expected_entry_count = (
                num_shuffles
                if manifest.name == "manifest.jsonl"
                else (
                    None
                    if shuffles_per_batch is None
                    else (num_shuffles + shuffles_per_batch - 1) // shuffles_per_batch
                )
            )
        _publish_simulation_manifest_v3(
            cfg,
            path=manifest,
            n_players=n_players,
            expected_entry_count=expected_entry_count,
        )


def write_simulation_done(
    cfg: AppConfig,
    n_players: int,
    *,
    num_shuffles: int,
    shuffles_per_batch: int,
    n_strategies: int,
    outputs: Sequence[Path],
    allow_unsealed_v3_outputs: bool = False,
    rewritten_outputs: Sequence[Path] = (),
) -> Path:
    """Write a completion marker for a per-N simulation run."""
    done_path = simulation_done_path(cfg, n_players)
    metadata: dict[str, object] = {
        "n_players": n_players,
        "seed": cfg.sim.seed,
        "root_seed": cfg.sim.seed,
        "k": n_players,
        "num_shuffles": num_shuffles,
        "shuffle_index_start": 0,
        "shuffle_index_end": num_shuffles - 1,
        "deterministic_batch_count": (
            (num_shuffles + shuffles_per_batch - 1) // shuffles_per_batch
        ),
        "shuffles_per_batch": shuffles_per_batch,
        "rng_scheme_version": urandom.RNG_SCHEME_VERSION,
        "rng_purpose_namespace": int(urandom.RandomPurpose.TOURNAMENT_SHUFFLE),
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        "tournament_method_version": TOURNAMENT_METHOD_VERSION,
        "n_strategies": n_strategies,
    }
    if cfg._game_profile_sha256 is not None:
        metadata["game_profile_sha256"] = cfg._game_profile_sha256
    _publish_simulation_outputs_v3(
        cfg,
        n_players=n_players,
        outputs=outputs,
        done_path=done_path,
        allow_unsealed_outputs=allow_unsealed_v3_outputs,
        rewritten_outputs=rewritten_outputs,
        num_shuffles=num_shuffles,
        shuffles_per_batch=shuffles_per_batch,
    )
    output_files = _completion_output_files(outputs, done_path)
    immutable_inputs = [
        cfg.strategy_manifest_root_path(),
        done_path.parent / "simulation_workload_plan.json",
    ]
    write_stage_done(
        done_path,
        inputs=immutable_inputs,
        outputs=output_files,
        cfg=cfg,
        stage="simulation",
        stage_config_sha=_simulation_stage_config_sha(cfg, n_players),
        cache_key_version=cfg.stage_cache_key_version("simulation"),
        metadata=metadata,
    )
    return done_path


def _resolve_per_n_output_dir(
    cfg: AppConfig,
    raw_value: Path | None,
    n_players: int,
) -> Path | None:
    """Resolve a per-player-count output directory from config or placeholders.

    Args:
        cfg: Application config supplying the results root.
        raw_value: Optional configured directory template or relative path.
        n_players: Player count used to fill placeholders or default prefixes.

    Returns:
        Resolved output directory, or ``None`` when no override is configured.
    """
    if not raw_value:
        return None

    raw_str = str(raw_value)
    placeholders = {
        "n": n_players,
        "n_players": n_players,
        "p": f"{n_players}p",
    }
    used_placeholders = False
    try:
        formatted_str = raw_str.format(**placeholders)
        used_placeholders = formatted_str != raw_str
    except KeyError:
        formatted_str = raw_str

    out_path = Path(formatted_str)
    if not used_placeholders:
        tail = out_path.name
        prefix = f"{n_players}p"
        if tail and not tail.startswith(prefix):
            out_path = out_path.parent / f"{prefix}_{tail}"
    if out_path.is_absolute():
        return out_path

    n_dir = cfg.results_root / f"{n_players}_players"
    return n_dir / out_path


def _strategy_manifest_digest(manifest: pd.DataFrame) -> str:
    """Compute a stable digest for the strategy manifest contents.

    Args:
        manifest: Strategy manifest frame to fingerprint.

    Returns:
        SHA-256 digest of the CSV-serialized manifest.
    """
    payload = manifest.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_manifest_matches(manifest: pd.DataFrame, path: Path, *, label: str) -> None:
    """Ensure an on-disk manifest matches the expected in-memory manifest.

    Args:
        manifest: Expected manifest frame for the current run.
        path: Existing manifest parquet path to validate.
        label: Human-readable manifest label for error messages.
    """
    if not path.exists():
        return

    existing = pd.read_parquet(path)
    if set(existing.columns) != set(manifest.columns):
        raise ValueError(f"{label} manifest schema mismatch at {path}; use a new output root.")
    existing = existing[manifest.columns]
    if not existing.equals(manifest):
        raise ValueError(f"{label} manifest mismatch at {path}; use a new output root.")


def _safe_cleanup_path(path: Path, *, directory: bool = False) -> os.stat_result | None:
    """Check the lexical path before resolution can hide a filesystem alias."""
    path = path.absolute()
    if ".." in path.parts or path.is_reserved() or path.resolve() != path:
        raise ValueError(f"Unsafe simulation cleanup path: {path}")
    for component in (*reversed(path.parents), path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            return None
        if (
            stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError(f"Unsafe simulation cleanup link or reparse point: {component}")
        if component != path or directory:
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"Unsafe simulation cleanup directory: {component}")
        elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"Unsafe simulation cleanup file: {component}")
    return info


def _force_output_directories(cfg: AppConfig, n_players: int) -> tuple[Path | None, Path | None]:
    row_dir = _resolve_row_output_dir(cfg, n_players)
    metric_dir = _resolve_metric_chunk_dir(cfg, n_players)
    root = cfg.results_root.absolute()
    n_dir = cfg.n_dir(n_players)
    _safe_cleanup_path(n_dir / "_execution" / "._publication_simulation.lock")
    for directory in (row_dir, metric_dir):
        if directory is not None:
            _safe_cleanup_path(directory, directory=True)
            if not directory.absolute().is_relative_to(root):
                # Authenticated simulation publications require this same boundary.
                raise ValueError(
                    f"Simulation output directory is outside the canonical run root: {directory}"
                )
    return row_dir, metric_dir


def _simulation_cell_files(
    cfg: AppConfig, n_players: int, row_dir: Path | None, metric_dir: Path | None
) -> set[Path]:
    n_dir = cfg.n_dir(n_players)
    candidates = {
        cfg.checkpoint_path(n_players),
        n_dir / f"{n_players}p_checkpoint.parquet",
        cfg.metrics_path(n_players),
        cfg.simulation_workload_plan_path(n_players),
        simulation_done_path(cfg, n_players),
    }
    for directory, manifest, pattern in (
        (row_dir, "manifest.jsonl", rf"rows_{cfg.sim.seed}_{n_players}p_[0-9]{{12,}}[.]parquet"),
        (metric_dir, "metrics_manifest.jsonl", r"metrics_[0-9]{6,}[.]parquet"),
    ):
        if directory is None:
            continue
        candidates.add(directory / manifest)
        if directory.is_dir():
            for child in directory.iterdir():
                name = child.name.lower() if os.name == "nt" else child.name
                name = name.removesuffix(".sidecar.json")
                if re.fullmatch(pattern, name):
                    candidates.add(directory / name)
    return candidates


def _validate_force_inventory(
    cfg: AppConfig, plan: TournamentWorkloadPlan, row_dir: Path | None, metric_dir: Path | None
) -> tuple[bool, dict[Path, tuple[os.stat_result, str]]]:
    """Retain sealed evidence and admit only exactly identified unsealed controls."""
    found: dict[Path, os.stat_result] = {}
    sidecars: set[Path] = set()
    files = _simulation_cell_files(cfg, plan.k, row_dir, metric_dir)
    for path in sorted(files):
        if (info := _safe_cleanup_path(path)) is not None:
            found[path] = info
        if (
            path.name != "simulation.done.json"
            and _safe_cleanup_path(sidecar_path(path)) is not None
        ):
            sidecars.add(path)

    workload = cfg.simulation_workload_plan_path(plan.k)
    if simulation_done_path(cfg, plan.k) in found:
        raise ValueError(
            "Existing simulation completion failed validation; preserve it and use a new output root"
        )
    sealed = bool(sidecars)
    if sealed:
        _validate_sealed_force_inventory(cfg, plan.k, found.keys(), sidecars)
    owned: dict[Path, tuple[os.stat_result, str]] = {}
    for path, info in sorted(found.items()):
        if path in sidecars:
            continue
        if path == workload and 0 < info.st_size <= 64 * 1024:
            try:
                with path.open("rb") as stream:
                    contents = stream.read(info.st_size + 1)
                old = json.loads(contents)
                old_plan = plan_tournament_workload(
                    root_seed=plan.root_seed,
                    k=plan.k,
                    strategy_count=plan.strategy_count,
                    resolution_delta=plan.resolution_delta,
                    confidence=plan.confidence,
                    batch_count=plan.batch_count,
                    min_shuffles_per_batch=plan.shuffles_per_batch,
                    shuffle_cap=old["shuffle_cap"],
                    projected_games_per_second=old["projected_games_per_second"],
                )
                if (
                    len(contents) == info.st_size
                    and old_plan.cap_exceeded
                    and canonical_json_bytes(old) == canonical_json_bytes(old_plan.to_dict())
                ):
                    owned[path] = (info, hashlib.sha256(contents).hexdigest())
                    continue
            except (OSError, ValueError, TypeError, KeyError, OverflowError):
                pass
        # The checked resume path authenticates append inventories and recovers
        # interrupted publications from immutable inputs. A standalone filename
        # or record without its seal does not authorize a forced deletion.
        if workload in sidecars and path.name in {"manifest.jsonl", "metrics_manifest.jsonl"}:
            continue
        raise ValueError(
            f"Cannot establish simulation cleanup ownership for {path}; use a new output root"
        )
    return sealed, owned


def _validate_sealed_force_inventory(
    cfg: AppConfig, n_players: int, files: Collection[Path], sidecars: set[Path]
) -> None:
    """Verify occupied output names before handing the cell to normal resume."""
    from farkle.utils.scientific_reuse import compatible_producer

    n_dir = cfg.n_dir(n_players)
    workload = cfg.simulation_workload_plan_path(n_players)
    roster = cfg.strategy_manifest_root_path()
    for source in (roster, workload):
        _safe_cleanup_path(source)
        _safe_cleanup_path(sidecar_path(source))
    fixed_operations = {
        cfg.checkpoint_path(n_players): "publish_simulation_checkpoint",
        n_dir / f"{n_players}p_checkpoint.parquet": "publish_simulation_checkpoint_summary",
        cfg.metrics_path(n_players): "publish_simulation_metrics_summary",
        workload: "publish_simulation_workload_plan",
    }
    code = producer_code_identity(cfg)
    dependencies = cfg.stage_config_sha("simulation")
    for path in sorted(sidecars):
        if path not in files:
            raise ValueError(f"Missing data for sealed simulation output: {path}")
        location = derive_canonical_location(cfg, path, stage_key="simulation")
        if path.name in {"manifest.jsonl", "metrics_manifest.jsonl"}:
            manifest = load_immutable_manifest_sidecar(path)
            stage = manifest.stage_identity
            valid = manifest.location == location and sha256_file(path) == manifest.manifest_sha256
        else:
            artifact = validate_authenticated_artifact_unbound(
                path, expected_location=location, validate_provenance=False
            )
            stage = artifact.stage_identity
            operation = fixed_operations.get(path) or (
                "publish_simulation_row_shard"
                if path.name.lower().startswith("rows_")
                else "publish_simulation_metric_chunk"
            )
            valid = (
                artifact.method_contract.procedure == operation
                and artifact.method_contract.player_counts == (n_players,)
                and exact_direct_sources(path, [roster] if path == workload else [roster, workload])
            )
        if (
            not valid
            or stage.immutable_design_identities.get("stage_dependencies_v1") != dependencies
            or not compatible_producer(stage, code)
        ):
            raise ValueError(
                f"Incompatible sealed simulation output: {path}; use a new output root"
            )


def _purge_simulation_outputs(inventory: Mapping[Path, tuple[os.stat_result, str]]) -> None:
    """Unlink the prevalidated file inventory without changing permissions."""
    for path, (inspected, digest) in inventory.items():
        current = _safe_cleanup_path(path)
        if (
            current is None
            or not os.path.samestat(current, inspected)
            or current.st_size != inspected.st_size
            or current.st_mtime_ns != inspected.st_mtime_ns
            or sha256_file(path) != digest
        ):
            raise ValueError(f"Simulation cleanup target changed after validation: {path}")
    for path in inventory:
        os.unlink(path)


def _has_existing_outputs(
    *,
    n_dir: Path,
    n_players: int,
    ckpt_path: Path,
    row_dir: Path | None,
    metric_chunk_dir: Path | None,
) -> bool:
    """Check whether any outputs already exist for a player-count run.

    Args:
        n_dir: Base directory for the current player-count outputs.
        n_players: Player count for the output family.
        ckpt_path: Checkpoint path to inspect.
        row_dir: Optional row-shard directory.
        metric_chunk_dir: Optional metric-chunk directory.

    Returns:
        ``True`` when any checkpoint, manifest, or shard artifacts already exist.
    """
    candidates = [
        ckpt_path,
        n_dir / f"{n_players}p_checkpoint.parquet",
        n_dir / f"{n_players}p_metrics.parquet",
    ]
    if any(path.exists() for path in candidates):
        return True
    if (n_dir / STRATEGY_MANIFEST_NAME).exists():
        return True
    if row_dir is not None and row_dir.exists():
        if (row_dir / "manifest.jsonl").exists():
            return True
        if next(row_dir.glob("rows_*.parquet"), None) is not None:
            return True
        if (row_dir / STRATEGY_MANIFEST_NAME).exists():
            return True
    if metric_chunk_dir is not None and metric_chunk_dir.exists():
        if (metric_chunk_dir / "metrics_manifest.jsonl").exists():
            return True
        if next(metric_chunk_dir.glob("metrics_*.parquet"), None) is not None:
            return True
    return False


def _has_unsealed_simulation_outputs(
    *,
    cfg: AppConfig,
    n_players: int,
    row_dir: Path | None,
    metric_chunk_dir: Path | None,
    strategy_manifest_path: Path,
    recoverable_checkpoint: Path | None = None,
) -> bool:
    """Detect pre-v3 bytes that must never be blessed by a later completion."""

    candidates = _simulation_cell_files(cfg, n_players, row_dir, metric_chunk_dir)
    candidates.add(strategy_manifest_path)

    def _requires_preexisting_seal(path: Path) -> bool:
        if recoverable_checkpoint is not None and path.resolve() == recoverable_checkpoint:
            return False
        if not path.is_file() or sidecar_path(path).is_file():
            return False
        if path.name in {"manifest.jsonl", "metrics_manifest.jsonl", "simulation.done.json"}:
            return False
        return not (
            path.suffix.lower() == ".parquet"
            and (path.name.startswith("rows_") or path.name.startswith("metrics_"))
        )

    return any(_requires_preexisting_seal(path) for path in candidates)


def _validate_manifest_record_identity_v3(
    cfg: AppConfig,
    manifest_path: Path,
    record: Mapping[str, Any],
) -> AuthenticatedSidecar:
    """Reject a partial manifest record not backed by one exact sealed shard."""

    raw_path = record.get("path")
    if not isinstance(raw_path, str):
        raise ValueError(f"simulation manifest entry is missing a path: {manifest_path}")
    shard = Path(raw_path)
    if not shard.is_absolute():
        shard = manifest_path.parent / shard
    metadata = validate_authenticated_artifact_metadata(shard, cfg=cfg)
    schema = metadata.artifact.arrow_schema
    expected = {
        "data_sha256": metadata.artifact.content_sha256,
        "byte_length": metadata.artifact.byte_length,
        "sidecar_sha256": sha256_file(sidecar_path(shard)),
        "schema_fingerprint_sha256": (None if schema is None else schema.fingerprint_sha256),
    }
    if schema is None or any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"simulation manifest shard identity mismatch: {shard}")
    return metadata


def _validate_resume_outputs(
    *,
    cfg: AppConfig,
    n_players: int,
    n_shuffles: int,
    strategies_manifest: pd.DataFrame,
    ckpt_path: Path,
    row_dir: Path | None,
    metric_chunk_dir: Path | None,
    require_authenticated_checkpoint: bool = False,
) -> dict[str, Any] | None:
    """Validate resume-mode artifacts against the expected run metadata.

    Args:
        cfg: Application config driving the resumed run.
        n_players: Player count being resumed.
        n_shuffles: Expected number of shuffles for the run.
        strategies_manifest: Strategy manifest expected for the run.
        ckpt_path: Checkpoint path whose metadata should align with the manifest.
        row_dir: Optional row-shard directory to validate.
        metric_chunk_dir: Optional metric-chunk directory to validate.
    """
    workload_plan = _plan_workload_from_config(
        cfg,
        n_strategies=len(strategies_manifest),
        n_players=n_players,
    )
    expected_meta = {
        "n_players": n_players,
        "num_shuffles": n_shuffles,
        "global_seed": cfg.sim.seed,
        "n_strategies": len(strategies_manifest),
        "strategy_manifest_sha": _strategy_manifest_digest(strategies_manifest),
        "rng_scheme_version": urandom.RNG_SCHEME_VERSION,
        "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        "tournament_method_version": TOURNAMENT_METHOD_VERSION,
        "rng_bit_generator": "PCG64DXSM",
        **_workload_checkpoint_metadata(workload_plan),
    }
    if cfg._game_profile_sha256 is not None:
        expected_meta["game_profile_sha256"] = cfg._game_profile_sha256
    root_manifest_path = cfg.strategy_manifest_root_path()
    if root_manifest_path.exists():
        if is_v3_config(cfg) and sidecar_path(root_manifest_path).is_file():
            validate_authenticated_artifact_metadata(root_manifest_path, cfg=cfg)
        _validate_manifest_matches(strategies_manifest, root_manifest_path, label="Strategy")
    else:
        raise ValueError(
            f"Canonical strategy manifest missing at {root_manifest_path}; rerun with --force."
        )

    has_row_manifest = row_dir is not None and (row_dir / "manifest.jsonl").exists()
    has_metrics_manifest = (
        metric_chunk_dir is not None and (metric_chunk_dir / "metrics_manifest.jsonl").exists()
    )
    payload: dict[str, Any] | None = None
    checkpoint_error: Exception | None = None
    checkpoint_sources: set[str] = set()
    if ckpt_path.exists():
        checkpoint_progress = observe_recovery(
            "authentication", f"Check checkpoint {ckpt_path}", total_files=1
        )
        try:
            candidate = tournament_mod._read_checkpoint(
                ckpt_path,
                cfg=(
                    cfg
                    if require_authenticated_checkpoint or sidecar_path(ckpt_path).is_file()
                    else None
                ),
                validate_sources=require_authenticated_checkpoint,
                source_identities=checkpoint_sources,
            )
            meta = candidate.get("meta")
            if not isinstance(meta, Mapping):
                raise ValueError(f"Checkpoint metadata missing at {ckpt_path}; rerun with --force.")
            current = dict(expected_meta)
            if require_authenticated_checkpoint or "simulation_stage_config_sha" in meta:
                current["simulation_stage_config_sha"] = _simulation_stage_config_sha(
                    cfg, n_players
                )
            for key, expected in current.items():
                if meta.get(key) != expected:
                    raise ValueError(
                        f"Checkpoint metadata mismatch for {key} at {ckpt_path}; rerun with --force."
                    )
            if require_authenticated_checkpoint:
                tournament_mod._coerce_counter(
                    candidate.get("win_totals", candidate), candidate.get("outcome_counts")
                )
            payload = candidate
            if checkpoint_progress is not None:
                checkpoint_progress.advance(files=1)
                checkpoint_progress.finish()
        except (
            AuthenticatedContractError,
            OSError,
            pickle.UnpicklingError,
            EOFError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            checkpoint_error = exc
    elif not (has_row_manifest or has_metrics_manifest):
        return None

    if checkpoint_error is not None and not require_authenticated_checkpoint:
        raise checkpoint_error
    if checkpoint_error is not None and not (has_row_manifest or has_metrics_manifest):
        raise ValueError(
            f"Checkpoint metadata mismatch at {ckpt_path}; no authenticated replay inventory."
        ) from checkpoint_error

    def require_replay_context() -> Callable[[Path, Mapping[str, Any], AuthenticatedSidecar], None]:
        from farkle.utils.scientific_identity import producer_config_identity
        from farkle.utils.scientific_reuse import compatible_producer

        stage_config = producer_config_identity(cfg, "simulation")
        dependencies = cfg.stage_config_sha("simulation")
        code = producer_code_identity(cfg)
        if payload is not None and checkpoint_sources:
            source_identities = checkpoint_sources
        else:
            template = _simulation_output_sidecar(
                cfg,
                ckpt_path,
                n_players=n_players,
                operation="publish_simulation_checkpoint",
                sources=[root_manifest_path, ckpt_path.parent / "simulation_workload_plan.json"],
            )
            source_identities = {item.sha256 for item in capture_v3_inputs(template).sources}

        def check(path: Path, record: Mapping[str, Any], metadata: AuthenticatedSidecar) -> None:
            shard = tournament_mod._resolve_manifest_record_path(path, record)
            stage = metadata.stage_identity
            if (
                stage.stage_config.sha256 != stage_config.sha256
                or stage.immutable_design_identities.get("stage_dependencies_v1") != dependencies
                or not compatible_producer(stage, code)
                or {item.sha256 for item in metadata.source_artifacts} != source_identities
                or (checkpoint_error is not None and sha256_file(shard) != record["data_sha256"])
            ):
                raise ValueError(f"simulation shard recovery identity mismatch: {shard}")

        return check

    checkpoint_meta = payload.get("meta", {}) if payload is not None else {}
    verify_replay_shard = (
        require_replay_context()
        if checkpoint_error is not None
        or "partial_aggregate" in checkpoint_meta
        or (require_authenticated_checkpoint and (has_row_manifest or has_metrics_manifest))
        else None
    )

    def require_manifest_seal(path: Path) -> None:
        if require_authenticated_checkpoint and sidecar_path(path).is_file():
            manifest = load_immutable_manifest_sidecar(path)
            manifest.location.require_path(cfg, path)
            if sha256_file(path) != manifest.manifest_sha256:
                raise ValueError(f"sealed simulation manifest bytes differ: {path}")

    expected_seeds = None
    replay_has_sealed_shards = False
    if row_dir is not None:
        manifest_path = row_dir / "manifest.jsonl"
        if manifest_path.exists():
            recovery = observe_recovery(
                "discovery",
                f"Discover and check row shard metadata in {manifest_path}",
                total_coordinates=n_shuffles,
            )
            require_manifest_seal(manifest_path)
            expected_seed_by_index = {
                index: urandom.coordinate_seed(
                    urandom.RandomPurpose.TOURNAMENT_SHUFFLE,
                    root_seed=cfg.sim.seed,
                    k=n_players,
                    shuffle_index=index,
                    dtype=np.uint32,
                )
                for index in range(n_shuffles)
            }
            expected_seeds = set(expected_seed_by_index.values())
            seen_indices: set[int] = set()
            seen_paths: set[Path] = set()
            duplicates = 0
            unexpected = 0
            coordinate_errors = 0
            all_sealed = True
            for record in iter_manifest(manifest_path):
                record_path = manifest_path.parent / str(record.get("path", ""))
                if require_authenticated_checkpoint:
                    resolved = record_path.resolve()
                    if resolved in seen_paths:
                        raise ValueError(f"Duplicate shard path in {manifest_path}: {record_path}")
                    seen_paths.add(resolved)
                if is_v3_config(cfg) and (
                    "data_sha256" in record or sidecar_path(record_path).is_file()
                ):
                    metadata = _validate_manifest_record_identity_v3(cfg, manifest_path, record)
                    if verify_replay_shard is not None:
                        verify_replay_shard(manifest_path, record, metadata)
                else:
                    all_sealed = False
                seed_val = record.get("shuffle_seed")
                index_val = record.get("shuffle_index")
                if seed_val is None or index_val is None:
                    coordinate_errors += 1
                    continue
                try:
                    seed_int = int(seed_val)
                    index_int = int(index_val)
                except (TypeError, ValueError):
                    coordinate_errors += 1
                    continue
                if recovery is not None:
                    recovery.advance(
                        files=int("data_sha256" in record),
                        coordinates=int(index_int not in seen_indices),
                    )
                if index_int in seen_indices:
                    duplicates += 1
                seen_indices.add(index_int)
                if (
                    seed_int not in expected_seeds
                    or expected_seed_by_index.get(index_int) != seed_int
                ):
                    unexpected += 1
                expected_batch = index_int // workload_plan.shuffles_per_batch
                if (
                    record.get("root_seed") != cfg.sim.seed
                    or record.get("n_players") != n_players
                    or record.get("deterministic_batch_id") != expected_batch
                    or record.get("rng_scheme_version") != urandom.RNG_SCHEME_VERSION
                    or record.get("rng_purpose_namespace")
                    != int(urandom.RandomPurpose.TOURNAMENT_SHUFFLE)
                    or record.get("outcome_schema_version") != OUTCOME_SCHEMA_VERSION
                    or record.get("tournament_method_version") != TOURNAMENT_METHOD_VERSION
                    or record.get("game_profile_sha256") != cfg._game_profile_sha256
                ):
                    coordinate_errors += 1
            if duplicates:
                raise ValueError(
                    f"Duplicate shuffle entries detected in {manifest_path}; rerun with --force."
                )
            if unexpected or coordinate_errors:
                raise ValueError(f"Row manifest mismatch at {manifest_path}; rerun with --force.")
            replay_has_sealed_shards = all_sealed and bool(seen_indices)
            if recovery is not None:
                recovery.finish()

    if metric_chunk_dir is not None:
        metrics_manifest = metric_chunk_dir / "metrics_manifest.jsonl"
        if metrics_manifest.exists():
            recovery = observe_recovery(
                "discovery",
                f"Discover and check metric block metadata in {metrics_manifest}",
                total_coordinates=n_shuffles,
            )
            require_manifest_seal(metrics_manifest)
            coordinate_errors = 0
            seen_batches: set[int] = set()
            seen_shuffle_indices: set[int] = set()
            seen_paths = set()
            duplicates = 0
            all_sealed = True
            for record in iter_manifest(metrics_manifest):
                record_path = metrics_manifest.parent / str(record.get("path", ""))
                if require_authenticated_checkpoint:
                    resolved = record_path.resolve()
                    if resolved in seen_paths:
                        raise ValueError(
                            f"Duplicate shard path in {metrics_manifest}: {record_path}"
                        )
                    seen_paths.add(resolved)
                if is_v3_config(cfg) and (
                    "data_sha256" in record or sidecar_path(record_path).is_file()
                ):
                    metadata = _validate_manifest_record_identity_v3(cfg, metrics_manifest, record)
                    if verify_replay_shard is not None:
                        verify_replay_shard(metrics_manifest, record, metadata)
                else:
                    all_sealed = False
                batch_value = record.get("deterministic_batch_id")
                block_value = record.get("process_block_index")
                start_value = record.get("shuffle_index_start")
                end_value = record.get("shuffle_index_end")
                count_value = record.get("shuffle_count")
                indices_value = record.get("shuffle_indices")
                seeds_value = record.get("shuffle_seeds")
                if (
                    batch_value is None
                    or block_value is None
                    or start_value is None
                    or end_value is None
                    or count_value is None
                    or not isinstance(indices_value, list)
                    or not isinstance(seeds_value, list)
                ):
                    coordinate_errors += 1
                    continue
                try:
                    batch_id = int(batch_value)
                    process_block_index = int(block_value)
                    start = int(start_value)
                    end = int(end_value)
                    count = int(count_value)
                    shuffle_indices = [int(value) for value in indices_value]
                    shuffle_seeds = [int(value) for value in seeds_value]
                except (TypeError, ValueError):
                    coordinate_errors += 1
                    continue
                new_coordinates = (
                    len(set(shuffle_indices) - seen_shuffle_indices) if recovery is not None else 0
                )
                if batch_id in seen_batches:
                    duplicates += 1
                seen_batches.add(batch_id)
                if seen_shuffle_indices.intersection(shuffle_indices):
                    duplicates += 1
                seen_shuffle_indices.update(shuffle_indices)
                if recovery is not None:
                    recovery.advance(
                        files=int("data_sha256" in record), coordinates=new_coordinates
                    )
                expected_seeds_for_indices = [
                    urandom.coordinate_seed(
                        urandom.RandomPurpose.TOURNAMENT_SHUFFLE,
                        root_seed=cfg.sim.seed,
                        k=n_players,
                        shuffle_index=index,
                        dtype=np.uint32,
                    )
                    for index in shuffle_indices
                ]
                if (
                    record.get("root_seed") != cfg.sim.seed
                    or record.get("n_players") != n_players
                    or record.get("rng_scheme_version") != urandom.RNG_SCHEME_VERSION
                    or record.get("rng_purpose_namespace")
                    != int(urandom.RandomPurpose.TOURNAMENT_SHUFFLE)
                    or record.get("outcome_schema_version") != OUTCOME_SCHEMA_VERSION
                    or record.get("tournament_method_version") != TOURNAMENT_METHOD_VERSION
                    or record.get("game_profile_sha256") != cfg._game_profile_sha256
                    or process_block_index != batch_id + 1
                    or not shuffle_indices
                    or len(set(shuffle_indices)) != len(shuffle_indices)
                    or shuffle_indices != sorted(shuffle_indices)
                    or start != shuffle_indices[0]
                    or end != shuffle_indices[-1]
                    or count != len(shuffle_indices)
                    or any(index < 0 or index >= n_shuffles for index in shuffle_indices)
                    or any(
                        index // workload_plan.shuffles_per_batch != batch_id
                        for index in shuffle_indices
                    )
                    or shuffle_seeds != expected_seeds_for_indices
                ):
                    coordinate_errors += 1
            if duplicates:
                raise ValueError(
                    f"Duplicate process-block entries detected in {metrics_manifest}; "
                    "rerun with --force."
                )
            if coordinate_errors:
                raise ValueError(
                    f"Metrics manifest mismatch at {metrics_manifest}; rerun with --force."
                )
            if not has_row_manifest:
                replay_has_sealed_shards = all_sealed and bool(seen_batches)
            if recovery is not None:
                recovery.finish()

    if checkpoint_error is not None:
        if not replay_has_sealed_shards:
            raise ValueError(
                f"Checkpoint metadata mismatch at {ckpt_path}; no authenticated replay inventory."
            ) from checkpoint_error
        LOGGER.warning(
            "Ignoring checkpoint %s and replaying sealed shards: %s", ckpt_path, checkpoint_error
        )
    return payload


def run_tournament(
    cfg: AppConfig,
    *,
    force: bool = False,
    oracle_game_profile: GameProfile | None = None,
) -> int:
    """Top-level dispatcher that runs single-N or multi-N based on the config.

    - If ``sim.n_players_list`` has one element, runs that N and returns total games (int).
    - If it has multiple elements, runs them all and returns the **sum** of total games.
    """
    _bind_game_profile(cfg, oracle_game_profile)
    return _dispatch_tournament(cfg, force=force, oracle_game_profile=oracle_game_profile)


def _bind_game_profile(cfg: AppConfig, oracle_game_profile: GameProfile | None) -> None:
    from farkle.utils.scientific_invocation import current

    profile_sha256 = oracle_game_profile.sha256 if oracle_game_profile is not None else None
    if cfg._game_profile_sha256 != profile_sha256:
        if cfg._run_lineage_sha256 is not None or current() is not None:
            raise ValueError("game-profile identity does not match the authenticated run lineage")
        cfg._game_profile_sha256 = profile_sha256


@stage_action("simulation", also=())
def _dispatch_tournament(
    cfg: AppConfig, *, force: bool, oracle_game_profile: GameProfile | None
) -> int:
    validate_workload(cfg)
    n_vals = list(cfg.sim.n_players_list)

    if len(n_vals) == 1:
        n = n_vals[0]
        LOGGER.info(
            "Running single-N tournament",
            extra={
                "stage": "simulation",
                "n_players": n,
                "resolution_delta": cfg.screening.resolution_delta,
                "batch_count": cfg.batching.target_batches,
                "seed": cfg.sim.seed,
                "n_jobs": cfg.sim.n_jobs,
                "expanded_metrics": cfg.sim.expanded_metrics,
            },
        )
        if oracle_game_profile is None:
            return run_single_n(cfg, n, force=force)
        return run_single_n(cfg, n, force=force, oracle_game_profile=oracle_game_profile)

    LOGGER.info(
        "Running multi-N tournaments",
        extra={
            "stage": "simulation",
            "n_players_list": n_vals,
            "resolution_delta": cfg.screening.resolution_delta,
            "batch_count": cfg.batching.target_batches,
            "seed": cfg.sim.seed,
            "n_jobs": cfg.sim.n_jobs,
            "expanded_metrics": cfg.sim.expanded_metrics,
        },
    )
    if oracle_game_profile is None:
        totals = run_multi(cfg, player_counts=n_vals, force=force)
    else:
        totals = run_multi(
            cfg,
            player_counts=n_vals,
            force=force,
            oracle_game_profile=oracle_game_profile,
        )
    return int(sum(totals.values()))


def simulation_roster_manifest(
    cfg: AppConfig, strategies: list[ThresholdStrategy] | None = None
) -> tuple[pd.DataFrame, str | None]:
    """Validate the requested ordered roster against any sealed root mapping."""
    from farkle.simulation.strategies import StopAtStrategy

    if strategies is None:
        strategies, _, _ = _resolve_strategies(cfg, None)
    ordered_roster = (
        freshness_sha256(
            {
                "ordered_roster": [
                    (
                        type(s).__name__,
                        s.strategy_id,
                        s.heuristic if isinstance(s, StopAtStrategy) else None,
                    )
                    for s in strategies
                ]
            }
        )
        if all(
            type(s) in (ThresholdStrategy, StopAtStrategy) and s.strategy_id is not None
            for s in strategies
        )
        else None
    )
    manifest = build_strategy_manifest(strategies)
    manifest_path = cfg.strategy_manifest_root_path()
    if manifest_path.exists() or sidecar_path(manifest_path).exists():
        from farkle.utils.frozen_h2h import validated_root_strategy_manifest

        original = validated_root_strategy_manifest(cfg)
        if original.stage_identity.immutable_design_identities.get(
            "stage_dependencies_v1"
        ) != cfg.stage_config_sha("simulation"):
            raise ValueError(
                f"Incompatible strategy manifest producer at {manifest_path}; use a new output root"
            )
        old_order = original.stage_identity.immutable_design_identities.get(
            "ordered_builtin_roster_v1"
        )
        if old_order is not None and old_order != ordered_roster:
            raise ValueError("Ordered strategy roster differs; use a new output root")
        _validate_manifest_matches(manifest, manifest_path, label="Strategy")
    return manifest, ordered_roster


def _pending_simulation_publications(cfg: AppConfig, n: int) -> dict[Path, str]:
    targets = {
        cfg.checkpoint_path(n): "publish_simulation_checkpoint",
        cfg.n_dir(n) / f"{n}p_checkpoint.parquet": "publish_simulation_checkpoint_summary",
        cfg.metrics_path(n): "publish_simulation_metrics_summary",
    }
    return {
        path: operation for path, operation in targets.items() if pending_publication_recovery(path)
    }


def _recover_simulation_publications(cfg: AppConfig, n: int) -> None:
    for path, operation in _pending_simulation_publications(cfg, n).items():
        template = _simulation_output_sidecar(
            cfg,
            path,
            n_players=n,
            operation=operation,
            sources=[cfg.strategy_manifest_root_path(), cfg.simulation_workload_plan_path(n)],
        )
        recover_simulation_output(cfg, path, template, check_path=_safe_cleanup_path)


def run_single_n(
    cfg: AppConfig,
    n: int,
    strategies: list[ThresholdStrategy] | None = None,
    *,
    force: bool = False,
    oracle_game_profile: GameProfile | None = None,
) -> int:
    _bind_game_profile(cfg, oracle_game_profile)
    from farkle.utils.manifest import publication_lock

    if force:
        _force_output_directories(cfg, n)
    if _pending_simulation_publications(cfg, n):
        _safe_cleanup_path(cfg.n_dir(n) / "_execution" / "._publication_simulation.lock")
    with publication_lock(cfg.n_dir(n) / "simulation"):
        return _run_single_n(
            cfg, n, strategies, force=force, oracle_game_profile=oracle_game_profile
        )


@stage_action("simulation", also=())
def _run_single_n(
    cfg: AppConfig,
    n: int,
    strategies: list[ThresholdStrategy] | None = None,
    *,
    force: bool = False,
    oracle_game_profile: GameProfile | None = None,
) -> int:
    """Run a Farkle tournament for a single tournament with player count *n*."""
    _recover_simulation_publications(cfg, n)
    # --- Grid & tests ---
    strategies, grid_size, _used_custom = _resolve_strategies(cfg, strategies)
    LOGGER.info(f"{grid_size} total strategies, used custom state: {_used_custom}")
    workload_plan = _plan_workload_from_config(
        cfg,
        n_strategies=grid_size,
        n_players=n,
    )
    n_shuffles = workload_plan.required_shuffles
    total_games = workload_plan.required_games

    # --- Output paths and pre-scheduling plan publication ---
    results_dir = cfg.results_root
    manifest, ordered_roster = simulation_roster_manifest(cfg, strategies)
    manifest_path = cfg.strategy_manifest_root_path()
    n_dir = results_dir / f"{n}_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    if simulation_is_complete(cfg, n, report_recovery=True):
        LOGGER.info("Simulation already complete; preserving published outputs for %sp", n)
        recorder, scope = current_supervisor_recorder(), current_supervisor_scope()
        if recorder is not None and scope is not None:
            recorder.record_completion_summary(
                f"{scope.scope}:simulation:k={n}",
                stage="simulation",
                summary={
                    "reused_attempted_games": total_games,
                    "newly_simulated_games": 0,
                    "reused_shuffles": n_shuffles,
                    "root_seed": cfg.sim.seed,
                    "player_count": n,
                    "recovery_phases": scope.recovery_summary(),
                },
            )
        return total_games
    workload_plan_path = n_dir / "simulation_workload_plan.json"
    row_dir = _resolve_row_output_dir(cfg, n)
    metric_chunk_dir = _resolve_metric_chunk_dir(cfg, n)
    ckpt_path = n_dir / f"{n}p_checkpoint.pkl"
    cleanup: dict[Path, tuple[os.stat_result, str]] = {}
    if force and not workload_plan.cap_exceeded:
        row_dir, metric_chunk_dir = _force_output_directories(cfg, n)
        sealed, cleanup = _validate_force_inventory(cfg, workload_plan, row_dir, metric_chunk_dir)
        force = not sealed
    had_unsealed_v3_outputs = is_v3_config(cfg) and _has_unsealed_simulation_outputs(
        cfg=cfg,
        n_players=n,
        row_dir=row_dir,
        metric_chunk_dir=metric_chunk_dir,
        strategy_manifest_path=cfg.strategy_manifest_root_path(),
        recoverable_checkpoint=(
            ckpt_path.resolve()
            if (
                (row_dir is not None and (row_dir / "manifest.jsonl").is_file())
                or (
                    metric_chunk_dir is not None
                    and (metric_chunk_dir / "metrics_manifest.jsonl").is_file()
                )
            )
            else None
        ),
    )
    LOGGER.info(
        "Tournament workload plan resolved",
        extra={"stage": "simulation", **workload_plan.to_dict()},
    )
    if workload_plan.cap_exceeded:
        if force:
            _safe_cleanup_path(workload_plan_path)
            _safe_cleanup_path(sidecar_path(workload_plan_path))
        if workload_plan_path.exists() or sidecar_path(workload_plan_path).exists():
            raise WorkloadCapExceeded(workload_plan)
        write_workload_plan(workload_plan_path, workload_plan, temp_dir=cfg.io.temp_dir)
        LOGGER.error(
            "Tournament workload blocked by configured cap",
            extra={"stage": "simulation", **workload_plan.to_dict()},
        )
        raise WorkloadCapExceeded(workload_plan)

    resume = not force
    recovery_checkpoint = None
    checkpoint_exists = ckpt_path.exists()
    LOGGER.info(
        "Preparing tournament outputs",
        extra={
            "stage": "simulation",
            "n_players": n,
            "checkpoint_path": str(ckpt_path),
            "force": force,
            "resume": resume,
            "checkpoint_exists": checkpoint_exists,
        },
    )
    if had_unsealed_v3_outputs and not force:
        raise ValueError(
            "artifact contract v3 found pre-existing simulation bytes without "
            "authenticated sidecars; preserve the evidence and use a new configured output root"
        )
    if force:
        _purge_simulation_outputs(cleanup)
    elif resume and _has_existing_outputs(
        n_dir=n_dir,
        n_players=n,
        ckpt_path=ckpt_path,
        row_dir=row_dir,
        metric_chunk_dir=metric_chunk_dir,
    ):
        if (scope := current_supervisor_scope()) is not None:
            scope.update(
                phase="simulation_recovery_discovery",
                state="working",
                progress={"root_seed": cfg.sim.seed, "current_player_count": n, "recovery": None},
            )
        recovery_checkpoint = _validate_resume_outputs(
            cfg=cfg,
            n_players=n,
            n_shuffles=n_shuffles,
            strategies_manifest=manifest,
            ckpt_path=ckpt_path,
            row_dir=row_dir,
            metric_chunk_dir=metric_chunk_dir,
            require_authenticated_checkpoint=True,
        )
    if not manifest.empty:
        if not manifest_path.exists():
            manifest_table = pa.Table.from_pandas(manifest, preserve_index=False)
            if is_v3_config(cfg):
                write_parquet_artifact_atomic(
                    manifest_table,
                    manifest_path,
                    sidecar=_simulation_output_sidecar(
                        cfg,
                        manifest_path,
                        n_players=n,
                        operation="publish_strategy_manifest",
                        ordered_roster=ordered_roster,
                        support_counts=sorted({int(value) for value in cfg.sim.n_players_list}),
                    ),
                )
            else:
                write_parquet_atomic(manifest_table, manifest_path, temp_dir=cfg.io.temp_dir)
        else:
            if is_v3_config(cfg):
                validate_authenticated_artifact_metadata(manifest_path, cfg=cfg)
            _validate_manifest_matches(manifest, manifest_path, label="Strategy")
        if row_dir is not None and row_dir != n_dir:
            row_dir.mkdir(parents=True, exist_ok=True)
        if LOGGER.isEnabledFor(logging.DEBUG):
            sample = manifest[["strategy_id", "strategy_str"]].head(5).to_dict("records")
            LOGGER.debug(
                "Strategy manifest written",
                extra={
                    "stage": "simulation",
                    "path": str(manifest_path),
                    "sample": sample,
                },
            )
    if is_v3_config(cfg):
        _write_workload_plan_simulation_output(
            cfg,
            workload_plan_path,
            workload_plan,
            n_players=n,
            strategy_manifest=manifest_path,
        )
    else:
        write_workload_plan(workload_plan_path, workload_plan, temp_dir=cfg.io.temp_dir)

    simulation_sources = [manifest_path, workload_plan_path]
    checkpoint_sidecar = (
        _simulation_output_sidecar(
            cfg,
            ckpt_path,
            n_players=n,
            operation="publish_simulation_checkpoint",
            sources=simulation_sources,
        )
        if is_v3_config(cfg)
        else None
    )
    row_sidecar = (
        _simulation_output_sidecar(
            cfg,
            (row_dir or n_dir) / "rows_template.parquet",
            n_players=n,
            operation="publish_simulation_row_shard",
            sources=simulation_sources,
        )
        if is_v3_config(cfg) and row_dir is not None
        else None
    )
    metric_chunk_sidecar = (
        _simulation_output_sidecar(
            cfg,
            (metric_chunk_dir or n_dir) / "metrics_template.parquet",
            n_players=n,
            operation="publish_simulation_metric_chunk",
            sources=simulation_sources,
        )
        if is_v3_config(cfg) and metric_chunk_dir is not None
        else None
    )

    # --- Tournament run ---
    tourn_cfg = TournamentConfig(
        n_players=n,
        num_shuffles=n_shuffles,
        desired_sec_per_chunk=cfg.sim.desired_sec_per_chunk,
        ckpt_every_sec=cfg.sim.ckpt_every_sec,
        progress_logging=cfg.sim.progress_logging,
        n_strategies=grid_size,
        mp_start_method=cfg.sim.mp_start_method,
        deterministic_batch_size=workload_plan.shuffles_per_batch,
    )
    resource_policy = resolve_stage_parallel_policy("simulation", cfg.sim, resources=cfg.resources)
    requested_workers = cfg.sim.n_jobs
    resolved_workers = normalize_n_jobs(
        requested_workers,
        cpu_count=resource_policy.total_cores,
        default=1,
    )
    nested_process_pool_suppressed = (
        is_process_pool_worker() and resource_policy.process_workers == 1
    )
    LOGGER.info(
        "Simulation worker handoff: seed=%s k=%s requested=%s resolved=%s effective=%s "
        "nested_suppressed=%s parent_pid=%s",
        cfg.sim.seed,
        n,
        requested_workers,
        resolved_workers,
        resource_policy.process_workers,
        nested_process_pool_suppressed,
        os.getpid(),
        extra={
            "stage": "simulation",
            "root_seed": cfg.sim.seed,
            "n_players": n,
            "requested_workers": requested_workers,
            "resolved_workers": resolved_workers,
            "effective_workers": resource_policy.process_workers,
            "nested_process_pool_suppressed": nested_process_pool_suppressed,
            "process_pool_marker": os.environ.get("FARKLE_PROCESS_POOL_ACTIVE"),
            "parent_pid": os.getpid(),
        },
    )
    apply_native_thread_limits(resource_policy)
    memory_guard = make_process_tree_memory_guard(cfg.resources)
    memory_guard.check_before_schedule(force=True)
    with producer_scope(cfg, "simulation"):
        restored_checkpoint = tournament_mod.run_tournament(
            global_seed=cfg.sim.seed,
            n_jobs=resource_policy.process_workers,
            max_in_flight_per_worker=cfg.resources.max_in_flight_per_worker,
            checkpoint_path=ckpt_path,
            collect_metrics=cfg.sim.expanded_metrics,
            row_output_directory=row_dir,
            metric_chunk_directory=metric_chunk_dir,
            num_shuffles=n_shuffles,
            config=tourn_cfg,
            strategies=strategies,
            resume=resume,
            checkpoint_metadata={
                "simulation_stage_config_sha": _simulation_stage_config_sha(cfg, n),
                "strategy_manifest_sha": _strategy_manifest_digest(manifest),
                "rng_scheme_version": urandom.RNG_SCHEME_VERSION,
                "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
                "tournament_method_version": TOURNAMENT_METHOD_VERSION,
                "rng_bit_generator": "PCG64DXSM",
                **(
                    {"game_profile_sha256": oracle_game_profile.sha256}
                    if oracle_game_profile is not None
                    else {}
                ),
                **_workload_checkpoint_metadata(workload_plan),
            },
            workload_plan=workload_plan,
            workload_plan_path=workload_plan_path,
            oracle_game_profile=oracle_game_profile,
            memory_guard=memory_guard,
            row_sidecar=row_sidecar,
            metric_chunk_sidecar=metric_chunk_sidecar,
            checkpoint_sidecar=checkpoint_sidecar,
            checkpoint_loader=lambda: recovery_checkpoint,
            write_final_metrics_artifact=not is_v3_config(cfg),
            write_workload_plan_artifact=not is_v3_config(cfg),
            execution_policy={
                "requested_workers": requested_workers,
                "resolved_workers": resolved_workers,
                "effective_workers": resource_policy.process_workers,
                "nested_process_pool_suppressed": nested_process_pool_suppressed,
                "process_pool_marker": os.environ.get("FARKLE_PROCESS_POOL_ACTIVE"),
                "runner_parent_pid": os.getpid(),
            },
            temp_dir=cfg.io.temp_dir,
        )

    # --- Final checkpoint post-processing ---
    payload = tournament_mod._read_checkpoint(ckpt_path, cfg=cfg if is_v3_config(cfg) else None)
    if restored_checkpoint is not None and payload != restored_checkpoint:
        raise ValueError(f"authenticated checkpoint changed during aggregate recovery: {ckpt_path}")
    raw_counts = payload.get("win_totals", payload)
    outcome_payload = payload.get("outcome_counts")
    win_totals = tournament_mod._coerce_counter(
        raw_counts, outcome_payload if isinstance(outcome_payload, Mapping) else None
    )

    metric_sums: dict[str, dict[int | str, float]] = payload.get("metric_sums", {})
    metric_sq_sums: dict[str, dict[int | str, float]] = payload.get(
        "metric_square_sums",
        payload.get("metric_sq_sums", {}),
    )

    # (A) Summary parquet
    summary_rows: list[dict[str, float | str]] = []
    for strat in sorted(win_totals.attempted_exposures, key=str):
        wins = int(win_totals.get(strat, 0))
        attempted = int(win_totals.attempted_exposures[strat])
        if attempted <= 0:
            continue
        completed = int(win_totals.completed_exposures[strat])
        safety_limit = int(win_totals.safety_limit_exposures[strat])
        row: dict[str, float | str] = {
            "strategy": strat,
            "wins": float(wins),
            "attempted_exposures": attempted,
            "completed_exposures": completed,
            "safety_limit_exposures": safety_limit,
            "losses": attempted - wins,
            "win_rate_per_attempt": wins / attempted,
            "win_rate": wins / attempted,
            "win_rate_given_completion": wins / completed if completed else float("nan"),
            "safety_limit_exposure_rate": safety_limit / attempted,
        }
        if metric_sums:
            for label in METRIC_LABELS:
                s = metric_sums.get(label, {})
                sum_val = s.get(strat, s.get(str(strat), 0.0))
                row[f"mean_{label}"] = (sum_val / wins) if wins > 0 else 0.0
        summary_rows.append(row)

    if summary_rows:
        ckpt_parquet = n_dir / f"{n}p_checkpoint.parquet"
        summary_table = pa.Table.from_pylist(summary_rows)
        if is_v3_config(cfg):
            _write_parquet_simulation_output(
                cfg,
                summary_table,
                ckpt_parquet,
                n_players=n,
                operation="publish_simulation_checkpoint_summary",
                sources=simulation_sources,
            )
        else:
            write_parquet_atomic(summary_table, ckpt_parquet, temp_dir=cfg.io.temp_dir)

    # (B) Expanded metrics parquet
    if cfg.sim.expanded_metrics:
        metrics_rows: list[dict[str, float | str]] = []
        for strat in sorted(win_totals.attempted_exposures, key=str):
            wins = int(win_totals.get(strat, 0))
            attempted = int(win_totals.attempted_exposures[strat])
            if attempted <= 0:
                continue
            completed = int(win_totals.completed_exposures[strat])
            safety_limit = int(win_totals.safety_limit_exposures[strat])
            base: dict[str, float | str] = {
                "strategy": strat,
                "wins": wins,
                "total_games_strat": attempted,
                "attempted_exposures": attempted,
                "completed_exposures": completed,
                "safety_limit_exposures": safety_limit,
                "losses": attempted - wins,
                "win_rate_per_attempt": wins / attempted,
                "win_rate": wins / attempted,
                "win_rate_given_completion": wins / completed if completed else float("nan"),
                "safety_limit_exposure_rate": safety_limit / attempted,
            }
            for label in METRIC_LABELS:
                sums_for_label = metric_sums.get(label, {})
                sq_for_label = metric_sq_sums.get(label, {})
                sum_val = sums_for_label.get(strat, sums_for_label.get(str(strat), 0.0))
                sq_val = sq_for_label.get(strat, sq_for_label.get(str(strat), 0.0))
                base[f"sum_{label}"] = float(sum_val)
                base[f"sq_sum_{label}"] = float(sq_val)
                mean_val = (sum_val / wins) if wins > 0 else 0.0
                base[f"mean_{label}"] = float(mean_val)
                if wins > 0:
                    ex2 = sq_val / wins
                    var = max(ex2 - (mean_val**2), 0.0)
                else:
                    var = 0.0
                base[f"var_{label}"] = float(var)
            ws = metric_sums.get("winning_score", {}).get(
                strat, metric_sums.get("winning_score", {}).get(str(strat), 0.0)
            )
            base["expected_score"] = (ws / attempted) if attempted > 0 else 0.0
            metrics_rows.append(base)

        if metrics_rows:
            metrics_file = n_dir / f"{n}p_metrics.parquet"
            metrics_table = pa.Table.from_pylist(metrics_rows)
            if is_v3_config(cfg):
                _write_parquet_simulation_output(
                    cfg,
                    metrics_table,
                    metrics_file,
                    n_players=n,
                    operation="publish_simulation_metrics_summary",
                    sources=simulation_sources,
                )
            else:
                write_parquet_atomic(metrics_table, metrics_file, temp_dir=cfg.io.temp_dir)

    outputs: list[Path] = [ckpt_path, workload_plan_path]
    ckpt_parquet = n_dir / f"{n}p_checkpoint.parquet"
    if ckpt_parquet.exists():
        outputs.append(ckpt_parquet)
    metrics_file = n_dir / f"{n}p_metrics.parquet"
    if metrics_file.exists():
        outputs.append(metrics_file)
    manifest_path = cfg.strategy_manifest_root_path()
    if manifest_path.exists():
        outputs.append(manifest_path)
    for directory, name in (
        (row_dir, "manifest.jsonl"),
        (metric_chunk_dir, "metrics_manifest.jsonl"),
    ):
        if directory is not None and (directory / name).exists():
            outputs.append(directory / name)
    if is_v3_config(cfg):
        write_simulation_done(
            cfg,
            n,
            num_shuffles=n_shuffles,
            shuffles_per_batch=workload_plan.shuffles_per_batch,
            n_strategies=grid_size,
            outputs=outputs,
            allow_unsealed_v3_outputs=force or not had_unsealed_v3_outputs,
            rewritten_outputs=[
                workload_plan_path,
                ckpt_path,
                *([ckpt_parquet] if ckpt_parquet.exists() else []),
                *([metrics_file] if metrics_file.exists() else []),
            ],
        )
    else:
        write_simulation_done(
            cfg,
            n,
            num_shuffles=n_shuffles,
            shuffles_per_batch=workload_plan.shuffles_per_batch,
            n_strategies=grid_size,
            outputs=outputs,
        )

    return total_games


def run_multi(
    cfg: AppConfig,
    player_counts: Sequence[int] | None = None,
    *,
    force: bool = False,
    oracle_game_profile: GameProfile | None = None,
) -> dict[int, int]:
    _bind_game_profile(cfg, oracle_game_profile)
    return _run_multi(cfg, player_counts, force=force, oracle_game_profile=oracle_game_profile)


@stage_action("simulation", also=())
def _run_multi(
    cfg: AppConfig,
    player_counts: Sequence[int] | None = None,
    *,
    force: bool = False,
    oracle_game_profile: GameProfile | None = None,
) -> dict[int, int]:
    """Run tournaments for multiple player counts."""
    results: dict[int, int] = {}
    player_counts = (
        list(player_counts) if player_counts is not None else list(cfg.sim.n_players_list)
    )
    if not player_counts:
        validate_workload(cfg, player_counts=player_counts)
    strategies, grid_size, _used_custom = _resolve_strategies(cfg, None)
    validate_workload(cfg, player_counts=player_counts, grid_size=grid_size)

    if force:
        row_owners: dict[Path, int] = {}
        metric_owners: dict[Path, int] = {}
        for n in player_counts:
            rows, metrics = _force_output_directories(cfg, n)
            for directory, owners in ((rows, row_owners), (metrics, metric_owners)):
                if directory is not None and owners.setdefault(directory.absolute(), n) != n:
                    raise ValueError(
                        f"Simulation player counts share a shard inventory: {directory}"
                    )
            if _pending_simulation_publications(cfg, n):
                from farkle.utils.manifest import publication_lock

                with publication_lock(cfg.n_dir(n) / "simulation"):
                    _recover_simulation_publications(cfg, n)
            if any(
                os.path.lexists(p) or os.path.lexists(sidecar_path(p))
                for p in _simulation_cell_files(cfg, n, rows, metrics)
            ) and not simulation_is_complete(cfg, n):
                plan = _plan_workload_from_config(cfg, grid_size, n)
                if not plan.cap_exceeded:
                    _validate_force_inventory(cfg, plan, rows, metrics)

    for n in player_counts:
        if oracle_game_profile is None:
            games = run_single_n(cfg, n, strategies=strategies, force=force)
        else:
            games = run_single_n(
                cfg,
                n,
                strategies=strategies,
                force=force,
                oracle_game_profile=oracle_game_profile,
            )
        results[n] = games
    return results


__all__ = [
    "run_tournament",
    "run_single_n",
    "run_multi",
    "validate_workload",
    "simulation_done_path",
    "simulation_is_complete",
    "write_simulation_done",
]
