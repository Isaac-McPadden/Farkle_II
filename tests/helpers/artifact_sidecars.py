"""Production-backed artifact builders for tests.

The approved authenticated-v3 entry points are
``make_authenticated_v3_config``, ``publish_v3_parquet``,
``publish_v3_strategy_manifest``, and ``publish_v3_simulation_run``.  They
create canonical temporary layouts, clean code/config identities, real Arrow
schema identities, authenticated upstream bindings, and v3 completion records
through production publication primitives.  Tests should establish an
accepted control with these builders before applying one mutation.

``mutate_artifact_bytes`` and ``mutate_json_identity_leaf`` deliberately alter
one stored property.  Schema or source mutations should instead republish a
changed table with ``publish_v3_parquet`` so the surrounding v3 identity stays
valid and the intended consumer reaches the changed semantic property.

Pair fixtures use the production ``SeedRunContext`` / ``RootPairRunContext``
constructors directly; this module does not duplicate that lifecycle.  The
legacy contract-v2 helpers at the bottom remain only for tests explicitly
covering that compatibility boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa

from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import (
    AppConfig,
    ArtifactScope,
    IOConfig,
    SimConfig,
    assign_config_sha,
)
from farkle.simulation import runner
from farkle.simulation.strategies import ThresholdStrategy, build_strategy_manifest
from farkle.simulation.workload_planner import TournamentWorkloadPlan
from farkle.utils.artifact_contract import (
    ArtifactSidecar,
    make_artifact_sidecar,
)
from farkle.utils.artifacts import write_parquet_artifact_atomic
from farkle.utils.authenticated_contract import (
    CodeIdentity,
    CodeIdentityPolicy,
    derive_canonical_location,
    validate_authenticated_artifact_unbound,
)
from farkle.utils.random import RNG_SCHEME_VERSION
from farkle.utils.schema_helpers import OUTCOME_SCHEMA_VERSION, TOURNAMENT_METHOD_VERSION
from farkle.utils.streaming_loop import run_streaming_shard
from farkle.utils.writer import atomic_path


@dataclass(frozen=True, slots=True)
class PublishedSimulationRun:
    """Canonical paths for one tiny authenticated simulation cell."""

    shard: Path
    row_manifest: Path
    strategy_manifest: Path
    workload_plan: Path
    completion: Path


def clean_test_code_identity(commit: str = "a" * 40) -> CodeIdentity:
    """Return a deterministic release-clean code identity for isolated tests."""

    return CodeIdentity(
        commit=commit,
        policy=CodeIdentityPolicy.RELEASE_CLEAN.value,
        state="clean",
        dirty_fingerprint_sha256=None,
    )


def make_authenticated_v3_config(
    tmp_path: Path,
    *,
    name: str = "authenticated",
    root_seed: int = 11,
    player_counts: tuple[int, ...] = (2,),
) -> AppConfig:
    """Create one deterministic temporary root with the accepted v3 identity."""

    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / name),
        sim=SimConfig(
            seed=root_seed,
            seed_list=[root_seed],
            n_players_list=list(player_counts),
        ),
    )
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    cfg._code_identity = clean_test_code_identity()
    cfg._run_lineage_sha256 = "1" * 64
    assign_config_sha(cfg)
    assert cfg.artifact_contract.artifact_contract_version == 3
    assert cfg.rng.scheme_version == 2
    assert cfg.artifact_contract.schema_version == 2
    assert cfg.artifact_contract.estimand_version == 2
    assert cfg.artifact_contract.conditioning_version == 2
    return cfg


def publish_v3_parquet(
    cfg: AppConfig,
    path: Path,
    table: pa.Table,
    *,
    stage_key: str,
    producer: str,
    operation: str,
    sources: tuple[Path, ...] = (),
    source_scope: ArtifactScope | str | None = None,
    conditioning: str = "unconditional",
    method_version: int | None = 1,
    method_parameters: dict[str, object] | None = None,
) -> Path:
    """Publish a small canonical Parquet artifact with a genuine v3 sidecar."""

    location = derive_canonical_location(cfg, path, stage_key=stage_key)
    counts = (
        [location.player_count]
        if location.player_count is not None
        else sorted({int(value) for value in cfg.sim.n_players_list})
    )
    parameters: dict[str, object] = {}
    if method_version is not None:
        parameters["method_version"] = method_version
    parameters.update(method_parameters or {})
    metadata = make_artifact_sidecar(
        cfg,
        path,
        producer=producer,
        scope=location.scope,
        source_scope=source_scope or location.scope,
        operation=operation,
        baseline="test_control",
        weighted_quantity="test_fixture_quantity",
        support_count_role="test_fixture_support",
        uncertainty_method="none",
        replication_unit="test_row",
        conditioning=conditioning,
        consistency_columns=table.schema.names,
        source_artifacts=list(sources),
        player_counts=cast(list[int], counts),
        required_player_counts=cast(list[int], counts),
        missing_cell_policy="fail",
        seed_scope="single_root",
        method_contract=cast(
            Any,
            {
                "kind": "operation",
                "procedure": operation,
                "parameters": parameters,
            },
        ),
    )
    write_parquet_artifact_atomic(table, path, sidecar=metadata)
    validate_authenticated_artifact_unbound(path, validate_provenance=False)
    return path


def publish_v3_strategy_manifest(
    cfg: AppConfig,
    strategies: tuple[ThresholdStrategy, ...],
) -> Path:
    """Publish the canonical integer-ID strategy manifest for a test root."""

    frame = build_strategy_manifest(strategies)
    if frame.empty:
        raise ValueError("authenticated strategy fixtures require at least one strategy")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    return publish_v3_parquet(
        cfg,
        cfg.strategy_manifest_root_path(),
        table,
        stage_key="simulation",
        producer="simulation",
        operation="publish_strategy_manifest",
        source_scope=ArtifactScope.DIAGNOSTICS,
        conditioning="all_attempted_games",
        # Simulation completion uses one operation-method identity across the
        # manifest, workload, shards, and checkpoints.  The tournament method
        # remains separately and explicitly versioned at 2; no generic
        # ``method_version`` key is introduced into the version registry.
        method_version=None,
        method_parameters={
            "tournament_method_version": TOURNAMENT_METHOD_VERSION,
            "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
        },
    )


def publish_v3_simulation_run(
    cfg: AppConfig,
    table: pa.Table,
    *,
    strategies: tuple[ThresholdStrategy, ...],
    shuffle_seed: int = 456,
) -> PublishedSimulationRun:
    """Publish one canonical row shard, manifest, workload, and completion."""

    if len(cfg.sim.n_players_list) != 1:
        raise ValueError("tiny simulation fixtures require exactly one player count")
    k = int(cfg.sim.n_players_list[0])
    row_dir = cfg.simulation_row_dir(k)
    if row_dir is None:
        raise ValueError("tiny simulation fixtures require sim.row_dir")
    row_dir.mkdir(parents=True, exist_ok=True)
    strategy_manifest = publish_v3_strategy_manifest(cfg, strategies)
    plan = TournamentWorkloadPlan(
        root_seed=int(cfg.sim.seed),
        k=k,
        strategy_count=len(strategies),
        confidence=0.95,
        resolution_delta=0.9,
        required_shuffles_unrounded=1,
        required_shuffles=1,
        batch_count=1,
        shuffles_per_batch=1,
        batch_construction="equal_contiguous",
        games_per_shuffle=1,
        required_games=table.num_rows,
        achieved_resolution=0.9,
        shuffle_cap=None,
        cap_exceeded=False,
        achieved_resolution_at_cap=None,
    )
    workload = cfg.n_dir(k) / "simulation_workload_plan.json"
    runner._write_workload_plan_simulation_output(
        cfg,
        workload,
        plan,
        n_players=k,
        strategy_manifest=strategy_manifest,
    )
    shard = row_dir / "rows_000000.parquet"
    row_manifest = row_dir / "manifest.jsonl"
    run_streaming_shard(
        out_path=str(shard),
        manifest_path=str(row_manifest),
        schema=table.schema,
        batch_iter=(table,),
        sidecar=runner._simulation_output_sidecar(
            cfg,
            shard,
            n_players=k,
            operation="publish_simulation_row_shard",
            sources=[strategy_manifest, workload],
        ),
        manifest_extra={
            "path": shard.name,
            "rows": table.num_rows,
            "root_seed": cfg.sim.seed,
            "n_players": k,
            "shuffle_index": 0,
            "shuffle_seed": shuffle_seed,
            "deterministic_batch_id": 0,
            "rng_scheme_version": RNG_SCHEME_VERSION,
            "outcome_schema_version": OUTCOME_SCHEMA_VERSION,
            "tournament_method_version": TOURNAMENT_METHOD_VERSION,
        },
    )
    completion = runner.write_simulation_done(
        cfg,
        k,
        num_shuffles=1,
        shuffles_per_batch=1,
        n_strategies=len(strategies),
        outputs=[strategy_manifest, workload, row_dir],
    )
    if not runner.simulation_is_complete(cfg, k):
        raise AssertionError("unmutated authenticated simulation fixture was not accepted")
    return PublishedSimulationRun(
        shard=shard,
        row_manifest=row_manifest,
        strategy_manifest=strategy_manifest,
        workload_plan=workload,
        completion=completion,
    )


def mutate_artifact_bytes(path: Path, *, suffix: bytes = b"mutation") -> None:
    """Change only an artifact's bytes, leaving its bound sidecar untouched."""

    path.write_bytes(path.read_bytes() + suffix)


def mutate_json_identity_leaf(
    path: Path,
    field_path: tuple[str | int, ...],
    value: object,
) -> None:
    """Change exactly one JSON leaf in a sidecar, manifest, or completion."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    target: Any = payload
    for part in field_path[:-1]:
        target = target[part]
    target[field_path[-1]] = value
    with atomic_path(str(path)) as temporary:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def sidecar_metadata(path: Path, *, scope: str = "concat_ks") -> ArtifactSidecar:
    """Return minimal valid metadata for a test artifact."""

    return ArtifactSidecar(
        artifact_contract_version=2,
        estimand_version=1,
        schema_version=1,
        artifact_name=path.name,
        producer="test_fixture",
        scope=scope,
        source_scope="by_k",
        operation="concatenate",
        method_contract={"kind": "operation", "procedure": "concatenate"},
        baseline="none",
        weighted_quantity="none",
        k_aggregation_method="none",
        k_weights=None,
        support_count_role="raw_support_provenance",
        uncertainty_method="none",
        replication_unit="none",
        conditioning="unconditional",
        consistency_columns=[],
        source_artifacts=[],
        grouping_keys=[],
        player_counts=[1],
        required_player_counts=[1],
        missing_cell_policy="not_applicable",
        seed_scope="single_root",
        rng_scheme_version=RNG_SCHEME_VERSION,
        config_hash="test-config",
        input_manifest_hashes=[],
        code_revision="test-revision",
    )


def write_parquet_test_artifact(table: pa.Table, path: Path, *, scope: str = "concat_ks") -> None:
    """Write a test Parquet with a compatible adjacent sidecar."""

    write_parquet_artifact_atomic(table, path, sidecar=sidecar_metadata(path, scope=scope))
