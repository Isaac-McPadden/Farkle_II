"""Read-only, bounded inspection of the two-root recovery plan."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from functools import cache, cached_property, partial
from pathlib import Path
from typing import Any, Literal

from farkle import analysis
from farkle.analysis.h2h_authorization import (
    authorization_token,
    cap_allows_plan,
    validate_authorization,
    validate_authorization_sources,
)
from farkle.analysis.release_audit import _committed_files, _working_blobs
from farkle.analysis.stage_runner import StagePlanItem
from farkle.config import AppConfig
from farkle.orchestration.run_contexts import RootPairRunContext
from farkle.orchestration.two_seed_pipeline import (
    ResumeIdentityError,
    _prepare_pipeline,
    _require_existing_resume_identity,
)
from farkle.simulation import runner
from farkle.simulation.game_profile import GameProfile
from farkle.simulation.run_tournament import TournamentConfig, _shard_coordinates
from farkle.utils.artifact_contract import ArtifactContractError, sidecar_path
from farkle.utils.authenticated_contract import (
    ArtifactMismatchError,
    AuthenticatedContractError,
    AuthenticatedSidecar,
    CodeIdentity,
    ImmutableManifestSidecar,
    StageIdentity,
    _CompletionVerification,
    load_authenticated_completion,
    require_completion_output_metadata,
)
from farkle.utils.evidence_reader import (
    EvidenceReadDeferred,
    preview_evidence_reads,
    read_evidence_bytes,
)
from farkle.utils.frozen_h2h import BUILTIN_ENGINE
from farkle.utils.manifest import validate_manifest_contract
from farkle.utils.partitioned_stage import _stamp_path, read_unit_stamp
from farkle.utils.release_identity import (
    _require_current_output_versions,
    require_authenticated_output_versions,
)
from farkle.utils.scientific_dependencies import (
    CLASSIFIED_PACKAGE_FILES,
    PREFLIGHT_RECIPE,
    Producer,
    dependencies_for,
    stage_producer,
)
from farkle.utils.scientific_identity import producer_config_identity
from farkle.utils.scientific_invocation import ROOT, dependency_key
from farkle.utils.scientific_reuse import (
    ROOT_INPUT_OPERATIONS,
    closure,
    has_scoped_producer_evidence,
)
from farkle.utils.stage_completion import CompletionState

Action = Literal[
    "reusable",
    "verification_required",
    "aggregate_reconstruction_required",
    "scientific_recomputation_required",
    "blocked",
]
_INVALID_EVIDENCE = (
    ArtifactContractError,
    AuthenticatedContractError,
    OSError,
    KeyError,
    TypeError,
    ValueError,
    ResumeIdentityError,
)
_MAX_COMPLETION_OUTPUTS = 24


@dataclass(frozen=True)
class Decision:
    scope: str
    action: Action
    reason: str
    path: str
    provisional: bool = True


@dataclass(frozen=True)
class ResumePreview:
    roots: tuple[int, int]
    results_root: str
    force: bool
    decisions: tuple[Decision, ...]


class _ChangedScience(ValueError):
    """An observed scientific component differs from the recorded producer."""

    sealed = False


class _Inspection:
    def __init__(self, code: CodeIdentity, *, force: bool) -> None:
        self.code = code
        self.force = force
        self.verification = _CompletionVerification()
        self.decisions: list[Decision] = []
        self._git_tree = cache(_committed_files)
        self._changed_source = cache(self._first_changed_source)
        self.changed: dict[str, str] = {}

    def add(
        self, scope: str, path: Path, action: Action, reason: str, *, provisional: bool = True
    ) -> None:
        self.decisions.append(Decision(scope, action, reason, str(path), provisional))

    def check(
        self,
        scope: str,
        path: Path,
        check: Callable[[], None],
        *,
        invalid: Action = "blocked",
        provisional: bool = True,
    ) -> None:
        try:
            check()
        except EvidenceReadDeferred as exc:
            self.add(scope, path, "verification_required", str(exc))
        except _INVALID_EVIDENCE as exc:
            self.add(scope, path, invalid, str(exc), provisional=provisional)

    @cached_property
    def _working_sources(self) -> dict[str, bytes]:
        # Git comparisons are prospective. The pinned invocation compares exact
        # source and installed runtime bytes, including checkout representation.
        present = tuple(
            sorted(name for name in CLASSIFIED_PACKAGE_FILES if (ROOT / name).is_file())
        )
        return _working_blobs(present)

    @cached_property
    def _working_preflight_sources(self) -> dict[str, bytes]:
        paths = {
            name
            for name in self._git_tree(self.code.commit)
            if name.startswith(("tests/", "scripts/")) and name.endswith(".py")
        }
        present = tuple(sorted(name for name in paths if (ROOT / name).is_file()))
        return _working_blobs(present)

    def _first_changed_source(self, owner: Producer, commit: str) -> str | None:
        required = dependencies_for(owner).source_files
        tree = self._git_tree(commit)
        working = self._working_sources
        if PREFLIGHT_RECIPE in required:
            working = working | self._working_preflight_sources
            required |= {
                name
                for name in tree.keys() | working.keys()
                if name.startswith(("tests/", "scripts/")) and name.endswith(".py")
            }
        external = tuple(
            sorted(name for name in required - working.keys() if (ROOT / name).is_file())
        )
        if external:
            working = working | _working_blobs(external)
        return next(
            (
                name
                for name in sorted(required)
                if (entry := tree.get(name)) is None or entry.split()[-1] != working.get(name)
            ),
            None,
        )

    def require_science(self, cfg: AppConfig, stage: StageIdentity) -> None:
        key = stage.stage_key
        owner = stage_producer(cfg, key)
        coordinate = dependency_key(owner, cfg.sim.seed_list or [cfg.sim.seed])
        if stage.stage_config.sha256 != producer_config_identity(cfg, key).sha256:
            self.changed[coordinate] = f"scoped scientific configuration differs for {key}"
            raise _ChangedScience(self.changed[coordinate])
        if stage.code != self.code and not has_scoped_producer_evidence(stage):
            raise ArtifactMismatchError(f"{key} lacks scoped evidence for cross-revision reuse")
        if stage.code.state == "clean":
            try:
                path = self._changed_source(owner, stage.code.commit)
            except (OSError, ValueError, RuntimeError) as exc:
                raise EvidenceReadDeferred(
                    f"producer source comparison deferred for {key}: {exc}"
                ) from exc
            if path:
                self.changed[coordinate] = f"scoped implementation differs for {key} at {path}"
                raise _ChangedScience(self.changed[coordinate])
        for upstream in closure(stage):
            if upstream in self.changed:
                raise _ChangedScience(self.changed[upstream])

    def metadata(self, cfg: AppConfig, path: Path, operation: str) -> AuthenticatedSidecar:
        metadata = self.verification.artifact(path, deep=False)
        metadata.artifact.location.require_path(cfg, path)
        if metadata.artifact.logical_operation != operation:
            raise ArtifactMismatchError(f"unexpected publication operation at {path}")
        return metadata

    def small_json(self, path: Path, metadata: AuthenticatedSidecar) -> dict[str, Any]:
        contents = read_evidence_bytes(path)
        if hashlib.sha256(contents).hexdigest() != metadata.artifact.content_sha256:
            raise ArtifactMismatchError(f"publication bytes differ from their sidecar at {path}")
        value = json.loads(contents)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}")
        return value

    def completion(self, cfg: AppConfig, stamp: Path, stage: str, required: Iterable[Path]) -> None:
        self.verification.watch(stamp)
        completion = load_authenticated_completion(stamp)
        inventory = {item.location.path(cfg).resolve(): item for item in completion.outputs}
        # Directory inventories are producer-owned. Preview never expands them.
        required_files = {path.resolve() for path in required if not path.is_dir()}
        if not required_files <= inventory.keys() and completion.state != "blocked_by_cap":
            raise ArtifactMismatchError(f"completion omits required output for {stage}")
        if len(inventory) > _MAX_COMPLETION_OUTPUTS:
            raise EvidenceReadDeferred(
                f"{stage} completion inventory requires execution verification"
            )
        config = producer_config_identity(cfg, stage)
        for path, output in inventory.items():
            if output.location.stage_key != stage:
                raise ArtifactMismatchError(f"completion declares a different stage at {path}")
            metadata, digest = self.verification.sidecar(path)
            if metadata is not None:
                try:
                    self.require_science(cfg, metadata.stage_identity)
                except _ChangedScience as exc:
                    exc.sealed = isinstance(metadata, AuthenticatedSidecar) and (
                        metadata.method_contract.procedure in ROOT_INPUT_OPERATIONS
                    )
                    raise
            require_completion_output_metadata(output, metadata, digest, stage_config=config)
            if isinstance(metadata, AuthenticatedSidecar):
                self.verification.artifact(path, deep=False)
                require_authenticated_output_versions(cfg, path, metadata)
            elif isinstance(metadata, ImmutableManifestSidecar):
                self.verification.watch(path)
                _require_current_output_versions(
                    metadata.stage_identity.versions, cfg=cfg, stage_key=stage
                )

    def stage(self, cfg: AppConfig, scope: str, item: StagePlanItem) -> None:
        stamp = item.completion_stamp
        assert stamp is not None
        scope = f"{scope}/{item.name}"
        frozen = item.name in {"candidate_freeze", "h2h_power"} and any(
            path.exists() for path in item.required_outputs
        )
        if not stamp.exists():
            if stamp.parent.exists():
                self.add(
                    scope,
                    stamp,
                    "verification_required",
                    "Completion is missing. Execution will inspect durable stage units.",
                )
                for k in cfg.sim.n_players_list:
                    if item.name == "ingest":
                        output = cfg.ingested_rows_raw(k)
                    elif item.name == "combine":
                        output = cfg.combined_rows_by_k(k)
                    else:
                        break
                    unit_scope = f"{scope}/{k}p"
                    self.check(
                        unit_scope,
                        _stamp_path(output),
                        partial(self.partition, cfg, unit_scope, item.name, k, output),
                        invalid="scientific_recomputation_required",
                    )
            else:
                self.add(
                    scope,
                    stamp,
                    "scientific_recomputation_required",
                    "stage has no published evidence",
                    provisional=False,
                )
            return
        try:
            self.completion(cfg, stamp, item.name, item.required_outputs)
        except EvidenceReadDeferred as exc:
            self.add(scope, stamp, "verification_required", str(exc))
        except _ChangedScience as exc:
            self.add(
                scope,
                stamp,
                "blocked" if frozen or exc.sealed else "scientific_recomputation_required",
                str(exc),
            )
        except _INVALID_EVIDENCE as exc:
            action: Action = (
                "verification_required" if frozen else "scientific_recomputation_required"
            )
            self.add(
                scope,
                stamp,
                action,
                f"Completion is unusable ({exc}). Execution will verify any durable inputs and units.",
            )
        else:
            if self.force and not frozen and item.name != "h2h_execute":
                self.add(
                    scope,
                    stamp,
                    "scientific_recomputation_required",
                    "Force requested. Sealed inputs still require compatibility checks.",
                )
            else:
                self.add(
                    scope,
                    stamp,
                    "verification_required",
                    "Completion metadata is compatible so far. Execution will verify its bytes and inputs.",
                )

    def partition(self, cfg: AppConfig, scope: str, stage: str, k: int, output: Path) -> None:
        stamp = _stamp_path(output)
        if not (output.exists() and stamp.exists()):
            self.add(
                scope,
                stamp,
                "scientific_recomputation_required",
                "Durable unit output or completion is missing. Execution will check retained publications.",
            )
            return
        unit = read_unit_stamp(output)
        if (
            unit["stage_name"] != stage
            or unit["root_seed"] != cfg.sim.seed
            or unit["unit_key"] != [k]
        ):
            raise ArtifactMismatchError(f"unit stamp coordinates differ at {stamp}")
        if output.stat().st_size != unit["output_size_bytes"]:
            raise ArtifactMismatchError(f"unit output length differs at {output}")
        if unit["unit_stamp_schema_version"] == 2:
            metadata, digest = self.verification.sidecar(output)
            if (
                not isinstance(metadata, AuthenticatedSidecar)
                or digest != unit["unit_sidecar_sha256"]
                or metadata.artifact.content_sha256 != unit["output_sha256"]
            ):
                raise ArtifactMismatchError(f"unit stamp and sidecar differ at {stamp}")
            self.require_science(cfg, metadata.stage_identity)
        self.add(
            scope,
            stamp,
            "verification_required",
            "Durable unit metadata found. Execution verifies its inputs and output bytes.",
        )

    def namespace(self, cfg: AppConfig, scope: str, root: Path) -> None:
        if not self.force:
            _require_existing_resume_identity(
                root, cfg=cfg, code_identity=self.code, prospective=True
            )
        if root.is_dir():
            self.add(
                scope,
                root,
                "verification_required" if self.force else "reusable",
                (
                    "Force requested. Execution checks retained publications."
                    if self.force
                    else "Resume namespace is admissible. Each publication requires separate checks."
                ),
                provisional=self.force,
            )
        else:
            self.add(
                scope,
                root,
                "scientific_recomputation_required",
                "results directory is absent",
                provisional=False,
            )

    def simulation(self, cfg: AppConfig, scope: str, k: int) -> None:
        stamp = runner.simulation_done_path(cfg, k)
        if runner._pending_simulation_publications(cfg, k):
            self.add(
                scope,
                stamp,
                "verification_required",
                "Execution must authenticate pending simulation publication recovery.",
            )
            return
        workload_path = stamp.parent / "simulation_workload_plan.json"
        checkpoint = cfg.checkpoint_path(k)
        rows = runner._resolve_row_output_dir(cfg, k)
        metrics = runner._resolve_metric_chunk_dir(cfg, k)
        row_manifest = rows / "manifest.jsonl" if rows is not None else None
        metric_manifest = metrics / "metrics_manifest.jsonl" if metrics is not None else None
        if stamp.is_file():
            try:
                self.completion(cfg, stamp, "simulation", ())
            except _ChangedScience as exc:
                self.add(scope, stamp, "blocked", f"immutable simulation evidence differs ({exc})")
                return
            except _INVALID_EVIDENCE:
                # Execution similarly falls through to checked simulation recovery.
                pass
            else:
                self.add(
                    scope,
                    stamp,
                    "verification_required",
                    "Simulation completion and inventory are provisionally compatible. Execution will verify the roster and shard identities.",
                )
                return

        manifests = tuple(
            path for path in (row_manifest, metric_manifest) if path is not None and path.is_file()
        )
        if not workload_path.exists() and not checkpoint.exists() and not manifests:
            if stamp.parent.exists() or any(
                path is not None and path.parent.exists()
                for path in (row_manifest, metric_manifest)
            ):
                self.add(
                    scope,
                    stamp,
                    "verification_required",
                    "No checkpoint inventory found. Execution will inspect retained simulation files.",
                )
            else:
                self.add(
                    scope,
                    stamp,
                    "scientific_recomputation_required",
                    "no simulation evidence for this player count",
                    provisional=False,
                )
            return

        if (
            self.force
            and not stamp.exists()
            and not sidecar_path(workload_path).exists()
            and all(not sidecar_path(path).exists() for path in manifests)
        ):
            self.add(
                scope,
                stamp,
                "verification_required",
                "Force requested. Execution will look for sealed files before deciding whether to recompute.",
            )
            return

        roster = cfg.strategy_manifest_root_path()
        if not roster.exists():
            raise ArtifactMismatchError(f"canonical strategy manifest missing at {roster}")
        self.require_science(
            cfg, self.metadata(cfg, roster, "publish_strategy_manifest").stage_identity
        )
        workload = self.metadata(cfg, workload_path, "publish_simulation_workload_plan")
        self.require_science(cfg, workload.stage_identity)
        plan = self.small_json(workload_path, workload)
        expected = runner._plan_workload_from_config(cfg, plan["strategy_count"], k)
        if not runner.workload_plan_matches(plan, expected):
            raise _ChangedScience(f"immutable simulation workload differs at {workload_path}")
        if expected.cap_exceeded:
            raise ValueError(
                f"configured workload cap blocks {expected.required_shuffles} shuffles"
            )
        tournament = TournamentConfig(
            n_players=k,
            num_shuffles=expected.required_shuffles,
            n_strategies=expected.strategy_count,
            deterministic_batch_size=expected.shuffles_per_batch,
        )
        coverage_by_manifest: dict[Path | None, set[int]] = {}
        for path in manifests:
            if sidecar_path(path).exists():
                sealed, _ = self.verification.sidecar(path)
                if not isinstance(sealed, ImmutableManifestSidecar):
                    raise ArtifactMismatchError(f"invalid simulation manifest seal at {path}")
                sealed.location.require_path(cfg, path)
                self.require_science(cfg, sealed.stage_identity)
            else:
                sealed = None
            contents = read_evidence_bytes(path)
            if sealed and hashlib.sha256(contents).hexdigest() != sealed.manifest_sha256:
                raise ArtifactMismatchError(f"sealed simulation manifest bytes differ at {path}")
            records = [json.loads(line) for line in contents.split(b"\n") if line.strip()]
            coverage = _shard_coordinates(records, tournament, rows=path == row_manifest)
            coverage_by_manifest[path] = coverage
            self.add(
                scope + "/manifest",
                path,
                "verification_required",
                f"Records cover {len(coverage)}/{tournament.num_shuffles} shuffles. Shard authentication is pending.",
            )

        row_coverage = coverage_by_manifest.get(row_manifest, set())
        metric_coverage = coverage_by_manifest.get(metric_manifest, set())
        durable = row_coverage if row_manifest is not None else metric_coverage
        missing_rows = row_manifest is not None and bool(metric_coverage - row_coverage)

        checkpoint_error = "aggregate checkpoint is missing"
        if checkpoint.exists():
            try:
                metadata = self.metadata(cfg, checkpoint, "publish_simulation_checkpoint")
                self.require_science(cfg, metadata.stage_identity)
                data = read_evidence_bytes(checkpoint)
                if hashlib.sha256(data).hexdigest() != metadata.artifact.content_sha256:
                    raise ArtifactMismatchError(
                        f"checkpoint bytes differ from their sidecar at {checkpoint}"
                    )
            except EvidenceReadDeferred as exc:
                self.add(scope + "/checkpoint", checkpoint, "verification_required", str(exc))
                return
            except _INVALID_EVIDENCE as exc:
                checkpoint_error = str(exc)
            else:
                self.add(
                    scope + "/checkpoint",
                    checkpoint,
                    "verification_required",
                    "Checkpoint bytes match their sidecar. Execution will verify sources and the aggregate receipt.",
                )
                if missing_rows:
                    self.add(
                        scope,
                        stamp,
                        "verification_required",
                        "Metric blocks outpace recorded row coverage. Execution will assess checkpoint and manifest repair.",
                    )
                elif manifests and len(durable) != tournament.num_shuffles:
                    self.add(
                        scope,
                        stamp,
                        "verification_required",
                        f"Manifests record {len(durable)}/{tournament.num_shuffles} shuffles. Execution will check the checkpoint for additional durable coordinates.",
                    )
                return
        if missing_rows:
            self.add(
                scope,
                stamp,
                "blocked",
                f"{checkpoint_error}. Metric blocks lack recorded coverage in the required row manifest.",
            )
        elif durable:
            self.add(
                scope + "/checkpoint",
                checkpoint,
                "aggregate_reconstruction_required",
                f"{checkpoint_error}. Replay {len(durable)}/{tournament.num_shuffles} recorded shuffles after authentication.",
            )
            if len(durable) != tournament.num_shuffles:
                self.add(
                    scope,
                    stamp,
                    "scientific_recomputation_required",
                    f"Manifest coverage is incomplete. {tournament.num_shuffles - len(durable)} shuffles have no recorded coverage.",
                )
        elif checkpoint.exists():
            self.add(
                scope + "/checkpoint",
                checkpoint,
                "blocked",
                f"{checkpoint_error}. No recorded shard coverage for replay.",
            )
        else:
            self.add(
                scope,
                stamp,
                "scientific_recomputation_required",
                "no aggregate checkpoint or recorded shuffles",
            )

    def h2h(self, cfg: AppConfig, scope: str, override: str | None) -> None:
        path = cfg.h2h_power_plan_path()
        schedule = cfg.h2h_block_manifest_path()
        authorization = cfg.h2h_cap_authorization_path()
        if not path.exists() and not schedule.exists():
            return
        if not (path.is_file() and schedule.is_file()):
            self.add(
                scope,
                path,
                "verification_required",
                "Frozen plan publication is incomplete. Execution must reconcile the retained design.",
            )
            return
        power_meta = self.metadata(cfg, path, "score_test_power_plan")
        self.require_science(cfg, power_meta.stage_identity)
        schedule_meta = self.metadata(cfg, schedule, "construct_pair_root_order_blocks")
        plan = self.small_json(path, power_meta)
        if (
            plan.get("planning_state") != CompletionState.COMPLETE_VALID.value
            or not isinstance(plan.get("frozen_design"), dict)
            or plan["frozen_design"].get("engine") != BUILTIN_ENGINE
        ):
            raise ValueError("frozen H2H plan lacks a complete built-in execution design")
        token = authorization_token(plan)
        self.add(
            scope + "/plan",
            path,
            "verification_required",
            f"Frozen plan {plan['schedule_hash']} requires up to {plan['maximum_total_attempts']} attempts. Execution will verify the block manifest and frozen inputs.",
        )
        cap = cfg.head2head.total_game_cap
        existing = None
        if authorization.exists():
            try:
                metadata = self.metadata(cfg, authorization, "h2h_cap_authorization")
                existing = self.small_json(authorization, metadata)
            except _INVALID_EVIDENCE:
                existing = None
            if existing is not None:
                self.require_science(cfg, metadata.stage_identity)
                validate_authorization(existing, plan)
                source_digests = {
                    path: power_meta.artifact.content_sha256,
                    schedule: schedule_meta.artifact.content_sha256,
                }
                validate_authorization_sources(
                    existing,
                    path,
                    schedule,
                    digest_of=source_digests.__getitem__,
                )
                if (
                    existing["configured_total_game_cap"] == cap
                    and existing["execution_authorization"] == "ready"
                ):
                    self.add(
                        scope,
                        authorization,
                        "verification_required",
                        "Retained authorization matches the frozen plan metadata. Execution will verify the exact block manifest bytes and frozen inputs.",
                    )
                    return
        if cap_allows_plan(plan, cap):
            self.add(
                scope,
                authorization,
                "verification_required",
                "Configured cap allows the plan. Execution must establish and authenticate authorization.",
            )
        elif override == token:
            self.add(
                scope,
                authorization,
                "verification_required",
                "Supplied authorization token matches the frozen plan. Execution must verify and record it.",
            )
        else:
            reason = (
                "Supplied authorization token differs"
                if override is not None
                else "Additional authorization is required"
            )
            self.add(
                scope,
                authorization,
                "blocked",
                f"{reason} for the above-cap frozen plan. Resume with --h2h-authorize {token} or approve interactively.",
            )


def preview_pipeline(
    cfg: AppConfig,
    *,
    seed_pair: tuple[int, int],
    force: bool = False,
    h2h_override: str | None = None,
    oracle_game_profile: GameProfile | None = None,
) -> ResumePreview:
    """Inspect the canonical resume entry points without pinning an execution."""
    pair, _, code = _prepare_pipeline(cfg, seed_pair, oracle_game_profile=oracle_game_profile)
    preview = _Inspection(code, force=force)
    pair_scope = f"pair {seed_pair[0]}/{seed_pair[1]}"
    with preview_evidence_reads():
        preview.check(
            pair_scope,
            pair.pair_root,
            partial(preview.namespace, pair.config, pair_scope, pair.pair_root),
            provisional=False,
        )
        for root in pair.root_contexts:
            preview.check(
                f"root {root.seed}",
                root.results_root,
                partial(preview.namespace, root.config, f"root {root.seed}", root.results_root),
                provisional=False,
            )
        ledger = pair.pair_root / "two_seed_pipeline_manifest.jsonl"
        if ledger.exists():
            preview.check(
                pair_scope + "/pipeline manifest",
                ledger,
                lambda: validate_manifest_contract(ledger, contents=read_evidence_bytes(ledger)),
                provisional=False,
            )
        if not any(item.action == "blocked" for item in preview.decisions):
            _inspect_stages(preview, pair, pair_scope, h2h_override, oracle_game_profile)
        preview.check(
            pair_scope + "/evidence changed",
            pair.pair_root,
            preview.verification.require_unchanged,
            invalid="verification_required",
        )
    return ResumePreview(seed_pair, str(pair.pair_root), force, tuple(preview.decisions))


def _inspect_stages(
    preview: _Inspection,
    pair: RootPairRunContext,
    pair_scope: str,
    h2h_override: str | None,
    oracle_game_profile: GameProfile | None,
) -> None:
    for root in pair.root_contexts:
        scope = f"root {root.seed}"
        for k in root.config.sim.n_players_list:
            cell = f"{scope}/simulation/{k}p"
            preview.check(
                cell,
                runner.simulation_done_path(root.config, k),
                partial(preview.simulation, root.config, cell, k),
            )
        for item in analysis.build_root_stage_plan(root.config, force=preview.force):
            preview.stage(root.config, scope, item)
    for item in analysis.build_root_pair_stage_plan(
        pair, force=preview.force, oracle_game_profile=oracle_game_profile
    ):
        preview.stage(pair.config, pair_scope, item)
    preview.check(
        pair_scope + "/h2h authorization",
        pair.config.h2h_power_plan_path(),
        lambda: preview.h2h(pair.config, pair_scope + "/h2h authorization", h2h_override),
    )


def render_preview(report: ResumePreview, *, output_format: str = "text") -> str:
    if output_format == "json":
        return json.dumps({"schema_version": 1, **asdict(report)}, indent=2)
    lines = [
        f"Resume preview for roots {report.roots[0]} and {report.roots[1]}",
        report.results_root,
    ]
    if report.force:
        lines.append("Force requested")
    for decision in report.decisions:
        action = decision.action.replace("_", " ")
        qualifier = " (provisional)" if decision.provisional else ""
        lines.append(
            f"{decision.scope} | {action}{qualifier} | {decision.reason} [{decision.path}]"
        )
    lines.append("Execution rechecks evidence and authorization before acting.")
    return "\n".join(lines)
