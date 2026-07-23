# src/farkle/orchestration/run_contexts.py
"""Run-context helpers for root and root-pair orchestration workflows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, cast

from farkle.analysis.stage_registry import StageLayout, resolve_root_pair_stage_layout
from farkle.config import AppConfig, assign_config_sha, compute_config_sha
from farkle.utils.authenticated_contract import CodeIdentity, canonical_json_bytes, identity_sha256
from farkle.utils.writer import atomic_path

SEED_PAIR_ANALYSIS_DIRNAME = "seed_pair_analysis"
RUN_CONTEXT_FILENAME = "run_context.json"


@dataclass(frozen=True)
class SeedRunContext:
    """Resolved paths and config for a single seed run."""

    seed: int
    config: AppConfig
    results_root: Path
    analysis_root: Path
    active_config_path: Path

    @property
    def run_context_path(self) -> Path:
        """Return the separate runtime/layout lineage artifact."""

        return self.results_root / RUN_CONTEXT_FILENAME

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "SeedRunContext":
        """Build a seed-run context from a resolved application config.

        Args:
            cfg: Loaded application config for a single-seed run.

        Returns:
            Seed context with resolved results, analysis, and config artifact paths.
        """
        return cls(
            seed=int(cfg.sim.seed),
            config=cfg,
            results_root=cfg.results_root,
            analysis_root=cfg.analysis_dir,
            active_config_path=cfg.results_root / "active_config.yaml",
        )


@dataclass
class RunContextConfig(AppConfig):
    """Config wrapper with run-specific path overrides."""

    _analysis_root_override: Path | None = field(default=None, init=False, repr=False)
    _root_input_layout_override: StageLayout | Mapping[str, str] | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def from_base(
        cls,
        base: AppConfig,
        *,
        analysis_root: Path | None = None,
        root_input_layout: StageLayout | Mapping[str, str] | None = None,
        stage_layout: StageLayout | None = None,
    ) -> "RunContextConfig":
        """Clone a base config with run-specific path overrides.

        Args:
            base: Source config to copy.
            analysis_root: Optional override for the analysis output directory.
            root_input_layout: Stage folders owned by each source root.
            stage_layout: Optional replacement stage layout for the cloned config.

        Returns:
            A config instance that preserves the base settings while applying overrides.
        """
        init_values = {
            item.name: getattr(base, item.name) for item in fields(AppConfig) if item.init
        }
        run_cfg = cls(**init_values)
        run_cfg.config_sha = base.config_sha
        run_cfg._stage_layout = base._stage_layout
        run_cfg._code_identity = base._code_identity
        run_cfg._run_lineage_sha256 = base._run_lineage_sha256
        run_cfg._analysis_root_override = analysis_root
        run_cfg._root_input_layout_override = root_input_layout
        if stage_layout is not None:
            run_cfg.set_stage_layout(cast(StageLayout, stage_layout))
        return run_cfg

    @property
    def analysis_dir(self) -> Path:
        """Return the analysis directory, honoring any run-specific override.

        Returns:
            Path where analysis artifacts for the active run should be written.
        """
        if self._analysis_root_override is not None:
            return self._analysis_root_override
        return super().analysis_dir

    def root_input_stage_folder(self, key: str) -> str | None:
        """Resolve the canonical folder name for a root-owned stage.

        Args:
            key: Root-stage key whose folder mapping should be resolved.

        Returns:
            Folder name for the stage key, or ``None`` when no mapping is available.
        """
        layout = self._root_input_layout_override
        if layout is None:
            return super().root_input_stage_folder(key)
        if isinstance(layout, Mapping):
            folder = layout.get(key)
            return str(folder) if folder is not None else None
        if isinstance(layout, StageLayout):
            return layout.folder_for(key)
        return None


@dataclass(frozen=True)
class RootPairRunContext:
    """Resolved paths and config for the one-time root-pair analysis run."""

    root_pair: tuple[int, int]
    root_contexts: tuple[SeedRunContext, SeedRunContext]
    pair_root: Path
    analysis_root: Path
    config: RunContextConfig

    @property
    def active_config_path(self) -> Path:
        return self.pair_root / "active_config.yaml"

    @property
    def run_context_path(self) -> Path:
        return self.pair_root / RUN_CONTEXT_FILENAME

    @classmethod
    def from_root_contexts(
        cls,
        root_contexts: tuple[SeedRunContext, SeedRunContext],
        *,
        pair_root: Path,
    ) -> "RootPairRunContext":
        """Build the pair context from exactly two completed root contexts."""

        root_pair = (root_contexts[0].seed, root_contexts[1].seed)
        if len(set(root_pair)) != 2:
            raise ValueError(f"root-pair context requires two distinct roots, found {root_pair}")
        first = root_contexts[0]
        input_layout = cast(StageLayout, first.config.stage_layout)
        combine_folder = input_layout.folder_for("combine")
        if combine_folder is None:
            raise KeyError("Root-stage layout must include 'combine' for pair inputs")
        input_layout_override: dict[str, str] = {
            placement.definition.key: placement.folder_name for placement in input_layout.placements
        }
        pair_analysis_root = pair_root / SEED_PAIR_ANALYSIS_DIRNAME
        pair_sim = replace(
            first.config.sim,
            seed=root_pair[0],
            seed_list=list(root_pair),
        )
        pair_base = replace(first.config, sim=pair_sim)
        assign_config_sha(pair_base)
        run_cfg = RunContextConfig.from_base(
            pair_base,
            analysis_root=pair_analysis_root,
            root_input_layout=input_layout_override,
            stage_layout=resolve_root_pair_stage_layout(pair_base),
        )
        return cls(
            root_pair=root_pair,
            root_contexts=root_contexts,
            pair_root=pair_root,
            analysis_root=pair_analysis_root,
            config=run_cfg,
        )


def _code_payload(code_identity: CodeIdentity) -> dict[str, object]:
    return {
        "commit": code_identity.commit,
        "policy": code_identity.policy,
        "state": code_identity.state,
        "dirty_fingerprint_sha256": code_identity.dirty_fingerprint_sha256,
    }


def configure_run_lineage(
    context: SeedRunContext | RootPairRunContext,
    *,
    code_identity: CodeIdentity,
    parent_lifecycle_roots: tuple[str, ...] = (),
) -> str:
    """Attach and return the immutable lineage identity used by stage freshness.

    Mutable execution controls and resolved physical paths are recorded in the
    run-context artifact but deliberately excluded from this identity.
    """

    cfg = context.config
    public_config_sha = compute_config_sha(cfg)
    cfg.config_sha = public_config_sha
    roots = (
        list(context.root_pair) if isinstance(context, RootPairRunContext) else [int(context.seed)]
    )
    lineage = {
        "run_context_contract_version": 1,
        "context_kind": "root_pair" if isinstance(context, RootPairRunContext) else "root",
        "roots": roots,
        "parent_lifecycle_roots": list(parent_lifecycle_roots),
        "stage_layout_identity_sha256": identity_sha256(cfg.stage_layout.to_resolved_layout()),
        "code_identity": _code_payload(code_identity),
    }
    lineage_sha = identity_sha256(lineage)
    cfg._code_identity = code_identity
    cfg._run_lineage_sha256 = lineage_sha
    return lineage_sha


def write_run_context_atomic(
    context: SeedRunContext | RootPairRunContext,
    *,
    code_identity: CodeIdentity,
    parent_lifecycle_roots: tuple[str, ...] = (),
    cli_overrides: tuple[str, ...] = (),
) -> str:
    """Publish the authenticated runtime/layout artifact separately from YAML."""

    lineage_sha = configure_run_lineage(
        context,
        code_identity=code_identity,
        parent_lifecycle_roots=parent_lifecycle_roots,
    )
    cfg = context.config
    resolved_paths = {
        "results_root": str(
            context.pair_root if isinstance(context, RootPairRunContext) else context.results_root
        ),
        "analysis_root": str(context.analysis_root),
        "active_config": str(context.active_config_path),
    }
    execution = {
        "sim_n_jobs": cfg.sim.n_jobs,
        "sim_mp_start_method": cfg.sim.mp_start_method,
        "sim_chunk_seconds": cfg.sim.desired_sec_per_chunk,
        "sim_checkpoint_seconds": cfg.sim.ckpt_every_sec,
        "ingest_n_jobs": cfg.ingest.n_jobs,
        "analysis_n_jobs": cfg.analysis.n_jobs,
        "analysis_mp_start_method": cfg.analysis.mp_start_method,
        "head2head_n_jobs": cfg.head2head.n_jobs,
        "parallel_seeds": cfg.orchestration.parallel_seeds,
    }
    payload = {
        "run_context_contract_version": 1,
        "run_lineage_sha256": lineage_sha,
        "public_config_sha256": cfg.config_sha,
        "parent_lifecycle_roots": list(parent_lifecycle_roots),
        "resolved_paths": resolved_paths,
        "resolved_stage_layout": cfg.stage_layout.to_resolved_layout(),
        "execution_controls": execution,
        "code_identity": _code_payload(code_identity),
        "cli_overrides": list(cli_overrides),
    }
    payload["run_context_sha256"] = identity_sha256(payload)
    path = context.run_context_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(str(path)) as temporary:
        Path(temporary).write_bytes(canonical_json_bytes(payload) + b"\n")
    return lineage_sha


def load_run_context(path: Path, *, active_config_path: Path | None = None) -> dict[str, Any]:
    """Load and authenticate a persisted runtime/layout context."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("run context must contain a JSON object")
    recorded = payload.pop("run_context_sha256", None)
    if recorded != identity_sha256(payload):
        raise ValueError("run context hash does not match its canonical payload")
    payload["run_context_sha256"] = recorded
    if active_config_path is not None:
        from farkle.config import load_app_config

        roots = payload.get("parent_lifecycle_roots")
        expected_roots = 2 if isinstance(roots, list) and len(roots) == 2 else None
        config = load_app_config(active_config_path, seed_list_len=expected_roots)
        if compute_config_sha(config) != payload.get("public_config_sha256"):
            raise ValueError("run context does not bind the adjacent public configuration")
    return payload


__all__ = [
    "RUN_CONTEXT_FILENAME",
    "SEED_PAIR_ANALYSIS_DIRNAME",
    "RootPairRunContext",
    "RunContextConfig",
    "SeedRunContext",
    "configure_run_lineage",
    "load_run_context",
    "write_run_context_atomic",
]
