# src/farkle/orchestration/run_contexts.py
"""Run-context helpers for root and root-pair orchestration workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import cast

from farkle.analysis.stage_registry import StageLayout, resolve_root_pair_stage_layout
from farkle.config import AppConfig


@dataclass(frozen=True)
class SeedRunContext:
    """Resolved paths and config for a single seed run."""

    seed: int
    config: AppConfig
    results_root: Path
    analysis_root: Path
    meta_analysis_dir: Path | None
    active_config_path: Path

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
            meta_analysis_dir=(
                cfg.meta_analysis_dir if cfg.io.meta_analysis_dir is not None else None
            ),
            active_config_path=cfg.results_root / "active_config.yaml",
        )


@dataclass
class RunContextConfig(AppConfig):
    """Config wrapper with run-specific path overrides."""

    _analysis_root_override: Path | None = field(default=None, init=False, repr=False)
    _interseed_input_dir_override: Path | None = field(default=None, init=False, repr=False)
    _interseed_input_layout_override: StageLayout | Mapping[str, str] | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def from_base(
        cls,
        base: AppConfig,
        *,
        analysis_root: Path | None = None,
        interseed_input_dir: Path | None = None,
        interseed_input_layout: StageLayout | Mapping[str, str] | None = None,
        stage_layout: StageLayout | None = None,
    ) -> "RunContextConfig":
        """Clone a base config with run-specific path overrides.

        Args:
            base: Source config to copy.
            analysis_root: Optional override for the analysis output directory.
            interseed_input_dir: Optional override for interseed input discovery.
            interseed_input_layout: Optional override for interseed stage folders.
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
        run_cfg._analysis_root_override = analysis_root
        run_cfg._interseed_input_dir_override = interseed_input_dir
        run_cfg._interseed_input_layout_override = interseed_input_layout
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

    @property
    def interseed_input_dir(self) -> Path | None:
        """Return the interseed input directory, honoring any run-specific override.

        Returns:
            Path used as the root for interseed inputs, or ``None`` when unset.
        """
        if self._interseed_input_dir_override is not None:
            return self._interseed_input_dir_override
        return super().interseed_input_dir

    def _interseed_input_folder(self, key: str | None) -> str | None:
        """Resolve the folder name for one interseed stage key.

        Args:
            key: Stage key whose folder mapping should be resolved.

        Returns:
            Folder name for the stage key, or ``None`` when no mapping is available.
        """
        if key is None:
            return None
        layout = self._interseed_input_layout_override
        if layout is None:
            return super()._interseed_input_folder(key)
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
        pair_analysis_root = pair_root / first.config.io.analysis_subdir
        pair_sim = replace(
            first.config.sim,
            seed=root_pair[0],
            seed_list=list(root_pair),
        )
        pair_base = replace(first.config, sim=pair_sim)
        run_cfg = RunContextConfig.from_base(
            pair_base,
            analysis_root=pair_analysis_root,
            interseed_input_dir=first.analysis_root,
            interseed_input_layout=input_layout_override,
            stage_layout=resolve_root_pair_stage_layout(pair_base),
        )
        return cls(
            root_pair=root_pair,
            root_contexts=root_contexts,
            pair_root=pair_root,
            analysis_root=pair_analysis_root,
            config=run_cfg,
        )


__all__ = ["RootPairRunContext", "RunContextConfig", "SeedRunContext"]
