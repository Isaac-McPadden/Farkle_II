"""Run-context helpers for seed and interseed orchestration workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from farkle.analysis.stage_registry import StageLayout, resolve_interseed_stage_layout
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
        return cls(
            seed=int(cfg.sim.seed),
            config=cfg,
            results_root=cfg.results_root,
            analysis_root=cfg.analysis_dir,
            meta_analysis_dir=cfg.meta_analysis_dir if cfg.io.meta_analysis_dir is not None else None,
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
        run_cfg = cls(
            io=base.io,
            sim=base.sim,
            analysis=base.analysis,
            ingest=base.ingest,
            combine=base.combine,
            metrics=base.metrics,
            trueskill=base.trueskill,
            head2head=base.head2head,
            hgb=base.hgb,
        )
        run_cfg.config_sha = base.config_sha
        run_cfg._stage_layout = base._stage_layout
        run_cfg._analysis_root_override = analysis_root
        run_cfg._interseed_input_dir_override = interseed_input_dir
        run_cfg._interseed_input_layout_override = interseed_input_layout
        if stage_layout is not None:
            run_cfg.set_stage_layout(stage_layout)
        return run_cfg

    @property
    def analysis_dir(self) -> Path:
        if self._analysis_root_override is not None:
            return self._analysis_root_override
        return super().analysis_dir

    @property
    def interseed_input_dir(self) -> Path | None:
        if self._interseed_input_dir_override is not None:
            return self._interseed_input_dir_override
        return super().interseed_input_dir

    def _interseed_input_folder(self, key: str | None) -> str | None:
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
class InterseedRunContext:
    """Resolved paths and config for the interseed analysis run."""

    seed_pair: tuple[int, int]
    seed: int
    analysis_root: Path
    input_root: Path
    input_layout: StageLayout
    config: RunContextConfig
    active_config_path: Path

    @classmethod
    def from_seed_context(
        cls,
        seed_context: SeedRunContext,
        *,
        seed_pair: tuple[int, int],
        analysis_root: Path,
    ) -> "InterseedRunContext":
        run_cfg = RunContextConfig.from_base(
            seed_context.config,
            analysis_root=analysis_root,
            interseed_input_dir=seed_context.analysis_root,
            interseed_input_layout=seed_context.config.stage_layout,
            stage_layout=resolve_interseed_stage_layout(seed_context.config),
        )
        return cls(
            seed_pair=seed_pair,
            seed=seed_context.seed,
            analysis_root=analysis_root,
            input_root=seed_context.analysis_root,
            input_layout=seed_context.config.stage_layout,
            config=run_cfg,
            active_config_path=seed_context.active_config_path,
        )


__all__ = ["InterseedRunContext", "RunContextConfig", "SeedRunContext"]
