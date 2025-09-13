from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
import dataclasses


@dataclass
class IOConfig:
    """File-system locations for the application."""

    results_dir: Path = Path("results")
    analysis_dir: Path = Path("analysis")


@dataclass
class SimConfig:
    """Simulation parameters."""

    n_players: int = 5
    num_shuffles: int = 100
    seed: int = 0


@dataclass
class AnalysisConfig:
    """Analysis-stage parameters."""

    run_trueskill: bool = True
    trueskill_beta: float = 4.166666666666667
    n_jobs: int = 1
    log_level: str = "INFO"


@dataclass
class AppConfig:
    """Top-level configuration container."""

    io: IOConfig = field(default_factory=IOConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` and return a new mapping."""

    result: dict[str, Any] = dict(base)
    for key, val in overlay.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(val, Mapping)
        ):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_app_config(*overlays: Path) -> AppConfig:
    """Load one or more YAML overlays and return an :class:`AppConfig`.

    Later overlays take precedence over earlier ones and are deep-merged."""

    data: dict[str, Any] = {}
    for path in overlays:
        with path.open("r", encoding="utf-8") as fh:
            overlay = yaml.safe_load(fh) or {}
        if not isinstance(overlay, Mapping):
            raise TypeError(f"Config file {path} must contain a mapping")
        data = _deep_merge(data, overlay)
    def build(cls, section: Mapping[str, Any]) -> Any:
        obj = cls()
        for f in dataclasses.fields(cls):
            if f.name in section:
                val = section[f.name]
                current = getattr(obj, f.name)
                if isinstance(current, Path) and not isinstance(val, Path):
                    val = Path(val)
                setattr(obj, f.name, val)
        return obj

    return AppConfig(
        io=build(IOConfig, data.get("io", {})),
        sim=build(SimConfig, data.get("sim", {})),
        analysis=build(AnalysisConfig, data.get("analysis", {})),
    )


def _coerce(value: str, current: Any) -> Any:
    """Coerce ``value`` to the type of ``current``."""

    if isinstance(current, bool):
        val_lower = value.lower()
        if val_lower in {"1", "true", "yes", "on"}:
            return True
        if val_lower in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value from {value!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, Path):
        return Path(value)
    return value


def apply_dot_overrides(cfg: AppConfig, pairs: list[str]) -> AppConfig:
    """Apply ``section.option=value`` overrides to *cfg*.

    Types are inferred from existing values for bool/int/float/Path/str."""

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid override {pair!r}")
        key, raw = pair.split("=", 1)
        if "." not in key:
            raise ValueError(f"Invalid override {pair!r}")
        section_name, option = key.split(".", 1)
        section = getattr(cfg, section_name)
        if not hasattr(section, option):
            raise AttributeError(f"Unknown option {option!r} in section {section_name!r}")
        current = getattr(section, option)
        new_value = _coerce(raw, current)
        setattr(section, option, new_value)
    return cfg


__all__ = [
    "IOConfig",
    "SimConfig",
    "AnalysisConfig",
    "AppConfig",
    "load_app_config",
    "apply_dot_overrides",
]

