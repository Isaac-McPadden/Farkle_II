# src/farkle/simulation/power_helpers.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from farkle.config import PowerDesign
from farkle.utils.stats import games_for_power


def _unpack_power_design(
    method: str,
    design: PowerDesign | Mapping[str, Any],
) -> dict[str, Any]:
    """
    Accepts a PowerDesign dataclass or a plain dict and returns a normalized dict
    of parameters suitable for forwarding to games_for_power(...).
    """
    if is_dataclass(design):
        d = asdict(design)
    elif isinstance(design, Mapping):
        d = dict(design)
    else:
        raise TypeError("design must be a PowerDesign dataclass or a mapping")

    # Normalize fields; ignore use_BY for Bonferroni
    d["use_BY"] = bool(d.get("use_BY")) if method == "bh" else False

    # Basic validation of tail spelling
    tail = str(d.get("tail", "two_sided")).lower().replace("-", "_").replace(" ", "_")
    if tail not in {"one_sided", "two_sided"}:
        raise ValueError("tail must be 'one_sided' or 'two_sided'")
    d["tail"] = tail

    # Map to the general-purpose function's names
    mapped = {
        "power": d["power"],
        "control": d["control"],
        "detectable_lift": d["detectable_lift"],
        "baseline_rate": d["baseline_rate"],
        "tail": d["tail"],
        "full_pairwise": bool(d.get("full_pairwise", True)),
        "use_BY": d["use_BY"],
        "min_games_floor": d.get("min_games_floor"),
        "max_games_cap": d.get("max_games_cap"),
        "bh_target_rank": d.get("bh_target_rank"),
        "bh_target_frac": d.get("bh_target_frac"),
    }
    return mapped

def games_for_power_from_design(
    *,
    n_strategies: int,
    k_players: int,
    method: str = "bh",
    design: "PowerDesign | Mapping[str, Any]",
) -> int:
    params = _unpack_power_design(method, design)
    return games_for_power(
        n_strategies=n_strategies,
        k_players=k_players,
        method=method,
        **params,
    )
