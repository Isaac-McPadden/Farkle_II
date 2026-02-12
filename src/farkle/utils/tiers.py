"""Helpers for reading and writing consolidated tier reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, MutableMapping

from farkle.utils.analysis_shared import TierMap, tiers_to_map
from farkle.utils.writer import atomic_path


def _normalize(mapping: Mapping[str, object]) -> TierMap:
    """Convert arbitrary tier values to ``{strategy: tier}`` integers."""

    # Boundary conversion belongs in one vectorized helper so analysis code can
    # avoid ad-hoc per-row coercion in hot paths.
    return tiers_to_map(mapping)


def _extract_section(payload: Mapping[str, object], label: str) -> dict[str, int]:
    """Return the normalized tier mapping for ``label`` if present."""

    section = payload.get(label)
    if isinstance(section, Mapping):
        tiers = section.get("tiers") if hasattr(section, "get") else None
        if isinstance(tiers, Mapping):
            normalized = _normalize(tiers)
            if normalized:
                return normalized
    return {}


def load_tier_payload(path: Path) -> dict:
    """Read a consolidated tier payload, returning ``{}`` on failure."""

    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        return {}


def tier_mapping_from_payload(payload: Mapping[str, object], prefer: str = "trueskill") -> dict[str, int]:
    """Extract the preferred tier mapping from a consolidated payload.

    The helper handles historical payloads where the root object is already a
    ``{strategy: tier}`` mapping as well as the new consolidated structure with
    named sections like ``{"trueskill": {"tiers": {...}}}``.
    """

    preferred = _extract_section(payload, prefer) if prefer else {}
    if preferred:
        return preferred

    # Look for any named section with tiers
    for value in payload.values():
        if isinstance(value, Mapping):
            tiers = value.get("tiers") if hasattr(value, "get") else None
            if isinstance(tiers, Mapping):
                normalized = _normalize(tiers)
                if normalized:
                    return normalized

    # Legacy: payload is already a mapping of strategy → tier
    if isinstance(payload, Mapping) and all(isinstance(v, (int, float, str)) for v in payload.values()):
        return _normalize(payload)

    return {}


def write_tier_payload(
    path: Path,
    *,
    trueskill: Mapping[str, object] | None = None,
    frequentist: Mapping[str, object] | None = None,
    active: str | None = None,
    meta: Mapping[str, object] | None = None,
) -> dict:
    """Merge tier sections and persist a consolidated payload atomically."""

    payload: MutableMapping[str, object] = load_tier_payload(path)
    if trueskill is not None:
        payload["trueskill"] = dict(trueskill)
    if frequentist is not None:
        payload["frequentist"] = dict(frequentist)
    if active:
        payload["active"] = active
    if meta:
        meta_block = payload.setdefault("meta", {})
        if isinstance(meta_block, MutableMapping):
            meta_block.update(meta)
        else:
            payload["meta"] = dict(meta)

    with atomic_path(str(path)) as tmp_path:
        Path(tmp_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return dict(payload)
