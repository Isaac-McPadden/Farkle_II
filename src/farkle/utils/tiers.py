"""Helpers for reading and writing consolidated tier reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, MutableMapping

from farkle.utils.analysis_shared import TierMap, tiers_to_map, to_int
from farkle.utils.writer import atomic_path


def _normalize_mapping(mapping: Mapping[str, object]) -> TierMap:
    """Convert strategy→tier payloads into canonical ``TierMap``."""

    normalized: TierMap = {}
    for strategy, tier in mapping.items():
        tier_int = to_int(tier)
        if tier_int is None:
            continue
        normalized[str(strategy)] = tier_int
    return normalized


def _normalize_tiers_payload(raw_tiers: object) -> TierMap:
    """Normalize legacy/current tier payload shapes into a ``TierMap``.

    Accepted legacy shape: ``list[list[str]]``.
    Canonical emitted shape: ``{strategy: tier_int}``.
    """

    if isinstance(raw_tiers, list):
        tier_lists: list[list[str]] = []
        for group in raw_tiers:
            if isinstance(group, list):
                tier_lists.append([str(strategy) for strategy in group])
        return tiers_to_map(tier_lists)

    if isinstance(raw_tiers, Mapping):
        return _normalize_mapping(raw_tiers)

    return {}


def _extract_section(payload: Mapping[str, object], label: str) -> dict[str, int]:
    """Return the normalized tier mapping for ``label`` if present."""

    section = payload.get(label)
    if isinstance(section, Mapping):
        tiers = section.get("tiers") if hasattr(section, "get") else None
        normalized = _normalize_tiers_payload(tiers)
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

    The helper handles historical payloads where tiers were ``list[list[str]]``
    or where the root object is already a ``{strategy: tier}`` mapping, and it
    emits the canonical ``TierMap`` for downstream analysis.
    """

    preferred = _extract_section(payload, prefer) if prefer else {}
    if preferred:
        return preferred

    # Look for any named section with tiers
    for value in payload.values():
        if isinstance(value, Mapping):
            tiers = value.get("tiers") if hasattr(value, "get") else None
            normalized = _normalize_tiers_payload(tiers)
            if normalized:
                return normalized

    # Legacy: payload is already a mapping of strategy → tier
    if isinstance(payload, Mapping):
        return _normalize_mapping(payload)

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
