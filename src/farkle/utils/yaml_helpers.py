# src/farkle/utils/yaml_helpers.py
"""
YAML parsing helpers. Currently exposes ``expand_dotted_keys`` to convert flat
mapping structures with dotted keys into nested dictionaries.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def expand_dotted_keys(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Return a nested dict from *mapping* that may contain dotted keys."""

    result: dict[str, Any] = {}
    for raw_key, raw_value in mapping.items():
        value = expand_dotted_keys(raw_value) if isinstance(raw_value, Mapping) else raw_value
        if isinstance(raw_key, str) and "." in raw_key:
            parts = [part for part in raw_key.split(".") if part]
            if not parts:
                continue
            target = result
            for part in parts[:-1]:
                existing = target.get(part)
                if existing is None:
                    existing = {}
                    target[part] = existing
                elif not isinstance(existing, dict):
                    raise TypeError(
                        f"Cannot expand dotted key {raw_key!r}; {part!r} is already set to a non-mapping value",
                    )
                target = existing
            last = parts[-1]
            existing_leaf = target.get(last)
            if isinstance(existing_leaf, dict) and isinstance(value, dict):
                existing_leaf.update(value)
            else:
                target[last] = value
        else:
            if isinstance(value, dict):
                existing = result.get(raw_key)
                if isinstance(existing, dict):
                    existing.update(value)
                else:
                    result[raw_key] = value
            else:
                result[raw_key] = value
    return result


__all__ = ["expand_dotted_keys"]
