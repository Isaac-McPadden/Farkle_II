from __future__ import annotations

import pytest

from farkle.utils.yaml_helpers import expand_dotted_keys


def test_expand_dotted_keys_creates_nested_dicts() -> None:
    data = {"a.b": 1, "a.c": 2, "root": 3}
    expanded = expand_dotted_keys(data)
    assert expanded == {"a": {"b": 1, "c": 2}, "root": 3}


def test_expand_dotted_keys_merges_existing_dicts() -> None:
    data = {"a": {"b": 1}, "a.c": {"deep": True}, "a.d.e": 5}
    expanded = expand_dotted_keys(data)
    assert expanded == {"a": {"b": 1, "c": {"deep": True}, "d": {"e": 5}}}


def test_expand_dotted_keys_conflicting_leaf_raises() -> None:
    with pytest.raises(TypeError):
        expand_dotted_keys({"a": 1, "a.b": 2})


def test_expand_dotted_keys_handles_nested_mappings() -> None:
    data = {"outer.inner": {"leaf": 1}, "outer": {"extra": True}, "outer.inner.deep": 2}
    expanded = expand_dotted_keys(data)
    assert expanded["outer"]["extra"] is True
    assert expanded["outer"]["inner"] == {"leaf": 1, "deep": 2}
