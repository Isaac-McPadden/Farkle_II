from __future__ import annotations

from scripts.check_terminology import find_violations


def test_repository_terminology_is_precise() -> None:
    assert find_violations() == []
