from __future__ import annotations

from scripts.check_rng_contract import find_violations


def test_repository_rngs_are_coordinate_owned() -> None:
    assert find_violations() == []
