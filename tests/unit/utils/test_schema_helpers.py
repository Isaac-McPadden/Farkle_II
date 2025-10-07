"""Tests for :mod:`farkle.utils.schema_helpers`."""

from farkle.utils.schema_helpers import rows_for_ram


def test_rows_for_ram_enforces_minimum() -> None:
    assert rows_for_ram(1, 1_000) == 10_000


def test_rows_for_ram_scales_with_parameters() -> None:
    expected = int((50 * 1024**2) / (20 * 4 * 1.5))
    assert rows_for_ram(50, 20) == expected


def test_rows_for_ram_respects_custom_value_size() -> None:
    expected = int((10 * 1024**2) / (10 * 2 * 2.0))
    assert rows_for_ram(10, 10, bytes_per_val=2, safety=2.0) == expected
