"""Shared fixtures and dependency guards for analysis tests."""

from __future__ import annotations

import importlib.util

import pytest


if importlib.util.find_spec("pydantic") is None:
    pytest.skip("pydantic is required for analysis tests", allow_module_level=True)


if importlib.util.find_spec("sklearn") is None:
    pytest.skip("scikit-learn is required for analysis tests", allow_module_level=True)

