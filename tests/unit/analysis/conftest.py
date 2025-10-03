"""Shared fixtures and helpers for analysis tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from farkle.config import AppConfig


@pytest.fixture
def app_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.io.results_dir = tmp_path
    return cfg
