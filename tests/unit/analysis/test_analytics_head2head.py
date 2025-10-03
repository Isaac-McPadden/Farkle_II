"""Tests for the lightweight head-to-head orchestrator."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from farkle.analysis import head2head
from farkle.config import AppConfig


@pytest.fixture
def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.io.results_dir = tmp_path
    data_dir = cfg.analysis_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / cfg.analysis.curated_rows_name).touch()
    return cfg


def test_run_skips_if_up_to_date(
    _cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _cfg
    out = cfg.analysis_dir / "bonferroni_pairwise.parquet"
    curated = cfg.curated_parquet
    out.touch()
    os.utime(curated, (1000, 1000))
    os.utime(out, (2000, 2000))

    def boom(*, root: Path, n_jobs: int) -> None:  # noqa: ARG001 - should not run
        raise AssertionError("head2head helper should not run when outputs are fresh")

    monkeypatch.setattr(head2head._h2h, "run_bonferroni_head2head", boom)

    with caplog.at_level(logging.INFO):
        head2head.run(cfg)

    assert "Head-to-head results up-to-date" in caplog.text


@pytest.mark.parametrize("error", [RuntimeError("boom"), ValueError("nope")])
def test_run_logs_warning_on_failure(
    _cfg: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    cfg = _cfg
    out = cfg.analysis_dir / "bonferroni_pairwise.parquet"
    curated = cfg.curated_parquet
    out.touch()
    curated.touch()
    os.utime(out, (1000, 1000))
    os.utime(curated, (2000, 2000))

    def boom(*, root: Path, n_jobs: int) -> None:  # noqa: ARG001
        raise error

    monkeypatch.setattr(head2head._h2h, "run_bonferroni_head2head", boom)

    with caplog.at_level(logging.INFO):
        head2head.run(cfg)

    assert any(rec.levelname == "WARNING" and rec.message == "Head-to-head skipped" for rec in caplog.records)
