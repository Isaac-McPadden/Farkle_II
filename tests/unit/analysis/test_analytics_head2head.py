"""Tests for the lightweight head-to-head orchestrator."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import pytest

from farkle.analysis import h2h_analysis, head2head
from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig, IOConfig


@pytest.fixture
def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(io=IOConfig(results_dir=tmp_path, append_seed=False))
    cfg.analysis.run_frequentist = True
    cfg.set_stage_layout(resolve_stage_layout(cfg, run_rng=True))
    data_dir = cfg.curate_stage_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / cfg.curated_rows_name).touch()
    curated = cfg.curated_parquet
    curated.parent.mkdir(parents=True, exist_ok=True)
    curated.touch()
    return cfg


def test_run_skips_if_up_to_date(
    _cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _cfg
    out = cfg.head2head_stage_dir / "bonferroni_pairwise.parquet"
    curated = cfg.curated_parquet
    out.touch()
    os.utime(curated, (1000, 1000))
    os.utime(out, (2000, 2000))

    def boom(cfg: AppConfig, *, root: Path | None = None, n_jobs: int, **kwargs) -> None:  # noqa: ARG001
        raise AssertionError("head2head helper should not run when outputs are fresh")

    monkeypatch.setattr(head2head._h2h, "run_bonferroni_head2head", boom)

    with caplog.at_level(logging.INFO):
        head2head.run(cfg)

    assert "Head-to-head results up-to-date" in caplog.text


def test_run_logs_warning_on_failure(
    _cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _cfg
    out = cfg.head2head_stage_dir / "bonferroni_pairwise.parquet"
    curated = cfg.curated_parquet
    out.touch()
    curated.touch()
    os.utime(out, (1000, 1000))
    os.utime(curated, (2000, 2000))

    called = False

    def boom(cfg: AppConfig, *, root: Path | None = None, n_jobs: int, **kwargs) -> None:  # noqa: ARG001
        nonlocal called
        called = True
        raise RuntimeError("boom")

    monkeypatch.setattr(head2head._h2h, "run_bonferroni_head2head", boom)

    with caplog.at_level(logging.INFO):
        head2head.run(cfg)

    assert called
    assert any(
        rec.levelname == "WARNING" and rec.message == "Head-to-head skipped"
        for rec in caplog.records
    )


def test_holm_bonferroni_ties_marked_non_sig(caplog: pytest.LogCaptureFixture) -> None:
    df = pd.DataFrame(
        [
            {"a": "A", "b": "B", "wins_a": 10, "wins_b": 5, "games": 15},
            {"a": "C", "b": "D", "wins_a": 7, "wins_b": 7, "games": 14},
        ]
    )
    with caplog.at_level(logging.WARNING):
        decisions = h2h_analysis.holm_bonferroni(df_pairs=df, alpha=0.05, tie_policy="neutral_edge")

    assert "Ties detected in head-to-head results" in caplog.text
    tie_rows = decisions[decisions["dir"] == "tie"]
    assert len(tie_rows) == 1
    tie_row = tie_rows.iloc[0]
    assert tie_row["adj_p"] == pytest.approx(1.0)
    assert bool(tie_row["is_sig"]) is False
    assert tie_row["tie_policy"] == "neutral_edge"
    assert bool(tie_row["tie_break"]) is False


def test_build_significant_graph_ignores_ties() -> None:
    df = pd.DataFrame(
        [
            {"a": "A", "b": "B", "dir": "a>b", "is_sig": True, "pval": 0.01, "adj_p": 0.01},
            {"a": "C", "b": "D", "dir": "tie", "is_sig": True, "pval": 1.0, "adj_p": 1.0},
        ]
    )
    graph = h2h_analysis.build_significant_graph(df)
    assert graph.has_edge("A", "B")
    assert "C" in graph.nodes
    assert "D" in graph.nodes
    assert not graph.has_edge("C", "D")
    assert ("C", "D") in graph.graph.get("neutral_pairs", [])


def test_build_significant_graph_uses_simulated_tie_breaks() -> None:
    df = pd.DataFrame(
        [
            {"a": "A", "b": "B", "dir": "a>b", "is_sig": False, "pval": 1.0, "adj_p": 1.0, "tie_break": True, "tie_policy": "simulate_game"},
        ]
    )
    graph = h2h_analysis.build_significant_graph(df, tie_policy="simulate_game", tie_break_seed=123)
    assert graph.number_of_edges() == 1
    edge = next(iter(graph.edges(data=True)))
    assert edge[2]["tie_break"] is True
    assert graph.graph.get("tie_break_edges")


def test_tie_break_simulation_is_seeded() -> None:
    df = pd.DataFrame(
        [
            {"a": "A", "b": "B", "wins_a": 5, "wins_b": 5, "games": 10},
        ]
    )
    first = h2h_analysis.holm_bonferroni(df, 0.05, tie_policy="simulate_game", tie_break_seed=4)
    second = h2h_analysis.holm_bonferroni(df, 0.05, tie_policy="simulate_game", tie_break_seed=4)
    assert first.equals(second)
