"""Tests for the lightweight head-to-head orchestrator."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import json
import warnings

import pandas as pd
import pytest

from farkle.analysis import h2h_analysis, head2head
from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig, IOConfig


@pytest.fixture
def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(io=IOConfig(results_dir=tmp_path, append_seed=False))
    cfg.analysis.run_frequentist = True
    cfg.set_stage_layout(resolve_stage_layout(cfg))
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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with caplog.at_level(logging.WARNING):
            decisions = h2h_analysis.holm_bonferroni(
                df_pairs=df, alpha=0.05, tie_policy="neutral_edge"
            )

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


def test_holm_bonferroni_skips_empty_tie_concat() -> None:
    df = pd.DataFrame(
        [
            {"a": "A", "b": "B", "wins_a": 11, "wins_b": 5, "games": 16},
            {"a": "C", "b": "D", "wins_a": 3, "wins_b": 8, "games": 11},
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decisions = h2h_analysis.holm_bonferroni(df_pairs=df, alpha=0.05)

    assert set(decisions["dir"]) == {"a>b", "b>a"}
    assert decisions[["a", "b"]].isna().sum().sum() == 0


def test_tie_break_simulation_is_seeded() -> None:
    df = pd.DataFrame(
        [
            {"a": "A", "b": "B", "wins_a": 5, "wins_b": 5, "games": 10},
        ]
    )
    first = h2h_analysis.holm_bonferroni(df, 0.05, tie_policy="simulate_game", tie_break_seed=4)
    second = h2h_analysis.holm_bonferroni(df, 0.05, tie_policy="simulate_game", tie_break_seed=4)
    assert first.equals(second)


def test_aggregate_pairwise_combines_mirrored_pairs() -> None:
    df = pd.DataFrame(
        [
            {"a": "B", "b": "A", "wins_a": 1, "wins_b": 2, "games": 3},
            {"a": "A", "b": "B", "wins_a": 4, "wins_b": 1, "games": 5},
        ]
    )
    aggregated = h2h_analysis._aggregate_pairwise(df)
    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    assert row["wins_a"] == 6
    assert row["wins_b"] == 2
    assert row["games"] == 8
    assert row["P1_win_rate"] == pytest.approx(6 / 8)


def test_resolve_alpha_prefers_bonferroni_design(_cfg: AppConfig) -> None:
    cfg = _cfg
    cfg.head2head.bonferroni_design = {"alpha": 0.0125, "control": 0.25}
    assert h2h_analysis._resolve_alpha(cfg) == pytest.approx(0.25)
    cfg.head2head.bonferroni_design = {}
    cfg.head2head.fdr_q = 0.1
    assert h2h_analysis._resolve_alpha(cfg) == pytest.approx(0.1)


def test_build_significant_graph_errors_on_cycles() -> None:
    df = pd.DataFrame(
        [
            {"a": "A", "b": "B", "dir": "a>b", "is_sig": True, "pval": 0.01, "adj_p": 0.01},
            {"a": "B", "b": "A", "dir": "a>b", "is_sig": True, "pval": 0.02, "adj_p": 0.02},
        ]
    )
    graph = h2h_analysis.build_significant_graph(df)
    with pytest.raises(RuntimeError):
        h2h_analysis.derive_sig_ranking(graph)


def test_write_graph_json_emits_payload(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"a": "A", "b": "B", "dir": "a>b", "is_sig": True, "pval": 0.01, "adj_p": 0.01},
        ]
    )
    graph = h2h_analysis.build_significant_graph(df, tie_policy="simulate_game", tie_break_seed=9)
    out_path = tmp_path / "graph.json"
    h2h_analysis._write_graph_json(graph, out_path)

    payload = json.loads(out_path.read_text())
    assert payload["tie_policy"] == "simulate_game"
    assert payload["tie_break_seed"] == 9
    assert payload["edges"][0]["source"] == "A"


def test_run_post_h2h_writes_outputs(_cfg: AppConfig, tmp_path: Path) -> None:
    cfg = _cfg
    df_pairs = pd.DataFrame(
        [
            {"a": "Aggro", "b": "Control", "wins_a": 10, "wins_b": 5, "games": 15},
            {"a": "Aggro", "b": "Mid", "wins_a": 7, "wins_b": 3, "games": 10},
        ]
    )
    pairwise_path = cfg.head2head_stage_dir / "bonferroni_pairwise.parquet"
    pairwise_path.parent.mkdir(parents=True, exist_ok=True)
    df_pairs.to_parquet(pairwise_path)

    h2h_analysis.run_post_h2h(cfg)

    decisions = cfg.head2head_stage_dir / "bonferroni_decisions.parquet"
    graph_json = cfg.head2head_stage_dir / "h2h_significant_graph.json"
    ranking_csv = cfg.head2head_stage_dir / "h2h_significant_ranking.csv"
    assert decisions.exists()
    assert graph_json.exists()
    assert ranking_csv.exists()

    manifest = cfg.analysis_dir / cfg.manifest_name
    assert manifest.exists()


def test_build_design_kwargs_normalizes_tail(_cfg: AppConfig) -> None:
    cfg = _cfg
    cfg.head2head.bonferroni_design = {"tail": "two-sided", "bh_target_rank": 3}
    design = head2head._build_design_kwargs(cfg)
    assert design["tail"] == "one_sided"
    assert design["bh_target_rank"] == 3


def test_maybe_autotune_tiers_handles_missing_inputs(_cfg: AppConfig, caplog: pytest.LogCaptureFixture) -> None:
    cfg = _cfg
    cfg.analysis.head2head_target_hours = 1.0
    with caplog.at_level("WARNING"):
        head2head._maybe_autotune_tiers(cfg, {})
    assert "Tier auto-tune skipped" in caplog.text


def test_search_candidate_uses_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    means = {"A": 10.0, "B": 8.0, "C": 7.5}
    stdevs = {"A": 1.0, "B": 1.2, "C": 0.9}
    design_kwargs = {"k_players": 2}

    monkeypatch.setattr(head2head, "_predict_runtime", lambda *args, **_: (5.0, 10, 100, 3))

    candidate = head2head._search_candidate(
        means=means,
        stdevs=stdevs,
        target_hours=1.0,
        tolerance_pct=10.0,
        games_per_sec=50.0,
        design_kwargs=design_kwargs,
        tiering_z=2.0,
        tiering_min_gap=None,
    )

    assert candidate is not None
    assert candidate.games_per_pair == 10
    assert candidate.total_pairs == 3


def test_predict_runtime_handles_elite_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(head2head, "games_for_power", lambda **_: 4)
    runtime, games_per_pair, total_games, pairs = head2head._predict_runtime(
        elite_count=3,
        games_per_sec=2.0,
        design_kwargs={"k_players": 2},
    )

    assert games_per_pair == 2
    assert total_games == 6
    assert pairs == 3
    assert runtime > 0

    zero_runtime = head2head._predict_runtime(elite_count=1, games_per_sec=1.0, design_kwargs={})
    assert zero_runtime == (0.0, 0, 0, 0)


def test_calibrate_h2h_games_per_sec_requires_two_strategies(tmp_path: Path) -> None:
    df = pd.DataFrame({"strategy": ["A"], "mu": [1.0], "sigma": [0.1]})
    with pytest.raises(ValueError):
        head2head._calibrate_h2h_games_per_sec(df, seed=1, n_jobs=1)
