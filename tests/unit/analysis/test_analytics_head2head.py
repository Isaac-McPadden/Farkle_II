"""Tests for the lightweight head-to-head orchestrator."""

from __future__ import annotations

import json
import logging
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import networkx as nx
import pandas as pd
import pytest

from farkle.analysis import h2h_analysis, head2head
from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.analysis.stage_state import read_stage_done, stage_done_path, write_stage_done
from farkle.config import AppConfig, IOConfig


@pytest.fixture
def _cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
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


def test_build_significant_graph_allows_cycles_in_tiers() -> None:
    df = pd.DataFrame(
        [
            {"a": "A", "b": "B", "dir": "a>b", "is_sig": True, "pval": 0.01, "adj_p": 0.01},
            {"a": "B", "b": "A", "dir": "a>b", "is_sig": True, "pval": 0.02, "adj_p": 0.02},
        ]
    )
    graph = h2h_analysis.build_significant_graph(df)
    tiers = h2h_analysis.derive_sig_ranking(graph)
    assert tiers == [["A", "B"]]


def test_topological_order_dag_disconnected_is_deterministic() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "C"), ("B", "C")])
    graph.add_node("D")

    assert h2h_analysis._topological_order(graph) == ["A", "B", "C", "D"]


def test_topological_order_raises_on_cycle() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "A")])

    with pytest.raises(RuntimeError, match="Ranking incomplete"):
        h2h_analysis._topological_order(graph)


def test_insert_sorted_covers_front_middle_and_end() -> None:
    items = [2, 4]

    h2h_analysis._insert_sorted(items, 1, key_fn=lambda value: value)
    h2h_analysis._insert_sorted(items, 3, key_fn=lambda value: value)
    h2h_analysis._insert_sorted(items, 5, key_fn=lambda value: value)

    assert items == [1, 2, 3, 4, 5]


def test_derive_sig_tiers_condenses_cycle() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("B", "A"), ("B", "C")])

    tiers, condensation, scc_tier_nodes = h2h_analysis._derive_sig_tiers(graph)

    assert tiers == [["A", "B"], ["C"]]
    assert condensation.number_of_nodes() == 2
    assert len(scc_tier_nodes) == 2


def test_derive_sig_tiers_dag_with_disconnected_components() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([("A", "B"), ("C", "D")])
    graph.add_node("E")

    tiers, condensation, dag_tier_nodes = h2h_analysis._derive_sig_tiers(graph)

    assert tiers == [["A"], ["B"], ["C"], ["D"], ["E"]]
    assert condensation.number_of_nodes() == 5
    assert len(dag_tier_nodes) == 5


def test_holm_and_tiers_are_stable_with_equal_p_and_ties() -> None:
    df_pairs = pd.DataFrame(
        [
            {"a": "A", "b": "B", "wins_a": 5, "wins_b": 5, "games": 10},
            {"a": "C", "b": "D", "wins_a": 9, "wins_b": 1, "games": 10},
            {"a": "E", "b": "F", "wins_a": 9, "wins_b": 1, "games": 10},
        ]
    )
    shuffled = df_pairs.iloc[[2, 0, 1]].reset_index(drop=True)

    decisions = h2h_analysis.holm_bonferroni(df_pairs, alpha=0.05, tie_policy="neutral_edge")
    decisions_shuffled = h2h_analysis.holm_bonferroni(
        shuffled, alpha=0.05, tie_policy="neutral_edge"
    )

    normalized = decisions.sort_values(["a", "b", "dir"]).reset_index(drop=True)
    normalized_shuffled = decisions_shuffled.sort_values(["a", "b", "dir"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(normalized, normalized_shuffled)

    tiers = h2h_analysis.derive_sig_ranking(
        h2h_analysis.build_significant_graph(decisions, tie_policy="neutral_edge")
    )
    tiers_shuffled = h2h_analysis.derive_sig_ranking(
        h2h_analysis.build_significant_graph(decisions_shuffled, tie_policy="neutral_edge")
    )
    assert tiers == tiers_shuffled


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


def test_run_post_h2h_writes_outputs(_cfg: AppConfig) -> None:
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

    decisions = cfg.post_h2h_stage_dir / "bonferroni_decisions.parquet"
    graph_json = cfg.post_h2h_stage_dir / "h2h_significant_graph.json"
    ranking_csv = cfg.post_h2h_stage_dir / "h2h_significant_ranking.csv"
    tiers_csv = cfg.post_h2h_stage_dir / "h2h_significant_tiers.csv"
    s_tiers_json = cfg.post_h2h_stage_dir / "h2h_s_tiers.json"
    assert decisions.exists()
    assert graph_json.exists()
    assert ranking_csv.exists()
    assert tiers_csv.exists()
    assert s_tiers_json.exists()

    manifest = cfg.analysis_dir / cfg.manifest_name
    assert manifest.exists()


def test_run_post_h2h_skipped_is_deterministic(_cfg: AppConfig) -> None:
    cfg = _cfg
    cfg.sim.seed = 101
    upstream_done = stage_done_path(cfg.head2head_stage_dir, "bonferroni_head2head")
    write_stage_done(
        upstream_done,
        inputs=[],
        outputs=[],
        config_sha=cfg.config_sha,
        status="skipped",
        reason="safeguard exceeded",
    )

    h2h_analysis.run_post_h2h(cfg)

    post_done = read_stage_done(stage_done_path(cfg.post_h2h_stage_dir, "post_h2h"))
    assert post_done["status"] == "skipped"
    assert post_done["reason"] == "safeguard exceeded"

    graph_path = cfg.post_h2h_stage_dir / "h2h_significant_graph.json"
    first_graph = json.loads(graph_path.read_text())
    assert first_graph["tie_break_seed"] == 101

    decisions_path = cfg.post_h2h_stage_dir / "bonferroni_decisions.parquet"
    assert pd.read_parquet(decisions_path).empty

    first_outputs = {
        "graph": graph_path.read_text(),
        "tiers": (cfg.post_h2h_stage_dir / "h2h_significant_tiers.csv").read_text(),
        "ranking": (cfg.post_h2h_stage_dir / "h2h_significant_ranking.csv").read_text(),
        "s_tiers": (cfg.post_h2h_stage_dir / "h2h_s_tiers.json").read_text(),
    }

    h2h_analysis.run_post_h2h(cfg)

    second_outputs = {
        "graph": graph_path.read_text(),
        "tiers": (cfg.post_h2h_stage_dir / "h2h_significant_tiers.csv").read_text(),
        "ranking": (cfg.post_h2h_stage_dir / "h2h_significant_ranking.csv").read_text(),
        "s_tiers": (cfg.post_h2h_stage_dir / "h2h_s_tiers.json").read_text(),
    }

    assert first_outputs == second_outputs


def test_run_exits_early_when_curated_missing(
    _cfg: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg
    cfg.curated_parquet.unlink()
    called = False

    def _boom(*args: Any, **kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(head2head, "_maybe_autotune_tiers", _boom)
    monkeypatch.setattr(head2head._h2h, "run_bonferroni_head2head", _boom)

    head2head.run(cfg)

    assert called is False


def test_run_calls_autotune_before_freshness_check_and_skips_when_legacy_fresh(
    _cfg: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg
    legacy_out = cfg.analysis_dir / "bonferroni_pairwise.parquet"
    legacy_out.parent.mkdir(parents=True, exist_ok=True)
    legacy_out.touch()
    os.utime(cfg.curated_parquet, (1000, 1000))
    os.utime(legacy_out, (2000, 2000))
    call_order: list[str] = []

    def _spy_autotune(cfg: AppConfig, design_kwargs: dict[str, Any]) -> None:  # noqa: ARG001
        call_order.append("autotune")

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("H2H simulation should not run when legacy output is fresh")

    monkeypatch.setattr(head2head, "_maybe_autotune_tiers", _spy_autotune)
    monkeypatch.setattr(head2head._h2h, "run_bonferroni_head2head", _boom)

    head2head.run(cfg)

    assert call_order == ["autotune"]


def test_run_calls_bonferroni_with_expected_args(
    _cfg: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg
    cfg.sim.seed = 987
    cfg.sim.n_jobs = 7
    cfg.head2head.n_jobs = 4
    design = {"tail": "Two-Sided", "foo": "drop"}
    cfg.head2head.bonferroni_design = design
    captured: dict[str, Any] = {}

    out = cfg.head2head_stage_dir / "bonferroni_pairwise.parquet"
    legacy_out = cfg.analysis_dir / "bonferroni_pairwise.parquet"
    if legacy_out.exists():
        legacy_out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.touch()
    cfg.curated_parquet.parent.mkdir(parents=True, exist_ok=True)
    cfg.curated_parquet.touch()
    os.utime(out, (1000, 1000))
    os.utime(cfg.curated_parquet, (2000, 2000))

    monkeypatch.setattr(head2head, "_maybe_autotune_tiers", lambda *_args, **_kwargs: None)

    def _capture_run(*args: Any, **kwargs: Any) -> None:
        captured["args"] = args
        captured.update(kwargs)

    monkeypatch.setattr(head2head._h2h, "run_bonferroni_head2head", _capture_run)

    head2head.run(cfg)

    args = cast(tuple[Any, ...], captured.get("args", ()))
    if "n_jobs" in captured:
        assert args == ()
        assert captured["n_jobs"] == 4
        assert captured["seed"] == 987
        design_payload = captured["design"]
    else:
        assert args
        assert 4 in args
        assert 987 in args
        design_payload = next(item for item in args if isinstance(item, dict))

    assert design_payload == {
        "k_players": 2,
        "method": "bonferroni",
        "full_pairwise": True,
        "endpoint": "pairwise",
        "tail": "one_sided",
    }


@pytest.mark.parametrize(
    ("h2h_jobs", "analysis_jobs", "sim_jobs", "expected"),
    [
        (4, 8, 10, 4),
        (0, 5, 10, 5),
        (None, None, 3, 3),
        (0, None, 0, 1),
    ],
)
def test_resolve_h2h_jobs_precedence(
    _cfg: AppConfig,
    h2h_jobs: int | None,
    analysis_jobs: int | None,
    sim_jobs: int | None,
    expected: int,
) -> None:
    cfg = _cfg
    cast(Any, cfg.head2head).n_jobs = h2h_jobs
    cast(Any, cfg.analysis).n_jobs = analysis_jobs
    cast(Any, cfg.sim).n_jobs = sim_jobs
    assert head2head._resolve_h2h_jobs(cfg) == expected


def test_build_design_kwargs_sanitizes_defaults_and_forces_tail(_cfg: AppConfig) -> None:
    cfg = _cfg
    cfg.head2head.bonferroni_design = {
        "k_players": 5,
        "endpoint": "not_pairwise",
        "tail": "Two-Sided",
        "method": "holm",
        "full_pairwise": False,
        "unknown": "drop-me",
    }
    design = head2head._build_design_kwargs(cfg)
    assert "unknown" not in design
    assert design["k_players"] == 5
    assert design["endpoint"] == "not_pairwise"
    assert design["method"] == "bonferroni"
    assert design["full_pairwise"] is True
    assert design["tail"] == "one_sided"

    cfg.head2head.bonferroni_design = {"tail": "ONE_SIDED"}
    defaults = head2head._build_design_kwargs(cfg)
    assert defaults["k_players"] == 2
    assert defaults["endpoint"] == "pairwise"
    assert defaults["tail"] == "one_sided"


def test_maybe_autotune_tiers_early_return_when_target_hours_non_positive(
    _cfg: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg
    cfg.analysis.head2head_target_hours = 0.0
    called = False

    def _boom(*args: Any, **kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(head2head.pd, "read_parquet", _boom)
    head2head._maybe_autotune_tiers(cfg, {})
    assert called is False


def test_maybe_autotune_tiers_handles_missing_and_empty_ratings(
    _cfg: AppConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _cfg
    cfg.analysis.head2head_target_hours = 1.0
    with caplog.at_level("WARNING"):
        head2head._maybe_autotune_tiers(cfg, {})
    assert "missing pooled ratings" in caplog.text

    ratings_path = cfg.trueskill_path("ratings_k_weighted.parquet")
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["strategy", "mu", "sigma"]).to_parquet(ratings_path)
    with caplog.at_level("WARNING"):
        head2head._maybe_autotune_tiers(cfg, {})
    assert "pooled ratings empty" in caplog.text


def test_maybe_autotune_tiers_uses_configured_throughput_and_candidate_none(
    _cfg: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg
    cfg.analysis.head2head_target_hours = 1.0
    cfg.analysis.head2head_games_per_sec = 321.0
    ratings_path = cfg.trueskill_path("ratings_k_weighted.parquet")
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"strategy": ["A", "B"], "mu": [20.0, 19.5], "sigma": [1.0, 1.1]}
    ).to_parquet(ratings_path)
    called = {"calibrate": False, "gps": None}

    def _boom(*args: Any, **kwargs: Any) -> float:
        called["calibrate"] = True
        raise AssertionError("Calibration should be skipped when throughput is configured")

    def _capture_search(**kwargs: Any) -> None:
        called["gps"] = kwargs["games_per_sec"]
        return None

    monkeypatch.setattr(head2head, "_calibrate_h2h_games_per_sec", _boom)
    monkeypatch.setattr(head2head, "_search_candidate", _capture_search)

    head2head._maybe_autotune_tiers(cfg, {})

    assert called["calibrate"] is False
    assert called["gps"] == 321.0


def test_maybe_autotune_tiers_handles_calibration_failure(
    _cfg: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg
    cfg.analysis.head2head_target_hours = 1.0
    cfg.analysis.head2head_games_per_sec = 0.0
    ratings_path = cfg.trueskill_path("ratings_k_weighted.parquet")
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"strategy": ["A", "B"], "mu": [20.0, 19.5], "sigma": [1.0, 1.1]}
    ).to_parquet(ratings_path)
    searched = False

    monkeypatch.setattr(
        head2head,
        "_calibrate_h2h_games_per_sec",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("nope")),
    )

    def _search(**kwargs: Any) -> None:
        nonlocal searched
        searched = True
        return None

    monkeypatch.setattr(head2head, "_search_candidate", _search)
    head2head._maybe_autotune_tiers(cfg, {})
    assert searched is False


def test_maybe_autotune_tiers_success_writes_payload_and_meta(
    _cfg: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg
    cfg.analysis.head2head_target_hours = 2.0
    cfg.analysis.head2head_games_per_sec = 50.0
    cfg.analysis.tiering_min_gap = 0.2
    ratings_path = cfg.trueskill_path("ratings_k_weighted.parquet")
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"strategy": ["A", "B", "C"], "mu": [25.0, 24.0, 10.0], "sigma": [1.0, 1.0, 1.0]}
    ).to_parquet(ratings_path)

    candidate = head2head.TierCandidate(
        z=1.75,
        runtime_hours=1.9,
        tiers={"A": 1, "B": 1, "C": 2},
        elite_count=2,
        games_per_pair=111,
        total_games=111,
        total_pairs=1,
    )
    payload: dict[str, Any] = {}
    atomic_calls: list[str] = []

    monkeypatch.setattr(head2head, "_search_candidate", lambda **kwargs: candidate)

    def _capture_payload(path: Path, **kwargs: Any) -> None:
        payload["path"] = path
        payload.update(kwargs)

    @contextmanager
    def _atomic(path: str):
        atomic_calls.append(path)
        yield path

    monkeypatch.setattr(head2head, "write_tier_payload", _capture_payload)
    monkeypatch.setattr(head2head, "atomic_path", _atomic)

    head2head._maybe_autotune_tiers(cfg, {"k_players": 2})

    assert payload["active"] == "trueskill"
    assert payload["trueskill"] == {"tiers": candidate.tiers, "z": 1.75, "min_gap": 0.2}
    meta_path = cfg.analysis_dir / "tiers_autotune.json"
    assert str(meta_path) in atomic_calls
    meta = json.loads(meta_path.read_text())
    assert set(["z", "predicted_hours", "games_per_pair", "total_games", "games_per_sec"]).issubset(
        meta
    )


def test_search_candidate_returns_none_when_means_empty() -> None:
    assert (
        head2head._search_candidate(
            means={},
            stdevs={},
            target_hours=1.0,
            tolerance_pct=5.0,
            games_per_sec=10.0,
            design_kwargs={},
            tiering_z=1.645,
            tiering_min_gap=None,
        )
        is None
    )


def test_search_candidate_baseline_low_high_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    means = {"A": 10.0, "B": 8.0}
    stdevs = {"A": 1.0, "B": 1.0}
    monkeypatch.setattr(head2head, "build_tiers", lambda *args, **kwargs: {"A": 1, "B": 1})
    runtimes = [5.0, 9.0, 11.0]
    monkeypatch.setattr(
        head2head,
        "_predict_runtime",
        lambda *args, **kwargs: (runtimes.pop(0), 20, 40, 1),
    )
    baseline = head2head._search_candidate(
        means=means,
        stdevs=stdevs,
        target_hours=5.0,
        tolerance_pct=5.0,
        games_per_sec=1.0,
        design_kwargs={},
        tiering_z=2.0,
        tiering_min_gap=None,
    )
    assert baseline is not None and baseline.z == 2.0

    runtimes = [1.0, 11.0, 5.0]
    low = head2head._search_candidate(
        means=means,
        stdevs=stdevs,
        target_hours=5.0,
        tolerance_pct=5.0,
        games_per_sec=1.0,
        design_kwargs={},
        tiering_z=2.0,
        tiering_min_gap=None,
    )
    assert low is not None and low.z == 0.5

    runtimes = [1.0, 5.0, 2.0]
    high = head2head._search_candidate(
        means=means,
        stdevs=stdevs,
        target_hours=5.0,
        tolerance_pct=5.0,
        games_per_sec=1.0,
        design_kwargs={},
        tiering_z=2.0,
        tiering_min_gap=None,
    )
    assert high is not None and high.z == 6.0


def test_search_candidate_binary_search_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    means = {"A": 10.0, "B": 9.0, "C": 8.0}
    stdevs = {"A": 1.0, "B": 1.0, "C": 1.0}
    monkeypatch.setattr(head2head, "build_tiers", lambda *args, **kwargs: {"A": 1, "B": 1, "C": 1})

    values = [0.1, 0.2, 9.0, 5.0]
    monkeypatch.setattr(head2head, "_predict_runtime", lambda *args, **kwargs: (values.pop(0), 1, 1, 1))
    hit = head2head._search_candidate(
        means=means,
        stdevs=stdevs,
        target_hours=5.0,
        tolerance_pct=1.0,
        games_per_sec=1.0,
        design_kwargs={},
        tiering_z=1.0,
        tiering_min_gap=None,
    )
    assert hit is not None and hit.runtime_hours == 5.0

    monkeypatch.setattr(head2head, "_predict_runtime", lambda *args, **kwargs: (20.0, 1, 1, 1))
    closest = head2head._search_candidate(
        means=means,
        stdevs=stdevs,
        target_hours=5.0,
        tolerance_pct=1.0,
        games_per_sec=1.0,
        design_kwargs={},
        tiering_z=1.0,
        tiering_min_gap=None,
    )
    assert closest is not None
    assert closest.runtime_hours == 20.0


def test_predict_runtime_handles_edges_and_rounding(monkeypatch: pytest.MonkeyPatch) -> None:
    assert head2head._predict_runtime(elite_count=1, games_per_sec=1.0, design_kwargs={}) == (0.0, 0, 0, 0)

    monkeypatch.setattr(head2head, "games_for_power", lambda **_: 5)
    runtime, games_per_pair, total_games, pairs = head2head._predict_runtime(
        elite_count=3,
        games_per_sec=2.0,
        design_kwargs={"k_players": 4},
    )
    assert games_per_pair == 8
    assert pairs == 3
    assert total_games == 24
    assert runtime == pytest.approx(24 / (2.0 * 3600.0))

    monkeypatch.setattr(head2head, "games_for_power", lambda **_: 0)
    assert head2head._predict_runtime(elite_count=4, games_per_sec=3.0, design_kwargs={"k_players": 2}) == (
        0.0,
        0,
        0,
        6,
    )


def test_calibrate_h2h_games_per_sec_requires_two_strategies(tmp_path: Path) -> None:
    df = pd.DataFrame({"strategy": ["A"], "mu": [1.0], "sigma": [0.1]})
    with pytest.raises(ValueError):
        head2head._calibrate_h2h_games_per_sec(df, seed=1, n_jobs=1)


def test_calibrate_h2h_games_per_sec_success(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "strategy": ["S1", "S2", "S3"],
            "mu": [5.0, 4.0, 1.0],
            "sigma": [1.0, 1.0, 1.0],
        }
    )
    calls: dict[str, Any] = {}

    monkeypatch.setattr(
        head2head,
        "parse_strategy_identifier",
        lambda identifier, **kwargs: f"parsed:{identifier}",
    )
    monkeypatch.setattr(head2head, "spawn_seeds", lambda n, seed: [seed + idx for idx in range(n)])

    def _simulate_many_games_from_seeds(**kwargs: Any) -> None:
        calls.update(kwargs)

    monkeypatch.setattr(head2head, "simulate_many_games_from_seeds", _simulate_many_games_from_seeds)
    times = iter([10.0, 14.0])
    monkeypatch.setattr(head2head.time, "perf_counter", lambda: next(times))

    gps = head2head._calibrate_h2h_games_per_sec(df, seed=11, n_jobs=3, sample_games=2000)

    assert calls["n_jobs"] == 3
    assert calls["strategies"] == ["parsed:S1", "parsed:S2"]
    seeds = cast(list[int], calls["seeds"])
    assert seeds[0] == 11
    assert len(seeds) == 2000
    assert gps == pytest.approx(500.0)
@pytest.mark.parametrize(
    ("df_pairs", "tie_policy", "error_type", "match"),
    [
        (
            pd.DataFrame([{"a": "A", "b": "B", "wins_a": 1, "wins_b": 0, "games": 1}]),
            "bad_policy",
            ValueError,
            "Unknown tie_policy",
        ),
        (
            pd.DataFrame([{"a": "A", "b": "B", "wins_a": 1}]),
            "neutral_edge",
            ValueError,
            "pairwise dataframe missing columns: games, wins_b",
        ),
        (
            pd.DataFrame([{"a": "A", "b": "B", "wins_a": 2, "wins_b": 1, "games": 99}]),
            "neutral_edge",
            RuntimeError,
            "Detected wins != games",
        ),
    ],
)
def test_holm_bonferroni_validation_errors_table(
    df_pairs: pd.DataFrame,
    tie_policy: str,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        h2h_analysis.holm_bonferroni(df_pairs=df_pairs, alpha=0.05, tie_policy=tie_policy)


def test_holm_bonferroni_all_rows_tied_with_simulation() -> None:
    df = pd.DataFrame(
        [
            {"a": "A", "b": "B", "wins_a": 3, "wins_b": 3, "games": 6},
            {"a": "C", "b": "D", "wins_a": 5, "wins_b": 5, "games": 10},
        ]
    )

    decisions = h2h_analysis.holm_bonferroni(
        df, alpha=0.05, tie_policy="simulate_game", tie_break_seed=11
    )

    assert len(decisions) == 2
    assert set(decisions["dir"]).issubset({"a>b", "b>a"})
    assert decisions["tie_break"].tolist() == [True, True]
    assert decisions["tie_policy"].tolist() == ["simulate_game", "simulate_game"]


def test_build_significant_graph_validation_and_error_branches() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        h2h_analysis.build_significant_graph(pd.DataFrame([{"a": "A", "b": "B"}]))

    invalid_dir = pd.DataFrame(
        [{"a": "A", "b": "B", "dir": "???", "is_sig": True, "pval": 0.01, "adj_p": 0.01}]
    )
    with pytest.raises(ValueError, match="Unknown direction"):
        h2h_analysis.build_significant_graph(invalid_dir)

    duplicate_edge_rows = pd.DataFrame(
        [
            {"a": "A", "b": "B", "dir": "a>b", "is_sig": True, "pval": 0.01, "adj_p": 0.01},
            {"a": "A", "b": "B", "dir": "a>b", "is_sig": True, "pval": 0.02, "adj_p": 0.02},
        ]
    )
    with pytest.raises(RuntimeError, match="Duplicate edge detected"):
        h2h_analysis.build_significant_graph(duplicate_edge_rows)


def test_build_significant_graph_skips_non_sig_without_tie_break() -> None:
    df = pd.DataFrame(
        [
            {
                "a": "A",
                "b": "B",
                "dir": "a>b",
                "is_sig": False,
                "pval": 0.7,
                "adj_p": 0.9,
                "tie_break": False,
            },
            {
                "a": "C",
                "b": "D",
                "dir": "a>b",
                "is_sig": False,
                "pval": 1.0,
                "adj_p": 1.0,
                "tie_break": True,
                "tie_policy": "simulate_game",
            },
        ]
    )
    graph = h2h_analysis.build_significant_graph(df, tie_policy="simulate_game", tie_break_seed=9)
    assert not graph.has_edge("A", "B")
    assert graph.has_edge("C", "D")


def test_load_union_candidates_branches(_cfg: AppConfig) -> None:
    cfg = _cfg
    candidates, meta, path = h2h_analysis._load_union_candidates(cfg)
    assert candidates == []
    assert meta is None
    assert path is None

    candidate_path = cfg.analysis_dir / "h2h_union_candidates.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text("not valid json", encoding="utf-8")
    candidates, meta, path = h2h_analysis._load_union_candidates(cfg)
    assert candidates == []
    assert meta is None
    assert path is None

    payload = {
        "candidates": ["Alpha", "Beta"],
        "ratings_count": 2,
        "metrics_path": "metrics.csv",
    }
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")
    candidates, meta, path = h2h_analysis._load_union_candidates(cfg)
    assert candidates == ["Alpha", "Beta"]
    assert meta == {"ratings_count": 2, "metrics_path": "metrics.csv"}
    assert path == candidate_path


def test_infer_h2h_input_metadata_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert h2h_analysis._infer_h2h_input_metadata(None)["pairwise_path"] is None

    pairwise_path = tmp_path / "pairs.parquet"
    pd.DataFrame([{"a": "A", "b": "B", "games": 2}]).to_parquet(pairwise_path)

    monkeypatch.setattr(
        h2h_analysis.pq,
        "read_schema",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    meta = h2h_analysis._infer_h2h_input_metadata(pairwise_path)
    assert meta["pooling_mode"] == "unknown"

    monkeypatch.undo()
    pooled_meta = h2h_analysis._infer_h2h_input_metadata(pairwise_path)
    assert pooled_meta["pooling_mode"] == "pooled"
    assert pooled_meta["pooled_implied_k"] == 2

    players_path = tmp_path / "players.parquet"
    pd.DataFrame([{"players": 2}, {"players": 4}]).to_parquet(players_path)
    monkeypatch.setattr(
        h2h_analysis.pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad read")),
    )
    read_fail_meta = h2h_analysis._infer_h2h_input_metadata(players_path)
    assert read_fail_meta["pooling_mode"] == "k_stratified"

    monkeypatch.undo()
    empty_players_path = tmp_path / "empty_players.parquet"
    pd.DataFrame({"players": []}).to_parquet(empty_players_path)
    empty_meta = h2h_analysis._infer_h2h_input_metadata(empty_players_path)
    assert empty_meta["pooling_mode"] == "k_stratified"
    assert empty_meta["k_values"] == []


def test_build_candidate_selection_metadata_branching(_cfg: AppConfig) -> None:
    cfg = _cfg
    metadata = h2h_analysis._build_candidate_selection_metadata(
        cfg=cfg,
        union_meta=None,
        union_path=None,
        fallback_used=False,
    )
    assert metadata["method"] == "union_top_ratings_metrics"
    assert metadata["source_path"] is None

    union_meta = {
        "ratings_count": 5,
        "metrics_count": 6,
        "combined_count": 9,
        "ratings_path": "ratings.csv",
        "metrics_path": "metrics.csv",
        "ignore_me": "x",
    }
    metadata = h2h_analysis._build_candidate_selection_metadata(
        cfg=cfg,
        union_meta=union_meta,
        union_path=Path("/tmp/union.json"),
        fallback_used=True,
    )
    assert metadata["method"] == "ranking_fallback"
    assert metadata["fallback_used"] is True
    assert metadata["ratings_count"] == 5
    assert metadata["metrics_count"] == 6
    assert metadata["combined_count"] == 9
    assert metadata["ratings_path"] == "ratings.csv"
    assert metadata["metrics_path"] == "metrics.csv"
    assert "ignore_me" not in metadata


def test_run_post_h2h_skips_when_pairwise_missing(_cfg: AppConfig) -> None:
    cfg = _cfg
    upstream_done = stage_done_path(cfg.head2head_stage_dir, "bonferroni_head2head")
    write_stage_done(
        upstream_done,
        inputs=[],
        outputs=[],
        config_sha=cfg.config_sha,
        status="completed",
    )

    h2h_analysis.run_post_h2h(cfg)

    done = read_stage_done(stage_done_path(cfg.post_h2h_stage_dir, "post_h2h"))
    assert done["status"] == "skipped"
    assert done["reason"] == "missing bonferroni pairwise parquet"
    assert pd.read_parquet(cfg.post_h2h_stage_dir / "bonferroni_decisions.parquet").empty


def test_run_post_h2h_falls_back_when_union_missing(_cfg: AppConfig) -> None:
    cfg = _cfg
    pairwise_path = cfg.head2head_stage_dir / "bonferroni_pairwise.parquet"
    pairwise_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"a": "A", "b": "B", "wins_a": 9, "wins_b": 1, "games": 10},
            {"a": "B", "b": "C", "wins_a": 9, "wins_b": 1, "games": 10},
        ]
    ).to_parquet(pairwise_path)

    h2h_analysis.run_post_h2h(cfg)

    payload = json.loads((cfg.post_h2h_stage_dir / "h2h_s_tiers.json").read_text(encoding="utf-8"))
    candidate_meta = payload["_meta"]["candidate_selection"]
    assert candidate_meta["fallback_used"] is True
    assert candidate_meta["method"] == "ranking_fallback"


def test_run_post_h2h_omits_ranking_output_for_cyclic_graph(_cfg: AppConfig) -> None:
    cfg = _cfg
    pairwise_path = cfg.head2head_stage_dir / "bonferroni_pairwise.parquet"
    pairwise_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"a": "A", "b": "B", "wins_a": 19, "wins_b": 1, "games": 20},
            {"a": "B", "b": "C", "wins_a": 19, "wins_b": 1, "games": 20},
            {"a": "C", "b": "A", "wins_a": 19, "wins_b": 1, "games": 20},
        ]
    ).to_parquet(pairwise_path)

    h2h_analysis.run_post_h2h(cfg)

    done = read_stage_done(stage_done_path(cfg.post_h2h_stage_dir, "post_h2h"))
    output_names = {Path(path).name for path in cast(list[str], done["outputs"])}
    assert "h2h_significant_ranking.csv" not in output_names
    assert not (cfg.post_h2h_stage_dir / "h2h_significant_ranking.csv").exists()
