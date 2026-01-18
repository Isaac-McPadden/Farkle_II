import json
import logging

import networkx as nx
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from farkle.analysis import agreement
from farkle.analysis.agreement import MethodData


def test_assert_no_ties_warns_on_duplicates(caplog):
    series = pd.Series([1.0, 1.0], index=["a", "b"])

    with caplog.at_level(logging.WARNING):
        agreement._assert_no_ties(series, "test scores")

    assert "Ties detected in test scores" in caplog.text


def test_rank_correlations_and_coverage():
    scores = {
        "a": pd.Series([1.0, 2.0], index=["s1", "s2"]),
        "b": pd.Series([2.0, 1.0], index=["s2", "s3"]),
        "c": pd.Series([3.0, 4.0], index=["s1", "s2"]),
    }

    spearman, kendall, coverage = agreement._rank_correlations(scores)

    assert spearman is not None
    assert kendall is not None
    assert coverage == {
        "a_vs_b": {"common": 1, "only_a": 1, "only_b": 1},
        "a_vs_c": {"common": 2, "only_a": 0, "only_c": 0},
        "b_vs_c": {"common": 1, "only_b": 1, "only_c": 1},
    }
    assert spearman["a_vs_c"] == pytest.approx(1.0)
    assert kendall["a_vs_c"] == pytest.approx(1.0)
    assert spearman["a_vs_b"] is None and kendall["b_vs_c"] is None


def test_filter_method_to_strategies(caplog):
    scores = pd.Series([1.0, 2.0, 3.0], index=["keep", "drop", "other"])
    tiers = {"keep": 1, "drop": 2, "other": 3}
    per_seed = [pd.Series([9.0, 8.0], index=["keep", "drop"])]
    method = MethodData(scores=scores, tiers=tiers, per_seed_scores=per_seed)

    with caplog.at_level(logging.WARNING):
        filtered = agreement._filter_method_to_strategies(method, ["keep", "missing"], "freq")

    assert "missing 1 strategies: missing" in caplog.text
    assert filtered.scores.index.tolist() == ["keep"]
    assert filtered.tiers == {"keep": 1}
    assert filtered.per_seed_scores[0].index.tolist() == ["keep"]


def test_normalize_tiers_and_agreements():
    tiers_a = agreement._normalize_tiers({"a": 2, "b": 2, "c": 5})
    tiers_b = agreement._normalize_tiers({"a": 10, "b": 20, "c": 20})

    assert tiers_a == {"a": 0, "b": 0, "c": 1}
    ari, nmi = agreement._tier_agreements({"a": tiers_a, "b": tiers_b})
    assert ari is not None
    assert nmi is not None
    common = sorted(set(tiers_a) & set(tiers_b))
    labels_a = [tiers_a[s] for s in common]
    labels_b = [tiers_b[s] for s in common]
    expected_ari = adjusted_rand_score(labels_a, labels_b)
    expected_nmi = normalized_mutual_info_score(labels_a, labels_b)
    assert ari == {"a_vs_b": pytest.approx(expected_ari)}
    assert nmi == {"a_vs_b": pytest.approx(expected_nmi)}


def test_summarize_seed_stability_handles_common_strategies():
    per_seed = [
        pd.Series([1.0, 2.0], index=["s1", "s2"]),
        pd.Series([1.5, 2.5], index=["s1", "s2"]),
    ]

    summary = agreement._summarize_seed_stability(per_seed)

    assert summary is not None
    assert summary["seeds"] == 2
    assert summary["strategies"] == 2
    assert summary["max_stddev"] == summary["top_strategies"][0]["stddev"]


def test_tiers_from_graph_orders_components():
    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")

    tiers = agreement._tiers_from_graph(graph)

    assert tiers["a"] == 1
    assert tiers["b"] == 2
    assert tiers["c"] == 3


def test_load_frequentist_and_trueskill(tmp_path, monkeypatch):
    cfg = agreement.AppConfig()
    cfg.io.results_dir = tmp_path
    cfg.io.append_seed = False
    players = 2

    cfg.trueskill_path = lambda filename: tmp_path / filename
    trueskill_path = cfg.trueskill_path("ratings_pooled.parquet")
    trueskill_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"strategy": ["a", "b"], "mu": [1.0, 2.0], "players": [players, players]},
    ).to_parquet(trueskill_path)

    tiers_path = tmp_path / "tiers.json"
    cfg.preferred_tiers_path = lambda: tiers_path
    tiers_path.write_text(json.dumps({str(players): {"tiers": {"a": 1, "b": 2}}}))

    monkeypatch.setattr(
        "farkle.analysis.stage_registry.StageLayout.require_folder",
        lambda self, key: key,
    )
    cfg.tiering_path = lambda filename: tmp_path / filename
    freq_path = cfg.tiering_path("frequentist_scores.parquet")
    freq_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "strategy": ["a", "b"],
            "win_rate": [0.5, 0.6],
            "players": [players, players],
            "tier": [1, 2],
        }
    ).to_parquet(freq_path)

    trueskill = agreement._load_trueskill(cfg, players)
    frequentist = agreement._load_frequentist(cfg, players)

    assert trueskill is not None and frequentist is not None
    assert trueskill.tiers == {"a": 1, "b": 2}
    assert frequentist.tiers == {"a": 1, "b": 2}
    assert frequentist.per_seed_scores == []
