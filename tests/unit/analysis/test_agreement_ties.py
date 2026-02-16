import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from farkle.analysis import agreement
from farkle.analysis.agreement import MethodData
from farkle.analysis.h2h_analysis import build_significant_graph, derive_sig_ranking
from farkle.utils.analysis_shared import tiers_to_map
from farkle.utils.tiers import tier_mapping_from_payload


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
    tiers_a = tier_mapping_from_payload({"a": 2, "b": 2, "c": 5})
    tiers_b = tier_mapping_from_payload({"a": 10, "b": 20, "c": 20})

    assert tiers_a == {"a": 2, "b": 2, "c": 5}
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

    summary: dict[str, Any] | None = agreement._summarize_seed_stability(per_seed)

    assert summary is not None
    assert summary["seeds"] == 2
    assert summary["strategies"] == 2
    assert summary["max_stddev"] == summary["top_strategies"][0]["stddev"]


def test_significant_graph_ranking_to_tier_map_integration():
    decisions = pd.DataFrame(
        [
            {"a": "a", "b": "b", "dir": "a>b", "is_sig": True, "pval": 0.01, "adj_p": 0.01},
            {"a": "b", "b": "c", "dir": "a>b", "is_sig": True, "pval": 0.01, "adj_p": 0.01},
        ]
    )

    graph = build_significant_graph(decisions)
    tier_lists = derive_sig_ranking(graph)
    tiers = tiers_to_map(tier_lists)

    assert tiers == {"a": 1, "b": 2, "c": 3}


def test_load_frequentist_and_trueskill(tmp_path):
    cfg = agreement.AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    players = 2

    assert agreement._load_trueskill(cfg, players, pooled_scope=False) is None
    assert agreement._load_frequentist(cfg, players) is None

    trueskill_path = cfg.trueskill_stage_dir / f"{players}p" / f"ratings_{players}.parquet"
    trueskill_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "strategy": ["a", "b", "ignored"],
            "mu": [1.0, 2.0, 9.9],
            "players": [players, players, 3],
            "sigma": [0.2, 0.3, 0.4],
        },
    ).to_parquet(trueskill_path)
    trueskill_path.with_suffix(".manifest.jsonl").write_text("{}\n")

    tiers_path = cfg.tiering_stage_dir / "tiers.json"
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({str(players): {"tiers": {"a": 1, "b": 2}}}))

    freq_path = cfg.tiering_path("frequentist_scores_k_weighted.parquet")
    freq_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "strategy": ["a", "b", "ignored"],
            "win_rate": [0.5, 0.6, 0.1],
            "n_players": [players, players, 3],
            "tier": [1, 2, 9],
            "metadata": ["x", "y", "z"],
        }
    ).to_parquet(freq_path)
    freq_path.with_suffix(".manifest.jsonl").write_text("{}\n")

    trueskill = agreement._load_trueskill(cfg, players, pooled_scope=False)
    frequentist = agreement._load_frequentist(cfg, players)

    assert trueskill is not None and frequentist is not None
    trueskill_norm = pd.DataFrame(
        {
            "strategy": trueskill.scores.index,
            "score": trueskill.scores.to_numpy(),
            "tier": [trueskill.tiers[s] for s in trueskill.scores.index],
        }
    )
    freq_norm = pd.DataFrame(
        {
            "strategy": frequentist.scores.index,
            "score": frequentist.scores.to_numpy(),
            "tier": [frequentist.tiers[s] for s in frequentist.scores.index],
        }
    )
    assert trueskill_norm.columns.tolist() == ["strategy", "score", "tier"]
    assert freq_norm.columns.tolist() == ["strategy", "score", "tier"]
    assert trueskill_norm["strategy"].tolist() == ["a", "b"]
    assert freq_norm["strategy"].tolist() == ["a", "b"]
    assert trueskill_norm["score"].dtype == "float64"
    assert freq_norm["score"].dtype == "float64"
    assert trueskill_norm["tier"].dtype == "int64"
    assert freq_norm["tier"].dtype == "int64"
    assert frequentist.per_seed_scores == []

    trueskill_path.unlink()
    pd.DataFrame({"mu": [1.0], "players": [players]}).to_parquet(trueskill_path)
    with pytest.raises(
        ValueError,
        match="ratings_k_weighted.parquet missing 'strategy' column",
    ):
        agreement._load_trueskill(cfg, players, pooled_scope=False)

    freq_path.unlink()
    pd.DataFrame({"win_rate": [0.1], "players": [players]}).to_parquet(freq_path)
    with pytest.raises(
        ValueError,
        match="frequentist_scores_k_weighted.parquet missing 'strategy' column",
    ):
        agreement._load_frequentist(cfg, players)

    pd.DataFrame(
        {
            "strategy": ["a", "b"],
            "players": [players, players],
            "metric_a": [0.1, 0.2],
            "metric_b": [1.0, 2.0],
        }
    ).to_parquet(freq_path)
    with pytest.raises(
        ValueError,
        match=(
            "frequentist_scores_k_weighted.parquet has multiple numeric columns; "
            "specify score column"
        ),
    ):
        agreement._load_frequentist(cfg, players)


def test_resolve_trueskill_seed_paths_deduplicates_alias_outputs(tmp_path: Path) -> None:
    cfg = agreement.AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"

    stage_dir = cfg.trueskill_stage_dir
    per_player_dir = stage_dir / "2p"
    per_player_dir.mkdir(parents=True, exist_ok=True)

    payload = pd.DataFrame(
        {
            "strategy": ["a", "b"],
            "mu": [10.0, 9.0],
            "sigma": [1.0, 1.1],
        }
    )

    payload.to_parquet(per_player_dir / "ratings_2_seed12.parquet")
    payload.to_parquet(per_player_dir / "ratings_2_seed13.parquet")
    payload.to_parquet(stage_dir / "trueskill_2p_seed12.parquet")
    payload.to_parquet(stage_dir / "trueskill_2p_seed13.parquet")

    seed_paths = agreement._resolve_trueskill_seed_paths(cfg, players=2, pooled_scope=False)

    assert [path.name for path in seed_paths] == [
        "ratings_2_seed12.parquet",
        "ratings_2_seed13.parquet",
    ]


def test_run_writes_per_scope_payload_and_summary_for_two_seed_pooled(tmp_path, monkeypatch):
    cfg = agreement.AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    cfg.sim.n_players_list = [2]
    cfg.analysis.agreement_include_pooled = True
    cfg.sim.seed_list = [11, 22]

    pooled_path = cfg.trueskill_path("ratings_k_weighted.parquet")
    pooled_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "strategy": ["a", "b"],
            "mu": [5.0, 4.0],
            "sigma": [1.0, 1.0],
        }
    ).to_parquet(pooled_path)

    tiers_path = cfg.preferred_tiers_path()
    tiers_path.parent.mkdir(parents=True, exist_ok=True)
    tiers_path.write_text(json.dumps({"pooled": {"tiers": {"a": 1, "b": 2}}}))

    freq_path = cfg.tiering_path("frequentist_scores_k_weighted.parquet")
    freq_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "strategy": ["a", "b"],
            "win_rate": [0.7, 0.3],
        }
    ).to_parquet(freq_path)

    monkeypatch.setattr(agreement, "_load_head2head", lambda _cfg: None)

    agreement.run(cfg)

    per_scope_path = cfg.agreement_output_path_pooled()
    assert per_scope_path.exists()
    summary_path = cfg.agreement_stage_dir / "agreement_summary.parquet"
    assert summary_path.exists()

    summary_df = pd.read_parquet(summary_path)
    assert len(summary_df) == 1
    assert summary_df.iloc[0]["players"] == "pooled"
