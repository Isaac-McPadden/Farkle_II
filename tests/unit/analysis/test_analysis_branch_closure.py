from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis import agreement, checks, hgb_feat, ingest, isolated_metrics, stage_registry
from farkle.config import AppConfig, IOConfig


def test_agreement_seed_selection_prefers_higher_priority_paths() -> None:
    paths = [
        Path("/tmp/root/ratings_2_seed5.parquet"),
        Path("/tmp/root/2p/ratings_2_seed5.parquet"),
        Path("/tmp/root/trueskill_2p_seed5.parquet"),
        Path("/tmp/root/ratings_2_seed9.parquet"),
        Path("/tmp/root/ignore_seedx.parquet"),
    ]

    selected = agreement._select_seed_paths(
        paths,
        key_fn=lambda path: (agreement._trueskill_seed_path_priority(path, 2), str(path)),
    )

    assert selected == [
        Path("/tmp/root/2p/ratings_2_seed5.parquet"),
        Path("/tmp/root/ratings_2_seed9.parquet"),
    ]


def test_agreement_select_score_column_raises_without_numeric_column() -> None:
    frame = pd.DataFrame({"strategy": ["A"], "tier": ["one"]})

    with pytest.raises(ValueError, match="lacks a numeric score column"):
        agreement._select_score_column(frame, ["win_rate"])


def test_checks_pre_metrics_raises_when_no_manifest_present(tmp_path: Path) -> None:
    combined = tmp_path / "pooled" / "all_ingested_rows.parquet"
    combined.parent.mkdir(parents=True)
    table = pa.table({"winner": ["A"], "wins": [1]})
    pq.write_table(table, combined)

    with pytest.raises(RuntimeError, match="no manifest files found"):
        checks.check_pre_metrics(combined)


def test_checks_post_combine_reports_unreadable_curated_file(tmp_path: Path) -> None:
    combined = tmp_path / "combined.parquet"
    curated = tmp_path / "broken.parquet"
    pq.write_table(pa.table({"winner": ["A"]}), combined)
    curated.write_text("not parquet")

    with pytest.raises(RuntimeError, match="unable to read"):
        checks.check_post_combine([curated], combined, max_players=1)


def test_checks_stage_artifact_families_ignores_missing_stage_dir(tmp_path: Path) -> None:
    checks.check_stage_artifact_families(
        tmp_path,
        stage_dirs={"combine": tmp_path / "does-not-exist"},
        k_values=(2,),
    )


def test_h2h_derive_sig_ranking_empty_graph_returns_empty_tiers() -> None:
    pytest.importorskip("networkx")
    from farkle.analysis import h2h_analysis

    assert h2h_analysis.derive_sig_ranking(h2h_analysis.nx.DiGraph()) == []


def test_h2h_topological_order_is_stable_for_multiple_zero_indegree_nodes() -> None:
    pytest.importorskip("networkx")
    from farkle.analysis import h2h_analysis

    graph = h2h_analysis.nx.DiGraph()
    graph.add_edges_from([("A", "C"), ("B", "C")])

    order = h2h_analysis._topological_order(graph)

    assert order == ["A", "B", "C"]


def test_stage_registry_stage_folder_for_unknown_stage_is_none(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    layout = stage_registry.resolve_stage_layout(cfg)

    assert layout.folder_for("unknown-stage") is None


def test_ingest_n_from_block_returns_zero_for_invalid_names() -> None:
    assert ingest._n_from_block("players_two") == 0


def test_hgb_feat_unique_players_falls_back_to_hints_when_metrics_missing(tmp_path: Path) -> None:
    players = hgb_feat._unique_players(tmp_path / "missing.parquet", hints=[4, 2])

    assert players == [2, 4]


def test_isolated_metrics_locator_from_config_uses_prefix_template(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=Path("analysis/root")))

    locator = isolated_metrics.locator_from_config(cfg, seeds=(3,), player_counts=(2,))

    path = locator.path_for(3, 2)
    assert str(path).endswith("analysis/root_seed_3/2_players/2p_metrics.parquet")


def test_isolated_metrics_collect_empty_returns_summary_with_missing_jobs(tmp_path: Path) -> None:
    locator = isolated_metrics.MetricsLocator(data_root=tmp_path, seeds=(1,), player_counts=(2,))

    frame, summary = isolated_metrics.collect_isolated_metrics(locator)

    assert frame.empty
    assert summary.has_missing is True
    assert summary.loaded_pairs == 0
    assert summary.expected_pairs == 1
