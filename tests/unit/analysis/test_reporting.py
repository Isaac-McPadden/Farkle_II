from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from farkle.analysis import reporting
from farkle.config import AppConfig, IOConfig, SimConfig


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_reporting_artifacts(cfg: AppConfig, *, players: int = 2) -> None:
    layout = reporting._analysis_layout(cfg)
    _write_frame(
        reporting._ratings_path(cfg.analysis_dir, layout=layout),
        pd.DataFrame(
            [
                {"strategy": "A", "players": players, "mu": 32.0, "sigma": 2.0},
                {"strategy": "B", "players": players, "mu": 29.0, "sigma": 2.5},
                {"strategy": "C", "players": players, "mu": 25.0, "sigma": 3.0},
            ]
        ),
    )
    _write_frame(
        cfg.meta_output_path(players, f"strategy_summary_{players}p_meta.parquet"),
        pd.DataFrame(
            [
                {
                    "strategy_id": "A",
                    "players": players,
                    "win_rate": 0.60,
                    "ci_lo": 0.55,
                    "ci_hi": 0.65,
                    "n_seeds": 2,
                },
                {
                    "strategy_id": "B",
                    "players": players,
                    "win_rate": 0.30,
                    "ci_lo": 0.25,
                    "ci_hi": 0.35,
                    "n_seeds": 2,
                },
            ]
        ),
    )
    _write_frame(
        cfg.analysis_dir / f"feature_importance_{players}p.parquet",
        pd.DataFrame({"name": ["keep_hot_dice", "bank_early"], "gain": [2.0, 1.0]}),
    )
    seed_root = reporting._stage_candidates(
        cfg.analysis_dir,
        "seed_summaries",
        layout=layout,
        filename=Path(f"{players}p"),
    )[0]
    _write_frame(
        seed_root / f"strategy_summary_{players}p_seed101.parquet",
        pd.DataFrame(
            [
                {
                    "strategy_id": "A",
                    "players": players,
                    "seed": 101,
                    "win_rate": 0.58,
                    "ci_lo": 0.52,
                    "ci_hi": 0.64,
                },
                {
                    "strategy_id": "B",
                    "players": players,
                    "seed": 101,
                    "win_rate": 0.31,
                    "ci_lo": 0.25,
                    "ci_hi": 0.37,
                },
            ]
        ),
    )
    _write_frame(
        seed_root / f"strategy_summary_{players}p_seed202.parquet",
        pd.DataFrame(
            [
                {
                    "strategy": "A",
                    "players": players,
                    "seed": 202,
                    "win_rate": 0.62,
                    "ci_lo": 0.56,
                    "ci_hi": 0.68,
                },
                {
                    "strategy": "B",
                    "players": players,
                    "seed": 202,
                    "win_rate": 0.29,
                    "ci_lo": 0.23,
                    "ci_hi": 0.35,
                },
            ]
        ),
    )

    tier_path = reporting._tier_path(cfg.analysis_dir, layout=layout)
    tier_path.parent.mkdir(parents=True, exist_ok=True)
    tier_path.write_text(
        json.dumps({"A": 0, "B": 1, "C": 2}),
        encoding="utf-8",
    )

    _write_frame(
        reporting._post_h2h_path(
            cfg.analysis_dir,
            "bonferroni_decisions.parquet",
            layout=layout,
        ),
        pd.DataFrame(
            [
                {"a": "A", "b": "B", "players": players, "wins_a": 7, "wins_b": 3, "games": 10},
                {"a": "B", "b": "C", "players": players, "wins_a": 6, "wins_b": 4, "games": 10},
            ]
        ),
    )
    ranking_path = reporting._post_h2h_path(
        cfg.analysis_dir,
        "h2h_significant_ranking.csv",
        layout=layout,
    )
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"strategy": ["A", "B", "C"], "rank": [1, 2, 3], "players": [players] * 3}
    ).to_csv(ranking_path, index=False)
    h2h_s_tiers_path = reporting._post_h2h_path(
        cfg.analysis_dir,
        "h2h_s_tiers.json",
        layout=layout,
    )
    h2h_s_tiers_path.parent.mkdir(parents=True, exist_ok=True)
    h2h_s_tiers_path.write_text(
        json.dumps({"A": "S+", "B": "S", "C": "S-"}),
        encoding="utf-8",
    )
    cfg.meta_output_path(players, f"meta_{players}p.json").write_text(
        json.dumps({"i2": 0.12, "tau2": 0.01, "ignored": "bad"}),
        encoding="utf-8",
    )
    (cfg.analysis_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "config_hash": "sha-report",
                "git_commit": "abc123",
                "run_timestamp": "2026-03-20T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )


def test_generate_report_for_players_writes_markdown_and_plots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(n_players_list=[2], seed=7),
    )
    _write_reporting_artifacts(cfg, players=2)
    original_write_text = Path.write_text

    def write_text_utf8(self: Path, data: str, *args, **kwargs):
        kwargs.setdefault("encoding", "utf-8")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(reporting.Path, "write_text", write_text_utf8)

    report_path = reporting.generate_report_for_players(cfg, 2, force=True)
    rerun_path = reporting.generate_report_for_players(cfg, 2, force=False)

    assert rerun_path == report_path
    assert report_path.exists()
    assert (cfg.analysis_dir / "plots" / "2p" / "ladder_2p.png").exists()
    assert (cfg.analysis_dir / "plots" / "2p" / "h2h_heatmap_2p.png").exists()
    assert (cfg.analysis_dir / "plots" / "2p" / "feature_importance_2p.png").exists()
    assert (cfg.analysis_dir / "plots" / "2p" / "seed_forest_2p.png").exists()

    body = report_path.read_text(encoding="utf-8")
    assert "# 2-player Farkle strategy report" in body
    assert "Head-to-head S-tier breakdown" in body
    assert "Meta-analysis metrics" in body
    assert "A: μ=32.00" in body
    assert "keep_hot_dice: 66.7%" in body


def test_run_report_discovers_players_and_skips_missing_artifacts(
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(n_players_list=[2, 3], seed=7),
    )
    _write_reporting_artifacts(cfg, players=2)
    original_write_text = Path.write_text

    def write_text_utf8(self: Path, data: str, *args, **kwargs):
        kwargs.setdefault("encoding", "utf-8")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(reporting.Path, "write_text", write_text_utf8)

    with caplog.at_level(logging.WARNING):
        reporting.run_report(cfg, force=True)

    assert (cfg.analysis_dir / "report_2p.md").exists()
    assert not (cfg.analysis_dir / "report_3p.md").exists()
    assert any(
        record.message == "Skipping report due to missing artifacts"
        and getattr(record, "players", None) == 3
        for record in caplog.records
    )


def test_reporting_helpers_cover_path_loading_and_text_branches(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    results_root = tmp_path / "results_seed_9"
    analysis_dir = results_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    assert reporting._analysis_dir(SimpleNamespace(results_root=results_root)) == analysis_dir
    assert reporting._analysis_dir(
        SimpleNamespace(
            io=SimpleNamespace(results_dir_prefix=Path("sample")),
            sim=SimpleNamespace(seed=4),
        )
    ) == Path("data") / "sample_seed_4" / "analysis"
    with pytest.raises(AttributeError, match="analysis_dir"):
        reporting._analysis_dir(SimpleNamespace())

    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "cfg-results"))
    legacy_meta = cfg.analysis_dir / "99_meta"
    legacy_meta.mkdir(parents=True, exist_ok=True)
    with caplog.at_level(logging.WARNING):
        candidates = reporting._stage_candidates(
            cfg.analysis_dir,
            "meta",
            layout=cfg.stage_layout,
            filename=Path("2p") / "artifact.parquet",
        )
    assert candidates[0] == (
        cfg.meta_stage_dir / "2p" / "artifact.parquet" / "2p" / "artifact.parquet"
    )
    assert any(
        record.message == "Legacy stage directory detected; prefer layout-aware helpers"
        for record in caplog.records
    )

    output = tmp_path / "output.txt"
    input_path = tmp_path / "input.txt"
    input_path.write_text("in", encoding="utf-8")
    output.write_text("out", encoding="utf-8")
    assert reporting._output_is_fresh(output, [input_path], force=True) is False
    assert reporting._output_is_fresh(output, [tmp_path / "missing.txt"], force=False) is False
    assert reporting._output_is_fresh(output, [input_path], force=False) is True

    with pytest.raises(reporting.ReportError, match="Expected winner to be a scalar"):
        reporting._extract_scalar(pd.Series([1, 2]), label="winner")
    assert reporting._extract_scalar(pd.Series([3]), label="winner") == 3

    assert reporting._format_rate(float("nan")) == "nan"
    assert reporting._top_feature_bullets(pd.DataFrame()) == ["Feature importances unavailable"]
    assert reporting._seed_stability_summary(
        pd.DataFrame({"strategy_id": ["A"], "seed": [1], "win_rate": [0.5]})
    ) == "Single seed available; stability not assessed."
    assert reporting._tier_buckets_from_ranked(["A", "A", "B"]) == {"S": ["A", "B"]}
    assert reporting._tiers_section({}) == "No tier information available."
    assert reporting._s_tiers_section({"A": "S+", "B": "bad"}) == [
        "### Head-to-head S-tier breakdown",
        "- S+: A",
    ]

    feature_path = analysis_dir / "feature_importance_2p.parquet"
    _write_frame(
        feature_path,
        pd.DataFrame({"feature": ["x", "y"], "importance": [0.0, 0.0]}),
    )
    feature_df = reporting._load_feature_importance(analysis_dir, 2)
    assert feature_df["importance"].sum() == 0.0
    _write_frame(feature_path, pd.DataFrame({"other": ["x"], "importance": [1.0]}))
    assert reporting._load_feature_importance(analysis_dir, 2).empty

    ranking_path = analysis_dir / "h2h_significant_ranking.csv"
    pd.DataFrame({"strategy": ["B", "A"], "players": [2, 2]}).to_csv(ranking_path, index=False)
    ranking = reporting._load_h2h_ranking(analysis_dir, 2)
    assert ranking.to_dict(orient="records") == [
        {"strategy": "B", "rank": 1},
        {"strategy": "A", "rank": 2},
    ]

    h2h_s_tiers = analysis_dir / "h2h_s_tiers.json"
    h2h_s_tiers.write_text("{bad json}", encoding="utf-8")
    assert reporting._load_h2h_s_tiers(analysis_dir) == {}
    h2h_s_tiers.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
    assert reporting._load_h2h_s_tiers(analysis_dir) == {}

    meta_json = analysis_dir / "meta_2p.json"
    meta_json.write_text("{bad json}", encoding="utf-8")
    bare_cfg = SimpleNamespace(analysis_dir=analysis_dir)
    assert reporting._load_meta_json(bare_cfg, 2) == {}
    meta_json.write_text(json.dumps(["bad"]), encoding="utf-8")
    assert reporting._load_meta_json(bare_cfg, 2) == {}
    meta_json.write_text(json.dumps({"i2": 0.2, "bad": "x"}), encoding="utf-8")
    assert reporting._load_meta_json(bare_cfg, 2) == {"i2": 0.2}

    run_meta = analysis_dir / "run_metadata.json"
    run_meta.write_text("{bad json}", encoding="utf-8")
    assert reporting._load_run_metadata(analysis_dir) == {}
    run_meta.write_text(json.dumps(["bad"]), encoding="utf-8")
    assert reporting._load_run_metadata(analysis_dir) == {}


def test_reporting_negative_load_and_plot_branches(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(n_players_list=[2], seed=7),
    )

    with pytest.raises(reporting.ReportError, match="Missing ratings parquet"):
        reporting._load_ratings(cfg.analysis_dir, 2)

    ratings_path = reporting._ratings_path(cfg.analysis_dir, layout=cfg.stage_layout)
    _write_frame(
        ratings_path,
        pd.DataFrame({"strategy": ["A"], "players": [2], "mu": [1.0]}),
    )
    with pytest.raises(reporting.ReportError, match="missing required columns"):
        reporting._load_ratings(cfg.analysis_dir, 2)

    _write_frame(ratings_path, pd.DataFrame({"strategy": ["A"], "mu": [1.0], "sigma": [2.0]}))
    with pytest.raises(reporting.ReportError, match="player-count column"):
        reporting._load_ratings(cfg.analysis_dir, 2)

    _write_frame(
        ratings_path,
        pd.DataFrame({"strategy": ["A"], "players": [3], "mu": [1.0], "sigma": [2.0]}),
    )
    with pytest.raises(reporting.ReportError, match="No ratings found for 2-player games"):
        reporting._load_ratings(cfg.analysis_dir, 2)

    _write_frame(
        ratings_path,
        pd.DataFrame({"strategy": ["A"], "players": [2], "mu": [1.0], "sigma": [2.0]}),
    )
    tier_path = reporting._tier_path(cfg.analysis_dir, layout=cfg.stage_layout)
    tier_path.parent.mkdir(parents=True, exist_ok=True)
    tier_path.write_text(json.dumps({"A": 0}), encoding="utf-8")
    _write_frame(
        reporting._post_h2h_path(
            cfg.analysis_dir,
            "bonferroni_decisions.parquet",
            layout=cfg.stage_layout,
        ),
        pd.DataFrame({"a": ["A"], "b": ["B"], "players": [2]}),
    )

    with caplog.at_level(logging.INFO):
        assert reporting.plot_h2h_heatmap_for_players(cfg, 2, force=True) is None
    assert any("win rates unavailable" in record.message for record in caplog.records)

    assert reporting.plot_feature_importance_for_players(cfg, 2, force=True) is None
    seed_root = reporting._stage_candidates(
        cfg.analysis_dir,
        "seed_summaries",
        layout=cfg.stage_layout,
        filename=Path("2p"),
    )[0]
    _write_frame(
        seed_root / "strategy_summary_2p_seed101.parquet",
        pd.DataFrame({"strategy_id": ["A"], "players": [2], "seed": [101], "win_rate": [0.5]}),
    )
    assert reporting.plot_seed_variability_for_players(cfg, 2, force=True) is None


def test_reporting_loader_fallback_and_seed_h2h_branches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    ratings_path = reporting._ratings_path(analysis_dir)
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_path.write_text("stub", encoding="utf-8")

    parquet_calls: list[tuple[str, tuple[str, ...] | None]] = []

    def fake_read_parquet(path: Path, columns=None):
        parquet_calls.append((Path(path).name, tuple(columns) if columns else None))
        if columns is not None:
            raise ValueError("column projection unavailable")
        return pd.DataFrame({"player_count": [4, 2, 4]})

    original_read_parquet = reporting.pd.read_parquet
    monkeypatch.setattr(reporting.pd, "read_parquet", fake_read_parquet)

    players = reporting._sim_player_counts(
        SimpleNamespace(
            analysis=SimpleNamespace(n_players_list=[3]),
            sim=SimpleNamespace(n_players_list=[]),
        ),
        analysis_dir,
    )

    assert players == [2, 3, 4]
    assert parquet_calls == [("ratings_k_weighted.parquet", ("strategy", "mu", "sigma", "players")), ("ratings_k_weighted.parquet", None)]
    with pytest.raises(reporting.ReportError, match="Unable to determine player counts"):
        reporting._sim_player_counts(
            SimpleNamespace(
                analysis=SimpleNamespace(n_players_list=[]),
                sim=SimpleNamespace(n_players_list=[]),
            ),
            tmp_path / "missing-analysis",
        )
    monkeypatch.setattr(reporting.pd, "read_parquet", original_read_parquet)

    cfg = AppConfig(
        io=IOConfig(
            results_dir_prefix=tmp_path / "results",
            meta_analysis_dir=tmp_path / "meta-analysis",
        ),
        sim=SimConfig(n_players_list=[2], seed=7),
    )
    layout = reporting._analysis_layout(cfg)

    assert reporting._load_meta_summary(cfg, 2).empty
    _write_frame(
        reporting._meta_artifact_path(cfg, 2, "strategy_summary_2p_meta.parquet"),
        pd.DataFrame({"other": [1], "players": [2]}),
    )
    assert reporting._load_meta_summary(cfg, 2).empty
    _write_frame(
        reporting._meta_artifact_path(cfg, 2, "strategy_summary_2p_meta.parquet"),
        pd.DataFrame(
            [
                {"strategy_id": "B", "players": 3, "win_rate": 0.9},
                {"strategy_id": "A", "players": 2, "win_rate": 0.4},
            ]
        ),
    )
    assert reporting._load_meta_summary(cfg, 2).to_dict(orient="records") == [
        {"strategy": "A", "players": 2, "win_rate": 0.4}
    ]

    seed_root = reporting._stage_candidates(
        cfg.analysis_dir,
        "seed_summaries",
        layout=layout,
        filename=Path("2p"),
    )[0]
    _write_frame(seed_root / "strategy_summary_2p_seed001.parquet", pd.DataFrame())
    _write_frame(
        cfg.analysis_dir / "strategy_summary_2p_seed002.parquet",
        pd.DataFrame({"strategy": ["A"], "players": [2], "win_rate": [0.55]}),
    )
    _write_frame(
        cfg.meta_analysis_dir / "strategy_summary_2p_seed003.parquet",
        pd.DataFrame({"strategy_id": ["B"], "players": [3]}),
    )

    seed_paths = reporting._seed_summary_paths(cfg, 2, layout=layout)
    assert cfg.meta_analysis_dir / "strategy_summary_2p_seed003.parquet" in seed_paths
    assert reporting._load_seed_summaries(cfg, 2, layout=layout).to_dict(orient="records") == [
        {"strategy_id": "A", "seed": 0, "win_rate": 0.55}
    ]

    assert reporting._load_h2h_decisions(tmp_path / "missing", 2).empty
    decisions_path = reporting._post_h2h_path(
        cfg.analysis_dir,
        "bonferroni_decisions.parquet",
        layout=layout,
    )
    _write_frame(decisions_path, pd.DataFrame())
    assert reporting._load_h2h_decisions(cfg.analysis_dir, 2, layout=layout).empty
    _write_frame(decisions_path, pd.DataFrame({"a": ["A"], "players": [2]}))
    assert reporting._load_h2h_decisions(cfg.analysis_dir, 2, layout=layout).empty
    _write_frame(
        decisions_path,
        pd.DataFrame(
            [
                {"a": "A", "b": "B", "players": 2, "win_rate": 0.6},
                {"a": "C", "b": "D", "players": 3, "win_rate": 0.4},
            ]
        ),
    )
    assert reporting._load_h2h_decisions(cfg.analysis_dir, 2, layout=layout).to_dict(
        orient="records"
    ) == [{"a": "A", "b": "B", "players": 2, "win_rate": 0.6}]

    assert reporting._load_h2h_ranking(tmp_path / "missing", 2).empty
    ranking_path = reporting._post_h2h_path(
        cfg.analysis_dir,
        "h2h_significant_ranking.csv",
        layout=layout,
    )
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"other": ["A"]}).to_csv(ranking_path, index=False)
    assert reporting._load_h2h_ranking(cfg.analysis_dir, 2, layout=layout).empty
    pd.DataFrame({"strategy": ["B", "A"], "players": [3, 2]}).to_csv(ranking_path, index=False)
    assert reporting._load_h2h_ranking(cfg.analysis_dir, 2, layout=layout).to_dict(
        orient="records"
    ) == [{"strategy": "A", "rank": 2}]

    assert reporting._load_h2h_s_tiers(tmp_path / "missing") == {}
    h2h_s_tiers_path = reporting._post_h2h_path(
        cfg.analysis_dir,
        "h2h_s_tiers.json",
        layout=layout,
    )
    h2h_s_tiers_path.parent.mkdir(parents=True, exist_ok=True)
    h2h_s_tiers_path.write_text(json.dumps({"A": "S+", "B": 2}), encoding="utf-8")
    assert reporting._load_h2h_s_tiers(cfg.analysis_dir, layout=layout) == {"A": "S+"}


def test_reporting_heatmap_order_body_and_seed_summary_branches(
    tmp_path: Path,
    monkeypatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    ratings = pd.DataFrame(
        [
            {"strategy": "A", "players": 2, "mu": 10.0, "sigma": 1.0},
            {"strategy": "B", "players": 2, "mu": 9.0, "sigma": 1.5},
        ]
    )
    ranking = pd.DataFrame({"strategy": ["B", "A"], "rank": [1, 2]})
    artifacts = reporting._ReportArtifacts(
        ratings=ratings,
        meta_summary=pd.DataFrame(),
        feature_importance=pd.DataFrame(),
        seed_summaries=pd.DataFrame(
            {
                "strategy_id": ["A", "A", "B", "B"],
                "seed": [1, 2, 1, 2],
                "win_rate": [0.60, 0.60, np.nan, np.nan],
            }
        ),
        tiers={"B": 0, "A": 1},
        h2h_decisions=pd.DataFrame(),
        h2h_ranking=pd.DataFrame(columns=["strategy", "rank"]),
        h2h_s_tiers={},
        heterogeneity={},
        run_metadata={},
    )

    assert reporting._determine_heatmap_order(artifacts, 2) == ["B", "A"]
    artifacts.h2h_ranking = ranking
    assert reporting._determine_heatmap_order(artifacts, 2) == ["B", "A"]
    artifacts.tiers = {}
    assert reporting._determine_heatmap_order(artifacts, 2) == ["B", "A"]

    moderate_seed_df = pd.DataFrame(
        {
            "strategy_id": ["A", "A", "B", "B"],
            "seed": [1, 2, 1, 2],
            "win_rate": [0.50, 0.52, 0.48, 0.50],
        }
    )
    large_seed_df = pd.DataFrame(
        {
            "strategy_id": ["A", "A", "B", "B"],
            "seed": [1, 2, 1, 2],
            "win_rate": [0.10, 0.90, 0.20, 0.80],
        }
    )
    unavailable_seed_df = pd.DataFrame(
        {
            "strategy_id": ["A", "A"],
            "seed": [1, 2],
            "win_rate": [np.nan, np.nan],
        }
    )
    assert "moderate" in reporting._seed_stability_summary(moderate_seed_df)
    assert "large" in reporting._seed_stability_summary(large_seed_df)
    assert reporting._seed_stability_summary(unavailable_seed_df) == "Seed variability data unavailable."

    body = reporting._build_report_body(
        2,
        artifacts,
        {"ladder": tmp_path / "ladder.png", "h2h": None, "features": None, "seed": None},
    )
    assert "Feature importances unavailable" in body
    assert "Head-to-head results not available for this player count." in body
    assert "Seed variability plot not generated." in body
    assert "(plot not generated)" in body
    assert "Head-to-head S-tier breakdown" not in body
    assert "Meta-analysis metrics" not in body

    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(n_players_list=[2], seed=7),
    )
    empty_ranking = pd.DataFrame(columns=["strategy", "rank"])

    monkeypatch.setattr(
        reporting,
        "_load_h2h_decisions",
        lambda *args, **kwargs: pd.DataFrame({"a": ["A"], "b": ["B"], "win_rate": [0.6]}),
    )
    monkeypatch.setattr(
        reporting,
        "_load_ratings",
        lambda *args, **kwargs: pd.DataFrame(columns=["strategy", "players", "mu", "sigma"]),
    )
    monkeypatch.setattr(reporting, "_load_tiers", lambda *args, **kwargs: {})
    monkeypatch.setattr(reporting, "_load_h2h_ranking", lambda *args, **kwargs: empty_ranking)
    monkeypatch.setattr(reporting, "_load_h2h_s_tiers", lambda *args, **kwargs: {})
    monkeypatch.setattr(reporting, "_output_is_fresh", lambda *args, **kwargs: False)

    with caplog.at_level(logging.INFO):
        assert reporting.plot_h2h_heatmap_for_players(cfg, 2, force=True) is None
    assert any("no strategies available" in record.message for record in caplog.records)

    monkeypatch.setattr(reporting, "_load_ratings", lambda *args, **kwargs: ratings.copy())
    heatmap_path = reporting.plot_h2h_heatmap_for_players(cfg, 2, force=True)
    assert heatmap_path is not None and heatmap_path.exists()

    monkeypatch.setattr(
        reporting,
        "_load_h2h_decisions",
        lambda *args, **kwargs: pd.DataFrame(
            {"a": ["A"], "b": ["B"], "wins_a": [0], "wins_b": [0], "games": [0]}
        ),
    )
    monkeypatch.setattr(
        reporting,
        "_as_float",
        lambda value: float("nan") if pd.isna(value) else float(value),
    )
    zero_division_path = reporting.plot_h2h_heatmap_for_players(cfg, 2, force=True)
    assert zero_division_path is not None and zero_division_path.exists()
