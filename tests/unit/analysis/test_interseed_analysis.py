from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from farkle.analysis import interseed_analysis
from farkle.analysis.game_stats_interseed import SeedInputs
from farkle.analysis.stage_registry import resolve_interseed_stage_layout
from farkle.config import AppConfig, IOConfig, SimConfig


def _make_cfg(tmp_path: Path, *, seed_list: list[int] | None = None) -> AppConfig:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=101),
    )
    if seed_list is not None:
        cfg.sim.seed_list = list(seed_list)
    return cfg


def test_interseed_run_success_writes_deterministic_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(tmp_path, seed_list=[9, 2])
    cfg.sim.n_players_list = [5, 2]

    prev_layout = cfg._stage_layout
    cfg.set_stage_layout(resolve_interseed_stage_layout(cfg))
    variance_path = cfg.variance_output_path("variance.parquet")
    meta_path = cfg.meta_output_path(2, "meta_2p.json")
    variance_path.parent.mkdir(parents=True, exist_ok=True)
    variance_path.write_text("ok")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text("ok")
    cfg._stage_layout = prev_layout

    monkeypatch.setattr(cfg, "interseed_ready", lambda: (True, ""))

    interseed_analysis.run(cfg, run_stages=False, force=True)

    summaries = list(cfg.analysis_dir.rglob(interseed_analysis.SUMMARY_NAME))
    assert len(summaries) == 1
    summary_path = summaries[0]
    assert summary_path.name == "interseed_summary.json"
    payload = json.loads(summary_path.read_text())

    assert payload["interseed_ready"] is True
    assert payload["stages"]["rng_diagnostics"]["enabled"] is True
    assert payload["stages"]["variance"]["outputs"] == [
        str(variance_path),
    ]
    # n_players_list is intentionally unsorted above; output ordering should be stable.
    assert payload["stages"]["meta"]["outputs"] == [
        str(meta_path),
    ]


def test_interseed_run_skips_when_inputs_not_ready(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, seed_list=[])

    interseed_analysis.run(cfg, run_stages=False)

    assert not list(cfg.analysis_dir.rglob(interseed_analysis.SUMMARY_NAME))


def test_run_s_tier_stability_mixed_seed_inputs_and_deterministic_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(tmp_path, seed_list=[3, 4, 7])

    seed3 = tmp_path / "seed3" / "analysis"
    seed4 = tmp_path / "seed4" / "analysis"
    seed7 = tmp_path / "seed7" / "analysis"
    for path in (seed3, seed4, seed7):
        path.mkdir(parents=True, exist_ok=True)

    # Seed 3: ranking + union candidates path (tests ranking-derived S-tier branch).
    seed3_h2h = seed3 / "head2head"
    seed3_h2h.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"strategy": "2", "rank": 2},
            {"strategy": 1, "rank": 1},
            {"strategy": "x", "rank": 3},
        ]
    ).to_csv(seed3_h2h / "h2h_significant_ranking.csv", index=False)
    (seed3_h2h / "h2h_union_candidates.json").write_text(json.dumps({"candidates": ["1", "2"]}))

    # Seed 4: direct S-tier json path (tests json loading branch).
    seed4_h2h = seed4 / "head2head"
    seed4_h2h.mkdir(parents=True, exist_ok=True)
    (seed4_h2h / "h2h_s_tiers.json").write_text(
        json.dumps({"2": "S", "1": "S+", "_meta": {"ignored": True}, "bad": 5})
    )

    # Seed 7: invalid/empty data should be ignored.
    seed7_h2h = seed7 / "head2head"
    seed7_h2h.mkdir(parents=True, exist_ok=True)
    (seed7_h2h / "h2h_s_tiers.json").write_text("not-json")

    monkeypatch.setattr(
        interseed_analysis,
        "_seed_analysis_dirs",
        lambda _cfg: [
            SeedInputs(seed=7, analysis_dir=seed7),
            SeedInputs(seed=4, analysis_dir=seed4),
            SeedInputs(seed=3, analysis_dir=seed3),
        ],
    )
    monkeypatch.setattr(cfg, "_interseed_input_folder", lambda stage: "head2head")

    interseed_analysis._run_s_tier_stability(cfg, force=True)

    json_output = cfg.interseed_stage_dir / "s_tier_stability.json"
    parquet_output = cfg.interseed_stage_dir / "s_tier_stability.parquet"
    assert json_output.name == "s_tier_stability.json"
    assert parquet_output.name == "s_tier_stability.parquet"

    payload = json.loads(json_output.read_text())
    assert payload["seeds"] == [3, 4]
    assert payload["pairs"][0]["seed_a"] == 3
    assert payload["pairs"][0]["seed_b"] == 4

    tier_frame = pd.read_parquet(parquet_output)
    assert tier_frame[["seed_a", "seed_b"]].drop_duplicates().values.tolist() == [[3, 4]]
    assert tier_frame["strategy_id"].tolist() == sorted(tier_frame["strategy_id"].tolist())


def test_run_s_tier_stability_skips_when_fewer_than_two_valid_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_cfg(tmp_path, seed_list=[11, 12])

    seed11 = tmp_path / "seed11" / "analysis"
    seed12 = tmp_path / "seed12" / "analysis"
    seed11.mkdir(parents=True, exist_ok=True)
    seed12.mkdir(parents=True, exist_ok=True)

    seed11_h2h = seed11 / "head2head"
    seed11_h2h.mkdir(parents=True, exist_ok=True)
    (seed11_h2h / "h2h_s_tiers.json").write_text(json.dumps({"a": "S+"}))

    seed12_h2h = seed12 / "head2head"
    seed12_h2h.mkdir(parents=True, exist_ok=True)
    (seed12_h2h / "h2h_s_tiers.json").write_text("{}")

    monkeypatch.setattr(
        interseed_analysis,
        "_seed_analysis_dirs",
        lambda _cfg: [
            SeedInputs(seed=11, analysis_dir=seed11),
            SeedInputs(seed=12, analysis_dir=seed12),
        ],
    )
    monkeypatch.setattr(cfg, "_interseed_input_folder", lambda stage: "head2head")

    interseed_analysis._run_s_tier_stability(cfg, force=True)

    assert not (cfg.interseed_stage_dir / "s_tier_stability.json").exists()
    assert not (cfg.interseed_stage_dir / "s_tier_stability.parquet").exists()


def test_helper_normalization_and_summary_branches(tmp_path: Path) -> None:
    ranking_path = tmp_path / "ranking.csv"
    pd.DataFrame(
        [
            {"strategy": 2, "rank": 2},
            {"strategy": "1", "rank": 1},
            {"strategy": "z", "rank": "bad"},
        ]
    ).to_csv(ranking_path, index=False)

    order, ranking = interseed_analysis._load_ranking(ranking_path)
    assert order == ["1", "2"]
    assert ranking == {"1": 1, "2": 2}

    data_a = interseed_analysis.SeedTierData(
        seed=1,
        analysis_dir=tmp_path,
        s_tiers={"A": "S+", "B": "S"},
        s_tiers_source="json",
        s_tiers_path=None,
        ranking={"A": 1, "B": 2},
        ranking_path=None,
        input_paths=[],
    )
    data_b = interseed_analysis.SeedTierData(
        seed=2,
        analysis_dir=tmp_path,
        s_tiers={"A": "S", "B": "S", "C": "S-"},
        s_tiers_source="json",
        s_tiers_path=None,
        ranking={"A": 2, "B": 1, "C": 3},
        ranking_path=None,
        input_paths=[],
    )

    merged_rows, summary = interseed_analysis._tier_flips(data_a, data_b)
    assert [row["strategy_id"] for row in merged_rows] == ["A", "B", "C"]
    assert summary == {"n_shared": 2, "n_flipped": 1, "flip_rate": 0.5}

    corr = interseed_analysis._rank_correlations(data_a.ranking, data_b.ranking)
    assert corr["n_common"] == 2
    assert corr["spearman"] is not None
    assert corr["kendall"] is not None
