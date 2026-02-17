from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from farkle.analysis import game_stats_interseed
from farkle.config import AppConfig, IOConfig, SimConfig


def test_seed_analysis_dirs_explicit_interseed_input_dir(tmp_path: Path) -> None:
    explicit_dir = tmp_path / "external_inputs"
    explicit_dir.mkdir(parents=True)
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results", interseed_input_dir=explicit_dir),
        sim=SimConfig(seed=42),
    )

    resolved = game_stats_interseed._seed_analysis_dirs(cfg)

    assert resolved == [game_stats_interseed.SeedInputs(seed=42, analysis_dir=explicit_dir)]


def test_seed_analysis_dirs_discovers_sibling_seed_folders(tmp_path: Path) -> None:
    prefix = tmp_path / "sim_results"
    for seed in (2, 5):
        (tmp_path / f"sim_results_seed_{seed}" / "analysis").mkdir(parents=True)
    # Non-matching sibling should be ignored.
    (tmp_path / "other_seed_9" / "analysis").mkdir(parents=True)

    cfg = AppConfig(io=IOConfig(results_dir_prefix=prefix), sim=SimConfig(seed=1))

    resolved = game_stats_interseed._seed_analysis_dirs(cfg)

    assert [entry.seed for entry in resolved] == [2, 5]
    assert all(entry.analysis_dir.exists() for entry in resolved)


def test_seed_analysis_dirs_dedups_and_filters_nonexistent(tmp_path: Path) -> None:
    prefix = tmp_path / "sim_results"
    (tmp_path / "sim_results_seed_3" / "analysis").mkdir(parents=True)
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=prefix),
        sim=SimConfig(seed=1, seed_pair=(3, 3)),
    )

    resolved = game_stats_interseed._seed_analysis_dirs(cfg)

    assert resolved == [
        game_stats_interseed.SeedInputs(
            seed=3,
            analysis_dir=tmp_path / "sim_results_seed_3" / "analysis",
        )
    ]


@pytest.mark.parametrize("stage_folder", ["preferred_game_stats", "legacy_game_stats"])
def test_seed_input_paths_resolves_preferred_and_legacy_stage_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_folder: str,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"), sim=SimConfig(seed=1))

    if stage_folder == "preferred_game_stats":
        monkeypatch.setattr(cfg, "_interseed_input_folder", lambda key: "preferred_game_stats")
    else:
        monkeypatch.setattr(cfg, "_interseed_input_folder", lambda key: None)
        cfg._stage_layout = type("Layout", (), {"folder_for": staticmethod(lambda key: "legacy_game_stats")})()

    seed_dir = tmp_path / "seed_analysis"
    target = seed_dir / stage_folder / "pooled" / "game_length_stats.parquet"
    target.parent.mkdir(parents=True)
    target.write_text("marker")

    found = game_stats_interseed._seed_input_paths(
        [game_stats_interseed.SeedInputs(seed=7, analysis_dir=seed_dir)],
        cfg,
        candidates=("game_length_stats.parquet", "game_length.parquet"),
    )

    assert found == [(7, target)]


def test_seed_input_paths_returns_empty_when_no_candidates(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"), sim=SimConfig(seed=1))
    seed_dir = tmp_path / "seed_analysis"
    seed_dir.mkdir()

    found = game_stats_interseed._seed_input_paths(
        [game_stats_interseed.SeedInputs(seed=7, analysis_dir=seed_dir)],
        cfg,
        candidates=(),
    )

    assert found == []


def test_load_seed_frames_normalizes_players_and_skips_empty(tmp_path: Path) -> None:
    populated = tmp_path / "seed_1.parquet"
    empty = tmp_path / "seed_2.parquet"
    pd.DataFrame([{"summary_level": "strategy", "strategy": 1, "players": 2, "metric": 5.0}]).to_parquet(populated)
    pd.DataFrame(columns=["summary_level", "strategy", "players", "metric"]).to_parquet(empty)

    frame = game_stats_interseed._load_seed_frames([(11, populated), (22, empty)])

    assert list(frame["seed"].unique()) == [11]
    assert "n_players" in frame.columns
    assert "players" not in frame.columns
    assert str(frame["n_players"].dtype) == "Int64"


def test_load_seed_frames_returns_empty_frame_when_all_inputs_empty(tmp_path: Path) -> None:
    empty = tmp_path / "seed_empty.parquet"
    pd.DataFrame(columns=["players"]).to_parquet(empty)

    frame = game_stats_interseed._load_seed_frames([(1, empty)])

    assert frame.empty


def test_aggregate_seed_stats_requires_n_players() -> None:
    frame = pd.DataFrame([{"summary_level": "strategy", "strategy": "A", "metric": 1.0, "seed": 1}])

    with pytest.raises(ValueError, match="missing n_players"):
        game_stats_interseed._aggregate_seed_stats(frame)


def test_aggregate_seed_stats_outputs_expected_ordering() -> None:
    frame = pd.DataFrame(
        [
            {"summary_level": "strategy", "strategy": "A", "n_players": 2, "metric": 10.0, "seed": 1},
            {"summary_level": "strategy", "strategy": "A", "n_players": 2, "metric": 14.0, "seed": 2},
        ]
    )

    out = game_stats_interseed._aggregate_seed_stats(frame)

    assert out.columns[:4].tolist() == ["summary_level", "strategy", "n_players", "n_seeds"]
    assert out.loc[0, "n_seeds"] == 2
    assert out.loc[0, "metric_seed_mean"] == pytest.approx(12.0)
    assert {"metric_seed_std", "metric_seed_ci_lo", "metric_seed_ci_hi"} <= set(out.columns)


def test_critical_values_uses_t_for_small_n_and_normal_for_large_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_seeds = pd.Series([2, game_stats_interseed.T_CRIT_N_SEEDS], index=["small", "large"])

    monkeypatch.setattr(game_stats_interseed.t, "ppf", lambda q, df: pd.Series(9.0, index=df.index))

    crit = game_stats_interseed._critical_values(n_seeds)

    assert crit.loc["small"] == pytest.approx(9.0)
    assert crit.loc["large"] == pytest.approx(game_stats_interseed.NORMAL_975)


def test_run_returns_early_when_no_seed_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"), sim=SimConfig(seed=1))
    monkeypatch.setattr(game_stats_interseed, "_seed_analysis_dirs", lambda _cfg: [])

    game_stats_interseed.run(cfg)

    assert not (cfg.interseed_stage_dir / game_stats_interseed.GAME_LENGTH_OUTPUT).exists()
    assert not (cfg.interseed_stage_dir / game_stats_interseed.MARGIN_OUTPUT).exists()


def test_run_returns_early_when_seed_files_are_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"), sim=SimConfig(seed=1))
    seeds = [game_stats_interseed.SeedInputs(seed=1, analysis_dir=tmp_path / "a")]

    monkeypatch.setattr(game_stats_interseed, "_seed_analysis_dirs", lambda _cfg: seeds)
    monkeypatch.setattr(game_stats_interseed, "_seed_input_paths", lambda *args, **kwargs: [(1, tmp_path / "in.parquet")])

    game_stats_interseed.run(cfg)

    assert not (cfg.interseed_stage_dir / game_stats_interseed.GAME_LENGTH_OUTPUT).exists()
    assert not (cfg.interseed_stage_dir / game_stats_interseed.MARGIN_OUTPUT).exists()


def test_run_up_to_date_bypasses_recompute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"), sim=SimConfig(seed=1))
    seeds = [game_stats_interseed.SeedInputs(seed=1, analysis_dir=tmp_path / "a")]
    input_paths = [(1, tmp_path / "in.parquet")]

    monkeypatch.setattr(game_stats_interseed, "_seed_analysis_dirs", lambda _cfg: seeds)
    monkeypatch.setattr(game_stats_interseed, "_seed_input_paths", lambda *args, **kwargs: input_paths)
    monkeypatch.setattr(game_stats_interseed, "stage_is_up_to_date", lambda *args, **kwargs: True)

    def _fail(*args, **kwargs):
        raise AssertionError("should not recompute when up-to-date")

    monkeypatch.setattr(game_stats_interseed, "_load_seed_frames", _fail)

    game_stats_interseed.run(cfg)


def test_run_writes_outputs_and_stage_stamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "results"), sim=SimConfig(seed=1))
    seeds = [
        game_stats_interseed.SeedInputs(seed=1, analysis_dir=tmp_path / "seed1"),
        game_stats_interseed.SeedInputs(seed=2, analysis_dir=tmp_path / "seed2"),
    ]
    game_inputs = [(1, tmp_path / "gl_1.parquet"), (2, tmp_path / "gl_2.parquet")]
    margin_inputs = [(1, tmp_path / "m_1.parquet"), (2, tmp_path / "m_2.parquet")]

    aggregated = pd.DataFrame(
        [
            {
                "summary_level": "strategy",
                "strategy": "A",
                "n_players": 2,
                "n_seeds": 2,
                "metric_seed_mean": 12.0,
            }
        ]
    )

    monkeypatch.setattr(game_stats_interseed, "_seed_analysis_dirs", lambda _cfg: seeds)

    def _seed_input_paths(_seeds, _cfg, *, candidates):
        return game_inputs if tuple(candidates) == game_stats_interseed.GAME_LENGTH_INPUTS else margin_inputs

    monkeypatch.setattr(game_stats_interseed, "_seed_input_paths", _seed_input_paths)
    monkeypatch.setattr(game_stats_interseed, "_load_seed_frames", lambda _paths: pd.DataFrame({"stub": [1]}))
    monkeypatch.setattr(game_stats_interseed, "_aggregate_seed_stats", lambda frame: aggregated)

    game_stats_interseed.run(cfg, force=True)

    game_output = cfg.interseed_stage_dir / game_stats_interseed.GAME_LENGTH_OUTPUT
    margin_output = cfg.interseed_stage_dir / game_stats_interseed.MARGIN_OUTPUT
    game_stamp = game_stats_interseed.stage_done_path(cfg.interseed_stage_dir, "interseed.game_length")
    margin_stamp = game_stats_interseed.stage_done_path(cfg.interseed_stage_dir, "interseed.margin")

    assert game_output.exists()
    assert margin_output.exists()
    assert game_stamp.exists()
    assert margin_stamp.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(game_output), aggregated)
    pd.testing.assert_frame_equal(pd.read_parquet(margin_output), aggregated)
