from pathlib import Path

import pandas as pd
import pytest

from farkle.analysis import isolated_metrics
from farkle.config import AppConfig


def test_metrics_locator_paths_and_validation(tmp_path):
    locator = isolated_metrics.MetricsLocator(
        data_root=tmp_path,
        seeds=[1, 2],
        player_counts=[2],
        override_roots={2: tmp_path / "alt"},
        results_template="seed_{seed}",
        metrics_template="{n}.parquet",
    )

    expected_default = tmp_path / "seed_1" / "2_players" / "2.parquet"
    expected_override = tmp_path / "alt" / "2_players" / "2.parquet"
    assert locator.path_for(1, 2) == expected_default
    assert locator.path_for(2, 2) == expected_override
    mapping = locator.as_mapping()
    assert mapping[1][2] == expected_default
    assert mapping[2][2] == expected_override


def test_collect_isolated_metrics_handles_missing_and_strict(tmp_path):
    locator = isolated_metrics.MetricsLocator(
        data_root=tmp_path,
        seeds=[7, 8],
        player_counts=[2],
    )

    metrics_path = locator.path_for(7, 2)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "strategy": ["A"],
            "wins": [4],
            "total_games_strat": [10],
            "sum_winning_score": [100.0],
            "sq_sum_winning_score": [1100.0],
            "sum_n_rounds": [40.0],
            "sq_sum_n_rounds": [500.0],
        }
    )
    frame.to_parquet(metrics_path, index=False)

    loaded, summary = isolated_metrics.collect_isolated_metrics(locator)
    assert not loaded.empty
    assert summary.loaded_pairs == 1
    assert summary.expected_pairs == 2
    assert summary.has_missing
    assert any("Missing metrics parquet" in note for note in summary.warnings)

    with pytest.raises(FileNotFoundError):
        isolated_metrics.collect_isolated_metrics(locator, strict=True)


def test_prepare_metrics_dataframe_recomputes_metrics_and_pads():
    cfg = AppConfig()
    cfg.sim.score_thresholds = [10]
    cfg.sim.dice_thresholds = [0]
    cfg.sim.smart_five_opts = [False]
    cfg.sim.smart_one_opts = [False]
    cfg.sim.consider_score_opts = [False]
    cfg.sim.consider_dice_opts = [False]
    cfg.sim.auto_hot_dice_opts = [False]
    cfg.sim.run_up_score_opts = [False]

    from farkle.simulation.simulation import generate_strategy_grid

    _, strategy_meta = generate_strategy_grid(
        score_thresholds=cfg.sim.score_thresholds,
        dice_thresholds=cfg.sim.dice_thresholds,
        smart_five_opts=cfg.sim.smart_five_opts,
        smart_one_opts=cfg.sim.smart_one_opts,
        consider_score_opts=cfg.sim.consider_score_opts,
        consider_dice_opts=cfg.sim.consider_dice_opts,
        auto_hot_dice_opts=cfg.sim.auto_hot_dice_opts,
        run_up_score_opts=cfg.sim.run_up_score_opts,
    )
    strategy_id = int(strategy_meta["strategy_id"].iloc[0])

    df = pd.DataFrame(
        {
            "strategy": [strategy_id],
            "wins": [5.0],
            "total_games_strat": [10],
            "sum_winner_hit_max_rounds": [2.0],
            "sum_winning_score": [200.0],
            "sq_sum_winning_score": [4200.0],
            "mean_winning_score": [20.0],
            "var_winning_score": [4.0],
        }
    )

    processed = isolated_metrics._prepare_metrics_dataframe(cfg, df, player_count=2)

    # Invariants: prepare helper preserves row counts of the strategy grid.
    assert len(processed) == len(isolated_metrics._STRATEGY_CACHE[id(cfg)])

    # Invariants: required normalized fields exist.
    required_fields = {
        "strategy",
        "n_players",
        "games",
        "wins",
        "win_rate",
        "win_prob",
        "false_wins_handled",
        "missing_before_pad",
        "mean_score",
        "sd_score",
    }
    assert required_fields.issubset(set(processed.columns))

    # Invariants on the populated strategy row after recompute correction.
    populated = processed.loc[processed["strategy"] == strategy_id].iloc[0]
    assert populated["wins"] == 3  # corrected by hit-flag subtraction
    assert populated["false_wins_handled"] == 2
    assert populated["games"] == 10
    assert populated["win_rate"] == pytest.approx(0.3)
    assert populated["n_players"] == 2

    # Invariants for padded rows: no synthetic partial outputs.
    padded = processed.loc[processed["strategy"] != strategy_id]
    if not padded.empty:
        assert (padded["wins"] == 0).all()
        assert (padded["win_rate"] == 0).all()


def test_collect_isolated_metrics_filters_player_counts_and_missing_seeds(tmp_path):
    locator = isolated_metrics.MetricsLocator(
        data_root=tmp_path,
        seeds=[11, 12],
        player_counts=[2, 4],
    )

    seed11_k2 = locator.path_for(11, 2)
    seed11_k2.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "strategy": [10, 11],
            "wins": [6, 4],
            "total_games_strat": [10, 10],
        }
    ).to_parquet(seed11_k2, index=False)

    loaded, summary = isolated_metrics.collect_isolated_metrics(locator)

    assert loaded[["seed", "k", "strategy"]].to_dict("records") == [
        {"seed": 11, "k": 2, "strategy": 10},
        {"seed": 11, "k": 2, "strategy": 11},
    ]
    assert summary.expected_pairs == 4
    assert summary.loaded_pairs == 1
    assert summary.has_missing is True
    assert summary.row_counts.loc[11, 2] == 2
    assert summary.row_counts.loc[11, 4] == 0
    assert summary.row_counts.loc[12, 2] == 0
    assert summary.strategy_counts.loc[11, 2] == 2
    assert "Seed 12 missing data for player counts [2, 4]" in summary.warnings
    assert "Player count 4 missing data for seeds [11, 12]" in summary.warnings


def test_collect_isolated_metrics_empty_isolates_schema_is_stable(tmp_path):
    locator = isolated_metrics.MetricsLocator(data_root=tmp_path, seeds=[31], player_counts=[2])

    frame, summary = isolated_metrics.collect_isolated_metrics(locator)

    assert frame.empty
    assert list(frame.columns) == [
        "strategy",
        "wins",
        "games",
        "win_rate",
        "winrate",
        "seed",
        "player_count",
        "k",
    ]
    assert summary.row_counts.index.to_list() == [31]
    assert summary.row_counts.columns.to_list() == [2]
    assert summary.row_counts.loc[31, 2] == 0
    assert summary.strategy_counts.loc[31, 2] == 0


def test_prepare_metrics_dataframe_rollups_and_optional_output_branches():
    cfg = AppConfig()
    cfg.sim.score_thresholds = [10]
    cfg.sim.dice_thresholds = [0]
    cfg.sim.smart_five_opts = [False]
    cfg.sim.smart_one_opts = [False]
    cfg.sim.consider_score_opts = [False]
    cfg.sim.consider_dice_opts = [False]
    cfg.sim.auto_hot_dice_opts = [False]
    cfg.sim.run_up_score_opts = [False]

    from farkle.simulation.simulation import generate_strategy_grid

    _, strategy_meta = generate_strategy_grid(
        score_thresholds=cfg.sim.score_thresholds,
        dice_thresholds=cfg.sim.dice_thresholds,
        smart_five_opts=cfg.sim.smart_five_opts,
        smart_one_opts=cfg.sim.smart_one_opts,
        consider_score_opts=cfg.sim.consider_score_opts,
        consider_dice_opts=cfg.sim.consider_dice_opts,
        auto_hot_dice_opts=cfg.sim.auto_hot_dice_opts,
        run_up_score_opts=cfg.sim.run_up_score_opts,
    )
    strategy_id = int(strategy_meta["strategy_id"].iloc[0])

    with_score = pd.DataFrame(
        {
            "strategy": [strategy_id],
            "wins": [5.0],
            "total_games_strat": [10],
            "sum_winning_score": [200.0],
            "sq_sum_winning_score": [4400.0],
            "mean_winning_score": [40.0],
            "var_winning_score": [100.0],
        }
    )
    out_with = isolated_metrics._prepare_metrics_dataframe(cfg, with_score, player_count=2)
    row = out_with.loc[out_with["strategy"] == strategy_id].iloc[0]

    assert row["expected_score"] == pytest.approx(20.0)
    assert row["mean_score"] == pytest.approx(40.0)
    assert row["sd_score"] == pytest.approx(10.0)

    without_score = pd.DataFrame(
        {
            "strategy": [strategy_id],
            "wins": [3.0],
            "total_games_strat": [10],
        }
    )

    with pytest.raises(KeyError, match="expected_score"):
        isolated_metrics._prepare_metrics_dataframe(cfg, without_score, player_count=2)


def test_build_isolated_metrics_control_flow_and_output_path(tmp_path, monkeypatch):
    cfg = AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    cfg.sim.seed = 123

    with pytest.raises(FileNotFoundError):
        isolated_metrics.build_isolated_metrics(cfg, player_count=4)

    src = cfg.results_root / "4_players" / "4p_metrics.parquet"
    src.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "strategy": [1],
            "wins": [1],
            "total_games_strat": [2],
            "sum_winning_score": [50.0],
            "sq_sum_winning_score": [2500.0],
            "mean_winning_score": [25.0],
            "var_winning_score": [1.0],
        }
    ).to_parquet(src, index=False)

    dst = cfg.metrics_isolated_path(4)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"already-up-to-date")
    src_ts = src.stat().st_mtime
    dst_ts = src_ts + 10
    import os

    os.utime(src, (src_ts, src_ts))
    os.utime(dst, (dst_ts, dst_ts))

    prepare_calls: list[int] = []
    write_calls: list[tuple[int, str]] = []

    def fake_prepare(_cfg, frame, player_count):
        prepare_calls.append(player_count)
        return frame.assign(n_players=player_count, games=2, win_rate=0.5, win_prob=0.5, expected_score=25.0)

    def fake_write(table, path):
        write_calls.append((table.num_rows, str(path)))

    monkeypatch.setattr(isolated_metrics, "_prepare_metrics_dataframe", fake_prepare)
    monkeypatch.setattr(isolated_metrics, "write_parquet_atomic", fake_write)

    short_path = isolated_metrics.build_isolated_metrics(cfg, player_count=4, force=False)
    assert short_path == cfg.metrics_isolated_path(4)
    assert prepare_calls == []
    assert write_calls == []

    rebuilt_path = isolated_metrics.build_isolated_metrics(cfg, player_count=4, force=True)
    assert rebuilt_path == cfg.metrics_isolated_path(4)
    assert prepare_calls == [4]
    assert write_calls and write_calls[0][1] == str(cfg.metrics_isolated_path(4))


@pytest.mark.parametrize(
    ("input_frame", "expected"),
    [
        (
            pd.DataFrame({"strategy": [1], "wins": [3], "total_games_strat": [6]}),
            {"games": 6, "winrate": 0.5},
        ),
        (
            pd.DataFrame({"strategy": [1], "win_rate": [0.42]}),
            {"winrate": 0.42},
        ),
        (
            pd.DataFrame({"strategy": [1, 2], "wins": [2, 0], "games": [4, 0]}),
            {"winrate": [0.5, 0.0]},
        ),
        (
            pd.DataFrame({"strategy": [1], "notes": ["missing metrics"]}),
            {"winrate": "nan"},
        ),
    ],
)
def test_load_job_branch_matrix(tmp_path, input_frame, expected):
    parquet_path = tmp_path / "job.parquet"
    input_frame.to_parquet(parquet_path, index=False)
    job = isolated_metrics.MetricJob(seed=9, player_count=3, path=parquet_path)

    loaded = isolated_metrics._load_job(job)

    assert {"strategy", "seed", "player_count", "k", "metrics_path", "winrate"}.issubset(
        set(loaded.columns)
    )
    assert loaded["seed"].tolist() == [9] * len(loaded)
    assert loaded["player_count"].tolist() == [3] * len(loaded)
    assert loaded["k"].tolist() == [3] * len(loaded)
    assert loaded["metrics_path"].tolist() == [str(parquet_path)] * len(loaded)

    if "games" in expected:
        assert loaded.loc[0, "games"] == expected["games"]
    if expected.get("winrate") == "nan":
        assert pd.isna(loaded.loc[0, "winrate"])
    elif isinstance(expected.get("winrate"), list):
        assert loaded["winrate"].tolist() == expected["winrate"]
    elif "winrate" in expected:
        assert loaded.loc[0, "winrate"] == pytest.approx(expected["winrate"])


def test_load_job_columns_union_and_metadata(tmp_path):
    parquet_path = tmp_path / "job_columns.parquet"
    pd.DataFrame(
        {
            "strategy": [1],
            "wins": [2],
            "total_games_strat": [4],
            "games": [4],
            "win_rate": [0.5],
            "winrate": [0.5],
            "notes": ["kept"],
        }
    ).to_parquet(parquet_path, index=False)
    job = isolated_metrics.MetricJob(seed=7, player_count=5, path=parquet_path)

    loaded = isolated_metrics._load_job(job, columns=["strategy", "notes"])

    assert "notes" in loaded.columns
    assert loaded.loc[0, "notes"] == "kept"
    assert loaded.loc[0, "seed"] == 7
    assert loaded.loc[0, "player_count"] == 5
    assert loaded.loc[0, "k"] == 5
    assert loaded.loc[0, "metrics_path"] == str(parquet_path)


def test_locator_from_config_data_root_prefix_and_seed_paths_copy(tmp_path):
    cfg = AppConfig()
    cfg.io.results_dir_prefix = Path("relative_results")
    cfg.sim.n_players_list = [2, 5]

    explicit = isolated_metrics.locator_from_config(
        cfg,
        seeds=[1],
        data_root=tmp_path / "explicit",
        override_roots={1: "custom_seed"},
    )
    assert explicit.path_for(1, 2) == tmp_path / "explicit" / "custom_seed" / "2_players" / "2p_metrics.parquet"

    rel_prefix = isolated_metrics.locator_from_config(cfg, seeds=[2])
    assert rel_prefix.path_for(2, 5) == Path("data") / "relative_results_seed_2" / "5_players" / "5p_metrics.parquet"

    cfg.io.results_dir_prefix = tmp_path / "absolute_prefix"
    abs_prefix = isolated_metrics.locator_from_config(cfg, seeds=[3])
    assert abs_prefix.path_for(3, 2) == tmp_path / "absolute_prefix_seed_3" / "2_players" / "2p_metrics.parquet"

    paths_copy = abs_prefix.seed_paths
    paths_copy[3] = tmp_path / "mutated"
    assert abs_prefix.path_for(3, 2) != tmp_path / "mutated" / "2_players" / "2p_metrics.parquet"


def test_metrics_locator_validation_errors():
    with pytest.raises(ValueError, match="at least one seed"):
        isolated_metrics.MetricsLocator(data_root=Path("."), seeds=[], player_counts=[2])
    with pytest.raises(ValueError, match="at least one player_count"):
        isolated_metrics.MetricsLocator(data_root=Path("."), seeds=[1], player_counts=[])


def test_summarize_warning_branches_empty_loaded_and_mismatch(tmp_path):
    empty_locator = isolated_metrics.MetricsLocator(data_root=tmp_path, seeds=[1], player_counts=[2])
    empty_path = empty_locator.path_for(1, 2)
    empty_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"strategy": [], "wins": []}).to_parquet(empty_path, index=False)

    empty_frame, empty_summary = isolated_metrics.collect_isolated_metrics(empty_locator)
    assert empty_frame.empty
    assert "No metrics data loaded; verify locator paths." in empty_summary.warnings

    mismatch_locator = isolated_metrics.MetricsLocator(data_root=tmp_path, seeds=[10, 11], player_counts=[2])
    p10 = mismatch_locator.path_for(10, 2)
    p11 = mismatch_locator.path_for(11, 2)
    p10.parent.mkdir(parents=True, exist_ok=True)
    p11.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"strategy": [1], "wins": [1], "games": [2]}).to_parquet(p10, index=False)
    pd.DataFrame({"strategy": [1, 2], "wins": [1, 1], "games": [2, 2]}).to_parquet(p11, index=False)

    _, mismatch_summary = isolated_metrics.collect_isolated_metrics(mismatch_locator)
    assert any("Strategy count mismatch" in warning for warning in mismatch_summary.warnings)


def test_helper_branches_games_mode_normalize_compress_and_padding(monkeypatch):
    assert isolated_metrics._games_mode(pd.Series([3, 3, 4])) == 3
    assert isolated_metrics._games_mode(pd.Series([None, 7, 8])) == 7
    assert isolated_metrics._games_mode(pd.Series([None, None])) == 0.0

    assert isolated_metrics._normalize_metric_name("winning_score") == "score"
    assert isolated_metrics._normalize_metric_name("winner_n_rounds") == "n_rounds"
    assert isolated_metrics._normalize_metric_name("already_normalized") == "already_normalized"

    compressed = isolated_metrics._compress_metric_columns(
        pd.DataFrame(
            {
                "var_winner_hit_max_rounds": [1.0],
                "sum_winner_hit_max_rounds": [2.0],
                "sq_sum_winner_hit_max_rounds": [3.0],
                "mean_winner_hit_max_rounds": [4.0],
                "var_lonely": [2.0],
                "var_winning_score": [9.0],
                "mean_winning_score": [5.0],
                "sum_winning_score": [10.0],
                "sq_sum_winning_score": [20.0],
            }
        )
    )
    assert "var_winner_hit_max_rounds" not in compressed.columns
    assert "var_lonely" in compressed.columns
    assert "mean_score" in compressed.columns
    assert "sd_score" in compressed.columns
    assert "var_winning_score" not in compressed.columns

    cfg = AppConfig()
    fake_index = pd.Index([101, 202], name="strategy", dtype="int64")
    monkeypatch.setattr(isolated_metrics, "generate_strategy_grid", lambda **_: (None, pd.DataFrame({"strategy_id": [101, 202]})))
    isolated_metrics._STRATEGY_CACHE.pop(id(cfg), None)
    first = isolated_metrics._strategy_index(cfg)
    second = isolated_metrics._strategy_index(cfg)
    assert first.equals(fake_index)
    assert second.equals(fake_index)

    padded = isolated_metrics._pad_strategies(cfg, pd.DataFrame({"wins": [1]}, index=pd.Index([101], name="strategy")))
    assert padded.index.tolist() == [101, 202]
    assert pd.isna(padded.loc[202, "wins"])
