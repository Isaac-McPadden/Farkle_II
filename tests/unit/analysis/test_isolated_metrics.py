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
