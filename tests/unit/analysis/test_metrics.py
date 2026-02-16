from __future__ import annotations

import pandas as pd
import pytest
from tests.helpers import metrics_samples as sample_data
from tests.helpers.diagnostic_fixtures import build_curated_fixture

from farkle.analysis import metrics
from farkle.analysis.stage_state import stage_done_path


@pytest.fixture
def sample_config(tmp_path, update_goldens):
    cfg = sample_data.stage_sample_run(tmp_path, refresh_inputs=update_goldens)
    metrics.run(cfg)
    return cfg


def test_collect_metrics_frames_preserves_stats(sample_config):
    iso_paths = [
        p
        for p in (
            sample_config.metrics_isolated_path(n) for n in sorted(sample_config.sim.n_players_list)
        )
        if p.exists()
    ]
    frame = metrics._collect_metrics_frames(iso_paths)

    required_prefix = [
        "strategy",
        "n_players",
        "games",
        "wins",
        "win_rate",
        "win_prob",
        "expected_score",
    ]
    assert list(frame.columns[: len(required_prefix)]) == required_prefix
    assert {2, 3}.issubset(set(frame["n_players"]))

    preserved_stats = {
        "total_games_strat",
        "sum_winning_score",
        "sq_sum_winning_score",
        "sum_n_rounds",
        "sq_sum_n_rounds",
        "false_wins_handled",
    }
    assert preserved_stats.issubset(set(frame.columns))

    by_seat = frame.groupby("n_players")["strategy"].nunique().to_dict()
    assert by_seat.get(2, 0) == 4
    assert by_seat.get(3, 0) == 4


def test_compute_seat_advantage_matches_manifest(sample_config):
    seat_df = metrics._compute_seat_advantage(sample_config, sample_config.curated_parquet)
    assert len(seat_df) == 12
    assert set(["seat", "wins", "games_with_seat", "win_rate"]).issubset(seat_df.columns)

    games_by_seat = seat_df.set_index("seat")["games_with_seat"].to_dict()
    assert games_by_seat[1] == 3
    assert games_by_seat[2] == 3
    assert games_by_seat[3] == 2
    assert games_by_seat[4] == 0

    seat1 = seat_df[seat_df["seat"] == 1].iloc[0]
    seat2 = seat_df[seat_df["seat"] == 2].iloc[0]
    assert seat1["wins"] == 2
    assert seat1["win_rate"] == pytest.approx(2 / 3)
    assert seat2["win_rate_delta_prev"] == pytest.approx(seat2["win_rate"] - seat1["win_rate"])


def test_seat_metrics_include_per_seat_stats(sample_config):
    seat_metrics = sample_config.metrics_input_path("seat_metrics.parquet")
    df = pd.read_parquet(seat_metrics)

    required = {
        "strategy",
        "seat",
        "n_players",
        "games",
        "wins",
        "win_rate",
        "mean_score",
        "mean_farkles",
        "mean_rounds",
    }
    assert required.issubset(df.columns)

    row = df[(df["seat"] == 1) & (df["n_players"] == 3)].iloc[0]
    assert row["games"] == 2
    assert row["wins"] == 1
    assert row["win_rate"] == pytest.approx(0.5)
    assert row["mean_rounds"] == pytest.approx(10.5)


def test_symmetry_checks_empty_when_no_symmetric_pairs(sample_config):
    symmetry_path = sample_config.metrics_input_path("symmetry_checks.parquet")
    df = pd.read_parquet(symmetry_path)

    required = {
        "strategy",
        "n_players",
        "observations",
        "mean_p1_farkles",
        "mean_p2_farkles",
        "farkle_diff",
        "mean_p1_rounds",
        "mean_p2_rounds",
        "rounds_diff",
        "farkle_flagged",
        "rounds_flagged",
    }
    assert required.issubset(df.columns)
    assert df.empty


def test_metrics_skip_when_up_to_date(sample_config, monkeypatch):
    done = stage_done_path(sample_config.metrics_stage_dir, "metrics")
    assert done.exists()

    def _boom(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("metrics should skip when up-to-date")

    monkeypatch.setattr("farkle.analysis.metrics._ensure_isolated_metrics", _boom)
    metrics.run(sample_config)


def test_win_rate_uncertainty_and_ordering():
    frame = pd.DataFrame(
        {
            "strategy": ["Aggro", "Control", "Zero"],
            "n_players": [2, 2, 2],
            "games": [50, 10, 0],
            "wins": [30, 1, 0],
            "win_rate": [0.6, 0.1, 0.0],
        }
    )

    out = metrics._add_win_rate_uncertainty(frame)

    assert list(out.columns[:8]) == [
        "strategy",
        "n_players",
        "games",
        "wins",
        "win_rate",
        "se_win_rate",
        "win_rate_ci_lo",
        "win_rate_ci_hi",
    ]
    assert out.loc[out["strategy"] == "Aggro", "se_win_rate"].item() == pytest.approx(
        (0.6 * 0.4 / 50) ** 0.5
    )
    assert out.loc[out["strategy"] == "Control", "win_rate_ci_hi"].item() == pytest.approx(
        0.1 + 1.96 * ((0.1 * 0.9 / 10) ** 0.5)
    )
    assert out.loc[out["strategy"] == "Zero", "se_win_rate"].item() == 0.0


def test_seat_metrics_and_advantage_from_synthetic_fixture(tmp_path):
    cfg, combined, _ = build_curated_fixture(tmp_path)

    seat_cfg = metrics.SeatMetricConfig(seat_range=(1, 2))
    seat_adv = metrics.compute_seat_advantage(cfg, combined, seat_cfg)
    assert seat_adv.loc[seat_adv["seat"] == 1, "win_rate"].item() == pytest.approx(2 / 3)
    assert seat_adv.loc[seat_adv["seat"] == 2, "win_rate"].item() == pytest.approx(1 / 3)

    seat_metrics = metrics.compute_seat_metrics(combined, seat_cfg)
    aggro_seat1 = seat_metrics[
        (seat_metrics["strategy"] == 1) & (seat_metrics["seat"] == 1)
    ].iloc[0]
    assert aggro_seat1["games"] == 3
    assert aggro_seat1["wins"] == 2
    assert aggro_seat1["mean_rounds"] == pytest.approx((6 + 9 + 12) / 3)

    control_seat2 = seat_metrics[
        (seat_metrics["strategy"] == 2) & (seat_metrics["seat"] == 2)
    ].iloc[0]
    assert control_seat2["wins"] == 1
    assert control_seat2["mean_score"] == pytest.approx((105 + 40 + 200) / 3)
