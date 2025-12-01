from __future__ import annotations

import pandas as pd
import pytest
from tests.helpers import metrics_samples as sample_data

from farkle.analysis import metrics


@pytest.fixture
def sample_config(tmp_path, update_goldens):
    cfg = sample_data.stage_sample_run(tmp_path, refresh_inputs=update_goldens)
    metrics.run(cfg)
    return cfg


def test_collect_metrics_frames_preserves_stats(sample_config):
    iso_paths = sorted((sample_config.analysis_dir / "data").glob("*p/*_isolated_metrics.parquet"))
    frame = metrics._collect_metrics_frames(iso_paths)

    assert set(frame["n_players"]) == {2, 3}

    alpha_two = frame[
        (frame["strategy"] == sample_data.STRATEGIES[0]) & (frame["n_players"] == 2)
    ].iloc[0]
    assert alpha_two["wins"] == 12
    assert alpha_two["games"] == 20
    assert alpha_two["win_rate"] == pytest.approx(0.6)
    assert alpha_two["expected_score"] == pytest.approx(100.0)


def test_compute_seat_advantage_matches_manifest(sample_config):
    seat_df = metrics._compute_seat_advantage(sample_config, sample_config.curated_parquet)
    assert len(seat_df) == 12

    seat1 = seat_df[seat_df["seat"] == 1].iloc[0]
    assert seat1["wins"] == 2
    assert seat1["games_with_seat"] == 3
    assert seat1["win_rate"] == pytest.approx(2 / 3)
    assert seat1["win_rate_delta_seat1"] == pytest.approx(0.0)

    seat2 = seat_df[seat_df["seat"] == 2].iloc[0]
    assert seat2["win_rate_delta_prev"] == pytest.approx(seat2["win_rate"] - seat1["win_rate"])

    seat4 = seat_df[seat_df["seat"] == 4].iloc[0]
    assert seat4["games_with_seat"] == 0
    assert seat4["win_rate"] == pytest.approx(0.0)


def test_seat_metrics_include_per_seat_stats(sample_config):
    seat_metrics = sample_config.analysis_dir / "seat_metrics.parquet"
    df = pd.read_parquet(seat_metrics)

    row = df[
        (df["strategy"] == sample_data.STRATEGIES[0])
        & (df["seat"] == 1)
        & (df["n_players"] == 3)
    ].iloc[0]

    assert row["games"] == 2
    assert row["wins"] == 1
    assert row["win_rate"] == pytest.approx(0.5)
    assert row["mean_rounds"] == pytest.approx(10.5)


def test_symmetry_checks_empty_when_no_symmetric_pairs(sample_config):
    symmetry_path = sample_config.analysis_dir / "symmetry_checks.parquet"
    df = pd.read_parquet(symmetry_path)

    assert df.empty
