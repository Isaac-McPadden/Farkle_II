from __future__ import annotations

import pandas as pd
import pytest

from farkle.analysis import metrics
from tests.helpers import metrics_samples as sample_data


@pytest.fixture
def sample_config(tmp_path, update_goldens):
    cfg = sample_data.stage_sample_run(tmp_path, refresh_inputs=update_goldens)
    metrics.run(cfg)
    return cfg


def test_collect_metrics_frames_preserves_stats(sample_config):
    iso_paths = sorted((sample_config.analysis_dir / "data").glob("*p/*_isolated_metrics.parquet"))
    frame = metrics._collect_metrics_frames(iso_paths)

    assert set(frame["n_players"]) == {2, 3}

    alpha_two = frame[(frame["strategy"] == sample_data.STRATEGIES[0]) & (frame["n_players"] == 2)].iloc[0]
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

    seat4 = seat_df[seat_df["seat"] == 4].iloc[0]
    assert seat4["games_with_seat"] == 0
    assert seat4["win_rate"] == pytest.approx(0.0)
