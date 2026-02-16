from __future__ import annotations

# tests/integration/test_metrics_stage.py
"""Integration tests for the metrics stage against stored goldens."""

import pandas as pd
import pytest
from tests.helpers.golden_utils import GoldenMismatchError
from tests.helpers.metrics_samples import stage_sample_run, validate_outputs

from farkle.analysis import metrics


def test_metrics_run_matches_goldens(tmp_path, update_goldens):
    """Run the metrics stage and compare outputs to goldens.

    Golden refresh should be triggered only when deterministic metrics behavior
    changes (schema/columns, output-path wiring, or computation logic) and then
    executed with ``--update-goldens`` to rewrite checked-in expectations.

    Args:
        tmp_path: Temporary directory for staging artifacts.
        update_goldens: Flag controlling golden regeneration.

    Returns:
        None
    """

    cfg = stage_sample_run(tmp_path, refresh_inputs=update_goldens)
    metrics.run(cfg)
    validate_outputs(cfg, update_goldens=update_goldens)

    df = pd.read_parquet(cfg.metrics_output_path())
    assert list(df.columns) == [
        "strategy",
        "n_players",
        "games",
        "wins",
        "win_rate",
        "se_win_rate",
        "win_rate_ci_lo",
        "win_rate_ci_hi",
        "win_prob",
        "expected_score",
        "total_games_strat",
        "sum_winning_score",
        "sq_sum_winning_score",
        "sum_n_rounds",
        "sq_sum_n_rounds",
        "false_wins_handled",
        "missing_before_pad",
    ]
    assert len(df) == 8


def test_golden_mismatch_requires_update_flag(tmp_path, update_goldens):
    """Ensure mismatched outputs raise without the update flag set.

    This test documents the explicit opt-in policy: when generated outputs drift
    from stored goldens, developers must pass ``--update-goldens`` to refresh
    fixtures; default runs should fail loudly instead of silently rewriting data.

    Args:
        tmp_path: Temporary directory for staging artifacts.
        update_goldens: Flag indicating whether goldens should be refreshed.

    Returns:
        None
    """

    cfg = stage_sample_run(tmp_path, refresh_inputs=False)
    metrics.run(cfg)

    if update_goldens:
        pytest.skip("Update mode refreshes goldens instead of enforcing mismatches.")

    metrics_path = cfg.metrics_output_path()
    df = pd.read_parquet(metrics_path).iloc[:-1]
    df.to_parquet(metrics_path, index=False)

    with pytest.raises(GoldenMismatchError, match="--update-goldens"):
        validate_outputs(cfg, update_goldens=False)
