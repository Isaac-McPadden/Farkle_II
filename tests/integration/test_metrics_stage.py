from __future__ import annotations

import pandas as pd
import pytest

from farkle.analysis import metrics
from tests.helpers.golden_utils import GoldenMismatchError
from tests.helpers.metrics_samples import stage_sample_run, validate_outputs


def test_metrics_run_matches_goldens(tmp_path, update_goldens):
    cfg = stage_sample_run(tmp_path, refresh_inputs=update_goldens)
    metrics.run(cfg)
    validate_outputs(cfg, update_goldens=update_goldens)


def test_golden_mismatch_requires_update_flag(tmp_path, update_goldens):
    cfg = stage_sample_run(tmp_path, refresh_inputs=False)
    metrics.run(cfg)

    if update_goldens:
        pytest.skip("Update mode refreshes goldens instead of enforcing mismatches.")

    metrics_path = cfg.analysis_dir / cfg.metrics_name
    df = pd.read_parquet(metrics_path).iloc[:-1]
    df.to_parquet(metrics_path, index=False)

    with pytest.raises(GoldenMismatchError):
        validate_outputs(cfg, update_goldens=False)
