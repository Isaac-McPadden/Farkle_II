from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from pipeline import analyze_all


def _setup(tmp_path: Path) -> Path:
    exp = tmp_path
    analysis = exp / "analysis"
    analysis.mkdir()
    # minimal inputs
    (exp / "sim.txt").write_text("data")  # simulation placeholder
    metrics_df = pd.DataFrame(
        {
            "strategy": ["A", "B"],
            "n_players": [5, 5],
            "games": [10, 10],
            "wins": [6, 4],
            "win_rate": [0.6, 0.4],
            "expected_score": [0.0, 0.0],
        }
    )
    metrics_df.to_parquet(analysis / "metrics.parquet")
    ratings_df = pd.DataFrame({"strategy": ["A", "B"], "mu": [1.0, 0.5], "sigma": [1.0, 1.0]})
    ratings_df.to_parquet(analysis / "ratings_pooled.parquet")
    return exp


def test_analyze_all_skips_when_up_to_date(tmp_path, monkeypatch):
    exp = _setup(tmp_path)
    analysis = exp / "analysis"

    # stub tools to create outputs
    def fake_ts(cfg):  # noqa: ANN001
        (analysis / "tiers.json").write_text("{}")

    monkeypatch.setattr("farkle.analysis.run_trueskill.run_trueskill_all_seeds", fake_ts)

    def fake_h2h(*, root, n_jobs=1):  # noqa: ARG001
        df = pd.DataFrame({"a": ["A"], "b": ["B"], "wins_a": [1], "wins_b": [0], "pvalue": [0.5]})
        df.to_parquet(analysis / "bonferroni_pairwise.parquet")

    monkeypatch.setattr(
        "farkle.analysis.run_bonferroni_head2head.run_bonferroni_head2head",
        fake_h2h,
    )

    def fake_hgb(*, root, output_path, seed=0):  # noqa: ARG001
        # Write a tiny valid JSON file (the real code expects JSON here)
        output_path.write_text("{}")

    monkeypatch.setattr("farkle.analysis.run_hgb.run_hgb", fake_hgb, raising=True)

    # first run: executes stages
    buf = io.StringIO()
    with redirect_stdout(buf):
        analyze_all(exp)
    assert buf.getvalue().strip().splitlines() == ["trueskill", "h2h", "hgb", "agreement"]

    # second run: all stages skipped
    buf = io.StringIO()
    with redirect_stdout(buf):
        analyze_all(exp)
    assert buf.getvalue().strip().splitlines() == [
        "SKIP trueskill (up to date)",
        "SKIP h2h (up to date)",
        "SKIP hgb (up to date)",
        "SKIP agreement (up to date)",
    ]
