from __future__ import annotations

from pathlib import Path
import io
from contextlib import redirect_stdout

import pandas as pd

from pipeline import analyze_all


def _setup(tmp_path: Path) -> Path:
    exp = tmp_path
    analysis = exp / "analysis"
    analysis.mkdir()
    # minimal inputs
    (exp / "sim.txt").write_text("data")  # simulation placeholder
    (analysis / "metrics.parquet").write_text("m")
    (analysis / "ratings_pooled.parquet").write_text("r")
    return exp


def test_analyze_all_skips_when_up_to_date(tmp_path, monkeypatch):
    exp = _setup(tmp_path)
    analysis = exp / "analysis"

    # stub tools to create outputs
    def fake_ts(*, root=None, dataroot=None, **_: object):  # noqa: ANN001, ARG001
        (analysis / "tiers.json").write_text("{}")
    monkeypatch.setattr("farkle.analysis.run_trueskill.run_trueskill", fake_ts)

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
    assert buf.getvalue().strip().splitlines() == ["trueskill", "h2h", "hgb"]

    # second run: all stages skipped
    buf = io.StringIO()
    with redirect_stdout(buf):
        analyze_all(exp)
    assert buf.getvalue().strip().splitlines() == [
        "SKIP trueskill (up to date)",
        "SKIP h2h (up to date)",
        "SKIP hgb (up to date)",
    ]
