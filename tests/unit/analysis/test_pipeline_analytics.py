from __future__ import annotations

from pathlib import Path
import io
from contextlib import redirect_stdout
import types
import sys

import pipeline


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
    def fake_ts(args):
        (analysis / "tiers.json").write_text("{}")
    monkeypatch.setattr("farkle.analysis.run_trueskill.main", fake_ts)

    def fake_h2h(*, root, n_jobs=1):  # noqa: ARG001
        (analysis / "bonferroni_pairwise.csv").write_text("a,b")
    monkeypatch.setattr(
        "farkle.analysis.run_bonferroni_head2head.run_bonferroni_head2head",
        fake_h2h,
    )

    def fake_hgb(*, root, output_path, seed=0):  # noqa: ARG001
        output_path.write_text("{}")
    monkeypatch.setitem(
        sys.modules,
        "farkle.analysis.run_hgb",
        types.SimpleNamespace(run_hgb=fake_hgb),
    )

    # first run: executes stages
    buf = io.StringIO()
    with redirect_stdout(buf):
        pipeline.analyze_all(exp)
    assert buf.getvalue().strip().splitlines() == ["trueskill", "h2h", "hgb"]

    # second run: all stages skipped
    buf = io.StringIO()
    with redirect_stdout(buf):
        pipeline.analyze_all(exp)
    assert buf.getvalue().strip().splitlines() == [
        "SKIP trueskill (up to date)",
        "SKIP h2h (up to date)",
        "SKIP hgb (up to date)",
    ]
