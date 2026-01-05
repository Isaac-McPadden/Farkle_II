from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

import farkle.analysis.stage_registry as stage_registry
from farkle.config import AppConfig, IOConfig
from farkle.analysis.stage_registry import resolve_stage_layout
from pipeline import analyze_all, _done_path, is_up_to_date, write_done


def _setup(tmp_path: Path) -> tuple[Path, AppConfig]:
    cfg = AppConfig(io=IOConfig(results_dir=tmp_path, append_seed=False))
    cfg.analysis.run_frequentist = True
    cfg.analysis.run_agreement = True
    cfg.set_stage_layout(resolve_stage_layout(cfg))
    exp = cfg.results_dir
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
    metrics_path = cfg.metrics_output_path("metrics.parquet")
    metrics_df.to_parquet(metrics_path)
    ratings_df = pd.DataFrame({"strategy": ["A", "B"], "mu": [1.0, 0.5], "sigma": [1.0, 1.0]})
    ratings_path = cfg.trueskill_pooled_dir / "ratings_pooled.parquet"
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_parquet(ratings_path)
    tiers = cfg.preferred_tiers_path()
    tiers.parent.mkdir(parents=True, exist_ok=True)
    tiers.write_text("{}")
    return exp, cfg


def test_analyze_all_skips_when_up_to_date(tmp_path, monkeypatch):
    original_resolve = stage_registry.resolve_stage_layout

    def _patched_layout(app_cfg, *, registry=None):
        app_cfg.analysis.run_frequentist = True
        app_cfg.analysis.run_agreement = True
        return original_resolve(app_cfg, registry=registry)

    monkeypatch.setattr(stage_registry, "resolve_stage_layout", _patched_layout)

    exp, cfg = _setup(tmp_path)

    def fake_agreement(exp_dir: Path) -> None:
        app_cfg = AppConfig(io=IOConfig(results_dir=exp_dir, append_seed=False))
        app_cfg.analysis.run_agreement = True
        app_cfg.set_stage_layout(stage_registry.resolve_stage_layout(app_cfg))
        outputs = [app_cfg.agreement_output_path(p) for p in app_cfg.sim.n_players_list]
        done = _done_path(outputs[0])
        inputs = [app_cfg.trueskill_path("ratings_pooled.parquet")]
        if is_up_to_date(done, inputs, outputs):
            print("SKIP agreement (up to date)")
            return
        for out in outputs:
            out.write_text("{}")
        write_done(done, inputs, outputs, "farkle.analytics.agreement")
        print("agreement")

    monkeypatch.setattr("pipeline.analyze_agreement", fake_agreement)

    # stub tools to create outputs
    def fake_ts(app_cfg):  # noqa: ANN001
        tiers_out = app_cfg.trueskill_stage_dir / "tiers.json"
        tiers_out.write_text("{}")

    monkeypatch.setattr("farkle.analysis.run_trueskill.run_trueskill_all_seeds", fake_ts)

    def fake_h2h(cfg: AppConfig, *, n_jobs=1):  # noqa: ARG001
        df = pd.DataFrame({"a": ["A"], "b": ["B"], "wins_a": [1], "wins_b": [0], "pvalue": [0.5]})
        out = cfg.head2head_stage_dir / "bonferroni_pairwise.parquet"
        df.to_parquet(out)

    monkeypatch.setattr(
        "farkle.analysis.run_bonferroni_head2head.run_bonferroni_head2head",
        fake_h2h,
    )

    def fake_hgb(*, root, output_path, seed=0, metrics_path=None, ratings_path=None):  # noqa: ARG001
        # Write a tiny valid JSON file (the real code expects JSON here)
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
