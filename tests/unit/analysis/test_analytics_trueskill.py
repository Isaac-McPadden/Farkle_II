import os
import time
from pathlib import Path

from farkle.analysis import trueskill
from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig, IOConfig


def _setup(tmp_path: Path) -> tuple[AppConfig, Path, Path]:
    cfg = AppConfig(io=IOConfig(results_dir=tmp_path, append_seed=False))
    cfg.analysis.run_frequentist = True
    cfg.set_stage_layout(resolve_stage_layout(cfg, run_rng=True))
    combined = cfg.curated_parquet
    combined.parent.mkdir(parents=True, exist_ok=True)
    combined.write_text("data")
    tiers = cfg.preferred_tiers_path()
    tiers.parent.mkdir(parents=True, exist_ok=True)
    tiers.write_text("{}")
    return cfg, combined, tiers


def test_run_skips_when_tiers_up_to_date(tmp_path, monkeypatch):
    cfg, combined, tiers = _setup(tmp_path)
    now = time.time()
    os.utime(combined, (now, now))
    os.utime(tiers, (now + 10, now + 10))

    def boom(cfg):  # noqa: ARG001
        raise AssertionError("should not call run_trueskill.run_trueskill_all_seeds")

    monkeypatch.setattr(trueskill.run_trueskill, "run_trueskill_all_seeds", boom)

    trueskill.run(cfg)


def test_run_invokes_legacy_when_stale(tmp_path, monkeypatch):
    cfg, combined, tiers = _setup(tmp_path)
    now = time.time()
    os.utime(tiers, (now, now))
    os.utime(combined, (now + 10, now + 10))

    called = {}

    def fake_run(app_cfg):  # noqa: ANN001
        called["cfg"] = app_cfg

    monkeypatch.setattr(trueskill.run_trueskill, "run_trueskill_all_seeds", fake_run)

    trueskill.run(cfg)

    assert called["cfg"] is cfg
