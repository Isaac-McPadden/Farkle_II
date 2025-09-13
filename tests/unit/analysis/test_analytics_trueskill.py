import os
import time
from pathlib import Path

from farkle.analysis.analysis_config import PipelineCfg
from farkle.analysis import trueskill


def _setup(tmp_path: Path) -> tuple[PipelineCfg, Path, Path]:
    cfg = PipelineCfg(results_dir=tmp_path)
    legacy = cfg.analysis_dir / "data" / cfg.curated_rows_name
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("data")
    tiers = cfg.analysis_dir / "tiers.json"
    tiers.write_text("{}")
    return cfg, legacy, tiers


def test_run_skips_when_tiers_up_to_date(tmp_path, monkeypatch):
    cfg, curated, tiers = _setup(tmp_path)
    now = time.time()
    os.utime(curated, (now, now))
    os.utime(tiers, (now + 10, now + 10))

    def boom(**kwargs):  # noqa: ARG001
        raise AssertionError("should not call run_trueskill.run_trueskill")

    monkeypatch.setattr(trueskill.run_trueskill, "run_trueskill", boom)

    trueskill.run(cfg)


def test_run_invokes_legacy_when_stale(tmp_path, monkeypatch):
    cfg, curated, tiers = _setup(tmp_path)
    now = time.time()
    os.utime(tiers, (now, now))
    os.utime(curated, (now + 10, now + 10))

    called = {}

    def fake_run(root: Path, dataroot: Path | None = None):  # noqa: ANN001
        called["root"] = root
        called["dataroot"] = dataroot

    monkeypatch.setattr(trueskill.run_trueskill, "run_trueskill", fake_run)

    trueskill.run(cfg)

    assert called["root"] == cfg.analysis_dir
