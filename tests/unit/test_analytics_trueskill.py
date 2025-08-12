import os
import time
from pathlib import Path

from farkle.analysis_config import PipelineCfg
from farkle.analytics import trueskill


def _setup(tmp_path: Path) -> tuple[PipelineCfg, Path, Path]:
    cfg = PipelineCfg(results_dir=tmp_path)
    legacy = cfg.analysis_dir / "data" / cfg.curated_rows_name
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("data")
    tiers = cfg.results_dir / "tiers.json"
    tiers.write_text("{}")
    return cfg, legacy, tiers


def test_run_skips_when_tiers_up_to_date(tmp_path, monkeypatch):
    cfg, curated, tiers = _setup(tmp_path)
    now = time.time()
    os.utime(curated, (now, now))
    os.utime(tiers, (now + 10, now + 10))

    def boom(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("should not call run_trueskill.main")

    monkeypatch.setattr(trueskill._rt, "main", boom)

    trueskill.run(cfg)


def test_run_invokes_legacy_when_stale(tmp_path, monkeypatch):
    cfg, curated, tiers = _setup(tmp_path)
    now = time.time()
    os.utime(tiers, (now, now))
    os.utime(curated, (now + 10, now + 10))

    captured = {}

    def fake_main(args):  # noqa: ANN001
        captured["args"] = args

    monkeypatch.setattr(trueskill._rt, "main", fake_main)

    trueskill.run(cfg)

    expected = ["--dataroot", str(cfg.results_dir), "--root", str(cfg.results_dir)]
    assert captured["args"] == expected
