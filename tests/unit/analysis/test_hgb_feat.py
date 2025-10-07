import os
import os
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from farkle.analysis import hgb_feat
from farkle.config import AppConfig, IOConfig


def _setup_cfg(tmp_path: Path) -> tuple[AppConfig, Path]:
    cfg = AppConfig(io=IOConfig(results_dir=tmp_path, append_seed=False))
    analysis_dir = cfg.analysis_dir
    combined = cfg.curated_parquet.parent
    combined.mkdir(parents=True, exist_ok=True)
    curated = combined / "all_ingested_rows.parquet"
    curated.touch()
    return cfg, curated


def test_hgb_feat_skips_when_up_to_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, curated = _setup_cfg(tmp_path)
    out = cfg.analysis_dir / "hgb_importance.json"
    out.write_text("{}")
    # out newer than curated -> skip
    os.utime(curated, (1000, 1000))
    os.utime(out, (1010, 1010))

    def boom(**kwargs):  # pragma: no cover - should not be called
        raise AssertionError("_hgb.run_hgb should not be called when up-to-date")

    monkeypatch.setattr(hgb_feat._hgb, "run_hgb", boom)
    hgb_feat.run(cfg)


def test_hgb_feat_runs_when_outdated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, curated = _setup_cfg(tmp_path)
    out = cfg.analysis_dir / "hgb_importance.json"
    out.write_text("{}")
    # out older than curated -> run
    os.utime(out, (1000, 1000))
    os.utime(curated, (1010, 1010))

    called = {}

    def fake_run(*, root: Path, output_path: Path, seed: int = 0):
        assert root == cfg.analysis_dir
        assert output_path == out
        called["root"] = root

    monkeypatch.setattr(hgb_feat._hgb, "run_hgb", fake_run)
    hgb_feat.run(cfg)
    assert called
    assert not any(cfg.analysis_dir.glob("*.pkl"))
