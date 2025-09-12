import os
import pickle
from pathlib import Path

import pytest

from farkle.analysis_config import PipelineCfg
from farkle.analysis import hgb_feat


def _setup_cfg(tmp_path: Path) -> tuple[PipelineCfg, Path]:
    cfg = PipelineCfg(results_dir=tmp_path)
    analysis_dir = cfg.analysis_dir
    # ensure directories exist
    combined = analysis_dir / "data" / "all_n_players_combined"
    combined.mkdir(parents=True, exist_ok=True)
    curated = combined / "all_ingested_rows.parquet"
    curated.touch()
    return cfg, curated


def test_hgb_feat_skips_when_up_to_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, curated = _setup_cfg(tmp_path)
    out = cfg.results_dir / "hgb_importance.json"
    out.write_text("{}")
    # out newer than curated -> skip
    os.utime(curated, (1000, 1000))
    os.utime(out, (1010, 1010))

    def boom(_args):  # pragma: no cover - should not be called
        raise AssertionError("_hgb.main should not be called when up-to-date")

    monkeypatch.setattr(hgb_feat._hgb, "main", boom)
    hgb_feat.run(cfg)
    assert not (cfg.analysis_dir / "ratings_pooled.pkl").exists()


def test_hgb_feat_runs_when_outdated_and_copies_ratings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, curated = _setup_cfg(tmp_path)
    out = cfg.results_dir / "hgb_importance.json"
    out.write_text("{}")
    # out older than curated -> run
    os.utime(out, (1000, 1000))
    os.utime(curated, (1010, 1010))

    ratings = {"A": (1.0, 2.0)}
    with open(cfg.results_dir / "ratings_pooled.pkl", "wb") as fh:
        pickle.dump(ratings, fh)

    called = {}

    def fake_main(argv):
        ratings_src = cfg.results_dir / "ratings_pooled.pkl"
        ratings_dst = cfg.analysis_dir / "ratings_pooled.pkl"
        assert ratings_dst.exists()
        with open(ratings_src, "rb") as fh_src, open(ratings_dst, "rb") as fh_dst:
            assert fh_src.read() == fh_dst.read()
        assert argv == ["--root", str(cfg.analysis_dir), "--output", str(out)]
        called["args"] = argv

    monkeypatch.setattr(hgb_feat._hgb, "main", fake_main)
    hgb_feat.run(cfg)
    assert called
