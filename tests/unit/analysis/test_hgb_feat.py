import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("sklearn")

from farkle.analysis import hgb_feat
from farkle.config import AppConfig, IOConfig


def _setup_cfg(tmp_path: Path) -> tuple[AppConfig, Path]:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    cfg.sim.n_players_list = [2]
    analysis_dir = cfg.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.metrics_output_path()
    ratings_path = analysis_dir / hgb_feat._hgb.RATINGS_NAME
    pd.DataFrame({"strategy": ["Strat(300,2)[SD][FOFS][AND][H-]"], "n_players": [2]}).to_parquet(
        metrics_path, index=False
    )
    pd.DataFrame({"strategy": ["Strat(300,2)[SD][FOFS][AND][H-]"], "mu": [0.0]}).to_parquet(
        ratings_path, index=False
    )
    os.utime(metrics_path, (1000, 1000))
    os.utime(ratings_path, (1000, 1000))
    combined = cfg.curated_parquet.parent
    combined.mkdir(parents=True, exist_ok=True)
    curated = combined / "all_ingested_rows.parquet"
    curated.touch()
    return cfg, curated


def test_hgb_feat_skips_when_up_to_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, curated = _setup_cfg(tmp_path)
    json_out = cfg.hgb_pooled_dir / "hgb_importance.json"
    parquet_out = cfg.hgb_per_k_dir(2) / hgb_feat._hgb.IMPORTANCE_TEMPLATE.format(players=2)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text("{}")
    pd.DataFrame({"feature": [], "importance_mean": [], "importance_std": []}).to_parquet(
        parquet_out, index=False
    )
    os.utime(curated, (1000, 1000))
    os.utime(json_out, (1010, 1010))
    os.utime(parquet_out, (1010, 1010))

    def boom(**kwargs):  # pragma: no cover - should not be called
        raise AssertionError("_hgb.run_hgb should not be called when up-to-date")

    monkeypatch.setattr(hgb_feat._hgb, "run_hgb", boom)
    hgb_feat.run(cfg)


def test_hgb_feat_runs_when_outdated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, curated = _setup_cfg(tmp_path)
    json_out = cfg.hgb_pooled_dir / "hgb_importance.json"
    parquet_out = cfg.hgb_per_k_dir(2) / hgb_feat._hgb.IMPORTANCE_TEMPLATE.format(players=2)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text("{}")
    pd.DataFrame({"feature": [], "importance_mean": [], "importance_std": []}).to_parquet(
        parquet_out, index=False
    )
    os.utime(json_out, (1000, 1000))
    os.utime(parquet_out, (1000, 1000))
    os.utime(curated, (1020, 1020))

    called = {}

    def fake_run(
        *,
        root: Path,
        output_path: Path,
        metrics_path: Path,
        ratings_path: Path,
        manifest_path: Path | None,
    ) -> None:
        assert root == cfg.hgb_stage_dir
        assert output_path == json_out
        assert metrics_path == cfg.metrics_input_path()
        assert ratings_path == cfg.trueskill_path(hgb_feat._hgb.RATINGS_NAME)
        assert manifest_path == cfg.strategy_manifest_root_path()
        called["root"] = root

    monkeypatch.setattr(hgb_feat._hgb, "run_hgb", fake_run)
    hgb_feat.run(cfg)
    assert called
    assert not any(cfg.analysis_dir.glob("*.pkl"))


def test_hgb_feat_returns_when_metrics_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    cfg.sim.n_players_list = [2]
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)

    called = False

    def boom(**kwargs):  # pragma: no cover - should not be called
        nonlocal called
        called = True
        raise AssertionError("_hgb.run_hgb should not be called when metrics are missing")

    monkeypatch.setattr(hgb_feat._hgb, "run_hgb", boom)

    hgb_feat.run(cfg)

    assert called is False


def test_hgb_feat_returns_when_ratings_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    cfg.sim.n_players_list = [2]
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_input_path().parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"strategy": ["S"], "n_players": [2]}).to_parquet(cfg.metrics_input_path(), index=False)

    called = False

    def boom(**kwargs):  # pragma: no cover - should not be called
        nonlocal called
        called = True
        raise AssertionError("_hgb.run_hgb should not be called when ratings are missing")

    monkeypatch.setattr(hgb_feat._hgb, "run_hgb", boom)

    hgb_feat.run(cfg)

    assert called is False


def test_unique_players_uses_players_fallback_column(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.parquet"
    pd.DataFrame({"players": [4, 2, 4, 5]}).to_parquet(metrics_path, index=False)

    players = hgb_feat._unique_players(metrics_path, hints=[3, 2, 3])

    assert players == [2, 3, 4, 5]


def test_unique_players_gracefully_handles_read_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics_path = tmp_path / "metrics.parquet"
    metrics_path.touch()

    def raise_read_error(path: Path, columns: list[str]):
        if columns == ["n_players"]:
            raise pa.ArrowInvalid("missing n_players")
        raise KeyError("missing players")

    monkeypatch.setattr(hgb_feat.pq, "read_table", raise_read_error)

    players = hgb_feat._unique_players(metrics_path, hints=[6, 2, 6])

    assert players == [2, 6]


def test_latest_mtime_returns_zero_for_missing_paths(tmp_path: Path) -> None:
    paths = [tmp_path / "does_not_exist_a.parquet", tmp_path / "does_not_exist_b.parquet"]

    assert hgb_feat._latest_mtime(paths) == 0.0
