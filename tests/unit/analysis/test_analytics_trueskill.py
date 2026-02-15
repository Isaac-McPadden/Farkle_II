import os
import time
from pathlib import Path

from farkle.analysis import trueskill
from farkle.analysis.stage_registry import resolve_stage_layout
from farkle.config import AppConfig, IOConfig


def _setup(tmp_path: Path) -> tuple[AppConfig, Path, Path]:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    cfg.analysis.run_frequentist = True
    cfg.set_stage_layout(resolve_stage_layout(cfg))
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


def test_interseed_run_uses_upstream_combine_and_writes_pooled_outputs(tmp_path, monkeypatch):
    from farkle.orchestration.run_contexts import InterseedRunContext, SeedRunContext

    seed_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "seed_results"))
    seed_cfg.set_stage_layout(resolve_stage_layout(seed_cfg))
    seed_context = SeedRunContext.from_config(seed_cfg)

    combine_folder = seed_cfg.stage_layout.require_folder("combine")
    upstream_curated = (
        seed_context.analysis_root / combine_folder / "pooled" / "all_ingested_rows.parquet"
    )
    upstream_curated.parent.mkdir(parents=True, exist_ok=True)
    upstream_curated.write_text("rows")

    interseed = InterseedRunContext.from_seed_context(
        seed_context,
        seed_pair=(101, 202),
        analysis_root=tmp_path / "pair" / "interseed_analysis",
    )
    cfg = interseed.config

    def fake_run(app_cfg):  # noqa: ANN001
        assert app_cfg.curated_parquet == upstream_curated
        pooled_dir = app_cfg.trueskill_pooled_dir
        pooled_dir.mkdir(parents=True, exist_ok=True)
        (pooled_dir / "ratings_long.parquet").write_text("long")
        (pooled_dir / "ratings_k_weighted.parquet").write_text("pooled")
        (app_cfg.trueskill_stage_dir / "tiers.json").write_text("{}")

    monkeypatch.setattr(trueskill.run_trueskill, "run_trueskill_all_seeds", fake_run)

    trueskill.run(cfg)

    pooled_dir = cfg.analysis_dir / "03_trueskill" / "pooled"
    assert (pooled_dir / "ratings_long.parquet").exists()
    assert (pooled_dir / "ratings_k_weighted.parquet").exists()


def test_run_logs_curated_candidates_when_missing(tmp_path, caplog):
    cfg = AppConfig(
        io=IOConfig(
            results_dir_prefix=tmp_path / "pair_results",
            interseed_input_dir=tmp_path / "upstream",
            interseed_input_layout={"combine": "02_combine"},
        )
    )

    caplog.set_level("INFO")
    trueskill.run(cfg)

    record = next(record for record in caplog.records if getattr(record, "reason", None))
    assert record.reason == "curated parquet missing"
    assert any(path.endswith("all_ingested_rows.parquet") for path in record.candidate_paths)
    assert record.interseed_input_root == str(tmp_path / "upstream")
