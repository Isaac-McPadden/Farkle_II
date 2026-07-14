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
    contribution = cfg.trueskill_candidate_contribution_path()
    contribution.parent.mkdir(parents=True, exist_ok=True)
    contribution.write_text("screening")
    return cfg, combined, contribution


def test_run_skips_when_screening_contribution_is_up_to_date(tmp_path, monkeypatch):
    cfg, combined, contribution = _setup(tmp_path)
    now = time.time()
    os.utime(combined, (now, now))
    os.utime(contribution, (now + 10, now + 10))

    def boom(cfg):  # noqa: ARG001
        raise AssertionError("should not call run_trueskill.run_trueskill_all_seeds")

    monkeypatch.setattr(trueskill.run_trueskill, "run_trueskill_all_seeds", boom)

    trueskill.run(cfg)


def test_run_invokes_rating_builder_when_stale(tmp_path, monkeypatch):
    cfg, combined, contribution = _setup(tmp_path)
    now = time.time()
    os.utime(contribution, (now, now))
    os.utime(combined, (now + 10, now + 10))

    called = {}

    def fake_run(app_cfg):  # noqa: ANN001
        called["cfg"] = app_cfg

    monkeypatch.setattr(trueskill.run_trueskill, "run_trueskill_all_seeds", fake_run)

    trueskill.run(cfg)

    assert called["cfg"] is cfg


def test_interseed_run_uses_upstream_rows_and_writes_screening_output(tmp_path, monkeypatch):
    from farkle.orchestration.run_contexts import InterseedRunContext, SeedRunContext

    seed_cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "seed_results"))
    seed_cfg.set_stage_layout(resolve_stage_layout(seed_cfg))
    seed_context = SeedRunContext.from_config(seed_cfg)

    combine_folder = seed_cfg.stage_layout.require_folder("combine")
    upstream_curated = (
        seed_context.analysis_root / combine_folder / "concat_ks" / "all_ingested_rows.parquet"
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
        (app_cfg.concat_ks_dir("trueskill") / "ratings_concat_ks.parquet").write_text("concat")
        contribution = app_cfg.trueskill_candidate_contribution_path()
        contribution.parent.mkdir(parents=True, exist_ok=True)
        contribution.write_text("screening")

    monkeypatch.setattr(trueskill.run_trueskill, "run_trueskill_all_seeds", fake_run)

    trueskill.run(cfg)

    assert (cfg.concat_ks_dir("trueskill") / "ratings_concat_ks.parquet").exists()
    assert cfg.trueskill_candidate_contribution_path().exists()


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
