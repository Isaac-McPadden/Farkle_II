from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from farkle.analysis import seed_symmetry
from farkle.analysis.stage_state import stage_done_path
from farkle.config import AppConfig, IOConfig, SimConfig


def _make_cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2]))


def test_seed_symmetry_run_writes_outputs_and_done(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    source = cfg.head2head_path("bonferroni_selfplay_symmetry.parquet")
    source.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "players": [2, 2],
            "seed": [7, 8],
            "strategy": ["A", "A"],
            "games": [20, 20],
            "seat1_win_rate": [0.5, 0.45],
            "seat2_win_rate": [0.5, 0.55],
            "seat_win_rate_diff": [0.0, -0.1],
            "mean_farkles_seat1": [1.0, 1.2],
            "mean_farkles_seat2": [1.0, 1.1],
            "mean_score_seat1": [100.0, 95.0],
            "mean_score_seat2": [100.0, 96.0],
        }
    ).to_parquet(source, index=False)

    seed_symmetry.run(cfg)

    out_seed = cfg.seed_symmetry_stage_dir / "seed_symmetry_checks.parquet"
    out_summary = cfg.seed_symmetry_stage_dir / "seed_symmetry_summary.parquet"
    assert out_seed.exists()
    assert out_summary.exists()

    done = stage_done_path(cfg.seed_symmetry_stage_dir, "seed_symmetry")
    payload = json.loads(done.read_text())
    assert payload["status"] == "success"
    assert payload["inputs"] == [str(source)]
    assert payload["outputs"] == [
        str(cfg.seed_symmetry_stage_dir / "seed_symmetry_checks.parquet"),
        str(cfg.seed_symmetry_stage_dir / "seed_symmetry_checks.csv"),
        str(cfg.seed_symmetry_stage_dir / "seed_symmetry_summary.parquet"),
        str(cfg.seed_symmetry_stage_dir / "seed_symmetry_summary.csv"),
    ]


def test_seed_symmetry_missing_input_marks_stage_skipped(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)

    seed_symmetry.run(cfg)

    done = stage_done_path(cfg.seed_symmetry_stage_dir, "seed_symmetry")
    payload = json.loads(done.read_text())
    assert payload["status"] == "skipped"
    assert payload["outputs"] == []


def test_seed_symmetry_skips_when_done_stamp_is_up_to_date(tmp_path: Path, monkeypatch) -> None:
    cfg = _make_cfg(tmp_path)
    source = cfg.head2head_path("bonferroni_selfplay_symmetry.parquet")
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("placeholder")

    done = stage_done_path(cfg.seed_symmetry_stage_dir, "seed_symmetry")
    monkeypatch.setattr(seed_symmetry, "stage_is_up_to_date", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        seed_symmetry.pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not recompute")),
    )

    seed_symmetry.run(cfg)

    assert not done.exists()
