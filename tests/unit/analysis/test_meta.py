from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import pandas as pd
import pytest

from farkle.analysis import meta
from farkle.config import AppConfig
from farkle.simulation.simulation import generate_strategy_grid


def _make_cfg(tmp_path: Path) -> AppConfig:
    cfg = AppConfig()
    cfg.sim.seed = 42
    cfg.io.results_dir_prefix = tmp_path / "results_meta"
    cfg.results_root.mkdir(parents=True, exist_ok=True)
    cfg.analysis.outputs = {}
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_estimate_rate_and_variance_validates_inputs():
    rate, var = meta._estimate_rate_and_variance(0, 0, None)
    assert math.isnan(rate)
    assert var == math.inf

    with pytest.raises(ValueError):
        meta._estimate_rate_and_variance(5, 2, None)

    rate, var = meta._estimate_rate_and_variance(10, 10, 1.0)
    assert 0.0 <= rate <= 1.0
    assert var >= meta.MIN_VARIANCE


@pytest.mark.parametrize(
    ("prob", "expected_sign"),
    [
        (0.0, -1),
        (1.0, 1),
    ],
)
def test_logit_clips_probability_boundaries(prob: float, expected_sign: int) -> None:
    value = meta._logit(prob)
    assert math.isfinite(value)
    assert math.copysign(1.0, value) == expected_sign


def test_inv_logit_handles_large_positive_and_negative_logits() -> None:
    assert meta._inv_logit(1000.0) == pytest.approx(1.0)
    assert meta._inv_logit(-1000.0) == pytest.approx(0.0)
    assert meta._inv_logit(0.0) == pytest.approx(0.5)


def test_wilson_logit_center_is_bounded_for_extreme_rates() -> None:
    assert 0.0 < meta._wilson_logit_center(0, 50) < 0.5
    assert 0.5 < meta._wilson_logit_center(50, 50) < 1.0


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("strategy_summary_4p_seed12.parquet", (4, 12)),
        ("strategy_summary_4p_seed.parquet", None),
        ("strategy_summary_4_seed12.parquet", None),
        ("summary_4p_seed12.parquet", None),
    ],
)
def test_parse_seed_file_valid_and_invalid_names(name: str, expected: tuple[int, int] | None) -> None:
    assert meta._parse_seed_file(Path(name)) == expected


def test_pool_winrates_prefers_fixed_effects_when_I2_low() -> None:
    df_seed1 = pd.DataFrame(
        [
            {
                "strategy_id": "S1",
                "players": 2,
                "seed": 1,
                "games": 50,
                "wins": 25,
                "win_rate": 0.50,
            },
            {
                "strategy_id": "S2",
                "players": 2,
                "seed": 1,
                "games": 50,
                "wins": 20,
                "win_rate": 0.40,
            },
        ]
    )
    df_seed2 = pd.DataFrame(
        [
            {
                "strategy_id": "S1",
                "players": 2,
                "seed": 2,
                "games": 50,
                "wins": 26,
                "win_rate": 0.52,
            },
            {
                "strategy_id": "S2",
                "players": 2,
                "seed": 2,
                "games": 50,
                "wins": 18,
                "win_rate": 0.36,
            },
        ]
    )

    result = meta.pool_winrates([df_seed1, df_seed2], use_random_if_I2_gt=80.0)
    assert result.method == "fixed"
    assert result.pooled.shape[0] == 2

    s1 = result.pooled.loc[result.pooled["strategy_id"] == "S1"].iloc[0]
    weight_a = 1.0 / (0.5 * 0.5 / 50.0)
    weight_b = 1.0 / (0.52 * 0.48 / 50.0)
    expected = (weight_a * 0.50 + weight_b * 0.52) / (weight_a + weight_b)
    assert s1["win_rate"] == pytest.approx(expected)
    expected_se = math.sqrt(1.0 / (weight_a + weight_b))
    assert s1["se"] == pytest.approx(expected_se)
    assert pytest.approx(0.0) == result.I2


def test_pool_winrates_returns_empty_for_none_or_empty_inputs() -> None:
    optional_inputs: list[pd.DataFrame | None] = [None, pd.DataFrame()]
    usable_inputs = [df for df in optional_inputs if df is not None]
    assert len(usable_inputs) == 1

    result = meta.pool_winrates(usable_inputs)

    assert result.pooled.empty
    assert result.pooled.columns.tolist() == meta.POOLED_COLUMNS
    assert result.method == "fixed"
    assert result.tau2 == pytest.approx(0.0)


def test_pool_winrates_raises_on_players_mismatch() -> None:
    df_a = pd.DataFrame(
        [{"strategy_id": "A", "players": 2, "seed": 1, "games": 10, "wins": 5, "win_rate": 0.5}]
    )
    df_b = pd.DataFrame(
        [{"strategy_id": "A", "players": 3, "seed": 2, "games": 10, "wins": 6, "win_rate": 0.6}]
    )

    with pytest.raises(ValueError, match="single player count"):
        meta.pool_winrates([df_a, df_b])


def test_pool_winrates_raises_on_strategy_set_mismatch() -> None:
    df_a = pd.DataFrame(
        [{"strategy_id": "A", "players": 2, "seed": 1, "games": 10, "wins": 5, "win_rate": 0.5}]
    )
    df_b = pd.DataFrame(
        [{"strategy_id": "B", "players": 2, "seed": 2, "games": 10, "wins": 6, "win_rate": 0.6}]
    )

    with pytest.raises(ValueError, match="Strategy presence mismatch"):
        meta.pool_winrates([df_a, df_b])


def test_pool_winrates_uses_random_effects_for_high_heterogeneity() -> None:
    df_seed1 = pd.DataFrame(
        [{"strategy_id": "S", "players": 2, "seed": 1, "games": 100, "wins": 1, "win_rate": 0.01}]
    )
    df_seed2 = pd.DataFrame(
        [{"strategy_id": "S", "players": 2, "seed": 2, "games": 100, "wins": 99, "win_rate": 0.99}]
    )

    result = meta.pool_winrates([df_seed1, df_seed2], use_random_if_I2_gt=0.0)

    assert result.method == "random"
    assert result.tau2 > 0.0


def test_pool_winrates_skips_strategies_without_usable_observations() -> None:
    df_seed1 = pd.DataFrame(
        [{"strategy_id": "S", "players": 2, "seed": 1, "games": 0, "wins": 0, "win_rate": 0.0}]
    )
    df_seed2 = pd.DataFrame(
        [{"strategy_id": "S", "players": 2, "seed": 2, "games": 0, "wins": 0, "win_rate": 0.0}]
    )

    result = meta.pool_winrates([df_seed1, df_seed2])

    assert result.pooled.empty
    assert result.method == "fixed"


def test_select_seed_entries_comparison_mode() -> None:
    entries = [(42, Path("a")), (7, Path("b")), (99, Path("c"))]

    selected_present = meta._select_seed_entries(
        entries,
        42,
        max_other_seeds=1,
        comparison_seed=7,
    )
    selected_missing_primary = meta._select_seed_entries(
        entries,
        100,
        max_other_seeds=1,
        comparison_seed=7,
    )
    selected_missing_comp = meta._select_seed_entries(
        entries,
        42,
        max_other_seeds=1,
        comparison_seed=101,
    )

    assert selected_present == [(7, Path("b")), (42, Path("a"))]
    assert selected_missing_primary == [(7, Path("b"))]
    assert selected_missing_comp == [(42, Path("a"))]


def test_select_seed_entries_limits_other_seeds_deterministically() -> None:
    entries = [(42, Path("a")), (1, Path("1")), (2, Path("2")), (3, Path("3")), (4, Path("4"))]

    selected_once = meta._select_seed_entries(
        entries,
        primary_seed=42,
        max_other_seeds=2,
        comparison_seed=None,
    )
    selected_twice = meta._select_seed_entries(
        entries,
        primary_seed=42,
        max_other_seeds=2,
        comparison_seed=None,
    )

    assert selected_once == selected_twice
    assert len(selected_once) == 3
    assert (42, Path("a")) in selected_once


def test_select_seed_entries_with_no_limit_includes_all_remaining() -> None:
    entries = [(42, Path("a")), (7, Path("b")), (99, Path("c"))]

    selected = meta._select_seed_entries(
        entries,
        primary_seed=42,
        max_other_seeds=None,
        comparison_seed=None,
    )

    assert selected == sorted(entries)


def test_meta_run_writes_pooled_outputs(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)

    df_seed1 = pd.DataFrame(
        [
            {
                "strategy_id": "Keep",
                "players": 2,
                "seed": 1,
                "games": 30,
                "wins": 18,
                "win_rate": 0.60,
            },
            {
                "strategy_id": "Drop",
                "players": 2,
                "seed": 1,
                "games": 30,
                "wins": 10,
                "win_rate": 0.33,
            },
        ]
    )
    df_seed2 = pd.DataFrame(
        [
            {
                "strategy_id": "Keep",
                "players": 2,
                "seed": 2,
                "games": 40,
                "wins": 20,
                "win_rate": 0.50,
            },
            # "Drop" missing here to trigger the presence rule.
        ]
    )
    df_seed1.to_parquet(cfg.analysis_dir / "strategy_summary_2p_seed1.parquet", index=False)
    df_seed2.to_parquet(cfg.analysis_dir / "strategy_summary_2p_seed2.parquet", index=False)

    meta.run(cfg, use_random_if_I2_gt=90.0)

    parquet_name = "strategy_summary_2p_meta.parquet"
    json_name = "meta_2p.json"
    parquet_path = cfg.meta_output_path(2, parquet_name)
    json_path = cfg.meta_output_path(2, json_name)
    assert parquet_path.exists()
    assert json_path.exists()

    pooled = pd.read_parquet(cfg.meta_input_path(2, parquet_name))
    assert pooled.columns.tolist() == meta.POOLED_COLUMNS
    assert pooled["strategy_id"].tolist() == ["Keep"]

    stats = json.loads(cfg.meta_input_path(2, json_name).read_text())
    assert stats["method"] == "fixed"
    assert stats["I2"] <= 90.0
    assert math.isfinite(stats["Q"])
    assert math.isfinite(stats["tau2"])


def test_meta_skips_when_single_seed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg = _make_cfg(tmp_path)
    caplog.set_level(logging.INFO)

    df_seed1 = pd.DataFrame(
        [
            {
                "strategy_id": "Solo",
                "players": 2,
                "seed": 1,
                "games": 20,
                "wins": 10,
                "win_rate": 0.50,
            }
        ]
    )
    df_seed1.to_parquet(cfg.analysis_dir / "strategy_summary_2p_seed1.parquet", index=False)

    meta.run(cfg, use_random_if_I2_gt=90.0)

    parquet_path = cfg.meta_output_path(2, "strategy_summary_2p_meta.parquet")
    assert not parquet_path.exists()
    assert any("requires multiple seeds" in rec.message for rec in caplog.records)


def test_meta_limits_other_seeds_and_respects_override(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.analysis.meta_max_other_seeds = 1

    frames_by_seed: dict[int, pd.DataFrame] = {}

    def _write(seed: int, wins: int) -> None:
        df = pd.DataFrame(
            [
                {
                    "strategy_id": "Only",
                    "players": 2,
                    "seed": seed,
                    "games": 10,
                    "wins": wins,
                    "win_rate": wins / 10.0,
                }
            ]
        )
        frames_by_seed[seed] = df
        df.to_parquet(cfg.analysis_dir / f"strategy_summary_2p_seed{seed}.parquet", index=False)

    _write(42, 6)
    _write(7, 5)
    _write(99, 2)

    meta.run(cfg, use_random_if_I2_gt=90.0)

    pooled = pd.read_parquet(cfg.meta_input_path(2, "strategy_summary_2p_meta.parquet"))
    assert pooled["n_seeds"].iloc[0] == 2
    expected_win_rate = meta.pool_winrates(
        [frames_by_seed[42], frames_by_seed[99]], use_random_if_I2_gt=90.0
    ).pooled["win_rate"].iloc[0]
    assert pooled["win_rate"].iloc[0] == pytest.approx(expected_win_rate)

    cfg.analysis.meta_comparison_seed = 7
    meta.run(cfg, force=True, use_random_if_I2_gt=90.0)

    pooled_override = pd.read_parquet(cfg.meta_input_path(2, "strategy_summary_2p_meta.parquet"))
    assert pooled_override["n_seeds"].iloc[0] == 2
    expected_override = meta.pool_winrates(
        [frames_by_seed[42], frames_by_seed[7]], use_random_if_I2_gt=90.0
    ).pooled["win_rate"].iloc[0]
    assert pooled_override["win_rate"].iloc[0] == pytest.approx(expected_override)


def test_apply_strategy_presence_filters_and_reports_missing():
    frames = [
        pd.DataFrame({"strategy_id": ["A", "B"], "seed": [1, 1]}),
        pd.DataFrame({"strategy_id": ["A", "C"], "seed": [2, 2]}),
    ]

    filtered, missing = meta._apply_strategy_presence(frames)

    assert all(df["strategy_id"].tolist() == ["A"] for df in filtered)
    assert missing == {"B": [2], "C": [1]}


def test_meta_strategy_intersection_non_empty_for_shared_grid() -> None:
    _, meta_frame = generate_strategy_grid(
        score_thresholds=[200],
        dice_thresholds=[0],
        smart_five_opts=[False],
        smart_one_opts=[False],
        consider_score_opts=[True],
        consider_dice_opts=[True],
        auto_hot_dice_opts=[False],
        run_up_score_opts=[False],
    )
    strategy_ids = meta_frame["strategy_id"].tolist()
    frames = [
        pd.DataFrame({"strategy_id": strategy_ids, "seed": [1] * len(strategy_ids)}),
        pd.DataFrame({"strategy_id": strategy_ids, "seed": [2] * len(strategy_ids)}),
    ]

    filtered, missing = meta._apply_strategy_presence(frames)

    assert missing == {}
    assert filtered
    assert filtered[0]["strategy_id"].tolist()


def test_collect_seed_summaries_prefers_stage_then_meta_then_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(tmp_path)
    stage_root = tmp_path / "stages" / "seed_summaries"
    meta_root = tmp_path / "meta_inputs"
    analysis_root = cfg.analysis_dir
    meta_root.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)
    cfg.io.meta_analysis_dir = meta_root

    monkeypatch.setattr(
        cfg,
        "stage_dir_if_active",
        lambda stage: stage_root if stage == "seed_summaries" else None,
    )

    stage_file = stage_root / "strategy_summary_2p_seed1.parquet"
    meta_file = meta_root / "strategy_summary_2p_seed2.parquet"
    analysis_file = analysis_root / "strategy_summary_2p_seed3.parquet"
    for path in (stage_file, meta_file, analysis_file):
        pd.DataFrame(
            [{"strategy_id": "A", "players": 2, "seed": int(path.stem.split("seed")[-1]), "games": 1, "wins": 1, "win_rate": 1.0}]
        ).to_parquet(path, index=False)

    # Duplicate seed in a lower-precedence dir should not override stage result.
    duplicate = meta_root / "strategy_summary_2p_seed1.parquet"
    pd.DataFrame(
        [{"strategy_id": "B", "players": 2, "seed": 1, "games": 1, "wins": 0, "win_rate": 0.0}]
    ).to_parquet(duplicate, index=False)

    collected = meta._collect_seed_summaries(cfg)
    assert collected[2][1] == stage_file
    assert collected[2][2] == meta_file
    assert collected[2][3] == analysis_file


def test_normalize_meta_frame_enforces_sort_and_dtypes() -> None:
    raw = pd.DataFrame(
        [
            {
                "strategy_id": 2,
                "players": 2.0,
                "win_rate": "0.4",
                "se": "0.1",
                "ci_lo": "0.2",
                "ci_hi": "0.6",
                "n_seeds": 2.0,
            },
            {
                "strategy_id": 1,
                "players": 2.0,
                "win_rate": "0.8",
                "se": "0.05",
                "ci_lo": "0.7",
                "ci_hi": "0.9",
                "n_seeds": 2.0,
            },
        ]
    )

    normalized = meta._normalize_meta_frame(raw)
    assert normalized["strategy_id"].tolist() == ["1", "2"]
    assert pd.api.types.is_integer_dtype(normalized["players"])
    assert pd.api.types.is_float_dtype(normalized["win_rate"])
    assert pd.api.types.is_integer_dtype(normalized["n_seeds"])


def test_meta_run_idempotent_skip_and_force_rewrite(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    df_seed1 = pd.DataFrame(
        [{"strategy_id": "A", "players": 2, "seed": 1, "games": 10, "wins": 5, "win_rate": 0.5}]
    )
    df_seed2 = pd.DataFrame(
        [{"strategy_id": "A", "players": 2, "seed": 2, "games": 10, "wins": 6, "win_rate": 0.6}]
    )
    df_seed1.to_parquet(cfg.analysis_dir / "strategy_summary_2p_seed1.parquet", index=False)
    df_seed2.to_parquet(cfg.analysis_dir / "strategy_summary_2p_seed2.parquet", index=False)

    meta.run(cfg, use_random_if_I2_gt=90.0)
    per_k_parquet = cfg.meta_output_path(2, "strategy_summary_2p_meta.parquet")
    per_k_json = cfg.meta_output_path(2, "meta_2p.json")
    long_path = cfg.meta_pooled_dir / "meta_long.parquet"
    assert per_k_parquet.exists()
    assert per_k_json.exists()
    assert long_path.exists()
    assert per_k_parquet.parent == cfg.meta_per_k_dir(2)

    parquet_mtime_1 = per_k_parquet.stat().st_mtime
    json_mtime_1 = per_k_json.stat().st_mtime
    long_mtime_1 = long_path.stat().st_mtime

    time.sleep(1.1)
    meta.run(cfg, force=False, use_random_if_I2_gt=90.0)
    assert per_k_parquet.stat().st_mtime == pytest.approx(parquet_mtime_1)
    assert per_k_json.stat().st_mtime == pytest.approx(json_mtime_1)
    assert long_path.stat().st_mtime == pytest.approx(long_mtime_1)

    time.sleep(1.1)
    meta.run(cfg, force=True, use_random_if_I2_gt=90.0)
    assert per_k_parquet.stat().st_mtime > parquet_mtime_1
    assert per_k_json.stat().st_mtime > json_mtime_1
    assert long_path.stat().st_mtime > long_mtime_1


def test_meta_run_recomputes_when_artifacts_partially_missing(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    for seed, wins in ((1, 5), (2, 7)):
        pd.DataFrame(
            [{"strategy_id": "A", "players": 2, "seed": seed, "games": 10, "wins": wins, "win_rate": wins / 10.0}]
        ).to_parquet(cfg.analysis_dir / f"strategy_summary_2p_seed{seed}.parquet", index=False)

    meta.run(cfg)
    parquet_path = cfg.meta_output_path(2, "strategy_summary_2p_meta.parquet")
    json_path = cfg.meta_output_path(2, "meta_2p.json")
    assert parquet_path.exists() and json_path.exists()

    json_path.unlink()
    time.sleep(1.1)
    before = parquet_path.stat().st_mtime
    meta.run(cfg, force=False)
    assert json_path.exists()
    assert parquet_path.stat().st_mtime == pytest.approx(before)


def test_parquet_matches_handles_missing_corrupt_and_column_mismatch(tmp_path: Path) -> None:
    good = pd.DataFrame(
        [{"strategy_id": "A", "players": 2, "win_rate": 0.5, "se": 0.1, "ci_lo": 0.3, "ci_hi": 0.7, "n_seeds": 2}]
    )
    missing_path = tmp_path / "missing.parquet"
    assert not meta._parquet_matches(missing_path, good)

    corrupt_path = tmp_path / "corrupt.parquet"
    corrupt_path.write_text("not parquet")
    assert not meta._parquet_matches(corrupt_path, good)

    mismatch_path = tmp_path / "mismatch.parquet"
    pd.DataFrame([{"strategy_id": "A", "players": 2, "win_rate": 0.5}]).to_parquet(
        mismatch_path, index=False
    )
    assert not meta._parquet_matches(mismatch_path, good)


def test_parquet_matches_true_for_equivalent_normalized_content(tmp_path: Path) -> None:
    existing = pd.DataFrame(
        [
            {"strategy_id": 2, "players": 2.0, "win_rate": 0.7, "se": 0.1, "ci_lo": 0.5, "ci_hi": 0.9, "n_seeds": 2.0},
            {"strategy_id": 1, "players": 2.0, "win_rate": 0.3, "se": 0.1, "ci_lo": 0.1, "ci_hi": 0.5, "n_seeds": 2.0},
        ]
    )
    new = pd.DataFrame(
        [
            {"strategy_id": "1", "players": 2, "win_rate": 0.3, "se": 0.1, "ci_lo": 0.1, "ci_hi": 0.5, "n_seeds": 2},
            {"strategy_id": "2", "players": 2, "win_rate": 0.7, "se": 0.1, "ci_lo": 0.5, "ci_hi": 0.9, "n_seeds": 2},
        ]
    )
    path = tmp_path / "same.parquet"
    existing.to_parquet(path, index=False)

    assert meta._parquet_matches(path, new)


def test_write_json_atomic_sorts_payload_and_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "meta.json"
    payload = {"z": 1.0, "a": "x"}

    meta._write_json_atomic(payload, path)

    assert path.exists()
    assert path.read_text() == '{\n  "a": "x",\n  "z": 1.0\n}'


def test_meta_run_returns_early_when_no_seed_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(tmp_path)
    calls: list[str] = []

    class _StageLog:
        def start(self) -> None:
            return

        def missing_input(self, message: str) -> None:
            calls.append(message)

    monkeypatch.setattr(meta, "stage_logger", lambda *_args, **_kwargs: _StageLog())

    meta.run(cfg)

    assert calls == ["no per-seed summaries found"]


def test_meta_run_skips_when_selection_leaves_only_one_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_cfg(tmp_path)
    for seed in (7, 99):
        pd.DataFrame(
            [
                {
                    "strategy_id": "A",
                    "players": 2,
                    "seed": seed,
                    "games": 10,
                    "wins": 5,
                    "win_rate": 0.5,
                }
            ]
        ).to_parquet(cfg.analysis_dir / f"strategy_summary_2p_seed{seed}.parquet", index=False)

    cfg.analysis.meta_comparison_seed = 7
    monkeypatch.setattr(meta, "stage_is_up_to_date", lambda *args, **kwargs: False)

    meta.run(cfg)

    assert not cfg.meta_output_path(2, "strategy_summary_2p_meta.parquet").exists()
    assert not cfg.meta_output_path(2, "meta_2p.json").exists()


def test_meta_run_rewrites_different_json_payload(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    for seed, wins in ((42, 6), (7, 4)):
        pd.DataFrame(
            [{"strategy_id": "A", "players": 2, "seed": seed, "games": 10, "wins": wins, "win_rate": wins / 10.0}]
        ).to_parquet(cfg.analysis_dir / f"strategy_summary_2p_seed{seed}.parquet", index=False)

    json_path = cfg.meta_output_path(2, "meta_2p.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"Q": -1, "I2": -1, "tau2": -1, "method": "fixed"}))

    meta.run(cfg)

    data = json.loads(json_path.read_text())
    assert data["Q"] >= 0.0
    assert data["I2"] >= 0.0


def test_meta_run_rewrites_invalid_json(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    for seed, wins in ((42, 6), (7, 4)):
        pd.DataFrame(
            [{"strategy_id": "A", "players": 2, "seed": seed, "games": 10, "wins": wins, "win_rate": wins / 10.0}]
        ).to_parquet(cfg.analysis_dir / f"strategy_summary_2p_seed{seed}.parquet", index=False)

    json_path = cfg.meta_output_path(2, "meta_2p.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text("{not-json")

    meta.run(cfg)

    data = json.loads(json_path.read_text())
    assert set(data) == {"Q", "I2", "tau2", "method"}


def test_meta_run_skips_zero_row_inputs(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    empty = pd.DataFrame(columns=["strategy_id", "players", "seed", "games", "wins", "win_rate"])
    empty.to_parquet(cfg.analysis_dir / "strategy_summary_2p_seed1.parquet", index=False)
    empty.to_parquet(cfg.analysis_dir / "strategy_summary_2p_seed2.parquet", index=False)

    meta.run(cfg)

    assert not cfg.meta_output_path(2, "strategy_summary_2p_meta.parquet").exists()
    assert not cfg.meta_output_path(2, "meta_2p.json").exists()
    assert not (cfg.meta_pooled_dir / "meta_long.parquet").exists()
