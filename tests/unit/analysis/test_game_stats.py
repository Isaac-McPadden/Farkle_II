import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from tests.helpers.diagnostic_fixtures import build_curated_fixture

from farkle.analysis import game_stats
from farkle.config import AppConfig, IOConfig, SimConfig


def test_rare_event_flags_cover_game_and_strategy_levels(tmp_path):
    cfg, _, per_n = build_curated_fixture(tmp_path)
    thresholds = (10, 60)

    output_path = tmp_path / "rare_events.parquet"
    rows = game_stats._rare_event_flags(
        [(2, per_n)],
        thresholds=thresholds,
        target_score=100,
        output_path=output_path,
        codec=cfg.parquet_codec,
    )

    assert rows > 0
    flags = pd.read_parquet(output_path)
    # Three game-level rows plus strategy-level and n-player summaries
    assert {"game", "strategy", "n_players"} <= set(flags["summary_level"].unique())

    aggro = flags[(flags["strategy"] == 1) & (flags["summary_level"] == "strategy")].iloc[0]
    assert aggro["observations"] == 5
    assert aggro["multi_reached_target"] == pytest.approx(0.6)
    assert aggro[f"margin_le_{thresholds[1]}"] == pytest.approx(1.0)


def test_summarize_rounds_handles_empty_and_values():
    empty = game_stats._summarize_rounds([])
    assert empty["observations"] == 0
    assert pd.isna(empty["mean_rounds"])

    stats = game_stats._summarize_rounds([1, 5, 9])
    assert stats["observations"] == 3
    assert stats["prob_rounds_le_5"] == pytest.approx(2 / 3)


def _build_parquet(tmp_path: Path, cfg):
    rows = pd.DataFrame(
        [
            {
                "seat_ranks": ["P1", "P2"],
                "n_rounds": 4,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 120,
                "P2_score": 110,
            },
            {
                "seat_ranks": ["P2", "P1"],
                "n_rounds": 8,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 50,
                "P2_score": 200,
            },
            {
                "seat_ranks": ["P1", "P2"],
                "n_rounds": 12,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 300,
                "P2_score": 100,
            },
        ]
    )

    per_n_path = cfg.ingested_rows_curated(2)
    per_n_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(per_n_path)

    combined_path = cfg.curated_parquet
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(combined_path)

    return per_n_path, combined_path


def test_run_generates_all_outputs(tmp_path: Path):
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2]))
    per_n_path, combined_path = _build_parquet(tmp_path, cfg)

    game_stats.run(cfg, force=True)

    game_length = cfg.game_stats_output_path("game_length.parquet")
    margin_path = cfg.game_stats_output_path("margin_stats.parquet")
    rare_events_path = cfg.game_stats_output_path("rare_events.parquet")

    assert game_length.exists()
    assert margin_path.exists()
    assert rare_events_path.exists()

    # Strategy summaries come from per-n inputs; global stats come from combined parquet
    game_df = pd.read_parquet(game_length)
    assert {"strategy", "n_players"} <= set(game_df.columns)
    assert any(game_df["summary_level"] == "n_players")

    margin_df = pd.read_parquet(margin_path)
    assert set(margin_df["summary_level"].unique()) == {"strategy"}
    assert all(
        col in margin_df.columns
        for col in ("mean_margin_runner_up", "median_margin_runner_up", "mean_score_spread")
    )

    per_k_game_length = cfg.per_k_subdir("game_stats", 2) / "game_length.parquet"
    per_k_margin = cfg.per_k_subdir("game_stats", 2) / "margin_stats.parquet"
    assert per_k_game_length.exists()
    assert per_k_margin.exists()

    per_k_game_df = pd.read_parquet(per_k_game_length)
    per_k_margin_df = pd.read_parquet(per_k_margin)
    assert set(per_k_game_df["n_players"].dropna().astype(int).unique()) <= {2}
    assert set(per_k_margin_df["n_players"].dropna().astype(int).unique()) <= {2}

    rare_df = pd.read_parquet(rare_events_path)
    assert {"game", "strategy", "n_players"} <= set(rare_df["summary_level"].unique())

    stamp = cfg.game_stats_stage_dir / "game_stats.done.json"
    assert stamp.exists()
    stamp_meta = json.loads(stamp.read_text())
    assert str(per_k_game_length) in stamp_meta["outputs"]
    assert str(per_k_margin) in stamp_meta["outputs"]



def test_run_requires_inputs(tmp_path: Path):
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))

    game_stats.run(cfg)

    assert not cfg.game_stats_output_path("game_length.parquet").exists()
    assert not cfg.game_stats_output_path("margin_stats.parquet").exists()
    assert not cfg.game_stats_output_path("rare_events.parquet").exists()


def test_compute_margins_and_aggregation(tmp_path: Path):
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    per_n_path, _ = _build_parquet(tmp_path, cfg)

    per_n_inputs = [(2, per_n_path)]
    margins = game_stats._per_strategy_margin_stats(per_n_inputs, thresholds=(100,))
    assert not margins.empty
    assert margins.loc[0, "prob_margin_runner_up_le_100"] == pytest.approx(1 / 3)
    assert margins.loc[0, "prob_score_spread_le_100"] == pytest.approx(1 / 3)

    rare_path = tmp_path / "rare_events.parquet"
    rare_rows = game_stats._rare_event_flags(
        per_n_inputs,
        thresholds=(100,),
        target_score=150,
        output_path=rare_path,
        codec=cfg.parquet_codec,
    )
    assert rare_rows > 0
    rare = pd.read_parquet(rare_path)
    assert set(rare["summary_level"].unique()) >= {"game", "strategy"}


def test_global_stats_warns_when_seat_ranks_missing(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch):
    class DummyDataset:
        schema = type("Schema", (), {"names": ["n_rounds"]})()

        @staticmethod
        def to_table(_columns=None):
            return pa.Table.from_pandas(pd.DataFrame({"n_rounds": [1, 2, 3]}))

    monkeypatch.setattr(game_stats.ds, "dataset", lambda path: DummyDataset())

    with caplog.at_level("WARNING"):
        result = game_stats._global_stats(Path("dummy"))

    assert result.empty
    assert "Combined parquet missing seat_ranks" in caplog.text


def test_global_stats_handles_numpy_array_seat_ranks(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyDataset:
        schema = type("Schema", (), {"names": ["seat_ranks", "n_rounds"]})()

        @staticmethod
        def to_table(_columns=None):
            return pa.Table.from_pandas(
                pd.DataFrame(
                    {
                        "seat_ranks": [
                            np.array(["P1", "P2"], dtype=object),
                            np.array(["P2", "P1"], dtype=object),
                        ],
                        "n_rounds": [4, 8],
                    }
                )
            )

    monkeypatch.setattr(game_stats.ds, "dataset", lambda _path: DummyDataset())
    monkeypatch.setattr(game_stats, "n_players_from_schema", lambda _schema: 12)

    result = game_stats._global_stats(Path("dummy"))

    assert not result.empty
    assert set(result["n_players"].astype(int).tolist()) == {2}


def _write_multi_k_curated_inputs(cfg: AppConfig) -> None:
    rows_2p = pd.DataFrame(
        [
            {
                "seat_ranks": ["P1", "P2"],
                "n_rounds": 5,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 120,
                "P2_score": 100,
            },
            {
                "seat_ranks": ["P2", "P1"],
                "n_rounds": 7,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_score": 70,
                "P2_score": 160,
            },
        ]
    )
    rows_3p = pd.DataFrame(
        [
            {
                "seat_ranks": ["P1", "P2", "P3"],
                "n_rounds": 6,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P3_strategy": 3,
                "P1_score": 210,
                "P2_score": 180,
                "P3_score": 120,
            },
            {
                "seat_ranks": ["P3", "P1", "P2"],
                "n_rounds": 10,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P3_strategy": 3,
                "P1_score": 190,
                "P2_score": 150,
                "P3_score": 230,
            },
        ]
    )

    for n_players, rows in ((2, rows_2p), (3, rows_3p)):
        per_n_path = cfg.ingested_rows_curated(n_players)
        per_n_path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(per_n_path)

    combined = pd.concat([rows_2p, rows_3p], ignore_index=True, sort=False)
    combined_path = cfg.curated_parquet
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(combined_path)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_writes_per_k_outputs_and_is_idempotent_for_multi_k(tmp_path: Path) -> None:
    k_values = [2, 3]
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path),
        sim=SimConfig(n_players_list=k_values, seed=123, seed_list=[123]),
    )
    _write_multi_k_curated_inputs(cfg)

    game_stats.run(cfg)

    pooled_targets = {
        "game_length.parquet": cfg.game_stats_output_path("game_length.parquet"),
        "margin_stats.parquet": cfg.game_stats_output_path("margin_stats.parquet"),
    }
    for output_path in pooled_targets.values():
        assert output_path.exists()

    expected_per_k_paths: list[Path] = []
    for k in k_values:
        for filename in pooled_targets:
            path = cfg.per_k_subdir("game_stats", k) / filename
            assert path.exists()
            expected_per_k_paths.append(path)

            per_k_df = pd.read_parquet(path)
            assert set(per_k_df["n_players"].dropna().astype(int).unique()) <= {k}

            pooled_df = pd.read_parquet(pooled_targets[filename])
            expected = pooled_df.loc[pooled_df["n_players"] == k].reset_index(drop=True)
            pd.testing.assert_frame_equal(per_k_df.reset_index(drop=True), expected)

    tracked_paths = list(pooled_targets.values()) + expected_per_k_paths
    before = {path: (path.stat().st_mtime_ns, _hash_file(path)) for path in tracked_paths}

    game_stats.run(cfg)

    after = {path: (path.stat().st_mtime_ns, _hash_file(path)) for path in tracked_paths}
    assert after == before


def test_run_resolves_rare_event_thresholds_from_histograms(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2]))
    _build_parquet(tmp_path, cfg)
    cfg.analysis.rare_event_margin_quantile = 0.5
    cfg.analysis.rare_event_target_rate = 0.4

    game_stats.run(cfg, force=True)

    rare_df = pd.read_parquet(cfg.game_stats_output_path("rare_events.parquet"))
    assert "margin_le_150" in rare_df.columns

    strat_row = rare_df[
        (rare_df["summary_level"] == "strategy") & (rare_df["strategy"] == 1.0)
    ].iloc[0]
    assert strat_row["multi_reached_target"] == pytest.approx(2 / 3)


def test_run_generates_margin_summary_columns_and_histogram_inputs(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2]))
    _build_parquet(tmp_path, cfg)
    cfg.analysis.game_stats_margin_thresholds = (25, 175)
    cfg.analysis.rare_event_margin_quantile = 0.9
    cfg.analysis.rare_event_target_rate = 0.2

    game_stats.run(cfg, force=True)

    margin_df = pd.read_parquet(cfg.game_stats_output_path("margin_stats.parquet"))
    pooled_margin_df = pd.read_parquet(cfg.game_stats_output_path("margin_k_weighted.parquet"))
    rare_df = pd.read_parquet(cfg.game_stats_output_path("rare_events.parquet"))

    for threshold in (25, 175):
        assert f"prob_margin_runner_up_le_{threshold}" in margin_df.columns
        assert f"prob_score_spread_le_{threshold}" in margin_df.columns
        assert f"prob_margin_runner_up_le_{threshold}" in pooled_margin_df.columns
        assert f"prob_score_spread_le_{threshold}" in pooled_margin_df.columns

    derived_margin_cols = [c for c in rare_df.columns if c.startswith("margin_le_")]
    assert derived_margin_cols == ["margin_le_200"]


def test_run_accepts_pooling_scheme_aliases(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2]))
    _build_parquet(tmp_path, cfg)
    cfg.analysis.pooling_weights = "equal_k"

    game_stats.run(cfg, force=True)

    assert cfg.game_stats_output_path("game_length_k_weighted.parquet").exists()
    assert cfg.game_stats_output_path("margin_k_weighted.parquet").exists()


def test_run_raises_for_invalid_pooling_scheme(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2]))
    _build_parquet(tmp_path, cfg)
    cfg.analysis.pooling_weights = "invalid-scheme"

    with pytest.raises(ValueError, match="Unknown pooling scheme"):
        game_stats.run(cfg, force=True)

def test_discover_per_n_inputs_handles_partial_layouts(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2, 3]))

    assert game_stats._discover_per_n_inputs(cfg) == []

    valid_dir = cfg.data_dir / "2p"
    valid_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir = cfg.data_dir / "badp"
    invalid_dir.mkdir(parents=True, exist_ok=True)
    missing_file_dir = cfg.data_dir / "3p"
    missing_file_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.DataFrame(
        [{"seat_ranks": ["P1", "P2"], "n_rounds": 3, "P1_strategy": 1, "P2_strategy": 2, "P1_score": 100, "P2_score": 90}]
    )
    rows.to_parquet(valid_dir / cfg.curated_rows_name)

    discovered = game_stats._discover_per_n_inputs(cfg)
    assert discovered == [(2, valid_dir / cfg.curated_rows_name)]


def test_run_force_and_uptodate_paths_and_partial_per_k(tmp_path: Path) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path),
        sim=SimConfig(n_players_list=[2, 3], seed=101, seed_list=[101]),
    )
    _build_parquet(tmp_path, cfg)

    game_stats.run(cfg, force=True)

    game_length = cfg.game_stats_output_path("game_length.parquet")
    margin = cfg.game_stats_output_path("margin_stats.parquet")
    per_k_2_game = cfg.per_k_subdir("game_stats", 2) / "game_length.parquet"
    per_k_3_game = cfg.per_k_subdir("game_stats", 3) / "game_length.parquet"

    assert game_length.exists()
    assert margin.exists()
    assert per_k_2_game.exists()
    assert per_k_3_game.exists()

    before_main = game_length.stat().st_mtime_ns
    game_stats.run(cfg, force=False)
    assert game_length.stat().st_mtime_ns == before_main

    # remove only one per-k output to force partial recompute path
    per_k_2_game.unlink()
    assert not per_k_2_game.exists()
    game_stats.run(cfg, force=False)
    assert per_k_2_game.exists()
    assert per_k_3_game.exists()


def test_run_raises_when_per_k_fanout_writer_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path),
        sim=SimConfig(n_players_list=[2, 3], seed=404, seed_list=[404]),
    )
    _write_multi_k_curated_inputs(cfg)

    game_stats.run(cfg, force=True)

    stage_stamp = cfg.game_stats_stage_dir / "game_stats.done.json"
    stage_stamp_before = stage_stamp.read_text()
    stage_stamp_mtime_before = stage_stamp.stat().st_mtime_ns

    stale_k = 2
    stale_output = cfg.per_k_subdir("game_stats", stale_k) / "game_length.parquet"
    stale_stamp = cfg.game_stats_stage_dir / f"game_stats.game_length.{stale_k}p.done.json"
    stale_output.unlink()
    stale_stamp.unlink()

    healthy_k = 3
    healthy_output = cfg.per_k_subdir("game_stats", healthy_k) / "game_length.parquet"
    healthy_stamp = cfg.game_stats_stage_dir / f"game_stats.game_length.{healthy_k}p.done.json"
    healthy_output_mtime_before = healthy_output.stat().st_mtime_ns
    healthy_stamp_before = healthy_stamp.read_text()

    original_writer = game_stats._write_per_k_game_length

    def _fail_one_k(**kwargs):
        if kwargs["k"] == stale_k:
            raise RuntimeError("boom: per-k writer failure")
        return original_writer(**kwargs)

    monkeypatch.setattr(game_stats, "_write_per_k_game_length", _fail_one_k)

    with pytest.raises(RuntimeError, match="per-k writer failure"):
        game_stats.run(cfg, force=False)

    assert stage_stamp.exists()
    assert stage_stamp.read_text() == stage_stamp_before
    assert stage_stamp.stat().st_mtime_ns == stage_stamp_mtime_before

    assert not stale_output.exists()
    assert not stale_stamp.exists()

    assert healthy_output.exists()
    assert healthy_output.stat().st_mtime_ns == healthy_output_mtime_before
    assert healthy_stamp.read_text() == healthy_stamp_before


def test_run_pooling_alias_and_invalid_via_run(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[2]))
    _build_parquet(tmp_path, cfg)

    cfg.analysis.pooling_weights = "count"
    game_stats.run(cfg, force=True)
    assert cfg.game_stats_output_path("game_length_k_weighted.parquet").exists()

    cfg.analysis.pooling_weights = "definitely-bad"
    with pytest.raises(ValueError, match="Unknown pooling scheme"):
        game_stats.run(cfg, force=True)


def _legacy_per_strategy_stats(per_n_inputs: list[tuple[int, Path]]) -> pd.DataFrame:
    long_frames: list[pd.DataFrame] = []
    for n_players, path in per_n_inputs:
        ds_in = game_stats.ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        for col in strategy_cols:
            scanner = ds_in.scanner(columns=["n_rounds", col], batch_size=65_536)
            for batch in scanner.to_batches():
                df = batch.to_pandas(categories=[col])
                melted = df.melt(id_vars=["n_rounds"], value_vars=[col], value_name="strategy")
                melted = melted.dropna(subset=["strategy"])
                if melted.empty:
                    continue
                melted["strategy"] = melted["strategy"].astype("category")
                melted["n_players"] = n_players
                long_frames.append(melted[["strategy", "n_players", "n_rounds"]])

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df["n_rounds"] = pd.to_numeric(long_df["n_rounds"], errors="coerce")
    long_df = long_df.dropna(subset=["n_rounds", "strategy"])
    grouped = long_df.groupby(["strategy", "n_players"], observed=True, sort=False)["n_rounds"]
    stats = grouped.agg(
        observations="count",
        mean_rounds="mean",
        median_rounds="median",
        std_rounds=lambda s: s.std(ddof=0),
        p10_rounds=lambda s: s.quantile(0.1),
        p50_rounds=lambda s: s.quantile(0.5),
        p90_rounds=lambda s: s.quantile(0.9),
    )
    prob_rounds_le_5 = (
        long_df["n_rounds"]
        .le(5)
        .groupby([long_df["strategy"], long_df["n_players"]], observed=True, sort=False)
        .mean()
        .rename("prob_rounds_le_5")
    )
    prob_rounds_le_10 = (
        long_df["n_rounds"]
        .le(10)
        .groupby([long_df["strategy"], long_df["n_players"]], observed=True, sort=False)
        .mean()
        .rename("prob_rounds_le_10")
    )
    prob_rounds_ge_20 = (
        long_df["n_rounds"]
        .ge(20)
        .groupby([long_df["strategy"], long_df["n_players"]], observed=True, sort=False)
        .mean()
        .rename("prob_rounds_ge_20")
    )
    stats = stats.join([prob_rounds_le_5, prob_rounds_le_10, prob_rounds_ge_20]).reset_index()
    stats.insert(0, "summary_level", "strategy")
    return stats[
        [
            "summary_level",
            "strategy",
            "n_players",
            "observations",
            "mean_rounds",
            "median_rounds",
            "std_rounds",
            "p10_rounds",
            "p50_rounds",
            "p90_rounds",
            "prob_rounds_le_5",
            "prob_rounds_le_10",
            "prob_rounds_ge_20",
        ]
    ]


def _legacy_per_strategy_margin_stats(
    per_n_inputs: list[tuple[int, Path]], *, thresholds: tuple[int, ...]
) -> pd.DataFrame:
    long_frames: list[pd.DataFrame] = []
    for n_players, path in per_n_inputs:
        ds_in = game_stats.ds.dataset(path)
        strategy_cols = [name for name in ds_in.schema.names if name.endswith("_strategy")]
        score_cols = [name for name in ds_in.schema.names if name.startswith("P") and name.endswith("_score")]

        for col in strategy_cols:
            scanner = ds_in.scanner(columns=[*score_cols, col], batch_size=65_536)
            for batch in scanner.to_batches():
                df = batch.to_pandas(categories=[col])
                if df.empty:
                    continue
                margin_cols = game_stats._compute_margin_columns(df, score_cols)
                df = df.assign(
                    margin_runner_up=margin_cols["margin_runner_up"],
                    score_spread=margin_cols["score_spread"],
                )
                melted = df.melt(
                    id_vars=["margin_runner_up", "score_spread"],
                    value_vars=[col],
                    value_name="strategy",
                )
                melted = melted.dropna(subset=["strategy"])
                if melted.empty:
                    continue
                melted["strategy"] = melted["strategy"].astype("category")
                melted["n_players"] = n_players
                long_frames.append(
                    melted[["strategy", "n_players", "margin_runner_up", "score_spread"]]
                )

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df["margin_runner_up"] = pd.to_numeric(long_df["margin_runner_up"], errors="coerce")
    long_df["score_spread"] = pd.to_numeric(long_df["score_spread"], errors="coerce")
    long_df = long_df.dropna(subset=["margin_runner_up", "strategy"])

    grouped = long_df.groupby(["strategy", "n_players"], observed=True, sort=False)
    runner_stats = grouped["margin_runner_up"].agg(
        observations="count",
        mean_margin_runner_up="mean",
        median_margin_runner_up="median",
        std_margin_runner_up=lambda s: s.std(ddof=0),
    )
    spread_stats = grouped["score_spread"].agg(
        mean_score_spread="mean",
        median_score_spread="median",
        std_score_spread=lambda s: s.std(ddof=0),
    )
    prob_frames = []
    for thr in thresholds:
        runner_prob = (
            long_df["margin_runner_up"]
            .le(thr)
            .groupby([long_df["strategy"], long_df["n_players"]], observed=True, sort=False)
            .mean()
            .rename(f"prob_margin_runner_up_le_{thr}")
        )
        spread_prob = (
            long_df["score_spread"]
            .le(thr)
            .groupby([long_df["strategy"], long_df["n_players"]], observed=True, sort=False)
            .mean()
            .rename(f"prob_score_spread_le_{thr}")
        )
        prob_frames.extend([runner_prob, spread_prob])

    stats = runner_stats.join([spread_stats, *prob_frames]).reset_index()
    stats.insert(0, "summary_level", "strategy")
    ordered_cols = [
        "summary_level",
        "strategy",
        "n_players",
        "observations",
        "mean_margin_runner_up",
        "median_margin_runner_up",
        "std_margin_runner_up",
        *[f"prob_margin_runner_up_le_{thr}" for thr in thresholds],
        "mean_score_spread",
        "median_score_spread",
        "std_score_spread",
        *[f"prob_score_spread_le_{thr}" for thr in thresholds],
    ]
    return stats[ordered_cols]


def test_refactored_batch_melt_matches_legacy_outputs(tmp_path: Path) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path), sim=SimConfig(n_players_list=[3]))
    rows = pd.DataFrame(
        [
            {
                "seat_ranks": ["P1", "P2", "P3"],
                "n_rounds": 4,
                "P1_strategy": 10,
                "P2_strategy": 20,
                "P3_strategy": 30,
                "P1_score": 900,
                "P2_score": 850,
                "P3_score": 200,
            },
            {
                "seat_ranks": ["P3", "P1", "P2"],
                "n_rounds": 12,
                "P1_strategy": 10,
                "P2_strategy": 20,
                "P3_strategy": 30,
                "P1_score": 700,
                "P2_score": 600,
                "P3_score": 950,
            },
            {
                "seat_ranks": ["P2", "P3", "P1"],
                "n_rounds": 22,
                "P1_strategy": 10,
                "P2_strategy": 20,
                "P3_strategy": 30,
                "P1_score": 500,
                "P2_score": 990,
                "P3_score": 640,
            },
        ]
    )
    per_n = cfg.ingested_rows_curated(3)
    per_n.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(per_n)
    per_n_inputs = [(3, per_n)]

    actual_rounds = game_stats._per_strategy_stats(per_n_inputs)
    legacy_rounds = _legacy_per_strategy_stats(per_n_inputs)
    pd.testing.assert_frame_equal(
        actual_rounds.sort_values(["strategy", "n_players"]).reset_index(drop=True),
        legacy_rounds.sort_values(["strategy", "n_players"]).reset_index(drop=True),
        check_dtype=False,
    )

    thresholds = (100, 300)
    actual_margin = game_stats._per_strategy_margin_stats(per_n_inputs, thresholds=thresholds)
    legacy_margin = _legacy_per_strategy_margin_stats(per_n_inputs, thresholds=thresholds)
    pd.testing.assert_frame_equal(
        actual_margin.sort_values(["strategy", "n_players"]).reset_index(drop=True),
        legacy_margin.sort_values(["strategy", "n_players"]).reset_index(drop=True),
        check_dtype=False,
    )
