from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import TracebackType
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

import farkle.simulation.runner as runner
from farkle.config import AppConfig, IOConfig, SimConfig
from farkle.simulation.strategies import ThresholdStrategy


def _cfg(tmp_path: Path, **sim_kwargs: Any) -> AppConfig:
    sim_defaults = {
        "n_players_list": [2],
        "num_shuffles": 2,
        "seed": 123,
        "recompute_num_shuffles": False,
    }
    sim_defaults.update(sim_kwargs)
    return AppConfig(io=IOConfig(results_dir_prefix=tmp_path / "out"), sim=SimConfig(**sim_defaults))


def _manifest_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"strategy_id": 1, "strategy_str": "s1", "score_threshold": 300},
            {"strategy_id": 2, "strategy_str": "s2", "score_threshold": 500},
        ]
    )


def test_filter_player_counts_invalid_vs_grid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(runner, "experiment_size", lambda **_: 6)

    valid, invalid, grid_size, source = runner._filter_player_counts(cfg, [0, -1, 2, 4, 6])

    assert valid == [2, 6]
    assert invalid == [0, -1, 4]
    assert grid_size == 6
    assert source == "experiment_size"


def test_compute_num_shuffles_precedence_and_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    per_n_cfg = SimConfig(num_shuffles=11, recompute_num_shuffles=False)
    cfg = _cfg(tmp_path, per_n={2: per_n_cfg}, recompute_num_shuffles=True, num_shuffles=99)

    monkeypatch.setattr(
        runner,
        "games_for_power_from_design",
        lambda n_strategies, k_players, method, design: (_ for _ in ()).throw(AssertionError()),
    )
    assert runner._compute_num_shuffles_from_config(cfg, n_strategies=8, n_players=2) == 11

    cfg_no_override = _cfg(tmp_path, recompute_num_shuffles=True, num_shuffles=77)
    monkeypatch.setattr(runner, "games_for_power_from_design", lambda **_: 5)
    assert runner._compute_num_shuffles_from_config(cfg_no_override, n_strategies=8, n_players=2) == 5

    cfg_fallback = _cfg(tmp_path, recompute_num_shuffles=False, num_shuffles=13)
    assert runner._compute_num_shuffles_from_config(cfg_fallback, n_strategies=8, n_players=2) == 13


def test_output_dir_and_done_helpers(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, row_dir=Path("rows"), metric_chunk_dir=Path("metrics"))

    row_dir = runner._resolve_row_output_dir(cfg, 2)
    metric_dir = runner._resolve_metric_chunk_dir(cfg, 2)
    assert row_dir == cfg.results_root / "2_players" / "2p_rows"
    assert metric_dir == cfg.results_root / "2_players" / "2p_metrics"

    done_path = runner.simulation_done_path(cfg, 2)
    assert done_path == cfg.results_root / "2_players" / "simulation.done.json"
    assert runner.simulation_is_complete(cfg, 2) is False

    marker = runner.write_simulation_done(
        cfg,
        2,
        num_shuffles=3,
        n_strategies=8,
        outputs=[cfg.results_root / "2_players" / "2p_checkpoint.pkl"],
    )
    assert marker == done_path
    assert runner.simulation_is_complete(cfg, 2) is True
    payload = json.loads(done_path.read_text())
    assert payload["n_players"] == 2
    assert payload["num_shuffles"] == 3


def test_manifest_digest_and_validate_mismatch(tmp_path: Path) -> None:
    manifest = _manifest_df()
    manifest_path = tmp_path / "manifest.parquet"
    runner.write_parquet_atomic(runner.pa.Table.from_pandas(manifest, preserve_index=False), manifest_path)

    digest_1 = runner._strategy_manifest_digest(manifest)
    digest_2 = runner._strategy_manifest_digest(manifest.iloc[::-1].reset_index(drop=True))
    assert digest_1 != digest_2

    bad_schema = manifest.rename(columns={"strategy_id": "id"})
    runner.write_parquet_atomic(runner.pa.Table.from_pandas(bad_schema, preserve_index=False), manifest_path)
    with pytest.raises(ValueError, match="schema mismatch"):
        runner._validate_manifest_matches(manifest, manifest_path, label="Strategy")

    runner.write_parquet_atomic(runner.pa.Table.from_pandas(manifest.iloc[::-1], preserve_index=False), manifest_path)
    with pytest.raises(ValueError, match="manifest mismatch"):
        runner._validate_manifest_matches(manifest, manifest_path, label="Strategy")


def test_purge_existing_and_resume_output_validation(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    n_players = 2
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = n_dir / "2p_checkpoint.pkl"
    row_dir = n_dir / "2p_rows"
    metric_dir = n_dir / "2p_metrics"
    row_dir.mkdir(parents=True)
    metric_dir.mkdir(parents=True)

    (n_dir / "2p_checkpoint.parquet").write_text("x")
    (n_dir / "2p_metrics.parquet").write_text("x")
    (row_dir / "rows_0.parquet").write_text("x")
    (metric_dir / "metrics_0.parquet").write_text("x")
    ckpt_payload = {
        "meta": {
            "n_players": 2,
            "num_shuffles": 2,
            "global_seed": cfg.sim.seed,
            "n_strategies": len(_manifest_df()),
            "strategy_manifest_sha": runner._strategy_manifest_digest(_manifest_df()),
        }
    }
    ckpt_path.write_bytes(pickle.dumps(ckpt_payload))
    runner.write_parquet_atomic(
        runner.pa.Table.from_pandas(_manifest_df(), preserve_index=False),
        cfg.strategy_manifest_root_path(),
    )

    assert runner._has_existing_outputs(
        n_dir=n_dir,
        n_players=n_players,
        ckpt_path=ckpt_path,
        row_dir=row_dir,
        metric_chunk_dir=metric_dir,
    )

    runner._validate_resume_outputs(
        cfg=cfg,
        n_players=2,
        n_shuffles=2,
        strategies_manifest=_manifest_df(),
        ckpt_path=ckpt_path,
        row_dir=row_dir,
        metric_chunk_dir=metric_dir,
    )

    runner._purge_simulation_outputs(
        n_dir=n_dir,
        n_players=n_players,
        ckpt_path=ckpt_path,
        row_dir=row_dir,
        metric_chunk_dir=metric_dir,
        strategy_manifest_path=cfg.strategy_manifest_root_path(),
    )

    assert not runner._has_existing_outputs(
        n_dir=n_dir,
        n_players=n_players,
        ckpt_path=ckpt_path,
        row_dir=row_dir,
        metric_chunk_dir=metric_dir,
    )


def _patch_tournament_writer(monkeypatch: pytest.MonkeyPatch, *, wrong_meta: bool = False) -> None:
    def fake_run_tournament(**kwargs: Any) -> None:  # noqa: ANN001
        ckpt = Path(kwargs["checkpoint_path"])
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        metadata = kwargs["checkpoint_metadata"]
        assert isinstance(metadata, dict)
        payload = {
            "win_totals": {"s0": 1},
            "meta": {
                "n_players": kwargs["n_players"],
                "num_shuffles": kwargs["num_shuffles"],
                "global_seed": kwargs["global_seed"],
                "n_strategies": len(kwargs["strategies"]),
                "strategy_manifest_sha": "bad" if wrong_meta else metadata["strategy_manifest_sha"],
            },
        }
        ckpt.write_bytes(pickle.dumps(payload))

    monkeypatch.setattr(runner.tournament_mod, "run_tournament", fake_run_tournament)


def test_run_single_n_force_overwrite_and_resume_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path, row_dir=Path("rows"), metric_chunk_dir=Path("metrics"), num_shuffles=1)
    _patch_tournament_writer(monkeypatch)

    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    stale = n_dir / "2p_checkpoint.parquet"
    stale.write_text("stale")

    assert runner.run_single_n(cfg, 2, force=True) > 0
    assert stale.exists()
    assert runner.simulation_is_complete(cfg, 2)

    assert runner.run_single_n(cfg, 2, force=False) > 0


@pytest.mark.parametrize(
    ("case", "force", "existing_outputs", "resume_side_effect", "expect_games", "expect_error"),
    [
        pytest.param("force", True, False, None, 3, None, id="force-cleanup"),
        pytest.param("resume-valid", False, True, None, 3, None, id="resume-valid"),
        pytest.param(
            "resume-invalid",
            False,
            True,
            ValueError("resume metadata mismatch"),
            None,
            ValueError,
            id="resume-invalid",
        ),
    ],
)
def test_run_single_n_branch_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    force: bool,
    existing_outputs: bool,
    resume_side_effect: Exception | None,
    expect_games: int | None,
    expect_error: type[Exception] | None,
) -> None:
    cfg = _cfg(tmp_path, row_dir=Path("rows"), metric_chunk_dir=Path("metrics"), num_shuffles=3)
    n_players = 2
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    stale_done = runner.simulation_done_path(cfg, n_players)
    stale_done.parent.mkdir(parents=True, exist_ok=True)
    stale_done.write_text('{"stale": true}')

    strategies = [ThresholdStrategy(300, 3), ThresholdStrategy(500, 2)]
    monkeypatch.setattr(runner, "_resolve_strategies", lambda cfg, strategies: (strategies or [], 2, True))
    monkeypatch.setattr(runner, "_compute_num_shuffles_from_config", lambda *_args, **_kwargs: 3)
    monkeypatch.setattr(runner, "build_strategy_manifest", lambda _strategies: _manifest_df())

    calls: dict[str, int] = {"worker": 0}

    def fake_run_tournament(**kwargs: Any) -> None:  # noqa: ANN001
        calls["worker"] += 1
        ckpt = Path(kwargs["checkpoint_path"])
        meta = kwargs["checkpoint_metadata"]
        assert isinstance(meta, dict)
        ckpt.write_bytes(
            pickle.dumps(
                {
                    "win_totals": {"s0": 1, "s1": 0},
                    "meta": {
                        "n_players": kwargs["n_players"],
                        "num_shuffles": kwargs["num_shuffles"],
                        "global_seed": kwargs["global_seed"],
                        "n_strategies": len(kwargs["strategies"]),
                        "strategy_manifest_sha": meta["strategy_manifest_sha"],
                    },
                }
            )
        )

    monkeypatch.setattr(runner.tournament_mod, "run_tournament", fake_run_tournament)

    purge_mock = MagicMock(side_effect=runner._purge_simulation_outputs)
    has_outputs_mock = MagicMock(return_value=existing_outputs)
    validate_mock = MagicMock(side_effect=resume_side_effect)
    monkeypatch.setattr(runner, "_purge_simulation_outputs", purge_mock)
    monkeypatch.setattr(runner, "_has_existing_outputs", has_outputs_mock)
    monkeypatch.setattr(runner, "_validate_resume_outputs", validate_mock)

    if expect_error is not None:
        with pytest.raises(expect_error):
            runner.run_single_n(cfg, n_players, strategies=strategies, force=force)
        assert calls["worker"] == 0
    else:
        assert runner.run_single_n(cfg, n_players, strategies=strategies, force=force) == expect_games
        assert calls["worker"] == 1
        assert runner.simulation_is_complete(cfg, n_players)

    assert purge_mock.call_count == (1 if force else 0), case
    assert has_outputs_mock.call_count == (0 if force else 1), case
    assert validate_mock.call_count == (1 if (not force and existing_outputs) else 0), case


def test_run_single_n_cleanup_purges_done_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, row_dir=Path("rows"), metric_chunk_dir=Path("metrics"), num_shuffles=1)
    _patch_tournament_writer(monkeypatch)

    n_players = 2
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    stale_done = n_dir / "simulation.done.json"
    stale_done.write_text('{"stale": true}')

    assert runner.run_single_n(cfg, n_players, force=True) > 0
    payload = json.loads(stale_done.read_text())
    assert payload["n_players"] == n_players
    assert payload["num_shuffles"] == 1


def test_run_single_n_resume_invalid_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, row_dir=Path("rows"), num_shuffles=1)
    _patch_tournament_writer(monkeypatch)

    assert runner.run_single_n(cfg, 2, force=False) > 0

    ckpt_path = cfg.results_root / "2_players" / "2p_checkpoint.pkl"
    payload = pickle.loads(ckpt_path.read_bytes())
    payload["meta"]["global_seed"] = -1
    ckpt_path.write_bytes(pickle.dumps(payload))

    _patch_tournament_writer(monkeypatch, wrong_meta=True)
    with pytest.raises(ValueError, match="Checkpoint metadata mismatch"):
        runner.run_single_n(cfg, 2, force=False)


def test_run_multi_invalid_counts_warns_and_returns_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _cfg(tmp_path, n_players_list=[5, 7])

    monkeypatch.setattr(
        runner,
        "generate_strategy_grid",
        lambda **_: ([ThresholdStrategy(300, 3) for _ in range(4)], None),
    )

    with caplog.at_level("WARNING", logger=runner.LOGGER.name):
        out = runner.run_multi(cfg)

    assert out == {}
    assert any("No valid player counts remain" in rec.getMessage() for rec in caplog.records)


def test_run_multi_filters_counts_and_calls_worker_once_per_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path, n_players_list=[1, 2, 3, 4, 6])

    monkeypatch.setattr(
        runner,
        "generate_strategy_grid",
        lambda **_: ([ThresholdStrategy(300, 3) for _ in range(6)], None),
    )

    worker_calls: list[tuple[int, bool]] = []

    def fake_run_single_n(
        cfg: AppConfig,
        n: int,
        strategies: list[ThresholdStrategy] | None = None,
        *,
        force: bool = False,
    ) -> int:
        assert strategies is not None
        worker_calls.append((n, force))
        return 10 * n

    monkeypatch.setattr(runner, "run_single_n", fake_run_single_n)
    out = runner.run_multi(cfg, force=True)

    assert out == {1: 10, 2: 20, 3: 30, 6: 60}
    assert worker_calls == [(1, True), (2, True), (3, True), (6, True)]


def test_filter_player_counts_zero_grid_allows_positive_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _cfg(tmp_path, smart_five_opts=[False], smart_one_opts=[True])
    monkeypatch.setattr(runner, "experiment_size", lambda **_: 0)

    valid, invalid, grid_size, source = runner._filter_player_counts(cfg, [0, -2, 1, 2, 3])

    assert valid == [1, 2, 3]
    assert invalid == [0, -2]
    assert grid_size == 0
    assert source == "experiment_size"


def test_compute_num_shuffles_invalid_power_config_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, recompute_num_shuffles=True)

    def fail_power(**_: Any) -> int:
        raise ValueError("invalid power configuration")

    monkeypatch.setattr(runner, "games_for_power_from_design", fail_power)

    with pytest.raises(ValueError, match="invalid power configuration"):
        runner._compute_num_shuffles_from_config(cfg, n_strategies=8, n_players=2)


def test_resolve_per_n_output_dir_edge_cases(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    assert runner._resolve_per_n_output_dir(cfg, None, 3) is None

    assert runner._resolve_per_n_output_dir(cfg, Path("rows/{n}/rows"), 3) == (
        cfg.results_root / "3_players" / "rows" / "3" / "rows"
    )
    assert runner._resolve_per_n_output_dir(cfg, Path("rows/{n_players}/rows"), 3) == (
        cfg.results_root / "3_players" / "rows" / "3" / "rows"
    )
    assert runner._resolve_per_n_output_dir(cfg, Path("rows/{p}/rows"), 3) == (
        cfg.results_root / "3_players" / "rows" / "3p" / "rows"
    )

    # Unknown placeholder falls back to raw string processing + prefix behavior.
    assert runner._resolve_per_n_output_dir(cfg, Path("{bad}_rows"), 3) == (
        cfg.results_root / "3_players" / "3p_{bad}_rows"
    )

    assert runner._resolve_per_n_output_dir(cfg, Path("rows"), 3) == (
        cfg.results_root / "3_players" / "3p_rows"
    )

    abs_template = tmp_path / "{n_players}p_rows"
    assert runner._resolve_per_n_output_dir(cfg, abs_template, 3) == tmp_path / "3p_rows"


def test_handle_remove_error_chmod_retry_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    removed: list[str] = []

    def fake_remover(path: str) -> None:
        removed.append(path)

    chmod_calls: list[str] = []
    monkeypatch.setattr(runner.os, "chmod", lambda path, mode: chmod_calls.append(path))
    err = PermissionError("original")
    try:
        raise err
    except PermissionError as exc:
        tb = exc.__traceback__
        assert tb is not None
        runner._handle_remove_error(fake_remover, "x.txt", (PermissionError, err, tb))

    assert chmod_calls == ["x.txt"]
    assert removed == ["x.txt"]

    def fail_chmod(_path: str, _mode: int) -> None:
        raise OSError("chmod failed")

    monkeypatch.setattr(runner.os, "chmod", fail_chmod)
    original = PermissionError("keep me")
    with pytest.raises(PermissionError, match="keep me"):
        try:
            raise original
        except PermissionError as exc:
            tb = exc.__traceback__
            assert tb is not None
            runner._handle_remove_error(fake_remover, "y.txt", (PermissionError, original, tb))


def test_remove_paths_missing_directory_and_permission_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing.txt"
    dir_path = tmp_path / "adir"
    dir_path.mkdir()
    file_path = tmp_path / "needs_retry.txt"
    file_path.write_text("x")

    rmtree_calls: list[Path] = []
    monkeypatch.setattr(
        runner.shutil,
        "rmtree",
        lambda path, onerror: rmtree_calls.append(Path(path)),
    )

    orig_unlink = Path.unlink
    unlink_calls = {"count": 0}

    def fake_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path == file_path and unlink_calls["count"] == 0:
            unlink_calls["count"] += 1
            raise PermissionError("deny")
        unlink_calls["count"] += 1
        return orig_unlink(path, missing_ok=missing_ok)

    chmod_calls: list[Path] = []
    monkeypatch.setattr(Path, "unlink", fake_unlink)
    monkeypatch.setattr(
        runner.os,
        "chmod",
        lambda p, mode: chmod_calls.append(Path(p)),
    )

    runner._remove_paths([missing, dir_path, file_path])

    assert rmtree_calls == [dir_path]
    assert chmod_calls == [file_path]
    assert unlink_calls["count"] >= 2
    assert not file_path.exists()


@pytest.mark.parametrize(
    "signal",
    [
        "checkpoint_pickle",
        "checkpoint_parquet",
        "metrics_parquet",
        "root_strategy_manifest",
        "row_manifest",
        "row_chunk",
        "row_strategy_manifest",
        "metrics_manifest",
        "metrics_chunk",
    ],
)
def test_has_existing_outputs_independent_signals(tmp_path: Path, signal: str) -> None:
    cfg = _cfg(tmp_path)
    n_players = 2
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = n_dir / "2p_checkpoint.pkl"
    row_dir = n_dir / "rows"
    metric_dir = n_dir / "metrics"
    row_dir.mkdir()
    metric_dir.mkdir()

    paths = {
        "checkpoint_pickle": ckpt_path,
        "checkpoint_parquet": n_dir / "2p_checkpoint.parquet",
        "metrics_parquet": n_dir / "2p_metrics.parquet",
        "root_strategy_manifest": n_dir / runner.STRATEGY_MANIFEST_NAME,
        "row_manifest": row_dir / "manifest.jsonl",
        "row_chunk": row_dir / "rows_000.parquet",
        "row_strategy_manifest": row_dir / runner.STRATEGY_MANIFEST_NAME,
        "metrics_manifest": metric_dir / "metrics_manifest.jsonl",
        "metrics_chunk": metric_dir / "metrics_000.parquet",
    }
    paths[signal].write_text("x")

    assert runner._has_existing_outputs(
        n_dir=n_dir,
        n_players=n_players,
        ckpt_path=ckpt_path,
        row_dir=row_dir,
        metric_chunk_dir=metric_dir,
    )


def test_has_existing_outputs_negative_case(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    assert not runner._has_existing_outputs(
        n_dir=n_dir,
        n_players=2,
        ckpt_path=n_dir / "2p_checkpoint.pkl",
        row_dir=n_dir / "rows",
        metric_chunk_dir=n_dir / "metrics",
    )


def _write_manifest_lines(path: Path, lines: list[dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def test_validate_resume_outputs_promotes_legacy_manifest(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = n_dir / "2p_checkpoint.pkl"
    manifest = _manifest_df()
    legacy_manifest = n_dir / runner.STRATEGY_MANIFEST_NAME
    runner.write_parquet_atomic(runner.pa.Table.from_pandas(manifest, preserve_index=False), legacy_manifest)
    ckpt_path.write_bytes(
        pickle.dumps(
            {
                "meta": {
                    "n_players": 2,
                    "num_shuffles": 2,
                    "global_seed": cfg.sim.seed,
                    "n_strategies": len(manifest),
                    "strategy_manifest_sha": runner._strategy_manifest_digest(manifest),
                }
            }
        )
    )

    runner._validate_resume_outputs(
        cfg=cfg,
        n_players=2,
        n_shuffles=2,
        strategies_manifest=manifest,
        ckpt_path=ckpt_path,
        row_dir=None,
        metric_chunk_dir=None,
    )

    assert cfg.strategy_manifest_root_path().exists()


@pytest.mark.parametrize("meta", [None, "bad"])
def test_validate_resume_outputs_checkpoint_meta_missing_or_non_mapping(
    tmp_path: Path, meta: str | None
) -> None:
    cfg = _cfg(tmp_path)
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = n_dir / "2p_checkpoint.pkl"
    ckpt_path.write_bytes(pickle.dumps({"meta": meta}))

    with pytest.raises(ValueError, match="Checkpoint metadata missing"):
        runner._validate_resume_outputs(
            cfg=cfg,
            n_players=2,
            n_shuffles=2,
            strategies_manifest=_manifest_df(),
            ckpt_path=ckpt_path,
            row_dir=None,
            metric_chunk_dir=None,
        )


@pytest.mark.parametrize(
    "bad_key,bad_value",
    [
        ("n_players", 99),
        ("num_shuffles", 99),
        ("global_seed", 999),
        ("n_strategies", 999),
        ("strategy_manifest_sha", "wrong"),
    ],
)
def test_validate_resume_outputs_checkpoint_meta_mismatch_keys(
    tmp_path: Path,
    bad_key: str,
    bad_value: int | str,
) -> None:
    cfg = _cfg(tmp_path)
    manifest = _manifest_df()
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = n_dir / "2p_checkpoint.pkl"
    meta = {
        "n_players": 2,
        "num_shuffles": 2,
        "global_seed": cfg.sim.seed,
        "n_strategies": len(manifest),
        "strategy_manifest_sha": runner._strategy_manifest_digest(manifest),
    }
    meta[bad_key] = bad_value
    ckpt_path.write_bytes(pickle.dumps({"meta": meta}))

    with pytest.raises(ValueError, match=f"Checkpoint metadata mismatch for {bad_key}"):
        runner._validate_resume_outputs(
            cfg=cfg,
            n_players=2,
            n_shuffles=2,
            strategies_manifest=manifest,
            ckpt_path=ckpt_path,
            row_dir=None,
            metric_chunk_dir=None,
        )


def test_validate_resume_outputs_no_checkpoint_and_no_valid_manifest_raises(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = n_dir / "2p_checkpoint.pkl"
    runner.write_parquet_atomic(
        runner.pa.Table.from_pandas(_manifest_df(), preserve_index=False),
        n_dir / runner.STRATEGY_MANIFEST_NAME,
    )

    with pytest.raises(ValueError, match="Existing outputs found without a checkpoint"):
        runner._validate_resume_outputs(
            cfg=cfg,
            n_players=2,
            n_shuffles=2,
            strategies_manifest=_manifest_df(),
            ckpt_path=ckpt_path,
            row_dir=None,
            metric_chunk_dir=None,
        )


@pytest.mark.parametrize(
    "rows,match",
    [
        ([{"shuffle_seed": 1, "n_players": 2}, {"shuffle_seed": 1, "n_players": 2}], "Duplicate shuffle"),
        ([{"shuffle_seed": 999, "n_players": 2}], "Row manifest mismatch"),
        ([{"shuffle_seed": 1, "n_players": 9}], "Row manifest mismatch"),
    ],
)
def test_validate_resume_outputs_row_manifest_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, int]],
    match: str,
) -> None:
    cfg = _cfg(tmp_path)
    n_dir = cfg.results_root / "2_players"
    row_dir = n_dir / "rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest_lines(row_dir / "manifest.jsonl", rows)
    monkeypatch.setattr(runner.urandom, "spawn_seeds", lambda n, seed: [1, 2])

    with pytest.raises(ValueError, match=match):
        runner._validate_resume_outputs(
            cfg=cfg,
            n_players=2,
            n_shuffles=2,
            strategies_manifest=_manifest_df(),
            ckpt_path=n_dir / "2p_checkpoint.pkl",
            row_dir=row_dir,
            metric_chunk_dir=None,
        )


@pytest.mark.parametrize(
    "rows,match",
    [
        ([{"chunk_index": 0, "n_players": 2}, {"chunk_index": 0, "n_players": 2}], "Duplicate chunk"),
        ([{"chunk_index": 0, "n_players": 9}], "Metrics manifest mismatch"),
    ],
)
def test_validate_resume_outputs_metrics_manifest_invalid(
    tmp_path: Path,
    rows: list[dict[str, int]],
    match: str,
) -> None:
    cfg = _cfg(tmp_path)
    n_dir = cfg.results_root / "2_players"
    metric_dir = n_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest_lines(metric_dir / "metrics_manifest.jsonl", rows)

    with pytest.raises(ValueError, match=match):
        runner._validate_resume_outputs(
            cfg=cfg,
            n_players=2,
            n_shuffles=2,
            strategies_manifest=_manifest_df(),
            ckpt_path=n_dir / "2p_checkpoint.pkl",
            row_dir=None,
            metric_chunk_dir=metric_dir,
        )


def test_validate_resume_outputs_row_and_metrics_manifests_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    n_dir = cfg.results_root / "2_players"
    row_dir = n_dir / "rows"
    metric_dir = n_dir / "metrics"
    row_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest_lines(
        row_dir / "manifest.jsonl",
        [{"shuffle_seed": 1, "n_players": 2}, {"shuffle_seed": 2, "n_players": 2}],
    )
    _write_manifest_lines(
        metric_dir / "metrics_manifest.jsonl",
        [{"chunk_index": 0, "n_players": 2}, {"chunk_index": 1, "n_players": 2}],
    )
    monkeypatch.setattr(runner.urandom, "spawn_seeds", lambda n, seed: [1, 2])

    runner._validate_resume_outputs(
        cfg=cfg,
        n_players=2,
        n_shuffles=2,
        strategies_manifest=_manifest_df(),
        ckpt_path=n_dir / "2p_checkpoint.pkl",
        row_dir=row_dir,
        metric_chunk_dir=metric_dir,
    )


def test_run_tournament_dispatch_error_and_multi_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_empty = _cfg(tmp_path, n_players_list=[])
    with pytest.raises(ValueError, match="must contain at least one"):
        runner.run_tournament(cfg_empty)

    cfg_invalid = _cfg(tmp_path, n_players_list=[7, 9])
    monkeypatch.setattr(runner, "_filter_player_counts", lambda *_: ([], [7, 9], 6, "experiment_size"))
    with pytest.raises(ValueError, match="No valid player counts remain"):
        runner.run_tournament(cfg_invalid)

    cfg_multi = _cfg(tmp_path, n_players_list=[2, 4])
    monkeypatch.setattr(runner, "_filter_player_counts", lambda *_: ([2, 4], [], 8, "experiment_size"))
    monkeypatch.setattr(runner, "run_multi", lambda cfg, player_counts, force=False: {2: 4, 4: 8})
    assert runner.run_tournament(cfg_multi, force=True) == 12


def test_run_single_n_empty_rows_and_legacy_sq_sum_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _cfg(tmp_path, expanded_metrics=True, num_shuffles=2)
    strategies = [ThresholdStrategy(300, 3)]

    monkeypatch.setattr(runner, "_resolve_strategies", lambda cfg, s: (strategies, 1, True))
    monkeypatch.setattr(runner, "_compute_num_shuffles_from_config", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(
        runner,
        "build_strategy_manifest",
        lambda _strategies: pd.DataFrame([{"strategy_id": "s0", "strategy_str": "s0"}]),
    )

    payloads = [
        {"win_totals": {"s0": -1}, "meta": {}},
        {
            "win_totals": {"s0": 2},
            "metric_sums": {"winning_score": {"s0": 8.0}},
            "metric_sq_sums": {"winning_score": {"s0": 40.0}},
            "meta": {},
        },
    ]

    def fake_worker(**kwargs: Any) -> None:  # noqa: ANN001
        ckpt = Path(kwargs["checkpoint_path"])
        meta_sha = kwargs["checkpoint_metadata"]["strategy_manifest_sha"]
        payload = payloads.pop(0)
        payload["meta"] = {
            "n_players": kwargs["n_players"],
            "num_shuffles": kwargs["num_shuffles"],
            "global_seed": kwargs["global_seed"],
            "n_strategies": len(kwargs["strategies"]),
            "strategy_manifest_sha": meta_sha,
        }
        ckpt.write_bytes(pickle.dumps(payload))

    captured_outputs: list[list[Path]] = []

    def fake_done(cfg: AppConfig, n_players: int, *, num_shuffles: int, n_strategies: int, outputs: list[Path]) -> Path:
        captured_outputs.append(outputs)
        return runner.simulation_done_path(cfg, n_players)

    monkeypatch.setattr(runner.tournament_mod, "run_tournament", fake_worker)
    monkeypatch.setattr(runner, "write_simulation_done", fake_done)

    with caplog.at_level("DEBUG", logger=runner.LOGGER.name):
        runner.run_single_n(cfg, 2, force=True)

    n_dir = cfg.results_root / "2_players"
    assert not (n_dir / "2p_checkpoint.parquet").exists()
    assert not (n_dir / "2p_metrics.parquet").exists()
    assert any("Strategy manifest written" in rec.getMessage() for rec in caplog.records)
    assert all(p.name.endswith("checkpoint.pkl") or p.name == runner.STRATEGY_MANIFEST_NAME for p in captured_outputs[0])

    runner.run_single_n(cfg, 2, force=True)
    metrics = pd.read_parquet(n_dir / "2p_metrics.parquet")
    assert float(metrics.loc[0, "sq_sum_winning_score"]) == 40.0
