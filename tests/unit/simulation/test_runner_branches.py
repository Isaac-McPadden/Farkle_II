from __future__ import annotations

import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from tests.helpers.artifact_sidecars import (
    make_authenticated_v3_config,
    publish_v3_simulation_run,
)

pa = pytest.importorskip("pyarrow")

import farkle.simulation.runner as runner
from farkle.config import AppConfig, SimConfig, assign_config_sha
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils.artifact_contract import write_artifact_with_sidecar_atomic


def _write_fake_checkpoint(kwargs: dict[str, Any], payload: object) -> None:
    path = Path(kwargs["checkpoint_path"])
    content = pickle.dumps(payload)
    sidecar = kwargs.get("checkpoint_sidecar")
    if sidecar is None:
        path.write_bytes(content)
        return

    def _write(staged: Path) -> None:
        staged.write_bytes(content)

    write_artifact_with_sidecar_atomic(path, sidecar, _write)


def _cfg(tmp_path: Path, **sim_kwargs: Any) -> AppConfig:
    test_batch_count = max(2, int(sim_kwargs.pop("num_shuffles", 2)))
    seed = int(sim_kwargs.pop("seed", 123))
    n_players_list = list(sim_kwargs.pop("n_players_list", [2]))
    cfg = make_authenticated_v3_config(
        tmp_path,
        name="out",
        root_seed=seed,
        player_counts=tuple(n_players_list),
    )
    for name, value in sim_kwargs.items():
        setattr(cfg.sim, name, value)
    cfg.screening.resolution_delta = 0.9
    cfg.batching.target_batches = test_batch_count
    cfg.batching.min_shuffles_per_batch = 1
    assign_config_sha(cfg)
    return cfg


def _manifest_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"strategy_id": 1, "strategy_str": "s1", "score_threshold": 300},
            {"strategy_id": 2, "strategy_str": "s2", "score_threshold": 500},
        ]
    )


def _workload_meta(cfg: AppConfig, manifest: pd.DataFrame | None = None) -> dict[str, object]:
    manifest = _manifest_df() if manifest is None else manifest
    plan = runner._plan_workload_from_config(
        cfg,
        n_strategies=len(manifest),
        n_players=2,
    )
    return runner._workload_checkpoint_metadata(plan)


def test_validate_workload_reports_every_invalid_count(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    with pytest.raises(ValueError, match=r"\[0, -1, 1, 4\].*6 strategies"):
        runner.validate_workload(cfg, player_counts=[0, -1, 1, 2, 4, 6], grid_size=6)
    runner.validate_workload(cfg, player_counts=[2, 6], grid_size=6)


def test_workload_plan_replaces_legacy_shuffle_sizing(tmp_path: Path) -> None:
    per_n_cfg = SimConfig()
    cfg = _cfg(tmp_path, per_n={2: per_n_cfg}, num_shuffles=99)
    cfg.batching.target_batches = 3

    plan = runner._plan_workload_from_config(cfg, n_strategies=8, n_players=2)

    assert plan.required_shuffles == 3
    assert plan.required_games == 12
    assert plan.batch_count == 3


def test_output_dir_and_done_helpers(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, row_dir=Path("rows"), metric_chunk_dir=Path("metrics"))

    row_dir = runner._resolve_row_output_dir(cfg, 2)
    metric_dir = runner._resolve_metric_chunk_dir(cfg, 2)
    assert row_dir == cfg.results_root / "2_players" / "2p_rows"
    assert metric_dir == cfg.results_root / "2_players" / "2p_metrics"

    done_path = runner.simulation_done_path(cfg, 2)
    assert done_path == cfg.results_root / "2_players" / "simulation.done.json"
    assert runner.simulation_is_complete(cfg, 2) is False
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text(
        '{"rng_scheme_version": 1, "outcome_schema_version": 2, "tournament_method_version": 2}'
    )
    assert runner.simulation_is_complete(cfg, 2) is False

    published = publish_v3_simulation_run(
        cfg,
        pa.table({"value": [1]}),
        strategies=(
            ThresholdStrategy(300, 3, strategy_id=1),
            ThresholdStrategy(500, 2, strategy_id=2),
        ),
    )
    marker = published.completion
    assert marker == done_path
    assert runner.simulation_is_complete(cfg, 2) is True
    payload = json.loads(done_path.read_text())
    assert payload["state"] == "complete_valid"
    assert {
        (item.get("artifact") or item.get("manifest"))["location"]["relative_path"]
        for item in payload["outputs"]
    } == {
        "2_players/2p_rows/manifest.jsonl",
        "2_players/simulation_workload_plan.json",
        "strategy_manifest.parquet",
    }


def test_manifest_digest_and_validate_mismatch(tmp_path: Path) -> None:
    manifest = _manifest_df()
    manifest_path = tmp_path / "manifest.parquet"
    runner.write_parquet_atomic(
        runner.pa.Table.from_pandas(manifest, preserve_index=False), manifest_path
    )

    digest_1 = runner._strategy_manifest_digest(manifest)
    digest_2 = runner._strategy_manifest_digest(manifest.iloc[::-1].reset_index(drop=True))
    assert digest_1 != digest_2

    bad_schema = manifest.rename(columns={"strategy_id": "id"})
    runner.write_parquet_atomic(
        runner.pa.Table.from_pandas(bad_schema, preserve_index=False), manifest_path
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        runner._validate_manifest_matches(manifest, manifest_path, label="Strategy")

    runner.write_parquet_atomic(
        runner.pa.Table.from_pandas(manifest.iloc[::-1], preserve_index=False), manifest_path
    )
    with pytest.raises(ValueError, match="manifest mismatch"):
        runner._validate_manifest_matches(manifest, manifest_path, label="Strategy")


def _write_blocked_plan(cfg: AppConfig) -> Path:
    cfg.screening.max_shuffles_per_root_k = 1
    plan = runner._plan_workload_from_config(cfg, n_strategies=2, n_players=2)
    cfg.screening.max_shuffles_per_root_k = None
    assert plan.cap_exceeded
    path = cfg.simulation_workload_plan_path(2)
    runner.write_workload_plan(path, plan)
    return path


def test_force_cleans_exact_prior_control_with_coincident_directories(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, row_dir=Path("shared"), metric_chunk_dir=Path("shared"))
    prior = _write_blocked_plan(cfg)
    rows, metrics = runner._force_output_directories(cfg, 2)
    assert rows is not None and rows == metrics
    rows.mkdir()
    unrelated = {
        rows / "keep.txt": b"private",
        rows / "rows_notes.parquet": b"notes",
        rows / "metrics_backup.parquet": b"backup",
        rows / "._tmp_notes": b"never issued by this run",
    }
    for path, data in unrelated.items():
        path.write_bytes(data)
    plan = runner._plan_workload_from_config(cfg, n_strategies=2, n_players=2)
    sealed, cleanup = runner._validate_force_inventory(cfg, plan, rows, metrics)
    assert not sealed
    assert list(cleanup) == [prior]
    runner._purge_simulation_outputs(cleanup)
    assert not prior.exists()
    assert {path: path.read_bytes() for path in unrelated} == unrelated


def test_force_rechecks_exact_bytes_before_removing_plan(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    prior = _write_blocked_plan(cfg)
    plan = runner._plan_workload_from_config(cfg, n_strategies=2, n_players=2)
    _, cleanup = runner._validate_force_inventory(cfg, plan, None, None)
    original = prior.stat()
    changed = prior.read_bytes().replace(b"blocked_by_cap", b"unrelated_data")
    assert len(changed) == original.st_size
    prior.write_bytes(changed)
    os.utime(prior, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(ValueError, match="target changed"):
        runner._purge_simulation_outputs(cleanup)
    assert prior.read_bytes() == changed


@pytest.mark.parametrize("target", ["row", "metric", "checkpoint", "manifest", "plan"])
def test_ambiguous_occupant_preserves_entire_inventory(tmp_path: Path, target: str) -> None:
    cfg = _cfg(tmp_path, row_dir=Path("shared"), metric_chunk_dir=Path("shared"))
    prior = _write_blocked_plan(cfg)
    rows = cfg.simulation_row_dir(2)
    assert rows is not None
    rows.mkdir()
    path = {
        "row": rows / f"rows_{cfg.sim.seed}_2p_000000000000.parquet",
        "metric": rows / "metrics_000000.parquet",
        "checkpoint": cfg.checkpoint_path(2),
        "manifest": rows / "manifest.jsonl",
        "plan": prior,
    }[target]
    path.write_bytes(b"unrelated")
    before = prior.read_bytes()
    with pytest.raises(ValueError, match="Cannot establish simulation cleanup ownership"):
        runner.run_single_n(
            cfg, 2, strategies=[ThresholdStrategy(300, 3), ThresholdStrategy(500, 2)], force=True
        )
    assert prior.read_bytes() == before
    assert path.read_bytes() == b"unrelated"


@pytest.mark.parametrize(
    "kind", ["external", "parent-alias", "symlink", "junction", "file-link", "file-symlink"]
)
def test_force_rejects_unsafe_paths_before_removing_anything(tmp_path: Path, kind: str) -> None:
    cfg = _cfg(tmp_path)
    prior = _write_blocked_plan(cfg)
    original = prior.read_bytes()
    target = tmp_path / "2p_private"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_bytes(b"keep")
    alias = cfg.n_dir(2) / "2p_alias"
    if kind == "external":
        cfg.sim.metric_chunk_dir = target
    elif kind == "parent-alias":
        cfg.sim.row_dir = cfg.n_dir(2) / "unused" / ".." / "2p_alias"
    elif kind == "symlink":
        try:
            alias.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlinks unavailable: {exc}")
        cfg.sim.row_dir = alias
    elif kind == "junction":
        if os.name != "nt":
            pytest.skip("Windows junction")
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(target)],
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        cfg.sim.metric_chunk_dir = alias
    elif kind == "file-link":
        os.link(sentinel, cfg.checkpoint_path(2))
    else:
        try:
            cfg.checkpoint_path(2).symlink_to(sentinel)
        except OSError as exc:
            pytest.skip(f"file symlinks unavailable: {exc}")
    try:
        with pytest.raises(
            ValueError, match="Unsafe simulation cleanup|outside the canonical run root"
        ):
            runner.run_single_n(
                cfg,
                2,
                strategies=[ThresholdStrategy(300, 3), ThresholdStrategy(500, 2)],
                force=True,
            )
        assert sentinel.read_bytes() == b"keep"
        assert prior.read_bytes() == original
    finally:
        if kind == "junction":
            alias.rmdir()


def test_force_checks_later_cells_before_cleanup(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path, n_players_list=[2, 4])
    first = cfg.simulation_workload_plan_path(2)
    cfg.screening.max_shuffles_per_root_k = 1
    plan = runner._plan_workload_from_config(cfg, n_strategies=5160, n_players=2)
    cfg.screening.max_shuffles_per_root_k = None
    runner.write_workload_plan(first, plan)
    saved = first.read_bytes()
    later = cfg.checkpoint_path(4)
    later.parent.mkdir()
    later.write_bytes(b"unrelated checkpoint name")
    monkeypatch.setattr(
        runner,
        "run_single_n",
        lambda *args, **kwargs: pytest.fail("started a cell before preflight"),
    )
    with pytest.raises(ValueError, match="Cannot establish simulation cleanup ownership"):
        runner.run_multi(cfg, force=True)
    assert first.read_bytes() == saved
    assert later.read_bytes() == b"unrelated checkpoint name"


def _patch_tournament_writer(monkeypatch: pytest.MonkeyPatch, *, wrong_meta: bool = False) -> None:
    def fake_run_tournament(**kwargs: Any) -> None:  # noqa: ANN001
        ckpt = Path(kwargs["checkpoint_path"])
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        metadata = kwargs["checkpoint_metadata"]
        assert isinstance(metadata, dict)
        payload = {
            "win_totals": {"s0": 1},
            "outcome_counts": {
                "games_attempted": 1,
                "games_completed": 1,
                "games_safety_limit": 0,
                "attempted_exposures": {"s0": 1, "s1": 1},
                "completed_exposures": {"s0": 1, "s1": 1},
                "safety_limit_exposures": {},
            },
            "meta": {
                "n_players": kwargs["config"].n_players,
                "num_shuffles": kwargs["num_shuffles"],
                "global_seed": kwargs["global_seed"],
                "n_strategies": len(kwargs["strategies"]),
                **metadata,
                "strategy_manifest_sha": "bad" if wrong_meta else metadata["strategy_manifest_sha"],
            },
        }
        _write_fake_checkpoint(kwargs, payload)

    monkeypatch.setattr(runner.tournament_mod, "run_tournament", fake_run_tournament)


def test_run_single_n_force_overwrite_and_resume_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path, row_dir=Path("rows"), metric_chunk_dir=Path("metrics"), num_shuffles=1)
    _patch_tournament_writer(monkeypatch)

    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    stale = _write_blocked_plan(cfg)

    strategies = [ThresholdStrategy(300, 3), ThresholdStrategy(500, 2)]
    assert runner.run_single_n(cfg, 2, strategies, force=True) > 0
    assert json.loads(stale.read_bytes())["status"] == "not_started"
    assert runner.simulation_is_complete(cfg, 2)

    assert runner.run_single_n(cfg, 2, strategies, force=False) > 0


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
    strategies = [ThresholdStrategy(300, 3), ThresholdStrategy(500, 2)]
    monkeypatch.setattr(
        runner, "_resolve_strategies", lambda cfg, strategies: (strategies or [], 2, True)
    )
    monkeypatch.setattr(runner, "build_strategy_manifest", lambda _strategies: _manifest_df())

    calls: dict[str, int] = {"worker": 0}

    def fake_run_tournament(**kwargs: Any) -> None:  # noqa: ANN001
        calls["worker"] += 1
        meta = kwargs["checkpoint_metadata"]
        assert isinstance(meta, dict)
        _write_fake_checkpoint(
            kwargs,
            {
                "win_totals": {"s0": 1, "s1": 0},
                "outcome_counts": {
                    "games_attempted": 1,
                    "games_completed": 1,
                    "games_safety_limit": 0,
                    "attempted_exposures": {"s0": 1, "s1": 1},
                    "completed_exposures": {"s0": 1, "s1": 1},
                    "safety_limit_exposures": {},
                },
                "meta": {
                    "n_players": kwargs["config"].n_players,
                    "num_shuffles": kwargs["num_shuffles"],
                    "global_seed": kwargs["global_seed"],
                    "n_strategies": len(kwargs["strategies"]),
                    **meta,
                },
            },
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
        assert (
            runner.run_single_n(cfg, n_players, strategies=strategies, force=force) == expect_games
        )
        assert calls["worker"] == 1
        assert runner.simulation_is_complete(cfg, n_players)

    assert purge_mock.call_count == (1 if force else 0), case
    assert has_outputs_mock.call_count == (0 if force else 1), case
    assert validate_mock.call_count == (1 if (not force and existing_outputs) else 0), case


def test_run_single_n_force_preserves_unrecognized_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, row_dir=Path("rows"), metric_chunk_dir=Path("metrics"), num_shuffles=1)
    _patch_tournament_writer(monkeypatch)

    n_players = 2
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    stale_done = n_dir / "simulation.done.json"
    stale_done.write_text('{"stale": true}')

    with pytest.raises(ValueError, match="completion failed validation"):
        runner.run_single_n(cfg, n_players, force=True)
    assert stale_done.read_text() == '{"stale": true}'


def test_run_single_n_cap_blocks_before_force_cleanup(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, num_shuffles=2)
    cfg.screening.max_shuffles_per_root_k = 1
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = n_dir / "2p_checkpoint.pkl"
    checkpoint.write_bytes(b"preserve-me")

    with pytest.raises(runner.WorkloadCapExceeded, match="max_shuffles_per_root_k"):
        runner.run_single_n(cfg, 2, force=True)

    assert checkpoint.read_bytes() == b"preserve-me"
    plan = json.loads((n_dir / "simulation_workload_plan.json").read_text(encoding="utf-8"))
    assert plan["status"] == "blocked_by_cap"
    assert plan["required_shuffles"] == 2
    assert plan["shuffle_cap"] == 1


def test_run_single_n_resume_invalid_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


@pytest.mark.parametrize("counts", [[5, 7], [2, 5], []])
def test_run_multi_rejects_all_invalid_support_before_starting_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, counts: list[int]
) -> None:
    cfg = _cfg(tmp_path, n_players_list=counts)
    monkeypatch.setattr(
        runner,
        "generate_strategy_grid",
        lambda **_: ([ThresholdStrategy(300, 3) for _ in range(4)], None),
    )
    monkeypatch.setattr(
        runner, "run_single_n", lambda *args, **kwargs: pytest.fail("started an invalid workload")
    )

    with pytest.raises(ValueError, match="sim.n_players_list"):
        runner.run_multi(cfg)
    assert not cfg.results_root.exists()


def test_run_multi_calls_worker_once_per_count_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path, n_players_list=[3, 2, 6])

    monkeypatch.setattr(
        runner,
        "generate_strategy_grid",
        lambda **_: ([ThresholdStrategy(300, 3) for _ in range(6)], None),
    )

    cfg.sim.n_jobs = 12
    worker_calls: list[tuple[int, bool, int | None]] = []
    active_counts: set[int] = set()

    def fake_run_single_n(
        cfg: AppConfig,
        n: int,
        strategies: list[ThresholdStrategy] | None = None,
        *,
        force: bool = False,
    ) -> int:
        assert strategies is not None
        assert not active_counts
        active_counts.add(n)
        try:
            worker_calls.append((n, force, cfg.sim.n_jobs))
            return 10 * n
        finally:
            active_counts.remove(n)

    monkeypatch.setattr(runner, "run_single_n", fake_run_single_n)
    out = runner.run_multi(cfg, force=True)

    assert out == {3: 30, 2: 20, 6: 60}
    assert worker_calls == [
        (3, True, 12),
        (2, True, 12),
        (6, True, 12),
    ]
    assert not active_counts


def test_validate_workload_rejects_zero_strategy_grid(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, smart_five_opts=[False], smart_one_opts=[True])
    with pytest.raises(ValueError, match=r"\[2\].*0 strategies"):
        runner.validate_workload(cfg)


def test_workload_plan_rejects_incompatible_strategy_count(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    with pytest.raises(ValueError, match="multiple of k"):
        runner._plan_workload_from_config(cfg, n_strategies=7, n_players=2)


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


def _write_manifest_lines(path: Path, lines: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


def _write_root_strategy_manifest(cfg: AppConfig, manifest: pd.DataFrame) -> None:
    path = cfg.strategy_manifest_root_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_parquet_atomic(runner.pa.Table.from_pandas(manifest, preserve_index=False), path)


def _row_manifest_record(
    cfg: AppConfig,
    shuffle_index: int,
    *,
    shuffle_seed: int | None = None,
    n_players: int = 2,
) -> dict[str, object]:
    return {
        "root_seed": cfg.sim.seed,
        "n_players": n_players,
        "shuffle_index": shuffle_index,
        "shuffle_seed": shuffle_index + 1 if shuffle_seed is None else shuffle_seed,
        "deterministic_batch_id": shuffle_index,
        "rng_scheme_version": runner.urandom.RNG_SCHEME_VERSION,
        "rng_purpose_namespace": int(runner.urandom.RandomPurpose.TOURNAMENT_SHUFFLE),
        "outcome_schema_version": runner.OUTCOME_SCHEMA_VERSION,
        "tournament_method_version": runner.TOURNAMENT_METHOD_VERSION,
    }


def _metric_manifest_record(
    cfg: AppConfig,
    deterministic_batch_id: int,
    *,
    n_players: int = 2,
) -> dict[str, object]:
    return {
        "root_seed": cfg.sim.seed,
        "n_players": n_players,
        "process_block_index": deterministic_batch_id + 1,
        "deterministic_batch_id": deterministic_batch_id,
        "shuffle_index_start": deterministic_batch_id,
        "shuffle_index_end": deterministic_batch_id,
        "shuffle_count": 1,
        "shuffle_indices": [deterministic_batch_id],
        "shuffle_seeds": [deterministic_batch_id + 1],
        "rng_scheme_version": runner.urandom.RNG_SCHEME_VERSION,
        "rng_purpose_namespace": int(runner.urandom.RandomPurpose.TOURNAMENT_SHUFFLE),
        "outcome_schema_version": runner.OUTCOME_SCHEMA_VERSION,
        "tournament_method_version": runner.TOURNAMENT_METHOD_VERSION,
    }


def test_validate_resume_outputs_rejects_noncanonical_manifest_location(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = n_dir / "2p_checkpoint.pkl"
    manifest = _manifest_df()
    legacy_manifest = n_dir / runner.STRATEGY_MANIFEST_NAME
    runner.write_parquet_atomic(
        runner.pa.Table.from_pandas(manifest, preserve_index=False), legacy_manifest
    )
    ckpt_path.write_bytes(
        pickle.dumps(
            {
                "meta": {
                    "n_players": 2,
                    "num_shuffles": 2,
                    "global_seed": cfg.sim.seed,
                    "n_strategies": len(manifest),
                    "strategy_manifest_sha": runner._strategy_manifest_digest(manifest),
                    "rng_scheme_version": runner.urandom.RNG_SCHEME_VERSION,
                    "rng_bit_generator": "PCG64DXSM",
                    **_workload_meta(cfg, manifest),
                }
            }
        )
    )

    with pytest.raises(ValueError, match="Canonical strategy manifest missing"):
        runner._validate_resume_outputs(
            cfg=cfg,
            n_players=2,
            n_shuffles=2,
            strategies_manifest=manifest,
            ckpt_path=ckpt_path,
            row_dir=None,
            metric_chunk_dir=None,
        )
    assert not cfg.strategy_manifest_root_path().exists()


@pytest.mark.parametrize("meta", [None, "bad"])
def test_validate_resume_outputs_checkpoint_meta_missing_or_non_mapping(
    tmp_path: Path, meta: str | None
) -> None:
    cfg = _cfg(tmp_path)
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = n_dir / "2p_checkpoint.pkl"
    _write_root_strategy_manifest(cfg, _manifest_df())
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
        ("rng_scheme_version", 1),
        ("outcome_schema_version", 999),
        ("tournament_method_version", 999),
        ("rng_bit_generator", "wrong"),
        ("workload_plan_version", 999),
    ],
)
def test_validate_resume_outputs_checkpoint_meta_mismatch_keys(
    tmp_path: Path,
    bad_key: str,
    bad_value: int | str,
) -> None:
    cfg = _cfg(tmp_path)
    manifest = _manifest_df()
    _write_root_strategy_manifest(cfg, manifest)
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = n_dir / "2p_checkpoint.pkl"
    meta = {
        "n_players": 2,
        "num_shuffles": 2,
        "global_seed": cfg.sim.seed,
        "n_strategies": len(manifest),
        "strategy_manifest_sha": runner._strategy_manifest_digest(manifest),
        "rng_scheme_version": runner.urandom.RNG_SCHEME_VERSION,
        "outcome_schema_version": runner.OUTCOME_SCHEMA_VERSION,
        "tournament_method_version": runner.TOURNAMENT_METHOD_VERSION,
        "rng_bit_generator": "PCG64DXSM",
        **_workload_meta(cfg, manifest),
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


def test_validate_resume_outputs_accepts_canonical_manifest_without_checkpoint(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    n_dir = cfg.results_root / "2_players"
    n_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = n_dir / "2p_checkpoint.pkl"
    manifest = _manifest_df()
    _write_root_strategy_manifest(cfg, manifest)
    runner._validate_resume_outputs(
        cfg=cfg,
        n_players=2,
        n_shuffles=2,
        strategies_manifest=manifest,
        ckpt_path=ckpt_path,
        row_dir=None,
        metric_chunk_dir=None,
    )


@pytest.mark.parametrize(
    "case,match",
    [
        ("duplicate", "Duplicate shuffle"),
        ("seed", "Row manifest mismatch"),
        ("players", "Row manifest mismatch"),
        ("outcome", "Row manifest mismatch"),
    ],
)
def test_validate_resume_outputs_row_manifest_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    match: str,
) -> None:
    cfg = _cfg(tmp_path)
    _write_root_strategy_manifest(cfg, _manifest_df())
    n_dir = cfg.results_root / "2_players"
    row_dir = n_dir / "rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    valid = _row_manifest_record(cfg, 0)
    rows = {
        "duplicate": [valid, valid],
        "seed": [_row_manifest_record(cfg, 0, shuffle_seed=999)],
        "players": [_row_manifest_record(cfg, 0, n_players=9)],
        "outcome": [{**valid, "outcome_schema_version": 1}],
    }[case]
    _write_manifest_lines(row_dir / "manifest.jsonl", rows)
    monkeypatch.setattr(
        runner.urandom,
        "coordinate_seed",
        lambda _purpose, *, shuffle_index, **_kwargs: shuffle_index + 1,
    )

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
    "case,match",
    [
        ("duplicate", "Duplicate process-block"),
        ("players", "Metrics manifest mismatch"),
        ("method", "Metrics manifest mismatch"),
    ],
)
def test_validate_resume_outputs_metrics_manifest_invalid(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    cfg = _cfg(tmp_path)
    _write_root_strategy_manifest(cfg, _manifest_df())
    n_dir = cfg.results_root / "2_players"
    metric_dir = n_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    valid = _metric_manifest_record(cfg, 0)
    rows = {
        "duplicate": [valid, valid],
        "players": [_metric_manifest_record(cfg, 0, n_players=9)],
        "method": [{**valid, "tournament_method_version": 1}],
    }[case]
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
    _write_root_strategy_manifest(cfg, _manifest_df())
    n_dir = cfg.results_root / "2_players"
    row_dir = n_dir / "rows"
    metric_dir = n_dir / "metrics"
    row_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest_lines(
        row_dir / "manifest.jsonl",
        [_row_manifest_record(cfg, 0), _row_manifest_record(cfg, 1)],
    )
    _write_manifest_lines(
        metric_dir / "metrics_manifest.jsonl",
        [_metric_manifest_record(cfg, 0), _metric_manifest_record(cfg, 1)],
    )
    monkeypatch.setattr(
        runner.urandom,
        "coordinate_seed",
        lambda _purpose, *, shuffle_index, **_kwargs: shuffle_index + 1,
    )

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
    with pytest.raises(ValueError, match="sim.n_players_list is empty"):
        runner.run_tournament(cfg_empty)

    cfg_invalid = _cfg(tmp_path, n_players_list=[2, 7, 11])
    with pytest.raises(ValueError, match=r"\[7, 11\].*5,160 strategies"):
        runner.run_tournament(cfg_invalid)

    cfg_multi = _cfg(tmp_path, n_players_list=[2, 4])
    monkeypatch.setattr(runner, "run_multi", lambda cfg, player_counts, force=False: {2: 4, 4: 8})
    assert runner.run_tournament(cfg_multi, force=True) == 12


def test_run_single_n_empty_rows_and_metric_sq_sum_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _cfg(tmp_path, expanded_metrics=True, num_shuffles=2)
    strategies = [ThresholdStrategy(300, 3), ThresholdStrategy(500, 2)]

    monkeypatch.setattr(runner, "_resolve_strategies", lambda cfg, s: (strategies, 2, True))
    monkeypatch.setattr(
        runner,
        "build_strategy_manifest",
        lambda _strategies: pd.DataFrame(
            [
                {"strategy_id": 0, "strategy_str": "s0"},
                {"strategy_id": 1, "strategy_str": "s1"},
            ]
        ),
    )

    payloads = [
        {"win_totals": {0: -1}, "meta": {}},
        {
            "win_totals": {0: 2},
            "outcome_counts": {
                "games_attempted": 2,
                "games_completed": 2,
                "games_safety_limit": 0,
                "attempted_exposures": {0: 2, 1: 2},
                "completed_exposures": {0: 2, 1: 2},
                "safety_limit_exposures": {},
            },
            "metric_sums": {"winning_score": {0: 8.0}},
            "metric_sq_sums": {"winning_score": {0: 40.0}},
            "meta": {},
        },
    ]

    def fake_worker(**kwargs: Any) -> None:  # noqa: ANN001
        meta_sha = kwargs["checkpoint_metadata"]["strategy_manifest_sha"]
        payload = payloads.pop(0)
        payload["meta"] = {
            "n_players": kwargs["config"].n_players,
            "num_shuffles": kwargs["num_shuffles"],
            "global_seed": kwargs["global_seed"],
            "n_strategies": len(kwargs["strategies"]),
            **kwargs["checkpoint_metadata"],
            "strategy_manifest_sha": meta_sha,
        }
        _write_fake_checkpoint(kwargs, payload)

    captured_outputs: list[list[Path]] = []

    def fake_done(
        cfg: AppConfig,
        n_players: int,
        *,
        num_shuffles: int,
        shuffles_per_batch: int,
        n_strategies: int,
        outputs: list[Path],
        allow_unsealed_v3_outputs: bool = False,
        rewritten_outputs: tuple[Path, ...] = (),
    ) -> Path:
        assert allow_unsealed_v3_outputs
        assert rewritten_outputs
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
    assert all(
        p.name.endswith("checkpoint.pkl")
        or p.name in {runner.STRATEGY_MANIFEST_NAME, "simulation_workload_plan.json"}
        for p in captured_outputs[0]
    )

    runner.run_single_n(cfg, 2, force=True)
    metrics = pd.read_parquet(n_dir / "2p_metrics.parquet")
    assert metrics["sq_sum_winning_score"].to_numpy(dtype=float)[0] == 40.0
