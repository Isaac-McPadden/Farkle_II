"""Additional branch-focused tests for the analysis pipeline CLI."""

from __future__ import annotations

import argparse
import logging
import types
from pathlib import Path
from typing import Any, cast

import pytest

from farkle.analysis import pipeline
from farkle.config import AppConfig, IOConfig, _stringify_paths_for_serialization


def _make_config(tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, AppConfig]:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_results_dir))
    cfg_path = tmp_results_dir / "configs/fast_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("io:\n  results_dir_prefix: dummy\n")

    def _load_app_config(path: Path) -> AppConfig:
        assert path == cfg_path
        return cfg

    monkeypatch.setattr("farkle.analysis.pipeline.load_app_config", _load_app_config, raising=True)
    monkeypatch.setattr("farkle.config.load_app_config", _load_app_config, raising=False)
    return cfg_path, cfg


def test_cli_errors_when_seed_pair_and_single_seed_flags_combined(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        pipeline.main(["--seed-pair", "1", "2", "--seed-a", "9", "ingest"])

    assert "Use --seed-pair or --seed-a/--seed-b, not both." in capsys.readouterr().err


def test_cli_errors_when_only_one_seed_from_pair_is_provided(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        pipeline.main(["--seed-a", "9", "ingest"])

    assert "--seed-a and --seed-b must be provided together." in capsys.readouterr().err


def test_main_falls_back_to_parse_args_when_parse_intermixed_unavailable(
    tmp_results_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path, _ = _make_config(tmp_results_dir, monkeypatch)
    called = {"intermixed": 0}

    def _raise_type_error(self: argparse.ArgumentParser, args=None, namespace=None):  # noqa: ANN001
        called["intermixed"] += 1
        raise TypeError("fallback")

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_intermixed_args",
        _raise_type_error,
        raising=True,
    )
    monkeypatch.setattr("farkle.analysis.pipeline.StageRunner.run", lambda *args, **kwargs: None)

    rc = pipeline.main(["--config", str(cfg_path), "ingest"])

    assert rc == 0
    assert called["intermixed"] == 1


@pytest.mark.parametrize(
    ("seed_args", "expected_pair"),
    [(["--seed-pair", "11", "22"], (11, 22)), (["--seed-a", "33", "--seed-b", "44"], (33, 44))],
)
def test_main_applies_seed_and_analysis_overrides(
    tmp_results_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed_args: list[str],
    expected_pair: tuple[int, int],
) -> None:
    cfg_path, cfg = _make_config(tmp_results_dir, monkeypatch)
    monkeypatch.setattr("farkle.analysis.pipeline.StageRunner.run", lambda *args, **kwargs: None)

    rc = pipeline.main(
        [
            "--config",
            str(cfg_path),
            *seed_args,
            "--rare-event-margin-quantile",
            "0.9",
            "--rare-event-target-rate",
            "0.01",
            "--disable-rng-diagnostics",
            "ingest",
        ]
    )

    assert rc == 0
    assert cfg.sim.seed_pair == expected_pair
    assert cfg.sim.seed_list == list(expected_pair)
    assert cfg.analysis.rare_event_margin_quantile == 0.9
    assert cfg.analysis.rare_event_target_rate == 0.01
    assert cfg.analysis.disable_rng_diagnostics is True


def test_main_logs_deprecated_flag_warnings(
    tmp_results_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg_path, _ = _make_config(tmp_results_dir, monkeypatch)
    monkeypatch.setattr("farkle.analysis.pipeline.StageRunner.run", lambda *args, **kwargs: None)

    caplog.set_level(logging.WARNING)
    rc = pipeline.main(
        [
            "--config",
            str(cfg_path),
            "--disable-trueskill",
            "--disable-head2head",
            "--interseed",
            "ingest",
        ]
    )

    assert rc == 0
    records = [
        rec
        for rec in caplog.records
        if rec.getMessage() == "Deprecated CLI flag ignored; stages now run based on inputs"
    ]
    assert {cast(Any, rec).flag for rec in records} == {
        "disable_trueskill",
        "disable_head2head",
        "run_interseed",
    }


def test_interseed_summary_skips_rng_stage_when_disabled(
    tmp_results_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path, _ = _make_config(tmp_results_dir, monkeypatch)
    calls: list[tuple[str, object]] = []

    def _optional_import(module: str, *, stage_log=None):  # noqa: ANN001
        if module == "farkle.analysis.interseed_analysis":
            return types.SimpleNamespace(
                run=lambda cfg, **kwargs: calls.append(("interseed", kwargs["run_rng_diagnostics"]))
            )
        if module == "farkle.analysis.rng_diagnostics":
            return types.SimpleNamespace(
                run=lambda cfg, **kwargs: calls.append(("rng", kwargs.get("lags")))
            )
        return None

    monkeypatch.setattr("farkle.analysis._optional_import", _optional_import, raising=True)

    def _run_only_interseed(plan, context, **_kwargs):  # noqa: ANN001
        next(item for item in plan if item.name == "interseed").action(context.config)

    monkeypatch.setattr("farkle.analysis.pipeline.StageRunner.run", _run_only_interseed)

    rc = pipeline.main(["--config", str(cfg_path), "--no-rng-diagnostics", "analytics"])

    assert rc == 0
    assert calls == [("interseed", False)]


def test_interseed_summary_runs_rng_stage_and_uses_sorted_unique_lags(
    tmp_results_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path, _ = _make_config(tmp_results_dir, monkeypatch)
    calls: list[tuple[str, object]] = []

    def _optional_import(module: str, *, stage_log=None):  # noqa: ANN001
        if module == "farkle.analysis.head2head":
            calls.append(("head2head_import", None))
            return None
        if module == "farkle.analysis.h2h_analysis":
            calls.append(("post_h2h_import", None))
            return None
        if module == "farkle.analysis.interseed_analysis":
            return types.SimpleNamespace(
                run=lambda cfg, **kwargs: calls.append(("interseed", kwargs["run_rng_diagnostics"]))
            )
        if module == "farkle.analysis.rng_diagnostics":
            return types.SimpleNamespace(
                run=lambda cfg, **kwargs: calls.append(("rng", kwargs.get("lags")))
            )
        return None

    monkeypatch.setattr("farkle.analysis._optional_import", _optional_import, raising=True)

    def _run_targeted_steps(plan, context, **_kwargs):  # noqa: ANN001
        for name in ("head2head", "post_h2h", "interseed"):
            next(item for item in plan if item.name == name).action(context.config)

    monkeypatch.setattr("farkle.analysis.pipeline.StageRunner.run", _run_targeted_steps)

    rc = pipeline.main(
        [
            "--config",
            str(cfg_path),
            "--rng-diagnostics",
            "--rng-lags",
            "5",
            "--rng-lags",
            "1",
            "--rng-lags",
            "5",
            "analytics",
        ]
    )

    assert rc == 0
    assert ("head2head_import", None) in calls
    assert ("post_h2h_import", None) in calls
    assert ("interseed", True) in calls
    assert ("rng", (1, 5)) in calls


def test_interseed_layout_is_restored_when_rng_stage_raises(
    tmp_results_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path, cfg = _make_config(tmp_results_dir, monkeypatch)

    def _optional_import(module: str, *, stage_log=None):  # noqa: ANN001
        if module == "farkle.analysis.interseed_analysis":
            return types.SimpleNamespace(run=lambda cfg, **kwargs: None)
        if module == "farkle.analysis.rng_diagnostics":
            return types.SimpleNamespace(
                run=lambda cfg, **kwargs: (_ for _ in ()).throw(RuntimeError("rng boom"))
            )
        return None

    monkeypatch.setattr("farkle.analysis._optional_import", _optional_import, raising=True)
    before_after: dict[str, object] = {}

    def _run_only_interseed(plan, context, **_kwargs):  # noqa: ANN001
        before_after["before"] = context.config._stage_layout
        next(item for item in plan if item.name == "interseed").action(context.config)

    monkeypatch.setattr("farkle.analysis.pipeline.StageRunner.run", _run_only_interseed)

    with pytest.raises(RuntimeError, match="rng boom"):
        pipeline.main(["--config", str(cfg_path), "--rng-diagnostics", "analytics"])

    assert cfg._stage_layout is before_after["before"]


def test_seed_symmetry_stage_runs_after_head2head(
    tmp_results_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg_path, _ = _make_config(tmp_results_dir, monkeypatch)
    calls: list[str] = []

    def _optional_import(module: str, *, stage_log=None):  # noqa: ANN001
        if module == "farkle.analysis.head2head":
            return types.SimpleNamespace(run=lambda cfg: calls.append("head2head_run"))
        if module == "farkle.analysis.h2h_analysis":
            return types.SimpleNamespace(run_post_h2h=lambda cfg: None)
        if module == "farkle.analysis.hgb_feat":
            return types.SimpleNamespace(run=lambda cfg: None)
        if module == "farkle.analysis.tiering_report":
            return types.SimpleNamespace(run=lambda cfg: None)
        if module == "farkle.analysis.trueskill":
            return types.SimpleNamespace(run=lambda cfg: None)
        return None

    monkeypatch.setattr("farkle.analysis._optional_import", _optional_import, raising=True)
    monkeypatch.setattr(
        "farkle.analysis.run_seed_symmetry",
        lambda cfg, force=False: calls.append(f"seed_symmetry_run:{force}"),
        raising=True,
    )

    def _run_targeted_steps(plan, context, **_kwargs):  # noqa: ANN001
        for name in ("head2head", "seed_symmetry"):
            next(item for item in plan if item.name == name).action(context.config)

    monkeypatch.setattr("farkle.analysis.pipeline.StageRunner.run", _run_targeted_steps)

    rc = pipeline.main(["--config", str(cfg_path), "analytics"])

    assert rc == 0
    assert calls == ["head2head_run", "seed_symmetry_run:False"]


def test_stringify_paths_recurses_for_nested_collections() -> None:
    nested = {
        "path": Path("alpha"),
        "items": [Path("beta"), {"tuple": (Path("gamma"), 3)}],
        "plain": ("ok", {"deep": [Path("delta"), 7]}),
    }

    actual = _stringify_paths_for_serialization(nested)

    assert actual == {
        "path": "alpha",
        "items": ["beta", {"tuple": ("gamma", 3)}],
        "plain": ("ok", {"deep": ["delta", 7]}),
    }
