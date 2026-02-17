from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

pytest.importorskip("pydantic")
pytest.importorskip("pyarrow")

import farkle.cli.main as cli_main


@pytest.fixture(autouse=True)
def _no_setup_logging(monkeypatch):
    monkeypatch.setattr(cli_main, "setup_info_logging", lambda: None)


@pytest.fixture
def preserve_root_logger():
    root = logging.getLogger()
    level = root.level
    handlers = list(root.handlers)
    yield
    root.setLevel(level)
    root.handlers[:] = handlers


def test_main_dispatches_watch(monkeypatch, preserve_root_logger):
    captured: dict[str, object] = {}

    def fake_watch_game(*, seed):
        captured["seed"] = seed

    monkeypatch.setattr(cli_main, "watch_game", fake_watch_game)

    cli_main.main(["watch", "--seed", "123"])

    assert captured == {"seed": 123}


def test_parse_level_accepts_string(preserve_root_logger):
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    level = cli_main._parse_level("debug")
    root.setLevel(level)

    assert root.level == logging.DEBUG


def test_parse_level_accepts_int(preserve_root_logger):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    level = cli_main._parse_level(logging.ERROR)
    root.setLevel(level)

    assert root.level == logging.ERROR


def test_analyze_metrics_ignores_rng_flags(monkeypatch, preserve_root_logger):
    calls: list[str] = []

    monkeypatch.setattr(cli_main.metrics, "run", lambda cfg: calls.append("metrics"))
    monkeypatch.setattr(
        cli_main.analysis_pkg,
        "run_all",
        lambda cfg, **kwargs: calls.append("run_all"),
    )

    cli_main.main(
        ["analyze", "metrics", "--rng-diagnostics", "--rng-lags", "2", "1", "2"]
    )

    assert calls == ["metrics"]


def test_stringify_paths_handles_nested_structures():
    nested = {
        "path": Path("/tmp/example"),
        "items": [Path("a"), {"b": Path("c")}, (Path("d"),)],
    }

    result = cli_main._stringify_paths(nested)

    assert result == {
        "path": str(Path("/tmp/example")),
        "items": ["a", {"b": "c"}, ("d",)],
    }


def test_write_active_config_persists_yaml(tmp_path: Path, monkeypatch):
    cfg = cli_main.AppConfig()
    cfg.io.results_dir_prefix = tmp_path / "results"
    cfg.analysis.outputs = {}
    cfg.sim.row_dir = tmp_path / "rows"

    cli_main._write_active_config(cfg, tmp_path)

    written = tmp_path / "active_config.yaml"
    assert written.exists()
    data = yaml.safe_load(written.read_text())
    assert data["io"]["results_dir_prefix"].endswith("results")
    assert data["sim"]["row_dir"].endswith("rows")


def test_main_time_dispatches(monkeypatch, preserve_root_logger):
    captured: dict[str, object] = {}

    def fake_measure_sim_times(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "measure_sim_times", fake_measure_sim_times)

    cli_main.main(["time", "--players", "3", "--n-games", "10", "--jobs", "2", "--seed", "9"])

    assert captured == {"n_games": 10, "players": 3, "seed": 9, "jobs": 2}


def _write_cfg(tmp_path: Path) -> Path:
    cfg = {
        "io": {"results_dir_prefix": str(tmp_path / "out")},
        "sim": {"seed": 7, "n_players_list": [2], "num_shuffles": 1, "recompute_num_shuffles": False},
    }
    path = tmp_path / "cfg.yml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_analyze_variance_dispatch(monkeypatch, tmp_path: Path, preserve_root_logger):
    called: dict[str, object] = {}

    monkeypatch.setattr(cli_main.analysis_pkg, "run_variance", lambda cfg, *, force=False: called.update(force=force))

    cfg_path = _write_cfg(tmp_path)
    cli_main.main(["--config", str(cfg_path), "analyze", "variance", "--force"])

    assert called == {"force": True}


def test_analyze_pipeline_dispatches_preprocess_and_analytics(
    monkeypatch, tmp_path: Path, preserve_root_logger
):
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        cli_main,
        "_run_preprocess",
        lambda cfg, **kwargs: calls.append(("analyze:pipeline", kwargs.get("compute_game_stats"))),
    )
    monkeypatch.setattr(
        cli_main.analysis_pkg,
        "run_all",
        lambda cfg, **kwargs: calls.append(("analyze:analytics", kwargs)),
    )

    cfg_path = _write_cfg(tmp_path)
    cli_main.main(["--config", str(cfg_path), "analyze", "pipeline"])

    assert calls == [
        ("analyze:pipeline", False),
        ("analyze:analytics", {"run_rng_diagnostics": False, "rng_lags": None}),
    ]


@pytest.mark.parametrize(
    ("argv", "expected_call"),
    [
        (["analyze", "ingest"], "ingest"),
        (["analyze", "curate"], "curate"),
        (["analyze", "combine"], "combine"),
        (["analyze", "metrics"], "metrics"),
        (["analyze", "analytics"], "analytics"),
    ],
)
def test_analyze_subcommands_dispatch(monkeypatch, preserve_root_logger, argv, expected_call):
    calls: list[str] = []

    monkeypatch.setattr(cli_main.ingest, "run", lambda cfg: calls.append("ingest"))
    monkeypatch.setattr(cli_main.curate, "run", lambda cfg: calls.append("curate"))
    monkeypatch.setattr(cli_main.combine, "run", lambda cfg: calls.append("combine"))
    monkeypatch.setattr(cli_main.metrics, "run", lambda cfg: calls.append("metrics"))
    monkeypatch.setattr(
        cli_main.analysis_pkg,
        "run_all",
        lambda cfg, **kwargs: calls.append("analytics"),
    )

    cli_main.main(argv)

    assert calls == [expected_call]


@pytest.mark.parametrize(
    "argv",
    [
        ["--seed-pair", "1", "2", "--seed-a", "1", "--seed-b", "2", "run"],
        ["--seed-a", "1", "run"],
        ["--seed-b", "2", "run"],
    ],
)
def test_main_rejects_conflicting_seed_flags(preserve_root_logger, argv):
    with pytest.raises(SystemExit) as excinfo:
        cli_main.main(argv)

    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    ("argv", "attr", "method"),
    [
        (["run"], "runner", "run_single_n"),
        (["analyze", "ingest"], "ingest", "run"),
        (["analyze", "variance"], "analysis_pkg", "run_variance"),
        (["two-seed-pipeline", "--seed-pair", "3", "4"], "two_seed_pipeline", "run_pipeline"),
    ],
)
def test_main_propagates_delegated_failure_codes(
    monkeypatch, preserve_root_logger, argv, attr, method
):
    module_obj = getattr(cli_main, attr)

    def _raise(*args, **kwargs):
        raise SystemExit(9)

    monkeypatch.setattr(module_obj, method, _raise)

    with pytest.raises(SystemExit) as excinfo:
        cli_main.main(argv)

    assert excinfo.value.code == 9
