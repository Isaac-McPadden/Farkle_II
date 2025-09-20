from __future__ import annotations

import logging

import pytest
import yaml

pytest.importorskip("pydantic")
pytest.importorskip("pyarrow")

import farkle.cli.main as cli_main
from farkle.analysis.analysis_config import PipelineCfg


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


def test_main_dispatches_run(monkeypatch, tmp_path, preserve_root_logger):
    recorded: dict[str, object] = {}

    def fake_run_tournament(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(cli_main, "run_tournament", fake_run_tournament)

    cli_main.main([
        "--log-level",
        "DEBUG",
        "run",
        "--metrics",
        "--row-dir",
        str(tmp_path),
    ])

    assert recorded["collect_metrics"] is True
    assert recorded["row_output_directory"] == tmp_path
    assert logging.getLogger().level == logging.DEBUG


def test_main_dispatches_time(monkeypatch, preserve_root_logger):
    called = False

    def fake_measure_sim_times():
        nonlocal called
        called = True

    monkeypatch.setattr(cli_main, "measure_sim_times", fake_measure_sim_times)

    cli_main.main(["time"])

    assert called is True


def test_main_dispatches_watch(monkeypatch, preserve_root_logger):
    captured: dict[str, object] = {}

    def fake_watch_game(*, seed):
        captured["seed"] = seed

    monkeypatch.setattr(cli_main, "watch_game", fake_watch_game)

    cli_main.main(["watch", "--seed", "123"])

    assert captured == {"seed": 123}


@pytest.mark.parametrize(
    "subcommand, expected_order",
    [
        ("ingest", ["ingest"]),
        ("curate", ["curate"]),
        ("combine", ["combine"]),
        ("metrics", ["metrics"]),
        ("pipeline", ["ingest", "curate", "combine", "metrics"]),
    ],
)
def test_main_dispatches_analyze_variants(monkeypatch, subcommand, expected_order, preserve_root_logger):
    calls: list[tuple[str, PipelineCfg]] = []

    def make_recorder(name: str):
        def _recorder(cfg: PipelineCfg) -> None:
            calls.append((name, cfg))

        return _recorder

    for name in ("ingest", "curate", "combine", "metrics"):
        module = getattr(cli_main, name)
        monkeypatch.setattr(module, "run", make_recorder(name))

    cli_main.main(["analyze", subcommand])

    assert [name for name, _ in calls] == expected_order
    assert all(isinstance(cfg, PipelineCfg) for _, cfg in calls)


def test_apply_override_creates_nested_keys():
    cfg: dict[str, object] = {}
    cli_main._apply_override(cfg, "sim.n_players=6")
    cli_main._apply_override(cfg, "sim.options.collect_metrics=true")
    cli_main._apply_override(cfg, "analysis.log_level=debug")

    assert cfg == {
        "sim": {"n_players": 6, "options": {"collect_metrics": True}},
        "analysis": {"log_level": "debug"},
    }


def test_load_config_with_overrides(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "global_seed": 7,
                "nested": {"value": 1},
            }
        ),
        encoding="utf-8",
    )

    cfg = cli_main.load_config(cfg_path, overrides=["nested.extra=2", "new.option='text'"])

    assert cfg["global_seed"] == 7
    assert cfg["nested"] == {"value": 1, "extra": 2}
    assert cfg["new"] == {"option": "text"}


def test_load_config_rejects_non_mapping(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")

    with pytest.raises(TypeError):
        cli_main.load_config(cfg_path)


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
