import pytest
import yaml
import datetime as _datetime

pytest.importorskip("pydantic")
pytest.importorskip("pyarrow")

if not hasattr(_datetime, "UTC"):
    _datetime.UTC = _datetime.timezone.utc  # type: ignore[attr-defined]

from farkle.cli import main as cli_main
from farkle.config import AppConfig, load_app_config
from farkle.utils.stats import games_for_power


def test_games_for_power_monotonic():
    n_small_delta = games_for_power(n_strategies=2, detectable_lift=0.05)
    n_large_delta = games_for_power(n_strategies=2, detectable_lift=0.10)
    # Larger effect size ??' fewer games required
    assert n_large_delta < n_small_delta


def test_cli_run(tmp_path, monkeypatch, capsys):
    called: dict[str, object] = {}

    def fake_run_single_n(cfg: AppConfig, n: int) -> None:
        called["cfg"] = cfg
        called["n"] = n

    monkeypatch.setattr(cli_main.runner, "run_single_n", fake_run_single_n)

    cfg = {
        "io": {"results_dir": str(tmp_path / "out"), "append_seed": False},
        "sim": {
            "seed": 42,
            "n_players_list": [2],
            "num_shuffles": 1,
            "recompute_num_shuffles": False,
        },
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cli_main.main(["--config", str(cfg_path), "run"])

    assert called["n"] == 2
    cfg_obj = called["cfg"]
    assert isinstance(cfg_obj, AppConfig)
    assert cfg_obj.sim.seed == 42
    assert cfg_obj.sim.num_shuffles == 1
    assert cfg_obj.sim.n_players_list == [2]
    assert cfg_obj.io.results_dir == tmp_path / "out"

    stderr = capsys.readouterr().err
    assert "CLI arguments parsed" in stderr
    assert "Dispatching run command" in stderr


@pytest.mark.parametrize(
    ("method", "full_pairwise", "endpoint"),
    [
        ("bh", True, "pairwise"),
        ("bonferroni", True, "pairwise"),
        ("bonferroni", False, "pairwise"),
    ],
)
def test_games_for_power_branches(method, full_pairwise, endpoint):
    n = games_for_power(
        n_strategies=3,
        method=method,
        full_pairwise=full_pairwise,
        endpoint=endpoint,
    )
    assert n > 0


def test_games_for_power_pairwise_requires_more_games():
    baseline = games_for_power(n_strategies=3, endpoint="pairwise", full_pairwise=False)
    expanded = games_for_power(n_strategies=3, endpoint="pairwise", full_pairwise=True)
    assert expanded >= baseline


def test_cli_missing_file():
    bad = "nope.yml"
    with pytest.raises(FileNotFoundError):
        cli_main.main(["--config", bad, "run"])


def test_cli_bad_yaml(tmp_path):
    cfg = tmp_path / "bad.yml"
    cfg.write_text("{:")  # invalid YAML
    with pytest.raises(yaml.YAMLError):
        cli_main.main(["--config", str(cfg), "run"])


def test_cli_missing_keys(tmp_path, monkeypatch):
    cfg = tmp_path / "missing.yml"
    cfg.write_text(yaml.safe_dump({}))
    called: dict[str, object] = {}

    def fake_run_single_n(cfg_obj: AppConfig, n: int) -> None:
        called["cfg"] = cfg_obj
        called["n"] = n

    monkeypatch.setattr(cli_main.runner, "run_single_n", fake_run_single_n)

    cli_main.main(["--config", str(cfg), "run"])

    assert called["n"] == 5
    cfg_obj = called["cfg"]
    assert isinstance(cfg_obj, AppConfig)
    assert cfg_obj.sim.n_players_list == [5]
    assert str(cfg_obj.io.results_dir).endswith("results_seed_0")


def test_load_config_missing_file(tmp_path):
    cfg_path = tmp_path / "missing.yml"
    with pytest.raises(FileNotFoundError):
        load_app_config(cfg_path)


def test_load_config_bad_yaml(tmp_path):
    cfg_path = tmp_path / "bad.yml"
    cfg_path.write_text("strategy_grid: [")
    with pytest.raises(yaml.YAMLError):
        load_app_config(cfg_path)


def test_load_config_missing_keys(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "sim": {
                    "n_players": 3,
                    "collect_metrics": True,
                }
            }
        )
    )
    cfg = load_app_config(cfg_path)
    assert isinstance(cfg, AppConfig)
    assert cfg.sim.n_players_list == [3]
    assert cfg.sim.expanded_metrics is True
