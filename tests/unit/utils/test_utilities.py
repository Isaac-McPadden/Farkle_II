import pytest
import yaml

pytest.importorskip("pydantic")
pytest.importorskip("pyarrow")

from farkle.cli import main as cli_main
from farkle.utils.stats import games_for_power


def test_games_for_power_monotonic():
    n_small_delta = games_for_power(n_strategies=2, delta=0.05)
    n_large_delta = games_for_power(n_strategies=2, delta=0.10)
    # Larger effect size ??' fewer games required
    assert n_large_delta < n_small_delta


def test_cli_run(tmp_path, monkeypatch, capsys):
    called: dict[str, object] = {}

    monkeypatch.setattr(cli_main, "run_tournament", lambda **kw: called.update(kw))

    cfg = {
        "global_seed": 42,
        "n_players": 2,
        "num_shuffles": 1,
        "checkpoint_path": str(tmp_path / "checkpoint.pkl"),
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cli_main.main(["--config", str(cfg_path), "run"])

    assert called["global_seed"] == 42
    assert called["n_players"] == 2
    assert called["num_shuffles"] == 1
    assert called["checkpoint_path"] == str(tmp_path / "checkpoint.pkl")

    stderr = capsys.readouterr().err
    assert "CLI arguments parsed" in stderr
    assert "Dispatching run_tournament" in stderr


@pytest.mark.parametrize(
    "method,full_pairwise",
    [("bh", True), ("bonferroni", True), ("bonferroni", False)],
)
def test_games_for_power_branches(method, full_pairwise):
    n = games_for_power(n_strategies=3, method=method, full_pairwise=full_pairwise)
    assert n > 0


def test_games_for_power_pairwise_deprecated():
    with pytest.warns(DeprecationWarning):
        a = games_for_power(n_strategies=3, pairwise=False)
    b = games_for_power(n_strategies=3, full_pairwise=False)
    assert a == b


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
    monkeypatch.setattr(cli_main, "run_tournament", lambda **kw: called.update(kw))

    cli_main.main(["--config", str(cfg), "run"])

    assert called == {}


def test_load_config_missing_file(tmp_path):
    cfg_path = tmp_path / "missing.yml"
    with pytest.raises(FileNotFoundError):
        cli_main.load_config(str(cfg_path))


def test_load_config_bad_yaml(tmp_path):
    cfg_path = tmp_path / "bad.yml"
    cfg_path.write_text("strategy_grid: [")
    with pytest.raises(yaml.YAMLError):
        cli_main.load_config(str(cfg_path))


def test_load_config_missing_keys(tmp_path):
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump({"strategy_grid": {}}))
    cfg = cli_main.load_config(str(cfg_path))
    assert cfg == {"strategy_grid": {}}
