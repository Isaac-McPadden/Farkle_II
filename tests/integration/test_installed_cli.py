from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml


def _installed_farkle_entrypoint() -> Path:
    name = "farkle.exe" if os.name == "nt" else "farkle"
    entrypoint = Path(sys.executable).with_name(name)
    assert entrypoint.is_file(), f"installed farkle entry point not found at {entrypoint}"
    return entrypoint


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    results_prefix = tmp_path / "results"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "io": {"results_dir_prefix": str(results_prefix)},
                "sim": {"n_players_list": [2], "seed": 7, "seed_list": [7, 8]},
            }
        ),
        encoding="utf-8",
    )
    return config_path, results_prefix


def _run_installed_cli(
    tmp_path: Path,
    args: list[str],
    *,
    intercept_pipeline: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path | None]:
    env = os.environ.copy()
    capture_path: Path | None = None
    if intercept_pipeline:
        capture_path = tmp_path / "pipeline_call.json"
        hook_dir = tmp_path / "python_hook"
        hook_dir.mkdir()
        (hook_dir / "sitecustomize.py").write_text(
            textwrap.dedent(
                """
                import json
                import os
                from pathlib import Path

                from farkle.orchestration import two_seed_pipeline
                from farkle.orchestration.seed_utils import seed_pair_root


                def _capture_pipeline(cfg, *, seed_pair, force=False):
                    payload = {
                        "seed_pair": list(seed_pair),
                        "configured_seed_list": list(cfg.sim.seed_list),
                        "pair_root": str(seed_pair_root(cfg, seed_pair)),
                        "force": force,
                    }
                    Path(os.environ["FARKLE_CLI_CAPTURE"]).write_text(
                        json.dumps(payload, sort_keys=True), encoding="utf-8"
                    )


                two_seed_pipeline.run_pipeline = _capture_pipeline
                """
            ),
            encoding="utf-8",
        )
        env["FARKLE_CLI_CAPTURE"] = str(capture_path)
        prior_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(hook_dir)
            if not prior_pythonpath
            else os.pathsep.join((str(hook_dir), prior_pythonpath))
        )

    completed = subprocess.run(
        [str(_installed_farkle_entrypoint()), *args],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, capture_path


def test_installed_entrypoint_resolves_documented_post_subcommand_seed_pair(
    tmp_path: Path,
) -> None:
    config_path, results_prefix = _write_config(tmp_path)

    completed, capture_path = _run_installed_cli(
        tmp_path,
        [
            "--config",
            str(config_path),
            "two-seed-pipeline",
            "--seed-pair",
            "42",
            "43",
        ],
        intercept_pipeline=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert capture_path is not None
    assert json.loads(capture_path.read_text(encoding="utf-8")) == {
        "configured_seed_list": [42, 43],
        "force": False,
        "pair_root": str(tmp_path / "results_seed_pair_42_43"),
        "seed_pair": [42, 43],
    }
    assert not (tmp_path / "results_seed_pair_7_8").exists()
    assert not results_prefix.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(["two-seed-pipeline", "--definitely-unknown"], id="unknown-option"),
        pytest.param(["two-seed-pipeline", "--seed-pai", "42", "43"], id="misspelled-option"),
        pytest.param(["two-seed-pipeline", "--seed-pair", "42"], id="missing-value"),
        pytest.param(
            ["--seed-pair", "42", "43", "two-seed-pipeline"],
            id="seed-options-are-command-owned",
        ),
    ],
)
def test_installed_entrypoint_parse_failure_creates_no_results(
    tmp_path: Path,
    arguments: list[str],
) -> None:
    config_path, results_prefix = _write_config(tmp_path)

    completed, _ = _run_installed_cli(
        tmp_path,
        ["--config", str(config_path), *arguments],
    )

    assert completed.returncode != 0
    assert "usage:" in completed.stderr.lower()
    assert not results_prefix.exists()
    assert list(tmp_path.glob("results_seed*")) == []


def test_installed_entrypoint_help_places_seed_options_on_command(tmp_path: Path) -> None:
    command_help, _ = _run_installed_cli(tmp_path, ["two-seed-pipeline", "--help"])
    global_help, _ = _run_installed_cli(tmp_path, ["--help"])

    assert command_help.returncode == 0
    assert "--seed-pair A B" in command_help.stdout
    assert "--seed-a SEED_A" in command_help.stdout
    assert global_help.returncode == 0
    assert "--seed-pair" not in global_help.stdout
