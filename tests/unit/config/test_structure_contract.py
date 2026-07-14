from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from farkle.config import AppConfig, ArtifactScope, IOConfig, load_app_config


@pytest.mark.parametrize(
    "filename",
    ["default_config.yaml", "farkle_mega_config.yaml", "fast_config.yaml"],
)
def test_active_configuration_examples_use_the_current_contract(filename: str) -> None:
    config_path = Path(__file__).resolve().parents[3] / "configs" / filename
    load_app_config(config_path)


@pytest.mark.parametrize(
    ("payload", "retired_key"),
    [
        ({"sim": {"n_players": 2}}, "sim.n_players"),
        ({"io": {"analysis_dir": "analysis"}}, "io.analysis_dir"),
        ({"metrics": {"enabled": True}}, "metrics"),
        ({"head2head": {"games_per_pair": 10}}, "head2head.games_per_pair"),
    ],
)
def test_retired_configuration_keys_fail_with_named_replacements(
    tmp_path: Path,
    payload: dict[str, object],
    retired_key: str,
) -> None:
    path = tmp_path / "retired.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=rf"Retired config (?:key|section) '{retired_key}'",
    ):
        load_app_config(path)


def test_canonical_scope_paths_are_disjoint_and_removed_helpers_stay_absent(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(io=IOConfig(results_dir_prefix=tmp_path))
    paths = {
        cfg.scope_dir("metrics", ArtifactScope.BY_K, k=2),
        cfg.scope_dir("metrics", ArtifactScope.CONCAT_KS),
        cfg.scope_dir("metrics", ArtifactScope.ACROSS_K),
        cfg.scope_dir("metrics", ArtifactScope.CROSS_SEED),
        cfg.scope_dir("metrics", ArtifactScope.DIAGNOSTICS),
        cfg.scope_dir("metrics", ArtifactScope.H2H_2P),
    }
    assert len(paths) == 6
    assert not hasattr(cfg, "combine_combined_dir")
    assert not hasattr(cfg, "resolve_input_stage_dir")
    assert not hasattr(cfg, "meta_analysis_dir")
