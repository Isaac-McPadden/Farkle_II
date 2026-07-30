"""Raw-derived tournament and real H2H analysis oracle through agreement."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.helpers.raw_simulation_oracle import (
    load_tiny_oracle_config,
    oracle_game_profile,
    run_analysis_through_candidate_freeze,
    run_pair_h2h_through_agreement,
    run_raw_simulation_roots,
)
from tests.helpers.tournament_analysis_oracle import (
    assert_authenticated_analysis_graph,
    assert_model_and_family_determinism,
    assert_pair_candidate_oracle,
    assert_pair_h2h_oracle,
    assert_root_pipeline_oracle,
)


@pytest.mark.integration
def test_raw_tournament_and_h2h_analysis_through_agreement(tmp_path: Path) -> None:
    """Exercise production analysis without manufacturing an intermediate table."""

    _config_path, cfg = load_tiny_oracle_config(tmp_path)
    profile = oracle_game_profile()
    contexts = run_raw_simulation_roots(cfg, profile)
    pair_context = run_analysis_through_candidate_freeze(cfg, contexts, profile)
    assert_root_pipeline_oracle(contexts)
    assert_pair_candidate_oracle(pair_context)
    assert_model_and_family_determinism(contexts, pair_context)
    frozen = run_pair_h2h_through_agreement(pair_context, profile)
    assert_pair_h2h_oracle(pair_context, frozen, profile)
    assert_authenticated_analysis_graph(
        contexts,
        pair_context,
        profile_sha256=profile.sha256,
    )
