"""High-level assertions for the real raw-simulation-to-report workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from farkle import analysis
from farkle.analysis.release_audit import audit_sidecar_completeness
from farkle.analysis.trueskill_screening import TRUESKILL_CONDITIONING
from farkle.config import AppConfig
from farkle.orchestration.run_contexts import RootPairRunContext, SeedRunContext
from farkle.simulation import runner
from farkle.simulation.game_profile import GameProfile
from farkle.utils.artifact_contract import sha256_file, sidecar_path
from farkle.utils.stage_completion import CompletionState

ORCHESTRATION_MANIFEST = "two_seed_pipeline_manifest.jsonl"


def _assert_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    assert not missing, f"missing canonical oracle artifacts: {missing}"


def _root_artifacts(context: SeedRunContext) -> list[Path]:
    cfg = context.config
    paths = [cfg.strategy_manifest_root_path()]
    for k in (2, 4):
        row_dir = cfg.simulation_row_dir(k)
        assert row_dir is not None
        metric_dir = cfg.n_dir(k) / f"{k}p_{cfg.sim.metric_chunk_dir}"
        paths.extend(
            [
                cfg.n_dir(k) / "simulation_workload_plan.json",
                cfg.checkpoint_path(k),
                cfg.checkpoint_path(k).with_suffix(".parquet"),
                cfg.metrics_path(k),
                runner.simulation_done_path(cfg, k),
                row_dir / "manifest.jsonl",
                *sorted(row_dir.glob("rows_*.parquet")),
                metric_dir / "metrics_manifest.jsonl",
                *sorted(metric_dir.glob("metrics_*.parquet")),
                cfg.ingested_rows_raw(k),
                cfg.ingest_manifest(k),
                cfg.ingested_rows_curated(k),
                cfg.manifest_for(k),
                cfg.combined_rows_by_k(k),
                cfg.metrics_all_player_batch_path(k),
                cfg.performance_batch_matrix_path(k),
                cfg.performance_by_k_path(k),
                cfg.seat_batch_counts_path(k),
                cfg.seat_effects_by_k_path(k),
                cfg.seat_population_by_k_path(k),
                cfg.game_stats_stage_dir / "by_k" / f"{k}p" / f"game_stats.{k}p.parquet",
                cfg.by_k_dir("game_stats", k) / "rare_events.parquet",
                cfg.by_k_dir("game_stats", k) / "rare_events.stats.json",
                cfg.trueskill_rating_path(k, root_seed=context.seed),
                cfg.trueskill_rating_path(k, root_seed=context.seed).with_suffix(".json"),
                cfg.trueskill_rating_path(k, root_seed=context.seed).parent / "artifact.parquet",
                cfg.hgb_importance_path(k),
                cfg.hgb_predictive_scores_path(k),
                cfg.hgb_fold_metrics_path(k),
            ]
        )
    paths.extend(
        [
            cfg.curated_dataset,
            cfg.combined_manifest_path(),
            cfg.performance_across_k_path(),
            cfg.performance_bootstrap_path(),
            cfg.performance_control_contrasts_path(),
            cfg.performance_player_count_effects_path(),
            cfg.seat_standardized_across_k_path(),
            cfg.seat_exposure_mixture_diagnostic_path(),
            cfg.seat_selfplay_diagnostic_path(),
            cfg.seat_mirrored_diagnostic_path(),
            cfg.game_stats_concat_path("game_length.parquet"),
            cfg.game_stats_concat_path("margin_stats.parquet"),
            cfg.game_stats_output_path("game_length_strategy_conditioned_equal_k_mean.parquet"),
            cfg.game_stats_output_path("margin_strategy_conditioned_equal_k_mean.parquet"),
            cfg.game_stats_output_path("rare_events.parquet"),
            cfg.exact_roll_distribution_path(),
            cfg.exact_roll_summary_path(),
            cfg.trueskill_candidate_contribution_path(),
            cfg.trueskill_screening_diagnostics_path(),
            cfg.hgb_stage_dir / "concat_ks" / "heldout_feature_importance_concat.parquet",
            cfg.hgb_stage_dir / "across_k" / "feature_importance_overall.parquet",
            cfg.hgb_stage_dir / "across_k" / "hgb_importance.json",
            cfg.hgb_future_proposals_path(),
            cfg.screening_path(),
            cfg.screening_path("descriptive_screening.json"),
        ]
    )
    return paths


def _pair_artifacts(pair_context: RootPairRunContext) -> list[Path]:
    cfg = pair_context.config
    paths = [
        *(cfg.root_combined_performance_by_k_path(k) for k in (2, 4)),
        cfg.root_combined_performance_across_k_path(),
        cfg.root_discrepancies_path(),
        cfg.root_joint_discrepancy_path(),
        cfg.root_rank_stability_path(),
        cfg.root_top_n_stability_path(),
        cfg.root_bootstrap_top_n_inclusion_path(),
        cfg.root_control_movement_path(),
        cfg.root_shortlist_changes_path(),
        cfg.root_matched_count_convergence_path(),
        cfg.root_half_drift_path(),
        cfg.trueskill_candidate_contribution_path(),
        cfg.h2h_candidate_family_path(),
        cfg.h2h_candidate_family_manifest_path(),
        cfg.h2h_power_plan_path(),
        cfg.h2h_block_manifest_path(),
        cfg.h2h_execution_state_path(),
        cfg.h2h_order_counts_path(),
        cfg.h2h_combined_order_counts_path(),
        cfg.h2h_pairwise_inference_path(),
        cfg.h2h_root_pairwise_diagnostics_path(),
        cfg.h2h_root_agreement_path(),
        cfg.h2h_dominance_edges_path(),
        cfg.h2h_cycle_groups_path(),
        cfg.h2h_dominance_fronts_path(),
        cfg.h2h_dominance_summary_path(),
        cfg.structure_agreement_pairs_path(),
        cfg.structure_agreement_summary_path(),
        cfg.structure_report_json_path(),
        cfg.structure_report_markdown_path(),
        cfg.structure_report_plot_path(),
        cfg.migration_report_path(),
    ]
    paths.extend(
        cfg.h2h_block_result_path(pair_id, root, order)
        for pair_id in range(3)
        for root in (11, 22)
        for order in (0, 1)
    )
    return paths


def assert_canonical_artifact_inventory(
    contexts: tuple[SeedRunContext, SeedRunContext],
    pair_context: RootPairRunContext,
) -> None:
    """Require every named Task 14 artifact at its config-resolved path."""

    for context in contexts:
        paths = _root_artifacts(context)
        _assert_paths(paths)
        for k in (2, 4):
            row_dir = context.config.simulation_row_dir(k)
            assert row_dir is not None
            assert len(list(row_dir.glob("rows_*.parquet"))) == 2
        assert audit_sidecar_completeness(context.analysis_root) == []
    pair_paths = _pair_artifacts(pair_context)
    _assert_paths(pair_paths)
    assert len(list(pair_context.config.h2h_block_results_dir().glob("*.parquet"))) == 12
    assert audit_sidecar_completeness(pair_context.analysis_root) == []


def _assert_identity(identity: dict[str, Any], path: Path) -> None:
    assert path.exists()
    assert identity["kind"] == "file"
    assert identity["byte_length"] == path.stat().st_size
    assert identity["content_sha256"] == sha256_file(path)
    metadata_path = sidecar_path(path)
    assert identity["sidecar_sha256"] == (
        sha256_file(metadata_path) if metadata_path.exists() else None
    )


def assert_pipeline_health_and_simulation_lifecycle(
    cfg: AppConfig,
    contexts: tuple[SeedRunContext, SeedRunContext],
    pair_context: RootPairRunContext,
    profile: GameProfile,
) -> None:
    """Authenticate raw completions and the orchestrator's final health view."""

    health_path = pair_context.pair_root / "pipeline_health.json"
    health = json.loads(health_path.read_text(encoding="utf-8"))
    assert health["seed_pair"] == [11, 22]
    assert health["status"] == "complete_success"
    assert health["config_sha"] == cfg.config_sha
    assert health["pair_public_config_sha256"] == pair_context.config.config_sha
    release_audit = health["release_audit"]
    assert release_audit["status"] == "passed"
    assert release_audit["accepted_release_identity"] == [3, 2, 2, 2, 2, 2]
    assert release_audit["failures"] == []
    assert release_audit["release_eligible"] is False
    assert set(release_audit["run_contexts"]) == {"root_11", "root_22", "pair"}
    expected_root_states = {
        "simulation_2p",
        "simulation_4p",
        "ingest",
        "curate",
        "combine",
        "metrics",
        "game_stats",
        "trueskill",
        "hgb",
        "screening",
    }
    for context in contexts:
        root_health = health["root_workflows"][str(context.seed)]
        assert root_health["simulation"] == "complete"
        assert root_health["analysis"] == "complete"
        assert root_health["error"] is None
        assert root_health["lifecycle_sha256"]
        assert set(root_health["stage_states"]) == expected_root_states
        assert set(root_health["stage_states"].values()) == {CompletionState.COMPLETE_VALID.value}
        for k in (2, 4):
            assert runner.simulation_is_complete(context.config, k)
            stamp = json.loads(
                runner.simulation_done_path(context.config, k).read_text(encoding="utf-8")
            )
            assert stamp["lifecycle_contract_version"] == 1
            assert stamp["state"] == CompletionState.COMPLETE_VALID.value
            assert len(stamp["stage_identity_sha256"]) == 64
            assert stamp["outputs"]

    pair_health = health["pair_workflow"]
    assert pair_health["status"] == "complete"
    assert pair_health["error"] is None
    assert Path(pair_health["analysis_root"]) == pair_context.analysis_root
    assert set(pair_health["stage_states"]) == {
        item.name for item in analysis.build_root_pair_stage_plan(pair_context)
    }
    assert set(pair_health["stage_states"].values()) == {CompletionState.COMPLETE_VALID.value}


def assert_report_oracle(pair_context: RootPairRunContext) -> None:
    """Assert report counts, conditioning, visibility, and conservative claims."""

    cfg = pair_context.config
    report = json.loads(cfg.structure_report_json_path().read_text(encoding="utf-8"))
    assert report["report_contract_version"] == 5
    assert report["execution_scope"] == "root_pair"
    assert report["roots"] == [11, 22]
    assert report["support"]["player_counts"] == [2, 4]
    assert report["support"]["k_weights"] == {"2": 0.5, "4": 0.5}
    assert report["safety_limits"]["games_attempted"] == 12
    assert report["safety_limits"]["games_completed"] == 11
    assert report["safety_limits"]["games_safety_limit"] == 1
    assert report["conditioning"]["trueskill"] == TRUESKILL_CONDITIONING
    assert "safety-limit attempts are excluded from rating updates" in (
        report["conditioning"]["trueskill"]
    )
    assert report["candidate_family"]["candidate_count"] == 3
    assert report["h2h"]["role"] == "external_two_player_finalist_diagnostic"
    assert report["h2h"]["games_attempted"] == 14
    assert report["h2h"]["games_completed"] == 11
    assert report["h2h"]["games_safety_limit"] == 3
    assert report["h2h"]["replacement_attempt_count"] == 2
    assert report["h2h"]["unresolved_pair_count"] == 3
    assert report["h2h"]["unresolved_nonviable_pair_count"] == 3
    assert report["h2h"]["operationally_nonviable_candidates"] == ["0", "1", "3"]
    assert report["h2h"]["unique_best"] is None
    assert report["h2h"]["unique_best_claim_permitted"] is False
    assert report["h2h"]["equivalence_enabled"] is False
    assert report["h2h"]["equivalent_pair_count"] == 0
    assert report["performance"]["interpretation"] == (
        "descriptive_complete_support_tournament_screening"
    )
    assert report["tournament_root_stability"]["interpretation"] == (
        "fixed_design_descriptive_reproducibility_with_monte_carlo_precision"
    )

    expected_claims = {
        "3 finalist comparisons remain unresolved.",
        (
            "Operationally nonviable frozen finalists (retained with no affected "
            "dominance/equivalence claims): ['0', '1', '3']."
        ),
        "No unique-best claim is permitted by the direct-dominance rule.",
    }
    assert expected_claims.issubset(set(report["claim_language"]))
    markdown = cfg.structure_report_markdown_path().read_text(encoding="utf-8")
    assert all(claim in markdown for claim in expected_claims)
    lower_markdown = markdown.lower()
    assert "comparisons satisfy the configured equivalence rule" not in lower_markdown
    assert "statistically significant" not in lower_markdown
    assert "reject" not in lower_markdown
    assert "safety-limit attempts were draws" not in lower_markdown
    assert "unique best among the frozen" not in lower_markdown

    inference = pq.read_table(cfg.h2h_pairwise_inference_path()).to_pandas()
    unresolved = inference["decision_class"].astype(str).str.startswith("unresolved")
    assert unresolved.all()
    assert not inference.loc[unresolved, "holm_reject"].any()
    assert not inference.loc[unresolved, "pair_claim_eligible"].any()
    assert not inference["decision_class"].eq("equivalent").any()
    assert pq.read_table(cfg.h2h_dominance_edges_path()).num_rows == 0

    membership = pq.read_table(cfg.h2h_candidate_family_path()).to_pandas()
    assert membership.loc[membership["final_family"], "strategy"].astype(int).tolist() == [0, 1, 3]
    migration = json.loads(cfg.migration_report_path().read_text(encoding="utf-8"))
    assert migration["artifacts_deleted"] is False


def snapshot_stable_workflow_files(
    contexts: tuple[SeedRunContext, SeedRunContext],
    pair_context: RootPairRunContext,
) -> dict[str, str]:
    """Hash every stable workflow file, excluding only append-only run views."""

    excluded = {
        (pair_context.pair_root / ORCHESTRATION_MANIFEST).resolve(),
        (pair_context.pair_root / "pipeline_health.json").resolve(),
        *(
            context.analysis_root.joinpath(context.config.manifest_name).resolve()
            for context in contexts
        ),
    }
    return {
        path.relative_to(pair_context.pair_root).as_posix(): sha256_file(path)
        for path in sorted(pair_context.pair_root.rglob("*"))
        if path.is_file() and path.resolve() not in excluded
    }


__all__ = [
    "assert_canonical_artifact_inventory",
    "assert_pipeline_health_and_simulation_lifecycle",
    "assert_report_oracle",
    "snapshot_stable_workflow_files",
]
