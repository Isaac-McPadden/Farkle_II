"""Synthetic guide-rendering inputs, explicitly distinct from producer evidence."""

from __future__ import annotations

import pyarrow as pa

from farkle.analysis.analysis_guide import guide_sources
from farkle.analysis.outcome_distributions import outcome_frequency_schema, outcome_support_schema
from farkle.analysis.outcome_precision import outcome_precision_schema
from farkle.analysis.outcome_publication import outcome_summary_schema
from farkle.config import AppConfig
from farkle.simulation.strategies import ThresholdStrategy, build_strategy_manifest
from farkle.utils.artifact_contract import make_artifact_sidecar
from farkle.utils.artifacts import write_json_artifact_atomic, write_parquet_artifact_atomic
from farkle.utils.authenticated_contract import derive_canonical_location
from farkle.utils.release_identity import _stage_key_for_path


def publish_task8_reporting_fixture(cfg: AppConfig) -> None:
    """Supply missing inputs for structural tests only, never a simulated run."""
    for path in guide_sources(cfg, (cfg.sim.seed,)):
        if path.exists():
            continue
        k = (
            int(path.parent.name[:-1])
            if path.parent.name.endswith("p") and path.parent.name[:-1].isdigit()
            else cfg.sim.n_players_list[0]
        )
        key = {"root_seed": cfg.sim.seed, "k": k, "strategy": 1, "seat": 0}
        schema = None
        rows = [
            {"fixture_label": "task8_synthetic_rendering_only", "root_seed": cfg.sim.seed, "k": k}
        ]
        if path.name == "outcome_support.parquet":
            schema = outcome_support_schema()
            rows = [
                {
                    **key,
                    "attempted": 4,
                    "completed": 3,
                    "safety": 1,
                    "wins": 1,
                    "losses_completed": 2,
                    "deficit_sum": 600,
                    "victory_margin_sum": 100,
                    "support_status": "observed",
                }
            ]
        elif path.name == "outcome_summaries.parquet":
            schema = outcome_summary_schema()
            rows = [
                {
                    **key,
                    "quantity": "completed_score",
                    "statistic": "mean",
                    "conditioning": "completed",
                    "denominator": 3,
                    "estimate": 800,
                    "status": "ok",
                }
            ]
        elif path.name == "outcome_precision.parquet":
            schema = outcome_precision_schema()
            rows = [
                {
                    **key,
                    "summary_id": "completed_win_rate",
                    "estimate": 1 / 3,
                    "denominator": 3,
                    "batch_count": 2,
                    "contributing_batches": 2,
                    "confidence": 0.95,
                    "replicates_requested": 20,
                    "replicates_valid": 0,
                    "replicates_invalid": 0,
                    "ci_low": None,
                    "ci_high": None,
                    "interval_width": None,
                    "mcse": None,
                    "replicate_min": None,
                    "replicate_max": None,
                    "status": "insufficient_batches",
                    "method_id": "outcome_batch_percentile_v1",
                }
            ]
        elif path.name == "outcome_frequencies.parquet":
            schema = outcome_frequency_schema()
            rows = [
                {
                    **key,
                    "distribution": "completed_score",
                    "value": 800,
                    "value_scale": 1,
                    "count": 3,
                }
            ]
        table = pa.Table.from_pylist(rows, schema=schema)
        if path.name == "strategy_manifest.parquet":
            table = pa.Table.from_pandas(
                build_strategy_manifest(
                    [
                        ThresholdStrategy(score_threshold=100 * sid, strategy_id=sid)
                        for sid in (1, 2)
                    ]
                ),
                preserve_index=False,
            )
        payload = {
            "fixture_label": "task8_synthetic_rendering_only",
            "root_seed": cfg.sim.seed,
            "required_ks": cfg.sim.n_players_list,
            "status": "insufficient_evidence",
        }
        sidecar = make_artifact_sidecar(
            cfg,
            path,
            producer="task8_synthetic_rendering_fixture",
            scope=derive_canonical_location(
                cfg, path, stage_key=_stage_key_for_path(cfg, path, "simulation")
            ).scope,
            source_scope="synthetic_fixture",
            operation="synthetic_guide_fixture",
            conditioning="synthetic_rendering_only",
            consistency_columns=list(payload) if path.suffix == ".json" else table.schema.names,
            player_counts=cfg.sim.n_players_list,
            required_player_counts=cfg.sim.n_players_list,
            missing_cell_policy="fail",
        )
        if path.suffix == ".json":
            write_json_artifact_atomic(payload, path, sidecar=sidecar)
        else:
            write_parquet_artifact_atomic(table, path, sidecar=sidecar)
