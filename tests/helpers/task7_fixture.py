"""Explicit synthetic Task 7 inputs for structural reporting tests only."""

from __future__ import annotations

from typing import Any, cast

import pyarrow as pa

from farkle.analysis.parameter_responses import response_paths, response_rows, response_schema
from farkle.analysis.seat_behavior import OA, OPPORTUNITY_SCHEMA, Q7, _zero, summary_rows
from farkle.config import AppConfig
from farkle.simulation.strategies import ThresholdStrategy, build_strategy_manifest
from farkle.utils.artifact_contract import make_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic


def publish_task7_reporting_fixture(cfg: AppConfig) -> None:
    """Publish labeled tiny deterministic products; this is not a producer oracle."""

    def publish(path, table, operation, k=None):
        metadata = make_artifact_sidecar(
            cfg,
            path,
            producer="test_task7_fixture",
            scope="across_k" if k is None else "by_k",
            source_scope="synthetic_fixture",
            operation=operation,
            conditioning="synthetic_structural_fixture",
            consistency_columns=table.schema.names,
            player_counts=cfg.sim.n_players_list if k is None else [k],
            required_player_counts=cfg.sim.n_players_list if k is None else [k],
            missing_cell_policy="fail",
        )
        write_parquet_artifact_atomic(table, path, sidecar=metadata)

    ks = sorted(set(cfg.sim.n_players_list))
    root = cfg.sim.seed
    publish(
        cfg.turn_opportunity_equal_k_path(),
        pa.Table.from_pylist(
            [
                {
                    "root_seed": root,
                    "required_ks": ks,
                    "weights": [1 / len(ks)] * len(ks),
                    "mean_turn_range_equal_k": 0.0,
                    "status": "ok",
                }
            ],
            schema=OA,
        ),
        "turn_opportunity_v1",
    )
    for k in ks:
        total = {
            **_zero(),
            "exposures": 1,
            "score_sum": 100,
            "turns_sum": 1,
            "rounds_sum": 1,
            "score_per_turn_sum": 100.0,
            "score_per_round_sum": 100.0,
        }
        rows = [
            r
            for sid in (1, 2)
            for seat in (0, 1)
            for c in ("completed", "all_attempted")
            for r in summary_rows(root, k, sid, seat, c, total)
        ]
        publish(
            cfg.seat_behavior_summaries_path(k),
            pa.Table.from_pylist(rows, schema=Q7),
            "seat_behavior_v1",
            k,
        )
        common = {"root_seed": root, "k": k, "completed_games": 1, "status": "ok"}
        opportunities = [
            {
                **common,
                "seat": 0,
                "turns_sum": k,
                "mean_turns": None,
                "turn_share": None,
                "mean_turn_range": 0.0,
                "turn_share_imbalance": 0.0,
            },
            *[
                {
                    **common,
                    "seat": s,
                    "turns_sum": 1,
                    "mean_turns": 1.0,
                    "turn_share": 1 / k,
                    "mean_turn_range": None,
                    "turn_share_imbalance": None,
                }
                for s in range(1, k + 1)
            ],
        ]
        publish(
            cfg.turn_opportunity_path(k),
            pa.Table.from_pylist(opportunities, schema=OPPORTUNITY_SCHEMA),
            "seat_behavior_v1",
            k,
        )
        manifest = build_strategy_manifest(
            [ThresholdStrategy(score_threshold=100 * sid, strategy_id=sid) for sid in (1, 2)]
        ).to_dict("records")
        performance = [{"strategy": sid, "win_rate": 0.5} for sid in (1, 2)]
        predictions = [
            {
                "strategy": sid,
                "root_seed": root,
                "players": k,
                "fold": sid - 1,
                "observed_win_rate": 0.5,
                "predicted_win_rate": 0.5,
            }
            for sid in (1, 2)
        ]
        folds = [
            {
                "root_seed": root,
                "players": k,
                "fold": fold,
                "train_strategies": 1,
                "heldout_strategies": 1,
                "mae": 0.0,
                "r2": 0.0,
            }
            for fold in (0, 1)
        ]
        products = list(
            response_rows(
                cast(list[dict[str, Any]], manifest),
                performance,
                predictions,
                folds,
                root=root,
                k=k,
                source_identity={"synthetic_fixture": "task7_reporting"},
            )
        )
        for product, path in zip(
            ("one_way", "two_way", "support"), response_paths(cfg, k), strict=True
        ):
            publish(
                path,
                pa.Table.from_pylist(
                    [r for p, r in products if p == product], schema=response_schema(product)
                ),
                "parameter_response_v1",
                k,
            )
        publish(
            cfg.hgb_fold_metrics_path(k), pa.Table.from_pylist(folds), "heldout_fold_diagnostics", k
        )
        publish(
            cfg.hgb_importance_path(k),
            pa.Table.from_pylist(
                [
                    {
                        "root_seed": root,
                        "players": k,
                        "feature": "score_threshold",
                        "association_importance_mean": 0.0,
                        "association_importance_fold_std": 0.0,
                        "interpretation": "synthetic_fixture",
                    }
                ]
            ),
            "heldout_permutation_importance",
            k,
        )

    from tests.helpers.task8_fixture import publish_task8_reporting_fixture

    publish_task8_reporting_fixture(cfg)
