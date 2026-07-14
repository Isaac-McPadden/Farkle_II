"""Descriptive strategy screening from canonical performance evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

from farkle.config import AppConfig, ArtifactScope
from farkle.utils.artifact_contract import make_artifact_sidecar
from farkle.utils.artifacts import (
    read_parquet_artifact,
    write_json_artifact_atomic,
    write_parquet_artifact_atomic,
)
from farkle.utils.stage_completion import stage_done_path, stage_is_up_to_date, write_stage_done


def _screening_sources(cfg: AppConfig) -> tuple[list[int], list[Path]]:
    player_counts = sorted({int(k) for k in cfg.sim.n_players_list})
    paths = [
        cfg.performance_across_k_path(),
        cfg.performance_bootstrap_path(),
        *(cfg.performance_by_k_path(k) for k in player_counts),
    ]
    return player_counts, paths


def _build_screening_frame(cfg: AppConfig, player_counts: list[int]) -> pd.DataFrame:
    across = read_parquet_artifact(
        cfg.performance_across_k_path(),
        expected_sidecar={
            "scope": ArtifactScope.ACROSS_K.value,
            "operation": "equal_k_mean",
            "conditioning": "unconditional",
        },
    ).to_pandas()
    bootstrap = read_parquet_artifact(
        cfg.performance_bootstrap_path(),
        expected_sidecar={
            "scope": ArtifactScope.ACROSS_K.value,
            "uncertainty_method": "joint_deterministic_batch_resampling",
        },
    ).to_pandas()
    if not across["complete_support"].all():
        incomplete = across.loc[~across["complete_support"], "strategy"].astype(int).tolist()
        raise ValueError(
            "descriptive screening requires complete configured k support; "
            f"incomplete strategies: {incomplete[:20]}"
        )
    output = across.merge(bootstrap, on=["root_seed", "strategy"], validate="one_to_one")
    output = output.sort_values(["equal_k_score", "strategy"], ascending=[False, True]).reset_index(
        drop=True
    )
    output["score_order_position"] = output.index + 1
    output["observed_top_n"] = output["score_order_position"] <= min(
        cfg.screening.candidate_contribution_size, len(output)
    )
    leader_score = float(output["equal_k_score"].max())
    output["within_across_k_practical_band"] = output["equal_k_score"] >= (
        leader_score - float(cfg.screening.delta_across_k or 0.0)
    )
    output["declared_control"] = output["strategy"].isin(cfg.screening.controls)
    output["mandatory_diagnostic"] = output["strategy"].isin(cfg.screening.mandatory_diagnostics)

    per_k_band_columns: list[str] = []
    practical_by_k = cfg.screening.practical_delta_by_k or {}
    for k in player_counts:
        frame = read_parquet_artifact(
            cfg.performance_by_k_path(k),
            expected_sidecar={"scope": ArtifactScope.BY_K.value, "player_counts": [k]},
        ).to_pandas()
        selected = frame[["root_seed", "strategy", "chance_delta", "win_rate", "raw_exposures"]]
        selected = selected.rename(
            columns={
                "chance_delta": f"chance_delta_k{k}",
                "win_rate": f"win_rate_k{k}",
                "raw_exposures": f"raw_exposures_k{k}",
            }
        )
        output = output.merge(selected, on=["root_seed", "strategy"], validate="one_to_one")
        band_column = f"within_k{k}_practical_band"
        per_k_band_columns.append(band_column)
        best = float(output[f"chance_delta_k{k}"].max())
        output[band_column] = output[f"chance_delta_k{k}"] >= (best - float(practical_by_k[k]))
    output["within_every_k_practical_band"] = output[per_k_band_columns].all(axis=1)
    return output


def run(cfg: AppConfig, *, force: bool = False) -> None:
    """Write descriptive evidence without inferential tiers or equality claims."""

    player_counts, sources = _screening_sources(cfg)
    missing = [path for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"descriptive screening inputs are missing: {missing}")
    output = cfg.screening_path()
    report = cfg.screening_path("descriptive_screening.json")
    done = stage_done_path(cfg.screening_stage_dir, "screening")
    if not force and stage_is_up_to_date(
        done,
        inputs=sources,
        outputs=[output, report],
        cfg=cfg,
        stage="screening",
        sidecar_artifacts=[output, report],
    ):
        return

    frame = _build_screening_frame(cfg, player_counts)
    common: dict[str, Any] = {
        "producer": "screening",
        "scope": ArtifactScope.ACROSS_K,
        "source_scope": ArtifactScope.ACROSS_K,
        "operation": "equal_k_mean",
        "baseline": "chance_1_over_k",
        "weighted_quantity": "chance_adjusted_win_rate",
        "k_aggregation_method": "equal_k",
        "support_count_role": "raw_player_game_exposures",
        "uncertainty_method": "descriptive_with_joint_batch_resampling",
        "replication_unit": "deterministic_shuffle_batch",
        "conditioning": "unconditional",
        "source_artifacts": sources,
        "grouping_keys": ["root_seed", "strategy"],
        "player_counts": player_counts,
        "required_player_counts": player_counts,
        "missing_cell_policy": "fail",
    }
    data_sidecar = make_artifact_sidecar(
        cfg,
        output,
        consistency_columns=frame.columns.tolist(),
        **common,
    )
    write_parquet_artifact_atomic(
        pa.Table.from_pandas(frame, preserve_index=False),
        output,
        sidecar=data_sidecar,
        codec=cfg.parquet_codec,
    )
    payload = {
        "artifact": output.name,
        "interpretation": (
            "Descriptive finite-grid screening evidence only; practical bands are not "
            "tests of equality, final tiers, or unique-best decisions."
        ),
        "player_counts": player_counts,
        "strategy_count": len(frame),
        "pareto_count": int(frame["pareto_member"].sum()),
        "maximin_leader": int(frame.loc[frame["maximin_leader"], "strategy"].iloc[0]),
        "control_count": int(frame["declared_control"].sum()),
        "mandatory_diagnostic_count": int(frame["mandatory_diagnostic"].sum()),
    }
    report_sidecar = make_artifact_sidecar(
        cfg,
        report,
        consistency_columns=list(payload),
        **common,
    )
    write_json_artifact_atomic(payload, report, sidecar=report_sidecar)
    write_stage_done(
        done,
        inputs=sources,
        outputs=[output, report],
        cfg=cfg,
        stage="screening",
        sidecar_artifacts=[output, report],
    )


__all__ = ["run"]
