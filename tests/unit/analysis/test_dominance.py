from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.dominance import build_dominance_outputs
from farkle.config import AppConfig, ArtifactScope, IOConfig, SimConfig
from farkle.utils.artifact_contract import make_artifact_sidecar, validate_artifact_sidecar
from farkle.utils.artifacts import write_parquet_artifact_atomic


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(
            seed=11,
            seed_list=[11, 22],
            seed_pair=(11, 22),
            n_players_list=[2],
        ),
    )


def _decision_row(
    pair_id: int,
    strategy_a: str,
    strategy_b: str,
    decision: str,
) -> dict[str, object]:
    favors_a = decision.endswith("_a")
    practical = decision.startswith("practical_dominance")
    directed = decision not in {"unresolved", "equivalent"}
    effect = (0.10 if practical else 0.01) * (1 if favors_a else -1) if directed else 0.0
    return {
        "family_hash": "c" * 64,
        "pair_id": pair_id,
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "d_ab": effect,
        "balanced_a_win_rate_alias": 0.5 + effect,
        "simultaneous_d_low": effect - 0.01,
        "simultaneous_d_high": effect + 0.01,
        "holm_reject": directed,
        "decision_class": decision,
    }


def _publish(
    cfg: AppConfig,
    strategies: tuple[str, ...],
    decisions: dict[tuple[str, str], str],
) -> None:
    rows = [
        _decision_row(pair_id, a, b, decisions[(a, b)])
        for pair_id, (a, b) in enumerate(combinations(strategies, 2))
    ]
    frame = pd.DataFrame(rows)
    path = cfg.h2h_pairwise_inference_path()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="seat_adjusted_score_inference",
        consistency_columns=frame.columns.tolist(),
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined",
    )
    write_parquet_artifact_atomic(table, path, sidecar=sidecar)


def _cycle_decisions(strategies: tuple[str, str, str, str]) -> dict[tuple[str, str], str]:
    a, b, c, d = strategies
    return {
        (a, b): "practical_dominance_a",
        (a, c): "practical_dominance_b",
        (a, d): "practical_dominance_a",
        (b, c): "practical_dominance_a",
        (b, d): "practical_dominance_a",
        (c, d): "practical_dominance_a",
    }


def test_cycles_remain_explicit_and_fronts_use_condensation_dag(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    strategies = ("A", "B", "C", "D")
    _publish(cfg, strategies, _cycle_decisions(strategies))

    artifacts = build_dominance_outputs(cfg)

    cycles = pq.read_table(artifacts.cycles).to_pandas()
    practical_cycle = cycles.loc[cycles["graph_type"] == "practical"]
    assert set(practical_cycle["strategy"]) == {"A", "B", "C"}
    assert practical_cycle["cycle_group"].nunique() == 1
    assert practical_cycle["cycle_size"].eq(3).all()

    fronts = pq.read_table(artifacts.fronts).to_pandas().set_index("strategy")
    assert fronts.loc["A", "practical_front"] == 1
    assert fronts.loc["B", "practical_front"] == 1
    assert fronts.loc["C", "practical_front"] == 1
    assert fronts.loc["D", "practical_front"] == 2
    assert not fronts["display_order_is_inferential"].any()

    summary = json.loads(artifacts.summary.read_text(encoding="utf-8"))
    assert summary["practical_cycle_group_count"] == 1
    assert summary["unique_best"] is None
    assert summary["unique_best_claim_permitted"] is False

    edges = pq.read_table(artifacts.edges).to_pandas()
    assert len(edges.loc[edges["graph_type"] == "practical"]) == 6
    assert len(edges.loc[edges["graph_type"] == "statistical"]) == 6
    for path in artifacts.all_paths:
        validate_artifact_sidecar(path, expected={"scope": "h2h_2p"})


def test_unique_best_requires_direct_practical_dominance_over_every_finalist(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    strategies = ("A", "B", "C", "D")
    decisions = {
        ("A", "B"): "practical_dominance_a",
        ("A", "C"): "practical_dominance_a",
        ("A", "D"): "practical_dominance_a",
        ("B", "C"): "unresolved",
        ("B", "D"): "unresolved",
        ("C", "D"): "unresolved",
    }
    _publish(cfg, strategies, decisions)

    artifacts = build_dominance_outputs(cfg)
    summary = json.loads(artifacts.summary.read_text(encoding="utf-8"))
    fronts = pq.read_table(artifacts.fronts).to_pandas().set_index("strategy")

    assert summary["unique_best"] == "A"
    assert summary["unique_best_claim_permitted"] is True
    assert fronts.loc["A", "practical_front"] == 1
    assert set(fronts.loc[["B", "C", "D"], "practical_front"]) == {2}
    assert summary["decision_counts"]["unresolved"] == 3


def test_identifier_renaming_does_not_change_graph_structure(tmp_path: Path) -> None:
    first_cfg = _cfg(tmp_path / "first")
    first = ("A", "B", "C", "D")
    _publish(first_cfg, first, _cycle_decisions(first))
    first_artifacts = build_dominance_outputs(first_cfg)

    renamed_cfg = _cfg(tmp_path / "renamed")
    renamed = ("W", "X", "Y", "Z")
    _publish(renamed_cfg, renamed, _cycle_decisions(renamed))
    renamed_artifacts = build_dominance_outputs(renamed_cfg)

    first_fronts = pq.read_table(first_artifacts.fronts).to_pandas()
    renamed_fronts = pq.read_table(renamed_artifacts.fronts).to_pandas()
    assert sorted(first_fronts["practical_front"].tolist()) == sorted(
        renamed_fronts["practical_front"].tolist()
    )
    first_cycles = pq.read_table(first_artifacts.cycles).to_pandas()
    renamed_cycles = pq.read_table(renamed_artifacts.cycles).to_pandas()
    assert sorted(first_cycles["cycle_size"].tolist()) == sorted(
        renamed_cycles["cycle_size"].tolist()
    )
    first_summary = json.loads(first_artifacts.summary.read_text(encoding="utf-8"))
    renamed_summary = json.loads(renamed_artifacts.summary.read_text(encoding="utf-8"))
    assert (
        first_summary["unique_best_claim_permitted"]
        == renamed_summary["unique_best_claim_permitted"]
    )


def test_dominance_rejects_incomplete_candidate_pairs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    strategies = ("A", "B", "C", "D")
    decisions = _cycle_decisions(strategies)
    decisions.pop(("C", "D"))
    rows = [
        _decision_row(pair_id, a, b, decision)
        for pair_id, ((a, b), decision) in enumerate(decisions.items())
    ]
    frame = pd.DataFrame(rows)
    path = cfg.h2h_pairwise_inference_path()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sidecar = make_artifact_sidecar(
        cfg,
        path,
        producer="test",
        scope=ArtifactScope.H2H_2P,
        source_scope=ArtifactScope.H2H_2P,
        operation="seat_adjusted_score_inference",
        consistency_columns=frame.columns.tolist(),
        player_counts=[2],
        required_player_counts=[2],
        missing_cell_policy="fail",
        seed_scope="both_roots_combined",
    )
    write_parquet_artifact_atomic(table, path, sidecar=sidecar)

    with pytest.raises(ValueError, match="incomplete"):
        build_dominance_outputs(cfg)
