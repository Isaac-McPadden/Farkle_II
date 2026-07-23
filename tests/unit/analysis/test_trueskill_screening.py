from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from farkle.analysis.trueskill_screening import (
    ScreeningRatingCell,
    build_percentile_contribution,
    build_screening_diagnostics,
    diagnose_rating_cell,
    publish_rating_cell_contract,
)
from farkle.config import AppConfig, IOConfig, SimConfig
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.schema_helpers import expected_schema_for


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        io=IOConfig(results_dir_prefix=tmp_path / "results"),
        sim=SimConfig(seed=11, seed_list=[11, 17], n_players_list=[2, 4]),
    )


def _ratings(path: Path, values: dict[str, tuple[float, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "strategy": list(values),
                "mu": [values[key][0] for key in values],
                "sigma": [values[key][1] for key in values],
            }
        ),
        path,
    )
    return path


def test_percentile_contribution_requires_complete_root_k_support(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cells: list[ScreeningRatingCell] = []
    for root in (11, 17):
        for k in (2, 4):
            values = {"A": (30.0 + root / 100, 2.0), "B": (20.0, 3.0)}
            if (root, k) == (11, 2):
                values["C"] = (40.0, 1.0)
            path = _ratings(tmp_path / f"ratings_{root}_{k}.parquet", values)
            cells.append(ScreeningRatingCell(root, k, path))

    output = build_percentile_contribution(cfg, cells)
    frame = pq.read_table(output).to_pandas().set_index("strategy")

    assert frame.loc["A", "complete_support"]
    assert frame.loc["A", "rating_cells_present"] == 4
    assert frame.loc["A", "candidate_contribution_rank"] == 1
    assert frame.loc["A", "mean_percentile_rank"] == pytest.approx((2 / 3 + 1 + 1 + 1) / 4)
    assert frame.loc["B", "mean_percentile_rank"] == pytest.approx((1 / 3 + 0.5 + 0.5 + 0.5) / 4)
    assert "C" not in frame.index
    assert "sigma" not in frame.columns
    validate_artifact_sidecar(
        output,
        expected={
            "scope": "across_k",
            "operation": "equal_root_k_percentile_mean",
            "seed_scope": "both_roots_combined",
        },
    )
    publish_rating_cell_contract(
        cfg,
        cells[0],
        completed_artifact_sha256=sha256_file(cells[0].ratings_path),
    )
    validate_artifact_sidecar(
        cells[0].ratings_path,
        expected={
            "scope": "by_k",
            "operation": "sequential_rating",
            "uncertainty_method": "trueskill_model_sigma_screening_only",
        },
    )


def test_rating_cell_contract_does_not_repair_a_present_corrupt_sidecar(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    ratings = _ratings(tmp_path / "ratings.parquet", {"A": (30.0, 2.0), "B": (20.0, 3.0)})
    cell = ScreeningRatingCell(11, 2, ratings)
    with pytest.raises(ArtifactContractError, match="independent cell completion"):
        publish_rating_cell_contract(cfg, cell)
    publish_rating_cell_contract(
        cfg,
        cell,
        completed_artifact_sha256=sha256_file(ratings),
    )
    metadata = sidecar_path(ratings)
    original_bytes = metadata.read_bytes()
    corrupt_bytes = original_bytes.replace(b'"scope": "by_k"', b'"scope": "nope"')
    assert corrupt_bytes != original_bytes
    metadata.write_bytes(corrupt_bytes)

    with pytest.raises(ArtifactContractError):
        publish_rating_cell_contract(cfg, cell)

    assert metadata.read_bytes() == corrupt_bytes


def _game_rows(path: Path, games: int = 10) -> Path:
    rows: list[dict[str, object]] = []
    for game in range(games):
        p1_wins = game % 2 == 0
        rows.append(
            {
                "root_seed": 11,
                "k": 2,
                "shuffle_index": game,
                "game_index": game,
                "deterministic_batch_id": 0,
                "winner_seat": "P1" if p1_wins else "P2",
                "winner_strategy": 1 if p1_wins else 2,
                "P1_strategy": 1,
                "P2_strategy": 2,
                "P1_rank": 1 if p1_wins else 2,
                "P2_rank": 2 if p1_wins else 1,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=expected_schema_for(2)), path)
    return path


def test_tau_order_and_heldout_diagnostics(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    rating_path = _ratings(
        tmp_path / "ratings.parquet",
        {"1": (25.0, 2.0), "2": (25.0, 2.0)},
    )
    game_path = _game_rows(tmp_path / "games.parquet")
    cell = ScreeningRatingCell(11, 2, rating_path, game_path)

    row = diagnose_rating_cell(
        cell,
        beta=cfg.trueskill.beta,
        tau=cfg.trueskill.tau,
        draw_probability=cfg.trueskill.draw_probability,
    )
    assert row["tau_zero_games"] == 10
    assert row["reversed_order_games"] == 10
    assert row["holdout_games"] == 2
    assert row["heldout_log_loss"] is not None
    assert row["heldout_brier_score"] is not None
    assert row["uniform_log_loss"] == pytest.approx(math.log(2))

    output = build_screening_diagnostics(cfg, [cell])
    assert output is not None
    diagnostics = pq.read_table(output).to_pandas()
    assert diagnostics.loc[0, "holdout_games"] == 2
    validate_artifact_sidecar(
        output,
        expected={
            "scope": "diagnostics",
            "conditioning": "finite_grid_trueskill_screening_only",
        },
    )
