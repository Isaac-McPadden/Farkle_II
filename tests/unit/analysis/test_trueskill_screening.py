from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.helpers.artifact_sidecars import (
    make_authenticated_v3_config,
    mutate_json_identity_leaf,
    publish_v3_parquet,
    publish_v3_strategy_manifest,
)

import farkle.analysis.trueskill_screening as screening_module
from farkle.analysis.trueskill_screening import (
    MU_SOFTMAX_HEURISTIC,
    MU_SOFTMAX_HEURISTIC_CLAIM,
    MU_SOFTMAX_HEURISTIC_OPERATION,
    TRUESKILL_CONDITIONING,
    ScreeningRatingCell,
    build_percentile_contribution,
    build_screening_diagnostics,
    classify_trueskill_row,
    diagnose_rating_cell,
    mu_softmax_heuristic_probabilities,
    publish_rating_cell_contract,
    trueskill_diagnostic_method_contract,
)
from farkle.config import AppConfig
from farkle.orchestration.run_contexts import RootPairRunContext, SeedRunContext
from farkle.simulation.strategies import ThresholdStrategy
from farkle.utils.artifact_contract import (
    ArtifactContractError,
    sha256_file,
    sidecar_path,
    validate_artifact_sidecar,
)
from farkle.utils.authenticated_contract import validate_authenticated_artifact_unbound
from farkle.utils.schema_helpers import expected_schema_for
from farkle.utils.stage_completion import stage_done_path


def _cfg(tmp_path: Path) -> AppConfig:
    return make_authenticated_v3_config(
        tmp_path,
        name="screening",
        root_seed=11,
        player_counts=(2, 4),
    )


def _publish_strategy_control(cfg: AppConfig) -> Path:
    path = cfg.strategy_manifest_root_path()
    if not path.exists():
        publish_v3_strategy_manifest(
            cfg,
            tuple(
                ThresholdStrategy(
                    score_threshold=200 + 100 * strategy_id,
                    dice_threshold=2,
                    strategy_id=strategy_id,
                )
                for strategy_id in range(1, 5)
            ),
        )
    return path


def _publish_game_source(cfg: AppConfig, k: int) -> Path:
    data: dict[str, pa.Array | list[object]] = {
        "termination_status": ["completed", "completed"],
        "outcome_schema_version": [2, 2],
        "winner_seat": ["P1", "P1"],
    }
    for seat in range(1, k + 1):
        data[f"P{seat}_strategy"] = pa.array([seat, seat], type=pa.int32())
        data[f"P{seat}_rank"] = [seat, seat]
    return publish_v3_parquet(
        cfg,
        cfg.ingested_rows_curated(k),
        pa.table(data),
        stage_key="curate",
        producer="curate",
        operation="curate_game_rows",
    )


def _publish_rating_control(
    cfg: AppConfig,
    k: int,
    values: dict[str, tuple[float, float]],
) -> ScreeningRatingCell:
    _publish_strategy_control(cfg)
    source = _publish_game_source(cfg, k)
    path = _ratings(cfg.trueskill_rating_path(k, root_seed=cfg.sim.seed), values)
    cell = ScreeningRatingCell(cfg.sim.seed, k, path, source)
    publish_rating_cell_contract(
        cfg,
        cell,
        completed_artifact_sha256=sha256_file(path),
    )
    validate_artifact_sidecar(path)
    return cell


def _pair_fixture(tmp_path: Path) -> tuple[AppConfig, tuple[SeedRunContext, SeedRunContext]]:
    first = SeedRunContext.from_config(
        make_authenticated_v3_config(
            tmp_path,
            name="root_11",
            root_seed=11,
            player_counts=(2, 4),
        )
    )
    second = SeedRunContext.from_config(
        make_authenticated_v3_config(
            tmp_path,
            name="root_17",
            root_seed=17,
            player_counts=(2, 4),
        )
    )
    roots = (first, second)
    pair = RootPairRunContext.from_root_contexts(roots, pair_root=tmp_path / "pair")
    return pair.config, roots


def _ratings(path: Path, values: dict[str, tuple[float, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    strategy_count = len(values)
    updates = [1] * strategy_count
    for index in range(max(0, 4 - sum(updates))):
        updates[index % strategy_count] += 1
    pq.write_table(
        pa.table(
            {
                "strategy": pa.array([int(value) for value in values], type=pa.int32()),
                "mu": [values[key][0] for key in values],
                "sigma": [values[key][1] for key in values],
                "strategy_attempted_exposures": updates,
                "strategy_completed_exposures": updates,
                "strategy_excluded_safety_limit_exposures": [0] * strategy_count,
                "strategy_performed_updates": updates,
                "rating_status": ["evidence_backed_completed_games"] * strategy_count,
                "cell_games_attempted": [2] * strategy_count,
                "cell_games_completed": [2] * strategy_count,
                "cell_games_excluded_safety_limit": [0] * strategy_count,
                "cell_performed_updates": [2] * strategy_count,
            }
        ),
        path,
    )
    return path


def test_percentile_contribution_requires_complete_root_k_support(tmp_path: Path) -> None:
    cfg, root_contexts = _pair_fixture(tmp_path)
    by_seed = {context.seed: context.config for context in root_contexts}
    cells: list[ScreeningRatingCell] = []
    for root in (11, 17):
        for k in (2, 4):
            values = {"1": (30.0 + root / 100, 2.0), "2": (20.0, 3.0)}
            if (root, k) == (11, 2):
                values["3"] = (40.0, 1.0)
            cells.append(_publish_rating_control(by_seed[root], k, values))

    output = build_percentile_contribution(cfg, cells)
    frame = pq.read_table(output).to_pandas().set_index("strategy")

    assert frame.loc[1, "complete_support"]
    assert frame.loc[1, "rating_cells_present"] == 4
    assert frame.loc[1, "candidate_contribution_rank"] == 1
    assert frame.loc[1, "mean_percentile_rank"] == pytest.approx((2 / 3 + 1 + 1 + 1) / 4)
    assert frame.loc[2, "mean_percentile_rank"] == pytest.approx((1 / 3 + 0.5 + 0.5 + 0.5) / 4)
    assert 3 not in frame.index
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


def test_percentile_contribution_excludes_prior_only_rows(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _publish_strategy_control(cfg)
    cells: list[ScreeningRatingCell] = []
    for k in (2, 4):
        source = _publish_game_source(cfg, k)
        path = cfg.trueskill_rating_path(k, root_seed=11)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "strategy": pa.array([1, 2, 3], type=pa.int32()),
                    "mu": [30.0, 20.0, 25.0],
                    "sigma": [2.0, 3.0, 25.0 / 3.0],
                    "strategy_performed_updates": [2, 2, 0],
                    "strategy_attempted_exposures": [3, 2, 1],
                    "strategy_completed_exposures": [2, 2, 0],
                    "strategy_excluded_safety_limit_exposures": [1, 0, 1],
                    "rating_status": [
                        "evidence_backed_completed_games",
                        "evidence_backed_completed_games",
                        "prior_only_unrated",
                    ],
                    "cell_games_attempted": [3, 3, 3],
                    "cell_games_completed": [2, 2, 2],
                    "cell_games_excluded_safety_limit": [1, 1, 1],
                    "cell_performed_updates": [2, 2, 2],
                }
            ),
            path,
        )
        cell = ScreeningRatingCell(11, k, path, source)
        publish_rating_cell_contract(
            cfg,
            cell,
            completed_artifact_sha256=sha256_file(path),
        )
        cells.append(cell)

    contribution = pq.read_table(build_percentile_contribution(cfg, cells)).to_pandas()

    assert contribution["strategy"].tolist() == [1, 2]
    assert contribution["mean_percentile_rank"].tolist() == [1.0, 0.5]


def test_rating_cell_contract_does_not_repair_a_present_corrupt_sidecar(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cell = _publish_rating_control(cfg, 2, {"1": (30.0, 2.0), "2": (20.0, 3.0)})
    ratings = cell.ratings_path
    metadata = sidecar_path(ratings)
    mutate_json_identity_leaf(metadata, ("artifact", "location", "scope"), "diagnostics")
    corrupt_bytes = metadata.read_bytes()

    with pytest.raises(RuntimeError, match="scope|identity|sidecar|location"):
        publish_rating_cell_contract(cfg, cell)

    assert metadata.read_bytes() == corrupt_bytes


def test_prechange_rating_schema_is_rejected_as_stale(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cell = _publish_rating_control(cfg, 2, {"1": (30.0, 2.0), "2": (20.0, 3.0)})
    ratings = cell.ratings_path
    pq.write_table(
        pa.table(
            {
                "strategy": pa.array([1, 2], type=pa.int32()),
                "mu": [30.0, 20.0],
                "sigma": [2.0, 3.0],
            }
        ),
        ratings,
    )

    with pytest.raises(ArtifactContractError, match="rating support is missing"):
        publish_rating_cell_contract(
            cfg,
            cell,
            completed_artifact_sha256=sha256_file(ratings),
        )


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
                "termination_status": "completed",
                "hit_safety_limit": False,
                "outcome_schema_version": 2,
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


def test_mu_softmax_heuristic_probability_assumptions_are_explicit() -> None:
    equal_mu = mu_softmax_heuristic_probabilities(
        [(25.0, 1.0), (25.0, 10.0)],
        beta=5.0,
    )
    different_mu = mu_softmax_heuristic_probabilities(
        [(30.0, 4.0), (20.0, 4.0)],
        beta=5.0,
    )

    assert equal_mu.tolist() == pytest.approx([0.5, 0.5])
    assert different_mu.tolist() == pytest.approx(
        [1.0 / (1.0 + math.exp(-2.0)), 1.0 / (1.0 + math.exp(2.0))]
    )


def test_mu_softmax_heuristic_claim_and_method_contract_are_exact() -> None:
    expected_claim = (
        "Held-out descriptive scores use mu_softmax_heuristic probabilities computed "
        "as softmax(mu / beta). TrueSkill sigma is ignored; these are heuristic "
        "probabilities, not TrueSkill predictive probabilities."
    )

    assert MU_SOFTMAX_HEURISTIC == "mu_softmax_heuristic"
    assert expected_claim == MU_SOFTMAX_HEURISTIC_CLAIM
    assert trueskill_diagnostic_method_contract().get("parameters") == {
        "method_version": 3,
        "outcome_schema_version": 2,
        "conditioning": "termination_status == completed",
        "safety_limit_policy": "excluded_without_update_or_rank_imputation",
        "diagnostic_method_version": 2,
        "heldout_probability_method": "mu_softmax_heuristic",
        "heldout_probability_formula": "softmax(mu / beta)",
        "heldout_probability_sigma_policy": "ignored",
        "heldout_fraction": 0.2,
        "heldout_rating_policy": "freeze_after_chronological_training_prefix",
        "heldout_target": "unique_rank_1_completed_game_winner",
        "interpretation": expected_claim,
    }


def test_tau_order_and_heldout_diagnostics(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _publish_strategy_control(cfg)
    rating_path = _ratings(
        cfg.trueskill_rating_path(2, root_seed=11),
        {"1": (25.0, 2.0), "2": (25.0, 2.0)},
    )
    game_path = cfg.ingested_rows_curated(2)
    _game_rows(game_path)
    publish_v3_parquet(
        cfg,
        game_path,
        pq.read_table(game_path),
        stage_key="curate",
        producer="curate",
        operation="curate_game_rows",
    )
    cell = ScreeningRatingCell(11, 2, rating_path, game_path)
    publish_rating_cell_contract(
        cfg,
        cell,
        completed_artifact_sha256=sha256_file(rating_path),
    )

    row = diagnose_rating_cell(
        cell,
        beta=cfg.trueskill.beta,
        tau=cfg.trueskill.tau,
        draw_probability=cfg.trueskill.draw_probability,
    )
    assert row["tau_zero_games"] == 10
    assert row["reversed_order_games"] == 10
    assert row["mu_softmax_heuristic_holdout_games"] == 2
    assert row["mu_softmax_heuristic_heldout_log_loss"] is not None
    assert row["mu_softmax_heuristic_heldout_brier_score"] is not None
    assert row["mu_softmax_heuristic_uniform_reference_log_loss"] == pytest.approx(math.log(2))
    assert row["mu_softmax_heuristic_claim"] == MU_SOFTMAX_HEURISTIC_CLAIM

    output = build_screening_diagnostics(cfg, [cell])
    assert output is not None
    diagnostics = pq.read_table(output).to_pandas()
    assert diagnostics.loc[0, "mu_softmax_heuristic_holdout_games"] == 2
    validate_artifact_sidecar(
        output,
        expected={
            "scope": "diagnostics",
            "operation": MU_SOFTMAX_HEURISTIC_OPERATION,
            "weighted_quantity": (
                "trueskill_screening_sensitivity_and_mu_softmax_heuristic_scores"
            ),
            "uncertainty_method": "descriptive_replay_and_mu_softmax_heuristic_scoring",
            "conditioning": TRUESKILL_CONDITIONING,
            "method_contract": trueskill_diagnostic_method_contract(),
        },
    )
    completion = json.loads(
        stage_done_path(
            cfg.trueskill_stage_dir,
            "trueskill_screening_diagnostics",
        ).read_text(encoding="utf-8")
    )
    assert completion["state"] == "complete_valid"
    authenticated = validate_authenticated_artifact_unbound(
        output,
        validate_provenance=False,
    )
    assert authenticated.versions.method_versions["trueskill_diagnostic_method_version"] == 2


def test_diagnostics_are_invariant_to_row_groups_and_byte_bounded_input_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rating_path = _ratings(tmp_path / "ratings.parquet", {"1": (25.0, 2.0), "2": (25.0, 2.0)})
    first = _game_rows(tmp_path / "first.parquet", games=11)
    second = tmp_path / "second.parquet"
    pq.write_table(pq.read_table(first), second, row_group_size=1)
    cell_first = ScreeningRatingCell(11, 2, rating_path, first)
    cell_second = ScreeningRatingCell(11, 2, rating_path, second)
    seen_batch_bytes: list[int] = []
    original = screening_module.iter_parquet_tables_by_bytes

    def bounded(*args: object, **kwargs: object):
        for item in original(*args, **kwargs):  # type: ignore[arg-type]
            seen_batch_bytes.append(item[2].nbytes)
            yield item

    monkeypatch.setattr(screening_module, "iter_parquet_tables_by_bytes", bounded)
    first_row = diagnose_rating_cell(
        cell_first,
        beta=25.0,
        tau=0.1,
        draw_probability=0.0,
        max_batch_bytes=512,
        batch_rows=100,
    )
    second_row = diagnose_rating_cell(
        cell_second,
        beta=25.0,
        tau=0.1,
        draw_probability=0.0,
        max_batch_bytes=512,
        batch_rows=1,
    )

    assert first_row == second_row
    assert seen_batch_bytes and max(seen_batch_bytes) <= 512


def test_diagnostic_cell_resume_rejects_missing_or_corrupt_completion_stamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    cell = _publish_rating_control(cfg, 2, {"1": (30.0, 2.0), "2": (20.0, 3.0)})
    build_screening_diagnostics(cfg, [cell])
    screening_module._diagnostic_cell_done_path(cfg, cell).write_text("{corrupt", encoding="utf-8")
    calls = 0
    original = screening_module.diagnose_rating_cell

    def counted(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(screening_module, "diagnose_rating_cell", counted)
    output = build_screening_diagnostics(cfg, [cell])

    assert output is not None
    assert calls == 1
    stamp = screening_module._diagnostic_cell_done_path(cfg, cell)
    assert stamp.exists()
    stamp.unlink()
    assert build_screening_diagnostics(cfg, [cell]) is not None
    assert calls == 2


def test_interrupted_diagnostic_cell_retries_without_accepting_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    cell = _publish_rating_control(cfg, 2, {"1": (30.0, 2.0), "2": (20.0, 3.0)})
    cell_output = cfg.trueskill_screening_diagnostic_cell_path(2, root_seed=11)

    def interrupted(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(screening_module, "diagnose_rating_cell", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_screening_diagnostics(cfg, [cell])
    assert not cell_output.exists()
    assert not screening_module._diagnostic_cell_done_path(cfg, cell).exists()

    monkeypatch.undo()
    assert build_screening_diagnostics(cfg, [cell]) is not None
    assert cell_output.exists()


def test_diagnostic_aggregate_is_worker_count_invariant_and_scans_once_per_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path)
    cells = [
        _publish_rating_control(cfg, 2, {"1": (30.0, 2.0), "2": (20.0, 3.0)}),
        _publish_rating_control(
            cfg,
            4,
            {"1": (30.0, 2.0), "2": (20.0, 3.0), "3": (21.0, 3.0), "4": (22.0, 3.0)},
        ),
    ]
    scans = 0
    original_iter = screening_module.pq.ParquetFile.iter_batches

    def counted_iter(self: pq.ParquetFile, *args: object, **kwargs: object):
        nonlocal scans
        scans += 1
        yield from original_iter(self, *args, **kwargs)

    monkeypatch.setattr(screening_module.pq.ParquetFile, "iter_batches", counted_iter)
    cfg.analysis.n_jobs = 1
    single = pq.read_table(build_screening_diagnostics(cfg, cells, force=True)).to_pydict()
    assert scans <= len(cells) * 2

    cfg.analysis.n_jobs = 2
    parallel = pq.read_table(build_screening_diagnostics(cfg, cells, force=True)).to_pydict()
    assert parallel == single


def test_diagnostics_report_mixed_support_and_prior_only_strategy(tmp_path: Path) -> None:
    rating_path = tmp_path / "ratings.parquet"
    pq.write_table(
        pa.table(
            {
                "strategy": pa.array([1, 2, 3], type=pa.int32()),
                "mu": [30.0, 20.0, 25.0],
                "sigma": [5.0, 5.0, 25.0 / 3.0],
                "strategy_performed_updates": [1, 1, 0],
                "strategy_attempted_exposures": [2, 1, 1],
                "strategy_completed_exposures": [1, 1, 0],
                "strategy_excluded_safety_limit_exposures": [1, 0, 1],
                "rating_status": [
                    "evidence_backed_completed_games",
                    "evidence_backed_completed_games",
                    "prior_only_unrated",
                ],
                "cell_games_attempted": [2, 2, 2],
                "cell_games_completed": [1, 1, 1],
                "cell_games_excluded_safety_limit": [1, 1, 1],
                "cell_performed_updates": [1, 1, 1],
            }
        ),
        rating_path,
    )
    game_path = tmp_path / "games.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "termination_status": "completed",
                    "outcome_schema_version": 2,
                    "winner_seat": "P1",
                    "P1_strategy": 1,
                    "P2_strategy": 2,
                    "P1_rank": 1,
                    "P2_rank": 2,
                },
                {
                    "termination_status": "safety_limit",
                    "outcome_schema_version": 2,
                    "winner_seat": None,
                    "P1_strategy": 1,
                    "P2_strategy": 3,
                    "P1_rank": None,
                    "P2_rank": None,
                },
            ]
        ),
        game_path,
    )

    row = diagnose_rating_cell(
        ScreeningRatingCell(11, 2, rating_path, game_path),
        beta=25.0,
        tau=0.1,
        draw_probability=0.0,
    )

    assert row["attempted_games"] == 2
    assert row["completed_games"] == 1
    assert row["excluded_safety_limit_games"] == 1
    assert row["performed_update_games"] == 1
    assert row["prior_only_strategy_count"] == 1
    assert row["prior_only_strategies"] == "3"
    assert row["mu_softmax_heuristic_training_games"] == 1
    assert row["mu_softmax_heuristic_holdout_games"] == 0


@pytest.mark.parametrize("malformed", [None, "1", 1.0, True])
def test_trueskill_row_rejects_noncanonical_strategy_ids(malformed: object) -> None:
    valid: dict[str, object] = {
        "termination_status": "completed",
        "outcome_schema_version": 2,
        "winner_seat": "P1",
        "P1_strategy": 1,
        "P2_strategy": 2,
        "P1_rank": 1,
        "P2_rank": 2,
    }
    assert classify_trueskill_row(valid, 2).ranks == [0, 1]
    row = {**valid, "P1_strategy": malformed}

    with pytest.raises(ValueError, match="P1_strategy"):
        classify_trueskill_row(row, 2)
