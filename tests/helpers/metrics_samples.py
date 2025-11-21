from __future__ import annotations

# tests/helpers/metrics_samples.py
"""Helpers for generating and validating metrics-stage sample artifacts."""

import json
import shutil
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.config import AppConfig, IOConfig, SimConfig
from farkle.utils.schema_helpers import expected_schema_for

from .golden_utils import assert_csv_golden, assert_parquet_golden, assert_stamp_has_paths

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "metrics_stage_v2"
INPUT_ROOT = DATA_ROOT / "inputs"
GOLDEN_ROOT = DATA_ROOT / "goldens"
VERSION_FILE = DATA_ROOT / "VERSION.txt"

_SIM_KWARGS = dict(
    n_players_list=[2, 3],
    score_thresholds=[200],
    dice_thresholds=[0],
    smart_five_opts=[True],
    smart_one_opts=[False],
    consider_score_opts=[True],
    consider_dice_opts=[True],
    auto_hot_dice_opts=[True],
    run_up_score_opts=[True],
)

_STRATEGIES = [
    "Strat(200,0)[SD][F-FS][AND][HR]",
    "Strat(200,0)[SD][F-FD][AND][HR]",
    "Strat(200,0)[SD][F-FS][OR][HR]",
    "Strat(200,0)[SD][F-FD][OR][HR]",
]

_COMBINED_ROWS: list[dict[str, object]] = [
    {
        "winner_seat": "P1",
        "winner_strategy": _STRATEGIES[0],
        "seat_ranks": ["P1", "P2", "P3"],
        "winning_score": 100,
        "n_rounds": 10,
        "P1_strategy": _STRATEGIES[0],
        "P2_strategy": _STRATEGIES[1],
        "P3_strategy": _STRATEGIES[2],
        "P1_rank": 1,
        "P2_rank": 2,
        "P3_rank": 3,
        "P1_score": 101,
        "P2_score": 102,
        "P3_score": 103,
        "P1_loss_margin": 11,
        "P2_loss_margin": 12,
        "P3_loss_margin": 13,
        "P1_loss_score": 201,
        "P2_loss_score": 202,
        "P3_loss_score": 203,
        "P1_rounds": 21,
        "P2_rounds": 22,
        "P3_rounds": 23,
    },
    {
        "winner_seat": "P2",
        "winner_strategy": _STRATEGIES[1],
        "seat_ranks": ["P2", "P1", "P3"],
        "winning_score": 110,
        "n_rounds": 11,
        "P1_strategy": _STRATEGIES[0],
        "P2_strategy": _STRATEGIES[1],
        "P3_strategy": _STRATEGIES[2],
        "P1_rank": 2,
        "P2_rank": 1,
        "P3_rank": 3,
        "P1_score": 111,
        "P2_score": 112,
        "P3_score": 113,
        "P1_loss_margin": 21,
        "P2_loss_margin": 22,
        "P3_loss_margin": 23,
        "P1_loss_score": 211,
        "P2_loss_score": 212,
        "P3_loss_score": 213,
        "P1_rounds": 31,
        "P2_rounds": 32,
        "P3_rounds": 33,
    },
    {
        "winner_seat": "P1",
        "winner_strategy": _STRATEGIES[0],
        "seat_ranks": ["P1", "P2"],
        "winning_score": 95,
        "n_rounds": 9,
        "P1_strategy": _STRATEGIES[0],
        "P2_strategy": _STRATEGIES[3],
        "P3_strategy": None,
        "P1_rank": 1,
        "P2_rank": 2,
        "P3_rank": 0,
        "P1_score": 90,
        "P2_score": 80,
        "P3_score": 0,
        "P1_loss_margin": 0,
        "P2_loss_margin": 0,
        "P3_loss_margin": 0,
        "P1_loss_score": 0,
        "P2_loss_score": 0,
        "P3_loss_score": 0,
        "P1_rounds": 15,
        "P2_rounds": 16,
        "P3_rounds": 0,
    },
]

_MANIFEST_ROWS = {2: 1, 3: 2}

_RAW_METRICS = {
    2: [
        {
            "strategy": _STRATEGIES[0],
            "wins": 12,
            "total_games_strat": 20,
            "sum_winning_score": 2000.0,
            "sq_sum_winning_score": 210000.0,
            "sum_n_rounds": 100.0,
            "sq_sum_n_rounds": 1100.0,
            "sum_winner_hit_max_rounds": 0.0,
        },
        {
            "strategy": _STRATEGIES[1],
            "wins": 8,
            "total_games_strat": 20,
            "sum_winning_score": 1400.0,
            "sq_sum_winning_score": 150000.0,
            "sum_n_rounds": 80.0,
            "sq_sum_n_rounds": 900.0,
            "sum_winner_hit_max_rounds": 1.0,
        },
        {
            "strategy": _STRATEGIES[2],
            "wins": 5,
            "total_games_strat": 20,
            "sum_winning_score": 900.0,
            "sq_sum_winning_score": 95000.0,
            "sum_n_rounds": 70.0,
            "sq_sum_n_rounds": 750.0,
            "sum_winner_hit_max_rounds": 0.0,
        },
        {
            "strategy": _STRATEGIES[3],
            "wins": 1,
            "total_games_strat": 20,
            "sum_winning_score": 200.0,
            "sq_sum_winning_score": 21000.0,
            "sum_n_rounds": 40.0,
            "sq_sum_n_rounds": 420.0,
            "sum_winner_hit_max_rounds": 0.0,
        },
    ],
    3: [
        {
            "strategy": _STRATEGIES[0],
            "wins": 9,
            "total_games_strat": 15,
            "sum_winning_score": 1500.0,
            "sq_sum_winning_score": 160000.0,
            "sum_n_rounds": 120.0,
            "sq_sum_n_rounds": 1300.0,
            "sum_winner_hit_max_rounds": 0.0,
        },
        {
            "strategy": _STRATEGIES[1],
            "wins": 4,
            "total_games_strat": 15,
            "sum_winning_score": 900.0,
            "sq_sum_winning_score": 95000.0,
            "sum_n_rounds": 90.0,
            "sq_sum_n_rounds": 950.0,
            "sum_winner_hit_max_rounds": 0.0,
        },
        {
            "strategy": _STRATEGIES[2],
            "wins": 2,
            "total_games_strat": 15,
            "sum_winning_score": 400.0,
            "sq_sum_winning_score": 42000.0,
            "sum_n_rounds": 60.0,
            "sq_sum_n_rounds": 650.0,
            "sum_winner_hit_max_rounds": 0.0,
        },
        {
            "strategy": _STRATEGIES[3],
            "wins": 0,
            "total_games_strat": 15,
            "sum_winning_score": 0.0,
            "sq_sum_winning_score": 0.0,
            "sum_n_rounds": 0.0,
            "sq_sum_n_rounds": 0.0,
            "sum_winner_hit_max_rounds": 0.0,
        },
    ],
}


def _write_version_note() -> None:
    """Record the generator version used for the golden artifacts.

    Returns:
        None
    """

    VERSION_FILE.write_text(
        "v2 metrics goldens generated via tests/helpers/metrics_samples.py using metrics.run (CSV goldens)\n"
    )


def _parquet_inputs_exist(root: Path) -> bool:
    """Check whether the expected parquet inputs already exist.

    Args:
        root: Directory containing the metrics inputs.

    Returns:
        True when all required parquet inputs are present.
    """

    expected = [
        root / "analysis" / "data" / "all_n_players_combined" / "all_ingested_rows.parquet",
        root / "2_players" / "2p_metrics.parquet",
        root / "3_players" / "3p_metrics.parquet",
    ]
    return all(path.exists() for path in expected)


def regenerate_inputs(target_root: Path) -> None:
    """Rebuild the synthetic metrics inputs in ``target_root``."""

    if target_root.exists():
        shutil.rmtree(target_root)
    analysis_root = target_root / "analysis" / "data"
    combined_dir = analysis_root / "all_n_players_combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_parquet = combined_dir / "all_ingested_rows.parquet"
    combined_csv = combined_dir / "all_ingested_rows.csv"
    combined_table = pa.Table.from_pylist(_COMBINED_ROWS, schema=expected_schema_for(3))
    pq.write_table(combined_table, combined_parquet)
    combined_table.to_pandas().to_csv(combined_csv, index=False)

    for n, count in _MANIFEST_ROWS.items():
        manifest_dir = analysis_root / f"{n}p"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest = manifest_dir / "manifest.jsonl"
        manifest.write_text(json.dumps({"row_count": count}))

    for n in _RAW_METRICS:
        metrics_dir = target_root / f"{n}_players"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_parquet = metrics_dir / f"{n}p_metrics.parquet"
        metrics_csv = metrics_dir / f"{n}p_metrics.csv"
        table = pa.Table.from_pylist(_RAW_METRICS[n])
        pq.write_table(table, metrics_parquet)
        table.to_pandas().to_csv(metrics_csv, index=False)

    _write_version_note()


def build_config(results_root: Path) -> AppConfig:
    """Construct an application config bound to a results directory.

    Args:
        results_root: Root directory containing metrics artifacts.

    Returns:
        Application configuration tailored for the sample metrics stage.
    """

    return AppConfig(io=IOConfig(results_dir=results_root, append_seed=False), sim=SimConfig(**_SIM_KWARGS))


def stage_sample_run(tmp_path: Path, *, refresh_inputs: bool) -> AppConfig:
    """Prepare a workspace populated with synthetic metrics inputs.

    Args:
        tmp_path: Base temporary directory provided by pytest.
        refresh_inputs: Whether to rebuild the static parquet inputs.

    Returns:
        Configuration pointing at the prepared workspace.
    """

    if refresh_inputs or not _parquet_inputs_exist(INPUT_ROOT):
        regenerate_inputs(INPUT_ROOT)
    workspace = tmp_path / "results"
    shutil.copytree(INPUT_ROOT, workspace, dirs_exist_ok=True)
    cfg = build_config(workspace)
    return cfg


def validate_outputs(cfg: AppConfig, *, update_goldens: bool) -> None:
    """Compare generated metrics artifacts against stored goldens.

    Args:
        cfg: Application configuration with analysis directory details.
        update_goldens: Whether to refresh existing golden artifacts.

    Returns:
        None
    """

    metrics_path = cfg.analysis_dir / cfg.metrics_name
    seat_csv = cfg.analysis_dir / "seat_advantage.csv"
    seat_parquet = cfg.analysis_dir / "seat_advantage.parquet"
    iso_paths = sorted((cfg.analysis_dir / "data").glob("*p/*_isolated_metrics.parquet"))
    metrics_golden = GOLDEN_ROOT / "metrics.csv"
    seat_golden = GOLDEN_ROOT / "seat_advantage.csv"

    assert_parquet_golden(metrics_path, metrics_golden, update=update_goldens, sort_by=["n_players", "strategy"])
    assert_csv_golden(seat_csv, seat_golden, update=update_goldens, sort_by=["seat"])
    assert_parquet_golden(seat_parquet, seat_golden, update=update_goldens, sort_by=["seat"])

    for iso_path in iso_paths:
        golden_iso = GOLDEN_ROOT / iso_path.relative_to(cfg.analysis_dir).with_suffix(".csv")
        assert_parquet_golden(iso_path, golden_iso, update=update_goldens, sort_by=["strategy"])

    stamp = cfg.analysis_dir / "metrics.done.json"
    assert_stamp_has_paths(
        stamp,
        expected_inputs=[cfg.curated_parquet, *[cfg.results_dir / f"{n}_players" / f"{n}p_metrics.parquet" for n in _MANIFEST_ROWS]],
        expected_outputs=[metrics_path, seat_csv, seat_parquet, *iso_paths],
    )


SIM_KWARGS = dict(_SIM_KWARGS)
STRATEGIES = tuple(_STRATEGIES)
