# src/farkle/analysis/hgb_feat.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from farkle.analysis import run_hgb as _hgb
from farkle.config import AppConfig

LOGGER = logging.getLogger(__name__)


def _unique_players(metrics_path: Path, hints: Iterable[int]) -> list[int]:
    players: set[int] = set(int(p) for p in hints)
    if not metrics_path.exists():
        return sorted(players)

    for column in ("n_players", "players"):
        try:
            table = pq.read_table(metrics_path, columns=[column])
        except (pa.lib.ArrowInvalid, KeyError, FileNotFoundError):
            continue
        except Exception:  # noqa: BLE001 - best effort read
            continue
        arr = table.column(0)
        players.update(int(v) for v in arr.to_pylist() if v is not None)
    return sorted(players)


def _latest_mtime(paths: Iterable[Path]) -> float:
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes, default=0.0)


def run(cfg: AppConfig) -> None:
    analysis_dir = cfg.analysis_dir
    metrics_path = analysis_dir / cfg.metrics_name
    ratings_path = analysis_dir / _hgb.RATINGS_NAME
    json_out = analysis_dir / "hgb_importance.json"

    players = _unique_players(metrics_path, cfg.sim.n_players_list)
    importance_paths = [
        analysis_dir / _hgb.IMPORTANCE_TEMPLATE.format(players=p) for p in players
    ]

    inputs = [cfg.curated_parquet, metrics_path, ratings_path]
    latest_input = _latest_mtime(inputs)
    outputs = [json_out, *importance_paths]

    if outputs and all(p.exists() and p.stat().st_mtime >= latest_input for p in outputs):
        LOGGER.info(
            "HGB feature importance up-to-date",
            extra={"stage": "hgb", "path": str(json_out)},
        )
        return

    LOGGER.info(
        "HGB feature importance running",
        extra={
            "stage": "hgb",
            "analysis_dir": str(analysis_dir),
        },
    )
    _hgb.run_hgb(root=analysis_dir, output_path=json_out)
    LOGGER.info(
        "HGB feature importance complete",
        extra={"stage": "hgb", "path": str(json_out)},
    )
