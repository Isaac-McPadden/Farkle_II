# src/farkle/run_full_field.py
"""
Full-field tournament sweep across all supported table sizes.

This module previously exposed a command line interface; the control flow has
been refactored into :func:`run_full_field` so the new configuration-driven CLI
can call it directly.
"""

import logging
import multiprocessing as mp
import shutil
from math import ceil
from pathlib import Path
from time import perf_counter

import pandas as pd
from scipy.stats import norm

from farkle.game.scoring_lookup import build_score_lookup_table
from farkle.utils.writer import atomic_path

SCORE_TABLE = build_score_lookup_table()

LOGGER = logging.getLogger(__name__)


def _concat_row_shards(out_dir: Path, n_players: int) -> int:
    """Combine row shard files and remove the temporary directory.

    If no shard files are found the function simply returns without
    writing anything. Reading/writing parquet requires ``pyarrow`` to be
    installed.

    out_dir is the parent folder of the row folder and where the finished
    parquet is saved.
    """
    row_dir = out_dir / f"{n_players}p_rows"
    files = sorted(row_dir.glob("*.parquet"))
    if not files:
        LOGGER.debug(
            "No row shards to consolidate",
            extra={"stage": "full_field", "n_players": n_players, "path": str(row_dir)},
        )
        return 0
    frames: list[pd.DataFrame] = []
    total_rows = 0
    for shard in files:
        df = pd.read_parquet(shard)
        rows = len(df)
        total_rows += rows
        LOGGER.debug(
            "Loaded row shard",
            extra={
                "stage": "full_field",
                "n_players": n_players,
                "path": str(shard),
                "rows": rows,
            },
        )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    out_path = out_dir / f"{n_players}p_rows.parquet"
    with atomic_path(str(out_path)) as tmp_path:
        df.to_parquet(tmp_path)
    shutil.rmtree(row_dir, ignore_errors=True)
    del df  # free memory before returning
    LOGGER.info(
        "Row shards consolidated",
        extra={
            "stage": "full_field",
            "n_players": n_players,
            "rows": total_rows,
            "path": str(out_path),
        },
    )
    return total_rows


def _combo_complete(out_dir: Path, n_players: int, *, force_clean: bool = False) -> bool:
    """Return ``True`` if the results for this combo already exist."""
    ckpt = out_dir / f"{n_players}p_checkpoint.pkl"
    rows = out_dir / f"{n_players}p_rows.parquet"
    row_dir = out_dir / f"{n_players}p_rows"

    # If the final parquet is present we consider the block complete;
    # leftover shard directories are debris but may be preserved unless forced.
    if rows.is_file() and row_dir.exists():
        if force_clean:
            LOGGER.warning(
                "Deleting existing row directory",
                extra={"stage": "full_field", "n_players": n_players, "path": str(row_dir)},
            )
            shutil.rmtree(row_dir, ignore_errors=True)
        else:
            LOGGER.warning(
                "Row directory exists alongside final parquet",
                extra={
                    "stage": "full_field",
                    "n_players": n_players,
                    "path": str(row_dir),
                    "hint": "rerun with --force-clean to remove",
                },
            )
    return ckpt.is_file() and rows.is_file()


def _reset_partial(out_dir: Path, n_players: int) -> None:
    """Delete partial outputs for a table size if needed."""

    row_dir = out_dir / f"{n_players}p_rows"
    rows = out_dir / f"{n_players}p_rows.parquet"
    if row_dir.exists() and not rows.exists():
        shutil.rmtree(row_dir, ignore_errors=True)
        ckpt = out_dir / f"{n_players}p_checkpoint.pkl"
        if ckpt.exists():
            ckpt.unlink()
        LOGGER.info(
            "Removed partial outputs",
            extra={
                "stage": "full_field",
                "n_players": n_players,
                "path": str(out_dir),
            },
        )


def run_full_field(
    results_dir: Path | str = Path("results_seed_0"), *, force_clean: bool = False
) -> None:
    """Run tournaments for each table size defined in ``PLAYERS``.

    The function iterates over the configured ``PLAYERS`` list. For each table
    size it computes the number of required shuffles, runs
    :func:`run_tournament` with those parameters and finally merges any parquet
    shards produced by the workers into a single file.
    """
    import farkle.simulation.run_tournament as tournament_mod  # required for main hook  # noqa: I001

    # ────────── GLOBAL CONFIG ─────────────────────────────────────────
    PLAYERS = [2, 3, 4, 5, 6, 7, 10, 12]
    GRID = 8_160  # total strategies
    DELTA = 0.03  # abs lift to detect
    POWER = 0.95  # 1 – β
    Q_FDR = 0.02  # BH, two-sided
    GLOBAL_SEED = 0
    JOBS = None  # None → all logical cores
    BASE_OUT = Path(results_dir)
    if not BASE_OUT.exists():
        candidate = Path("data") / BASE_OUT
        if candidate.exists():
            BASE_OUT = candidate
    LOGGER.info(
        "Full-field sweep starting",
        extra={
            "stage": "full_field",
            "results_dir": str(BASE_OUT),
            "global_seed": GLOBAL_SEED,
            "n_jobs": JOBS,
            "players": PLAYERS,
        },
    )
    if force_clean:
        LOGGER.info(
            "force_clean enabled",
            extra={"stage": "full_field", "results_dir": str(BASE_OUT)},
        )
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------

    # pre-compute critical z-scores
    Z_ALPHA = norm.isf(Q_FDR / 2)  # two-sided BH
    Z_BETA = norm.isf(1 - POWER)  # power target

    def shuffles_required(n_players: int) -> int:
        """Return observations/strategy (≡ shuffles) for given table size."""
        p0 = 1 / n_players
        var = p0 * (1 - p0) + (p0 + DELTA) * (1 - p0 - DELTA)
        n = ((Z_ALPHA + Z_BETA) ** 2 * var) / (DELTA**2)
        return ceil(n)

    mp.set_start_method("spawn", force=True)

    for n_players in PLAYERS:
        if n_players < 2:
            LOGGER.warning(
                "Table size below minimum",
                extra={"stage": "full_field", "n_players": n_players},
            )
            continue
        nshuf = shuffles_required(n_players)
        gps = GRID // n_players  # games per shuffle
        ngames = nshuf * gps
        LOGGER.debug(
            "Calculated batch sizing",
            extra={
                "stage": "full_field",
                "n_players": n_players,
                "shuffles": nshuf,
                "games_per_shuffle": gps,
                "total_games": ngames,
            },
        )

        out_dir = BASE_OUT / f"{n_players}_players"
        (out_dir).mkdir(parents=True, exist_ok=True)

        if _combo_complete(out_dir, n_players, force_clean=force_clean):
            LOGGER.info(
                "Skipping completed table size",
                extra={"stage": "full_field", "n_players": n_players, "path": str(out_dir)},
            )
            continue

        _reset_partial(out_dir, n_players)
        LOGGER.info(
            "Starting tournament sweep",
            extra={
                "stage": "full_field",
                "n_players": n_players,
                "shuffles": nshuf,
                "games_per_shuffle": gps,
                "total_games": ngames,
                "path": str(out_dir),
            },
        )

        # update shuffle count on the already-imported module
        tournament_mod.NUM_SHUFFLES = nshuf  # type: ignore

        t0 = perf_counter()
        tournament_mod.run_tournament(
            n_players=n_players,
            num_shuffles=nshuf,
            global_seed=GLOBAL_SEED,
            checkpoint_path=out_dir / f"{n_players}p_checkpoint.pkl",
            n_jobs=JOBS,
            collect_metrics=True,
            row_output_directory=out_dir / f"{n_players}p_rows",
        )
        dt = perf_counter() - t0
        LOGGER.info(
            "Tournament sweep completed",
            extra={
                "stage": "full_field",
                "n_players": n_players,
                "elapsed_s": dt,
                "total_games": ngames,
            },
        )
        rows = _concat_row_shards(out_dir, n_players)
        if rows:
            LOGGER.debug(
                "Row shard consolidation summary",
                extra={
                    "stage": "full_field",
                    "n_players": n_players,
                    "rows": rows,
                    "path": str(out_dir / f"{n_players}p_rows.parquet"),
                },
            )
    LOGGER.info("Full-field sweep complete", extra={"stage": "full_field"})
