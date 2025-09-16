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

logger = logging.getLogger(__name__)


def _concat_row_shards(out_dir: Path, n_players: int) -> None:
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
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out_path = out_dir / f"{n_players}p_rows.parquet"
    with atomic_path(str(out_path)) as tmp_path:
        df.to_parquet(tmp_path)
    shutil.rmtree(row_dir, ignore_errors=True)
    del df  # free memory before returning


def _combo_complete(out_dir: Path, n_players: int, *, force_clean: bool = False) -> bool:
    """Return ``True`` if the results for this combo already exist."""
    ckpt = out_dir / f"{n_players}p_checkpoint.pkl"
    rows = out_dir / f"{n_players}p_rows.parquet"
    row_dir = out_dir / f"{n_players}p_rows"

    # If the final parquet is present we consider the block complete;
    # leftover shard directories are debris but may be preserved unless forced.
    if rows.is_file() and row_dir.exists():
        if force_clean:
            logger.warning("Deleting existing row directory %s", row_dir)
            shutil.rmtree(row_dir, ignore_errors=True)
        else:
            logger.warning(
                "Row directory %s exists alongside final parquet; "
                "rerun with --force-clean to remove it",
                row_dir,
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
    PLAYERS = [2, 3, 4, 5, 6, 8, 10, 12]
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
    if force_clean:
        logger.info("force_clean=True - partial row directories will be removed when found")
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
            print(f"Must have two or more players, skipping {n_players} player(s)")
            continue
        nshuf = shuffles_required(n_players)
        gps = GRID // n_players  # games per shuffle
        ngames = nshuf * gps

        out_dir = BASE_OUT / f"{n_players}_players"
        (out_dir).mkdir(parents=True, exist_ok=True)

        if _combo_complete(out_dir, n_players, force_clean=force_clean):
            print(f"↩ skipping {n_players}-player - already done", flush=True)
            continue

        _reset_partial(out_dir, n_players)

        print(
            f"▶ {n_players:>2}-player  |  {nshuf:>7,} shuffles  "
            f"{gps:>5} games per shuffle  →  {ngames / 1e6:5.2f} M games",
            flush=True,
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
        print(f"✅ finished {n_players}-player in {dt / 60:5.1f} min\n", flush=True)
        print(f"Concatenating and cleaning up {n_players}-player parquet shards...")
        _concat_row_shards(out_dir, n_players)
        print(f"{n_players}-player parquet shards consolidation process complete.")
    print("🏁  All table sizes completed.")
