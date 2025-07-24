# src/farkle/run_full_field.py
"""
run_full_field.py  -  Phase-1 full-grid screen for all table sizes
   • BH FDR (two-sided) Q = 0.02   →  zα = Φ⁻¹(0.99) ≈ 2.326
   • Power               = 0.95    →  zβ = Φ⁻¹(0.05) ≈ 1.645
   • Detectable lift Δ   = 0.03     (3-percentage-point edge)

Run with: python -m farkle.run_full_field
"""

import multiprocessing as mp
import shutil
from math import ceil
from pathlib import Path
from time import perf_counter

import pandas as pd
from scipy.stats import norm

from farkle.scoring_lookup import build_score_lookup_table

SCORE_TABLE = build_score_lookup_table()


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
    df.to_parquet(out_dir / f"{n_players}p_rows.parquet")
    shutil.rmtree(row_dir, ignore_errors=True)


def _combo_complete(out_dir: Path, n_players: int) -> bool:
    """Return ``True`` if the results for this combo already exist."""

    ckpt = out_dir / f"{n_players}p_checkpoint.pkl"
    rows = out_dir / f"{n_players}p_rows.parquet"
    row_dir = out_dir / f"{n_players}p_rows"
    return ckpt.is_file() and rows.is_file() and not row_dir.exists()


def _reset_partial(out_dir: Path, n_players: int) -> None:
    """Delete partial outputs for a table size if needed."""

    row_dir = out_dir / f"{n_players}p_rows"
    rows = out_dir / f"{n_players}p_rows.parquet"
    if row_dir.exists() and not rows.exists():
        shutil.rmtree(row_dir, ignore_errors=True)
        ckpt = out_dir / f"{n_players}p_checkpoint.pkl"
        if ckpt.exists():
            ckpt.unlink()


def main():
    """Run tournaments for each table size defined in ``PLAYERS``.

    The function iterates over the configured ``PLAYERS`` list. For each
    table size it computes the number of required shuffles, runs
    :func:`run_tournament` with those parameters and finally merges any
    parquet shards produced by the workers into a single file.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    import farkle.run_tournament as tournament_mod  # required for main hook  # noqa: I001

    # ────────── GLOBAL CONFIG ─────────────────────────────────────────
    PLAYERS = [2, 3, 4, 5, 6, 8, 10, 12]
    GRID = 8_160  # total strategies
    DELTA = 0.03  # abs lift to detect
    POWER = 0.95  # 1 – β
    Q_FDR = 0.02  # BH, two-sided
    GLOBAL_SEED = 42
    JOBS = None  # None → all logical cores
    BASE_OUT = Path(f"data/results_seed_{GLOBAL_SEED}")
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
        nshuf = shuffles_required(n_players)
        gps = GRID // n_players  # games per shuffle
        ngames = nshuf * gps

        out_dir = BASE_OUT / f"{n_players}_players"
        (out_dir).mkdir(parents=True, exist_ok=True)

        if _combo_complete(out_dir, n_players):
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
        print(f"Cleaning up {n_players}-player parquet shards...")
        _concat_row_shards(out_dir, n_players)
        print(f"{n_players}-player parquet shards consolidation process complete.")
    print("🏁  All table sizes completed.")


if __name__ == "__main__":
    # run: python -m farkle.run_full_field
    main()
