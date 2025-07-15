#!/usr/bin/env python
"""
run_full_field.py  -  Phase-1 full-grid screen for all table sizes
   • BH FDR (two-sided) Q = 0.02   →  zα = Φ⁻¹(0.99) ≈ 2.326
   • Power               = 0.95    →  zβ = Φ⁻¹(0.05) ≈ 1.645
   • Detectable lift Δ   = 0.03     (3-percentage-point edge)
"""

import importlib
import multiprocessing as mp
from math import ceil
from pathlib import Path
from time import perf_counter

from scipy.stats import norm

# Seed 1 Global Config
    # # ────────── GLOBAL CONFIG ─────────────────────────────────────────
    # PLAYERS          = [2, 3, 4, 5, 6, 8, 10, 12]
    # GRID             = 8_160           # total strategies
    # DELTA            = 0.03            # abs lift to detect
    # POWER            = 0.95            # 1 – β
    # Q_FDR            = 0.02            # BH, two-sided
    # GLOBAL_SEED      = 42
    # JOBS             = None            # None → all logical cores
    # BASE_OUT         = Path("data/results_seed_42")
    # BASE_OUT.mkdir(parents=True, exist_ok=True)
    # # ------------------------------------------------------------------

    # # pre-compute critical z-scores
    # Z_ALPHA = norm.isf(Q_FDR / 2)      # two-sided BH
    # Z_BETA  = norm.isf(1 - POWER)      # power target

def main():
    import farkle.run_tournament as rt # required for main hook  # noqa: I001
    # ────────── GLOBAL CONFIG ─────────────────────────────────────────
    PLAYERS          = [2, 3, 4, 5, 6, 8, 10, 12]
    GRID             = 8_160           # total strategies
    DELTA            = 0.03            # abs lift to detect
    POWER            = 0.95            # 1 – β
    Q_FDR            = 0.02            # BH, two-sided
    GLOBAL_SEED      = 0
    JOBS             = None            # None → all logical cores
    BASE_OUT         = Path("data/results_seed_0")
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------

    # pre-compute critical z-scores
    Z_ALPHA = norm.isf(Q_FDR / 2)      # two-sided BH
    Z_BETA  = norm.isf(1 - POWER)      # power target

    def shuffles_required(n_players: int) -> int:
        """Return observations/strategy (≡ shuffles) for given table size."""
        p0   = 1 / n_players
        var  = p0 * (1 - p0) + (p0 + DELTA) * (1 - p0 - DELTA)
        n    = ((Z_ALPHA + Z_BETA) ** 2 * var) / (DELTA ** 2)
        return ceil(n)

    mp.set_start_method("spawn", force=True)

    for n in PLAYERS:
        nshuf  = shuffles_required(n)
        gps    = GRID // n # games per shuffle
        ngames = nshuf * gps

        out_dir = BASE_OUT / f"{n}_players"
        (out_dir).mkdir(parents=True, exist_ok=True)

        print(f"▶ {n:>2}-player  |  {nshuf:>7,} shuffles  "
            f"{gps:>5} gps  →  {ngames/1e6:5.2f} M games", flush=True)

        # fresh module copy for clean monkey-patch
        rt = importlib.reload(rt)
        rt.N_PLAYERS         = n # type: ignore
        rt.GAMES_PER_SHUFFLE = gps # type: ignore
        rt.NUM_SHUFFLES      = nshuf # type: ignore

        t0 = perf_counter()
        rt.run_tournament(
            global_seed          = GLOBAL_SEED,
            checkpoint_path      = out_dir / f"{n}p_checkpoint.pkl",
            n_jobs               = JOBS,
            collect_metrics      = True,
            row_output_directory = out_dir / f"{n}p_rows",
        )
        dt = perf_counter() - t0
        print(f"✅ finished {n}-player in {dt/60:5.1f} min\n", flush=True)

    print("🏁  All table sizes completed.")

if __name__=='__main__':
    main()