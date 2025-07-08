#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all_tournaments.sh
# ---------------------------------------------------------------------------
# Launch a Farkle Monte-Carlo tournament for each desired table size.
# Every run is isolated in its own sub-folder under ./results/.
#
#  • EDIT the PLAYERS array if you need different table sizes.
#  • The inner Python snippet tweaks run_tournament’s module-level constants
#    (N_PLAYERS and GAMES_PER_SHUFFLE) before calling run_tournament().
# ---------------------------------------------------------------------------

set -euo pipefail

# -------- configure here ----------------------------------------------------
PLAYERS=(2 3 4 5 6 8 10 12)          # eight values → eight folders
SEED=42                             # global RNG seed for full reproducibility
JOBS="$(nproc)"                    # worker processes (all CPUs by default)
BASE_OUT="data/results"                 # top-level output directory
# ---------------------------------------------------------------------------

mkdir -p "$BASE_OUT"

for N in "${PLAYERS[@]}"; do
  OUTDIR="$BASE_OUT/${N}_players"
  mkdir -p "$OUTDIR"
  echo "▶️  Starting tournament with ${N} players (output → ${OUTDIR})"

# --- inside the loop ---
  python - <<PYTHON
import pathlib, multiprocessing as mp, farkle.run_tournament as rt

rt.N_PLAYERS = $N
rt.GAMES_PER_SHUFFLE = 8160 // rt.N_PLAYERS      # keep logic in sync

mp.set_start_method("spawn", force=True)
rt.run_tournament(
    global_seed=$SEED,
    checkpoint_path=pathlib.Path("${OUTDIR}") / "checkpoint.pkl",
    n_jobs=$JOBS,
    collect_metrics=True,
    row_output_directory=pathlib.Path("${OUTDIR}") / "rows"   # <- changed
)
PYTHON


  echo "✅  Finished ${N}-player run"
done

echo "🏁  All tournaments completed."
