# Codex Testing and Review Map

Use `.venv` and distinguish a passing regression test from evidence that a
statistical claim is valid.

Exploratory notebooks are archived research records and are excluded from the
production Ruff and Black gates. Maintained Python source, tests, and scripts
remain in both gates.

## High-value targets

- Workload/RNG/resume: `tests/unit/simulation/test_workload_planner.py`,
  `test_run_tournament*.py`, and `tests/unit/utils/test_random.py`.
- Row/turn arithmetic: `tests/unit/game/test_engine*.py`,
  `tests/unit/simulation/test_simulation.py`, and
  `tests/unit/analysis/test_all_player_metrics.py`.
- Scopes/sidecars/state: `tests/unit/utils/test_artifact_contract.py`,
  `test_stage_registry.py`, `test_stage_state.py`, and `test_release_audit.py`.
- Performance/screening: `test_performance.py` and `test_screening.py`.
- Seat/game/RNG/roll: `test_seat_analysis.py`, `test_game_stats*.py`,
  `test_rng_diagnostics_branches.py`, and `test_roll_enumeration.py`.
- TrueSkill/HGB: `test_run_trueskill_*.py`, `test_trueskill_screening.py`, and
  `test_hgb_feat.py`.
- Root/H2H/dominance: `test_root_stability.py`, `test_candidate_family.py`,
  `test_h2h_schedule.py`, `test_h2h_inference.py`, and `test_dominance.py`.
- Agreement/reporting: `test_structure_agreement.py` and
  `test_structure_reporting.py`.
- Full workflow: `tests/integration/test_structure_toy_oracle.py`.

## Review questions

For statistical code, verify:

1. the declared estimand and conditioning;
2. formula fidelity on a small hand calculation;
3. complete root/k support and declared weights;
4. uncertainty replication unit and dependence assumptions;
5. multiplicity and decision rule;
6. sidecar method/scope compatibility;
7. report language permitted by the evidence.

For resumable code, compare coordinate manifests and logical outputs across
worker counts, interruption, and resume. File byte identity is required where
the format is deterministic; otherwise compare canonical logical content.

## Release gates

```powershell
.\.venv\Scripts\python scripts/check_terminology.py
.\.venv\Scripts\python scripts/check_structure_release.py
.\.venv\Scripts\python -m pytest tests/integration/test_structure_toy_oracle.py
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m black --check .
.\.venv\Scripts\python -m mypy src
.\.venv\Scripts\python -m pyright
```

The release audit validates checked-in runnable configs, confirms retired
source entry points are absent, and checks canonical artifact/sidecar pairing.
The toy oracle covers two roots, multiple k values, immutable H2H blocks,
interruption, resume with a different worker count, hashes, decisions, fronts,
cycles, report claims, and sidecar completeness.
