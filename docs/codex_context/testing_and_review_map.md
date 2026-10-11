# Codex Testing and Review Map

Task 10 adds `test_windows_launcher_qualification.py` for the actual public CLI
under a strict Windows Job Object, normal rules, an integration profile, S:
results/C: scratch, a cap stop, explicit exact-plan consent, byte-stable replay
and missing-Markdown recovery without replaying producer Parquets.
`test_ingest_coalescing.py` streams the production high-k physical dimensions
without production games and checks every ordered coordinate, row count and
footer admission. Arrow coalescing tests include nullable lists, byte ceilings,
chunk compaction and invalid-input rejection. Final evidence, static checks and
separate historical/development boundaries are in
[Task 10](../implementation/task10_report.md). The four former terminology
findings are resolved in Task 10; the Task 8 account below retains its original
validation boundary.

Task 10 accepted all 1,584 collected cases with no omissions/waivers, across
integration/top-level, broad-unit and complete affected-module followups, plus
the fresh 2,153.02-second Windows launcher. Low fixture worker thresholds use
fresh processes rather than larger limits; the unavailable-Git identity check
needs a non-Git temporary directory. The historical Task 5A capacity tests use
their declared ignored 54/55 dataset read-only, not as new scientific evidence.
The simulation mutation control now uses `publish_v3_simulation_run`; bare
unsealed checkpoint bytes are deliberately rejected. Canonical producer
files remain in replay comparisons; CLI `log.txt` joins the append-only views.

Task 8 adds `test_analysis_guide.py` for frozen coverage, actual extraction, support/status distinctions, limitations, JSON-only rendering, authentication, no-force reuse, corruption and interrupted rendering recovery. `tests/helpers/task8_fixture.py` is explicitly synthetic rendering evidence. `task8_oracle.py` reconciles guide values, schemas, support/status counts, plan authorization and diagnostic coverage to actual producers in the raw workflow oracle. Final acceptance passes 153 focused tests plus the fresh raw oracle (984.954 seconds), both required preflights, root/root/pair audit and unchanged no-force replay. Signed nonfinite source values retain explicit JSON tokens; artifact identity digests and byte digests are checked separately. Acceptance and snapshot identities are in [Task 8](../implementation/task8_report.md) and [validation](../implementation/task8_validation.json). The four inherited terminology findings remain unresolved.

Task 7 adds `test_seat_behavior.py`, `test_parameter_responses.py`, and
`test_seat_behavior_workers.py`: raw-ratio arithmetic, support/zero semantics,
matched contrasts and exclusions, slice/worker invariance, missing-unit recovery,
interruption, byte bounds and observed memory/scratch. The full pipeline oracle
also checks these products independently against tiny raw rows. Structural
reporting tests use explicitly synthetic `tests/helpers/task7_fixture.py`;
those fixtures are not producer evidence. Accepted results are in
[Task 7](../implementation/task7_report.md).

Use `.venv` and distinguish a passing regression test from evidence that a
statistical claim is valid.

Task 6 adds `test_h2h_authorization.py`: exact-plan approval, prompt/refusal,
noninteractive override, preserved partial progress, interrupted authorization
publication, source binding refresh, and final audit. The full raw pipeline
oracle now approves 24 maximum attempts against an unchanged cap of 23 and
replays without prompting. The structural toy oracle supplies explicitly
insufficient synthetic RNG evidence; it does not replace real preflight.
See [Task 6](../implementation/task6_report.md).

Task 5C adds `tests/unit/analysis/test_rng_monitor.py` and the standalone
`scripts/task5c_preflight.py` bounded validation command. The complete pipeline
oracle executes real preflights and verifies PE/MC/MR ownership. Its original
240-second workflow budget excludes separately measured new preflight time;
each preflight retains its own 1,800-second ceiling. See the
[Task 5C report](../implementation/task5c_report.md).

Exploratory notebooks are archived research records and are excluded from the
production Ruff and Black gates. Maintained Python source, tests, and scripts
remain in both gates.

## High-value targets

- Workload/RNG/resume: `tests/unit/simulation/test_workload_planner.py`,
  `test_run_tournament*.py`, and `tests/unit/utils/test_random.py`.
- Simulation process topology/sequencing: `tests/unit/utils/test_parallel_files.py`,
  `tests/unit/orchestration/test_seed_workflows.py`,
  `tests/unit/simulation/test_runner_branches.py`,
  `tests/unit/simulation/test_simulation_publication_v3.py`, and
  `tests/unit/scripts/test_benchmark_task4c_simulation_execution.py`. The real
  protected Windows-spawn acceptance driver is
  `scripts/benchmark_task4c_simulation_execution.py`.
- Row/turn arithmetic: `tests/unit/game/test_engine*.py`,
  `tests/unit/simulation/test_simulation.py`, and
  `tests/unit/analysis/test_all_player_metrics.py`.
- Scopes/sidecars/state: `tests/unit/utils/test_artifact_contract.py`,
  `test_stage_registry.py`, `test_stage_state.py`, `test_release_audit.py`,
  `test_authenticated_graph.py`, and `test_partitioned_stage.py`. Completion
  namespace tests use the production `_bootstrap_top_n_ranges/units` and
  `_joint_discrepancy_ranges/units` layouts.
- Performance/screening: `test_performance.py` and `test_screening.py`.
- First-release completed-outcome precision: `test_outcome_precision.py` checks
  shared root/k batch draws against direct sampled-index arithmetic, sparse and
  undefined support, loss complements, authenticated range recovery, scratch
  interruption and tile repartition. `scripts/benchmark_outcome_precision.py`
  records bounded fixture timing/RSS separately from source preparation.
- Seat/game/RNG/roll: `test_seat_analysis.py`, `test_game_stats*.py`,
  `test_rng_diagnostics_branches.py`, and `test_roll_enumeration.py`.
- First-release RNG preflight: [Task 5A](../implementation/task5a_contract.md)
  maps ten correctness requirements to existing tests in its
  [recipe](../implementation/task5a_recipe.json). This is a specification,
  not a preflight execution or demonstrated monitor. The new all-k worker/resume
  matrix, production-topology applicability, fault/clean capability, and optional
  exhaustive-stage integration remain Tasks 5B/5C. Descriptive band excursions
  are not automatic failures, and missing capability means insufficient evidence.
- Task 5B: `test_batch_monitor.py` checks direct arithmetic, strict thresholds,
  support withholding, count integrity, deterministic fixtures, hidden within-batch
  dependence, common root faults, and unit recovery. The config-seeded driver
  `scripts/task5b_capability.py` retains per-fixture evidence for 64 clean families
  and strong/weak alternatives at 30 and 100 batches. Acceptance and scope are in
  [the protocol](../implementation/task5b_protocol.md); measured capability and
  false alerts are not a universal independence or replacement-coverage guarantee.
- TrueSkill/HGB: `test_run_trueskill_*.py`, `test_trueskill_screening.py`, and
  `test_hgb_feat.py`.
- Root/H2H/dominance: `test_root_stability.py`, `test_candidate_family.py`,
  `test_h2h_schedule.py`, `test_h2h_inference.py`, and `test_dominance.py`.
- First-release H2H conclusion semantics: `test_h2h_conclusions.py` checks mixed
  advantage/equivalence, strict boundaries, nonviability, nondefault tolerances,
  nontransitive graph handling, and unchanged actual schedules across margins.
  It also reproduces the current 150-finalist exact power/interval illustration;
  `scripts/task4_h2h_example.py` retains the numerical evidence without simulation.
  `test_structure_reporting.py` checks configured tolerance and actual intervals
  in authenticated JSON/Markdown with both conclusions visible.
- Task 7A artifact-density diagnostics:
  `tests/unit/scripts/test_benchmark_task7a_artifact_density.py` and
  `scripts/benchmark_task7a_artifact_density.py`. The retained owned root is
  `data/farkle-task7a-artifact-density-v1`; the driver inventories accepted trees
  read-only and requires exact logical equality for benchmark-only layouts.
- Task 7B closed bundle gate:
  `tests/unit/scripts/test_benchmark_task7b_h2h_bundling.py`,
  `scripts/benchmark_task7b_h2h_bundling.py`, and
  `docs/remediation/task7b_h2h_bundling_report.md`. Retained evidence covers
  three baseline and candidate repetitions for 180 and 760 logical blocks at
  12-worker Windows spawn. The verdict is `performance_gate_failed`; production
  tests and source remain on the per-block layout.
- Agreement/reporting: `test_structure_agreement.py` and
  `test_structure_reporting.py`.
- Full workflow: `tests/integration/test_structure_toy_oracle.py` and
  `tests/integration/test_simulation_to_report_oracle.py`; the latter requires
  a fresh-byte final audit, successful pipeline health and `run_end`, then a
  byte-stable ordinary replay.

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
.\.venv\Scripts\python scripts/check_structure_release.py --artifact-root data/results_seed_pair_A_B/results_seed_A/analysis
.\.venv\Scripts\python -m pytest tests/integration/test_structure_toy_oracle.py
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m black --check .
.\.venv\Scripts\python -m mypy src
.\.venv\Scripts\python -m pyright
```

The release audit validates checked-in runnable configs, confirms retired
source entry points are absent, and checks canonical artifact/sidecar pairing.
At least one explicit fresh `--artifact-root` is required; repeat the option to
audit multiple independent roots.
The toy oracle covers two roots, multiple k values, immutable H2H blocks,
interruption, resume with a different worker count, hashes, decisions, fronts,
cycles, report claims, and sidecar completeness.
