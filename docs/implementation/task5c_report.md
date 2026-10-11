# Task 5C — required preflight and lightweight production monitoring

Contract: September 24 `first_release_task0_v1`. This implements the Task 5A
correctness workload and Task 5B descriptive monitor, with authenticated evidence
under the existing RNG diagnostics stage. Validation results are recorded in
`task5c_validation.json`; they are separate from production observations.

## Runtime behavior

`analysis.disable_rng_diagnostics: true` now omits **only exhaustive game-level
lags**. The stage remains required and publishes `preflight_evidence.json`,
`monitor_coverage.json`, and `production_monitor.parquet`. Existing false values
still request exhaustive diagnostics. The explicit analysis-call override has
the same scope. A zero matchup cap remains the internal unlimited convention;
it is never used as a disable switch. No simulation target, batch boundary,
bootstrap replicate count, candidate selection, or H2H allocation changed.

The stage depends on combine and metrics. Its completion owns the three final
artifacts, per-k monitor units, and the exact retained preflight inventory.
Required omissions/corruption invalidate completion. An omitted optional lag
path is `omitted_by_design`; an enabled capped path retains `blocked_by_cap`.
Independent-root comparisons and ordinary workflow telemetry remain in place.
Reporting contract 7 includes each root's preflight/monitor status, warnings,
coverage and exhaustive status. The byte-deep audit validates proof inventories.
RNG cache version is 7; preflight/monitor method versions are 1. Legacy lag
arithmetic remains method 4.

## Bounded preflight and topology extension

The original matrix is unchanged: roots 1101/2202, all eight required k values,
120 deterministic legal mega-grid strategies, four shuffles, two batches,
target 10,000, max rounds 200, and the declared safety override. It compares
canonical rows, exact integer counts, and scientific summaries for one worker,
two workers, and interruption followed by a two-to-one-worker resume. One
shuffle is an atomic data/sidecar/completion unit. Injection-start records prevent
an incomplete injection from being repeated automatically. Missing/corrupt
ordinary units are individually recomputed; valid units retain their bytes.

`target_workers_v1` is an explicit correctness extension for a requested worker
count from 3 through 16. It executes one additional complete matrix at that
count and compares against the one-worker reference. At 16 workers this adds
1,688 attempts and 7,680 player exposures: the declared matrix budget becomes
7,174 attempts and 32,640 exposures including the controlled replay. Supplemental
oracles retain their separately bounded inputs. This is topology correctness
evidence, not a production throughput qualification.

All 27 frozen supplemental pytest selections must execute without failed,
errored, skipped, or missing cases. Their parameterizations and invocation are
retained in typed evidence. The seeded capability assessment reruns the frozen
64-seed, six-scenario, two-batch-count recipe, retaining every check result in
64 bounded per-seed Parquet units with typed nested check records. This avoids
repeated structural parsing of large JSON records during graph authentication.
Its protocol hashes and acceptance thresholds are
checked before assessment. Validation games and synthetic counts remain in
`diagnostics/preflight`; no inference consumer reads them.

The 1,800-second deadline is checked at durable unit boundaries; the supplemental
test subprocess has a timeout that terminates its process tree. Individual game
units retain the engine's finite safety limits. Evidence records elapsed time,
sampled process-tree RSS and parent I/O counters. RSS sampling is at unit
completion, not a continuous maximum; parent I/O explicitly excludes worker I/O.
There is no fixed 1 GiB validity claim. Configured worker counts outside 1..16
or an automatic/unspecified count fail with an applicability explanation.

The standalone command `.venv/Scripts/python.exe scripts/task5c_preflight.py`
runs only this validation workload under `data/task5c-validation_seed_102`.
It does not run production simulation. The regular RNG stage produces its own
current-run proof. Reuse requires current code (including dirty fingerprint),
config, plan, environment, sources, sidecars and completion. This implementation
conservatively revalidates per root; it does not import foreign completion stamps.

## Monitor scope and interpretation

For each root/k, select at most 16 canonical strategy IDs before reading outcomes
using Task 5B's evenly spaced rule. Sum certified actual-seat batch counts into
seat 0; retain original batch IDs. The six responses are W/A, S/A, W/C, D/C, D/L
and V/W. Full batch/roster support and conservation are checked before arithmetic.
Independent k units use the existing process policy. Input reads and selected
vectors are byte bounded; final output is at most 1,536 diagnostic rows per root.

The frozen lag-one and half-drift rules require B=30 or B=100. Zero-denominator
batches are not dropped; constant, undefined, short or unassessed series report
`insufficient_evidence`. An effect warning is `investigate`, surfaces in logging
and reporting, and leaves sampling targets unchanged. Unsupported per-production
check capability counts are null; actual assessment counts are separately
reported at their assessed family/scenario level.

The two-batch correctness matrix is explicitly insufficient statistical
dependence evidence. No result proves independence. Batch summaries cannot see
within-batch faults preserving totals; unselected strategies, actual-seat series,
higher lags and rare tails remain outside this monitor. Agreement between roots
can coexist with a common fault. Thresholds do not claim calibrated family-wise
error control and do not replace exhaustive diagnostics.

## Validation boundaries

The October 1 standalone validation completed in 206.14 seconds, including 43
supplemental correctness cases and the requested 1/2/16-worker matrix. Exact
no-force reuse took 19.50 seconds and preserved bytes and timestamps for all
1,068 owned files. Deliberately corrupting one capability unit invalidated the
proof; repair took 22.09 seconds, restored the original bytes, and preserved all
other unit timestamps. Sampled process-tree RSS peaked at 3,798,147,072 bytes.
These validation runs overlapped other tests; their timings are observations,
not isolated throughput benchmarks. All 87 focused regression tests passed.
The full simulation-to-report oracle and its no-force replay passed in
1,059.62 seconds, including 457.51 seconds across preflight execution/reuse calls.
Both final release audits passed, and the replay preserved the authenticated
workflow bytes. The isolated validation snapshot was
`fb52bfd108c9a5e49087e497c913dd2949937f73`; all 256 source/test/script/config
files matched the workspace. Ruff, Mypy (94 source files), targeted Pyright,
Black and the normal Git whitespace check passed.

Focused tests exercise intentional omission, exact reuse, lost required outputs,
retained-proof ownership, warnings without new games, and enabled cap status.
Legacy ordering/collision/cap/resume tests remain applicable. The complete raw
simulation-to-report oracle runs both real preflights and checks their evidence
in the final graph. The full oracle's ceiling is the previous 240-second baseline
plus two 1,800-second preflight budgets. Proof authentication also occurs in
health, graph construction, reporting and final audit, so subtracting only the
preflight function's duration does not isolate the former workflow runtime.
Each preflight invocation also retains its own 1,800-second ceiling. This is an
explicit validation-runtime allowance, not a simulation-budget increase or a
production-throughput qualification.

Existing uncommitted Tasks 0/9/1A/1B/2/3/4/5A/5B work is preserved. Task 6
(interactive H2H cap authorization) is the next outstanding task.
