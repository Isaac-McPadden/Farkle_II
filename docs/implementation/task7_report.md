# Task 7 - behavior, opportunity, robustness, and parameter summaries

Contract: September 24 `first_release_task0_v1`, implementation backlog Task 7.
Status: complete; bounded acceptance passed on October 6, 2026.

## Products and statistical meaning

The metrics child `seat_behavior` publishes B7/Q7/O7/O and the root-level OA
artifact at the exact AppConfig paths frozen by Task 0. B7 retains actual seats,
deterministic batches, and separate completed/safety-limit sufficient statistics.
Certified zero strata are explicit; unavailable means have null estimates and
`no_denominator`. Required actual-seat counters cannot be missing. Completed
turn/round denominators must be positive; a safety exposure with a zero denominator
must have zero score. Its mean-of-game-ratios term is zero, and zero support is
shown explicitly. Checked int64 sums fail on overflow.

Q7 publishes actual seats plus seat=0, with completed, safety-limit and
all-attempted views. Points/turn is available as ratio-of-sums and equally
exposure-weighted game ratios. The round proxy and exact-minus-proxy gap remain
separate. Count rates use actual turns or exposures; highest-turn uses exposures;
heuristic dice/use ratios are null without uses. All-attempt support and compatible
numerators reconcile with the existing all-player metrics before completion.
No behavior confidence intervals or state-specific causal claims are added.

O7 accumulates each game once, even when a strategy appears in multiple seats.
O reports completed-game turn ranges, seat turn means/shares and total-variation
imbalance relative to 1/k. OA is the equal-k mean of mean turn range separately
within each root, with every required k and explicit weights. A missing k fails;
a valid k without completed games makes the overall value unavailable.

The HGB child `parameter_responses` publishes R1/R2/RI using authenticated
all-attempt `performance.win_rate`, canonical manifest attributes, held-out
predictions and fold support. The ten HGB features retain at most 25 ordered
levels by the frozen index rule. Threshold sentinels never become active levels.
One-way estimates use backgrounds supporting every selected level. The three
prespecified pairs use adjacent selected-level rectangles and identical
backgrounds for all four corners. Backgrounds have equal weights. Missing
required input rows fail; unsupported cells have null estimates and an explicit
status. Equivalent inactive settings cannot count twice.

Observed nonadditivity is mean(p11-p10-p01+p00). OOF corner means are separate;
there is no cross-fold predicted interaction column. RI retains exact matched
strategy IDs, fold IDs and matching attributes. Support IDs bind source hashes,
selection policy and matched identities. Unknown declared strategy types receive
explicit excluded-type records. There are at most 250 one-way and 1,728 two-way
rows per root/k. Support inventories stream rather than growing in memory.

## Durability and integration

The behavior producer reuses the authenticated outcome workload/roster and
complete-support certificate. It projects bounded curated slices, writes sorted
fixed-width records and merges with fan-in at most 16. Floating game-ratio terms
retain their canonical coordinate order through merging, so I/O boundaries do
not change the published sums. Each batch then reconciles condition-specific
exposures to B before emitting zero strata. Summary workers use bounded strategy
tiles over retained B7; opportunity reductions retain only per-seat totals.

`_seat_behavior_units` owns the immutable plan, source/merge/batch/summary units,
unit manifests and final publication inventory. Existing partitioned-stage,
byte-aware reader, atomic Parquet/sidecar and child-completion helpers own
publication and resume. Independent units use the configured process policy.
The release graph retains these checkpoints and the final byte-deep audit checks
the complete declared unit inventory. Missing canonical products cannot be
replaced by an empty success marker. Scratch staging uses `io.temp_dir`.

Metrics cache version is 6, HGB stage cache version is 2, and structure report
contract version is 8. New child method versions are 1. HGB fitting, fold/RNG
construction, simulation targets and all production configuration values remain
unchanged. Reporting authenticates the new products, renders behavior and
opportunity values/support, links exact parameter/support tables, and surfaces
held-out fold metrics and permutation association importance. Existing per-k,
equal-k, Pareto and maximin summaries remain descriptive.

## Validation

The focused regression suite passed all 136 cases in 382.390 seconds, with no
failures, errors or skips. Coverage includes behavior arithmetic and denominator
rules, matched responses, authenticated publication, missing/corrupt checkpoint
recovery, changed slice widths, real Windows-spawn worker equivalence, HGB and
metrics integration, report rendering, release audit, existing outcome products,
H2H authorization, and the coordinate-owned RNG audit.

The worker fixture used 48 twelve-player games, eight source units and a 2 MiB
source budget. One- and two-worker publications had identical bytes, and deleting
one source unit rebuilt that unit while preserving the others. The measured peak
process-tree RSS was 700,706,816 bytes, scratch use was 442,431 bytes, and fixture
runtime was 4.362 seconds. These small-fixture measurements do not qualify
production capacity.

Ruff (`src`, `tests`, `scripts`), Mypy (98 source files), Pyright (six affected
runtime modules), Black (22 changed/new Python files), and `git diff --check`
passed. Validation runs use isolated snapshot
`2fd71c911bf6873267c2ccb9b2d813322fb2e7ae`; all 98 production Python files match
the working tree. Existing uncommitted changes and production configuration are
preserved. Counts, identities, JUnit hashes and measurement properties are retained
in [task7_validation.json](task7_validation.json).

The full raw simulation-to-report oracle passed in 898.881 seconds (900.648
seconds including pytest setup/teardown). Both required RNG preflights ran,
accounting for 368.303 seconds. The oracle independently reconciled behavior and
opportunity products to raw games, checked bounded parameter outputs and the
authenticated artifact inventory, exercised one interactive H2H cap approval,
and completed the final root/root/pair audit. An ordinary no-force replay
preserved authenticated output hashes and H2H block count without prompting
again. The configured H2H cap remained unchanged.

Acceptance exposed missing Parquet column and JSON key inventories in the new
checkpoint sidecars. Both are fixed, and the regression now checks both formats.
The accepted results above are from the final snapshot after those corrections.

Structural reporting fixtures use explicitly synthetic Task 7 data. They test
report ownership and rendering, and do not replace producer or raw-workflow
validation. No production simulation, new model fit for A89, or production cap
override is run by this task. Task 8 is the next implementation task after
acceptance; full production capacity qualification remains Task 10.

The separate repository terminology gate still reports four pre-existing uses
of `pooled` in unchanged `outcome_publication.py` (995, 1031, 1080, 1186),
including a durable unit directory name. This gate is not reported as passing.
