# Task 7B deterministic H2H bundling benchmark plan

## Status and decision boundary

This plan is published before Task 7B production edits and before Task 7B
measurements.  The only eligible production change is a narrowly versioned H2H
physical checkpoint layout that replaces one Parquet/sidecar per logical block
with deterministic bundles of at most 16 contiguous canonical schedule rows.
H2H method v2, all statistical and RNG identities, the logical block schema and
order, power planning, candidate membership, game coordinates, attempt caps,
the 5,000-attempt checkpoint bound, aggregation, inference, dominance,
agreement, reporting semantics, and the final fresh byte-deep release audit are
frozen.

The starting repository is branch `main` at
`2c7c97183c544e5ada8a050747727fb39ce7c283`, aligned with `origin/main`.
The only starting worktree entry is the user-owned untracked
`docs/remediation/PLAN Revised.md`.  Accepted Task 4--7A evidence is committed.
Historical and accepted result trees are immutable and will not be resumed,
migrated, resealed, re-sidecarred, renamed, deleted, or modified.

The closed verdict must be exactly one of:

- `accepted_bounded_remediation`;
- `performance_gate_failed`;
- `equivalence_gate_failed`;
- `recovery_or_tail_gate_failed`;
- `authentication_migration_requires_approval`;
- `indeterminate_more_evidence_required`.

Only the first verdict permits the proposed production layout to remain as the
Task 7B result.  Task 8, another optimization, a production run, and a fresh
full two-seed integration run are outside this task.

## Preserved baseline and owned roots

The pre-change baseline is the current `execute_h2h_schedule` implementation at
the starting code identity.  It uses the real game engine, semantic PCG64DXSM
coordinates, Windows `spawn`, shared bounded process scheduling, one-block
Parquet/sidecar publication, execution-state checkpoints, final block
validation, aggregate publication, stage completion, authenticated graph
construction where the fixture reaches it, and the fresh release-audit path.
Task 7A's direct Parquet comparison is supporting evidence only and is not the
Task 7B performance baseline.

Task 7B owns only `data/farkle-task7b-h2h-bundling-v1`.  The exact resolved root
must contain `task7b_ownership.json` with task `7B`, benchmark schema version,
settings, starting code identity, and resolved root before any benchmark write.
The root contains disjoint `baseline`, `candidate`, `recovery`, and `adversarial`
children.  Baseline measurements and artifacts are made read-only by policy
after their pre-edit capture and are never regenerated or overwritten after a
production edit.  Candidate resume skips any complete atomic measurement unless
`--force` is supplied.  Force may replace only a named case/repetition child
after resolving it beneath the exact owned root and checking the marker; it may
not recursively clean the owned root, historical roots, or unrelated data.

Every durable measurement and checkpoint is written with `atomic_path`.
Recognized staging files are never completion inputs.  Retained evidence will
include the ownership marker, complete baseline and candidate repetition JSON,
logical-oracle files, recovery/interruption records, audit records, environment
record, final raw evidence, and summary.  Superseded disposable payload may be
cleaned only by exact marker-authorized child path and will be named in the
report.

## Physical candidate and bounded scheduler design

Bundle membership is derived once from the authenticated schedule sorted by the
existing canonical key `(pair_id, root_index, order)`.  Consecutive half-open
schedule-index ranges of width 16 form the inventory; the final range may be
shorter.  A bundle key contains layout version, bundle ordinal, start index, and
exclusive stop index, and its filename encodes the same values.  Crossing the
root/order cycle is permitted only because current blocks are independent
semantic RNG coordinates and aggregation already restores the same canonical
order; focused tests must prove that the crossing changes no value or ordering.
Using root/order-local groups would not be contiguous in the frozen global
schedule and is therefore not selected.

One bundle publication is one Parquet, one adjacent authenticated sidecar, and
one adjacent typed partition-unit stamp.  A deterministic authenticated final
inventory is published only after every expected bundle validates.  The
inventory must reject duplicate, overlapping, omitted, noncontiguous,
out-of-order, relocated, unexpected, old-layout, wrong-identity, wrong-schema,
wrong-row-count, or duplicate-coordinate support.  Consumers use one scanner
that validates the planned inventory and yields the original logical rows in
canonical order; no consumer synthesizes per-block or bundle filenames.

Workers continue to receive individual block-extension tasks, so simulation
coordinates and load granularity do not change.  Parent state holds at most one
row per completed but unpublished block.  Submission backpressure stops adding
new futures at an explicit deterministic completed-row/byte bound while allowing
already submitted work to drain.  The initial process-map window is unchanged;
the precommitted completed-row bound is
`max(64, 4 * resolved_workers * max_in_flight_per_worker * 16)` rows, and the
byte bound is 16 MiB.  Publication of a ready bundle removes its rows from the
completed buffer.  The bounds are deliberately much larger than the ordinary
15-worker window so one slow block does not normally drain worker occupancy,
but remain independent of total schedule size.  If either bound is reached,
submission pauses until publication frees capacity.  Cancellation uses the
existing bounded cancel/terminate/join/kill path; no orphan worker is accepted.

For future blocks requiring more than one 5,000-attempt extension, every bundle
is published after all of its active members for that generation return.
Terminal members are reused in later bundle rewrites, while nonterminal members
resume their exact authenticated prefixes.  A complete valid bundle is the
recovery unit.  Invalid bundle data, sidecar, stamp, or support causes only that
bundle's members to be replayed.  Unaffected bundle bytes and modification times
must remain exact.  Maximum exposure is 16 blocks, 5,000 attempts per block, or
80,000 unauthenticated attempt coordinates.

## Benchmark cases

All timing cases use the repository `.venv`, the synchronized repository volume,
Snappy, the same deterministic strategy family and schedule within a comparison,
the real game engine, Windows `spawn`, the same resource policy, and identical
game coordinates across worker counts and layouts.

1. **Warm-up.** One unreported 12-block real-engine execution under the active
   layout primes imports, Arrow, worker spawn, and filesystem paths.  It is not
   included in medians.
2. **Accepted 180-block shape.** Ten frozen candidates, two roots, and two seat
   orientations produce 45 unordered pairs and exactly 180 blocks through the
   executable planner.  The bounded completed target is the accepted integration
   target of 1,372 per block.  This is the primary publication benchmark and
   exercises real game work without manufacturing outcomes.
3. **Production-shaped bounded inventory.** Twenty candidates produce 190 pairs
   and 760 root/orientation blocks through executable scheduling.  The bounded
   target is likewise 1,372 per block.  This case is large enough to expose bundle planning,
   out-of-order completion, repeated publication readiness, backpressure,
   aggregate construction, authentication, and tail behavior while staying
   bounded.
4. **Exceptional functional matrix.** Small deterministic real-path fixtures
   cover ordinary viable blocks, low-safety replacements, one deliberately
   nonviable tail, all-nonviable support, an exact 16-row boundary, a final
   partial bundle, multiple roots/orders, one worker, multiple workers, and
   worker-count invariance.
5. **Interruption and exact resume.** Inject at the requested boundaries before
   first publication, with partial in-memory occupancy, around atomic data and
   sidecar/stamp publication, after some bundles, before/after final inventory,
   before aggregate, after aggregate, before completion, and during worker
   cancellation.  Resume records reused/replayed bundle and block coordinates,
   bytes, mtimes, actual replay attempts, and wall time.
6. **Adversarial inventory.** Independently mutate or remove bundle data,
   sidecar, unit stamp, inventory, schema/nullability, row support, path,
   identity, ordering, range coverage, and post-snapshot bytes.  Add duplicate,
   overlapping, omitted, relocated, unexpected, stale, mixed-old/new, and
   duplicate-coordinate cases.  No case may publish false aggregate, stage,
   graph, audit, or pipeline success.
7. **Logical/downstream oracle.** Compare baseline and candidate ordered block
   streams, aggregate, execution state, inference, dominance, agreement,
   reporting, authenticated completion, and final release-audit results.
8. **Candidate-count sensitivity.** Derive exact schedule counts by invoking the
   executable scheduler for 75, 77, 100, 125, 143, and 150 candidates under two
   roots/two orientations.  Reconcile but do not substitute the expected
   `4 * C * (C - 1) / 2` relationship.

Each primary performance case has at least three measured baseline repetitions
captured before production edits and three candidate repetitions afterward.
Case order alternates 180/760 then 760/180 by repetition; candidate execution
uses the reverse starting order.  Repetition-specific strategy and schedule
seeds are fixed, not randomized at runtime.  This balances case/order effects,
although the necessarily pre-edit/post-edit layout phases leave host-time as an
explicit confounder.  Every repetition is reported; medians determine gates.

## Timer boundaries and telemetry

The baseline block-publication timer begins immediately before construction of
each terminal/partial block Parquet and ends after its sidecar has atomically
published and validated.  Its repetition value includes parent checkpoint
handling, Parquet construction, sidecar construction, hashing, atomic
replacement, validation, and the aggregation-facing block inventory
finalization.  The candidate timer has the identical semantic boundary around
bundle readiness, table construction, sidecar/stamp publication, validation,
and final bundle-inventory publication.  Simulation, execution-state,
aggregate, and stage-completion timers are separate.

The total H2H timer begins immediately before source authentication in
`execute_h2h_schedule` and ends after stage completion publication.  Critical
worker simulation is the maximum summed simulation time for any worker.  Pool
construction begins before executor construction; shutdown ends after clean
worker termination.  Final-result-to-final-bundle and final-bundle tail timers
use parent monotonic timestamps.

Per repetition record wall and parent/child CPU where available; peak
process-tree RSS and native threads; requested/resolved/created/live/peak
workers; pending/in-flight/idle behavior; pool generations/initializers/shutdown;
attempt/completion/replacement/safety totals; bundle/block planned, reused,
rejected, replayed, buffered, published, and durable counts; maximum buffered
rows/bytes; oldest unpublished age; waiting bundles; last-bundle tail;
execution-state, aggregate, completion, authentication, validation, cleanup and
release-audit times; files created/opened; Parquet/data/footer/sidecar/stamp/
manifest/completion bytes; hash calls/bytes/time; retries, pauses, downshifts,
failures, and cleanup failures.  Unavailable measurements are `null`, never
invented as zero.  Durable parent state is the sole completion numerator.

## Exact equivalence oracles

The baseline capture retains the ordered schedule, family/pair membership,
exact Arrow schema including field order/types/nullability, every logical row,
coordinate support, attempts, completions, replacements, safety counts, wins,
statuses, aggregate integer counts, execution state, inference outputs and
decisions, dominance, agreement, report outputs, completions, sidecars,
manifests, and canonical digests.

The primary logical digest streams typed Arrow record batches in canonical
coordinate order and includes the schema identity before row bytes.  Candidate
comparison requires exact equality of coordinate set/order, schema, attempted
prefixes, root/order balance, all integer values/statuses, candidate/pair
support, downstream floating-point bit patterns, decisions/classifications,
and downstream canonical bytes.  No tolerance is allowed.  Permitted byte
differences are only bundle Parquets/paths/row groups, their sidecars/stamps and
inventory, layout telemetry, and completion provenance that binds those new
physical bytes.  The report must enumerate and explain every differing file.

## Recovery, performance, and stop gates

Acceptance requires all of:

- exact logical and downstream equivalence;
- at least 70% reduction in median real-path block-publication wall time for the
  180-block primary case;
- no more than 5% median regression in critical worker-simulation time;
- no more than 5% median regression in total bounded H2H wall time, unless the
  complete predeclared repetitions clearly establish measurement noise;
- no material worker-utilization loss or unresolved publication/head-of-line
  stall;
- observed completed-buffer occupancy within both fixed bounds;
- exact reuse of unaffected bundle bytes/mtimes;
- at most 16 blocks and 80,000 attempts exposed by one lost bundle; and
- conservative full-bundle replay at representative 12--15 workers no greater
  than 60 seconds.  One-worker and reduced-worker replay are reported
  separately and are not treated as production topology.

Any logical/schema/numeric difference, false completion, authentication-policy
expansion, required broad contract migration, buffer-bound violation, orphan
worker, replay outside the affected bundle, representative replay above 60 s,
unexplained >5% simulation/total regression, or <70% publication reduction is a
stop condition.  It cannot be waived after results are known.

## Resource bounds and sensitivity model

No case may exceed 4 GiB newly written durable/temporary data, 8 GiB
process-tree RSS, 16 workers, 190 native threads, or 30 minutes for one
repetition.  A three-failure case, ownership mismatch, immutable-evidence
mutation, partial-file qualification, or digest/schema mismatch stops that
case.  Processing remains byte-bounded; the benchmark does not load a production
result tree or full production dataset into memory.

For candidate counts 75, 77, 100, 125, 143, and 150, the report uses executable
schedule/block/bundle/file counts and measured per-block/per-bundle publication
and final-authentication coefficients.  It reports block-publication saving,
final-authentication saving, central/conservative total saving, revised pipeline
runtime, and whether Task 7A's original six-hour/5%-central and three-hour
conservative gates pass.  It separately states the 150-candidate maximum
envelope, the unknown realized family, method-overlap sensitivity, and the
calculated break-even candidate count.  It will not imply that maximum-envelope
savings are guaranteed.
