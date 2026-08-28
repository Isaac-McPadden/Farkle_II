# Task 7A artifact-density benchmark plan

## Status and decision boundary

This plan is published before any Task 7A measurement. Task 7A is diagnostic
only: production modules, configurations, statistical methods, canonical logical
schemas, authentication policy, checkpoint behavior, and accepted result trees
will not be changed. Results may recommend at most one separately approved Task
7B remediation.

Repository provenance at plan publication is branch `main`, HEAD
`9f879de1cdb4e6b92808799899c1e6af06980ca1`, tracking `origin/main` at the same
commit. The tracked worktree is clean. Eight untracked
`docs/remediation/task6_rerun_20260827_*` documents predate Task 7A and are
preserved as user evidence.

## Evidence roots and ownership

Read-only immutable evidence:

- `data/results_rng_opt_fast_seed_pair_56_57`: accepted post-Task-6 integration
  evidence, including both roots and the pair workflow;
- `data/results_efficiency_updates_4D_fast_seed_pair_54_55`: accepted historical
  pre-Task-6 integration evidence used only for provenance-sensitive comparison;
- retained `data/farkle-task5a-production-capacity-v1` and current Task 6
  benchmark roots, including the pre-existing 2026-08-27 rerun roots.

Task 7A will never resume, reseal, re-sidecar, migrate, rename, delete, or write
inside those roots.

The only disposable/retained Task 7A root is
`data/farkle-task7a-artifact-density-v1`. Its exact root must contain
`task7a_ownership.json` with task `7A`, schema version `1`, and the canonical
resolved root before any write or resume. Cleanup may target only named child
case directories recorded in that marker, after resolving each child and proving
it is beneath that exact owned root. No glob-selected or recursive broad-path
cleanup is permitted. The final measurement checkpoint and bounded fixtures are
retained for review unless this report explicitly names a safely cleaned case.

## Cases and identifying purpose

1. **Immutable integration inventory.** Traverse seeds 56/57 by root, pair,
   stage, role, and type. Parse every Parquet footer, JSON document, and alleged
   JSONL file structurally. This identifies current file/open counts, row-group
   density, footer bytes, typed-empty behavior, metadata repetition, physical
   duplication, H2H density, and the release-audit byte/open surface. Repeat the
   combine-only footer inventory for seeds 54/55 to establish whether the
   approximately 25-row historical groups survived Task 6.
2. **Production-width Parquet fixtures.** Use the code-defined raw simulation
   Arrow schema for representative low/high `k` values (2 and 12), deterministic
   PCG64DXSM-generated values, production writer settings, and bounded rows.
   Compare equal logical rows split across many shuffle-shaped files and fewer
   deterministic range-shaped benchmark-only files. This identifies per-file,
   footer, open, scan, hash, and padded-column coefficients without implementing
   a production layout.
3. **Shard-count sweep.** Hold logical bytes/rows fixed where practical and use
   balanced file counts sufficient to expose metadata/provider latency. Measure
   write, open/footer, projected downstream scan, and SHA-256 costs. Logical
   Arrow content and coordinate order are the equivalence oracle.
4. **Authentication byte/count sweep.** Cross controlled artifact counts with
   controlled total bytes, using production hashing and artifact-validation
   helpers where a valid sidecar is required. This separates per-open/call cost
   from byte throughput and records calls, bytes, CPU, and wall time.
5. **H2H finalized-block sizing.** Inventory current block data and sidecars,
   then deterministically model bounded grouping from actual block sizes and
   schemas. If fixture writes are necessary, they remain benchmark-only and use
   the production block schema/writer. No checkpoint or publication code changes.
6. **Current combine traversal.** Read current post-Task-6 combine partitions
   with the production deterministic scanner/projection path at bounded support.
   Record row groups, reads, decoded estimates, wall/CPU, and logical digest.
7. **RNG decomposition.** Prefer accepted telemetry and Task 6 checkpoints.
   Run only the smallest existing Task 6 bounded case needed to fill a missing
   coefficient. Reuse accepted stage wall time rather than rerunning a large
   workload.

Together, these cases distinguish count-bound metadata/open overhead,
byte-bound hashing/decoding, row-width/padding, and downstream traversal. They do
not implement any candidate production representation.

## Measurement protocol

- Use `\.venv\Scripts\python` (Python 3.12) and Windows-compatible spawn guards.
- Use `time.perf_counter_ns` for wall boundaries and `time.process_time_ns` plus
  child-process CPU sampling for CPU. Boundaries exclude fixture planning but
  include the operation under test; end-to-end case timers also include durable
  publication and cleanup separately.
- Sample process-tree RSS and native threads during active cases. Record requested,
  resolved, created, live, and peak workers when a production pool is exercised.
- Each timing case has one unreported warm-up and at least three measured
  repetitions. Two-case comparisons use ABBA order; larger sweeps use a fixed
  seed-derived balanced Latin/reversed order. Report every repetition and median.
- Filesystem/provider is the current synchronized OneDrive repository volume;
  local OS temporary space is reported separately. Record volume, hydration
  observations available from file attributes, host-available memory, CPU count,
  Python/Arrow versions, and wall-clock period. Provider activity, cache state,
  antivirus, CPU frequency, thermals, and other host load are confounders; reversed
  order and conservative intervals limit but cannot eliminate them.
- Track created/opened files, durable/temporary bytes read/written, Parquet rows,
  groups and serialized footer bytes, JSON/sidecar/manifest/checkpoint/completion
  bytes, hash calls/bytes/time, authentication calls/time, spills/merges when
  exposed, cleanup time/failures, parent/worker CPU, peak process-tree RSS, and
  peak native threads. Unavailable counters are explicitly null, never zero.
- Resource bounds: at most 512 MiB fixture payload, 4 GiB peak Task 7A temporary
  disk, 4 GiB process-tree RSS, four worker processes, and 20 minutes per case;
  no case may exceed 30 minutes. Stop the case on a resource bound, three failed
  repetitions, ownership mismatch, evidence mutation, schema/digest mismatch,
  or a production-code write attempt.

## Equivalence oracles

- Parquet layout comparisons must have equal production Arrow schema, row count,
  canonical coordinate order, nullability, and a streaming logical-content
  digest over typed batches. Physical bytes may differ.
- Hash/authentication sweeps compare expected SHA-256 identities and production
  validator outcomes.
- Current artifact inventories are reconciled against manifests, sidecars,
  completions, run contexts, and final release-audit/health evidence without
  publishing anything.
- JSONL validity requires every nonblank physical line to parse as exactly one
  JSON value; a pretty-printed single object is not classified as JSON Lines.
- Typed-empty files are compared to code-defined intended Arrow schemas, including
  type and nullability, rather than merely column names.

Any oracle failure stops candidate ranking for the affected family and yields
`more_evidence_required` unless the failure independently proves a contract defect.

## Precommitted ranking and materiality rules

Candidates are classified as physical/checkpoint layout, authentication/public
artifact-contract migration, canonical logical-schema change, or statistical
method change. The last two are ineligible for Task 7B. A broad authentication or
public-contract migration can only yield `contract_migration_requires_approval`.

For each eligible physical candidate, scale the measured per-file, per-byte,
per-row, or per-block coefficients with current executable production dimensions.
Report central, plausible, and conservative savings; storage, RAM, temporary-space,
Windows-spawn, recovery-quantum, identity/version, compatibility, and test costs;
and subtract any added read/write/authentication work.

A Task 7B recommendation must be one independently verifiable remediation and
must satisfy all of:

- at least 6 hours central two-root critical-path opportunity **and** at least 5%
  of the refreshed central total;
- at least 3 hours net benefit under conservative throughput/overhead assumptions;
- bounded peak RAM and temporary space within current execution policy;
- unchanged scientific results and a precise logical-equivalence oracle;
- no required concurrent authentication-policy, public-contract, canonical-schema,
  statistical-method, local-scratch, or process-isolation migration.

Rank primarily by conservative net critical-path hours, then file/open reduction,
then implementation/recovery risk. Storage savings alone do not clear the gate.
If no candidate clears it, recommend no implementation. Task 8 is triggered only
by measured cross-stage retention, repeatable inherited-state resource failure, or
a bounded fresh-process A/B with material benefit.

## Retained outputs and stop criteria

Retain the ownership marker, case checkpoints, raw repetition records, environment
record, inventory summaries, and final benchmark JSON beneath the owned root, plus
the checked-in Task 7A script/test/report/JSON deliverables. Atomic publication is
required for every checkpoint.

Stop Task 7A after publishing one closed verdict, the current-state capacity model,
verification results, retained/cleaned-root inventory, and final worktree status.
Do not begin Task 7B, Task 8, a production run, or a fresh two-seed integration run.
