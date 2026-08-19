# Task 3B RNG route-unit coarsening

## Verdict

Task 3B is implemented with a production default of 32 contiguous source row
groups per durable route unit. The change materially reduces the file/open/spill
architecture while preserving the RNG diagnostic method and exact canonical
results.

At 1,024 source row groups and eight diagnostic partitions, the one-worker
median changed from 216.170 s at size 1 to 93.186 s at size 32 (0.431x, or a
2.32x speedup). Size 32 reduced:

- route files and implied unit stamps from 2,048 to 64 (32x);
- reducer route opens from 16,384 to 512 (32x);
- selection-membership loads from 1,024 to 32 (32x);
- initial spills from 5,632 to 384 (14.7x);
- merge outputs from 188 to 12 (15.7x);
- route bytes from 12,218,368 to 8,897,152 (27.2%);
- merge bytes from 335,907 to 166,131 (50.5%).

Median sampled RSS changed from 224.3 MiB to 225.9 MiB (0.7%). A separate
two-worker size-32 run completed in 40.554 s with 642.6 MiB peak process-tree
RSS, within the configured 2,304 MiB aggregate contract. There were no retries,
downshifts, memory pauses, failures, or cleanup failures.

Every size and worker-count comparison had one exact final digest per scale.
The preserved pre-edit full-stage baseline also proved byte-identical diagnostic
and selection Parquets, identical schemas/order/logical values, and bitwise
identical floating-point fields.

Machine-readable evidence is in
[`task3b_rng_coarsening.json`](task3b_rng_coarsening.json); the reproducible,
resumable driver is
[`benchmark_task3b_rng_coarsening.py`](../../scripts/benchmark_task3b_rng_coarsening.py).

## Design and coverage contract

The canonical ordered inventory remains `(source path, row-group index)` in
combined-manifest order. The planner validates contiguous zero-based ordinals
and emits half-open `(start, stop)` keys with filenames such as
`row-groups-00032-00063.arrow`. It handles a shorter-than-range inventory, exact
multiples, an uneven final range, and ranges crossing per-k source files.

Each worker processes the range sequentially: one source row group at a time,
then one projected Arrow batch at a time through the existing byte-bounded
reader. It never concatenates a route range, source file, partition, or dataset.
Reducers coalesce only records from one route unit and only up to the existing
derived spill-byte budget; oversized individual projected batches retain their
existing bounded treatment.

Route-layout v2 stores four Arrow schema metadata fields: layout version, route
kind, partition count, and `source_batch_then_partition` layout. Writers emit
exactly one record batch per diagnostic partition for each projected source
batch, including empty batches. Reducers validate the metadata, schema, complete
partition blocks, and batch-count divisibility before selecting
`partition + n * partition_count`. Focused tests recover every partition from a
range containing multiple row groups and multiple projected batches.

After the shared partition runner validates and publishes a route manifest, the
parent reads that manifest once and passes its ordered route inventory to the
reducers. Reducers no longer synthesize old `row-group-{ordinal}` names, and no
new repeated deep-authentication pass was introduced.

Count reduction remains exact unsigned-integer addition. Observation runs retain
the complete sort key: group type, k, compact ID, every padded participant ID,
root seed, shuffle index, game index, and seat index. Duplicate coordinates are
rejected both within a newly coalesced run and across merge inputs. The compact
digest remains only an accelerator; injected digest collisions retain separate
full participant keys through counting, selection, membership, routing, and
final grouping.

## Checkpoint identity and compatibility

This is an operational checkpoint-layout change only:

- RNG partition checkpoint schema advances from 1 to 2;
- route layout is explicitly version 2;
- route size, source-row-group count, and the complete planned unit key/filename
  inventory are hashed into a `route_layout` input identity;
- both route phases bind that identity, and downstream eligibility/statistics
  state binds the resulting authenticated route manifests.

Method v4, stage cache-key v6, RNG scheme v2, canonical diagnostic/selection
schemas, statistical config SHA, grouping, eligibility, cap, lag, and coordinate
semantics are unchanged. Method-v4 checkpoints from the old layout cannot be
silently reused. Historical seed-48/49 artifacts remain valid read-only evidence
but are not resumable under the new repository identity; none were opened,
modified, resealed, migrated, or backfilled.

Interruption tests prove that a writer failure after producing a temporary ranged
file publishes neither the unit nor a false-complete stamp. Resume preserves
valid completed range mtimes and executes the pending range exactly once.
Corruption tests quarantine one invalid ranged route or downstream partition and
reuse every unaffected valid unit.

## Exact-equivalence gate

Before editing, a disposable full-stage baseline at starting HEAD
`0364f63054bc919c18fe25fdc08e013e489c898b` used 96 rows, 14 source row groups,
eight partitions, lags 1/2/5, cap 12, and one worker. The complete old checkpoint
tree, route manifests, unit stamps, sidecars, completions, output tables, schemas,
and summary were preserved under the host temporary directory.

The post-change default run used identical inputs and settings. Results:

| Comparison | Result |
| --- | --- |
| Candidate / eligible / selected / skipped / capped counts | Exact |
| Observation and lagged-pair integer counts | Exact |
| Stable partitions, compact IDs, and full participant keys | Exact |
| Membership, eligibility, estimability, and completeness | Exact |
| Coordinate coverage and duplicate rejection | Exact |
| Schemas, nullability, and deterministic row order | Exact |
| All canonical float bit patterns | Exact |
| Diagnostic Parquet bytes | Identical |
| Selection Parquet bytes | Identical |

The only diagnostic-summary differences were
`partition_manifest_sha256` (expected new checkpoint graph) and
`peak_sampled_process_tree_rss_mb` (221.1641 versus 221.1562 MiB). Sidecars,
completion bytes, unit stamps, route manifests, route filenames, route bytes,
and their hashes are expected to differ because they bind code/checkpoint/layout
provenance. No statistical or capacity field differed.

## Benchmark method

The benchmark invokes the production `_CountRouteWriter`, `_EligibilityWriter`,
`_StatsRouteWriter`, and `_StatsPartitionWriter` directly. Fixtures have 256 and
1,024 physical Parquet row groups, one deterministic game row per group, four
repeating full matchup keys, eight diagnostic partitions, and the ten exact
projected columns read by the stage. Fixture and selection creation occur before
the workload timer.

There is one warm-up. One-worker sizes 1, 16, and 32 use two repetitions in
forward then reverse order. Size 32 also runs once with two Windows-spawn
workers. Each scenario has a distinct owned directory and atomically publishes
`measurement.json`; an ownership marker binds all resume settings. A resumed
run reuses valid measurements and removes only an incomplete direct-child
scenario.

Route files/bytes, reducer opens, memberships, actual spills/merges, projected
compressed source-column bytes, route hashing time/bytes, phase/total wall time,
CPU time, process-tree RSS, host-available memory, native threads, workers, and
failure counters are recorded. Unit-stamp counts are exact structural counts
implied by `run_partitioned_stage`, not separately written benchmark stamps.
Hash timing is an explicit SHA-256 pass over actual route files; it does not
pretend to be kernel ETW or a full partition-runner authentication trace.

## Repetition-level results

| Row groups | Range | Workers | Rep / position | Wall (s) | CPU (s) | Peak RSS (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 1 | 1 / 1 | 44.466 | 41.344 | 209.9 |
| 256 | 16 | 1 | 1 / 2 | 14.877 | 14.484 | 211.2 |
| 256 | 32 | 1 | 1 / 3 | 12.817 | 12.656 | 211.4 |
| 256 | 32 | 1 | 2 / 1 | 12.217 | 12.141 | 211.9 |
| 256 | 16 | 1 | 2 / 2 | 14.019 | 13.766 | 211.9 |
| 256 | 1 | 1 | 2 / 3 | 38.282 | 35.078 | 213.0 |
| 256 | 32 | 2 | 1 / 1 | 15.400 | 16.422 | 620.8 |
| 1,024 | 1 | 1 | 1 / 1 | 212.806 | 205.141 | 222.9 |
| 1,024 | 16 | 1 | 1 / 2 | 99.689 | 104.125 | 223.4 |
| 1,024 | 32 | 1 | 1 / 3 | 93.711 | 98.844 | 225.8 |
| 1,024 | 32 | 1 | 2 / 1 | 92.662 | 97.984 | 226.1 |
| 1,024 | 16 | 1 | 2 / 2 | 98.734 | 103.250 | 225.7 |
| 1,024 | 1 | 1 | 2 / 3 | 219.534 | 212.250 | 225.7 |
| 1,024 | 32 | 2 | 1 / 1 | 40.554 | 82.078 | 642.6 |

The reversed order retained the same direction and similar effect size. Two
workers are counterproductive at 256 groups because spawn/pool overhead dominates,
but beneficial at 1,024 groups. Worker-count changes do not change the digest.

## Median phases and structural counts

| Row groups | Range | Count route | Count reduce | Stats route | Stats reduce | Total wall | Total CPU | RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 7.116 | 11.628 | 8.120 | 13.176 | 41.374 | 38.211 | 211.5 MiB |
| 256 | 16 | 5.301 | 1.847 | 5.369 | 1.810 | 14.448 | 14.125 | 211.5 MiB |
| 256 | 32 | 4.718 | 1.071 | 5.268 | 1.362 | 12.517 | 12.398 | 211.7 MiB |
| 1,024 | 1 | 55.408 | 47.048 | 61.360 | 48.118 | 216.170 | 208.695 | 224.3 MiB |
| 1,024 | 16 | 41.823 | 6.181 | 43.920 | 6.969 | 99.212 | 103.688 | 224.5 MiB |
| 1,024 | 32 | 41.911 | 3.471 | 43.365 | 4.239 | 93.186 | 98.414 | 225.9 MiB |

| 1,024-row-group metric | Size 1 | Size 16 | Size 32 | Size-32 / size-1 |
| --- | ---: | ---: | ---: | ---: |
| Route files / stamps | 2,048 | 128 | 64 | 0.031x |
| Route bytes | 12,218,368 | 9,004,288 | 8,897,152 | 0.728x |
| Reducer opens | 16,384 | 1,024 | 512 | 0.031x |
| Membership loads | 1,024 | 64 | 32 | 0.031x |
| Initial spills | 5,632 | 768 | 384 | 0.068x |
| Initial spill bytes | 248,832 | 181,440 | 173,664 | 0.698x |
| Aggregate merge passes | 24 | 24 | 12 | 0.500x |
| Merge outputs | 188 | 36 | 12 | 0.064x |
| Merge bytes | 335,907 | 332,505 | 166,131 | 0.495x |
| Projected source bytes | 1,843,200 | 1,843,200 | 1,843,200 | 1.000x |

Size 32 is selected over size 16 because it halves files, opens, membership
loads, and spills again; reduces median wall time another 6.1%; and adds no
material one-worker memory cost. Its recovery quantum is at most 32 source row
groups, which remains useful relative to the former 12,900-row-group workload
and prospective 34,400-row-group production root. The choice is not based on a
single fastest repetition.

## Telemetry and resource behavior

Parent completion telemetry now distinguishes total/completed source row groups,
total/reused/new count-route units, total/reused/new stats-route units, and route
range size. Worker telemetry separately counts source row groups, route units,
membership loads, reducer opens, spill runs/bytes, merge passes/outputs/bytes,
retries, and downshifts. ETAs retain the generic durable-unit count only within
explicitly named count-route or stats-route phases. No periodic durable telemetry
write was added.

The benchmark's minimum sampled host-available memory ranged from 4.33 to 5.95
GiB. One-worker peak RSS was 209.9-226.1 MiB. Two-worker peak RSS was 620.8 MiB
at 256 groups and 642.6 MiB at 1,024 groups. The extra memory is worker-count
parallelism, not range-width growth; focused uneven-row-group tests retain the
per-batch byte bound. No retry, downshift, pause, failure, or cleanup-failure
counter was nonzero.

## Runtime projection and limits

The bounded route/reduce fixture demonstrates structural acceptance and a 2.32x
median speedup at 1,024 row groups. It is deliberately sparse and is not a
production-shaped record-volume benchmark, so extrapolating the 93.2 s directly
to an integration or production root would be unsound. The plan's 5-12 minute
RNG-per-root integration projection and 55-90 minute complete integration-pair
projection are therefore retained as low-confidence projections, not measured
runtimes. Task 5A remains responsible for production-shaped capacity evidence.

Known limits:

- benchmark unit stamps are counted structurally rather than physically written;
- projected source bytes exclude repeated Parquet footer reads and OS cache effects;
- hashing timing covers actual route bytes but not the complete shared
  partition-runner publication/authentication lifecycle;
- the two-worker result has one measured repetition per scale;
- no full integration root pair or production pipeline was run.

## Verification

The focused verification set passed 92 tests: 24 RNG branch/contract tests, 15
partitioned-stage tests, six telemetry tests, 19 artifact-contract tests, 16
resource-config tests, four Task 3B benchmark tests, and eight applicable
stage-registry tests. It includes Windows-spawn one/two-worker execution,
two/eight diagnostic partitions, size-1/coarsened canonical equivalence,
collisions, duplicate coordinates, cap invariance, not-estimable output,
interruption/resume, route and downstream corruption, uneven row groups,
telemetry reconciliation, cleanup, and benchmark resume.

Static results:

- `ruff check .`: pass;
- `mypy src`: pass (84 source files);
- `git diff --check`: pass;
- Black on all four changed Python files: pass;
- Pyright on the changed production module, benchmark, and benchmark tests:
  pass with zero errors;
- repository-wide Black: fails on seven unchanged files already unformatted at
  the starting HEAD;
- repository-wide Pyright: 28 errors in unchanged pre-existing test typing (and
  four pre-existing typed-dict accesses in the retained RNG metadata test); no
  error is in changed Task 3B production/benchmark code;
- one pre-existing stage-registry spawn test remains red because its unchanged
  helper returns a tuple and compares it to `StageLayout.keys()`'s list. The
  other eight stage-registry tests pass.

Starting repository state was clean `main` at
`0364f63054bc919c18fe25fdc08e013e489c898b`. The ending state is intentionally
dirty only with the Task 3B source, focused tests, benchmark/evidence, and two
context-document updates listed in the final review.

No Task 4 work, H2H/checkpoint change, local-root feature, statistical setting
change, historical-artifact mutation, commit, or push occurred. Stop after Task
3B for human review.
