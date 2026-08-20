# Task 4A H2H execution and checkpoint efficiency

## Verdict

Task 4A is implemented with a deterministic cap-bounded checkpoint policy of
at most 5,000 attempted game coordinates per durable block state. The selected
policy keeps every currently admitted integration, production-like, and
maximum-planning block within one worker task and one block publication while
retaining a fixed recovery bound for any future larger target.

The scientific contract is unchanged. H2H method v2, score-test and exact-power
identities, candidate ordering, completed targets, maximum-attempt caps, root
and order balance, safety exclusion, replacements, RNG coordinates, output
paths, authentication, and final integer values remain identical.

Across the three normal two-worker targets (1,372, 1,974, and 2,191 completed
games per block), the fixed-1,000 median was 81.625 s and the selected-policy
median was 57.302 s: 0.702x wall time, or a 1.42x speedup. Chunks, block
publications, pool generations, and observed initializer loads changed from
28/28/7/14 to 12/12/3/6. Peak process-tree RSS changed from 224.23 MiB to
224.50 MiB, a 0.12% increase.

Across all six two-worker fixtures, including low-safety replacements, a mixed
exceptional tail, and all-nonviable blocks, the median changed from 68.641 s to
58.635 s. Chunks and block publications fell from 40 to 24, generations from
10 to 6, and initializer loads from 20 to 12. All 18 policy/scenario groups
had exact logical values, schemas and ordering, and byte-identical block and
aggregate Parquets within each scenario.

Machine-readable evidence is in
[`task4a_h2h_execution.json`](task4a_h2h_execution.json). The resumable driver is
[`benchmark_task4a_h2h_execution.py`](../../scripts/benchmark_task4a_h2h_execution.py).

## Starting-state reconciliation

The task request expected clean committed HEAD
`0364f63054bc919c18fe25fdc08e013e489c898b` plus eight dirty Task 3B files.
The actual starting state was clean `main`/`origin/main` at
`d64d61b732db097e00a76a1d4d1f661a9ec7146f`. Repository history explains the
difference completely: `d64d61b` is a single `Coarsened RNG route units` commit
whose changed-file inventory is exactly the eight accepted Task 3B files in the
Task 3B report. No unrelated change was present, so Task 4A proceeded from that
accepted committed state.

## Alternatives evaluated

The bounded benchmark ran the real `execute_h2h_schedule` path with the real
game engine, Windows `spawn`, worker initialization, semantic RNG coordinates,
authenticated block writes, execution-state writes, final block validation,
aggregate publication, and completion publication.

Three policies were compared:

1. Fixed 1,000 attempts, reproducing the original behavior.
2. Target-aligned upper bounds of 1,372, 1,974, or 2,191 attempts.
3. The selected cap-bounded upper bound of 5,000 attempts.

Target alignment handles ordinary no-safety blocks in one generation, but it
requires another generation and publication when replacements are needed and
two generations for a nonviable block. The cap-bounded policy covers the
frozen 2.0x attempt allowance in the same task. It therefore retains one
generation for ordinary, replacement, mixed-tail, and nonviable branches.

Persistent process-pool reuse was not implemented. It would not reduce the
selected policy's one generation for any admitted target or cap, while adding
new lifetime, cancellation, long-tail, memory-accounting, and Windows-spawn
complexity. Large future blocks still split deterministically every 5,000
attempts and use the existing bounded process-map recovery path.

The original fixed-1,000 implementation also made final column order depend on
whether a block had an intermediate checkpoint. The first equivalence run
correctly failed because the selected one-chunk output placed `wins_a` and
`wins_b` differently despite identical values. `_block_progress` now emits the
established fixed-1,000 terminal order explicitly. The benchmark schema was
advanced to v2 and every measurement was rerun. This is a normalization bug fix,
not a canonical schema change: the selected output now matches the original
fixed-1,000 terminal bytes.

## Recovery bound

Chunk size is an execution-only control under the repository's remediation
contract. It is absent from statistical configuration identity. A valid block
sidecar authenticates the exact prefix `0 .. games_attempted - 1`; changing the
upper bound may therefore resume that prefix, and the next plan starts exactly
at `games_attempted`.

Task 1B measured about 1,712 attempts/s over 15 occupied workers, or a
conservative reference rate of about 114 attempts/s per worker. The selected
fallback gives:

- 5,000 / 114 = 43.86 s for the largest possible durable interval;
- 4,382 / 114 = 38.44 s for the current maximum 2,191-target planning cap;
- 3,948 / 114 = 34.63 s for a production-like nonviable block;
- 1,974 / 114 = 17.32 s for an ordinary viable block, which stops as soon as
  its completed target is reached.

Thus the deterministic worst interval is within the approved 30–60 s recovery
window on the reference machine. If interruption is uniform within an interval,
expected replay is at most half the full interval, 21.93 s for the generic
5,000-attempt fallback. The deterministic maximum number of replayed
unauthenticated coordinates is 5,000; authenticated coordinates are never
replayed.

The selected two-worker normal fixtures measured at least 580.84 attempts/s per
worker, and the selected one-worker 1,974 fixture measured 269.08 attempts/s.
Those imply smaller replay times, but the policy uses the slower production
reference rather than aggregate or warm-cache benchmark throughput.

## Benchmark results

### Two-worker normal targets

| Policy | Wall times (s) | Median (s) | Chunks / block writes | Pool generations | Initializer loads | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 1,000 | 81.625, 73.099, 100.868 | 81.625 | 28 / 28 | 7 | 14 | 224.23 MiB |
| Target-aligned | 59.692, 48.047, 50.664 | 50.664 | 12 / 12 | 3 | 6 | 224.26 MiB |
| Cap-bounded 5,000 | 51.995, 57.302, 68.316 | 57.302 | 12 / 12 | 3 | 6 | 224.50 MiB |

All policies attempted and completed exactly 22,148 games. The target-aligned
median was lower in these no-safety runs, but it lost that structural advantage
in the exceptional fixtures. Selecting it would optimize the easiest branch at
the cost of more generations and publications for replacements and nonviable
blocks.

### All two-worker fixtures

| Policy | Individual wall times (s) | Median | Chunks / writes | Generations / loads | Checkpoint bytes | Sidecar publications / bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 1,000 | 81.625, 73.099, 100.868, 56.087, 64.183, 60.505 | 68.641 | 40 / 40 | 10 / 20 | 1,069,616 | 82 / 885,654 |
| Target-aligned | 59.692, 48.047, 50.664, 77.217, 58.890, 87.544 | 59.291 | 33 / 33 | 9 / 17 | 892,209 | 75 / 810,969 |
| Cap-bounded 5,000 | 51.995, 57.302, 68.316, 54.617, 61.563, 59.969 | 58.635 | 24 / 24 | 6 / 12 | 645,744 | 66 / 709,494 |

Execution-state cadence was held constant at 36 writes per policy. The
four-block fixtures intentionally exercise its small-schedule cadence and make
its cost visible; production schedules throttle at the existing larger block
interval.

The selected-policy summed phase evidence across all six fixtures was:

- 39.473 worker-seconds of simulation;
- 306.825 s of pool-generation wall exposure, which includes parent work while
  workers remain alive;
- 0.330 s of executor construction and 2.777 s of shutdown;
- 148.606 s of block Parquet/sidecar publication;
- 140.124 s of execution-state publication;
- 0.462 s of initial source authentication;
- 0.883 s of final block authentication;
- 15.657 s of aggregate publication; and
- 11.438 s of completion publication.

These categories deliberately expose both worker-time and parent wall-time and
therefore are not summed as mutually exclusive CPU accounting. Scheduler/IPC
residuals are also reported per measurement in the raw benchmark summaries.

### Exceptional outcomes

| Fixture | Attempted | Completed | Safety | Replacements | Nonviable blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low safety | 1,036 | 1,024 | 12 | 12 | 0 |
| Mixed tail | 640 | 384 | 256 | 128 | 1 |
| Always safety | 1,024 | 0 | 1,024 | 512 | 4 |

Every nonviable block stopped exactly at its frozen cap. All policies produced
the same wins, losses, safety attempts, replacements, completion statuses, root
totals, order totals, coordinate-prefix hashes, schemas, logical-row digests,
block bytes, and aggregate bytes.

### One-worker versus multi-worker

At target 1,974, the one-worker fixed/target/selected wall times were
78.426/72.313/56.254 s. Fixed used eight chunks and two logical generations;
selected used four chunks and one generation. For each policy, one-worker and
two-worker counts, schemas, logical digests, block-Parquet digests, and
aggregate-Parquet digests were identical.

## Coordinate, resume, and failure evidence

Focused tests and the benchmark cover:

- fresh, partial, small-target, large-target, and near-cap chunk planning;
- exact contiguous boundaries `0, 5000, 10000, 12000` for a larger block;
- ordinary exact-target completion;
- deterministic safety attempts and replacement coordinates;
- all-safety termination at the exact cap;
- resume from the smallest unauthenticated attempt;
- data-without-sidecar rejection and replay of only the affected block;
- corrupted authenticated checkpoints remaining untrusted;
- stable terminal schema and column order with or without intermediate state;
- exact root/order and aggregate integer reconciliation;
- execution-state throttling and missing-final-stamp recovery;
- one-worker and Windows-spawn multi-worker equivalence;
- pool generation and manifest-initializer counts;
- bounded submission, native-thread limits, resource guards, worker exception
  classification, cancellation, termination, and cleanup.

Worker timing metadata uses underscore-prefixed transient result fields. The
normalizer removes them before publication, so telemetry cannot alter canonical
rows or freshness.

## Production-shaped structural projection

For the historical production-like 11,704 blocks, fixed 1,000 produced about
23,488 block publications and four logical generations including the 40-block
nonviable tail. With the admitted 1,974 target and 3,948 cap, the selected
policy structurally projects 11,704 block publications and one generation: a
50.2% publication reduction and 75% generation reduction.

This is a low-confidence structural projection, not a runtime measurement. The
optimization does not reduce the dominant approximately 23.18 million game
attempts, and Task 5A remains responsible for production capacity evidence.

## Identity and compatibility decision

- H2H method remains v2.
- `SCORE_TEST_ID` and `POWER_METHOD_ID` are unchanged.
- No statistical, artifact, checkpoint, or route schema version changed.
- Canonical paths and artifact inventory are unchanged.
- The 5,000-attempt value is execution-only and does not contaminate
  statistical configuration identity.
- A different operational bound may resume an authenticated prefix because the
  prefix, target, cap, family, schedule, RNG, outcome, and code identities are
  validated independently.
- Old seed-48/49 artifacts remain read-only historical evidence and were not
  opened as work state, modified, resealed, or migrated.

## Verification

The final focused collection contains 117 tests across H2H scheduling,
benchmarking, telemetry, writers, artifact contracts, orchestration, shared
parallel behavior, and the structural integration oracle. The first combined
run passed 116 and exposed one new test bug: `zip(strict=True)` was used on two
intentionally offset boundary arrays. The test was corrected to
`strict=False` and passes on rerun; there is no remaining Task 4A failure.
Shared parallel's 31 tests include Windows-spawn worker exceptions and forced
termination and all pass. The dedicated real-engine worker-equivalence and toy
interruption/resume tests both pass.

Static results:

- `ruff check .`: pass;
- `mypy src`: pass for 84 source files;
- Black on all six changed Python files: pass;
- Pyright on changed production, benchmark, and test files: zero errors;
- `git diff --check`: pass;
- machine-readable evidence JSON parsing: pass;
- repository-wide Pyright: the same 28 pre-existing test errors;
- repository-wide Black: six unchanged files remain unformatted. The prior
  baseline was seven; `parallel.py` was the seventh and is now formatted because
  Task 4A necessarily touched it for pool lifecycle telemetry; and
- stage registry: eight pass and the unchanged tuple-versus-list spawn
  assertion fails with the previously recorded provenance.

## Temporary roots and limits

The following owned disposable roots are intentionally preserved for review:

- `data/farkle-task4a-quick-w2` — benchmark-schema-v1 Windows-spawn warm-up;
- `data/farkle-task4a-bounded-w2-v2` — accepted 18-measurement matrix;
- `data/farkle-task4a-bounded-w1-v2` — accepted one-worker comparison;
- `data/farkle-task4a-bounded-w2` — superseded first equivalence run that
  exposed the column-order defect.

The superseded root is not accepted evidence. No historical or canonical result
tree was mutated. No full integration pair or production pipeline was run.

Task 4B was not started. No commit, push, pull request, local-working-root
feature, RNG diagnostic change, power/inference change, or authentication-graph
deduplication was performed.
