# Task 7B — Deterministic H2H block-range bundling

## Closed verdict

`performance_gate_failed`

The migration is not accepted. The production implementation and its production-test edits were restored to the starting revision. The current one-Parquet-plus-sidecar-per-block layout remains authoritative.

The decisive primary result was a median real-path publication reduction from 79.800 seconds to 26.624 seconds for the preregistered 180-block case: 66.64%, below the non-waivable 70% gate. The 760-block case did pass at 79.70%, and total wall time improved in both cases, but the closed plan identifies the 180-block case as primary. Median critical-worker simulation time also exceeded the 5% regression threshold in both phase-separated comparisons. No post-result waiver is applied.

## Provenance and scope

- Starting and ending HEAD: `2c7c97183c544e5ada8a050747727fb39ce7c283` (`main`, aligned with `origin/main` at entry).
- Initial worktree: one user-owned untracked file, `docs/remediation/PLAN Revised.md`; it was not modified.
- Benchmark root: `data/farkle-task7b-h2h-bundling-v1`, guarded by `task7b_ownership.json`.
- Baseline was captured before any production edit and was never regenerated or overwritten afterward.
- Final summary SHA-256 values are `aa211509a85f8d423780cb791404bedae0c355538058c756087a1dee3c202d9b` (baseline) and `79bbb9c930e7189ce5f359036e715110262c292e03f9bc6a8328f398619ac2eb` (candidate).
- No historical accepted artifact tree, global authentication policy, statistical method, RNG identity, candidate selection, H2H target, cap, or power rule changed.
- No commit, push, pull request, Task 8 work, production run, local scratch design, or process isolation was performed.

## Evaluated design

The rejected candidate used physical-layout identity version 2 and deterministic half-open ranges over the canonical `(pair_id, root_index, order)` schedule stream. Each bundle contained at most 16 adjacent rows; ranges could cross pair, root, and seat-order boundaries because those are physically independent immutable RNG coordinates. Names encoded bundle ID and exact start/stop offsets.

Each candidate bundle was an atomic Snappy Parquet plus authenticated v3 sidecar and exact `PartitionedUnit`-style stamp. A canonical final manifest authenticated every expected range, output hash, stamp hash, row support, layout identity, scientific/config identity, and code identity. The scanner rejected old-layout files, mixed layouts, extras, missing ranges, invalid sidecars/stamps, and noncanonical coordinates. The aggregate continued to expose the original ordered logical row stream.

Worker tasks remained block-wise. Out-of-order completions accumulated only until every member of that block's bundle generation was ready, after which the complete bundle was published and buffered results were released. The declared parent buffer limits were:

- rows: `max(64, 4 * workers * max_in_flight_per_worker * 16)` (768 at 12 workers);
- serialized result bytes: 16 MiB.

Observed final-matrix peaks were 38 rows / 47,142 bytes in the 180-block case and 42 rows / 52,248 bytes in the 760-block case. No submission pause, memory pause, retry downshift, orphan worker, or unresolved publication tail occurred. The largest measured final-worker-to-durable-bundle tail was 0.836 seconds.

The candidate used 12 requested/resolved/created/live workers under Windows `spawn`. A transient Windows `ProcessPoolExecutor` observability race was diagnosed during calibration; the evaluated candidate used a bounded topology-observation interval and still failed if the declared topology was not reached before a result completed. That shared change was rolled back with the rejected migration.

## Preserved baseline and measured repetitions

One warm-up per phase was excluded. Every listed repetition used the real H2H engine, target 1,372 completed games per block, identical deterministic schedules and game coordinates within its case, the synchronized repository volume, Snappy, Windows spawn, 12 workers, real sidecar/authentication publication, aggregation, inference, and a fresh release audit.

| Case | Phase | Repetition | Publication (s) | Total wall (s) |
|---|---:|---:|---:|---:|
| 180 blocks | baseline | 1 | 103.744 | 164.714 |
| 180 blocks | baseline | 2 | 79.800 | 133.576 |
| 180 blocks | baseline | 3 | 77.644 | 134.183 |
| 180 blocks | candidate | 1 | 15.636 | 123.060 |
| 180 blocks | candidate | 2 | 26.624 | 106.158 |
| 180 blocks | candidate | 3 | 29.648 | 114.496 |
| 760 blocks | baseline | 1 | 482.821 | 652.289 |
| 760 blocks | baseline | 2 | 355.655 | 548.593 |
| 760 blocks | baseline | 3 | 460.539 | 641.263 |
| 760 blocks | candidate | 1 | 153.963 | 515.313 |
| 760 blocks | candidate | 2 | 93.501 | 440.635 |
| 760 blocks | candidate | 3 | 85.287 | 438.687 |

| Gate metric | 180 blocks | 760 blocks |
|---|---:|---:|
| Baseline publication median | 79.800 s | 460.539 s |
| Candidate publication median | 26.624 s | 93.501 s |
| Publication reduction | **66.64% (fail)** | 79.70% (pass) |
| Baseline total-wall median | 134.183 s | 641.263 s |
| Candidate total-wall median | 114.496 s | 440.635 s |
| Total-wall change | -14.67% | -31.29% |
| Baseline critical-worker simulation median | 38.751 s | 180.491 s |
| Candidate critical-worker simulation median | 54.411 s | 214.213 s |
| Critical-worker simulation change | **+40.41% (fail)** | **+18.68% (fail)** |

The simulation regressions are plausibly affected by host-time/load separation because logical coordinates are exact, but all three candidate medians are observed evidence and the plan forbids waiving a greater-than-5% result after inspection.

## Publication, authentication, files, and bytes

Median secondary timings:

| Metric | 180 baseline | 180 candidate | 760 baseline | 760 candidate |
|---|---:|---:|---:|---:|
| Pool generation wall | 105.424 s | 108.276 s | 515.136 s | 419.244 s |
| Execution-state publication | 4.499 s | 5.422 s | 4.079 s | 11.244 s |
| Final block/bundle authentication | 3.237 s | 0.443 s | 13.091 s | 2.247 s |
| Aggregate publication | 6.196 s | 0.463 s | 25.045 s | 1.841 s |
| Stage completion publication | 12.994 s | 1.222 s | 54.651 s | 5.068 s |
| Fresh release audit | 2.472 s | 0.345 s | 11.171 s | 1.013 s |

Physical checkpoint inventory:

| Case | Current Parquets | Candidate bundles | Current data bytes | Candidate data bytes | Current sidecar bytes | Candidate sidecar bytes |
|---|---:|---:|---:|---:|---:|---:|
| 180 | 180 | 12 | 4,771,796 | 339,081 | 1,981,980 | 132,228 |
| 760 | 760 | 48 | 20,147,580 | 1,361,149 | 8,368,360 | 528,912 |

Median authentication telemetry fell materially. For 180 blocks, artifact/manifest/sidecar opens changed from 1,734/774/5,952 to 319/109/999, SHA-256 calls from 4,621 to 781, and bytes hashed from 93,429,462 to 14,065,715. For 760 blocks the corresponding changes were 6,954/3,094/23,932 to 931/253/2,835, 18,541 to 2,185 SHA calls, and 437,602,169 to 49,999,540 hashed bytes. Every measured release audit returned no failures.

The heartbeat's final resource sample occurs after pool shutdown, so it is not a process-tree peak oracle. Recorded final parent RSS ranged from about 254 to 326 MB and native threads from 49 to 52; the configured 12 GiB hard limit, 8 GiB warning threshold, 16-worker maximum, and 190-thread ceiling were not approached. Worker topology telemetry independently recorded all 12 candidate workers. No benchmark repetition exceeded the 30-minute or 4-GiB-written limit.

## Exact logical and downstream equivalence

All three repetitions within each phase shared one typed Arrow logical digest. Cross-phase comparison was exact for:

- canonical schedule digest and ordered coordinate support;
- Arrow field order, types, nullability, and 180/760 logical rows;
- games attempted/completed, replacement and safety counts, wins, statuses, and RNG-coordinate prefix hashes;
- aggregate logical digest and execution-state payload;
- all captured inference logical digests and decisions;
- family/pair membership and release-audit outcome.

Expected physical differences were limited to candidate bundle Parquets, row groups, paths, sidecars, unit stamps, final bundle manifest, layout telemetry, aggregate/source provenance, and completion provenance that bound those bytes. Aggregate Parquet bytes could differ physically because v3 source identities changed, while its typed logical content remained exact.

## Interruption, resume, and adversarial evidence

Before the performance stop fired, the focused candidate suite passed nine bundle selections covering exact 16-row/final partial ranges, cross-root/order ranges, stable names and Parquet bytes, interruption before an exact stamp, missing sidecar replay, corrupt final stamp replay, extra-artifact rejection, mixed-layout rejection, and forward/reverse worker-completion order. The existing targeted H2H/partition/parallel/benchmark suite also passed after its layout-specific assertions were updated.

Those tests established that an unstamped range was replayed as one bundle, valid unaffected bundle bytes were preserved, and no old/new mixed inventory was accepted. However, the full requested interruption matrix, representative 12–15-worker timed resume, and every adversarial mutation were intentionally not completed after the closed performance stop condition. Recovery therefore cannot independently qualify the migration, even though the tested cases were correct.

## Calibration and failed-run evidence

No failed or partial benchmark root was deleted. Retained calibration evidence includes memory-reserve, short-task Windows topology, provider permission, interrupted candidate, and redundant-validation attempts under the owned root's `_failed_*` children. The first complete candidate implementation measured 183.427 seconds of 760-block publication (60.2% reduction); it was retained as `_failed_candidate_v1_redundant_validation`. A bounded redundant post-stamp semantic read was removed for the final evaluated candidate. The final candidate summary is immutable at `data/farkle-task7b-h2h-bundling-v1/candidate/summary.json`.

## Candidate-count sensitivity and capacity

For two roots and two seat orders, blocks equal `2 * C * (C - 1)` and width-16 bundles equal `ceil(blocks / 16)`.

| Candidates | Pairs | Blocks | Bundles | Current checkpoint files | Rejected candidate checkpoint files |
|---:|---:|---:|---:|---:|---:|
| 75 | 2,775 | 11,100 | 694 | 22,200 | 2,084 |
| 77 | 2,926 | 11,704 | 732 | 23,408 | 2,198 |
| 100 | 4,950 | 19,800 | 1,238 | 39,600 | 3,716 |
| 125 | 7,750 | 31,000 | 1,938 | 62,000 | 5,816 |
| 143 | 10,153 | 40,612 | 2,539 | 81,224 | 7,619 |
| 150 | 11,175 | 44,700 | 2,794 | 89,400 | 8,384 |

The rejected layout would have changed maximum unauthenticated exposure from one block/5,000 attempts to at most 16 blocks/80,000 attempts. Scaling the two measured publication coefficients to 44,700 blocks produces a counterfactual saving range of 3.78–6.01 hours. The primary coefficient misses Task 7A's six-hour and 5%-central gates; both coefficient endpoints exceed only the three-hour conservative threshold. Because Task 7B failed, none of these counterfactual values updates capacity.

The operative model remains Task 7A: 127.30 hours central, 90.77–200.36 plausible, 240.44 hours planning upper, approximately 252.33 GB central durable storage, 231,857 final files, 25.77 GB central peak temporary storage, 4.83 GB central projected process-tree RSS, 16 processes, and 160 central/190 upper native threads. It does not fit six 18-hour windows centrally and remains `not_capacity_ready`.

## Verification and rollback

Commands completed during evaluation included:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/analysis/test_h2h_schedule.py tests/unit/utils/test_partitioned_stage.py tests/unit/utils/test_parallel_files.py tests/unit/scripts/test_benchmark_task7b_h2h_bundling.py -q
.\.venv\Scripts\python.exe -m pytest tests/unit/analysis/test_h2h_schedule.py -q -k 'bundle or completion_order'
.\.venv\Scripts\python.exe -m pytest tests/unit/utils/test_parallel_files.py tests/unit/utils/test_partitioned_stage.py -q
.\.venv\Scripts\python.exe -m ruff check <changed Python files>
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\pyright.exe <changed production files>
.\.venv\Scripts\python.exe scripts/benchmark_task7b_h2h_bundling.py --output data/farkle-task7b-h2h-bundling-v1 --phase baseline --workers 12
.\.venv\Scripts\python.exe scripts/benchmark_task7b_h2h_bundling.py --output data/farkle-task7b-h2h-bundling-v1 --phase candidate --workers 12
```

The production candidate and production-test changes were then restored exactly to HEAD. Post-rollback checks and JSON/worktree validation are recorded below before handoff.

Post-rollback results:

- the combined H2H schedule/inference/dominance, partitioned-stage, parallel, and benchmark-driver selection passed;
- the final benchmark-driver-only rerun passed 4/4;
- Ruff and Black passed on the retained Python deliverables;
- `mypy src` passed all 87 source files;
- Pyright initially identified one benchmark-only `object`-to-`float` cast, which was corrected; the final run passed with zero errors/warnings;
- both machine-readable deliverables parsed, then all 4,467 retained JSON files and all nine retained JSONL files under the owned root parsed with zero errors;
- `git diff --check` passed (Git emitted only the repository's expected LF-to-CRLF worktree warnings).

Final `git status --short` contains only the four context-document edits and the new Task 7B plan/driver/test/report/JSON/capacity files, plus the untouched user-owned `docs/remediation/PLAN Revised.md`. There are no production-source or production-test modifications.

## Retained roots, changed files, and limitations

Retained roots include the immutable baseline and final candidate, `_failed_candidate_v1_redundant_validation`, the quick optimized diagnostic, the independent quick real-path run, and the 12-worker topology diagnostic. Material data was not deleted; the failed candidate directory was moved within the exact owned root so evidence remained recoverable.

Final deliverable changes are limited to:

- `docs/remediation/task7b_h2h_bundling_benchmark_plan.md`;
- `scripts/benchmark_task7b_h2h_bundling.py`;
- `tests/unit/scripts/test_benchmark_task7b_h2h_bundling.py`;
- this report and its machine-readable JSON;
- `docs/remediation/task7b_production_capacity.json`;
- four Codex context/orientation files.

The user-owned `docs/remediation/PLAN Revised.md` remains untouched. The principal limitation is decisive rather than inferential: the primary median missed its preregistered threshold. The phase-separated simulation timings and synchronized-provider variance are additional unresolved risks. No production readiness claim is made.

Because the rejected candidate source was rolled back, reproducing candidate timings requires a newly reviewed implementation of the recorded design. The retained driver encodes separate baseline and width-16 physical-count oracles and will reject the current per-block source if it is accidentally invoked as a candidate; it cannot silently relabel the current layout.
