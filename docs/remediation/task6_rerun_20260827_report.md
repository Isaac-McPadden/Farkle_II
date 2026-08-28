# Task 6 RNG benchmark rerun — 2026-08-27

## Result

The optimized RNG diagnostics remain correct and materially faster than the
historical Task 3B implementation. This session was consistently slower than the
original Task 6 benchmark session for the sparse, dense, and production-mix cases,
so the capacity projection has been recalibrated with the fresh conservative
speedup rather than retaining the earlier 5.06× factor.

The refreshed projection is **127.30 hours / 5.30 days central**, **3.78–8.35 days
plausible**, and **10.02 days conservative planning upper**. The verdict remains
`not_capacity_ready`; the 48-hour full-pipeline budget still does not pass.

No prior Task 5A or Task 6 document or evidence root was overwritten. No production
pipeline was run.

## Fresh benchmark results

All cases used the same settings as the committed Task 6 evidence and two measured
repetitions after a warm-up.

| Case | Fresh median | Prior Task 6 | Fresh/prior | Peak RSS | Correctness |
| --- | ---: | ---: | ---: | ---: | --- |
| Sparse, 256 row groups | 3.430 s | 2.474 s | 1.386× slower | 212.6 MiB | historical digest reproduced |
| Sparse, 1,024 row groups | 13.322 s | 10.283 s | 1.296× slower | 219.7 MiB | historical digest reproduced |
| Dense, 32,768 games | 16.224 s | 12.704 s | 1.277× slower | 224.1 MiB | prior optimized digest reproduced |
| Production-`k` mix, 32,768 games | 24.825 s | 18.139 s | 1.369× slower | 273.3 MiB | prior optimized digest reproduced |
| 32 partitions, 4,096 games | 7.297 s | 7.223 s | 1.010× slower | 216.8 MiB | repetitions identical |
| 64 partitions, 4,096 games | 11.590 s | 8.994 s | 1.289× slower | 216.0 MiB | repetitions identical |

Fresh measured wall times were:

- sparse 256: 2.994 s and 3.865 s;
- sparse 1,024: 13.830 s and 12.815 s;
- dense: 16.141 s and 16.306 s;
- production mix: 24.878 s and 24.771 s;
- 32 partitions: 7.341 s and 7.253 s;
- 64 partitions: 11.838 s and 11.342 s.

The identical historical sparse digests remain
`6a3cd6d1...fd41393` and `56d8b723...f01c892`. Dense and production-mix digests
also exactly reproduce the prior Task 6 values. This confirms that the runtime
shift is operational performance variation, not changed benchmark outputs.

The current 32- and 64-partition physical bundle digests differ by construction.
An independent canonical comparison excluding the partition-routing column found
exact equality: 3,825 eligibility rows with digest `91726979...ea54` and 5,955 stats
rows with digest `591391db...ba0` in both cases.

## Partition decision

The fresh 64-partition case is **58.8% slower** than 32 partitions (11.590 s versus
7.297 s), with no RAM benefit. This is stronger evidence for retaining the current
32-partition production setting than the earlier 24.5% penalty.

The prior partition-tuning files are not byte-comparable fixtures to this rerun:
the current generator produced 12,288 reduced count records versus 12,271 and a
177,940-byte source versus 173,571 bytes previously. Therefore only the paired
fresh 32/64 comparison is used for the tuning conclusion.

## Refreshed capacity projection

Against the historical pre-optimization Task 3B width-32 medians, the fresh sparse
speedups are:

- **3.650×** at 256 row groups (`12.517 / 3.430`);
- **6.995×** at 1,024 row groups (`93.186 / 13.322`).

Using the lower 3.650× speedup as the conservative architecture coefficient changes
the central two-root RNG projection from the historical 167.53 hours to **45.91
hours**. This is close to independently inflating the prior 33.11-hour Task 6 RNG
projection by the fresh production-mix slowdown (about 45.32 hours).

| Stage | Refreshed central | Critical-path share |
| --- | ---: | ---: |
| Simulation | 63.80 h | 50.1% |
| RNG diagnostics | 45.91 h | 36.1% |
| H2H and pair analyses | 13.94 h | 11.0% |
| Other root analyses | 1.65 h | 1.3% |
| Authentication/finalization | 1.50 h | 1.2% |
| Fixed orchestration | 0.50 h | 0.4% |
| **Total** | **127.30 h / 5.30 d** | **100%** |

The plausible range is **90.77–200.36 hours / 3.78–8.35 days**. Applying the
existing correlated planning allowance gives **240.44 hours / 10.02 days**.
Simulation is now the largest projected stage, while RNG remains a substantial
second constraint.

## Interpretation

The rerun does not invalidate the optimization: even the slower session retains a
3.65×–6.99× improvement over the accepted historical implementation, exact outputs,
bounded RAM, and a substantially reduced RNG projection. It does show that the
original 5.06× conservative coefficient was optimistic for the current host session.

Only two measured repetitions exist per case, so the session-to-session slowdown
cannot be assigned confidently to CPU frequency, host load, thermals, storage state,
or another operational factor. The revised interval treats it as throughput
variation rather than claiming a code regression; the RNG source is unchanged since
the optimization commit, and the subsequent commit only changed fast-config seeds.

## New evidence files

- `task6_rerun_20260827_rng_sparse.json`
- `task6_rerun_20260827_rng_dense.json`
- `task6_rerun_20260827_rng_production_mix.json`
- `task6_rerun_20260827_rng_partitions32.json`
- `task6_rerun_20260827_rng_partitions64.json`
- `task6_rerun_20260827_production_capacity.json`
- `task6_rerun_20260827_summary.json`

Retained owned roots use the corresponding
`data/farkle-task3b-task6-rerun-20260827-*` names plus
`data/farkle-task5a-production-capacity-v3-rerun-20260827`.

## Verification and retention

- Focused Task 3B/Task 6 benchmark and Task 5A capacity-model tests: **11 passed**.
- Seven new JSON documents parse; the published capacity JSON is byte-logically
  identical to the retained capacity checkpoint.
- Every measured group reproduced its digest across both repetitions.
- Summary ratios were independently recomputed from raw case medians.
- `git diff --check`: **passed**.
- All six new data roots have their expected ownership marker. They contain 1,211
  files and approximately 123 MiB in total and are retained for review.
- Git status contains only the eight new `task6_rerun_20260827_*` documents; no
  existing tracked document was modified.
