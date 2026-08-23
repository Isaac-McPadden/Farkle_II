# Task 6 RNG diagnostics architecture optimization

## Verdict

The architecture is implemented and the bounded performance gate passes. RNG
diagnostic method v4, RNG scheme v2, cache-key v6, grouping, eligibility, cap,
lag definitions, output schemas, and coordinate order are unchanged.

Against the accepted Task 3B width-32 baseline, the same production writers and
reducers now measure:

| Source row groups | Historical median | Current median | Speedup | Peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 12.517 s | 2.474 s | 5.06x | 212 MiB |
| 1,024 | 93.186 s | 10.283 s | 9.06x | 220 MiB |

Both current repetitions reproduce the exact historical final-file digest at
each scale. The conservative 5.06x observed speedup changes the central
two-root RNG-only projection from 167.5 to **33.1 hours**, below the 48-hour
target. Applying that factor to the former RNG plausible interval gives about
23.8-51.3 hours. This remains an extrapolation from a bounded structural
benchmark, not a production execution or full-pipeline readiness approval.

Machine-readable evidence:

- `task6_rng_architecture_benchmark.json`: sparse structural comparison with
  the historical byte oracle;
- `task6_rng_architecture_dense_benchmark.json`: 32,768-game dense route/reduce
  exercise, 12.70 s median and about 225 MiB peak RSS;
- `task6_rng_production_mix_benchmark.json`: paired 32,768-game workload with
  the production `k` distribution, full 12-seat records, 18.14 s median, and
  about 256 MiB peak RSS;
- `task6_rng_partitions32_benchmark.json` and
  `task6_rng_partitions64_benchmark.json`: bounded partition retuning evidence.
- `task6_production_capacity.json`: refreshed configuration-derived physical
  dimensions and stage-specific capacity projection.

## Architecture

- Combine checkpoint schema v3 buffers decoded source batches and emits full
  configured-size physical row groups except the final row group for each `k`.
  Logical schema and row order are unchanged; combine method remains v2.
- RNG partition/checkpoint schema and route layout advance to v3. The planner
  greedily groups canonical source row groups using expanded fixed-width bytes:
  64 MiB target and 96 MiB hard ceiling. Production dimensions resolve to 613
  combined row groups and 393 route units per root; the maximum projected unit
  is 78,208,000 bytes.
- Route workers reuse one Parquet reader per consecutive source file. Count
  keys are reduced across the complete route unit before stable partitioning.
  Count and statistics routes publish one contiguous Arrow batch per partition.
- Reducers retain pending arrays across route-file boundaries, use 256 MiB
  initial-run budgets, and group vectorized merges by both fan-in 32 and a
  deterministic byte ceiling. Every nonterminal generation consumes at least
  two runs.
- Matchup IDs use a compiled implementation of the existing personalized
  BLAKE2b-64 calculation. Lag moments use a non-fastmath compiled segmented
  scan with the same sequential operation order as the scalar accumulator.
  Array merges replace Python tuple/heap processing.

## Tuning decisions

The partition benchmark measured 7.22 s at 32 partitions and 8.99 s at 64
partitions on the same 4,096-game workload. Sixty-four partitions were 24.5%
slower and therefore fail the required 15% improvement gate; production stays
at 32. Fan-in remains 32, with the byte ceiling reducing effective fan-in only
when necessary for memory safety.

Reducer temporary work already resolves to
`C:\Users\imcpa\AppData\Local\Temp`, outside OneDrive. The accepted Task 3A
location effect remains 6.3%, below the 10% gate, so no configurable local route
workspace or publication-copy path was added. Sidecar and string cleanup remain
deferred secondary work.

## Correctness and lifecycle

- Historical Task 3B final benchmark digests are reproduced exactly:
  `6a3cd6d1...fd41393` at 256 row groups and
  `56d8b723...f01c892` at 1,024 row groups.
- Focused tests cover byte-plan bounds, route-local cross-batch reduction,
  partition-contiguous recovery, exact BLAKE2 vectors, bitwise scalar/compiled
  lag accumulation, digest collisions, duplicate coordinates, partition and
  worker invariance, corruption quarantine, and interruption/resume.
- Operational v2 checkpoints cannot be reused under v3. Historical artifacts
  remain untouched evidence; no migration, re-sidecarring, or production run
  was performed.

The refreshed capacity model projects 114.50 hours central, 3.40-7.52 days
plausible, and 9.03 days at the conservative planning upper. The complete
pipeline therefore remains `not_capacity_ready`; no production run was
authorized.
