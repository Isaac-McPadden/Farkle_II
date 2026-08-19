# Task 3A bounded storage-location experiment

## Verdict

The synchronized OneDrive tree was directionally slower in every paired RNG and H2H
measurement, and several metadata-sensitive subphases showed repeatable provider overhead.
The total effect was not material under the predeclared 1.10 wall-time ratio threshold:

- RNG median wall time was 15.338 s on OneDrive and 14.424 s locally, a 1.063 ratio
  (6.3% slower).
- H2H median wall time was 22.657 s on OneDrive and 22.137 s locally, a 1.023 ratio
  (2.3% slower).
- Every required fixture, output, manifest, sidecar, completion, block checkpoint, and final
  checkpoint-state digest was byte-identical across all six measured location runs.
- No retry, downshift, memory-pause, worker-failure, or cleanup-failure event occurred.

This is evidence of real synchronized-tree overhead in file-heavy subphases, but it does not
justify designing a configurable local working/checkpoint root at this point. No such design,
Task 3B work, Task 4 work, production configuration change, or production artifact change was
implemented.

The machine-readable evidence is
[`task3a_storage_benchmark.json`](task3a_storage_benchmark.json). The reproducible driver is
[`benchmark_task3a_storage.py`](../../scripts/benchmark_task3a_storage.py).

## Scope and safety

The historical seed-48/49 artifacts were not opened as benchmark state and were not modified,
resumed, resealed, or migrated. The experiment did not run the pipeline or the fast-config
integration pair.

The benchmark resolved and validated both absent target paths before creating anything. It
required an exact `farkle-task3a-` name prefix, stable absolute parents, separation from the
repository for the local root, containment within the repository and configured OneDrive root
for the synchronized root, and a benchmark-owned marker before every recursive cleanup.

| Location | Disposable path | Provider | Volume / filesystem | Physical device |
| --- | --- | --- | --- | --- |
| OneDrive | `S:\Libraries\OneDrive\Documents\Code Projects Parent Folder\Code Projects\Farkle Mk II\data\farkle-task3a-onedrive-benchmark` | OneDrive; client PID 51064 present; not paused | `S:`, NTFS, volume `0cbe99bc-3f8d-4c4c-b7d0-baee809f1749`, label `2TB SSD` | Disk 3, `SHGP31-2000GM`, NVMe, serial reported as `FFFF_FFFF_FFFF_FFFF.` |
| Local | `S:\farkle-task3a-local-benchmark` | None detected; outside all configured OneDrive roots | Same `S:` NTFS volume | Same Disk 3 device |

The OneDrive parent carried archive and pinned attributes; the local `S:\` parent did not carry
cloud-provider attributes. Both paths therefore used the same volume, partition, filesystem, and
reported physical device. There is no different-device confounder in the primary comparison.
Both disposable roots and all per-run directories were removed successfully. Final root cleanup
took 1.751 ms on OneDrive and 0.967 ms locally, with zero failures; both paths were independently
confirmed absent afterward.

## Exact workload and environment

The host was `Isaac_Desktop`, Windows 11 build 26200, with 16 logical CPUs. The repository venv
used CPython 3.12.10 and NumPy 2.3.5. Repository HEAD was
`ceaeb6257e4903ef7522d372fe543da9e90141a7`; the worktree was dirty only with the uncommitted
Task 3A implementation/evidence.

Both workloads used deterministic seed 30,048,049, Windows `spawn`, two requested and two
effective workers, a process-map window of four, one Python/Arrow/native thread per process, and
the same repository worker policy at both locations. Each location received one full warm-up.
Measured order was AB/BA/AB, where A is OneDrive and B is local. Locations never ran
concurrently.

The RNG workload used:

- one byte-identical 4 MiB source fixture containing 131,072 fixed-width 32-byte records;
- 32 route units of 4,096 records;
- 8 deterministic route partitions;
- every reducer opening all 32 route artifacts, for 256 reducer route opens and 32 MiB of route
  reads per run;
- fan-in 4, 256 initial spills, 88 merge outputs, and three merge generations per reducer (24
  aggregate merge passes);
- repository atomic publication, hash-bound v2 sidecars, validation, one deterministic checkpoint,
  one manifest, and one completion.

The H2H workload used:

- one byte-identical source contract read for every scheduled chunk;
- 16 blocks, 4 chunks per block, and 4,000 deterministic game coordinates per chunk, totaling
  256,000 game coordinates per run;
- four process-pool generations, identical checkpoint cadence, 64 block rewrites, and four
  deterministic execution-state rewrites;
- 512 fixed 32-byte payload records per block/chunk;
- repository atomic publication, hash-bound v2 sidecars, validation, manifest, aggregate, and
  completion calls.

Fixture creation occurred before each workload timer. Workload wall time includes process-pool
startup, source reads, computation, routing/reduction, hashing, publication, authentication, and
correctness hashing. Per-run recursive cleanup is reported separately. File counts cover
benchmark-owned logical operations plus known calls inside the v2 sidecar helper. Open/close
latencies cover explicitly instrumented Python operations; helper-internal latency is included in
publication timing. This was Python-level instrumentation, not kernel ETW tracing.

## Repetition-level results

| Repetition | Order | Location | RNG wall / CPU (s) | H2H wall / CPU (s) | RNG paired ratio | H2H paired ratio |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 1 | OneDrive | 16.254 / 14.875 | 24.199 / 20.016 | 1.117 | 1.061 |
| 1 | 2 | Local | 14.551 / 13.172 | 22.811 / 19.422 | — | — |
| 2 | 1 | Local | 14.424 / 12.641 | 20.797 / 17.375 | — | — |
| 2 | 2 | OneDrive | 15.241 / 13.797 | 22.217 / 18.516 | 1.057 | 1.068 |
| 3 | 1 | OneDrive | 15.338 / 13.812 | 22.657 / 18.719 | 1.131 | 1.023 |
| 3 | 2 | Local | 13.559 / 11.938 | 22.137 / 18.438 | — | — |

The paired OneDrive/local ratios were 1.117, 1.057, and 1.131 for RNG (paired median 1.117), and
1.061, 1.068, and 1.023 for H2H (paired median 1.061). The ratio of location medians is the primary
summary below. The pooled first-position/second-position median ratio was 1.054 for RNG and 1.020
for H2H, so first-run order modestly increased time. Reversing the order in repetition 2 did not
reverse the location direction.

| Workload | OneDrive wall median [range] (s) | Local wall median [range] (s) | OD/local | OneDrive CPU median [range] (s) | Local CPU median [range] (s) | OD/local |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RNG | 15.338 [15.241, 16.254] | 14.424 [13.559, 14.551] | 1.063 | 13.812 [13.797, 14.875] | 12.641 [11.938, 13.172] | 1.093 |
| H2H | 22.657 [22.217, 24.199] | 22.137 [20.797, 22.811] | 1.023 | 18.719 [18.516, 20.016] | 18.438 [17.375, 19.422] | 1.015 |

## Operation counts and bytes

Counts and byte volumes were identical in every measured repetition at both locations.

| Measurement per run | RNG | H2H |
| --- | ---: | ---: |
| Source bytes read | 4,194,304 | 4,096 |
| Route bytes read | 33,554,432 | 0 |
| Temporary/staging bytes written | 20,972,882 | 2,648,464 |
| Durable publication bytes written | 8,438,901 | 2,728,450 |
| File creates | 422 | 142 |
| File opens / closes | 1,271 / 1,271 | 527 / 527 |
| Spill runs / spill bytes | 256 / 4,194,304 | 0 / 0 |
| Merge passes / merge outputs | 24 / 88 | 0 / 0 |
| Merge input / output bytes | 12,582,912 / 12,582,912 | 0 / 0 |
| Hash calls / hashed bytes | 143 / 25,220,203 | 189 / 7,442,955 |
| Sidecar publications / bytes | 41 / 48,672 | 66 / 79,174 |
| Checkpoint writes / bytes | 1 / 137 | 68 / 2,723,399 |
| Scheduler events | 124 | 200 |
| Retries / downshifts / memory pauses / worker failures | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

H2H's 66 sidecar publications comprise 64 block-checkpoint publications plus manifest and
aggregate publications; final unique sidecars are fewer because each block sidecar is rewritten
at the unchanged per-chunk cadence. Its 68 checkpoint writes comprise those 64 block rewrites
plus four execution-state rewrites.

## Phase and filesystem timings

| Workload / metric | OneDrive median [range] (s) | Local median [range] (s) | OD/local |
| --- | ---: | ---: | ---: |
| RNG measured open latency | 0.296 [0.294, 0.301] | 0.227 [0.217, 0.234] | 1.302 |
| RNG measured close latency | 1.797 [1.760, 1.900] | 1.351 [1.351, 1.471] | 1.330 |
| RNG spill/merge | 0.634 [0.632, 0.687] | 0.473 [0.470, 0.500] | 1.341 |
| RNG hashing | 0.116 [0.113, 0.124] | 0.104 [0.104, 0.119] | 1.118 |
| RNG sidecar publication | 0.445 [0.425, 0.458] | 0.359 [0.353, 0.516] | 1.240 |
| RNG checkpoint rewrite | 0.0079 [0.0062, 0.0104] | 0.0046 [0.0041, 0.0049] | 1.700 |
| RNG cleanup | 0.170 [0.167, 0.174] | 0.123 [0.112, 0.129] | 1.376 |
| H2H measured open latency | 0.0455 [0.0413, 0.0460] | 0.0350 [0.0341, 0.0356] | 1.301 |
| H2H measured close latency | 0.254 [0.250, 0.290] | 0.232 [0.232, 0.233] | 1.096 |
| H2H hashing | 0.136 [0.136, 0.143] | 0.131 [0.128, 0.132] | 1.042 |
| H2H sidecar publication | 0.718 [0.713, 0.767] | 0.596 [0.587, 0.597] | 1.206 |
| H2H checkpoint rewrite | 0.723 [0.713, 0.772] | 0.596 [0.590, 0.598] | 1.213 |
| H2H aggregate checkpoint queue time | 26.602 [24.503, 28.438] | 23.234 [22.734, 25.511] | 1.145 |
| H2H cleanup | 0.0267 [0.0233, 0.0279] | 0.0184 [0.0177, 0.0196] | 1.450 |

RNG median merge throughput was 18.93 MiB/s on OneDrive versus 25.39 MiB/s locally. The local
H2H median checkpoint throughput was approximately 4.36 MiB/s; OneDrive was approximately
3.59 MiB/s. Queue time is the sum of per-task elapsed time from submission to worker start, not
wall time, so it can exceed the workload duration and should be interpreted only as scheduler
contention exposure.

The phase ratios show repeatable provider cost where metadata activity is dense. In absolute
terms, however, the median OneDrive penalties were approximately 0.91 s for the complete RNG
workload and 0.52 s for the complete H2H workload. They did not reach the 10% total-wall threshold.

## Resource measurements

Requested and effective worker counts were two in every run. Native threads were capped at one
per process. Peak process-tree RSS was stable:

- RNG: 268.3–268.9 MiB on OneDrive and 267.9–269.4 MiB locally.
- H2H: 262.3–262.5 MiB on OneDrive and 262.4–262.5 MiB locally.

Minimum sampled host-available memory ranged from 4.85 to 6.58 GiB across measured runs, showing
a time-varying host background load but no benchmark memory-pressure event. Aggregate current
commit, Windows Job peak, and Job hard limit were unavailable because the standalone benchmark
was not launched inside the pipeline's aggregate Windows Job boundary; the report records these
as null/zero rather than relabeling process-tree RSS. No telemetry monitoring error occurred.

## Correctness digests

Each digest below had exactly one value across three OneDrive and three local measured runs.

| Artifact class | RNG SHA-256 | H2H SHA-256 |
| --- | --- | --- |
| Source fixture | `6ee59943cab5e22533abbf75c5c7186c319710564c09d623800c779f68d6c2a3` | `9daff6d76918e1ce368a28906f02d9c81ced6ce40d0ab43f87058926cd78e3ae` |
| Canonical fixture output / aggregate | `5b684c2ea6bb64623864bfdc9992ec01fb41577e4e723f8b1787438102d83267` | `8b74d489f3112e0ba0b3f0aa9cddd075ea2b90022f617cd208132707f69bfaad` |
| Manifest | `46d25f797970516f58a296059b25d7a52873d7cd41f13989904d1179091134d7` | `cf5b632cd60b38c077494e0a26c48951ab52ec25c7b03dec168ada9ace431815` |
| Sidecar bundle | `2533c3941af0c45338bdaf5faa1d6984c5632030a1486f35c07e4bc3f61ae309` | `5759edd4fa98e1063cd46ccba72b40466bfdac1855689bfb9346e6aa0f06eae8` |
| Completion | `780833967333f36122ce19621c39096e0a590f2be9e315bb90c360faaafbbbdc` | `95340de7854b37baa63fb7c7d47386906e061c60565fb0fd32667a7a8e48db5d` |
| Final deterministic checkpoint state | `5d51aca10b3b6775d7cd55a44af8d01c89d312a49308a69c5a2c92797b6b6b71` | `5109677fd878dfdb63e1ebc43d61ddbcdd62a139575175bb3838e49eee6b2362` |
| H2H final block-checkpoint bundle | — | `4f9f3792339f877614b5c9ff865e3f68e669ab81db0ae8bfba844017ad811140` |

There was no difference to investigate. Authentication was not weakened, paths inside each
fixture were path-independent and deterministic, and no statistical method, schema, checkpoint
cadence, production path, or production behavior changed.

## Confounders and decision boundary

- OneDrive remained in its normal observed operating state. Provider activity was not paused or
  controlled, so background synchronization is part of the treatment and may vary over longer
  periods.
- Three paired repetitions are sufficient to establish the direction in this bounded fixture but
  not to estimate long-horizon OneDrive variance or rare provider stalls.
- Available host memory varied by about 1.7 GiB over the experiment. AB/BA/AB ordering reduces,
  but does not eliminate, time-order and background-load effects.
- The benchmark uses the same physical device, which is the correct isolation for provider
  overhead but does not answer whether a faster separate scratch device would help.
- Python-level timing does not provide kernel filesystem/ETW queue decomposition.
- The workload preserves production-shaped operation structure at bounded scale; it does not
  project full fast-config or production runtime.

Decision: synchronized-tree overhead is measurable and directionally repeatable, especially for
opens, closes, merge files, sidecars, checkpoints, and cleanup. It is not material and repeatable
at the complete-workload level under the declared 10% threshold. Task 3A therefore does not, by
itself, justify a configurable local working/checkpoint root. Stop for review before any Task 3B,
Task 4, local-root architecture, or production configuration work.
