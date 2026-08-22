# Task 5A production capacity and readiness gate

## Verdict

`not_capacity_ready`

The current official two-seed production pipeline is projected at **10.47 days**
(251.3 hours) on this machine and synchronized OneDrive storage. The plausible
interval is **7.48–16.44 days**; the conservative planning upper bound is **19.73
days** (474 hours). None of the 8, 12, 24, or 48 hour comparison bands pass.

This is a capacity decision, not a correctness failure. The accepted fast run is
healthy, the bounded model reconstructs its 1,663.156 s measured runtime within the
declared 10% tolerance, and the official statistical scope is unchanged. A human
would need to approve an unattended budget of at least 474 hours for the current
planning upper bound to pass. No such budget is selected here.

Task 5B was not triggered: there is no new reproducible cross-stage-retention or
resource-failure evidence. No production run, optimization, subprocess isolation,
statistical-scope reduction, commit, or push was performed.

Machine-readable evidence is in
[`task5a_production_capacity.json`](task5a_production_capacity.json); the retained
owned evidence root is `data/farkle-task5a-production-capacity-v1`.

## Current executable production dimensions

Every value below was resolved from `configs/farkle_mega_config.yaml` and current
code at the clean `df0551e` branch tip. Labels mean: **measured** = existing runtime
or file evidence; **configured** = literal active configuration; **derived** = exact
current planning/code arithmetic; **projected** = capacity estimate.

| Quantity | Current value | Label |
| --- | ---: | --- |
| Roots | 102, 103; sequential | configured |
| Strategies | 5,160 | derived |
| Player counts | 2, 3, 4, 5, 6, 8, 10, 12; sequential | configured |
| Shuffles per root/k | 4,300 (100 batches × 43) | derived |
| Games/root, k=2/3/4/5 | 11,094,000 / 7,396,000 / 5,547,000 / 4,437,600 | derived |
| Games/root, k=6/8/10/12 | 3,698,000 / 2,773,500 / 2,218,800 / 1,849,000 | derived |
| Attempted games/root | 39,013,900 | derived |
| Completed games/root | 39,013,900 central; safety-limit variation remains a risk | projected |
| Player exposures/root | 177,504,000 | derived |
| Source Parquets/rows/row groups per root | 34,400 / 39,013,900 / 34,400 | derived |
| Production row-group widths by k | 2,580 / 1,720 / 1,290 / 1,032 / 860 / 645 / 516 / 430 rows | derived |
| RNG source row groups/root | 34,400 | derived |
| Route size and route units/root/route | 32 row groups; 1,075 units | configured/derived |
| Count-route records/root | 216,517,900 (games + exposures) | derived |
| Statistics-route records/root | 181,405,390 central; 177,504,000–216,517,900 | projected |
| Reducers and reducer route opens/root | 32; 68,800 across both routes | configured/derived |
| Initial spill files/root | 68,800 | projected from current per-route flush boundary |
| Merge fan-in | 32 | configured |
| Actual production merge depth | 3 generations per reducer route: 1,075→34→2→1 | derived |
| Merge output files/root | 2,368 across both routes | derived |
| Performance bootstrap units/root | 40 (2,000 replicates / 50) | derived |
| Root-stability partition units | 40 top-N + 40 joint discrepancy | derived |
| Candidate upper envelope | 150 | derived |
| H2H pairs/root-orientation blocks | 11,175 / 44,700 | derived |
| H2H target completed per block | 2,191 | derived by current exact first crossing |
| H2H completed/maximum attempted | 97,937,700 / 195,875,400 | derived |
| H2H checkpoint bound/checkpoints | 5,000 attempts; one durable extension per current block cap | configured/derived |
| Projected artifacts/sidecars/completions | 114,430 / 114,430 / 43 | projected |
| Projected partition stamps/manifests | 4,782 / 70 | projected |
| Projected final file inventory | 237,281 files | projected |
| Top-level fresh byte-deep audits | exactly one | configured contract |

The source-byte projection is approximately 25–35 GiB per root after allowing for
both row bytes and 34,400 small-file headers. Count and statistics route bytes add
roughly 30 GiB and 17–38 GiB per root; normalized ingest/combine data and other
products bring central durable storage to **235 GiB** across the pair.

### Reconciliation with the historical Plan

- The historical 39,013,900 games/root, 177,504,000 exposures/root, 34,400 row
  groups/root, 216.5 million count records/root, 150 candidates, and 97.94 million
  H2H completed games still match current executable planning.
- The current grid is **5,160 strategies**, not the older 7,140 comment still visible
  in a simulation docstring. Executable strategy generation is authoritative.
- Task 3B changed the merge topology. With 32-row-group route units there are 1,075
  possible initial runs per reducer route, so the current depth is three, not the
  Plan's pre-coarsening four. A deterministic structural case with 32,769 initial
  runs exercises the real four-pass fan-in topology
  (`32,769→1,025→33→2→1`). Production is not charged for a nonexistent fourth pass.
- The H2H target remains exactly 2,191 per root/orientation block at 150 candidates;
  the resulting 44,700 blocks remain below the current one-extension 5,000-attempt
  checkpoint bound because each 2.0× cap is 4,382.

## Benchmark design and boundedness

The plan was published before measurement in
[`task5a_benchmark_plan.md`](task5a_benchmark_plan.md). The smallest identifying set
reuses accepted, current-implementation real-path evidence instead of rerunning a
large matrix:

| Evidence case | Raw result used | Model role |
| --- | --- | --- |
| Task 4C simulation | 24,000 k=2 games: 86 s at 1 worker, 43 s at 2, 21 s at 12; 12-worker peak 3.33 GiB, 182 CPU-s, 12 workers | startup, scaling, CPU/RAM, real engine |
| Task 4D simulation | two roots; 106.891 s and 98.063 s; all k=2/4/5 pools created and peaked at 12 workers | exposure throughput and startup/shutdown dispersion |
| Task 3B RNG sweep | 256/1,024 row groups; route widths 1/16/32; two repetitions; width-32 medians 12.52/93.19 s at one worker | source-unit, route/open, spill, merge structure |
| Task 4D RNG | 1,800 row groups/root, 57 units/route, count/stats rows 189,600/~168k, 229.6 s/root | real schema/record-density throughput |
| Task 4A H2H | real engine targets 1,372/1,974/2,191 plus viable/replacement/nonviable branches | target, completion, resume, fixed block cost |
| Task 4D H2H | 184,320 attempts/completions, 144 blocks, 15 initializer loads, 527.85 worker CPU-s, 118.61 s wall | game/block/publication separation and actual topology |
| Task 4B authentication | 9/128/512 artifacts; 512 optimized finalization 3.844 s and snapshot 1.470 s | open/artifact coefficient |
| Task 4D audit | 4,004 canonical artifacts, 97,353 cumulative opens, 10.99 GB cumulative bytes hashed, final audit 76.328 s | bytes/open reconciliation at integration scale |

The evidence root contains only dimensional/model checkpoints. Existing benchmark
and integration trees were read as immutable evidence. The seed-54/55 tree was not
written, resealed, reinterpreted, or used as resumable work.

## Model and fitted coefficients

Stage-specific terms are used; roots and `k` remain sequential.

| Coefficient/term | Estimate | Residual/dispersion | Interpolation or extrapolation |
| --- | ---: | --- | --- |
| Simulation exposure throughput | 1,545.7 exposures/s | Task 4D root wall times differ 8.7%; Task 4C scaling is sublinear | k=3/6/8/10/12 and 15-worker rate extrapolated; -35%/+39% |
| RNG combined route throughput | 1,557.1 records/s | root stage times differ 0.007%; record densities differ slightly | record volume extrapolated ~1,100×; explicit 18% third-generation factor; -35%/+39% |
| Other root source-unit term | 70% of calibrated scale term | root aggregate times differ 2.1% | 1,800→34,400 row groups extrapolated |
| Other root row term | 30% using square-root row growth | not separately identifiable from one integration scale | row count extrapolated; deliberately wide 0.45×–2.40× stage interval |
| H2H game term | 20,219 s | 15 worker rates span 334.7–369.1 attempts/s | 531× game extrapolation |
| H2H block checkpoint term | 21,234 s | one 144-block integration anchor plus Task 4A target sweep | 310× block extrapolation |
| H2H publication term | 5,429 s | one integration anchor; Task 4A policy-equivalence cases bound behavior | 310× block extrapolation |
| Finalization | 1.5 h central | Task 4B nonlinear 9/128/512 sweep; Task 4D 76.3 s audit anchor | artifact count and bytes extrapolated separately; 0.75–3.75 h |

Robust summaries use medians where repetitions exist. The model does not report
sub-second production precision. Timer-resolution digits in JSON are retained only
for reproducibility; the report rounds them.

### Fast integration validation

The reconstruction uses measured stage envelopes and removes the overlapping
`pair_analysis` wrapper before summing individual pair stages. It predicts
1,663.156 s versus 1,663.156 s measured (0.0% reconstruction residual, tolerance
10%). This validates stage accounting, not production linearity. Production remains
a 19× source-unit, ~1,100× RNG-record, 531× H2H-game, and ~28× file-count
extrapolation, which is why the interval is deliberately broad.

The measured fast peak process-tree RSS was 3.91 GiB; the production estimate does
not scale RSS with total records because processing is streamed and roots/`k` are
sequential. The projected central peak is 4.5 GiB, planning peak 8 GiB, below the
configured 12 GiB hard limit.

Task 4C's 12-worker simulation used 182 CPU-s over 21 s wall, about 8.7 logical
cores on average (58% of the 15-core execution budget) across startup, execution,
publication, and drain. Task 4D H2H recorded 527.85 worker CPU-s; over the 38.05 s
critical simulation window that is about 92.5% of the 15-worker capacity, while
the full 118.61 s envelope is lower because block publication is intentionally on
the critical path. These are the CPU-seconds/utilization anchors; production CPU
hours are not presented more precisely than the stage intervals.

## Production critical path

| Stage | Central | Plausible range | Critical-path share |
| --- | ---: | ---: | ---: |
| RNG diagnostics | 167.5 h | 120.6–259.7 h | 66.7% |
| Simulation | 63.8 h | 45.9–98.9 h | 25.4% |
| H2H and pair analyses | 13.9 h | 10.0–21.6 h | 5.5% |
| Other root analyses | 4.1 h | 1.8–9.8 h | 1.6% |
| Authentication/finalization | 1.5 h | 0.75–3.75 h | 0.6% |
| Fixed orchestration | 0.5 h | 0.25–1.0 h | 0.2% |
| **Total** | **251.3 h / 10.47 d** | **179.4–394.7 h / 7.48–16.44 d** | **100%** |

The conservative planning upper bound is 473.6 hours / 19.73 days; it applies a
further 20% correlated-host/storage allowance rather than summing incompatible
best/worst observations.

## Capacity requirements

| Resource | Central | Planning bound | Basis |
| --- | ---: | ---: | --- |
| Process-tree RAM | 4.5 GiB | 8 GiB; configured hard limit 12 GiB | measured 3.91 GiB; streamed/sequential topology |
| Process count | 16 | 16 | parent + 15 H2H/production workers |
| Native threads | 160 | 190 | measured peak 145 |
| Durable storage | 235 GiB | 350 GiB | source, route, normalized analysis, H2H, sidecars |
| Peak temporary storage | 24 GiB | 48 GiB | four simultaneously admitted reducer workspaces and merge generations |
| Final files | 237,281 | uncertainty included in storage interval | source shards, sidecars, checkpoints, completions, manifests |

Temporary space is in addition to durable space; a conservative free-space gate is
therefore approximately **400 GiB**. This is a logical/provider-visible estimate;
reflink physical allocation and OneDrive hydration can change physical occupancy.

## Sensitivity and interruption exposure

- A 25% throughput loss adds roughly one third of the simulation, RNG, and H2H
  compute terms, pushing the run several days later.
- Applying Task 3A's 6.3% RNG metadata sensitivity to I/O-heavy stages adds roughly
  11 hours centrally. Task 3A remains the accepted decision: the overall synchronized
  penalty was below the 10% materiality threshold, so no local working root is added.
- Candidate count has a quadratic pair term and a multiplicity-dependent power term.
  The 150-candidate envelope is the capacity gate; a smaller realized frozen family
  would reduce H2H, not RNG or simulation.
- A 0.99 completion ratio increases attempted H2H game work by about 1.02%; lower
  completion can approach the frozen 2.0× attempt cap and is covered by the upper
  interval.
- Statistics-route density can vary from 177.5 to 216.5 million records/root; merge
  depth stays three while throughput and bytes change.
- A half-worker retry/downshift can nearly double the active simulation/RNG stage and
  is incompatible with the central estimate; one successful retry is nevertheless
  resumable from authenticated units.
- Simulation interruption loses at most the active deterministic process block and
  reuses authenticated row/metric manifests. RNG interruption reuses completed
  32-row-group route units or reducer partitions. H2H interruption replays at most
  the active 5,000-attempt extension. Long RNG stages still create the largest wall
  exposure even though durable recovery units bound recomputation.

The 0.203 s `backpressure_seconds` at low sampled RSS occurred around pool admission
transitions. There were zero warning crossings, retries, downshifts, memory failures,
or near-hard-boundary events. It is consistent with a transient admission/host-reserve
sample rather than an 8 GiB RSS high-water event; no architecture change is justified.

The integration heartbeat's old H2H `workers=0/0/?` presentation was observational,
not zero execution. Independent Task 5A evidence sees 15 worker initializer loads,
15 per-worker rates, 527.85 worker CPU-seconds, one pool generation, 1.439 s shutdown,
and the 3.91 GiB process-tree peak. Configured/resolved/effective topology is
0(auto)/16/15 workers; created/live fields were unavailable in that heartbeat, so
they remain labelled unavailable rather than zero.

## Runtime budget gate and next decision

| Human comparison band | Central fits? | Planning upper fits? |
| ---: | :---: | :---: |
| 8 h | No | No |
| 12 h | No | No |
| 24 h | No | No |
| 48 h | No | No |

Dominant dimensions, in order, are:

1. RNG routing/hash/sort/merge record volume and Python record handling.
2. Simulation player-exposure volume across eight sequential `k` values.
3. H2H candidate-pair/power allocation and per-block publication.
4. Repeated source scans and metadata-heavy publication.
5. Authentication/final audit and synchronized-storage latency.

The next step requires a separate human decision among additional algorithmic/I/O
work, faster/larger hardware or separately approved local scratch architecture,
acceptance of a roughly 20-day planning budget, or an explicitly reviewed change to
official statistical scope. The benchmark does **not** recommend silently reducing
strategies, player counts, resolution, candidate contributions, power, or completion
targets. Given the 66.7% share, RNG implementation/I/O work is the first engineering
dimension to investigate if the current runtime is unacceptable.

## Changed files and retained evidence

- `scripts/benchmark_task5a_production_capacity.py`
- `tests/unit/scripts/test_benchmark_task5a_production_capacity.py`
- `docs/remediation/task5a_benchmark_plan.md`
- `docs/remediation/task5a_production_capacity_report.md`
- `docs/remediation/task5a_production_capacity.json`
- `docs/codex_context/context_prompt.md`
- `docs/codex_context/metadata.md`

Retained: `data/farkle-task5a-production-capacity-v1`, containing an ownership marker
and resumable machine-readable result. No superseded benchmark payload remains.
Cleanup was not required; the root is retained for review. Historical and successful
integration trees were not modified.

## Verification

- Focused Task 5A driver/dimension/model/topology/telemetry tests: **6 passed**.
- Repository-wide Ruff: **passed**.
- Mypy over `src`: **passed, 87 source files**.
- Black on changed Python files and final `--check`: **passed**.
- Pyright on changed benchmark and test files: **passed, 0 errors/warnings**.
- Task 5A output, retained checkpoint, and ownership-marker JSON validation:
  **3 valid files**.
- `git diff --check`: **passed**; Git emitted only the repository's normal
  LF-to-CRLF checkout warnings for two context Markdown files.
- Resumable no-force behavior is covered by the driver checkpoint path; force
  recomputation atomically replaces named owned checkpoints without recursively
  deleting the OneDrive directory.

No unrelated test or static-analysis failure was encountered. The full official
production pipeline was not run.
