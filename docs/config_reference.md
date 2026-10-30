# Configuration Reference

`farkle.config.AppConfig` is the only supported configuration contract. YAML
keys that are not current dataclass fields are rejected. Retired keys produce
an actionable replacement message and are never reinterpreted.

## Top-level sections

| Section | Purpose |
| --- | --- |
| `io` | Results-root prefix, analysis subdirectory, and optional scratch staging path |
| `sim` | Roots, player counts, strategy grid, simulation workers, and checkpoints |
| `profile` | Declared run purpose |
| `screening` | Wilson-width target, practical thresholds, bootstrap size, and candidate inputs |
| `batching` | Deterministic batch construction |
| `robustness` | Pareto, maximin, and two-root stability diagnostics |
| `k_aggregation` | Equal-k or explicitly declared player-count weights |
| `analysis` | Root-local diagnostics, workers, output names, and rare-event settings |
| `ingest` | Streaming parquet settings |
| `combine` | Maximum normalized schema width for the partitioned concat dataset |
| `trueskill` | Root/k TrueSkill screening parameters |
| `head2head` | Candidate cap, power, allocation, and inference settings |
| `hgb` | Held-out predictive-association settings |
| `resources` | Process-tree memory, CPU/native-thread, and byte-batch execution budgets |

## Roots and player counts

- `sim.seed_list` is required to identify the workflow roots.
- A standalone workflow requires one entry. `two-seed-pipeline` requires two.
- `sim.seed` is the active root of a root-local `AppConfig`; orchestration sets
  it while cloning each root context.
- `sim.n_players_list` is the complete configured player-count support.
- Missing root/k cells are errors. They cannot silently change an estimand.

## Implementation-owned protocols

`farkle.protocols` owns RNG scheme 2 / `PCG64DXSM`, artifact contract 3,
derived schema/estimand/conditioning 2, the 95% confidence level, H2H seat
scenarios `[0, 0.03, 0.06]`, and the balanced-tail cap policy. Pareto and maximin
reports are required; root execution is sequential. These facts are recorded in
authenticated metadata and cannot be selected through YAML or `--set`.

Public configuration schema 2 removes the former `rng`, `artifact_contract`,
and `orchestration` sections and the keys listed in the
[cleanup audit](config_option_audit.md). Even declarations matching the former
defaults are rejected with instructions to remove them. `analysis.results_glob`
is retired because input ownership follows explicit configured root/k paths.

Run-context contract 4 records `implementation_contract` in its canonical hash;
sidecars retain the actual artifact/RNG/version vector and configuration-schema
method identity. `active_config.yaml` contains user settings only. Compute and
stage hashes bind both the settings and the implemented protocol. This is an
explicit identity migration: use fresh output roots, preserve historical
contexts/sidecars, and never relabel old artifacts as current outputs.

## Statistical settings

- production `screening.resolution_delta = 0.03`, the maximum full 95% Wilson width
- `screening.practical_delta_by_k` must contain every configured k
- `screening.delta_across_k` must be positive
- `batching.target_batches = 100`
- `batching.min_shuffles_per_batch >= 30`
- `k_aggregation.method` is `equal-k` or `declared-mapping`
- authenticated release identity:
  artifact contract 3,
  `schema_version = 2`, `estimand_version = 2`, and
  `conditioning_version = 2`
- RNG scheme 2 and outcome schema 2 remain required by that identity

For `declared-mapping`, `k_aggregation.k_weights` must be positive, sum to
one, and cover the complete configured k support. Equal-k is the canonical
performance estimand; a declared alternative has a separately identified
operation and artifact.

## Screening and robustness

`screening` fields:

- `resolution_delta` (interval confidence is fixed at 95%)
- `practical_delta_by_k`, `delta_across_k`
- `bootstrap_replicates`
- `candidate_contribution_size` (default `75`)
- `controls`, `mandatory_diagnostics`
- `max_shuffles_per_root_k`, `projected_games_per_second`

`robustness` fields:

Pareto and maximin reports are always produced. Configurable fields:
- `delta_seed_stability`
- `joint_discrepancy_alpha` (upper-tail fraction for a descriptive joint
  reference quantile, not an inferential test level)
- `matched_count_fractions`

`analysis.rng_diagnostic_lags` declares the sorted unique positive lag set used
by RNG diagnostics. `analysis.rng_max_matchup_groups` is a deterministic cap on
eligible matchup groups; cap exhaustion publishes `blocked_by_cap`, never a
successful completion. `analysis.rng_diagnostic_partitions` (1 through 256)
sets the stable external partition count used for counting, eligibility,
ordering, and resumable lag reduction. All three fields participate in
RNG-diagnostic freshness.

RNG physical execution uses four execution-only `resources.stage_batch_bytes`
entries: `rng_diagnostics_source` bounds projected source batches,
`rng_diagnostics_route_unit` targets expanded route-unit bytes,
`rng_diagnostics_route_unit_hard` rejects an indivisible oversized source row
group, and `rng_diagnostics_reduce_run` bounds initial reducer runs. The exact
route and reducer phase names may also appear in
`resources.estimated_worker_memory_mb`; these values control admission only and
do not change RNG statistical freshness.

The workload planner chooses the smallest shuffle count meeting the Wilson
target, then rounds upward to the configured equal contiguous batch count. A cap that is too
small produces `blocked_by_cap` before simulation work begins.

## Run profiles

The [configuration option audit](config_option_audit.md) also identifies fixed
protocol descriptions and unused fields elsewhere in the public schema.

`profile.purpose` declares `smoke`, `integration`, or `production` intent. It
is the only profile setting. Purpose does not select batch counts or establish
statistical adequacy. Numerical workload settings determine the work.

Generated profile metadata (contract version 4) calculates `reduced_resolution`
from configured and planned Wilson widths relative to 0.03. It lists every
production qualification check, its actual value, requirement, and outcome.
The versioned `first_release_production_v1` policy retains the published
production design: resolution at most 0.03 with 95% intervals; exactly 100
batches with at least 30 shuffles each; at least 2,000 bootstrap replicates;
at least min(75, grid size) candidates contributed per ranking method; no
candidate-cap contraction of the potential family; H2H family alpha at most
0.02, target power at least 0.80, and planning effect at most 0.03. These are
release-policy requirements, not universal statistical sufficiency thresholds.
The finite grid, player-count selection, worker counts, and RNG partitions do
not independently establish or invalidate qualification. Stronger resolution,
power or bootstrap settings are accepted within this contract.

`production_eligible` requires both a qualifying configuration and declared
production intent. Development runs remain development evidence even when
their settings satisfy the design. A production request with unmet requirements
remains ineligible and reports the reasons. Immutable pre-audit profile metadata
uses `release_status: see_pipeline_health`, pointing to the final operational
result without claiming an audit is still pending. Final `pipeline_health.json`
profile metadata records the actual audit status and release eligibility.
Only the final pipeline audit publishes `release_eligible`, requiring a passed
fresh root/root/pair audit, completed roots, clean release code provenance,
qualifying production settings, standard game rules, authenticated correctness
coverage, and completed required monitoring. Sparse/constant-series diagnostic
limitations and descriptive investigation alerts remain explicit in
`diagnostic_limitations`; they do not become new statistical pass/fail rules.
Optional exhaustive diagnostics may remain omitted by design. Correctness
preflight is not statistical dependence evidence, and monitor clearance is
not an independence guarantee.

Remove the retired YAML keys `profile.reduced_resolution`,
`profile.production_eligible`, and `profile.release_eligible`; the loader rejects
these declarations with migration guidance. Labels remain outside statistical
compute identity, while numerical settings independently affect that identity.
The declared purpose and calculated qualification are authenticated in run
contexts and publishing stages. Historical artifacts are not relabeled.

`fast_config.yaml` is the existing machine-targeted integration family. It
preserves 80 strategies and k=2/4/5 over the configured root pair, with a 0.08 configured
resolution, 500 bootstrap replicates, eight RNG partitions, top-eight
contributions per ranking method, and a balanced-tail frozen-candidate cap of
12. The configured 20-batch granularity and locked 30-shuffle floor
resolve the unrounded 597-shuffle requirement to 600 shuffles per root/k and
achieve a maximum Wilson width of about 0.0797603. Production qualification requires
100 batches. This is development evidence only.

`fast_monitor_config.yaml` starts from that same design and uses exactly 30
batches, a supported batch count for the descriptive monitor. With 30 shuffles
per batch it plans 900 shuffles per root/k: 36,000 / 18,000 / 14,400 games for
k=2/4/5 per root (136,800 tournament games over two roots). It uses seeds 68/69
and the separate `results_monitor_30` output prefix. All other design and
resource controls match the fast profile. This addresses batch-count support;
zero denominators, constant series and sparse conditional outcomes can still
yield `insufficient_evidence`. It remains integration evidence, with the same
production qualification limitations apart from the changed batch count.

## H2H contract

- `family_alpha = 0.02`
- `target_power = 0.80`
- `practical_delta = 0.03`
- `sensitivity_deltas` includes `0.03` and `0.04`
- implemented seat scenarios: `[0, 0.03, 0.06]` (metadata, not a setting)
- `delta_equivalence = null` disables equivalence
- implemented candidate-cap policy: `balanced-tail` (metadata, not a setting)
- `total_game_cap` is operational and does not alter the schedule hash
- `allow_single_root` controls explicitly labeled single-root execution

The planner exhaustively finds the first admissible completed-game block size
whose exact implemented two-proportion score-test power reaches the Bonferroni
target over every locked seat scenario. Power is conditional on reaching that
completed support. Work is equal across roots and seat orders; single-root work
is equal across seat orders. `total_game_cap` authorization and progress live in
mutable execution state, so a cap-only raise never rewrites the immutable power
plan or block manifest.

Balanced-tail contraction lowers both method-specific cutoffs simultaneously,
uses stable source ranks (with strategy identifier tie-breaking for equal
win-rate scores), and preserves configured controls and mandatory diagnostics.
It may finish below the cap when a simultaneous tail removal drops multiple
unique candidates; it does not introduce a new inferential selection rule.

## Simulation, analysis, and model settings

`sim` owns `n_jobs`, process start method, checkpoint cadence, row/metric
locations, and the strategy option grid. `analysis` owns root-analysis workers,
optional RNG diagnostics, game-stat thresholds, rare-event settings, and the
three overridable output names (`curated_rows_name`, `metrics_name`, and
`manifest_name`).

Worker settings do not inherit across sections. `sim.n_jobs` controls only
simulation workers, `analysis.n_jobs` controls root-analysis workers, and
`head2head.n_jobs` controls H2H execution workers. A value of `0` is the explicit
auto mode and resolves to the detected logical CPU count; positive values are
explicit. `sim.n_jobs: null` retains its compatibility default of one worker.
YAML is loaded first and a matching `--set` value wins.
Roots execute sequentially and each root receives its section's resolved budget.
The authenticated run context records requested, resolved, and effective counts.
All four section-owned `n_jobs` values are execution-only: they are excluded
from the global compute-config hash and every statistical stage-config hash,
but remain authenticated in the run context.

The resource memory fields have distinct meanings:

- `scheduler_memory_budget_mb` is used only for worker admission.
- `process_tree_warning_threshold_mb` is the cooperative RSS high-water warning.
- `aggregate_memory_hard_limit_mb` is the explicit cooperative abort and
  Job Object/cgroup boundary.
- `minimum_system_available_memory_mb` is the host reserve required before
  launch and before new work is scheduled.
- `parent_process_memory_mb` is the parent's scheduling estimate inside the
  scheduler budget; it is not host-reserved memory.

The static ordering is `0 < parent_process_memory_mb <
scheduler_memory_budget_mb <= process_tree_warning_threshold_mb <
aggregate_memory_hard_limit_mb`. `logical_cpu_budget` (`0` means detected
logical CPUs) and `native_threads_per_worker` jointly cap processes; per-stage
worker estimates cap processes to `(scheduler_memory_budget_mb -
parent_process_memory_mb) / estimated_worker_memory_mb`. Concurrent roots share
both the logical-CPU and schedulable-memory envelopes deterministically. Native
BLAS/OpenMP/Arrow thread settings are capped in executor children, nested
process pools collapse to one process, and `stage_batch_bytes` bounds projected
Arrow batches. Resource controls and resolved policies are authenticated in
run-context contract v4 but are excluded from statistical freshness; focused
worker-count, ordering, and batch-boundary invariance tests protect that split.
RNG route units are planned greedily in canonical source-row-group order using
the expanded fixed-width observation size. The production defaults target 64
MiB, permit one row group through 96 MiB, and use 256 MiB reducer runs.
The `all_player_metrics` byte budget is applied to the k-dependent wide source
projection; `performance` bounds mapped-matrix column work, and
`performance_bootstrap` supplies the per-worker memory estimate for independent
replicate ranges.

The library/default YAML is a conservative portable profile with automatic CPU
detection. `fast_config.yaml` and `farkle_mega_config.yaml` are explicitly
machine-targeted for the Ryzen 7 3700X / 32 GiB host: 15 logical CPUs, an 8192
MiB scheduler and warning budget, a 12288 MiB aggregate hard limit, a 512 MiB
parent estimate, and one native thread per worker. The integration profile uses
a 4096 MiB minimum host reserve; the mega production profile retains 8192 MiB.
Those values are not universal defaults. Preflight
rejects an explicit CPU budget above detected logical CPUs or a hard-limit plus
host-reserve total above detected physical memory.

The CLI-only `--resource-limiters` mode is runtime provenance and is never
written to `active_config.yaml`. `enabled` applies the complete configured
CPU, native-thread, memory-admission, RSS-backpressure, and OS-boundary policy.
`memory-only` retains every memory control while ignoring
`logical_cpu_budget` and `native_threads_per_worker`. `disabled` establishes no
Job/cgroup boundary and performs no resource-derived worker reduction or memory
admission. Explicit stage `n_jobs`, nested-process suppression, bounded queues,
and byte-bounded streaming remain active in every mode.

`io.temp_dir` is either `null` or an absolute path. `null` stages each file
beside its canonical destination. A configured path is created lazily and is
used for growing artifact, sidecar, manifest, JSON, CSV, Parquet, partition,
and analysis-workspace files. It is public path provenance but is excluded from
all statistical hashes. Prefer a non-synchronized directory on the same
filesystem as the results tree for inexpensive atomic renames.

The two-root preflight log/manifest event also reports a declared upper-envelope
projection for tournament row shards, H2H pair/root/order blocks, their adjacent
sidecars, and their total. This is operational file-system capacity information,
not a statistical sample-size calculation. Fixed-count workflow artifacts,
manifests, logs, and completion stamps are explicitly outside that projected
high-cardinality total. No file-count threshold is currently configured, so the
projection warns or blocks nothing.

The reviewed fast run produced 75,609 files, so bounded, manifest-indexed shard
aggregation remains justified future operational work. It is intentionally not
part of this change; per-coordinate atomic H2H recovery remains unchanged.

`trueskill` contains `beta`, `tau`, and `draw_probability`. Canonical ratings
are always root/k cells. `hgb` contains `max_depth`, `n_estimators`,
`heldout_folds`, `permutation_repeats`, and `future_proposal_limit`.

## Canonical paths

Do not assemble analysis paths manually. Use:

- `cfg.stage_dir(stage)` and `cfg.stage_subdir(stage, ...)`
- `cfg.by_k_dir(stage, k)`
- `cfg.concat_ks_dir(stage)`
- `cfg.across_k_dir(stage)`
- `cfg.cross_seed_dir(stage)`
- `cfg.diagnostics_dir(stage)`
- `cfg.h2h_2p_dir(stage)`
- the artifact-specific helpers on `AppConfig`

These helpers resolve paths without creating directories. Artifact writers,
checkpoint publishers, and completion-stamp writers create parents only when
the active stage publishes work. `h2h_2p_dir` always requires its owning stage.

Two-seed orchestration keeps the individual roots under `results_seed_X` and
`results_seed_Y`; pair-owned outputs live under the sibling
`seed_pair_analysis` directory.

The only scopes are `by_k`, `concat_ks`, `across_k`, `cross_seed`,
`diagnostics`, and `h2h_2p`. Scope-mismatched paths fail validation.

## Example

```yaml
sim:
  seed_list: [42, 43]
  n_players_list: [2, 4]

screening:
  practical_delta_by_k: {2: 0.03, 4: 0.03}
  delta_across_k: 0.03

k_aggregation:
  method: equal-k
```

The implementation-owned v3 release path rejects contract-v2 artifacts,
sidecars, completion stamps, and mixed v2/v3 descendant graphs. Release
workflows must use a new output prefix; old bytes cannot be promoted by adding
or replacing sidecars.

```powershell
farkle --config configs/fast_config.yaml --set sim.n_jobs=8 run
farkle --config configs/fast_config.yaml two-seed-pipeline --seed-pair 42 43
```
