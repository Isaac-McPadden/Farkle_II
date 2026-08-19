# Configuration Reference

`farkle.config.AppConfig` is the only supported configuration contract. YAML
keys that are not current dataclass fields are rejected. Retired keys produce
an actionable replacement message and are never reinterpreted.

## Top-level sections

| Section | Purpose |
| --- | --- |
| `io` | Results-root prefix and analysis subdirectory |
| `sim` | Roots, player counts, strategy grid, simulation workers, and checkpoints |
| `rng` | RNG scheme and bit generator |
| `profile` | Run purpose and production/release claim eligibility |
| `screening` | Wilson-width target, practical thresholds, bootstrap size, and candidate inputs |
| `batching` | Deterministic batch construction |
| `robustness` | Pareto, maximin, and two-root stability diagnostics |
| `k_aggregation` | Equal-k or explicitly declared player-count weights |
| `artifact_contract` | Versions used by sidecars and freshness keys |
| `analysis` | Root-local diagnostics, workers, output names, and rare-event settings |
| `ingest` | Streaming parquet settings |
| `combine` | Maximum normalized schema width for the partitioned concat dataset |
| `trueskill` | Root/k TrueSkill screening parameters |
| `head2head` | Candidate cap, power, allocation, and inference settings |
| `hgb` | Held-out predictive-association settings |
| `orchestration` | Root execution concurrency |
| `resources` | Process-tree memory, CPU/native-thread, and byte-batch execution budgets |

## Roots and player counts

- `sim.seed_list` is required to identify the workflow roots.
- A standalone workflow requires one entry. `two-seed-pipeline` requires two.
- `sim.seed` is the active root of a root-local `AppConfig`; orchestration sets
  it while cloning each root context.
- `sim.n_players_list` is the complete configured player-count support.
- Missing root/k cells are errors. They cannot silently change an estimand.

## Locked statistical settings

- `rng.scheme_version = 2`
- `rng.bit_generator = PCG64DXSM`
- production `screening.resolution_delta = 0.03`, the maximum full 95% Wilson width
- `screening.practical_delta_by_k` must contain every configured k
- `screening.delta_across_k` must be positive
- `batching.target_batches = 100`
- `batching.min_shuffles_per_batch >= 30`
- `k_aggregation.method` is `equal-k` or `declared-mapping`
- authenticated release identity:
  `artifact_contract.artifact_contract_version = 3`,
  `schema_version = 2`, `estimand_version = 2`, and
  `conditioning_version = 2`
- `rng.scheme_version = 2` and outcome schema 2 are required by that identity

For `declared-mapping`, `k_aggregation.k_weights` must be positive, sum to
one, and cover the complete configured k support. Equal-k is the canonical
performance estimand; a declared alternative has a separately identified
operation and artifact.

## Screening and robustness

`screening` fields:

- `resolution_delta`, `interval_confidence`
- `practical_delta_by_k`, `delta_across_k`
- `bootstrap_replicates`
- `candidate_contribution_size` (default `75`)
- `controls`, `mandatory_diagnostics`
- `max_shuffles_per_root_k`, `projected_games_per_second`

`robustness` fields:

- `report_pareto`, `report_maximin`
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

The workload planner chooses the smallest shuffle count meeting the Wilson
target, then rounds upward to 100 equal contiguous batches. A cap that is too
small produces `blocked_by_cap` before simulation work begins.

## Run profiles

`profile.purpose` is `smoke`, `integration`, or `production`. Production
profiles are full-resolution, production-eligible, and release-eligible. Smoke
and integration profiles must be explicitly reduced-resolution,
non-production, and non-release. These claim labels are authenticated in the
run context and in stages that publish them, but remain outside statistical
compute identity; the numerical workload settings independently change that
identity.

`fast_config.yaml` is the existing machine-targeted integration family. It
preserves 80 strategies, roots 48/49, and k=2/4/5, with a 0.08 configured
resolution, 500 bootstrap replicates, eight RNG partitions, top-eight
contributions per ranking method, and a balanced-tail frozen-candidate cap of
12. The locked batching floor resolves 0.08 to 3,000 shuffles (100 batches of
30) per root/k, not approximately 600, and achieves a maximum Wilson width of
about 0.035761. This is development evidence only.

## H2H contract

- `family_alpha = 0.02`
- `target_power = 0.80`
- `practical_delta = 0.03`
- `sensitivity_deltas` includes `0.03` and `0.04`
- `seat1_advantage_scenarios = [0, 0.03, 0.06]`
- `delta_equivalence = null` disables equivalence
- `candidate_cap_policy = balanced-tail`
- `total_game_cap` is operational and does not alter the schedule hash
- `allow_single_root` controls explicitly labelled single-root execution

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
When `orchestration.parallel_seeds` is false (the default), roots remain
sequential and each root receives the resolved section budget. When it is true,
each section's own resolved budget is divided across concurrent roots. The
authenticated run context records requested, resolved, and effective counts.
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
run-context contract v2 but are excluded from statistical freshness; focused
worker-count, ordering, and batch-boundary invariance tests protect that split.
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

artifact_contract:
  artifact_contract_version: 3
  schema_version: 2
  estimand_version: 2
  conditioning_version: 2
```

The public v3 switch intentionally invalidates all contract-v2 artifacts,
sidecars, completion stamps, and mixed v2/v3 descendant graphs. Release
workflows must use a new output prefix; old bytes cannot be promoted by adding
or replacing sidecars.

```powershell
farkle --config configs/fast_config.yaml --set sim.n_jobs=8 run
farkle --config configs/fast_config.yaml two-seed-pipeline --seed-pair 42 43
```
