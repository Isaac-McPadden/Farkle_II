# Task 14 end-to-end oracle blueprint

Status: design only; no production or test implementation is part of this task

## 1. Purpose and acceptance boundary

Task 14 must add the missing evidence that a raw simulation can traverse the
canonical two-root workflow and arrive at an authenticated report without
manufactured tournament metrics, manufactured TrueSkill ratings, or a
substituted H2H block runner.

The existing `tests/integration/test_structure_toy_oracle.py` remains useful for
larger structural, interruption, worker-count, cycle, and reporting cases. It
is not this oracle: it writes root metrics and ratings directly and supplies a
fake H2H block runner. The new test must instead execute real raw games, ingest
their row shards, run every enabled root stage, freeze the family from those
root results, execute the production H2H scheduler and game reducer, and run
inference through reporting.

The new oracle is deliberately a test design, not a release-valid scientific
configuration. In particular, its two deterministic batches of one shuffle
each are smaller than the production release contract of 100 batches with at
least 30 shuffles per batch. The test must label these as test-only workload
parameters and must not weaken `AppConfig.validate_statistical_contract`.

## 2. Exact entry point

The final integration test must invoke this orchestration entry point exactly:

```python
from farkle.config import assign_config_sha, load_app_config
from farkle.orchestration.two_seed_pipeline import run_pipeline

cfg = load_app_config(config_path, seed_list_len=2)
assign_config_sha(cfg)
run_pipeline(
    cfg,
    seed_pair=(11, 22),
    oracle_game_profile=oracle_game_profile,
)
```

`oracle_game_profile` is the required narrow test seam specified in Section 5.
The production default must remain absent/`None`.

This choice exercises the same canonical orchestrator called by
`farkle --config CONFIG two-seed-pipeline --seed-pair 11 22`, including root
context construction, simulation, root analysis, pair context construction,
pair analysis, manifests, and `pipeline_health.json`. It avoids adding a
test-only option to the public CLI grammar. CLI parsing and installed-entrypoint
dispatch are already independently covered by
`tests/integration/test_installed_cli.py`; that test should remain the parser
oracle.

The final test must not call root stages individually and must not call
`execute_h2h_schedule` with `block_runner=...`.

## 3. Tiny configuration

Write the YAML under the test's `tmp_path`; do not add a runnable configuration
under `configs/`. Paths shown as `RESULTS_PREFIX` are replaced with an absolute
temporary path.

```yaml
io:
  results_dir_prefix: RESULTS_PREFIX
  analysis_subdir: analysis

sim:
  n_players_list: [2, 4]
  seed: 11
  seed_list: [11, 22]
  n_jobs: 1
  expanded_metrics: true
  row_dir: rows
  metric_chunk_dir: metric_chunks
  desired_sec_per_chunk: 1
  ckpt_every_sec: 1
  score_thresholds: [500]
  dice_thresholds: [2]
  smart_five_opts: [false]
  smart_one_opts: [false]
  consider_score_opts: [true]
  consider_dice_opts: [true]
  auto_hot_dice_opts: [false, true]
  run_up_score_opts: [false]
  include_stop_at: false
  include_stop_at_heuristic: false

analysis:
  disable_rng_diagnostics: true
  n_jobs: 1
  log_level: INFO
  rare_event_target_score: 100
  game_stats_margin_thresholds: [500]

ingest:
  row_group_size: 64
  batch_rows: 64
  n_jobs: 1

combine:
  max_players: 4

trueskill:
  beta: 1.0
  tau: 0.0
  draw_probability: 0.0

head2head:
  n_jobs: 1
  family_alpha: 0.5
  target_power: 0.1
  practical_delta: 0.2
  sensitivity_deltas: [0.2, 0.04]
  seat1_advantage_scenarios: [0.0, 0.03, 0.06]
  delta_equivalence: null
  candidate_cap: 3
  candidate_cap_policy: balanced-tail
  min_candidate_completion_rate: 0.99
  max_attempt_multiplier: 2.0
  total_game_cap: 24
  allow_single_root: true

screening:
  resolution_delta: 0.9
  interval_confidence: 0.95
  practical_delta_by_k: {2: 0.2, 4: 0.2}
  delta_across_k: 0.2
  bootstrap_replicates: 1
  candidate_contribution_size: 1
  controls: [0, 1, 3]
  mandatory_diagnostics: []

batching:
  target_batches: 2
  min_shuffles_per_batch: 1

robustness:
  report_pareto: true
  report_maximin: true
  delta_seed_stability: 0.2
  joint_discrepancy_alpha: 0.05
  matched_count_fractions: [1.0]

artifact_contract:
  artifact_contract_version: 2
  estimand_version: 1
  schema_version: 1

k_aggregation:
  method: equal-k
  k_weights: null

hgb:
  max_depth: 1
  n_estimators: 1
  heldout_folds: 2
  permutation_repeats: 1
  future_proposal_limit: 1

orchestration:
  parallel_seeds: false
```

This grid has exactly four canonical strategies and is divisible by both
configured k values:

| ID | `require_both` | `auto_hot_dice` | all other active choices |
| ---: | --- | --- | --- |
| 0 | true | false | threshold 500, dice 2, score+dice, no smart flags |
| 1 | false | false | same |
| 2 | true | true | same |
| 3 | false | true | same |

The workload planner must publish, for every root/k cell:

```text
strategy_count = 4
target_batches = 2
shuffles_per_batch = 1
required_shuffles = 2
games_per_shuffle(k=2) = 2
games_per_shuffle(k=4) = 1
```

Thus each root produces four 2-player attempts and two 4-player attempts.

### Minimum model settings

The minimum HGB settings actually admitted by production code are two held-out
folds and one permutation repeat. Scikit-learn also admits one boosting
iteration and depth one, so the oracle uses:

```text
heldout_folds = 2
permutation_repeats = 1
n_estimators/max_iter = 1
max_depth = 1
```

Four strategy configurations give each k enough support for the two-fold
split. `future_proposal_limit=1` keeps the future-only output exercised without
expanding current support.

TrueSkill has no fold or iteration setting: it makes one sequential pass over
the canonical row order. Production currently performs only float coercion and
does not define a validated numeric minimum; the dependency even accepts
`beta=0`. The minimum meaningful boundary values for dynamics and draws are
`tau=0.0` and `draw_probability=0.0`. The oracle uses the smallest deliberately
nondegenerate beta in this design, `beta=1.0`; zero beta must not be used as a
runtime-minimization trick.

## 4. Existing helpers and fixtures to reuse

Reuse:

- `tmp_path`, the autouse frozen-clock fixture, and `update_goldens` from
  `tests/conftest.py`;
- `make_test_app_config` from `tests/helpers/config_factory.py` for focused
  Task 14B seam tests, while the final orchestration test loads the YAML above;
- `assert_parquet_golden` and `assert_stamp_has_paths` from
  `tests/helpers/golden_utils.py`;
- the `_logical_table` and one-sidecar assertion pattern from
  `tests/integration/test_structure_toy_oracle.py`, moved to a shared helper
  rather than copied;
- `audit_sidecar_completeness` from `farkle.analysis.release_audit`;
- `load_run_context`, `resolve_stage_state`, `read_stage_done`,
  `validate_artifact_sidecar`, `sidecar_path`, and `sha256_file` for
  authenticated assertions;
- `raw_simulation_schema_for(k)` and `all_player_batch_schema()` for exact
  schema comparison rather than handwritten dtype lists.

Do not reuse:

- `sim_artifacts` from `tests/conftest.py`;
- `build_curated_fixture` from `tests/helpers/diagnostic_fixtures.py`;
- `write_parquet_test_artifact`/`sidecar_metadata` from
  `tests/helpers/artifact_sidecars.py`;
- `_write_root_cells`, `_write_trueskill_contribution`,
  `_toy_block_runner`, or `_noncompletion_oracle_runner` from the structural
  toy oracle.

Those helpers manufacture a boundary that this test exists to exercise.

## 5. Required test seam

### Current gap

No current orchestration-level setting carries the already-supported
`_play_game(target_score=..., max_rounds=...)` arguments into tournament worker
processes or production H2H blocks. Monkeypatching `_play_game` in the parent is
not reliable under spawned workers, and passing a fake H2H `block_runner` would
repeat the defect in the existing structural oracle.

Task 14B therefore requires one seam before the final oracle can exist.

### Permitted seam

Add an immutable, picklable game-profile descriptor accepted only as an
optional keyword argument by `two_seed_pipeline.run_pipeline`. It must:

1. contain default `target_score` and `max_rounds` values;
2. contain coordinate-specific `max_rounds` overrides for tournament
   `(root_seed, k, shuffle_index, game_index)` and H2H
   `(root_seed, pair_id, order, attempt_index)` coordinates;
3. resolve only arguments already accepted by `_play_game`;
4. be passed into spawned tournament workers and the production
   `h2h_schedule._simulate_block`;
5. leave `_play_game`, `FarkleGame.play`, `_winner_seat_counts`,
   `_block_progress`, and all reducers untouched;
6. have a canonical SHA-256 identity included in run lineage, simulation
   freshness, the H2H schedule hash/block manifest, and relevant sidecar method
   parameters; and
7. default to the current production rules when omitted.

It must not accept precomputed winners, ranks, rows, counts, metrics, or H2H
block results. It must not expose a general callback that can return an
upstream artifact. The final integration test must assert that the production
`_simulate_block` is used.

The oracle profile is:

```text
default target_score = 100
default max_rounds = 200

tournament override:
  (root=11, k=2, shuffle=0, game=0) -> max_rounds=0

H2H overrides:
  (root=11, pair=0, order=0, attempt=0) -> max_rounds=0
  (root=11, pair=1, order=0, attempt=0) -> max_rounds=0
  (root=11, pair=1, order=0, attempt=1) -> max_rounds=0
```

`max_rounds=0` still invokes the real engine. It deterministically returns a
schema-v2 safety-limit attempt with zero rounds, zero turns, no winner, and
null ranks. Every other coordinate plays real seeded dice to completion under
the small target.

## 6. Hand oracle for raw tournament rows

With RNG-v2, roots 11/22, the four strategies above, and the game profile in
Section 5, the raw rows must be:

| root | k | shuffle | game | seated strategy IDs | status | winner strategy | rounds | total player turns | final scores in seat order |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 11 | 2 | 0 | 0 | `[0,2]` | safety_limit | null | 0 | 0 | `[0,0]` |
| 11 | 2 | 0 | 1 | `[1,3]` | completed | 1 | 2 | 5 | `[1950,1100]` |
| 11 | 2 | 1 | 0 | `[2,1]` | completed | 2 | 2 | 4 | `[500,0]` |
| 11 | 2 | 1 | 1 | `[0,3]` | completed | 0 | 1 | 2 | `[600,0]` |
| 11 | 4 | 0 | 0 | `[0,1,2,3]` | completed | 2 | 1 | 4 | `[700,0,800,0]` |
| 11 | 4 | 1 | 0 | `[3,2,1,0]` | completed | 3 | 1 | 5 | `[3050,2900,0,0]` |
| 22 | 2 | 0 | 0 | `[3,0]` | completed | 3 | 1 | 2 | `[600,0]` |
| 22 | 2 | 0 | 1 | `[1,2]` | completed | 2 | 1 | 2 | `[500,1100]` |
| 22 | 2 | 1 | 0 | `[2,0]` | completed | 2 | 1 | 2 | `[950,0]` |
| 22 | 2 | 1 | 1 | `[3,1]` | completed | 3 | 1 | 3 | `[750,550]` |
| 22 | 4 | 0 | 0 | `[1,2,0,3]` | completed | 2 | 1 | 5 | `[0,700,0,0]` |
| 22 | 4 | 1 | 0 | `[0,2,1,3]` | completed | 1 | 1 | 4 | `[700,0,1100,0]` |

The exact cross-cell totals are 18,550 final-score points and 38 player turns.
These are useful discriminators for the exact turn-denominator metrics; they
must not replace assertions on the raw rows themselves.

### Root/k counts

| root | k | attempted A | completed C | safety S | exposures E | completed exposures | safety exposures | wins W | losses L |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 2 | 4 | 3 | 1 | 8 | 6 | 2 | 3 | 5 |
| 11 | 4 | 2 | 2 | 0 | 8 | 8 | 0 | 2 | 6 |
| 22 | 2 | 4 | 4 | 0 | 8 | 8 | 0 | 4 | 4 |
| 22 | 4 | 2 | 2 | 0 | 8 | 8 | 0 | 2 | 6 |
| total | — | 12 | 11 | 1 | 32 | 30 | 2 | 11 | 21 |

Each batch is one shuffle and exposes every strategy exactly once:

| root | k | batch | attempted | completed | safety | winning strategies |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 11 | 2 | 0 | 2 | 1 | 1 | `[1]` |
| 11 | 2 | 1 | 2 | 2 | 0 | `[2,0]` |
| 11 | 4 | 0 | 1 | 1 | 0 | `[2]` |
| 11 | 4 | 1 | 1 | 1 | 0 | `[3]` |
| 22 | 2 | 0 | 2 | 2 | 0 | `[3,2]` |
| 22 | 2 | 1 | 2 | 2 | 0 | `[2,3]` |
| 22 | 4 | 0 | 1 | 1 | 0 | `[2]` |
| 22 | 4 | 1 | 1 | 1 | 0 | `[1]` |

### Cross-root/cross-k strategy counts

Every strategy has eight attempted exposures:

| strategy | attempted | completed | safety | wins | losses |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 7 | 1 | 1 | 7 |
| 1 | 8 | 8 | 0 | 2 | 6 |
| 2 | 8 | 7 | 1 | 5 | 3 |
| 3 | 8 | 8 | 0 | 3 | 5 |

The raw rows, checkpoint counters, per-batch all-player metrics, per-k
performance, root-combined performance, and report support must all conserve:

```text
A = C + S
E = k*A
E_C = k*C
E_S = k*S
E = E_C + E_S
W = C
L = E - W = (k - 1)*C + k*S
```

The same identities must be asserted by root, k, deterministic batch, and
strategy. For the safety row, `winner_seat`, `winner_strategy`,
`winning_score`, `victory_margin`, all `P*_rank`, and every entry in
`seat_ranks` are null; no downstream winner-conditioned table may count it.

## 7. Frozen family and actual H2H oracle

Configured controls 0, 1, and 3 are protected before H2H. Strategy 2 may rank
well in the tournament, but the balanced-tail cap contracts unprotected method
tails until the final family is exactly:

```text
candidates = [0, 1, 3]
unordered_pair_count = 3
root_count = 2
seat orders = 2
total_block_count = 12
```

The candidate manifest and membership bytes, their sidecars, and `family_hash`
must be captured before `h2h_execute` and remain unchanged afterward.

For the tiny planning settings, production exact-power code must return:

```text
n_completed_required_per_root_order_block = 1
max_attempts_per_root_order_block = 2
total_completed_required = 12
maximum_total_attempts = 24
planning_state = complete_valid
execution_authorization = ready
```

The real H2H block reducer must produce:

| pair | A,B | root | order | attempted | completed | safety | wins A | wins B | replacements | block status |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0,1 | 11 | 0 | 2 | 1 | 1 | 1 | 0 | 1 | complete |
| 0 | 0,1 | 11 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | complete |
| 0 | 0,1 | 22 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | complete |
| 0 | 0,1 | 22 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | complete |
| 1 | 0,3 | 11 | 0 | 2 | 0 | 2 | 0 | 0 | 1 | unresolved_nonviable |
| 1 | 0,3 | 11 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | complete |
| 1 | 0,3 | 22 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | complete |
| 1 | 0,3 | 22 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | complete |
| 2 | 1,3 | 11 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | complete |
| 2 | 1,3 | 11 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | complete |
| 2 | 1,3 | 22 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | complete |
| 2 | 1,3 | 22 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | complete |

This simultaneously proves a deterministic replacement (pair 0/root
11/order 0 advances from attempt 0 to attempt 1) and cap exhaustion (pair
1/root 11/order 0 authenticates the contiguous prefix `[0,2)` and stops).

H2H totals must be:

```text
games_attempted = 14
games_completed = 11
games_safety_limit = 3
replacement_attempt_count = 2
wins_a + wins_b = 11
resolved blocks = 11
unresolved blocks = 1
```

The execution lifecycle is still `complete_valid`: every scheduled block
reached a terminal execution state. Its separate substantive status is
`unresolved_nonviable`; it is not `blocked_by_cap`, which is reserved for the
global authorization cap.

### Operational viability after freeze

Incident-attempt support is:

| candidate | attempted | completed | safety | replacements | completion rate | operationally viable at 0.99 | inferentially viable |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 10 | 7 | 3 | 2 | 0.7 | false | false |
| 1 | 9 | 8 | 1 | 1 | 8/9 | false | true |
| 3 | 9 | 7 | 2 | 1 | 7/9 | false | false |

Thus strategy 3 (and, deliberately, the other incident candidates) remains a
member of the immutable frozen family but becomes operationally nonviable only
after actual H2H attempts are observed. No candidate is removed and the
multiplicity family remains all three unordered pairs.

Pair 0 has `q_AB=1`, `q_BA=1`, and `d_AB=0`. Pair 2 has `q_AB=1/2`,
`q_BA=1`, and `d_AB=-1/4`. Pair 1 has no formal test and null effect/interval
fields. Because every pair contains at least one operationally nonviable
candidate, all three final `decision_class` values are
`unresolved_nonviable`, all `holm_reject` values are false, and all pairs
remain `multiplicity_family_member=true`.

## 8. Required workflow stages and artifacts

With RNG diagnostics disabled, each root must have exactly these stage
directories:

```text
00_ingest
01_curate
02_combine
03_metrics
04_game_stats
05_trueskill
06_hgb
07_screening
```

The pair analysis root must have exactly:

```text
00_root_stability
01_trueskill
02_candidate_freeze
03_h2h_power
04_h2h_execute
05_h2h_inference
06_h2h_digest
07_agreement
08_reporting
```

At minimum, assert the following artifact families. Resolve every path through
`AppConfig`; folder numbers must never be hard-coded in test logic.

### Per root

- simulation: the strategy manifest; per-k workload plan, pickle checkpoint,
  checkpoint parquet, expanded metrics, row manifest/shards, metric-chunk
  manifest/shards, and authenticated simulation completion stamp;
- ingest: by-k raw ingested parquet and ingest manifest;
- curate: by-k `game_rows.parquet` and curate manifest;
- combine: both by-k partition files/manifests, the partitioned dataset, the
  `concat_ks` parquet, and combined manifest;
- metrics: both all-player batch tables, both per-k performance tables,
  across-k performance/bootstrap/control tables, player-count diagnostics,
  all per-k seat count/effect/population tables, and all four cross-k/diagnostic
  seat tables;
- game stats: both per-k summaries, concat and equal-k game-length/margin
  tables, rare-event summary and its authenticated shards, and both exact-roll
  artifacts;
- TrueSkill: both rating cells and their JSON/checkpoint/done companions,
  across-k candidate contribution, and screening diagnostics;
- HGB: both sets of importance/prediction/fold tables, concatenated importance,
  across-k importance JSON/table, and future-only proposals;
- screening: descriptive screening parquet and JSON.

### Pair root

- all root-stability paths returned by `RootStabilityArtifacts`, including both
  per-k combined-performance tables;
- pair TrueSkill candidate contribution;
- candidate membership and immutable family manifest;
- immutable H2H power plan and block manifest;
- execution state, 12 block parquets, and root-order union;
- four H2H inference artifacts;
- four dominance/digest artifacts;
- agreement pair table and summary;
- structure report JSON, Markdown, PNG, and migration report.

Every derived analysis artifact above must have exactly one adjacent compatible
sidecar. `audit_sidecar_completeness(root_analysis_dir)` and
`audit_sidecar_completeness(pair_analysis_dir)` must both return `[]`. Raw
simulation source files are authenticated by the simulation completion stamp;
do not incorrectly require analysis sidecars on source checkpoints.

## 9. Lifecycle, hash, schema, scope, and source assertions

### Run contexts and health

- Authenticate both root `run_context.json` files and the pair
  `run_context.json` with `load_run_context(..., active_config_path=...)`.
- Assert run-context contract version 1, each canonical
  `run_context_sha256`, root lists, resolved stage layouts, and absolute
  resolved paths.
- Assert the pair context contains the two root lifecycle identities in root
  order and that its `run_lineage_sha256` differs from both parents.
- Assert the oracle game-profile hash is present in the authenticated lineage
  extension introduced by Task 14B and is identical in both root contexts and
  the pair context.
- Assert `pipeline_health.json` is `complete_success`, both root workflows are
  complete, the pair workflow is complete, and every listed stage state is
  `complete_valid`.

### Completion stamps

For every simulation, root-stage, pair-stage, and independently resumable
TrueSkill/H2H cell stamp:

- assert completion schema version 4 and lifecycle contract version 1 where
  the general stage-completion format applies;
- assert `status=success` and `completion_state=complete_valid`;
- assert stage key, cache-key version, public/stage config hashes, freshness
  SHA-256, code identity, run-lineage SHA-256, and stage-identity SHA-256;
- recalculate every input/output byte or directory-tree identity;
- assert sidecar SHA identities are present for sidecar-owned outputs; and
- call `resolve_stage_state` with the owning config and require
  `CompletionState.COMPLETE_VALID`.

The one H2H block that exhausts its attempt cap does not make the execution
stage stamp partial. Assert `execution_state=complete_valid`,
`substantive_status=unresolved_nonviable`, and `unresolved_block_count=1`.

### Schemas and semantic versions

- raw and curated by-k schemas must equal `raw_simulation_schema_for(k)`;
- combined rows must equal `expected_schema_for(12)` (the current rectangular
  combine schema) while conserving all 12 row identities;
- all-player metrics must equal `all_player_batch_schema()`;
- strategy ID columns must retain canonical integer Arrow types;
- outcome schema version is exactly 2 and tournament method version is exactly
  2 in rows, manifests, checkpoints, freshness, and simulation stamps;
- RNG scheme is exactly 2/PCG64DXSM;
- H2H method version is 2, TrueSkill method version is 3, HGB method version is
  2, and structure-report contract version is 4;
- every sidecar has artifact-contract version 2, estimand version 1, schema
  version 1, and an exact `consistency_columns` match to its artifact.

### Sidecar identity, scope, and lineage

For every derived artifact:

- recalculate `artifact_sha256` and `artifact_size_bytes`;
- require `config_hash` to match the owning root or pair public config;
- validate `method_contract.procedure == operation`;
- require `required_player_counts=[2,4]` for complete-support root/pair products,
  `[k]` for by-k products, and `[2]` for H2H products;
- require `seed_scope=single_root` for root artifacts and
  `both_roots_combined` for pair products where applicable;
- require the canonical scope dictated by its path: `by_k`, `concat_ks`,
  `across_k`, `cross_seed`, `diagnostics`, or `h2h_2p`; and
- require every `source_artifacts` path to be an actual current-workflow
  artifact and to resolve through the authenticated graph: it is either a
  declared stage input or a same-stage output whose bytes and sidecar are
  themselves bound by that stage's completion stamp. No retired path or
  alternate stage folder is allowed.

In addition, assert:

- candidate `source_identity.win_rate.sha256` equals the current
  root-combined performance bytes, while the TrueSkill source identity equals
  both the current pair contribution bytes and its sidecar bytes;
- one `family_hash` is repeated in membership rows, family manifest, power
  plan, block manifest, all block results, execution state, inference,
  dominance, agreement, and report;
- one `schedule_hash` is repeated in the plan, block manifest, all block
  results, execution state, and H2H method contracts;
- each block's `attempt_coordinate_range_hash` recalculates from the exact
  authenticated contiguous attempt prefix;
- power-plan and block-manifest bytes/hashes are unchanged by execution,
  inference, and a no-force rerun; and
- HGB and TrueSkill cell sidecars bind their production-resolved code revision,
  while the general lifecycle code identity supplies the authoritative code
  binding for artifacts whose sidecar currently records the default revision.

### Idempotent second invocation

Invoke `run_pipeline` a second time without `force`. Snapshot all derived
artifact and completion-stamp hashes before the second invocation and require
them to remain identical. Append-only orchestration manifests and the rewritten
health view are excluded from byte-equality, but their final events must again
report success. No new H2H attempt coordinate may be executed.

## 10. Report oracle

The JSON report must state:

```text
report_contract_version = 4
execution_scope = root_pair
roots = [11, 22]
support.player_counts = [2, 4]
support.k_weights = {"2": 0.5, "4": 0.5}
safety_limits.games_attempted = 12
safety_limits.games_completed = 11
safety_limits.games_safety_limit = 1
candidate_family.candidate_count = 3
h2h.games_attempted = 14
h2h.games_completed = 11
h2h.games_safety_limit = 3
h2h.replacement_attempt_count = 2
h2h.unresolved_pair_count = 3
h2h.unresolved_nonviable_pair_count = 3
h2h.operationally_nonviable_candidates = ["0", "1", "3"]
h2h.unique_best = null
h2h.unique_best_claim_permitted = false
```

Because configured support includes k=4, H2H's role must be
`external_two_player_finalist_diagnostic`. Candidate viability rows and
by-pair/root/order rows must reproduce Sections 7 and 8 exactly.

The claim language and Markdown must contain:

```text
3 finalist comparisons remain unresolved.
Operationally nonviable frozen finalists (retained with no affected dominance/equivalence claims): ['0', '1', '3'].
No unique-best claim is permitted by the direct-dominance rule.
```

It must not contain a unique-best assertion, an equivalence assertion, a root
significance/rejection assertion, or any claim that safety-limit attempts were
draws. The migration report must say that no artifact was deleted.

## 11. Runtime target

The routine-CI target is:

```text
p50 developer machine: <= 30 seconds
p95 ordinary CI worker: <= 60 seconds
cold Windows/spawn allowance: <= 90 seconds
```

The test remains a routine integration test, not a nightly-only test. It runs
12 tournament games, at most 24 authorized H2H attempts (14 actually used),
eight one-iteration held-out HGB fits, and four tiny TrueSkill cells. If it
exceeds the cold allowance, profile stage startup and plot/model import costs;
do not reduce coverage by manufacturing artifacts or replacing H2H execution.

## 12. Division of work and assertions for Tasks 14B-14E

### Task 14B — authenticated game-profile seam and fixture

- implement the immutable/picklable limit-only profile;
- propagate it through spawned tournament workers and production
  `_simulate_block`;
- bind its hash to run lineage, simulation freshness, H2H schedule identity,
  and method contracts;
- add focused unit tests proving default behavior is unchanged, coordinate
  matching is exact, spawn serialization works, and the profile cannot return
  outcomes or artifacts;
- add the YAML/profile builder and expected row/H2H coordinate constants.

### Task 14C — raw simulation and root oracle helpers

- add a helper that validates the 12 raw rows, schemas, nullability, exact
  root/k and strategy counts, turns/scores, workload plans, and conservation
  identities;
- add a helper that validates root stage layouts, root completion states, root
  contexts, and root artifact/sidecar completeness;
- add focused tests for those helpers against the canonical orchestrator output
  fixture.

### Task 14D — family, H2H, and pair oracle helpers

- validate immutable family membership/provenance and exact planning counts;
- validate all 12 real block results, contiguous attempt hashes, replacement,
  cap exhaustion, execution state, candidate viability, inference effects,
  multiplicity retention, and dominance suppression;
- validate pair lifecycle, scope, sidecar, source, family-hash, and
  schedule-hash propagation.

### Task 14E — final orchestration/report test

- invoke `run_pipeline` exactly as specified in Section 2;
- call the Task 14C and 14D assertion helpers;
- assert report JSON/Markdown/plot/migration outputs and exact claim language;
- run the no-force idempotency pass and compare authenticated hashes;
- enforce/log the runtime budget without making a timing threshold the only
  correctness assertion.

The final test body should read as orchestration plus four high-level calls,
not as hundreds of inline assertions:

```python
run_pipeline(cfg, seed_pair=(11, 22), oracle_game_profile=profile)
assert_raw_and_root_oracle(contexts, expected)
assert_pair_h2h_oracle(pair_context, expected)
assert_authenticated_artifact_graph(contexts, pair_context)
assert_report_oracle(pair_context.config, expected)
assert_idempotent_rerun(...)
```

This division keeps the final test readable while ensuring every helper is
itself exercised against artifacts produced by the actual canonical workflow.
