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
| `screening` | Wilson-width target, practical thresholds, bootstrap size, and candidate inputs |
| `batching` | Deterministic batch construction |
| `robustness` | Pareto, maximin, and two-root stability diagnostics |
| `k_aggregation` | Equal-k or explicitly declared player-count weights |
| `artifact_contract` | Versions used by sidecars and freshness keys |
| `analysis` | Root-local diagnostics, workers, output names, and rare-event settings |
| `ingest` | Streaming parquet settings |
| `combine` | Maximum supported player count |
| `trueskill` | Root/k TrueSkill screening parameters |
| `head2head` | Candidate cap, power, allocation, and inference settings |
| `hgb` | Held-out predictive-association settings |
| `orchestration` | Root execution concurrency |

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
- `screening.resolution_delta = 0.03`, the maximum full 95% Wilson width
- `screening.practical_delta_by_k` must contain every configured k
- `screening.delta_across_k` must be positive
- `batching.target_batches = 100`
- `batching.min_shuffles_per_batch >= 30`
- `k_aggregation.method` is `equal-k` or `declared-mapping`
- `artifact_contract.artifact_contract_version = 2`

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
- `joint_discrepancy_alpha`
- `matched_count_fractions`

The workload planner chooses the smallest shuffle count meeting the Wilson
target, then rounds upward to 100 equal contiguous batches. A cap that is too
small produces `blocked_by_cap` before simulation work begins.

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

The planner validates the implemented two-proportion score rejection rule at
the Bonferroni planning threshold. Work is equal across roots and seat orders;
single-root work is equal across seat orders.

## Simulation, analysis, and model settings

`sim` owns `n_jobs`, process start method, checkpoint cadence, row/metric
locations, and the strategy option grid. `analysis` owns analysis workers,
optional RNG diagnostics, game-stat thresholds, rare-event settings, and the
three overridable output names (`curated_rows_name`, `metrics_name`, and
`manifest_name`).

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
  artifact_contract_version: 2
```

```powershell
farkle --config configs/fast_config.yaml --set sim.n_jobs=8 run
farkle --config configs/fast_config.yaml two-seed-pipeline --seed-pair 42 43
```
