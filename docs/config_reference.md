# Config Reference

This is a concise reference for the `farkle.config.AppConfig` structure used by
the CLI and orchestration code.

## Top-level sections

- `io`
  Filesystem roots and analysis subdirectory settings.
- `sim`
  Simulation seeds, player counts, strategy grid, worker counts, and progress settings.
- `screening`
  Wilson-width resolution target, operational shuffle cap, and optional runtime rate.
- `batching`
  Equal contiguous shuffle-batch construction.
- `robustness`
  Finite-grid summaries and two-root reproducibility thresholds.
- `k_aggregation`
  Equal-k or explicitly declared player-count weights.
- `artifact_contract`
  Versions included in sidecar compatibility and freshness decisions.
- `analysis`
  Analysis-stage behavior, logging, aggregation, rare-event settings, and optional
  interseed inputs.
- `ingest`
  Streaming parquet write settings for ingestion.
- `combine`
  Settings for combined combined row outputs.
- `metrics`
  Metric-computation settings such as seat ranges.
- `trueskill`
  TrueSkill screening hyperparameters.
- `head2head`
  Head-to-head simulation and post-processing settings.
- `hgb`
  Histogram gradient boosting hyperparameters.
- `orchestration`
  Top-level orchestration toggles such as parallel seed execution.

## Seed model

- `sim.seed_list` is the canonical seed source.
- Single-seed commands expect one seed in `seed_list`.
- Two-seed orchestration expects two seeds in `seed_list`.
- `sim.seed` and `sim.seed_pair` remain compatibility aliases and are
  normalized on load.

## Important `sim` fields

- `n_players_list`
  Player counts to simulate.
- `row_dir`
  Optional row-output directory.
- `metric_chunk_dir`
  Optional resumable metric-chunk output directory.
- `n_jobs`
  Worker count for simulation.
- `per_n`
  Optional per-player-count runtime overrides. It does not override the resolved workload.

## Screening workload

- `screening.resolution_delta`
  Maximum full Wilson interval width. The default is `0.03` at 95% confidence.
- `screening.max_shuffles_per_root_k`
  Optional operational safety cap. Insufficient caps stop before scheduling and identify this key.
- `screening.projected_games_per_second`
  Optional positive throughput estimate used for the pre-scheduling runtime projection.
- `batching.target_batches`
  Locked to `100` equal contiguous batches.
- `batching.min_shuffles_per_batch`
  Minimum shuffles in every batch; values below `30` are rejected.

The planner finds the smallest shuffle count meeting the worst-case Wilson-width
target, then rounds upward to the batch contract. `sim.power_method`,
`sim.recompute_num_shuffles`, and `sim.power_design` are rejected retired keys.

## Two-root robustness

- `robustness.delta_seed_stability`
  Positive raw chance-delta threshold used to scale root discrepancies.
- `robustness.joint_discrepancy_alpha`
  Family level for the joint maximum-standardized-discrepancy diagnostic.
- `robustness.matched_count_fractions`
  Unique increasing cumulative batch fractions in `(0, 1]`; the final value
  must be `1.0`.

Two-root combination treats roots as independent RNG domains for one fixed
simulation design. These settings control reproducibility diagnostics, not
random-effects inference or population intervals over roots.

## Candidate-family freeze

- `screening.candidate_contribution_size`
  Number of entries contributed by each canonical method ranking; default `75`.
- `screening.controls`
  Strategy identifiers protected as declared controls.
- `screening.mandatory_diagnostics`
  Strategy identifiers protected for prespecified diagnostics.
- `head2head.candidate_cap`
  Optional maximum family size. When absent, the complete declared union is
  retained.
- `head2head.candidate_cap_policy`
  Locked to `balanced-tail`: both nonprotected method cutoffs decrease by one in
  every contraction round. The result may be smaller than the cap.

Every protected strategy must occur in at least one canonical contribution,
and the cap cannot be smaller than the protected family.

## H2H power and allocation

- `head2head.family_alpha`: Holm familywise alpha; locked default `0.02`.
- `head2head.target_power`: minimum worst-scenario planning power; default
  `0.80`.
- `head2head.practical_delta`: reported seat-adjusted target effect; default
  `0.03`.
- `head2head.sensitivity_deltas`: includes the target effect and `0.04`.
- `head2head.seat1_advantage_scenarios`: locked to `0`, `0.03`, and `0.06`.
- `head2head.total_game_cap`: operational cap checked before block scheduling.
- `head2head.allow_single_root`: permits explicitly labelled single-root H2H.

Power is calculated for the same independent two-proportion score procedure
used downstream, at Bonferroni `family_alpha / pair_count` as a conservative
Holm planning threshold. Allocation is equal across roots and seat orders.

## Important `analysis` fields

- `log_level`
- `n_jobs`
- `disable_rng_diagnostics`
- `agreement_strategies`
- `agreement_include_combined`
- `game_stats_margin_thresholds`
- `k_aggregation_method`
- `k_weights`
- `rare_event_target_score`
- `rare_event_margin_quantile`
- `rare_event_target_rate`
- `rng_max_matchup_groups`
- `head2head_target_hours`
- `head2head_tolerance_pct`
- `head2head_games_per_sec`
- `head2head_force_calibrate`
- `outputs`

Deprecated `analysis.run_*` and `analysis.disable_*` flags are still accepted
for compatibility, but most no longer control stage scheduling. The current
pipeline schedules stages from inputs and preconditions instead.

## `io` notes

- `results_dir_prefix`
  Results root, usually written under `data/`.
- `analysis_subdir`
  Name of the analysis directory under the results root.
- `meta_analysis_dir`
  Legacy path retained until the migration cleanup; canonical two-root outputs
  use `cfg.cross_seed_dir(...)`.
- `interseed_input_dir`
  Optional alternate analysis root used to resolve interseed inputs.
- `interseed_input_layout`
  Optional explicit mapping for interseed input stage folders.

## Output naming

`analysis.outputs` can override selected filenames, including:

- `curated_rows_name`
- `metrics_name`
- `manifest_name`

If you need a path, prefer `AppConfig` helpers over manual filename assembly.

## `hgb`

- `max_depth`: maximum HGB tree depth.
- `n_estimators`: boosting iteration count (`max_iter` in sklearn).
- `heldout_folds`: deterministic strategy-configuration folds; must be at
  least 2.
- `permutation_repeats`: permutation repeats within each held-out fold; must be
  positive.
- `future_proposal_limit`: maximum draft candidates for a future simulation;
  zero disables proposal generation.

HGB evaluation is configuration-held-out. These settings do not authorize
insertion of proposed configurations into the current analysis.

## Common override examples

```bash
farkle --config configs/fast_config.yaml --set sim.n_jobs=8 run
```

```bash
farkle --config configs/fast_config.yaml --set analysis.log_level=DEBUG analyze pipeline
```

```bash
farkle --config configs/fast_config.yaml --set orchestration.parallel_seeds=true two-seed-pipeline --seed-pair 10 11
```

## Recommendation

When extending the config model:

- add a dataclass field in `config.py`
- make the path or behavior discoverable through `AppConfig` when appropriate
- document the new field here and in any user-facing CLI docs that surface it
