# Config Reference

This is a concise reference for the `farkle.config.AppConfig` structure used by
the CLI and orchestration code.

## Top-level sections

- `io`
  Filesystem roots and analysis subdirectory settings.
- `sim`
  Simulation seeds, player counts, strategy grid, worker counts, and progress
  settings.
- `analysis`
  Analysis-stage behavior, logging, pooling, rare-event settings, and optional
  interseed inputs.
- `ingest`
  Streaming parquet write settings for ingestion.
- `combine`
  Settings for pooled combined row outputs.
- `metrics`
  Metric-computation settings such as seat ranges.
- `trueskill`
  TrueSkill hyperparameters and pooled weighting overrides.
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
- `num_shuffles`
  Tournament size per player count.
- `row_dir`
  Optional row-output directory.
- `metric_chunk_dir`
  Optional resumable metric-chunk output directory.
- `n_jobs`
  Worker count for simulation.
- `per_n`
  Optional per-player-count simulation overrides.
- `power_design`
  Nested power-analysis inputs used by planning helpers.

## Important `analysis` fields

- `log_level`
- `n_jobs`
- `disable_rng_diagnostics`
- `agreement_strategies`
- `agreement_include_pooled`
- `game_stats_margin_thresholds`
- `pooling_weights`
- `pooling_weights_by_k`
- `rare_event_target_score`
- `rare_event_margin_quantile`
- `rare_event_target_rate`
- `rng_max_matchup_groups`
- `tiering_seeds`
- `tiering_z_star`
- `tiering_min_gap`
- `head2head_target_hours`
- `head2head_tolerance_pct`
- `head2head_games_per_sec`
- `head2head_force_calibrate`
- `meta_random_if_I2_gt`
- `meta_max_other_seeds`
- `meta_comparison_seed`
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
  Optional alternate root for meta-analysis artifacts.
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
