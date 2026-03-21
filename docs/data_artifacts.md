# Simulation and analysis data artifacts

This document summarizes the artifact contracts that the codebase currently
guarantees. It intentionally avoids historical row counts and deleted module
references.

## Path rules

- Resolve analysis paths through `AppConfig` helpers.
- Do not hard-code numbered stage folders in downstream scripts.
- Use `analysis/config.resolved.yaml` to inspect the layout chosen for a run.
- Interseed layouts can be renumbered when RNG diagnostics are disabled.

Useful helpers live on `farkle.config.AppConfig`, including:

- `cfg.results_root`
- `cfg.analysis_dir`
- `cfg.stage_dir("<stage>")`
- `cfg.metrics_input_path("metrics.parquet")`
- `cfg.trueskill_path("ratings_k_weighted.parquet")`
- `cfg.head2head_path("bonferroni_pairwise.parquet")`
- `cfg.preferred_tiers_path()`

## Simulation artifacts

Results are rooted under `cfg.results_root`, typically:

```text
data/<results_dir_prefix>_seed_<seed>
```

Common simulation artifacts:

- `active_config.yaml`
  The effective config written for the run.
- `log.txt`
  Command log output for `run`, `analyze`, and `two-seed-pipeline`.
- `strategy_manifest.parquet`
  Strategy manifest used to map stable strategy identifiers.
- `<n>_players/<n>p_checkpoint.pkl`
  Tournament checkpoint used for resumability.
- `<n>_players/win_counts.csv`
  Per-strategy win totals for that player count.
- `<n>_players/<n>p_metrics.parquet`
  Final per-strategy metric aggregates for that player count.
- Optional row outputs under `<n>_players`
  Depending on `cfg.sim.row_dir`, runs may emit row parquet shards or
  consolidated row parquet files for ingestion.

## Analysis artifacts by stage

### Ingest

Per-player-count ingest outputs live under:

```text
cfg.ingest_block_dir(k)
```

Key files:

- `<k>p_ingested_rows.raw.parquet`
- `<k>p_ingested_rows.raw.manifest.jsonl`

### Curate

Per-player-count curated outputs live under:

```text
cfg.curate_block_dir(k)
```

Key files:

- `game_rows.parquet` by default, or `analysis.outputs.curated_rows_name`
- `manifest.jsonl` by default, or `analysis.outputs.manifest_name`

### Combine

Pooled combine outputs are resolved through:

- `cfg.curated_parquet`
- `cfg.curated_dataset`
- `cfg.combined_manifest_path()`

Current preferred locations are under:

```text
cfg.stage_dir("combine") / "pooled"
```

The code supports both:

- A monolithic combined parquet, typically `all_ingested_rows.parquet`
- A partitioned dataset directory, typically `all_ingested_rows_partitioned`

### Metrics

Metrics outputs are resolved through:

- `cfg.metrics_output_path()`
- `cfg.metrics_input_path()`
- `cfg.metrics_isolated_path(k)`

Common files:

- `pooled/metrics.parquet`
- `pooled/seat_advantage.csv`
- `pooled/seat_advantage.parquet`
- `<k>p/<k>p_isolated_metrics.parquet`

### Coverage and game stats

Common stage roots:

- `cfg.stage_dir("coverage_by_k")`
- `cfg.stage_dir("game_stats")`

Typical outputs:

- `coverage_by_k.parquet`
- `game_length_k_weighted.parquet`
- `margin_k_weighted.parquet`
- `rare_events.parquet`

Game-stat helpers also support legacy filenames through `AppConfig`
canonicalization and fallback logic.

### TrueSkill and tiering

Common paths:

- `cfg.trueskill_stage_dir`
- `cfg.trueskill_pooled_dir`
- `cfg.tiering_stage_dir`
- `cfg.preferred_tiers_path()`

Typical outputs:

- `ratings_<k>.parquet`
- `ratings_<k>.json`
- `ratings_k_weighted.parquet`
- `ratings_k_weighted.json`
- `tiers.json`

### Head-to-head and post-processing

Common paths:

- `cfg.head2head_stage_dir`
- `cfg.post_h2h_stage_dir`
- `cfg.head2head_path("bonferroni_pairwise.parquet")`
- `cfg.post_h2h_path("bonferroni_decisions.parquet")`

Typical outputs:

- `bonferroni_pairwise.parquet`
- `bonferroni_pairwise_ordered.parquet`
- `bonferroni_selfplay_symmetry.parquet`
- `bonferroni_decisions.parquet`
- `h2h_significant_graph.json`
- `h2h_s_tiers.json`

### HGB

Common paths:

- `cfg.hgb_stage_dir`
- `cfg.hgb_pooled_dir`

Typical outputs:

- `hgb_importance.json`

### Interseed analytics

Interseed runs use a dedicated layout resolved by
`resolve_interseed_stage_layout(...)`. Relevant stage roots include:

- `cfg.stage_dir("rng_diagnostics")` when RNG diagnostics are enabled
- `cfg.variance_stage_dir`
- `cfg.meta_stage_dir`
- `cfg.agreement_stage_dir`
- `cfg.interseed_stage_dir`

Typical outputs include:

- RNG diagnostic parquet outputs
- variance summaries
- interseed game-stat summaries
- meta-analysis outputs
- pooled/interseed TrueSkill outputs
- agreement outputs
- `interseed_summary.json`

## Metadata artifacts

The pipeline relies on metadata files in addition to tabular outputs.

- `analysis/config.resolved.yaml`
  The resolved config snapshot, including the active stage layout.
- `manifest.jsonl`
  Append-only run manifest written by stage orchestration.
- `<artifact>.done.json`
  Stage completion metadata used for skip-if-fresh checks.
- `two_seed_pipeline_manifest.jsonl`
  Pair-level orchestration manifest for dual-seed runs.
- `pipeline_health.json`
  Health/status artifact emitted by two-seed orchestration.

## Legacy compatibility

The codebase still reads several legacy file names and legacy directory
locations. That compatibility is implemented in `AppConfig` path helpers.

Practical guidance:

- Write new code against canonical helper methods.
- Do not assume a legacy alias is the preferred output location.
- Do not add fixed deprecation dates to docs unless they are enforced in code.

## What not to assume

- Fixed stage numbers across every run type
- Historical row counts from earlier experiments
- Deleted modules or old helper classes
- Flat `analysis/` artifact locations when stage helpers exist
