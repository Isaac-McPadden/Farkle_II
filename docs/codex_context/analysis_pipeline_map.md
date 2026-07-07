# Codex Analysis Pipeline Map

Generated for Codex orientation. Treat this file as a cache, not authority.
Verify `src/farkle/analysis/stage_registry.py`, `src/farkle/analysis/pipeline.py`,
and `src/farkle/analysis/__init__.py` before changing pipeline behavior.

- Sources inspected: stage registry, analysis CLI pipeline, analysis plan
  builders, config path helpers, artifact docs

## Stage Layout

The default registry in `stage_registry.py` resolves numbered folders at
runtime. Do not hard-code the numbers in new code.

Current default registry order:

1. `ingest`
2. `curate`
3. `combine`
4. `metrics`
5. `coverage_by_k`
6. `game_stats`
7. `seed_summaries`
8. `trueskill`
9. `frequentist`
10. `head2head`
11. `seed_symmetry`
12. `post_h2h`
13. `hgb`
14. `variance`
15. `meta`
16. `h2h_tier_trends`
17. `agreement`
18. `interseed`

Interseed-specific registry order:

1. `rng_diagnostics` as folder stub `rng`, unless disabled
2. `variance`
3. `interseed_game_stats`
4. `meta`
5. `trueskill`
6. `agreement`
7. `interseed`
8. `h2h_tier_trends`

## Pipeline Entrypoints

- Package CLI `farkle analyze ...` is implemented in `src/farkle/cli/main.py`.
  Its `analyze pipeline` subcommand runs `_run_preprocess(...)` followed by
  `analysis.run_all(...)`; it does not delegate to `src/farkle/analysis/pipeline.py`.
- `src/farkle/analysis/pipeline.py` is a standalone analysis pipeline CLI with
  its own parser and `StageRunner` execution path. Verify which entrypoint is in
  use before changing CLI behavior.
- `pipeline.py` resolves `AppConfig`, writes `analysis/config.resolved.yaml`,
  assigns a config SHA, appends manifest events, then runs selected stages
  through `StageRunner`.
- `analysis.__init__.build_per_seed_stage_plan(...)` builds the per-seed plan
  used by orchestration: ingest, curate, combine, metrics, coverage, game stats,
  seed summaries, TrueSkill, frequentist, head-to-head, seed symmetry,
  post-H2H, and HGB.
- `analysis.__init__.build_interseed_analysis_plan(...)` builds the interseed
  plan with readiness checks against `cfg.interseed_ready()`.
- `two_seed_pipeline.py` coordinates two per-seed runs, then interseed work.

## Stage Families

- Ingest: normalizes raw simulation row outputs into per-player-count raw
  parquet shards and manifests.
- Curate: normalizes ingested rows into current schemas and canonical curated
  parquet files.
- Combine: merges per-player-count curated outputs into pooled combined data.
- Metrics: computes per-strategy wins, games, win rates, expected scores, seat
  advantage, seat metrics, isolated metrics, and weighted pooled metrics.
- Coverage and game stats: report missing strategy/seed/player-count coverage,
  game length, margins, rare events, and related summaries.
- Seed summaries, variance, meta: produce seed-aware summaries and cross-seed
  uncertainty/pooling artifacts.
- TrueSkill, frequentist, head-to-head, post-H2H: produce ranking, tiering, and
  pairwise-comparison outputs.
- HGB: model-based strategy-feature importance and partial-dependence artifacts.
- Agreement: compares ranking/tiering outputs across methods.
- Interseed: summarizes paired-seed comparisons and optional RNG diagnostics.

## Artifact Path Rules

Use `AppConfig` helpers:

- Stage roots: `cfg.stage_dir("metrics")`, `cfg.stage_subdir(...)`,
  `cfg.stage_dir_if_active(...)`.
- Core paths: `cfg.results_root`, `cfg.analysis_dir`, `cfg.curated_dataset`,
  `cfg.metrics_input_path(...)`, `cfg.metrics_output_path(...)`.
- Statistical outputs: `cfg.trueskill_path(...)`, `cfg.head2head_path(...)`,
  `cfg.post_h2h_path(...)`, `cfg.frequentist_path(...)`,
  `cfg.meta_output_path(...)`, `cfg.variance_output_path(...)`,
  `cfg.agreement_output_path(...)`.
- Tiers: `cfg.preferred_tiers_path()`.

The code still reads legacy paths in many places. New writes should prefer
canonical helpers, not legacy flat `analysis/` paths.

## Cache And Resume

- Stage freshness is controlled by `.done.json` files via
  `src/farkle/analysis/stage_state.py`.
- Stage definitions carry `cache_scope` and `cache_key_version`; `AppConfig`
  projects stage-specific config hashes.
- Simulation uses checkpoints, manifests, strategy manifests, row manifests,
  and metric chunk manifests to validate resume state.
- Atomic writes are expected for final artifacts. Partial files should not be
  left behind.

## High-Risk Review Points

- Stage registry and plan builders are not identical views of execution. Check
  the specific command or orchestrator path before assuming a stage runs.
- Deprecated `analysis.run_*` and `analysis.disable_*` config fields mostly no
  longer control scheduling, except RNG diagnostics and specific CLI handling.
- Interseed input paths can use an alternate root and alternate layout, so path
  resolution must go through config helpers.
- `metrics`, `seed_summaries`, `variance`, and `meta` use related but not
  identical pooling and uncertainty conventions.
- Head-to-head creates several artifacts with different statistical meanings:
  pairwise simulation results, ordered seat-balance rows, self-play symmetry,
  Holm-adjusted decisions, graph outputs, and S-tier JSON.

## Fast Orientation Commands

Use read-only commands first:

```powershell
git status --short
rg -n "StageDefinition|build_.*plan|stage_map|stage_done" src/farkle/analysis src/farkle/config.py
rg -n "metrics_output_path|trueskill_path|head2head_path|preferred_tiers_path" src/farkle/config.py
rg -n "write_stage_done|stage_is_up_to_date|atomic_path|write_parquet_atomic" src tests
```
