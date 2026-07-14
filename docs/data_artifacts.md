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

Row-preserving cross-k outputs are resolved through:

- `cfg.curated_parquet`
- `cfg.curated_dataset`
- `cfg.combined_manifest_path()`

Canonical outputs are under:

```text
cfg.stage_dir("combine") / "concat_ks"
```

The code supports both:

- A monolithic concatenated parquet, `all_ingested_rows.parquet`
- A partitioned concatenated dataset, `all_ingested_rows_partitioned`

Both layouts preserve the typed root, k, shuffle, game, deterministic-batch,
RNG, and turn-accounting columns from canonical `curate/by_k` rows. Combine
verifies source/output row identity without loading the dataset into memory.
Retired locations are ignored and inventoried under
`combine/diagnostics/migration_report.json`.

### Metrics

Metrics outputs are resolved through:

- `cfg.metrics_output_path()`
- `cfg.metrics_input_path()`
- `cfg.metrics_isolated_path(k)`
- `cfg.metrics_all_player_batch_path(k)`

Common files:

- `combined/metrics.parquet`
- `combined/seat_advantage.csv`
- `combined/seat_advantage.parquet`
- `<k>p/<k>p_isolated_metrics.parquet`
- `<k>p/all_player_batch_metrics.parquet`
- `<k>p/performance.parquet`
- `<k>p/seat_batch_counts.parquet`
- `<k>p/seat_effects.parquet`
- `<k>p/seat_population_effects.parquet`
- `across_k/performance_equal_k.parquet`
- `across_k/performance_bootstrap.parquet`
- `across_k/performance_control_contrasts.parquet`
- `across_k/seat_effects_standardized_across_k.parquet`
- `diagnostics/seat_exposure_mixture.parquet`
- `diagnostics/seat_selfplay_p1.parquet`
- `diagnostics/seat_mirrored_games.parquet`
- `cross_seed/performance_root_combination_<k>p.parquet`
- `cross_seed/performance_root_combination_across_k.parquet`
- `cross_seed/root_discrepancies.parquet`
- `cross_seed/root_joint_discrepancy.parquet`
- `cross_seed/root_rank_stability.parquet`
- `cross_seed/root_top_n_stability.parquet`
- `cross_seed/root_control_movement.parquet`
- `cross_seed/root_shortlist_changes.parquet`
- `cross_seed/root_matched_count_convergence.parquet`
- `cross_seed/root_half_drift.parquet`

`all_player_batch_metrics.parquet` is the canonical unconditional exposure
artifact. It contains one row per `(root_seed, k, deterministic_batch_id,
strategy)` and streams every player exposure, including losing and zero-score
exposures. Its sufficient statistics support these distinct estimands:

- `turn_return_turn_weighted = sum(final_score) / sum(n_turns)`
- `turn_return_game_weighted_exact = mean(final_score / n_turns)`
- `turn_return_round_proxy = mean(final_score / n_rounds)`
- `round_proxy_gap`, `round_proxy_relative_gap`, and exact turn/round mismatch
  prevalence

The artifact sidecar declares `conditioning=unconditional`. Its schema rejects
`win_conditioned_*` columns. Winner-only isolated metrics use the explicit
`win_conditioned_*` prefix and cannot satisfy this consumer contract.

Canonical performance uses raw wins and exposures from those batch rows.
Per-k outputs include the exact `1/k` chance baseline, chance delta, Wilson
resolution check, and batch MCSE. Across-k scores require every configured k
and use the equal-k mean of `win_rate - 1/k`; their variance is the sum of the
independent-k variance contributions with equal weights. The output also keeps
Pareto membership and the maximin descriptive leader separate.

Bootstrap summaries resample complete batch vectors, so all strategies in a k
share the same selected batch indices within each replicate. They report rank,
top-N, practical-shortlist, and declared-control contrast stability. Every
performance artifact is hash-bound to an adjacent sidecar.

Canonical seat counts contain one row per `(root_seed, k,
deterministic_batch_id, strategy, seat)` with raw wins and player-game
exposures. Strategy-specific and population-wide effects are computed within k
as `seat win rate - 1/k`. The across-k artifact includes only identical common
k support and uses the configured equal-k or declared-k weights.

`seat_exposure_mixture.parquet` is a secondary diagnostic, not the standardized
estimand. It combines raw exposures and uses
`sum(exposures_k / k) / sum(exposures_k)` as its corresponding chance baseline.
Self-play first-seat effects and paired two-player mirrored-game differences
are separate diagnostics; the mirrored output also records unpaired games.
Every canonical seat artifact has a hash-bound sidecar.

Two-root performance first combines raw wins and player-game exposures within
each k. The combined estimate is therefore exactly
`sum(root wins) / sum(root exposures)`, not a mean of root rates. The declared
equal-k or configured-k calculation is applied only after those within-k
estimates exist. Each output retains the two root-specific estimates beside the
combined result.

The remaining `cross_seed` artifacts are reproducibility diagnostics for one
fixed simulation design: raw and MCSE-standardized differences, threshold
fractions, a joint maximum-discrepancy batch bootstrap, rank and top-N overlap,
control and shortlist movement, matched-count convergence, and contiguous
first-half/second-half drift. They do not estimate a population of roots and do
not publish random-effects heterogeneity or two-root population intervals.

### H2H candidate freeze

The pre-scheduling two-player family is published under
`cfg.h2h_2p_dir("head2head")`:

- `candidate_family.parquet`
- `candidate_family.json`
- `power_plan.json`
- `block_manifest.parquet`
- `blocks/pair_<id>_root_<root>_order_<order>.parquet`
- `root_order_counts.parquet`
- `combined_order_counts.parquet`
- `pairwise_inference.parquet`
- `dominance_edges.parquet`
- `cycle_groups.parquet`
- `dominance_fronts.parquet`
- `dominance_summary.json`

The membership table records every canonical win-rate and TrueSkill source
rank, method-list membership, protected control/diagnostic status, admission
reasons, simultaneous cutoff round, cap removal, and the family hash. The JSON
manifest records the initial and final method cutoffs, every contraction round,
overlap statistics, admission counts, content hashes, and projected unordered
pair/root/order block counts.

The default contribution is the top 75 from each canonical method ranking. An
optional candidate cap reduces both nonprotected method cutoffs together until
their union plus protected entries fits. No cap retains the complete declared
union. A single-root family is supported but explicitly labelled in the
manifest and sidecars.

`power_plan.json` sizes the independent two-proportion score procedure at the
conservative Bonferroni threshold `family_alpha / unordered_pair_count`. It
finds the smallest allocation that attains target power in the worst declared
common seat-1-advantage scenario and reports the sensitivity grid. If the
projected games exceed `head2head.total_game_cap`, it publishes
`blocked_by_cap` guidance and does not publish a block manifest.

A ready block manifest divides every pair equally across roots and both seat
orders. Each immutable block records the family hash and its exact
`(root, pair, order, game_index)` RNG coordinates. Completed blocks are written
atomically with sidecars; interruption and worker-count changes therefore skip
valid blocks without changing any stream. `root_order_counts.parquet` is the
row-preserving union consumed by seat-adjusted inference.

Inference first combines raw wins and games across roots separately within
`a_b` and `b_a`; it never mixes the seat orders before estimating their rates.
For each pair, `q_AB` is A's seat-1 win rate in `a_b`, `q_BA` is B's seat-1 win
rate in `b_a`, and `d_AB = 0.5 * (q_AB - q_BA)`. The score-test null estimate,
ordinary score-inversion interval, Bonferroni simultaneous interval, Holm
adjustment, practical threshold, and decision class are all recorded.

The balanced A-win rate is retained only as an equality-checked point-estimate
alias for `0.5 + d_AB`. Equivalence is emitted only when an explicit margin is
configured; nonsignificance otherwise remains unresolved.

Practical and statistical dominance are separate directed graphs. Strongly
connected groups remain explicit in `cycle_groups.parquet` and are collapsed
only to construct each condensation DAG. Fronts are repeated zero-indegree
layers of that DAG; strategies in the same cycle remain separate rows in the
same front. Within-front order uses round-robin mean, practical wins/losses,
tournament score, and stable identifier solely for display and adds no
inferential edge.

`dominance_summary.json` permits a unique-best claim only when one strategy has
a direct practical-dominance edge to every other frozen finalist. Neither a
front position nor a path through other strategies satisfies that rule.

### Coverage and game stats

Common stage roots:

- `cfg.stage_dir("coverage_by_k")`
- `cfg.stage_dir("game_stats")`

Typical outputs:

- `coverage_by_k.parquet`
- `game_length_strategy_conditioned_equal_k_mean.parquet`
- `margin_strategy_conditioned_equal_k_mean.parquet`
- `diagnostics/roll_outcome_distribution_exact.parquet`
- `diagnostics/roll_summary_exact.parquet`
- `rare_events.parquet`

The two across-k game summaries are descriptive, strategy-conditioned equal-k
means over complete configured support. Their sidecars distinguish them from
the row-preserving `concat_ks/game_length.parquet` and
`concat_ks/margin_stats.parquet` artifacts. Exact roll outputs use the named
production scoring-selection rule and finite ordered-outcome counts; they are
not Monte Carlo estimates.

### TrueSkill and descriptive screening

Common paths:

- `cfg.trueskill_stage_dir`
- `cfg.trueskill_combined_dir`
- `cfg.screening_stage_dir`
- `cfg.preferred_tiers_path()`

Typical outputs:

- `by_k/<k>p/ratings_<k>_seed<root>.parquet`
- `concat_ks/ratings_concat_ks.parquet`
- `across_k/candidate_percentile_contribution.parquet`
- `diagnostics/screening_diagnostics.parquet`
- `descriptive_screening.parquet`
- `descriptive_screening.json`

TrueSkill ratings remain root/k-specific. The cross-k candidate contribution is
the complete-support equal mean of within-root/per-k `mu` percentile ranks and
does not propagate model sigma. Tau-zero, reversed-order, and held-out
predictive-calibration results are descriptive diagnostics. Screening outputs
report finite-grid evidence and practical bands; they do not emit inferential
tiers or equality decisions.

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
- `cfg.hgb_combined_dir`
- `cfg.hgb_importance_path(k)`
- `cfg.hgb_predictive_scores_path(k)`
- `cfg.hgb_fold_metrics_path(k)`
- `cfg.hgb_future_proposals_path()`

Canonical outputs:

- per-k held-out predictive scores and fold metrics;
- per-k permutation associations calculated only on held-out strategy
  configurations, including between-fold variability and finite-grid support;
- an equal-k descriptive association summary in `hgb_importance.json`;
- `future_simulation_proposals.parquet`, whose rows have
  `included_in_current_analysis=false` and no current-run strategy ID.

HGB output describes predictive associations over the configured finite grid.
It is exploratory, does not identify causal option effects, and cannot add a
proposed strategy to the analysis that generated the proposal.

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
- combined/interseed TrueSkill outputs
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
