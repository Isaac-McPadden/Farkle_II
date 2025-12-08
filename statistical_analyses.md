pipeline.py (top-level): Front-door helpers that run the end-to-end analytics pipeline on an experiment directory, tracking .done.json stamps so heavy statistical stages only re-run when inputs change.

farkle.analysis.pipeline: CLI orchestrator for the analysis pipeline (ingest → curate → combine → metrics → analytics), wiring together config, manifests, and sequential execution of the statistical stages.

farkle.analysis (package __init__): Lightweight orchestrator that runs the individual analytics modules (TrueSkill, head-to-head, frequentist tiering, HGB, agreement, seed summaries, meta) according to config, tying the higher-level statistical analyses together.

farkle.analysis.ingest: Streams raw simulation outputs, validates basic schemas, and writes per–player-count parquet shards that form the raw data foundation for all downstream statistical processing.

farkle.analysis.curate: Normalizes ingested shards into curated parquet files with stable schemas and JSON manifests, ensuring statistically analyzable, reproducible datasets for later stages.

farkle.analysis.combine: Merges per–player-count curated parquets into a unified all_ingested_rows.parquet with a superset schema, creating a single combined dataset for global metrics and seat-effect analyses.

farkle.analysis.checks: Provides validation routines that enforce schema expectations, non-negativity of counts, and manifest consistency so that statistical computations run on internally coherent data.

farkle.analysis.metrics: Aggregates combined data into per-strategy metrics (wins, games, win rates, expected scores) and seat-advantage tables, producing the core descriptive statistics for strategies and positions.

farkle.analysis.metrics (win probability/uncertainty): Adds a win_prob alias for symmetric matchups and computes standard errors/CI bounds for win rates before writing consolidated metrics.

farkle.analysis.game_stats: Derives per-game lengths, margins of victory, close-game shares, and rare tie-like flags from curated rows, emitting aggregated length/margin tables per strategy and player count.

farkle.analysis.seat_stats: Extends seat-advantage outputs with per-seat win rates, score/farkle/round averages, and symmetry diagnostics comparing seats in symmetric matchups.

farkle.analysis.isolated_metrics: Collects per-seed, per–player-count tournament metrics into “isolated” parquet frames, enabling seed-aware downstream analyses such as meta-analysis, tiering, and feature models.

farkle.analysis.seed_summaries: Builds per-seed, per–player-count strategy summaries with Wilson confidence intervals, capturing uncertainty on win rates for each (seed, players) combination.

farkle.analysis.variance: Summarizes randomness vs player count using win-rate variance across seeds, signal-to-noise heuristics, and variance decomposition of win rates, scores, and game lengths.

farkle.analysis.meta: Performs fixed- and random-effects meta-analysis of the per-seed summaries to pool win-rate estimates across seeds, computing heterogeneity diagnostics (e.g., I²) and pooled strategy performance.

farkle.analysis.rng_diagnostics: Computes autocorrelation diagnostics over game seeds for win indicators and game length to flag RNG ordering artifacts; optional and lightweight.

farkle.analysis.trueskill: Thin pipeline wrapper that triggers the TrueSkill analysis, ensuring that TrueSkill-based tier files are (re)computed when curated game data change.

farkle.analysis.run_trueskill: Implements the TrueSkill rating workflow by scanning tournament data, updating per-game ratings, and writing per-player-count and pooled rating tables plus tier assignments for strategies.

farkle.analysis.head2head: Designs and runs Bonferroni head-to-head simulations between top strategies, using power calculations and simulation results to propose tier configurations under statistical error and runtime constraints.

farkle.analysis.run_bonferroni_head2head: Executes the pairwise game simulations and binomial tests with Bonferroni-style power targets, writing detailed head-to-head outcome tables used for later multiple-comparisons correction and ranking.

farkle.analysis.h2h_analysis: Applies Holm–Bonferroni adjustments to head-to-head p-values, constructs a directed significance graph, and derives a statistically justified ranking based on significant pairwise differences.

farkle.analysis.tiering_report: Combines isolated metrics, weighting across player counts, to compute frequentist win-rate scores and tiers, then compares them to TrueSkill tiers and emits consolidated tiering outputs.

farkle.analysis.hgb_feat: Wrapper that coordinates histogram gradient boosting feature-importance runs per player count, checking timestamps and delegating to run_hgb when new statistical feature analyses are needed.

farkle.analysis.run_hgb: Trains histogram gradient boosting models that predict TrueSkill performance from parsed strategy features, exporting permutation-based feature importances and optional diagnostics for interpretability.

farkle.analysis.agreement: Loads TrueSkill, frequentist, and head-to-head results to compute rank and tier agreement metrics (e.g., rank correlations, ARI, NMI, seed stability), quantifying consistency between statistical methods.

farkle.analysis.reporting: Reads the various analysis artifacts (ratings, meta summaries, head-to-head tiers, feature importances, seed summaries) and produces Markdown reports and plots, turning the statistical results into human-readable summaries.