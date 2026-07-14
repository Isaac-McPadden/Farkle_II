# Codex Statistical Methods Map

Generated for Codex orientation. Treat this file as a cache, not authority.
For any statistical review, verify the listed source files directly and
separate implementation fidelity from model validity.

- Sources inspected: `utils/stats.py`, `utils/mdd_helpers.py`, `analysis/mdd.py`,
  `analysis/metrics.py`, `analysis/seed_summaries.py`, `analysis/meta.py`,
  `analysis/variance.py`, `analysis/run_trueskill.py`,
  `analysis/run_bonferroni_head2head.py`, `analysis/h2h_analysis.py`,
  `analysis/frequentist_ranking.py`, `analysis/agreement.py`

## Review Frame

For every statistical claim, review in this order:

1. Intended estimator or decision rule.
2. Implementation mapping from formula to code.
3. Model assumptions and independence assumptions.
4. Artifact/report interpretation.
5. Tests, hand checks, and missing counterexamples.

## Wilson Binomial Confidence Interval

- Code: `src/farkle/utils/stats.py::wilson_ci`.
- Used by: `analysis/seed_summaries.py`, boundary handling in `analysis/meta.py`.
- Inputs: successes `k`, trials `n`, two-sided `alpha`.
- Formula: Wilson score interval with
  `z = norm.ppf(1 - alpha / 2)`,
  denominator `1 + z^2 / n`, center `p + z^2 / (2n)`, and Wilson margin.
- Edge behavior: rejects `n <= 0`; callers such as `seed_summaries` use
  `(0, 1)` for zero-game summaries.
- Tests: `tests/test_stats_wilson.py`, `tests/unit/utils/test_stats.py`,
  `tests/unit/analysis/test_seed_summaries.py`.
- Review risks: zero-game policy, integer count coercion, whether Wilson vs
  normal intervals are consistently labeled in downstream artifacts.

## Metrics Win-Rate Uncertainty

- Code: `src/farkle/analysis/metrics.py::_add_win_rate_uncertainty`.
- Inputs: per-strategy `wins`, `games`, `win_rate`.
- Formula: normal approximation `se = sqrt(p * (1 - p) / n)`,
  `win_rate_ci_lo = p - 1.96 * se`, `win_rate_ci_hi = p + 1.96 * se`,
  clipped to `[0, 1]`.
- Outputs: `se_win_rate`, `win_rate_ci_lo`, `win_rate_ci_hi`.
- Tests: `tests/unit/analysis/test_metrics.py`,
  `tests/unit/analysis/test_metrics_branches.py`,
  `tests/integration/test_metrics_stage.py`.
- Review risks: normal approximation differs from Wilson intervals in seed
  summaries; zero-game handling leaves CI equal to win rate; verify denominator
  is games for the intended Bernoulli unit.

## Aggregation Across Player Counts

- Code: `analysis/metrics.py::_compute_weighted_metrics`,
  `analysis/seed_summaries.py::_build_combined_seed_summary`,
  `utils/aggregation.py`.
- Aggregation modes: `game-count`, `equal-k`, and `config`.
- Game-count: row weights are game counts.
- Equal-k: each player count contributes equal total mass by scaling rows by
  `games / total_games_for_k`.
- Config: configured per-k weights are normalized by total games for that k;
  missing k weights are treated as zero after warning.
- Tests: `tests/unit/analysis/test_metrics*.py`,
  `tests/unit/analysis/test_seed_summaries.py`,
  `tests/unit/utils/test_stage_io_and_aggregation.py`.
- Review risks: combined rows can mean "across k" rather than "across seeds";
  artifact names with `combined` need scope-specific interpretation.

## Frequentist MDD And Tiers

- Code: `src/farkle/utils/mdd_helpers.py`,
  `src/farkle/analysis/mdd.py`,
  `src/farkle/analysis/frequentist_ranking.py`.
- Inputs: isolated metrics by strategy, player count, seed, wins, games, and
  optional `weights_by_k`.
- `prepare_cell_means`: computes per-row win rate, then averages row-level
  rates within `(strategy, k, seed)` and sums games. It does not recompute
  grouped rate as `sum(wins) / sum(games)` when multiple rows exist per cell.
- Default smoothing: Jeffreys-style `(wins + 0.5) / (games + 1)` when using
  wins/games rather than an existing win-rate column.
- `tau2_seed`: across-seed variance by `(strategy, k)` minus binomial variance,
  clipped at zero, aggregated robustly by median by default.
- `tau2_sxk`: weighted across-k dispersion identity; implementation matches the
  identity documented in `mdd_helpers.py`.
- MDD formula: `mdd = z_star * sqrt(2 * var_theta)`, where
  `var_theta = sum(w_k^2 * binom_k / R) + tau2_seed / R + tau2_sxk * sum(w_k^2)`.
- Frequentist tiers: sort combined win rates descending; start a new tier when
  the next rate falls below the current threshold by more than MDD.
- Tests: `tests/unit/utils/test_mdd.py`,
  `tests/unit/analysis/test_frequentist_ranking.py`.
- Review risks: cell averaging convention, Jeffreys smoothing interaction with
  supplied `win_rate`, missing k coverage, whether MDD is interpreted as
  pairwise detectable difference rather than a confidence interval.

## Power Sizing

- Code: `src/farkle/utils/stats.py::games_for_power`,
  `src/farkle/simulation/power_helpers.py`,
  `src/farkle/simulation/runner.py::_compute_num_shuffles_from_config`,
  `src/farkle/analysis/head2head.py::_predict_runtime`.
- Methods: Bonferroni FWER or BH/BY-style planning level.
- Endpoint `pairwise`: two-sample proportion sizing around `p0`, then converts
  pair co-appearances to games per strategy by dividing by `k_players - 1`.
- Endpoint `top1`: one-sample proportion sizing against baseline `1/k` unless
  overridden.
- Tests: `tests/unit/utils/test_stats.py`,
  `tests/unit/simulation/test_simulation.py`,
  `tests/unit/simulation/test_runner_wrapper.py`.
- Review risks: planning approximation is not the same as final hypothesis
  testing; BH target rank/fraction is a design choice; floor/cap may dominate
  computed sample size.

## TrueSkill Ratings And Tiers

- Code: `src/farkle/analysis/trueskill.py`,
  `src/farkle/analysis/run_trueskill.py`.
- Inputs: curated rows with strategy columns and ranking information.
- Update rule: streams games and calls `trueskill.TrueSkill(...).rate(...)`
  with ranks from `P#_rank`, `seat_ranks`, or fallback winner-plus-tied-losers.
- Hyperparameters: `cfg.trueskill.beta`, `tau`, `draw_probability`.
- Resume: per-block checkpoints and done stamps.
- Aggregation: precision weighting using `tau = 1 / sigma^2`; combined `mu` is
  `sum(tau * mu) / sum(tau)`, combined `sigma = sqrt(1 / sum(tau))`; optional
  per-k weights affect combined k outputs.
- Tiers: `utils.stats.build_tiers` groups overlapping confidence tiers from
  means and sigmas.
- Tests: `tests/unit/analysis/test_run_trueskill_*.py`,
  `tests/unit/analysis/test_analytics_trueskill.py`,
  `tests/unit/analysis/test_run_trueskill_aggregation.py`.
- Review risks: TrueSkill is a sequential rating model, not a classical
  independent-binomial estimator; game order matters; sigma should not be read
  as a simple sampling standard error without validating the model assumptions.

## Head-To-Head Simulation And Holm Decisions

- Code: `src/farkle/analysis/head2head.py`,
  `src/farkle/analysis/run_bonferroni_head2head.py`,
  `src/farkle/analysis/h2h_analysis.py`.
- Candidate selection: union of top combined TrueSkill ratings and frequentist
  scores; `use_tier_elites` is currently logged as ignored in favor of combined
  score artifacts.
- Simulation: each pair is split across both seat orders (`a_b` and `b_a`) and
  writes pairwise, ordered-pairwise, and self-play symmetry artifacts.
- Initial p-value: `run_bonferroni_head2head.py` stores
  `pval_one_sided = binomtest(wins_a, games, alternative="greater").pvalue`.
- Post-processing: `h2h_analysis.py` aggregates canonical unordered pairs,
  recomputes two-sided binomial p-values, applies Holm-Bonferroni adjusted
  p-values, builds a significant directed graph, then derives tiers/rankings.
- Tie policies: `neutral_edge` marks ties non-significant; `simulate_game`
  gives deterministic non-significant tie-break direction from a seeded RNG.
- Tests: `tests/unit/analysis/test_run_bonferroni_head2head*.py`,
  `tests/unit/analysis/test_head2head.py`,
  `tests/unit/analysis/test_h2h_analysis.py`,
  `tests/unit/analysis/test_agreement_ties.py`.
- Review risks: one-sided stored pair p-values differ from two-sided
  post-H2H decisions; pairwise tests are dependent because strategies share
  tournament context; final graph ranking can be partial or cyclic.

## Meta-Analysis Across Seeds

- Code: `src/farkle/analysis/meta.py`.
- Inputs: `strategy_summary_{players}p_seed*.parquet` files.
- Strategy presence: only strategies present in every selected seed summary
  contribute to the combined estimate.
- Per-seed variance: `p * (1 - p) / games`, with boundary rates replaced by a
  Wilson-logit-centered estimate and a minimum variance floor.
- Fixed effect: inverse-variance weighted mean per strategy.
- Heterogeneity: sums Q across strategies, computes global I2 and
  DerSimonian-Laird style tau2. If I2 exceeds `meta_random_if_I2_gt`, uses one
  shared tau2 for all strategy pools.
- Output CI: normal approximation `combined_rate +- 1.959963984540054 * se`.
- Tests: `tests/unit/analysis/test_meta.py`.
- Review risks: one global tau2 across all strategies, strategy pruning,
  assumptions that seeds are independent exchangeable runs, and probability
  scale aggregation near boundaries.

## Cross-Seed Variance

- Code: `src/farkle/analysis/variance.py`.
- Inputs: metrics parquet and per-seed summaries.
- Computes sample variance of `win_rate` across seeds by
  `(strategy_id, players)`, `std`, and `se = std / sqrt(n_seeds)`.
- Components: separate cross-seed variance summaries for win rate, score mean,
  and turns mean, with normal-style CIs on component means.
- Minimum seeds: `MIN_SEEDS = 2`.
- Tests: `tests/unit/analysis/test_variance.py`,
  `tests/unit/analysis/test_variance_branch_closure.py`.
- Review risks: signal-to-noise currently compares against `0.5`, which is not
  the fair top-1 win rate for k-player games when k is not 2.

## Agreement Metrics

- Code: `src/farkle/analysis/agreement.py`.
- Methods compared: TrueSkill, frequentist, and combined head-to-head when
  available.
- Rank metrics: Spearman and Kendall on common strategy sets.
- Tier metrics: adjusted Rand index and normalized mutual information on common
  tier maps.
- Stability: per-seed score standard deviations for common strategies.
- Tests: `tests/unit/analysis/test_agreement_payload.py`,
  `tests/unit/analysis/test_agreement_ties.py`.
- Review risks: agreement is descriptive, not proof of correctness; missing
  strategies are summarized as coverage, and tied scores are warned but allowed.

## RNG Diagnostics

- Code: `src/farkle/analysis/rng_diagnostics.py`.
- Intended use: optional interseed diagnostics over game seeds, win indicators,
  and game length to flag ordering artifacts.
- Tests: `tests/unit/analysis/test_rng_diagnostics.py`,
  `tests/unit/analysis/test_rng_diagnostics_branches.py`.
- Review risks: diagnostics can flag suspicious patterns but cannot prove RNG
  independence or simulation validity by themselves.

## High-Value Missing Hand Checks

- Meta-analysis: two strategies, two seeds, one boundary-rate case, expected Q,
  I2, tau2, fixed/random switch, and CI values.
- Head-to-head: three pairs with known p-values and Holm adjusted p-values,
  including one tie and one reversed direction.
- TrueSkill: tiny deterministic ordered game stream confirming rank extraction,
  checkpoint resume equivalence, and precision-aggregation arithmetic.
- Metrics vs seed summaries: same wins/games through normal CI and Wilson CI to
  document intentional interval differences.
