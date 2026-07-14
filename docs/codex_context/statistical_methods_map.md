# Codex Statistical Methods Map

Generated for Codex orientation. Treat this file as a cache, not authority.
For any statistical review, verify the listed source files directly and
separate implementation fidelity from model validity.

- Sources inspected: `utils/stats.py`, `analysis/metrics.py`,
  `analysis/performance.py`, `analysis/screening.py`, `analysis/seed_summaries.py`,
  `analysis/meta.py`,
  `analysis/variance.py`, `analysis/run_trueskill.py`,
  `analysis/run_bonferroni_head2head.py`, `analysis/h2h_analysis.py`,
  `analysis/agreement.py`

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

## Canonical Performance Estimators

- Code: `src/farkle/analysis/all_player_metrics.py` and
  `src/farkle/analysis/performance.py`.
- Per-k estimand: raw wins divided by raw player-game exposures, with chance
  baseline `1/k` and `chance_delta = win_rate - 1/k`.
- Per-k uncertainty: full-sample Wilson interval plus batch
  `MCSE = sample_sd(batch win rates) / sqrt(number of batches)` and a t interval.
- Across-k estimand: equal-k mean of chance delta over complete configured
  support only. Its analytic variance is the sum of squared equal weights times
  the independent-k batch-MCSE variances.
- Robustness: exact finite-grid Pareto membership and a separately identified
  maximin descriptive leader.
- Resampling: each replicate selects complete deterministic-batch vectors, so
  all strategies share the same selected batches within k. Outputs summarize
  ranks, top-N membership, practical-shortlist inclusion, and declared-control
  contrasts.
- Tests: `tests/unit/analysis/test_all_player_metrics.py` and
  `tests/unit/analysis/test_performance.py` contain hand-computed return,
  baseline, MCSE, equal-k variance, support, and determinism checks.

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

## Descriptive Performance Screening

- Code: `src/farkle/analysis/screening.py`.
- Inputs: validated per-k performance, complete-support equal-k performance,
  and joint deterministic-batch resampling summaries.
- Evidence retained: root and per-k chance deltas, equal-k score order,
  bootstrap top-N and shortlist inclusion, declared controls and mandatory
  diagnostics, Pareto membership, and the separate maximin leader.
- Practical bands compare observed scores with explicitly configured
  thresholds. They are descriptive and are not tests of equality, final tiers,
  or unique-best decisions.
- Tests: `tests/unit/analysis/test_screening.py`.

## Screening Workload Planning

- Code: `src/farkle/simulation/workload_planner.py`,
  `src/farkle/simulation/runner.py::_plan_workload_from_config`.
- Method: find the smallest shuffle count whose worst-case full 95% Wilson
  interval width meets `screening.resolution_delta`, then round upward to equal
  contiguous batches with the configured minimum batch size.
- Output: required shuffles and games, batch construction, achieved resolution,
  operational cap state, and projected runtime.
- Tests: `tests/unit/simulation/test_workload_planner.py`,
  `tests/unit/simulation/test_runner_wrapper.py`.
- H2H power remains a distinct procedure and must match the final H2H test.

## Canonical Seat Effects

- Code: `src/farkle/analysis/seat_analysis.py`.
- Inputs: validated normalized per-k row partitions with one root and complete
  configured k support.
- Canonical counts: raw wins and player-game exposures for every
  `(root, k, deterministic batch, strategy, seat)` cell.
- Within-k estimands: strategy-specific and population-wide seat win rates
  minus the exact `1/k` chance baseline.
- Across-k estimand: equal-k or declared-k weighted effects only for identical
  common k support. Missing strategy/k/seat cells are not silently reweighted.
- Secondary diagnostic: the exposure-weighted cross-k mixture reports its own
  exposure-weighted chance baseline and is not interchangeable with the
  standardized estimand.
- Additional diagnostics: self-play P1 effects and paired mirrored two-player
  P1-win differences, with unmatched orientations retained as support counts.
- Tests: `tests/unit/analysis/test_seat_analysis.py`.

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
  shared tau2 for all strategy groups.
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
- Output fields are `diagnostic_band_lower` and `diagnostic_band_upper`. The
  sidecar records the `1.96/sqrt(n)` approximation and explicitly states that
  inclusion within a reference band does not establish independence.
- Tests: `tests/unit/analysis/test_rng_diagnostics.py`,
  `tests/unit/analysis/test_rng_diagnostics_branches.py`.
- Review risks: diagnostics can flag suspicious patterns but cannot prove RNG
  independence or simulation validity by themselves.

## Exact Roll Enumeration

- Code: `src/farkle/analysis/roll_enumeration.py`.
- Method: enumerate every one of the `6**d` ordered outcomes for each dice count
  from one through six and evaluate it with the production scoring engine.
- Selection rule: `production_max_immediate_score_v1`, with zero score for a
  farkle. Outputs include the exact score/scoring-dice distribution, farkle and
  hot-dice probabilities, expected immediate score, score quantiles, and
  expected scoring dice.
- No RNG or sampling interval is involved; sidecars label the finite
  enumeration and ordered-outcome denominator.
- Tests: `tests/unit/analysis/test_roll_enumeration.py`.

## High-Value Missing Hand Checks

- Meta-analysis: two strategies, two seeds, one boundary-rate case, expected Q,
  I2, tau2, fixed/random switch, and CI values.
- Head-to-head: three pairs with known p-values and Holm adjusted p-values,
  including one tie and one reversed direction.
- TrueSkill: tiny deterministic ordered game stream confirming rank extraction,
  checkpoint resume equivalence, and precision-aggregation arithmetic.
- Metrics vs seed summaries: same wins/games through normal CI and Wilson CI to
  document intentional interval differences.
