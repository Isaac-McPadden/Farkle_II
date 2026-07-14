# Statistical Analyses

This index describes the current production analysis modules. The detailed
method contracts live in
[`docs/codex_context/statistical_methods_map.md`](docs/codex_context/statistical_methods_map.md).

- `all_player_metrics`: streams unconditional root/k/batch/strategy sufficient
  statistics for wins, exposures, exact turn returns, proxy gaps, and aborts.
- `performance`: computes per-k chance-relative performance, Wilson resolution,
  batch MCSE, complete-support cross-k scores, resampling diagnostics, Pareto,
  maximin, controls, and player-count effects.
- `screening`: descriptive candidate evidence only; it does not establish
  equality, tiers, or a unique best strategy.
- `seat_analysis`: within-k strategy/population seat effects, common-support
  standardization, self-play, and mirrored-game diagnostics.
- `game_stats`: game-level lengths, score configurations, margins, close games,
  rare events, and matchup ecology.
- `rng_diagnostics`: dependence-aware descriptive reference bands without an
  independence claim.
- `roll_enumeration`: exact ordered-outcome enumeration for one through six
  dice.
- `run_trueskill` and `trueskill_screening`: canonical root/k sequential
  ratings, percentile-rank candidate contribution, and sensitivity/calibration
  diagnostics.
- `run_hgb`: held-out finite-grid prediction and permutation association
  importance; full-grid fits are future-proposal-only.
- `root_stability`: raw-count two-root combination and descriptive stability;
  it makes no root-population inference.
- `candidate_family`: freezes the H2H family with balanced-tail contraction and
  complete provenance.
- `h2h_schedule`: validates score-test power and owns immutable
  pair/root/order blocks.
- `h2h_inference`: estimates seat-adjusted pair effects, score intervals, Holm
  decisions, simultaneous practical bounds, and root diagnostics.
- `dominance`: constructs practical/statistical graphs, cycles, fronts, and the
  direct-dominance unique-best rule.
- `structure_agreement`: compares canonical screening inputs and
  selection-conditioned H2H evidence.
- `structure_reporting`: writes sidecar-validated JSON, Markdown, plot, and
  migration outputs with controlled claim language.

The canonical across-k performance estimand is the equal-k mean of
`win_rate - 1/k` over complete configured k support. Root combination is total
wins divided by total exposures within each k before any cross-k calculation.

H2H uses Holm familywise alpha `0.02`, target power `0.80`, practical effect
`0.03`, sensitivity `0.04`, and common first-seat scenarios `0`, `0.03`, and
`0.06`. Equivalence is unavailable unless an explicit margin is configured.
