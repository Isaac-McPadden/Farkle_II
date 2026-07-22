# Canonical Data Artifacts

Every derived artifact has one logical scope, one precisely named operation,
and exactly one adjacent hash-bound sidecar. Current consumers reject an
artifact when its scope, method contract, content hash, freshness metadata, or
source identity is incompatible.

## Scopes

| Scope | Meaning |
| --- | --- |
| `by_k` | One concrete player count |
| `concat_ks` | Row-preserving concatenation across player counts |
| `across_k` | A declared cross-k estimator or summary |
| `cross_seed` | Two-root raw-count combination or stability diagnostic |
| `diagnostics` | Explicitly descriptive diagnostic output |
| `h2h_2p` | Two-player finalist simulation and inference |

Paths are resolved through `AppConfig`. A file from one scope cannot satisfy a
consumer for another scope.

## Sidecar contract

For data file `x.parquet`, metadata is written to
`x.parquet.sidecar.json`. Writers stage both files atomically, publish data,
publish the sidecar, and write completion state last. The sidecar binds:

- artifact, estimand, schema, and RNG versions;
- exact artifact SHA-256 and byte size;
- producer, source scope, output scope, and operation;
- a tagged `method_contract`;
- baseline, weighted quantity, support-count role, and conditioning;
- k support, declared weights, missing-cell policy, and root scope;
- uncertainty method and replication unit;
- source artifacts, manifest hashes, config hash, and code revision.

Artifact publication, sidecar validation, hashing, and canonical artifact reads
use bounded retries for transient filesystem failures. Retry classification is
based on portable exception and error semantics rather than operating-system or
sync-provider names, so local disks, network filesystems, and synchronized
storage follow the same contract. Persistent errors still fail closed.

Streamed ingest rows, row-preserving curated files, rare-event shards and
counter checkpoints, and auxiliary per-k TrueSkill exports receive the same
adjacent hash-bound sidecars as final summaries. A no-force resume may publish
a genuinely missing sidecar for bytes already covered by a valid completion
stamp; it validates rather than replaces any sidecar that already exists.

Method tags are `operation`, `h2h`, `trueskill`, `diagnostic_band`,
`conditional_metrics`, `turn_metrics`, and `root_combination`. The method
procedure must exactly equal the sidecar operation identifier.

## Lifecycle and freshness

The only lifecycle states are:

- `not_started`
- `partial_resumable`
- `complete_valid`
- `complete_stale`
- `blocked_by_cap`

Freshness includes every statistical contract version, RNG version, chance
baseline, k support and weighting, conditioning, multiplicity, and candidate
family identity. Old completion stamps cannot validate replacement outputs.

## Simulation and row artifacts

Simulation outputs are rooted at `cfg.results_root`. The runner owns immutable
`(root, k, shuffle_index)` coordinates. Outcome-schema-v2 rows carry root, k,
shuffle index, game index, deterministic batch ID, RNG provenance, exact
`n_turns`, and an explicit `termination_status`. A `safety_limit` row is an
attempted game with no winner, winning score, victory margin, or player ranks;
all nullable winner/rank fields remain null rather than crediting P1.

Tournament checkpoints and resumable metric chunks use tournament-method-v2.
They conserve `attempted_games = completed_games + safety_limit_games` and
`total_wins = completed_games`, with attempted, completed, and safety-limit
exposure counters by strategy. Both version identifiers participate explicitly
in freshness identity, so winner-only predecessor artifacts cannot resume into
this contract.

Ingest and curate operate by k and preserve those identifiers. The combine
stage writes a row-preserving `concat_ks` union. It verifies source/output row
identity and total count without changing values or keys.

## All-player metrics and performance

`cfg.metrics_all_player_batch_path(k)` contains one row per
`(root, k, batch, strategy)`. It includes every attempted player-game exposure,
including all safety-limit participants as losses, zero-point turns, and
maximum-round abort exposure counts. It publishes attempted, completed,
safety-limit, win, and loss counts plus safety-limit exposure rates. Exact
estimators include:

- turn-weighted return: total final score / total turns;
- game-weighted exact return: mean of final score / turns;
- rounds-proxy return and its absolute/relative gap;
- turn/round mismatch prevalence.

Winner-only outputs use `win_conditioned_*` fields and cannot satisfy the
unconditional schema.

Per-k performance uses `win_rate_per_attempt = wins / attempted exposures` as
the primary rate. Its chance delta remains `win_rate_per_attempt - 1/k`, so a
safety-limit attempt intentionally penalizes every participant. Wilson checks
and batch MCSE use that same per-attempt denominator. The completed-only
`win_rate_given_completion` is a labelled diagnostic and is accompanied by its
completed exposure count. Across-k canonical performance is the
complete-support equal-k mean of the per-attempt chance delta; raw attempted,
completed, safety-limit, win, and loss counts remain reconstructible. Joint
batch-vector resampling supplies rank, top-N, contrast, and shortlist
diagnostics.

Separate outputs report Pareto membership, maximin descriptive leadership,
controls, finite chance-relative log odds, pairwise k contrasts, per-k spread,
and cross-k Spearman/Kendall agreement. Boundary logits are unavailable.

## Seat, game, RNG, and roll diagnostics

Seat counts are canonical at `(root, batch, strategy, k, seat)` and separately
record attempted, completed, and safety-limit exposures. Strategy and
population effects use the per-attempt win indicator within k relative to
`1/k`; a safety-limit attempt gives no seat a win. Cross-k standardization
requires identical common support and declared weights. Exposure-weighted
cross-k mixtures are separate diagnostics. Self-play reports a primary
per-attempt P1 rate and a labelled completed-only diagnostic. Paired mirrored
effects are winner-conditioned products and therefore exclude safety-limit
attempts while reporting their excluded counts.

Game-length and multi-target game summaries use attempted games; their
strategy-conditioned counterparts use one seated-strategy exposure per seat
per attempted game. Winner/margin/close-game products use completed games or
completed seated-strategy exposures only. Every game-stat output states its
observational unit and its completed/safety-limit support. Canonical seat
strategy columns match `P<seat>_strategy`; `winner_strategy` is never treated
as another exposure. RNG reference fields use
`diagnostic_band_*`; they do not assert independence. Roll diagnostics exactly
enumerate all ordered outcomes for one through six dice.

## TrueSkill and HGB

TrueSkill ratings are canonical only per root and k. The candidate contribution
normalizes percentile rank within each complete root/k cell and averages those
percentiles. Tau-zero, reversed-order, and held-out predictive-calibration
results are diagnostics.

HGB scoring and permutation importance are evaluated on held-out strategy
configurations and report fold variability and finite-grid support. Results are
associations. Full-grid fits may write future-only proposal rows without adding
them to the current strategy manifest.

## Two-root stability

Two-root performance first combines raw wins and exposures within each k, then
computes cross-k estimates. Root-specific estimates remain alongside the
combined result. Stability outputs include raw and standardized discrepancies,
threshold fractions, joint discrepancy diagnostics, rank correlations,
95th-percentile rank movement, root bootstrap top-N inclusion, controls,
shortlist changes, matched-count convergence, and first/second-half drift.

Roots are deterministic simulation domains for one fixed design. These outputs
make no root-population inference.

## H2H family, schedule, and inference

The family is the union of the top 75 canonical win-rate entries, top 75
TrueSkill entries, configured controls, and mandatory diagnostics. A cap
contracts both unprotected tails simultaneously. Family tables and manifests
record every rank, admission reason, cutoff round, removal, overlap, content
hash, and projected workload.

The power plan binds the family, effects, alpha, target power, seat scenarios,
allocation, RNG version, and score procedure in a schedule hash. Immutable
pair/root/order blocks carry that hash and coordinate-owned RNG identity.
The power plan remains immutable after publication. The execution stage owns a
separate `execution_state.json` containing the family and schedule hashes,
lifecycle state, completed block count, and total block count. Inference
requires this artifact to be `complete_valid` and to match the power plan.
If final completion-stamp publication is interrupted, a no-force resume
authenticates the frozen block set and canonical execution outputs, then writes
only the missing stamp; it does not replay blocks or rebuild the aggregate.

Pair outputs are grouped by their owning stages under
`results_seed_pair_X_Y/seed_pair_analysis`: root stability, TrueSkill,
candidate freeze, H2H power, H2H execution, H2H inference, H2H digest,
agreement, and reporting. Canonical scope directories remain beneath those
stages.

Inference reports root-combined and labelled root-specific `q_AB`, `q_BA`, and
`d_AB = 0.5(q_AB - q_BA)`, score-inversion intervals, Holm decisions,
Bonferroni simultaneous practical bounds, and root agreement diagnostics.
Equivalence is unavailable unless a margin is explicitly configured.

Practical and statistical graphs retain unresolved comparisons. Strongly
connected cycle groups record members, strongest and weakest internal edges,
and a deterministic representative shortest cycle. Fronts are zero-indegree
layers of the condensation graph. A unique-best claim requires direct
practical dominance over every finalist.

## Agreement, reports, and migration

Canonical Markdown, JSON, and plot outputs require sidecar-validated inputs.
Reports state roots, k support, weights, controls, conditioning, candidate
provenance, intervals, unresolved evidence, cycles, and the decision rule for
each meaning of “best.” Outside two-player analyses, H2H is labelled as an
external finalist diagnostic. Report-contract-v2 also states the primary
per-attempt tournament rate and publishes attempted/completed/safety-limit
games by root and k plus strategy exposure counts and rates.

The migration report inventories ignored old on-disk artifacts and identifies
their canonical replacements. It never deletes user results, and current code
never reads those files.
