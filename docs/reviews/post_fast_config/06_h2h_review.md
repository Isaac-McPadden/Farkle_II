# Adversarial post-remediation review: complete H2H workflow

## Scope, evidence, and verdict

Reviewed commit `6be5f5fa11df77155621bfc81188c7515f38f8de` and the completed
`configs/fast_config.yaml` run at
`data/results_seed_pair_32_33/seed_pair_analysis`. This review followed the H2H
workflow from canonical win-rate and TrueSkill candidate contribution through
family freeze, power planning, allocation, simulation, inference, dominance
digestion, agreement, and reporting. It did not modify application code, tests,
configuration, or existing run artifacts.

**Verdict: unsound within this review's scope.** The candidate-union arithmetic,
pair/root/order schedule, accepted H2H estimand, score calculations, Holm and
Bonferroni procedures, graph digestion, and report claim restrictions are
largely implemented correctly and reproduce from the stored aggregates.
However, the simulator converts maximum-round ties into P1 wins and H2H throws
away the abort state. Six completed-run comparisons therefore use 47,376
aborted 0-0 games as if they were completed Bernoulli trials and report narrow
intervals from zero demonstrated completed games. This is a blocker regardless
of the otherwise-correct downstream arithmetic. The run is also not bound to
the reviewed code revision, and narrower defects remain in allocation
validation, exact-power minimality, cap-resume immutability, worker-setting
ownership, and workload provenance.

## Accepted estimand and implementation trace

The implemented primary estimand agrees with the accepted contract:

- Schedule order 0 places A in seat 1 and B in seat 2; order 1 reverses them
  (`src/farkle/analysis/h2h_schedule.py:460-502`).
- Raw counts are combined across roots within order, not across orders
  (`src/farkle/analysis/h2h_inference.py:370-414`).
- `q_ab = x_ab/n_ab`, `q_ba = x_ba/n_ba`, and
  `d_ab = 0.5 * (q_ab - q_ba)`; the code independently checks that
  `0.5 + d_ab` equals the balanced A win rate
  (`src/farkle/analysis/h2h_inference.py:417-495`).
- The zero-effect test uses the pooled constrained-null variance for two
  independent proportions and a two-sided normal tail
  (`src/farkle/analysis/h2h_inference.py:62-93`). Ordinary and Bonferroni
  intervals invert the uncorrected score procedure, including a boundary-safe
  fallback (`src/farkle/analysis/h2h_inference.py:96-271`).
- Holm adjusted p-values control zero-effect decisions; Bonferroni simultaneous
  score intervals separately control practical-dominance and optional
  equivalence decisions (`src/farkle/analysis/h2h_inference.py:274-289,417-531`).
- Practical and statistical edges are separate. Unresolved/equivalent pairs
  create no edge; SCCs and condensation fronts retain cycles and partial order;
  within-front sorting is explicitly noninferential
  (`src/farkle/analysis/dominance.py:58-144,266-410`). A unique best requires a
  direct practical edge over every other finalist
  (`src/farkle/analysis/dominance.py:626-654`).
- Because fast config includes `k = 2, 4, 5`, reporting labels H2H an
  `external_two_player_finalist_diagnostic` and suppresses any primary
  multi-k unique-best claim (`src/farkle/analysis/structure_reporting.py:439-447,498-527`).

All statistical H2H artifacts from candidate freeze through agreement are in
stage-local `h2h_2p/` scopes. Their sidecars declare `player_counts=[2]` and
`required_player_counts=[2]`; the overall structure report correctly lives in
`diagnostics/`. Current consumers contain no retired pooled, weighted-mu,
players-zero, seed-random-effects, or forced-total-order path.

## Findings

### H1 - Maximum-round aborts become P1 wins, and H2H manufactures precision after discarding the abort state

**Severity: Blocker. Confidence: High.**

**Classification:** Confirmed statistical and data-validity defect.

**Evidence.** `FarkleGame.play` sets `max_rounds_hit` after 200 rounds but then
stable-sorts scores, chooses the first player as winner, and assigns unique
ranks even when every score is tied (`src/farkle/game/engine.py:408-480`).
`_play_game` emits that rank-1 player as `winner_seat`
(`src/farkle/simulation/simulation.py:359-413`). H2H's
`_winner_seat_counts` accepts only P1/P2, and `_simulate_block` reduces each
game to the two win counts without retaining completion, tie, abort, or
termination reason (`src/farkle/analysis/h2h_schedule.py:768-830`). Inference
then requires only `wins_seat1 + wins_seat2 == games_completed` and treats all
such rows as independent completed games
(`src/farkle/analysis/h2h_inference.py:329-367`).

The completed `root_order_counts.parquet` has six comparisons among strategies
`0`, `1`, `20`, and `21`, comprising 24 root/order blocks and 47,376 nominal
games. Every block reports `games_completed=1974`, `wins_seat1=1974`, and
`wins_seat2=0`. A read-only replay of pair 0 `(0,1)`, root 32, order AB, game 0
from its stored coordinate produced `P1_score=P2_score=0`,
`P1_hit_max_rounds=P2_hit_max_rounds=1`, and `n_rounds=200`, but still emitted
`winner_seat=P1`, `P1_rank=1`, `P2_rank=2`. The stored inference consequently
reports `q_ab=q_ba=1`, `d_ab=0`, `p=1`, and a simultaneous interval of about
`[-0.002544, 0.002544]` for each of those six pairs. With a typical configured
equivalence margin such as 0.02, these zero-completion comparisons would be
labelled equivalent.

This defect also contaminates the tournament win-rate and TrueSkill inputs to
candidate selection: maximum-round ties receive arbitrary winners/ranks before
both source methods see them. The later family freeze can be mechanically exact
while still freezing a family selected from invalid outcomes.

**Consequence.** The six affected H2H comparisons have no demonstrated
completed-game support, yet receive the narrowest-looking no-effect evidence in
the run. The aggregate has discarded the information needed to distinguish an
abort from a win, so it cannot be repaired from H2H counts. Candidate selection
and every downstream H2H claim are conditioned on contaminated tournament
outcomes as well.

**Smallest reasonable remediation.** Give games an explicit completion status
and termination reason. A maximum-round tie/abort must not receive a winner or
rank-1 exposure. Retain attempted, completed, tied, and aborted counts in every
H2H block and aggregate; exclude non-completions from win denominators and make
zero-completed-support comparisons unresolved/incomparable. If power is defined
over completed games, schedule deterministic replacement coordinates while
retaining attempt provenance. Rebuild tournament, TrueSkill, family, H2H,
inference, dominance, agreement, and report artifacts after the fix.

### H2 - The completed H2H artifacts cannot be tied to the reviewed code revision

**Severity: High. Confidence: High.**

**Classification:** Confirmed provenance defect inherited by the H2H workflow.

**Evidence.** `make_artifact_sidecar` defaults `code_revision` to the literal
`"unknown"` and H2H producers do not override it
(`src/farkle/utils/artifact_contract.py:272-341`). Every inspected H2H sidecar,
including `candidate_family.json.sidecar.json`,
`power_plan.json.sidecar.json`, `root_order_counts.parquet.sidecar.json`, and
`pairwise_inference.parquet.sidecar.json`, records
`"code_revision": "unknown"`. Pair completion stamps additionally record
`config_sha: null`; for example `03_h2h_power/h2h_power.done.json` has a
stage-scoped hash but no full configuration identity. The H2H artifacts were
written on 2026-07-17, while the reviewed commit is dated 2026-07-18. That date
ordering does not itself prove different code, but the missing code identity
makes equivalence unverifiable.

**Consequence.** Exact recalculation can show that current code reproduces
stored formulas from stored counts, but the artifacts cannot prove that the
reviewed source produced the simulated games or derived bytes. A code-only
statistical correction does not necessarily invalidate H2H caches.

**Smallest reasonable remediation.** Make a reproducible code identity
(release commit plus dirty-tree/source digest, or equivalent) mandatory in
sidecars and stage keys, carry an explicit full-config lineage into the pair
context, and bind validated upstream artifact SHA-256 identities rather than
only source paths and path/size/mtime freshness.

### H3 - Inference accepts unequal root/order allocations when the imbalances cancel after root pooling

**Severity: Medium. Confidence: High.**

**Classification:** Confirmed validation defect; the completed fast artifact is
not affected.

**Evidence.** `_read_counts` rejects incomplete rows, win-count imbalance,
duplicate `(pair_id, root_seed, order)` keys, and the wrong root set, but it does
not compare each block's support with the immutable plan
(`src/farkle/analysis/h2h_inference.py:292-367`).
`_combine_within_order` requires the expected cell count and two orders, while
`_pairwise_estimates` checks equality only after summing roots
(`src/farkle/analysis/h2h_inference.py:370-437`). A read-only in-memory probe
gave root 11 supports `(AB=100, BA=200)` and root 22 supports
`(AB=200, BA=100)`. The implementation accepted the four cells, pooled each
order to 300, and emitted ordinary inference. This violates the documented
equal-root/order allocation while passing the aggregate balance test.

The completed fast run itself has exactly 1,974 games in every one of 11,400
root/order blocks and therefore passes the stronger intended invariant.

**Consequence.** A malformed producer or reauthenticated recovery artifact can
give the two orders different root mixtures. Root-specific outcome differences
would then be confounded with order, while the primary effect and planned power
still appear valid.

**Smallest reasonable remediation.** Before pooling, compare the counts table
against the immutable block manifest or, at minimum, require the exact Cartesian
set of planned `(pair, root, order)` cells, exact strategy/seat/order labels,
the schedule hash, positive support, and
`games_completed == games_per_root_order_block` in every cell. Add a compensated-
imbalance rejection test.

### H4 - The exact-power search assumes a monotonicity that the implemented discrete rejection rule does not have

**Severity: Medium. Confidence: High.**

**Classification:** Confirmed planning defect; the fast plan passes the local
minimum check but the general smallest-allocation claim is unsupported.

**Evidence.** `implemented_score_test_power` correctly sums the joint binomial
law for the actual score rejection region
(`src/farkle/analysis/h2h_schedule.py:98-226`). `_minimum_block_games`, however,
binary-searches the predicate `exact power >= target` as if exact discrete power
were monotone in games (`src/farkle/analysis/h2h_schedule.py:280-344`). It is
not. Under valid parameter domains with one pair/root,
`family_alpha=0.20`, `practical_delta=0.10`, the locked seat scenarios
`(0, .03, .06)`, and target power 0.40, one game per order has worst-scenario
power 0.5128 and is the true minimum, but `_minimum_block_games` returns 14;
block 13 is below target, so the published previous-block check misleadingly
appears to prove minimality. The same counterexample occurs at targets 0.30 and
0.50.

For the completed fast plan, independent calls reproduce the reported
Bonferroni alpha `0.02/2850`, 1,974 games per root/order block, worst-scenario
power `0.8003606088973029`, and previous-block power
`0.7996629814552753`. Thus the chosen fast allocation meets target and its
immediate predecessor fails; this does not establish a globally first crossing
under the current algorithm.

**Consequence.** Supported configurations can be materially overallocated,
falsely described as the smallest exact allocation, and unnecessarily blocked
by `total_game_cap`. The returned allocation is explicitly checked to meet
target, so this defect is not evidence of underpowering.

**Smallest reasonable remediation.** Use a search that does not assume global
monotonicity of discrete power, or establish and enforce a parameter region plus
a mathematically valid lower bound from which all earlier allocations are
proven inadequate. Test global first-crossing behavior over the admitted
configuration domain; do not treat only `n-1` as proof of minimality.

### H5 - Raising the cap rewrites the published power-plan artifact

**Severity: Medium. Confidence: High.**

**Classification:** Confirmed lifecycle/provenance defect; the statistical
schedule hash itself remains stable for a cap-only change.

**Evidence.** The documented contract says the power plan remains immutable
after publication (`docs/data_artifacts.md:149-158`). The implementation embeds
the operational cap, execution authorization, and cap guidance in the power-plan
payload (`src/farkle/analysis/h2h_schedule.py:592-625`), writes that payload
before publishing a `blocked_by_cap` stamp (`src/farkle/analysis/h2h_schedule.py:650-676`),
and rewrites the same `power_plan.json` when a raised cap makes the stage stale.
The cap-resume test checks only that `schedule_hash` is unchanged
(`tests/unit/analysis/test_h2h_schedule.py:213-236`), not that the published plan
artifact is immutable. The schedule hash deliberately excludes the cap and
does preserve family, roots, effect, alpha, target power, allocation, RNG, and
method IDs (`src/farkle/analysis/h2h_schedule.py:423-452`).

**Consequence.** Cap-only resume preserves the statistical design, but it
destroys the original blocked plan bytes and conflates immutable design with
mutable execution authorization. The original cap decision cannot be audited
from the final artifact, contrary to the lifecycle contract.

**Smallest reasonable remediation.** Publish the immutable statistical plan
once and keep cap/authorization state in a separate atomic lifecycle artifact.
On a cap-only resume, authenticate the existing plan and change only operational
authorization; never replace the plan.

### H6 - Canonical execution ignores `head2head.n_jobs`

**Severity: Low. Confidence: High.**

**Classification:** Confirmed configuration/implementation disagreement.

**Evidence.** Fast config sets `head2head.n_jobs: 0` (automatic CPU count) and
`analysis.n_jobs: 4` (`configs/fast_config.yaml:40-42,57-59`). The canonical
H2H tail explicitly calls `execute_h2h_schedule(...,
n_jobs=inner.analysis.n_jobs)` (`src/farkle/analysis/__init__.py:136-146`). That
argument overrides `cfg.head2head.n_jobs` in the executor
(`src/farkle/analysis/h2h_schedule.py:1087-1092`). Pair orchestration may further
rewrite the analysis worker budget, so the completed run did not use the H2H
worker setting it records in active configuration.

**Consequence.** Results remain deterministic, but the dedicated H2H resource
control is ineffective and operational planning can be unexpectedly slower or
more aggressive.

**Smallest reasonable remediation.** Let `execute_h2h_schedule` read
`cfg.head2head.n_jobs`, or pass that field explicitly. Add a canonical-stage
test that distinguishes it from `analysis.n_jobs`.

### H7 - Candidate provenance projects self-play work that the H2H schedule never creates

**Severity: Low. Confidence: High.**

**Classification:** Confirmed artifact-contract/documentation defect.

**Evidence.** Candidate freeze records
`selfplay_root_blocks = candidate_count * root_count`
(`src/farkle/analysis/candidate_family.py:534-540`). The fast manifest therefore
projects 152 self-play root blocks. Scheduling uses only unordered combinations
of distinct candidates and creates two seat orders per root
(`src/farkle/analysis/h2h_schedule.py:460-502`); its plan and completed execution
contain exactly the 11,400 non-self pair/root/order blocks and no self-play.
No current H2H documentation or consumer assigns a role to self-play blocks.

**Consequence.** The frozen manifest disagrees with actual scheduled workload
and can mislead cap/runtime estimates or future consumers, although it does not
change the current inference.

**Smallest reasonable remediation.** Remove the field if H2H self-play is not a
current method, or explicitly implement and separately label a noninferential
self-play diagnostic without adding it to the unordered-pair family.

## Completed-run reconciliation

The following checks passed for the bytes that exist. They establish internal
wiring and arithmetic, not outcome validity after H1.

### Candidate contribution and freeze

- Both canonical source tables contain all 80 strategies with complete support.
  The win-rate contribution is the combined-root across-k score; the TrueSkill
  contribution is the equal mean of within-root/k `mu` percentile ranks over
  all six root/k cells. Raw `mu` and `sigma` are not cross-cell inputs
  (`src/farkle/analysis/trueskill_screening.py:86-180`).
- Independent reranking reproduces top 75 from each source, 74 shared entries,
  one method-specific entry from each source, and the exact 76-member union.
  Reconstructing the identity JSON reproduces family hash
  `07a2f7168e2c00249602d7581e06b4c3efd52a0ac02dbb25185bf688c8141733`.
- Fast config has no controls, mandatory diagnostics, or candidate cap, so the
  completed artifact cannot exercise their protection/contraction. Source and
  focused tests confirm protected unioning and simultaneous one-step reduction
  of both unprotected method tails (`src/farkle/analysis/candidate_family.py:244-315,431-557`).

### Power, allocation, and simulation identity

- The family has exactly 2,850 unordered pairs. `block_manifest.parquet`
  contains the exact lexically sorted combination family once, four rows per
  pair (`roots 32/33 x orders AB/BA`), 11,400 unique immutable keys and block
  IDs, one family hash, one schedule hash, and 1,974 games in every block.
- The power plan uses `alpha_per_pair = 7.017543859649123e-06`, exact joint-
  binomial score-rule power, target effect 0.03, and mappings
  `(q_ab,q_ba)=(.53,.47),(.56,.50),(.59,.53)` for common seat effects
  `0,.03,.06`. The sensitivity grid for effects 0.03 and 0.04 recalculates to
  every stored value. Projected and completed workload is 22,503,600 games,
  below the 100,000,000 cap.
- Schedule rows swap seat strategies correctly. H2H game seeds derive from
  `(root, k=2, pair, order, game)` and use explicit PCG64DXSM-derived streams
  (`src/farkle/analysis/h2h_schedule.py:784-824` and
  `src/farkle/utils/random.py:63-150`).
- `root_order_counts.parquet` matches the complete schedule identity exactly;
  every block is present once, has positive support, and satisfies
  `wins_seat1 + wins_seat2 = games_completed = games_required = 1974`.

### Inference and classifications

- Independent vectorized recalculation from `combined_order_counts.parquet`
  matches every stored `q_ab`, `q_ba`, `d_ab`, balanced-A alias, constrained-null
  p-value, Holm adjusted p-value, and rejection decision exactly (maximum
  absolute discrepancy zero after handling the six zero-variance equal-boundary
  rows by the declared rule).
- All 2,850 p-values enter one stable Holm procedure exactly once. There are
  2,315 Holm rejections. Classifications independently reproduce 832 practical-A,
  877 practical-B, 348 statistical-only-A, 258 statistical-only-B, and 535
  unresolved comparisons. Equivalence is disabled and no row is labelled
  equivalent.
- All ordinary and simultaneous interval endpoints are finite, including 294
  observed boundary-count pairs. An exhaustive 3,056-case synthetic check over
  equal/unequal small sample sizes, boundary counts, and four alpha levels found
  no containment or zero-null test/interval-duality mismatch. Strong-effect,
  no-effect, common-seat-effect, and explicit-equivalence fixtures also passed.

### Dominance, agreement, and reporting

- `dominance_edges.parquet` contains 2,315 statistical edges and 1,709 practical
  edges (4,024 typed rows total). Unresolved pairs create no edges. The fast run
  has no SCC cycles and no direct practical dominator; this is an observed fast-
  run wiring fact, not evidence that cycles are impossible.
- The rock-paper-scissors toy oracle preserves the three-node SCC, puts its
  members in one condensation front, retains unresolved comparisons, and does
  not create a unique best. Unit tests separately require direct practical
  dominance over every finalist.
- Agreement is explicitly conditioned on the frozen family and root comparison
  is labelled fixed-root reproducibility, not root-population inference
  (`src/farkle/analysis/structure_agreement.py:73-211`).
- The 1.8 KiB Markdown report remains readable at 76 finalists: it reports 535
  unresolved pairs, zero cycle groups, no unique-best claim, the external-2P
  role, and points to the interval artifact. Detailed pair rows remain in
  Parquet/JSON rather than being forced into a display total order. The JSON is
  about 1.29 MiB. No 100-150-finalist scale fixture was run, but the current
  output split is directionally appropriate for that range.

## Validation performed

- Passed 44 focused unit tests in `test_candidate_family.py`,
  `test_h2h_schedule.py`, `test_h2h_inference.py`, `test_dominance.py`,
  `test_structure_agreement.py`, and `test_structure_reporting.py`.
- Passed `tests/integration/test_structure_toy_oracle.py`, including interrupted
  block execution/resume, family and schedule hashes, score decisions, cycles,
  fronts, reports, and sidecars.
- Performed read-only artifact reconciliation for family membership/hash,
  unordered-pair completeness, block allocation, schedule/count identity,
  estimator identities, Holm adjustment, classifications, graph counts, and
  report claims.
- Performed in-memory adversarial checks for exact-power nonmonotonicity,
  compensated root/order imbalance, score interval boundaries/duality, and a
  stored H2H abort coordinate. No temporary tracked files were created.

Passing these controls demonstrates that the current formulas and graph/report
wiring operate as implemented. It does not mitigate H1: correct inference
arithmetic over fabricated wins is still unsound inference.
