# Post-fast-config remediation contract

Status: governing implementation contract

Review basis: `docs/reviews/Farkle_II_post_fast_config_review.md`

Reviewed revision: `6be5f5fa11df77155621bfc81188c7515f38f8de`

Contract task: Task 0

## 1. Authority and scope

This document reconciles the final adversarial synthesis with the accepted user
decisions. It governs Tasks 1-11 below. If an implementation, configuration,
test, older document, or existing artifact disagrees with this document, this
document controls until it is superseded by an explicit recorded decision.

The completed fast-run tree reviewed in the synthesis is known-bad regression
evidence. No remediation task may read it as an input, repair it, rewrite it,
delete it, or authenticate its bytes retroactively. A later audit may inspect it
read-only only when that audit is explicitly scoped to regression evidence.

The existing estimands remain unless this contract says otherwise: tournament
performance remains per player-game exposure with the existing across-k rule;
seat-adjusted H2H remains `d_AB = 0.5(q_AB - q_BA)`; and H2H remains a
two-player, frozen-family, completed-game analysis. The outcome contract below
changes the tournament denominator from the implementation's fabricated-winner
behavior to the accepted per-attempt denominator. That change is intentional
and requires version invalidation and full regeneration.

### Task ownership

| Task | Owner scope |
| ---: | --- |
| 0 | This governing contract |
| 1 | Outcome and safety-limit semantics (B1) |
| 2 | Full-coordinate RNG scheme v2 (H1) |
| 3 | Authenticated lifecycle and provenance (B2) |
| 4 | Metric-only resume commit ordering (H2) |
| 5 | Canonical game-stat seat selection (H3) |
| 6 | Artifact scope, schema, method, and source authentication (H4) |
| 7 | Strict public CLI parsing (H5) |
| 8 | Actual end-to-end oracle and clean-fast audit (H6) |
| 9 | H2H planning, cell validation, and recovery (M1-M3) |
| 10 | Diagnostic method and claim corrections (M4-M6) |
| 11 | Risk probes and optional hardening (P1-P3) |

## 2. Safety-limit outcome schema

### 2.1 Closed outcome states

Every successfully persisted attempted game has exactly one non-null
`termination_status` value from this closed enum:

- `completed`: normal game rules produced a final-round result and exactly one
  winner.
- `safety_limit`: the configured maximum-round limit was reached without normal
  completion.

An exception, killed worker, corrupt row, or missing result is not a third game
outcome. It is incomplete execution and must be retried or leave the owning
coordinate pending. The name `aborted` may appear only as a clearly documented
legacy/reporting alias for `safety_limit`; it is not a canonical enum value.

For `completed` rows:

- `winner_seat` and `winner_strategy` are non-null;
- exactly one participant has `rank == 1`;
- all `k` ranks are non-null and are the permutation `1..k` under the existing
  completed-game ranking rule; and
- winner-dependent fields such as winning score and victory margin are
  non-null.

For `safety_limit` rows:

- `winner_seat`, `winner_strategy`, winning score, victory margin, every
  participant rank, and every winner-conditioned field are null;
- no participant has rank 1, including when one score happens to be larger;
- a compact rank representation retains `k` null entries rather than inventing
  an order or dropping participants; and
- final scores, turns, rolls, farkles, round count, seated strategies, and the
  full semantic RNG coordinate remain recorded because the attempt occurred.

The safety limit is not a draw. It is an attempted game in which every
participant loses. No stable sort, seat order, score order, or display
tie-break may create a winner or rank.

### 2.2 Canonical counts and conservation

For any closed set of games at one concrete `k`, define:

- `games_attempted = A`;
- `games_completed = C`;
- `games_safety_limit = S`;
- `attempted_exposures = E`;
- `completed_exposures = E_C`;
- `safety_limit_exposures = E_S`;
- `wins = W`; and
- `losses = L`.

The required integer identities, with zero tolerance, are:

```text
A = C + S
E = k*A
E_C = k*C
E_S = k*S
E = E_C + E_S
W = C
L = E - W = (k - 1)*C + k*S
```

For each strategy `i`, counts are player-game exposures, not distinct games:

```text
E_i = E_C_i + E_S_i
L_i = E_i - W_i
0 <= W_i <= E_C_i <= E_i
sum_i(E_i) = k*A
sum_i(E_C_i) = k*C
sum_i(E_S_i) = k*S
sum_i(W_i) = C
```

The same identities must hold additively by root, k, deterministic batch, and
every canonical aggregation that claims complete support. Counts must remain
explicit; `games_completed` must never be used to mean attempted games.

### 2.3 Tournament rates

The canonical tournament win rate is the per-attempt rate:

```text
win_rate = wins / attempted_exposures
```

This is the primary rate used by per-k performance, chance deltas, batch MCSE,
across-k performance, screening, and the win-rate candidate contribution.
Safety-limit attempts therefore penalize every seated strategy. The existing
`1/k` chance baseline and existing tournament, equal-k, and seat-adjusted
estimands otherwise remain unchanged.

An optional diagnostic may report:

```text
win_rate_completed_only = wins / completed_exposures
```

It must be named and described as conditional on completed games, must never be
aliased to `win_rate`, and is null rather than zero when
`completed_exposures == 0`.

Required safety reporting includes `games_attempted`, `games_completed`,
`games_safety_limit`, `completion_game_rate = C/A`,
`safety_limit_game_rate = S/A`, attempted/completed/safety-limit exposure
counts, and strategy-level
`safety_limit_exposure_rate = E_S_i/E_i`. Rates with a zero denominator are
null with an explicit reason. Reports show these by root and k and, where
relevant, by strategy, deterministic batch, H2H pair, root, and seat order;
pooled totals may not hide a failing cell.

## 3. Consumer policy

### 3.1 All-player metrics and game statistics

Unconditional all-player metrics retain exactly one observation per seated
player per attempted game, including every participant in a safety-limit game.
Score, turns, rolls, farkles, return metrics, maximum-round exposure, and other
defined player measurements use all attempted exposures unless their field name
and sidecar explicitly state another conditioning set. Every safety-limit
exposure is a loss.

Unconditional game statistics retain exactly one observation per attempted
game. Game length and other termination-independent measurements include
safety-limit games and report termination strata. Winner-, rank-, margin-, and
winning-score-conditioned statistics use completed games only and carry the
exact conditioning string `termination_status == "completed"`. A null winner
field may not be imputed for grouping.

Strategy-conditioned game statistics use only anchored canonical seat columns
matching `^P[1-9][0-9]*_strategy$`. Their observational unit is one seated
strategy exposure per attempted game. `winner_strategy` is never a seat column.

### 3.2 TrueSkill

TrueSkill performs no update for a safety-limit game. It consumes completed
games only, preserves their chronological/declared sequential order, and
records attempted, completed, and excluded safety-limit counts for every
root/k cell.

The exact conditioning language in TrueSkill artifacts, sidecars, and reports
is:

> Descriptive TrueSkill screening conditional on games that completed under
> the configured safety-round limit; safety-limit attempts are excluded from
> rating updates and are reported separately. This is not the canonical
> per-attempt tournament win-rate estimand.

The percentile candidate contribution remains descriptive and inherits this
conditioning. It cannot override the primary per-attempt tournament rate or
support formal inference.

### 3.3 Candidate viability

Candidate selection first freezes the existing union-of-contributors family;
termination evidence must not silently remove a selected member or shrink the
multiplicity family. Viability is then an explicit, auditable status.

A pair is inferentially viable only when every planned root/order cell reaches
its required completed-game target within its maximum attempts. A candidate is
globally viable for a candidate-level H2H claim only when every incident pair
is inferentially viable. Nonviable candidates remain in the frozen family and
reports; they cannot be declared unique best. The unresolved edges they induce
remain unresolved rather than being treated as losses, ties, equivalence, or
evidence for another candidate.

**Decision D1 - accepted production completion-rate threshold.** Accepted on
2026-07-22: `head2head.min_candidate_completion_rate = 0.99`, evaluated over
all H2H attempts incident to a candidate and also reported separately by
pair/root/order. Falling below it marks the candidate
`operationally_nonviable` for a candidate-level production claim even if
individual pairs reached their formal completed targets; it does not discard
valid pairwise completed-game inference. A lower threshold would permit more
safety-limit-heavy candidates and workload; a higher threshold would withhold
more candidate-level claims. The accepted value is immutable design state,
participates in freshness, and cannot change after family/plan publication.

### 3.4 H2H attempts, replacements, and inference

The immutable power plan targets `n_completed_required` completed games in
each `(pair_id, root_seed, order)` cell. Power and formal inference condition
on completed games only.

Each attempted H2H game has the immutable coordinate:

```text
(rng_scheme_version, purpose, root_seed, pair_id, order, attempt_index)
```

`attempt_index` starts at zero and increases by one within its cell. The first
`n_completed_required` indices are initial attempts. Any later index is a
deterministic replacement attempt. Replacement scheduling always chooses the
smallest not-yet-authenticated attempt index; it never draws a replacement seed
from a prior RNG and never renumbers completed attempts. Resume reconstructs
the next index from authenticated attempt coordinates, not a counter that can
advance ahead of durable results.

Every cell records `games_attempted`, `games_completed`,
`games_safety_limit`, `wins_seat1`, and `wins_seat2`, with:

```text
games_attempted = games_completed + games_safety_limit
wins_seat1 + wins_seat2 = games_completed
```

Execution stops when the completed target is reached or the cell's declared
attempt cap is exhausted. No extra attempts are made after the target.

**Decision D2 - accepted maximum-attempt multiplier.** Accepted on 2026-07-22:
`head2head.max_attempt_multiplier = 2.0`, giving
`max_attempts = ceil(2.0 * n_completed_required)` per root/order cell. The
multiplier must be at least 1, is frozen in the design, and is applied before
execution. A larger value increases the chance of reaching completed support
and increases worst-case workload; a smaller value resolves sooner as
nonviable. The accepted value participates in freshness and cannot change
after plan publication. The sum of per-cell maximum attempts must also fit the
separately declared total-game authorization; if it does not, execution enters
`blocked_by_cap` before attempting any H2H game and the immutable statistical
design remains unchanged.

If any cell exhausts its cap before reaching the target, the unordered pair is
`unresolved_nonviable`. It receives no formal effect, p-value, dominance,
equivalence, or directional claim. Partial completed counts and their plainly
labelled descriptive rates may be reported but are not fed to the formal
procedure. With zero completed games, `q_AB`, `q_BA`, `d_AB`, standard errors,
intervals, and p-values are null; no zero, one-half, or perfect-win value is
imputed.

Multiplicity is fixed before outcomes are observed. The Holm family and the
Bonferroni simultaneous-bound family contain every unordered pair in the
frozen candidate family, including unresolved/nonviable pairs. Operationally,
unresolved p-values are treated as non-rejections with family size unchanged;
they are not removed to obtain a smaller correction. Equivalence still
requires an explicitly configured margin, completed support in every cell,
and simultaneous interval containment. Nonsignificance is never equivalence.

## 4. RNG scheme v2

### 4.1 Direct semantic-coordinate streams

RNG scheme v2 uses NumPy `PCG64DXSM` initialized directly from a
`SeedSequence` whose entropy is the version, a permanent purpose namespace,
and every full-width semantic coordinate. Each integer is encoded without
loss. No generated `uint32`, `uint64`, hash prefix, table seed, game seed, or
other reduced scalar may become a new RNG root.

The required tournament identities are:

| Stream/identity | Purpose namespace | Semantic fields |
| --- | --- | --- |
| Shuffle ownership | `tournament_shuffle` (100) | `root_seed, k, shuffle_index` |
| Shuffle permutation | `shuffle_permutation` (101) | `root_seed, k, shuffle_index` |
| Game identity | `tournament_game` (102) | `root_seed, k, shuffle_index, game_index` |
| Seat/player dice | `tournament_player` (103, new) | `root_seed, k, shuffle_index, game_index, seat_index` |

The required H2H identities are:

| Stream/identity | Purpose namespace | Semantic fields |
| --- | --- | --- |
| Pair design | `h2h_pair` (200) | `root_seed, pair_id` |
| Seat order | `h2h_order` (201) | `root_seed, pair_id, order` |
| Attempt/game identity | `h2h_game` (202) | `root_seed, pair_id, order, attempt_index` |
| Seat/player dice | `h2h_player` (203, new) | `root_seed, pair_id, order, attempt_index, seat_index` |

`k=2` is stored in H2H provenance but the H2H-specific namespace prevents a
tournament/H2H collision. Namespace integers already published by v1 remain
reserved forever. New namespace integers 103 and 203 may not be reused.
Strategy generation, bootstrap, HGB, diagnostic, and display namespaces retain
their permanent values and must also use direct semantic coordinates.

Scalar `shuffle_seed` and `game_seed` fields may remain only as display or
legacy diagnostic fingerprints. They are non-authoritative, collision-tolerant,
and never identify ownership, freshness, replay, or a child stream. Canonical
ownership uses the complete coordinate tuple.

### 4.2 Version and compatibility rules

The remediation version registry is:

| Identity | Required value after implementation | Compatibility effect |
| --- | ---: | --- |
| `rng_scheme_version` | 2 | All simulation/H2H outcomes and descendants from v1 are stale |
| `outcome_schema_version` | 2 | Rows without canonical termination/nullability are stale |
| `tournament_method_version` | 2 | Fabricated-winner/per-completion rate artifacts are stale |
| `h2h_method_version` | 2 | Attempt-unaware or non-completed inference artifacts are stale |
| `artifact_contract_version` | 3 | Sidecars without semantic schema/source authentication are stale |
| `schema_version` | 2 | Existing derived schemas are incompatible |
| `estimand_version` | 2 | Existing tournament outcome summaries are incompatible |
| `conditioning_version` | 2 | TrueSkill/H2H conditioning must be explicit |

Names may be represented as typed method-contract fields rather than top-level
configuration keys, but every value above must occur in the canonical
freshness identity. Compatibility is equality, not `>=`. Unknown or absent
versions fail closed. No adapter may label v1 outcomes as v2.

### 4.3 Resume and worker-count invariants

For one public configuration and code identity, changing worker count,
start method, chunk size, task completion order, checkpoint interval, or
interruption/resume timing must not change:

- the set of semantic attempt coordinates;
- the stream or outcome at any coordinate;
- deterministic batch ownership;
- the logical rows and sufficient statistics; or
- canonical sorted artifact bytes and hashes where the format is defined as
  canonical.

Execution timestamps and progress logs may differ and live only in mutable
execution state. A checkpoint may claim a coordinate only after its row or all
of its sufficient statistics are durably and atomically committed. Resume
rejects a checkpoint whose design/freshness identity differs.

## 5. Authenticated lifecycle

### 5.1 Canonical stage identity

Every simulation cell and root/pair analysis stage has one freshness identity:

```text
stage_identity = SHA256(canonical_json({
  lifecycle_contract_version,
  stage_key,
  stage_cache_key_version,
  stage_config_identity,
  method_versions,
  rng_scheme_version,
  outcome_schema_version,
  schema_and_estimand_versions,
  code_identity,
  upstream_identities,
  immutable_design_identities
}))
```

Canonical JSON uses UTF-8, sorted keys, normalized path-independent logical
identifiers, and a specified numeric representation. Missing identity fields
are errors. Size and mtime may aid diagnostics but never freshness.

`stage_config_identity` is the SHA-256 of the canonical public configuration
fields that can affect that stage's logical result plus inherited statistical
dependencies. The stage registry owns the field allowlist. Semantic changes
stale the stage; runtime-only controls such as worker count, chunk size,
checkpoint frequency, log level, and temporary paths do not. Output locations
are authenticated separately as physical artifact identities.

### 5.2 Code identity and dirty trees

Production and release-evidence runs require a clean Git worktree and a full
40-character commit SHA. Failure to determine the commit or any tracked,
staged, or untracked repository change makes the run ineligible for release.
The clean-tree check and commit are recorded before execution and rechecked
before final release audit.

Dirty-tree development runs are permitted only by an explicit non-production
opt-in. Their code identity is the HEAD SHA plus a deterministic digest of all
tracked diffs and all untracked executable/project files in the declared code
inventory. They are permanently labelled `development_dirty` and cannot be
promoted, blessed, or used as release evidence after the fact. `unknown` is
never a valid code revision.

### 5.3 Upstream, output, and sidecar identity

Each ordinary upstream artifact is bound by logical role, canonical scope,
relative canonical path, exact content SHA-256, exact sidecar SHA-256, and the
sidecar's validated contract identity. A path, size, mtime, or self-hash alone
is insufficient.

Large sharded inputs may use an authenticated immutable manifest root instead
of listing every file in every child stamp. The root is computed over a
canonical coordinate-sorted list of `(logical coordinate, canonical relative
path, data SHA-256, sidecar SHA-256, schema fingerprint)`. The manifest bytes,
entry count, coordinate support, and root digest are themselves authenticated.
Changing, adding, deleting, duplicating, or reordering a logical entry changes
or invalidates the root.

Every output has:

1. an artifact identity binding canonical physical scope/path, byte length,
   exact content SHA-256, Arrow/JSON schema fingerprint including field order,
   types and nullability, and logical operation; and
2. exactly one adjacent sidecar binding the stage identity, method contract,
   conditioning, support, upstream identities, and artifact identity.

The final completion stamp contains the stage identity and a canonical list of
every required output's artifact and sidecar hashes. Publication order is data,
sidecar, then completion stamp, using atomic replacement. A missing sidecar may
be finalized only from an independent valid completion identity that already
binds the exact output bytes. A present mismatched sidecar or changed output is
stale/corrupt and must fail or recompute; existing bytes are never
re-authenticated merely because they exist.

### 5.4 Immutable design and mutable execution

Candidate family, power assumptions, completed-game targets, root/order
allocation, maximum-attempt policy, multiplicity family, equivalence margin,
RNG/method versions, and schedule coordinates are immutable design. Once
published, their bytes and hashes never change. Raising an operational workload
authorization must not rewrite them.

Mutable execution state is separate and contains only such facts as lifecycle
state, authorized cap, completed/pending coordinate roots, counts, checkpoint
generation, and timestamps. It binds the immutable design hash. It is atomic
and resumable but is not an input to the statistical estimand. Invalid block
data/sidecar pairs leave only that coordinate pending; they are regenerated
deterministically and never blessed.

### 5.5 Public configuration and runtime context

`active_config.yaml` contains only public declared configuration fields. It
must round-trip through the public configuration loader to the same canonical
public config hash. Private dataclass fields, resolved stage layouts, path
overrides, object representations, and runtime-only state are forbidden.

A separate authenticated run-context manifest records resolved root/pair
paths, selected roots, parent public-config hashes, stage layout identity,
command and CLI overrides, code identity, environment identity needed for
audit, and execution controls. The context points to the public configuration;
it does not mutate or masquerade as that configuration. Root-pair context must
bind both exact parent lifecycle roots and reproduce the declared pair/context
hashes on reload.

## 6. Finding disposition and traceability

`Gate` means the item must be fixed, disabled, or explicitly withheld as
specified before the affected release claim. A conditional gate does not imply
that the known-bad tree should be repaired.

| Finding | Disposition and owner | Hand-checkable acceptance test | Regeneration boundary | Release gate |
| --- | --- | --- | --- | --- |
| B1 | Fix under Task 1 using Sections 2-3. | Force one 0-0 max-round game through engine, tournament metrics, H2H, and reporting: `A=1,C=0,S=1`, all `k` lose, winner/ranks null, no H2H inference; add a mixed completed/safety hand oracle. | Full simulation, all root analyses, family selection, all H2H, inference, and reports. | Yes. |
| B2 | Replace existence/mtime lifecycle under Task 3 using Section 5. | Independently mutate one config value, upstream byte, method parameter, code identity, output byte, and sidecar; each becomes stale/recomputes, unchanged skips, `--force` recomputes, and public/pair configs round-trip. | Fresh full run for release; old bytes cannot acquire provenance retroactively. | Yes. |
| H1 | Replace scalar-root tournament/H2H streams under Task 2 using Section 4. | Enumerate production-scale semantic coordinates, include a deliberate v1 32-bit collision, and prove distinct v2 streams plus identical logical results across worker counts and resume. | Full simulation and all descendants, including finalist/H2H tail. | Yes. |
| H2 | Commit statistics before ownership under Task 4. | Interrupt a two-shuffle metric-only run with row/metric-chunk outputs disabled; resumed and uninterrupted checkpoints match every sum, square sum, win, and coordinate. | Only affected interrupted metric-only runs; superseded by fresh fast run. | Yes if metric-only mode remains enabled; otherwise that mode must fail closed. |
| H3 | Anchor seat columns and observational unit under Task 5. | Two hand games yield each strategy's seat exposures exactly, never exposures plus wins, through per-k, rare-event, concat, and across-k outputs. | Game-stat stages and their consumers; superseded by full rerun. | Yes for publishing game-stat outputs. |
| H4 | Enforce physical scope, real schema, method parameters, and sources under Task 6. | Wrong directory, absent/wrong column, type/nullability mismatch, changed source, wrong family/schedule/multiplicity hash, and mixed-scope payload all fail. | Relocate/regenerate affected artifacts and sidecars; no simulation rerun solely for metadata, but B2 requires a fresh run. | Yes. |
| H5 | Use strict command-owned parsing under Task 7. | Installed CLI accepts documented post-subcommand roots 42/43; arbitrary unknown option exits nonzero before creating output. | None for known-correct invocations; audit/regenerate runs relying on ignored options. | Yes. |
| H6 | Add actual pipeline oracle/audit under Task 8. | Tiny configured two-root CLI run uses real stages from simulation through report and matches hand counts, hashes, orientation, lifecycle, unresolved behavior, and claim text. | None; evidence gate only. | Yes. |
| M1 | Replace unproved monotone binary search under Task 9. | Brute-force the admitted small-n range and recover the first crossing, including the review's `n=1` versus returned `n=14` counterexample. | Power plan/schedule/H2H only when allocation changes; current 1,974 allocation needs no change for M1 alone. | Conditional: implementation gate before production planning; no current-count regeneration solely for M1. |
| M2 | Validate every cell against frozen manifest under Task 9. | Reject compensating `(100,200)/(200,100)` root/order imbalance; accept exact Cartesian support and seat mapping. | Recompute malformed H2H aggregate/inference only. | Yes before formal H2H release. |
| M3 | Separate immutable plan/authorization and coordinate recovery under Task 9. | Raising authorization leaves design bytes/hash unchanged; interruption between block data and sidecar replays exactly that coordinate and never blesses bytes. | Only overwritten plan or unauthenticated block work. | Yes for resumable H2H release. |
| M4 | Remove unadjusted significance claims under Task 10. | Root-stability artifacts/reports contain no significance/rejection labels when ordinary intervals exclude zero; descriptive estimates remain. | Root-stability artifacts and reports. | Conditional: affected artifact must be fixed or withheld. |
| M5 | Correct globally ordered RNG diagnostic under Task 10. | A sequence split across Arrow batches/seats matches a one-frame coordinate-sorted oracle; health cannot be `complete_success` with stale stage hashes. | RNG diagnostics and health, after v2 rerun. | Conditional: required for a complete-success clean-fast claim. |
| M6 | Use a model-consistent predictor or rename under Task 10. | Equal-mu/different-sigma and different-mu/equal-sigma fixtures distinguish a model-consistent method; alternatively exact `mu_softmax_heuristic` labels contain no calibration claim. | TrueSkill diagnostic outputs if estimator changes; metadata/reports if renamed. | Conditional: calibration claims are gated; percentile screening is not. |
| P1 | Investigate and harden ingest under Task 11. | Swapped shard, duplicate game key, wrong internal shuffle, invalid outcome, missing identity, repeated seated strategy, and bad margin all fail streaming validation. | Only inputs/audits found malformed. | Risk probe; becomes a gate if evidence finds a violation. |
| P2 | Resolve sparse support semantics under Task 11. | A declared zero-exposure row has the documented no-effect/failure result; a missing strategy/batch cell fails when rectangular support is required. | None for the rectangular fast design; affected sparse analyses only. | Decision/evidence risk, not a current fast-run gate. |
| P3 | Add production-JIT and ID-boundary evidence under Task 11. | Clean subprocess runs hand-checkable compiled scoring and tiny seeded simulation; nullable, nonnumeric, and mixed-ID boundaries fail canonical validation. | Only if compiled behavior or canonical IDs change. | Hardening risk; becomes a gate on any observed divergence/coercion. |

## 7. Release readiness

A release is ready only when all of the following are true:

1. Tasks 1-8 and the release-gating portions of Task 9 are implemented and
   accepted. Task 10 outputs are fixed or explicitly withheld according to the
   table. Any Task 11 probe that discovers a violation is resolved before
   release. Configuration and method identities match the accepted D1 and D2
   values recorded in Sections 3.3 and 3.4.
2. Focused tests for every gated finding pass, followed by Ruff, Pyright,
   Mypy for `src`, the full hermetic test suite, the compiled-Numba subprocess
   check, and the actual tiny two-root pipeline oracle. The terminology and
   coverage gates must be hermetic and enforce their documented thresholds.
3. The release revision is a recorded, clean 40-character Git commit before
   and after the run. No sidecar or stamp contains `unknown`, a dirty identity,
   or a null required config/code/source hash.
4. `configs/fast_config.yaml` is materialized with the accepted new fields and
   run once from an empty, newly declared output root. The resolved pair/root
   directories must not have existed at start. The known-bad reviewed tree is
   neither an input nor an output and remains untouched.
5. The exact public config bytes/hash, reload round-trip, CLI command and
   overrides, code identity, environment/package lock identity, start/end
   times, resolved output paths, and root/pair run-context hashes are retained.
6. Every stage finishes `complete_valid`; none is `complete_stale`,
   `partial_resumable`, or `blocked_by_cap`. If H2H reaches an attempt cap, the
   execution stage may still finish valid only when the affected pairs and
   candidates are explicitly `unresolved_nonviable`, all formal fields are
   null, and no downstream claim consumes them. Top-level health must describe
   that substantive outcome rather than conceal it.
7. A read-only release audit validates every canonical physical scope, real
   schema/type/nullability fingerprint, exact upstream/manifest root, method
   and multiplicity parameter, output hash, sidecar hash, completion stamp,
   and immutable family/schedule identity with zero unexplained violations.
8. Independent recounts from clean rows satisfy every Section 2 identity with
   integer tolerance zero at global, root, k, batch, strategy, and H2H
   pair/root/order levels. Reports show safety-limit counts/rates and label all
   conditioning exactly.
9. Independent statistical oracles reproduce the primary per-attempt
   tournament rates, equal-k aggregation, completed-only TrueSkill filtering,
   completed-game H2H counts/effects, fixed-family Holm adjustment,
   Bonferroni practical bounds, and unresolved/equivalence guards within the
   declared numerical tolerances.
10. Worker-count and forced interruption/resume comparisons use a separate
    clean fixture and produce identical coordinate manifests, logical outputs,
    and canonical hashes. No production-scale pipeline is required for this
    evidence.

The clean fast run is an integration and release oracle, not production-scale
scientific evidence. Only after all evidence above passes may a production run
be authorized. Existing v1/implicit-outcome artifacts are stale at the
simulation boundary; no existing completed fast-run artifact can be carried
forward.

## 8. Implementation sequence and next task

The exact next task unblocked by this contract is **Task 1: implement the
safety-limit outcome schema, null winner/rank behavior, per-attempt tournament
accounting, completed-only consumer conditioning, H2H attempt accounting, and
the focused hand oracles in Sections 2-3**, including the accepted D1 value.
Task 2 then changes the RNG root construction; Task 3 installs the lifecycle
that makes both changes invalidate old work. Task 9 must implement the accepted
D2 value when it freezes the H2H plan. No task may use the known-bad completed
fast-run tree as a migration source.
