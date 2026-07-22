# Adversarial post-remediation review: orchestration and reproducibility

## Scope and verdict

Reviewed commit `6be5f5fa11df77155621bfc81188c7515f38f8de` and the completed
`configs/fast_config.yaml` run at `data/results_seed_pair_32_33`. This was a
review-only examination of pipeline control flow, resource budgeting,
deterministic RNG, atomicity, interruption recovery, and cache validity. No
application code, test, configuration, or existing run artifact was changed.

**Verdict: unsound within this review's scope.** The canonical root/pair order,
H2H coordinate recovery, atomic publication, and most RNG namespace design are
well constructed. However, a two-seed no-force run treats simulation completion
as the mere existence of six marker files, root TrueSkill can both reuse stale
results and re-authenticate corrupted bytes, tournament game coordinates are
collapsed to collision-prone 32-bit seeds, and neither artifacts nor stage
freshness bind the checked-out code revision. These are reproducibility defects,
not merely missing hardening.

## Run trace and controls that passed

- The latest successful manifest invocation contains each root stage exactly
  once and in this order for each root: `ingest`, `curate`, `combine`, `metrics`,
  `game_stats`, `rng_diagnostics`, `trueskill`, `hgb`, `screening`. Its one pair
  workflow contains: `root_stability`, pair TrueSkill contribution,
  `candidate_freeze`, `h2h_power`, `h2h_execute`, `h2h_inference`, `h2h_digest`,
  `agreement`, `reporting`. This agrees with
  `src/farkle/analysis/__init__.py:71-118,247-307` and
  `src/farkle/orchestration/two_seed_pipeline.py:143-173,248-356`. The manifest
  also records an earlier interrupted/failed pair attempt and later retries;
  no invocation contains a duplicated stage.
- With `orchestration.parallel_seeds=false`, the run was serial across roots.
  Its manifest records 16 detected cores and policies of 12 simulation workers,
  3 ingest workers, and 12 analysis workers. The division logic is explicit at
  `src/farkle/orchestration/two_seed_pipeline.py:86-122`; the pair H2H call
  inherits `analysis.n_jobs` through `src/farkle/analysis/__init__.py:142-143`.
  Thus the actual serial run did not multiply the 12-worker budget across two
  simultaneous roots.
- The frozen family contains 76 finalists. The power plan records 2,850
  unordered pairs, 1,974 games per root/order block, 11,400 blocks, and
  22,503,600 projected games under the 100,000,000-game cap. The schedule and
  results each contain exactly 11,400 unique `(pair_id, root_seed, order)` and
  `block_id` values; every `games_completed` equals `games_required`. The family
  hash is `07a2f716...41733` and schedule hash is
  `b0fd81bb...b3c09` throughout the plan, schedule, execution state, and result
  table. The execution state reports 11,400/11,400 complete blocks.
- H2H plan identity deliberately excludes the operational cap while including
  family, roots, orders, effect, multiplicity, power target, RNG version, and
  method IDs (`src/farkle/analysis/h2h_schedule.py:423-452`). Raising only the
  cap can therefore publish the same frozen schedule and resume it without
  changing the statistical claim. A blocked plan is written before execution
  and receives `blocked_by_cap` state
  (`src/farkle/analysis/h2h_schedule.py:528-709`).
- H2H block RNG uses root, `k=2`, pair, order, and game coordinates
  (`src/farkle/analysis/h2h_schedule.py:784-824`). Existing blocks are checked
  against sidecar hashes plus block/family/schedule identities, aggregation is
  sorted by pair/root/order, and a missing final execution stamp can be rebuilt
  after authenticating completed state, aggregate, and every block
  (`src/farkle/analysis/h2h_schedule.py:841-969,1008-1188`). This prevents normal
  interruption/resume from duplicating or reordering H2H observations.
- Tournament shuffle, permutation, and game coordinates include root and `k`;
  bootstrap streams include root, `k`, and replicate. This supplies distinct
  cross-`k` RNG domains and supports the declared independent-`k` variance
  assumption (`src/farkle/utils/random.py:63-121`,
  `src/farkle/simulation/run_tournament.py:174-201`, and
  `src/farkle/analysis/performance.py:312-329`). Parallel tournament metric
  reductions are explicitly sorted by chunk index
  (`src/farkle/simulation/run_tournament.py:696-715`), and ingest restores
  manifest rows to shuffle-index order (`src/farkle/analysis/ingest.py:111-179`).
- Data files, completion stamps, health files, and execution state use same-
  directory temporary files and atomic replacement. Artifact publication first
  invalidates an old sidecar, replaces data, then publishes the new hash-bound
  sidecar (`src/farkle/utils/artifact_contract.py:477-529`); an interruption in
  that window is fail-closed as a missing sidecar. Manifests use locked,
  fsynced, single-record append (`src/farkle/utils/manifest.py:77-166`). No
  leftover `._tmp_`, `.part`, or `.partial` files were found in the completed
  run.
- Current consumers use canonical explicit paths. A source search found retired
  pooled/weighted/meta-analysis names only in migration or release audit rules,
  not current readers.

## Findings

### 1. Simulation completion is existence-only and can silently attach a new configuration to old or incomplete simulations

**Severity: Blocker. Confidence: High.**

**Classification:** Confirmed defect.

**Evidence.** `simulation_is_complete` returns only
`simulation_done_path(...).exists()` (`src/farkle/simulation/runner.py:231-239`),
and `seed_has_completion_markers` merely applies that predicate to configured
player counts (`src/farkle/orchestration/seed_utils.py:57-59`). The two-seed
workflow writes the active configuration first, then skips simulation whenever
those marker filenames exist (`src/farkle/orchestration/two_seed_pipeline.py:192-200`).
It never validates marker schema, root, player count, RNG version, strategy-grid
digest, workload plan, checkpoint metadata, listed output existence, or output
bytes on this path.

The actual `results_seed_32/2_players/simulation.done.json` lists six paths and
basic counts but contains no configuration hash, strategy-manifest hash, code
revision, input/output content hashes, or sidecar identities. A read-only probe
changed the in-memory score-threshold grid from the fast config to `[999]`;
`seed_has_completion_markers` returned `true` both before and after the change.
The adjacent active config would nevertheless be replaced with the changed
configuration before the skip.

The deeper resume validator does compare checkpoint metadata, expected strategy
manifest contents, RNG version, workload details, and manifest coordinates
(`src/farkle/simulation/runner.py:474-675`), but it is unreachable when all done
markers exist. It also cannot protect a missing output referenced by an
existence-only marker.

**Consequence.** A changed strategy grid, simulation method, workload setting,
or damaged/deleted output can be presented as a completed simulation under a
new `active_config.yaml`. Downstream stages may recompute from the old rows and
produce apparently current sidecars. This breaks the conditioning claim and
the fundamental idempotent/resumable provenance boundary.

**Smallest reasonable remediation.** Replace the simulation marker with the
same versioned lifecycle contract used by analysis: bind a simulation-scoped
configuration hash, strategy-manifest digest, workload-plan/method versions,
RNG scheme, code revision, and content identities for every canonical output or
shard manifest. Validate all of it before skipping. A missing/stale final stamp
should enter authenticated `partial_resumable` recovery; it must not imply that
existing bytes are valid merely because a filename exists.

### 2. Tournament game RNG coordinates are reduced to 32 bits and collisions occur in the completed fast run

**Severity: High. Confidence: High.**

**Classification:** Confirmed defect.

**Evidence.** A full tournament game coordinate is reduced with
`coordinate_seed(..., dtype=np.uint32)` at
`src/farkle/simulation/run_tournament.py:192-201`. That 32-bit value becomes the
new `root_seed` from which all players in `_play_game` derive their streams
(`src/farkle/simulation/simulation.py:325-384`). Distinct
`(root, k, shuffle_index, game_index)` coordinates can therefore share the same
player RNG streams within a `k` cell.

This is not hypothetical. Recalculating duplicate `game_seed` values from the
canonical curated rows found 12 duplicate excess values:

| root | k | games | duplicate excess |
| ---: | ---: | ---: | ---: |
| 32 | 2 | 172,000 | 2 |
| 32 | 4 | 86,000 | 2 |
| 32 | 5 | 68,800 | 0 |
| 33 | 2 | 172,000 | 5 |
| 33 | 4 | 86,000 | 1 |
| 33 | 5 | 68,800 | 2 |

All collided coordinate pairs were in different deterministic batches. For
example, root 32/k=2 seed `2963478802` occurs at shuffle/game/batch
`(194,18,4)` and `(4052,4,94)`. The shuffle resume identity is also reduced to
`uint32` (`src/farkle/simulation/run_tournament.py:635-648`) and is treated as
unique during duplicate detection and pending-work filtering
(`src/farkle/simulation/runner.py:540-594` and
`src/farkle/simulation/run_tournament.py:660-693`), although no shuffle-seed
collision happened in this fast run.

**Consequence.** Nominally distinct games share random streams, including
across deterministic batches used for MCSE. The observed fast-run fraction is
small, but collision probability grows quadratically with workload and the
scheme therefore does not scale to production Monte Carlo volumes. A future
shuffle-seed collision can also make interruption recovery reject a valid
manifest or omit a pending shuffle.

**Smallest reasonable remediation.** Do not use a truncated generated seed as
an intermediate identity. Derive player generators directly from the full
tournament coordinate, or at minimum carry a 64-bit coordinate seed for games
and shuffle identities. Treat this as an RNG-scheme change, increment the
scheme version, add collision/uniqueness tests at production-scale coordinate
counts, and invalidate affected simulations.

### 3. Root TrueSkill cache ignores its inputs and hyperparameters, and its recovery path can bless corrupted bytes

**Severity: High. Confidence: High.**

**Classification:** Confirmed defect.

**Evidence.** Per-`k` TrueSkill uses a private version-1 done stamp containing
only shard key, parquet path, row count, and creation time
(`src/farkle/analysis/run_trueskill.py:320-367`). If that stamp and parquet
exist, it skips without checking the curated-row identity, `beta`, `tau`, draw
probability, method version, code revision, or sidecar
(`src/farkle/analysis/run_trueskill.py:558-598`). Checkpoint recovery similarly
checks only the row pathname, not its bytes or TrueSkill environment
(`src/farkle/analysis/run_trueskill.py:600-614`).

After the skip, `publish_rating_cell_contract` catches *every*
`ArtifactContractError`, rereads the current parquet, and republishes it with a
new valid sidecar (`src/farkle/analysis/trueskill_screening.py:41-83`). A
temporary proof copied a fast-run rating artifact and its sidecar, changed one
`mu` value by 123 so validation failed, then called this function. Validation
subsequently passed and the altered value was preserved. Thus a content-hash
mismatch is handled as sidecar migration rather than corruption.

**Consequence.** Changes to TrueSkill settings or curated input can leave old
ratings in place, and valid-but-altered rating bytes can become freshly
authenticated. Those ratings feed the pair contribution, frozen finalist
family, and therefore the entire H2H family of claims.

**Smallest reasonable remediation.** Replace the private stamps with canonical
stage completion records binding the exact curated artifact and sidecar hash,
all TrueSkill parameters, ordered-input/method version, and code revision. Only
a provably complete immutable artifact with a known independent identity should
be eligible for missing-sidecar finalization. A present but mismatched sidecar
must fail or deterministically recompute the rating cell; it must never be
regenerated around the suspect bytes.

### 4. The run cannot be tied to the reviewed code, and most dependency links are not content-addressed

**Severity: High. Confidence: High.**

**Classification:** Confirmed provenance and cache-validity defect. Replacing
mtime fingerprints with content identities is also a hardening requirement for
adversarial filesystem changes.

**Evidence.** `make_artifact_sidecar` defaults `code_revision` to the literal
`"unknown"` (`src/farkle/utils/artifact_contract.py:272-341`). All 11,586
sidecars under `results_seed_pair_32_33` contain `code_revision: "unknown"`.
Stage configuration hashes include stage name, cache-key version, freshness,
and selected configuration, but no source revision or method implementation
digest (`src/farkle/config.py:1822-1843`). A source-code change with unchanged
configuration therefore leaves stage hashes unchanged.

Completion inputs are fingerprints of path, size, and `mtime_ns`, not content
hashes (`src/farkle/utils/stage_completion.py:126-154,236-243`). Sidecars bind
their own artifact bytes, but `source_artifacts` are normally only path strings
and many `input_manifest_hashes` are empty. For example, the root-combined
across-`k` and H2H block-manifest sidecars list source paths but no source
content hashes. The initial ingest sidecar hashes the row manifest, but that
manifest itself records shard path and row count without shard content hashes.

Pair provenance has an additional break: `dataclasses.replace` creates
`pair_base` without preserving the `init=False` `config_sha`, which is then
copied as `None` (`src/farkle/orchestration/run_contexts.py:149-160` and
`src/farkle/orchestration/run_contexts.py:48-82`). The actual pair
`active_config.done.json`, all pair completion stamps, and pair StageRunner
events therefore have `config_sha: null`; pair sidecars independently compute
`134bbb7f...dcb6a`, while the enclosing pipeline manifest records
`1909cc1f...48fcc`. No artifact records the reviewed commit hash or an explicit
relationship between those two configuration identities.

**Consequence.** The completed run proves that some compatible configuration
executed, but it cannot prove that commit `6be5f5f` produced the bytes. Code-only
method corrections will not invalidate caches. Restoring altered inputs with
their former size and timestamp can also evade dependency freshness.

**Smallest reasonable remediation.** Make a reproducible code identity
(release commit plus dirty-tree digest, packaged source hash, or equivalent)
mandatory in sidecars and stage keys. Carry an explicit run-to-derived-context
configuration lineage and assign the pair config hash after transformation.
Record and compare upstream artifact SHA-256 identities (preferably obtained
from validated sidecars) rather than only path/size/mtime.

### 5. HGB freshness ignores HGB configuration and method identity

**Severity: Medium. Confidence: High.**

**Classification:** Confirmed defect.

**Evidence.** Although the registry declares `cache_scope=("io", "hgb")`, the
HGB wrapper does not use `stage_is_up_to_date` or a completion stamp. It skips
solely when outputs and sidecars exist, output mtimes are no older than inputs,
and output self-hashes validate (`src/farkle/analysis/hgb_feat.py:34-91`). A
read-only probe constructed the fast root-32 context, changed
`hgb.max_depth` from 6 to 999, replaced the trainer with a call recorder, and
called `hgb_feat.run`; the trainer was not called.

**Consequence.** Changes to maximum depth, estimator count, held-out folds,
permutation repeats, or proposal limit can silently reuse results from the old
model while the active configuration states the new settings. HGB is
descriptive rather than causal, but its predictive-association artifacts are
still mis-provenanced.

**Smallest reasonable remediation.** Use the standard versioned HGB stage stamp
with validated inputs, outputs, `cfg.stage_config_sha("hgb")`, method version,
and code identity. Compare sidecar configuration/method expectations before a
skip.

### 6. The canonical lifecycle and dependency declarations are not uniformly enforced

**Severity: Medium. Confidence: High.**

**Classification:** Confirmed lifecycle-contract defect.

**Evidence.** The generic lifecycle resolver correctly distinguishes
`not_started`, `partial_resumable`, `complete_valid`, `complete_stale`, and
`blocked_by_cap` (`src/farkle/utils/stage_completion.py:157-251`). The H2H tail
also declares required outputs and completion stamps. In contrast, all root
`StagePlanItem`s have empty required outputs and no completion stamps
(`src/farkle/analysis/__init__.py:97-117`), so StageRunner treats a returning
action as success (`src/farkle/analysis/stage_runner.py:151-180`). HGB can log a
missing input and return (`src/farkle/analysis/hgb_feat.py:43-52`) while its
stage event is healthy. Simulation has its separate existence marker, and a
completed fast-run workload plan still reports `status: "not_started"` because
that field describes only the pre-execution plan.

The registry exposes `depends_on`, but every root and pair definition leaves it
empty (`src/farkle/analysis/stage_registry.py:28-39,103-225`). Dependency
validation in `resolve_stage_layout` is consequently inert. Actual ordering is
correct because it is duplicated manually in plan builders, not because a DAG
contract enforces it.

**Consequence.** Lifecycle state has different meanings by stage, health can be
green without a root-stage output contract, and accidental plan reordering or
omission is not caught by dependency declarations. Operators cannot uniformly
distinguish the five promised states across the whole workflow.

**Smallest reasonable remediation.** Give each root stage a stage-owned final
stamp and explicit required outputs, populate registry dependencies, and have
StageRunner validate the same output/sidecar contract for root and pair stages.
Represent simulation planning and execution as separate state fields so a
completed workload plan is not externally labeled `not_started`.

### 7. A missing H2H block sidecar is fail-closed but not resumable

**Severity: Medium. Confidence: High.**

**Classification:** Confirmed interruption-recovery defect.

**Evidence.** `_valid_existing_block` calls `validate_artifact_sidecar` without
catching `ArtifactContractError` (`src/farkle/analysis/h2h_schedule.py:841-865`).
The missing-final-stamp recovery path catches sidecar failure only for execution
state and the aggregate; its subsequent per-block loop is also uncaught
(`src/farkle/analysis/h2h_schedule.py:921-969`). Therefore a present block with
a missing or invalid sidecar aborts before it can be classified as pending and
deterministically regenerated. By contrast, a genuinely missing block path is
correctly returned as pending.

**Consequence.** One interrupted sidecar publication among 11,400 blocks can
make the frozen H2H schedule non-resumable without manual artifact surgery.
Failing closed is correct for authenticity, but failing to regenerate that one
coordinate is not.

**Smallest reasonable remediation.** Treat a missing/invalid block sidecar as
an invalid checkpoint coordinate, not as valid bytes and not as a fatal global
error. Re-run that frozen block to a temporary path and atomically replace both
data and sidecar. Do not attach a new sidecar to the unauthenticated existing
block.

### 8. Run-level configuration checks defeat stage scoping for root analysis

**Severity: Low. Confidence: High.**

**Classification:** Confirmed cache-efficiency defect.

**Evidence.** `resolve_stage_state` compares the full run `config_sha` before
the stage-scoped hash (`src/farkle/utils/stage_completion.py:203-220`). Root
stamps contain the full pipeline hash. A probe changed only
`analysis.log_level`; the ingest stage hash remained identical, but the actual
root-32 ingest stamp changed from `complete_valid` to `complete_stale` because
the run hash changed. Simulation, conversely, remained “complete” after a
strategy-grid change because of Finding 1.

**Consequence.** Analysis-only operational changes can trigger expensive root
re-ingest/reanalysis, while simulation-affecting changes can fail to invalidate
simulation. The scoped invalidation design does not achieve its stated purpose.

**Smallest reasonable remediation.** Use the full config hash as provenance,
not as a universal cache predicate. Cache validity should compare only the
versioned stage scope, validated upstream identities, method/code identity, and
freshness contract. Add paired tests proving that log/progress/worker changes do
not invalidate deterministic statistical bytes while strategy or simulation
method changes do invalidate simulation.

### 9. Canonical orchestration overrides both configured analysis and H2H worker settings

**Severity: Low. Confidence: High.**

**Classification:** Configuration/implementation disagreement.

**Evidence.** `fast_config.yaml` requests `analysis.n_jobs: 4` and
`head2head.n_jobs: 0`, but the orchestrator derives its analysis policy solely
from `sim.n_jobs` and writes that value back into each root config
(`src/farkle/orchestration/two_seed_pipeline.py:92-111,125-140`). Unlike ingest,
the analysis policy is not capped by its stage-specific configured value. The
H2H tail then explicitly passes `inner.analysis.n_jobs`, preventing
`execute_h2h_schedule` from consulting `head2head.n_jobs`
(`src/farkle/analysis/__init__.py:142-143` and
`src/farkle/analysis/h2h_schedule.py:1087-1090`). The actual manifest confirms
12 analysis/H2H workers, not 4 or independently resolved auto mode.

**Consequence.** The run stayed within the serial 12-process global budget, so
this did not oversubscribe the reviewed machine. It nevertheless makes two
documented configuration controls ineffective and can allocate more analysis
workers than the operator requested.

**Smallest reasonable remediation.** Define one explicit precedence rule. If
`sim.n_jobs` is a global ceiling, cap stage-specific requests beneath it (as
ingest already does) and resolve H2H's own setting beneath the remaining pair
budget. If inheritance is intentional, remove or deprecate the ineffective
settings and state the resolved policy in the persisted effective config.

### 10. RNG documentation omits an active permanent namespace

**Severity: Low. Confidence: High.**

**Classification:** Documentation and contract-checker defect.

**Evidence.** `RandomPurpose.ROOT_STABILITY_BOOTSTRAP = 401` is used by root
stability resampling (`src/farkle/utils/random.py:18-35` and
`src/farkle/analysis/root_stability.py:471-488,722-734`), but
`docs/rng_contract.md` lists 400 followed by 500 and claims its namespace table
is permanent. `scripts/check_rng_contract.py` passed and did not catch the
omission.

**Consequence.** The public RNG contract is incomplete, increasing the risk of
future namespace reuse or an unreviewed stream-identity change.

**Smallest reasonable remediation.** Add namespace 401 to the contract and make
the checker compare every `RandomPurpose` enum member/value against the
documented registry.

## Validation performed

- Passed 92 focused tests covering RNG utilities and contract enforcement,
  workload planning, tournament determinism/resume, stage lifecycle/runner/
  registry, H2H schedule lifecycle, and seed orchestration.
- `scripts/check_rng_contract.py` passed.
- Read-only artifact checks verified stage order, worker policies, completion
  state, family/schedule identity, H2H coordinate uniqueness, block completion,
  sidecar revision/config distributions, temporary-file absence, and actual
  game-seed collisions.
- Temporary proofs for TrueSkill corruption handling were created under the
  system temporary directory and removed automatically. No large simulation was
  launched.

Passing these controls confirms that the currently tested path executes and
that substantial parts of its resume design work. It does not mitigate the
confirmed cache-authentication and RNG-collision defects above.
