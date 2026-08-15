# Codex Context Prompt

Use this prompt in a new Codex conversation for this repository.

```text
You are working in Farkle Mk II, a deterministic Monte Carlo simulation and
statistical analysis project. Read AGENTS.md and the files under
docs/codex_context before broad source exploration. Treat those maps as a
verified orientation cache, then inspect the relevant source before editing.

Preserve these invariants:
- explicit coordinate-owned PCG64DXSM streams;
- atomic, idempotent, resumable work;
- AppConfig-owned paths and only the six canonical artifact scopes;
- exactly one compatible sidecar for each derived artifact;
- streaming or partitioned handling for large data;
- complete root/k support and explicit cross-k weights;
- explicit outcome-schema-v2 termination status: a safety-limit attempt has no
  winner or ranks, counts as a loss for every tournament participant, and is
  excluded from winner-conditioned products;
- tournament primary rates and batch MCSE use all attempted player-game
  exposures; completed-only rates are labelled diagnostics;
- separation of descriptive screening, inference, dominance, and display order;
- RNG lag diagnostics use authenticated external partitions: strategy seat
  exposures follow the full RNG-v2 tournament-player coordinate, matchup games
  follow the game coordinate without participant expansion, the compact matchup
  digest is only a partition/display accelerator and every identity comparison
  includes the sorted padded participant key, eligibility precedes fixed-ring
  allocation, deterministic cap exhaustion is blocked-by-cap, and zero-centered
  descriptive reference bands make no independence claim;
- two-root stability is fixed-design descriptive reproducibility: intervals are
  Monte Carlo precision, practical threshold positions are not tests, and joint
  discrepancy reference quantiles/frequencies are not multiple-testing inference;
- current code never reads old on-disk analysis artifacts;
- pair analysis lives under `results_seed_pair_X_Y/seed_pair_analysis`, path
  lookup is non-mutating, and every H2H phase owns its artifacts;
- explicit artifact-contract-v3 runs publish simulation through frozen
  candidate-family artifacts with canonical path/scope, actual Arrow or typed
  non-Parquet format, complete version/method, code/config, exact source or
  immutable-manifest identities, and authenticated completion; old completion
  schemas are stale and v2 bytes cannot be promoted or re-sidecarred;
- TrueSkill root/k completion binds ordered row bytes and rating sidecars, and
  HGB completion binds target/features, whole-strategy folds, model/RNG method,
  outputs, and sidecars; HGB per-k row unions live in `concat_ks`;
- active configs contain only reloadable public fields; resolved paths, layouts,
  parent lifecycle roots, code identity, and execution controls live in the
  separately authenticated `run_context.json`;
- public process-tree execution is launched inside one aggregate Windows Job
  Object or delegated cgroup-v2 `memory.max` boundary using an explicit aggregate
  hard limit; scheduler admission memory, RSS high-water warning, host-available
  reserve, parent-process estimate, logical CPU budget, and per-stage worker
  estimates have separate meanings, and requested/resolved/effective execution
  policy is authenticated outside statistical freshness; setup fails closed in
  strict mode, nested pools are suppressed, and partition manifests publish only
  after every unit validates; RSS high water provides nonsticky submission
  backpressure, Windows Job committed-memory peak and cgroup current/peak remain
  distinct telemetry, recognized resource failures cancel pending work and may
  retry pending partition coordinates once at half workers, valid publications
  are never quarantined for a soft warning, and roots fail fast before pair work;
- byte-bounded streaming, atomic durable units, and exact resume are permanent
  engineering invariants; a fixed 1 GiB process-tree ceiling is not;
- simulation v3 publication is producer-owned: staged data is hashed and
  sidecarred without a canonical data rewrite, coordinate-sorted native
  manifests reuse those identities, routine completion/resume is metadata-fast,
  and the explicit release audit remains byte-deep;
- curate preserves its authenticated provenance boundary with independent
  copy-on-write reflinks where supported and deterministic bounded physical-copy
  fallback; combine keeps normalized by-k Parquets as its only canonical data
  bytes and exposes `concat_ks` through one authenticated ordered partition
  manifest and deterministic scanner, with explicit deep verification and
  non-canonical release materialization;
- ingest and combine resolve authenticated upstream lifecycle/artifact identity
  once per root/k in the parent, pass immutable pickle-safe source snapshots to
  workers and validators, bind ingest freshness to simulation manifest/completion
  roots, classify allocator failures separately from malformed Parquet, and cap
  process workers by pending independent units;
- TrueSkill projected reads use byte-bounded Arrow batches; screening
  diagnostics publish an authenticated root/k result and completion stamp
  before deterministic aggregation, using one forward projected scan and a
  bounded reverse replay spool;
- all-player and seat batch metrics publish deterministic single-entry stream
  manifests with source-bound sidecars before per-k or stage completion;
  all-player metrics use k/schema-aware byte-bounded projections and stable
  row/seat accumulation; canonical performance uses memory-mapped root/k count
  matrices plus authenticated deterministic replicate-range shards, and only
  publishes bootstrap summaries after a complete validated range manifest;
- two-root stability reuses those immutable root/k matrices (or publishes a
  compatible fallback cache once), with independently authenticated top-N and
  joint-discrepancy ranges; final completion is stale unless both manifests validate;
- the H2H power plan and block manifest are immutable after publication;
  exact-power allocation is the global first admissible crossing, power is
  conditional on completed support, and cap authorization/execution progress
  is a separate resumable artifact;
- H2H targets completed games per pair/root/order, retains safety-limit attempts
  outside score-test counts, resumes deterministic contiguous replacements up
  to the frozen 2.0x attempt cap, keeps no-test pairs in the multiplicity family,
  and evaluates the frozen candidates against the 0.99 incident-attempt
  completion threshold without shrinking the family;
- final reporting authenticates the migration inventory as a stage output and
  states TrueSkill's completed-game conditioning and safety-limit exclusion;
- transient artifact I/O uses bounded, provider-neutral retries, and completed
  H2H execution can finalize a missing stamp without replaying block data.

Before a statistical change, identify the estimand, conditioning, chance
baseline, replication unit, uncertainty procedure, multiplicity rule, and
permitted report claim. Before a recovery change, compare coordinate manifests
and logical outputs across worker counts and interruption/resume.

Check git status before editing and use the repository .venv for validation.
```

Update this prompt and `metadata.md` when stage order, statistical contracts,
path/scope behavior, sidecars, RNG identity, or release gates materially change.
