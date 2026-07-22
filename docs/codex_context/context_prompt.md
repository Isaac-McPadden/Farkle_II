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
- current code never reads old on-disk analysis artifacts;
- pair analysis lives under `results_seed_pair_X_Y/seed_pair_analysis`, path
  lookup is non-mutating, and every H2H phase owns its artifacts;
- the H2H power plan is immutable after publication and execution progress is
  a separate resumable artifact;
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
