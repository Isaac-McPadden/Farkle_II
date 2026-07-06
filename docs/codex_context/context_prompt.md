# Codex Context Prompt

Paste the prompt below into a new Codex chat when working on this repository.
The prompt is intentionally short and tells Codex to use the orientation pack
as cache, not authority.

- Last verified commit: `6cd587c3cb0625567e7d8412a2f97d1e343ef3e4`
- Working tree note: generated from a dirty tree with existing edits in
  `src/farkle/utils/mdd_helpers.py` and `tests/unit/utils/test_mdd.py`

## New-Chat Prompt

```text
You are working in the Farkle Mk II repo, a deterministic Monte Carlo
simulation and statistical analysis project for Farkle strategies.

Before doing broad grep, read these cached orientation docs:

- docs/codex_context/repo_map.md
- docs/codex_context/analysis_pipeline_map.md
- docs/codex_context/statistical_methods_map.md
- docs/codex_context/testing_and_review_map.md

Treat those docs as orientation only, not authority. For any code change,
statistical claim, path behavior, or test conclusion, verify the relevant source
files directly. Preserve the project invariants from AGENTS.md:

- all randomness is explicit and seeded
- prefer NumPy/project RNG helpers over hidden ambient randomness
- stages are idempotent and resumable
- long writes use atomic helpers and never leave partial outputs
- output paths are resolved from AppConfig and config, not guessed
- large data processing should stream or shard rather than materialize
- reuse helpers in src/farkle/utils before adding ad hoc helpers
- parallelize when order does not matter

When reviewing statistics, separate:

1. intended statistical method or estimator
2. implementation fidelity
3. model assumptions and independence assumptions
4. artifact/report interpretation
5. missing tests or hand-check examples

When reviewing tests, distinguish "test passes" from "claim is proven." Prefer
small hand-checkable examples for statistical code.

Current orientation pack was verified against commit
6cd587c3cb0625567e7d8412a2f97d1e343ef3e4 and was generated from a dirty tree
that already had edits in src/farkle/utils/mdd_helpers.py and
tests/unit/utils/test_mdd.py. Check git status before editing.
```

## Maintenance Rule

Regenerate or update this prompt when any of these change:

- stage registry or stage execution order
- statistical methods or formulas
- config seed/path semantics
- artifact contracts
- test harness assumptions
- major source directory ownership
