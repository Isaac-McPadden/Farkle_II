# Codex Context Metadata

This file records the verification state for the cached Codex orientation docs.
Treat the maps as orientation only, not authority; verify relevant source files
before making code changes or accepting statistical claims.

- Last verified source commit: `50a5e86d226120b38c27d2666cb72dbe21a5cacb`
- Verified date: `2026-07-07`
- Working tree at verification: clean (`git status --short` produced no output)
- Verification scope: `repo_map.md`, `analysis_pipeline_map.md`,
  `statistical_methods_map.md`, `testing_and_review_map.md`, and
  `context_prompt.md`
- Incremental correction: `2026-07-13`, based on source commit `209a3f2`,
  verified the metrics-stage description and canonical performance formulas
  against `all_player_metrics.py`, `performance.py`, their path helpers, and
  their hand-computed tests. Other sections retain the verification state above.
- Incremental correction: `2026-07-13`, based on source commit `0850dd8`,
  removed the retired detectable-difference method and verified descriptive
  screening language, stage placement, artifact contracts, and tests against
  `screening.py` and `test_screening.py`.

## Update Rule

Update this metadata when the orientation pack is regenerated, revalidated
against source, or materially corrected after source inspection. Do not update
the verified commit for ordinary commits unless the affected context docs were
checked against the relevant source files.
