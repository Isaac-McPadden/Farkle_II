# Codex Repository Map

This is an orientation cache. Verify relevant source before changing code or
accepting a statistical claim.

## Source ownership

- `src/farkle/config.py`: typed configuration, strict migration rejection,
  canonical scopes, freshness keys, and every path helper.
- `src/farkle/game`: scoring, player state, exact turn accounting, and game
  completion behavior.
- `src/farkle/simulation`: workload sizing, coordinate-owned tournament tasks,
  checkpoints, row production, and process execution.
- `src/farkle/analysis`: streaming ingest/curate/concatenation, estimators,
  diagnostics, candidate freeze, H2H, dominance, agreement, and reporting.
- `src/farkle/orchestration`: root contexts, root-pair context, and two-root
  execution.
- `src/farkle/utils`: RNG namespaces, atomic writers, manifests, sidecars,
  lifecycle state, streaming loops, and shared statistics.
- `src/farkle/cli/main.py`: public CLI parsing and dispatch.

## Analysis modules

- Row foundation: `ingest.py`, `curate.py`, `combine.py`, `checks.py`.
- Tournament statistics: `all_player_metrics.py`, `performance.py`,
  `screening.py`, `seat_analysis.py`.
- Diagnostics: `game_stats.py`, `rng_diagnostics.py`, `roll_enumeration.py`.
- Screening models: `run_trueskill.py`, `trueskill.py`,
  `trueskill_screening.py`, `run_hgb.py`, `hgb_feat.py`.
- Root-pair work: `root_stability.py`, `candidate_family.py`,
  `h2h_schedule.py`, `h2h_inference.py`, `dominance.py`.
- Delivery: `structure_agreement.py`, `structure_reporting.py`,
  `migration_audit.py`, `release_audit.py`.
- Execution: `stage_registry.py`, `stage_runner.py`, package `__init__.py`.

## Contracts to locate first

- Scopes/path APIs: `ArtifactScope` and `AppConfig` in `config.py`.
- Artifact compatibility: `utils/artifact_contract.py`.
- Lifecycle/freshness: `utils/stage_completion.py`.
- RNG identity: `utils/random.py` and `docs/rng_contract.md`.
- Row coordinates: `docs/turn_and_row_contract.md`.
- Root/pair order: `analysis/__init__.py` and `analysis/stage_registry.py`.

Old on-disk outputs are not source inputs. `migration_audit.py` only inventories
them. Do not add lookup fallbacks or current readers for retired layouts.
