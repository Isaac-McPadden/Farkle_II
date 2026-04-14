# Farkle Mk II Human Review Checklist

Use this checklist to verify that the codebase matches its documented claims at the exact revision and config you are reviewing.

## Review Setup

- [ ] Freeze the exact commit, config file(s), and seed or seed pair used for the review.
- [ ] Treat the repo claims in `README.md`, `cli_args.md`, `docs/config_reference.md`, `docs/data_artifacts.md`, `statistical_analyses.md`, and `farkle_rules.md` as the statements to verify.
- [ ] Record which workflow you are reviewing: single-seed run, per-seed analysis pipeline, or two-seed orchestration.
- [ ] Keep a running log of commands executed, artifacts inspected, and any claim that is only approximately true or only true under specific configs.

## Suggested Sequence

Recommended order for a full human-in-the-loop review:

1. `Glance / Boilerplate`
   Estimate: 30-60 minutes.
   Goal: confirm that the docs, packaging, stage layout, config model, and test surface still describe the same system.
   Evidence to collect:
   `git rev-parse HEAD`, the exact config file path, the relevant doc paths, and a short note on any doc/code mismatch.

2. `Light Double Check` for config, CLI, determinism, and resumability
   Estimate: 1-2 hours.
   Goal: verify that the wiring claims are true before spending time validating deeper math or gameplay.
   Evidence to collect:
   one or two representative CLI invocations, the resulting `active_config.yaml`, `analysis/config.resolved.yaml`, one manifest file, and one `.done.json` file with notes on whether the fields match expectations.

3. `Careful Examination` of Farkle rules and tournament semantics
   Estimate: 2-4 hours.
   Goal: verify that the game engine and shuffle construction really implement the intended rules and deterministic tournament structure.
   Evidence to collect:
   at least one traced game example, one manually checked scoring example per major rule family, and one small seeded tournament run whose shuffle/game counts you verify by hand.

4. `Careful Examination` of stage-by-stage data correctness
   Estimate: 3-6 hours.
   Goal: confirm that each analysis stage consumes the right inputs, writes the right outputs, and preserves the intended semantics from upstream data.
   Evidence to collect:
   for each major stage family, one input artifact, one output artifact, the associated stage stamp, and a short note on what invariant you verified.

5. `Careful Examination` of statistical validity
   Estimate: 2-5 hours.
   Goal: confirm that uncertainty, pooling, power sizing, corrections, and agreement metrics match the documented design rather than merely producing files.
   Evidence to collect:
   the formula or code path used, a tiny worked example or hand-check, and the artifact fields where the result appears.

6. `Careful Examination` of two-seed orchestration and provenance
   Estimate: 1-3 hours.
   Goal: verify that pair-level success, health reporting, and `config_sha` provenance are trustworthy.
   Evidence to collect:
   the pair manifest, `pipeline_health.json`, one per-seed manifest, and notes on whether blocking failures are surfaced correctly.

7. `Review Close-Out`
   Estimate: 30-60 minutes.
   Goal: separate confirmed claims, qualified claims, and disproven claims.
   Evidence to collect:
   a final issue list grouped into `confirmed`, `qualified`, `incorrect`, and `not yet proven`.

If you are time-constrained, the highest-value order is:

1. Config/CLI/determinism
2. Game rules and tournament semantics
3. Metrics, game stats, and TrueSkill/head-to-head outputs
4. Resumability/provenance
5. Everything else

## Evidence Template

For each section or claim, capture the same small bundle of evidence:

- Claim being verified.
- Source of the claim:
  doc path, code path, or both.
- Review method:
  read-only code inspection, targeted test, small seeded run, artifact inspection, or manual calculation.
- Inputs used:
  config path, seed or seed pair, player counts, and command line.
- Output examined:
  artifact paths, manifest paths, log paths, and stage stamp paths.
- Result:
  confirmed, qualified, failed, or inconclusive.
- Notes:
  exact caveats, scope limits, or follow-up needed.

## Practical Effort Split

For a realistic full-pass review, a good default budget is:

- 10-15% on docs and boilerplate alignment.
- 20-25% on config, determinism, paths, and resumability.
- 25-35% on game logic and tournament semantics.
- 25-35% on analysis and statistical correctness.
- 10-15% on orchestration, provenance, and close-out.

If the main question is "does it do what we say it does?", bias time toward gameplay semantics, tournament construction, and statistical interpretation over lint/tooling polish.

## Glance / Boilerplate

### Docs, Packaging, and Entry Points

- [ ] `README.md`, `cli_args.md`, and `docs/config_reference.md` still match the actual CLI in `src/farkle/cli/main.py`.
- [ ] `README.md`, `docs/data_artifacts.md`, and `src/farkle/analysis/stage_registry.py` agree on active stage names and ordering.
- [ ] `pyproject.toml`, `requirements.txt`, `src/farkle/__main__.py`, and `src/farkle/cli/main.py` agree on package name, Python version, and entry point behavior.
- [ ] Config presets under `configs/` still load cleanly and reflect the documented seed model and path semantics.
- [ ] Top-level docs do not refer to deleted modules, obsolete outputs, or hard-coded stage numbers that the code no longer guarantees.

### Repo Surface and Test Surface

- [ ] The major code areas are covered by tests: `game`, `simulation`, `analysis`, `orchestration`, `utils`, `cli`, and `config`.
- [ ] Existing integration tests still represent the workflows they claim to represent, especially simulation, metrics, and top-level CLI paths.
- [ ] Golden-output tests are still pointed at current schemas and not silently validating stale artifacts.

### Fast Code-Hygiene Pass

- [ ] New code paths use `AppConfig` helpers instead of manually assembling analysis paths.
- [ ] Persisted artifacts use atomic-write helpers or equivalent safe replacement logic.
- [ ] No obvious hidden randomness entry points exist outside explicit seeded helpers or clearly seeded local RNG objects.
- [ ] Legacy compatibility branches still look secondary, not accidentally the only live path.
- [ ] Canonical artifact names in docs match the constants and helpers actually used by the code.

## Light Double Check

### Config, CLI, and Path Semantics

- [ ] `sim.seed_list`, `sim.seed`, and `sim.seed_pair` normalization in `src/farkle/config.py` matches the documented single-seed and two-seed rules.
- [ ] CLI overrides in `src/farkle/cli/main.py` actually mutate the config fields the help text says they mutate.
- [ ] `results_root`, `analysis_dir`, stage directories, and per-stage artifact helpers resolve correctly for both relative and absolute `results_dir_prefix` inputs.
- [ ] `active_config.yaml` and `analysis/config.resolved.yaml` are written where the docs say they are and contain the resolved state actually used for the run.

### Determinism and Stable IDs

- [ ] Strategy-grid ordering and `strategy_id` assignment are deterministic across repeated runs and config ordering changes.
- [ ] Strategy parsing and manifest-based decoding stay consistent between simulation outputs and analysis inputs.
- [ ] All simulation and analysis randomness is rooted in explicit seeds and child-seed spawning, not ambient process state.
- [ ] Any use of Python's `random` module is explicitly seeded and justified, especially in `simulation/strategies.py`, `simulation/watch_game.py`, `simulation/time_farkle.py`, `analysis/meta.py`, and `utils/random.py`.

### Resumability, Atomicity, and Cache Behavior

- [ ] Long-running stages use `atomic_path`, `write_parquet_atomic`, `run_streaming_shard`, or equivalent finalization-safe helpers.
- [ ] `.done.json` stamps are present for the stages that claim resumability and contain the expected config and stage hashes.
- [ ] `stage_is_up_to_date` checks are invalidated by the right inputs and do not silently reuse stale outputs.
- [ ] Manifest v2 events are used consistently and legacy manifests rotate aside instead of being appended to incompatibly.
- [ ] Resume validation rejects mismatched strategy manifests, duplicate chunk entries, wrong player counts, and unexpected seeds.

### Pipeline and Artifact Wiring

- [ ] The documented `ingest -> curate -> combine -> metrics -> analytics` flow matches `src/farkle/analysis/__init__.py`, `src/farkle/analysis/pipeline.py`, and `src/farkle/analysis/stage_runner.py`.
- [ ] Interseed stage-layout renumbering when RNG diagnostics are disabled matches the implementation in `src/farkle/analysis/stage_registry.py`.
- [ ] Required artifact names and locations in `docs/data_artifacts.md` line up with the actual writer paths and validation logic.
- [ ] Legacy-path fallback logic preserves backward compatibility without silently preferring old locations over canonical ones.

## Careful Examination

### Farkle Rules, Scoring, and Single-Game Behavior

- [ ] `farkle_rules.md` matches the actual scoring behavior implemented by `src/farkle/game/scoring.py` and `src/farkle/game/scoring_lookup.py`.
- [ ] Single-roll scoring returns the correct score, used-dice count, and reroll count for all supported scoring combinations, not just sampled cases.
- [ ] Smart-5 and Smart-1 discard logic never breaks scoring sets, produces negative discard counts, or violates bank-vs-reroll semantics.
- [ ] The 500-point entry gate is implemented exactly as intended and does not double-apply or get skipped after a bust.
- [ ] Hot-dice handling and dice-left transitions match the intended rules.
- [ ] Final-round logic in `src/farkle/game/engine.py` gives each remaining player exactly one last turn after the first trigger and handles `run_up_score` correctly.
- [ ] Winner, rank, and margin values agree between `GameMetrics`, flattened game rows, and downstream analysis assumptions.

### Tournament Construction and Simulation Semantics

- [ ] Each shuffle uses a full permutation of the strategy grid and produces exactly `n_strategies / n_players` games with no omissions or duplicates.
- [ ] `experiment_size`, strategy-grid construction, and divisibility checks match the tournament math claimed in docs and logs.
- [ ] Resume mode continues interrupted tournaments without duplicate rows, duplicate metric chunks, or dropped shuffle seeds.
- [ ] Checkpoint reload, row-manifest replay, and metric-chunk replay all reconstruct the same aggregates as a clean run.
- [ ] Multi-`n` runs and per-`n` overrides cannot leak outputs into the wrong directory tree or reuse the wrong checkpoint family.
- [ ] Parallel worker budgeting and native-thread caps behave sensibly during nested orchestration and do not oversubscribe the machine.

### Stage-by-Stage Data Correctness

- [ ] `ingest` correctly normalizes winner columns, seat columns, legacy strategy identifiers, and missing-column schemas.
- [ ] `curate` preserves the intended schema, writes matching manifests, and migrates legacy artifacts only when safe.
- [ ] `combine` pads and casts schemas correctly, and the partitioned dataset plus monolithic compatibility parquet represent the same data.
- [ ] `metrics` produces internally consistent `games`, `wins`, `win_rate`, uncertainty, `expected_score`, and pooled-weight outputs.
- [ ] `coverage_by_k` correctly reports missing strategies, missing seeds, and coverage gaps rather than masking them with padding.
- [ ] `game_stats` writes the exact counts and means it claims to write, and any approximate quantile logic is clearly bounded and documented.
- [ ] `seed_summaries`, `variance`, and `meta` respect per-seed boundaries and use the documented inclusion and pooling rules.
- [ ] `trueskill` and `run_trueskill` read the right rows, update ratings in the intended order, checkpoint safely, and write per-`k` plus pooled outputs that match the docs.
- [ ] `head2head` and `run_bonferroni_head2head` select candidate strategies, size experiments, simulate pairings, and compute p-values and correction targets exactly as claimed.
- [ ] `h2h_analysis` applies the intended post-processing and produces tiers and rankings that are traceable back to the pairwise artifacts.
- [ ] `tiering_report` combines cross-`k` frequentist scores and weights the way the docs claim.
- [ ] `run_hgb` and `hgb_feat` build features from strategy encodings correctly and do not leak downstream targets into training inputs.
- [ ] `agreement` compares compatible rankings and tiers and does not accidentally compare differently scoped artifacts.
- [ ] `interseed_analysis`, `game_stats_interseed`, `rng_diagnostics`, and `h2h_tier_trends` really consume the paired-seed inputs they claim to consume.

### Statistical Validity

- [ ] Confidence intervals, standard errors, and variance formulas use the documented estimators and consistent denominator conventions throughout the project.
- [ ] Meta-analysis fixed-vs-random switching, I^2 thresholding, and seed-selection behavior match the documented design.
- [ ] Multiple-comparison correction and power calculations in the head-to-head pipeline match the stated Bonferroni and Holm logic and runtime heuristics.
- [ ] Agreement metrics, pooled weights, and tier-boundary rules are reproducible from saved artifacts without hidden state.
- [ ] Any artifact labeled "pooled" clearly means what the docs imply: pooled across player counts, pooled across seeds, or both.

### Two-Seed Orchestration and Provenance

- [ ] `src/farkle/orchestration/two_seed_pipeline.py` only marks success when both per-seed pipelines and the interseed family have their required outputs.
- [ ] `pipeline_health.json` accurately reports missing or invalid artifacts and agrees with manifest events.
- [ ] `config_sha` validation actually catches mixed-config or stale-artifact runs.
- [ ] Shared meta-analysis directories and interseed input overrides cannot cross wires between unrelated seed pairs.
- [ ] Parallel per-seed execution does not race on shared manifests or shared output roots.

## Review Close-Out

- [ ] Write down every place where the code is correct but the docs overstate certainty or scope.
- [ ] Write down every place where behavior exists in code but is undocumented, especially legacy-path fallbacks, deprecated flags, or optional-stage skip paths.
- [ ] Separate "works as implemented" from "implements the intended Farkle rules or statistical design".
- [ ] For each accepted claim, keep at least one reproducible command, artifact path, or test reference as evidence.
- [ ] Note any gap where the test suite proves mechanics but not the higher-level claim you care about.
