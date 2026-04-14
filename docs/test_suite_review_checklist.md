# Farkle Mk II Test Suite Review Checklist

Use this checklist to review whether the test suite itself is trustworthy, current, and aligned with the codebase claims.

This checklist is not about whether the application is correct. It is about whether the tests are good evidence for correctness.

## Review Setup

- [ ] Freeze the exact commit being reviewed and note the Python version and installed optional test dependencies.
- [ ] Treat `pytest.ini`, `tests/conftest.py`, `tests/helpers/*`, `tests/FAILURE_NOTES.md`, and representative files under `tests/unit`, `tests/integration`, and `tests/unit/analysis_light` as the core test-suite contract.
- [ ] Record whether you are reviewing test code only, running the suite, or both.
- [ ] Keep a log of skipped tests, xfails, monkeypatch-heavy tests, and tests that depend on golden artifacts or synthetic fixtures.

## Suggested Sequence

Recommended order for a full review of the tests:

1. Test harness and collection rules
   Estimate: 20-40 minutes.
   Goal: understand how pytest collects tests, which dependencies are optional, and which global fixtures affect determinism.
   Evidence to collect:
   `pytest.ini`, `tests/conftest.py`, and a short note on autouse fixtures, skip gates, and warning filters.

2. Shared helpers, goldens, and synthetic datasets
   Estimate: 45-90 minutes.
   Goal: verify that helper code provides stable, honest test inputs rather than masking real behavior.
   Evidence to collect:
   `tests/helpers/golden_utils.py`, `tests/helpers/config_factory.py`, `tests/helpers/metrics_samples.py`, and any fixture-generating helper used by multiple modules.

3. Representative unit tests by subsystem
   Estimate: 2-4 hours.
   Goal: verify that tests assert meaningful invariants rather than implementation trivia.
   Evidence to collect:
   one or two representative files from `game`, `simulation`, `analysis`, `config`, `cli`, `orchestration`, and `utils`, with notes on what behavior they truly prove.

4. Integration and pipeline-stabilizer tests
   Estimate: 1-3 hours.
   Goal: verify that slower tests exercise realistic workflows and artifact contracts, not only heavily patched happy paths.
   Evidence to collect:
   `tests/integration/*`, `tests/unit/analysis_light/*`, and any goldens or sample artifacts they rely on.

5. Failure inventory and stale-test review
   Estimate: 30-60 minutes.
   Goal: separate temporarily broken-but-useful tests from obsolete tests that no longer match the code.
   Evidence to collect:
   `tests/FAILURE_NOTES.md`, current skip/import behavior, and a short list of tests that appear stale or too tightly coupled to removed APIs.

6. Close-out
   Estimate: 20-40 minutes.
   Goal: classify the suite into trustworthy coverage, weak evidence, stale coverage, and missing coverage.
   Evidence to collect:
   a final summary grouped into `strong`, `qualified`, `stale`, and `missing`.

If time is limited, prioritize this order:

1. Harness determinism and skip behavior
2. Game and simulation tests
3. Metrics / analysis tests and goldens
4. Integration tests
5. Everything else

## Evidence Template

For each reviewed test area, capture:

- What behavior the tests claim to verify.
- Which files provide the evidence.
- Whether the tests exercise public behavior, private helpers, or both.
- What is patched, stubbed, skipped, or import-gated.
- Whether the assertions would catch a real regression in the underlying claim.
- Result:
  strong evidence, qualified evidence, weak evidence, stale, or missing.

## Glance / Boilerplate

### Collection and Suite Shape

- [ ] `pytest.ini` still reflects the intended suite shape: test discovery, markers, warning filters, and strict xfail behavior.
- [ ] The suite is still meaningfully split across `tests/unit`, `tests/integration`, and `tests/unit/analysis_light`.
- [ ] Test file names and directory layout still match actual collection behavior.
- [ ] The suite has visible coverage for `game`, `simulation`, `analysis`, `cli`, `config`, `orchestration`, and `utils`.

### Shared Harness Assumptions

- [ ] Autouse fixtures in `tests/conftest.py` are intentional and documented by behavior, especially time freezing and seed forcing.
- [ ] Global monkeypatches in `tests/conftest.py` do not silently invalidate important runtime behavior the suite claims to test.
- [ ] Optional dependency stubs for `numba`, `matplotlib`, and `sklearn` are narrow enough to support tests without falsifying behavior.
- [ ] `pytest.importorskip(...)` usage is appropriate and does not hide core regressions behind missing extras.

### Failure Notes and Maintenance Signals

- [ ] `tests/FAILURE_NOTES.md` still reflects real current failure modes rather than a stale historical report.
- [ ] Known broken tests are categorized clearly as dependency issues, API drift, logic regressions, or obsolete expectations.
- [ ] The suite is not carrying dead tests for APIs that no longer exist unless they are intentionally parked for migration work.

## Light Double Check

### Determinism and Repeatability

- [ ] Tests that claim determinism are actually deterministic under the suite harness, not just by accident.
- [ ] Randomness-sensitive tests either seed their own RNGs explicitly or rely on the suite-level deterministic fixtures in a defensible way.
- [ ] Time-sensitive tests are compatible with `freezegun` behavior and do not accidentally assert on moving timestamps.
- [ ] Hash/order-sensitive tests are safe under `PYTHONHASHSEED=0` and would fail for genuine ordering regressions.

### Dependency Gating and Optional Paths

- [ ] Tests skipped via `pytest.importorskip` correspond to truly optional functionality, not core packaged behavior.
- [ ] If a test requires `pydantic`, `hypothesis`, `pyarrow`, or other extras, the requirement matches the feature being tested.
- [ ] Optional-backend tests still prove fallback behavior rather than merely executing imports.
- [ ] Stubbed dependencies do not change interfaces so much that the tested code path is no longer representative.

### Helper and Fixture Quality

- [ ] `tests/helpers/config_factory.py` creates configs that reflect realistic defaults for the behavior under test.
- [ ] Golden helpers compare the right artifact content and normalize only what should be order-insensitive.
- [ ] Synthetic artifact builders in `tests/helpers/metrics_samples.py` and similar files preserve the schemas and invariants the production code expects.
- [ ] Shared fixtures do not overfit to one implementation detail and accidentally constrain future refactors.

### Public Behavior vs Private Coupling

- [ ] CLI tests focus primarily on public dispatch and config semantics, not deleted or historical entry points.
- [ ] Config tests validate user-visible semantics and normalization rules, not just dataclass internals.
- [ ] Utility tests around manifests, stage stamps, and RNG helpers validate contracts that production code actually depends on.
- [ ] Private-helper tests are justified where the helper encodes real business logic or a critical resumability contract.

## Careful Examination

### Do the Tests Prove the Important Claims?

- [ ] Game tests verify Farkle rules, final-round semantics, hot-dice behavior, and smart-discard counters in ways that would catch real rule regressions.
- [ ] Scoring tests include both exact examples and broader invariants, and the golden cases still reflect the intended scoring table.
- [ ] Simulation tests verify shuffle construction, seed handling, checkpointing, resume semantics, and aggregation determinism rather than only patched control flow.
- [ ] Analysis tests verify output meaning, stage invalidation, artifact naming, and statistical-field consistency rather than only file existence.
- [ ] Orchestration tests verify pair-level health, manifest semantics, and config provenance rather than only event sequencing.

### Are the Assertions Strong Enough?

- [ ] Assertions check the actual invariant that matters, not a proxy that could stay true while behavior regresses.
- [ ] Tests inspect artifact contents when content matters, not only existence.
- [ ] Tests that patch worker functions or heavy modules still validate final semantics rather than tautologies created by the patch itself.
- [ ] Branch-table tests cover meaningful behavior differences and are not just enumerating paths without validating outcomes.
- [ ] Property-based tests, where used, are scoped to meaningful invariants and not so weak that many wrong implementations would pass.

### Goldens and Stabilizer Tests

- [ ] Golden files are refreshed only when intended output semantics change, not as a routine way to bless regressions.
- [ ] Golden comparisons are normalized only where ordering is irrelevant; they do not erase meaningful regressions.
- [ ] `analysis_light` stabilizer tests still match the current artifact contracts and stage semantics.
- [ ] Synthetic goldens are realistic enough to catch schema, naming, and stage-output regressions in the metrics/analysis pipeline.

### Realism of Integration Coverage

- [ ] Integration tests exercise actual end-to-end seams with minimal patching where practical.
- [ ] Tests described as integration are not mostly unit tests wearing integration labels.
- [ ] CLI integration coverage still targets the real front door in `src/farkle/cli/main.py`.
- [ ] Pipeline integration coverage includes enough artifact inspection to prove outputs are usable by downstream stages.

### Review of Current Risk Hotspots

- [ ] Tests in `tests/unit/simulation/test_run_tournament_metrics.py` still align with the current metric-chunk and checkpoint contracts.
- [ ] Tests in `tests/unit/simulation/test_runner_branches.py` and related runner tests are reviewed for over-coupling to helper names and exact internal paths.
- [ ] Tests in `tests/unit/analysis/test_metrics.py` and `tests/unit/analysis_light/test_pipeline_stabilizers.py` are reviewed for stale schema expectations and golden drift.
- [ ] Tests in `tests/unit/cli/*` are reviewed for references to removed CLI helpers or outdated dispatch semantics.
- [ ] Tests in `tests/unit/config/test_app_config.py` are reviewed for compatibility with current config aliasing and deprecation behavior.
- [ ] Tests in `tests/unit/utils/test_manifest_events.py` and related utility tests are reviewed for coverage of the manifest v2 and `.done.json` contracts actually used by the pipeline.

### Broken, Stale, or Misleading Tests

- [ ] Any test currently failing because of removed helpers, renamed APIs, or changed semantics is explicitly classified as either:
  still valuable but needs rewrite, or obsolete and should be removed.
- [ ] Tests that skip because an extra dependency is missing are tracked separately from tests that fail due to real regressions.
- [ ] Historical comments or xfail notes that refer to old issues are still accurate.
- [ ] No test suite area is giving false confidence because the most important tests are permanently skipped in the default environment.

### Missing Coverage

- [ ] For each major repo claim, there is at least one test area that would fail if the claim stopped being true.
- [ ] There is explicit coverage for deterministic seed handling across simulation and analysis boundaries.
- [ ] There is explicit coverage for resumability and cache invalidation, not only fresh-run behavior.
- [ ] There is explicit coverage for artifact compatibility across stage boundaries.
- [ ] There is explicit coverage for statistically meaningful outputs, not just the mechanics of writing parquet and JSON files.

## Review Close-Out

- [ ] Write down which test areas provide strong regression protection.
- [ ] Write down which areas are valuable but too implementation-coupled and should be refocused on public behavior.
- [ ] Write down which areas appear stale because the product code moved and the tests did not.
- [ ] Write down which claims in the main review checklist lack strong automated coverage.
- [ ] Separate "tests are failing because code regressed" from "tests are failing because tests no longer describe the current system".
