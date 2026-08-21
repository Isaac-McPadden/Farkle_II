# Task 4D release-audit recovery and telemetry correctness

Date: 2026-08-20  
Starting HEAD: `6496720a912cb5df0ec2cd3beef90ea1c267e092`  
Scope: Task 4D only; Task 5A/5B were not started.

## Outcome

The production failure was a deterministic namespace-classification defect,
not a post-snapshot mutation. The audit now distinguishes canonical stage
completions from typed partition-unit resumability, active configuration state,
checkpoints/substages, and unrelated or malformed done-like files. The expected
canonical inventory is still freshly read and byte-authenticated.

Telemetry now uses parent-authoritative durable counters, deduplicates semantic
worker events across processes and retries, retains seed/root/stage/phase/scope
context, reports unknown denominators and ETA as indeterminate, and derives log
and sink output from one locked snapshot. No scientific method, RNG coordinate,
canonical artifact, schema, path, ordering, method version, or contract version
changed.

The preserved root could not be resumed under the current worktree because its
authenticated public-config and clean-code identity do not match the requested
configuration and development-dirty code identity. The new preflight now stops
that condition before publication. A production-layout 20-unit fixture passed
the final fresh-byte audit, and the real raw-simulation-to-report oracle passed
its final audit, published `complete_success` health and terminal `run_end`, and
completed a byte-stable ordinary replay. The Task 4D stop gate is therefore met
by the allowed equivalent fixture route.

## Proven root cause

`capture_authenticated_graph_snapshot` correctly received an explicit list of
canonical simulation/stage completions. The final audit passed that list to
`_build_audit_index`, but the explicit-inventory branch recursively promoted
nearly every `*.done.json` under the graph root into the canonical namespace.
`PartitionedStage` intentionally publishes durable `*.unit.done.json` stamps,
authenticated by exact unit output hashes and the ordered final partition
manifest. The audit consequently compared an explicit canonical-stage set to a
suffix-derived set containing internal resume state.

The incident's 20 additions were the ten 0-500/50 top-N ranges and ten
0-500/50 joint-discrepancy ranges under the real root-stability paths. Broader
inspection also found valid partition stamps under root ingest/combine `by_k`
trees and operational lifecycle substamps. A two-directory exception would
therefore have been incomplete.

Observed classification of the preserved tree with the centralized contract:

| Scope | Canonical | Partition unit | Operational | Active config | Unrelated |
|---|---:|---:|---:|---:|---:|
| Root 52 | 12 | 166 | 34 | 1 | 0 |
| Root 53 | 12 | 166 | 34 | 1 | 0 |
| Pair analysis | 9 | 20 | 0 | 0 | 0 |

## Completion-file classification contract

`farkle.utils.completion_files.CompletionNamespace` is now shared by snapshot
construction and release auditing.

- Canonical: exact executable-plan completion paths, known canonical basenames
  at any other location (so relocation/duplication fails), fixed
  `<graph>/<k>_players/simulation.done.json`, and a newly introduced top-level
  analysis stage completion.
- Partition unit: a `*.unit.done.json` file only when its JSON has the exact
  PartitionedStage unit-stamp envelope and its declared adjacent output exists.
- Active configuration: `active_config.done.json` only.
- Operational: checkpoint/state suffixes, substages below an existing canonical
  stage, `by_k`/diagnostic/checkpoint state, or the typed lifecycle envelope.
- Unrelated/malformed: every remaining done-like file.

Partition stamps remain on disk and continue to be validated by their unit and
final-manifest contracts. They are intentionally outside the released canonical
stage-completion graph; they are not ignored as unauthenticated files or treated
as canonical stages.

Fail-closed behavior is unchanged or strengthened: each expected canonical
stamp is freshly opened, hashed, parsed, and authenticated; graph/config/code/
provenance and snapshot generation/lifetime checks remain; missing, mutated,
relocated, duplicated, and unexpected canonical completions fail. Malformed or
tampered unit state fails its partition contract and cannot qualify for resume.

## Telemetry corrections

The over-100% values came from adding worker observations to already durable
parent/checkpoint counts and then selecting a maximum. Duplicate/retried events
could arrive from different PIDs, delayed unique events could arrive out of
order, and scheduler-future completion was being confused with durable unit
publication. Chunk text also used a global chunk index against a per-resume
remaining count. Generic StageRunner `phase=action` values lost useful context.

Corrections:

- displayed games/shuffles/chunks use only the parent-owned durable numerator;
- worker messages carry semantic event IDs and process sequence numbers;
- a bounded exact-ID set rejects duplicate/retried events across PIDs while
  retaining unique out-of-order observations;
- worker aggregates remain observational and never become completion counts;
- partition progress is `reused + unique durable completed unit keys`;
- simulation chunks use completed durable chunk indices over the global plan;
- terminal ETA is zero only at genuine completion; unknown ETA/denominator is
  explicitly `indeterminate`;
- StageRunner and two-seed phase scopes carry actual stage names, seed or root
  pair, execution scope, k, scheduler state, and available durable progress;
- the heartbeat thread copies active scopes once under lock, and the formatted
  log plus `pipeline_telemetry.json` sink consume that same snapshot.

Before (production incident):

```text
games=18780/12000 (156.5%) shuffles=1010/600 chunk 21/20 seed=? phase=action
```

After (protected benchmark):

```text
stage=simulation phase=simulation state=working seed=94512 k=2 games=12000/24000 (50.0%) shuffles=300/600 chunks=10/20 eta=40s
stage=simulation phase=simulation state=waiting_on_futures seed=94512 k=2 games=296/2400 (12.3%) shuffles=74/600 chunks=0/20 eta=indeterminate
stage=simulation phase=simulation state=working seed=94512 k=2 games=2400/2400 (100.0%) shuffles=600/600 chunks=20/20 eta=0s
```

## Tests and release-gate evidence

- Expanded Task 4B/4C/4D regression selection: **261 passed** in 288.7 s.
- Real raw-simulation-to-authenticated-report oracle: **1 passed** in 187.8 s.
  It ran both roots and pair analysis, passed the fresh-byte audit, asserted
  `pipeline_health.status == complete_success`, asserted the last manifest event
  was `run_end/complete_success`, then proved byte-stable no-force replay.
- The production-shaped unit fixture uses the actual
  `_bootstrap_top_n_ranges/units` and `_joint_discrepancy_ranges/units` paths,
  creates all 20 range stamps, passes the final audit, and reuses all ten units
  in each partition family.
- Missing, mutated, extra, relocated, duplicated, post-snapshot, provenance,
  mixed-identity, and interrupted-audit cases fail closed.
- Partition output/stamp corruption is rejected; ordinary resume repairs only
  the two invalid units and reuses the other ten.
- Duplicate/retried cross-PID and delayed out-of-order worker events, coherent
  terminal snapshots, determinate bounds, stage context, sequential roots/k,
  and full worker budget are covered.

Static results:

- repository-wide Ruff: passed;
- Mypy: passed, 87 source files;
- Black: passed on 16 changed Python files;
- Pyright: passed on changed production/test files, 0 errors;
- `git diff --check`: passed (line-ending conversion warnings only);
- deliverable JSON parse: passed.

## Preserved-root recovery

Target: `data/results_efficiency_updates_4B_fast_seed_pair_52_53`  
Command semantics: `farkle --config configs/fast_config.yaml two-seed-pipeline`,
without `--force`.

The first ordinary attempt exposed that identity validation occurred too late.
It published operational running state and rewrote root-52 run context/active
configuration before root authentication rejected the clean historical
completion under the dirty current code identity. It recovered the existing
600 row shards for root 52/k=2 and did not launch a tournament pool, but could
not authenticate the completion. This attempt did mutate operational files in
the preserved tree; it did **not** rewrite any sampled canonical simulation or
analysis artifact. Those changes were not hidden or rolled back.

Task 4D consequently added an early pair-context identity preflight. A second
ordinary attempt stopped in 7.9 s before health, manifest, run-context, or
active-config publication. Pair manifest, health, and run-context modification
times were unchanged across that attempt. The exact conflict was:

```text
public config SHA 91ad5... != fce4b...;
code identity clean commit 6496720... != development_dirty fingerprint 063c...
```

Representative canonical artifacts were unchanged across recovery attempts:

| Artifact | SHA-256 | `mtime_ns` | Bytes |
|---|---|---:|---:|
| Root 52, 2p simulation completion | `6e03cda...aa42` | 1787220869343266300 | 10632 |
| Root 52, metrics completion | `0500fc...965a` | 1787221152596197900 | 54924 |
| Root 53, 5p simulation completion | `76a016...fc0f` | 1787221762568183000 | 10632 |
| Root 53, screening completion | `f8a8c3...872a` | 1787222414337691200 | 4924 |
| Pair root-stability completion | `0c083f...53f2` | 1787222472492117200 | 21795 |
| Pair reporting completion | `bbc40e...29ff` | 1787222786062200000 | 2351 |
| Pair root-discrepancies Parquet | `2c9e90...7927f` | 1787222466023909900 | 27421 |

No completed tournament or canonical analysis stage was recomputed. The
preserved root cannot legitimately publish a new final audit/health/run-end
under the mismatched identity. Process inspection after both attempts found no
Farkle worker processes.

## Performance

Protected benchmark evidence:
`data/farkle-task4c-task4d-telemetry-v1/task4c_simulation_execution.json`
(`SHA-256 5939fc7f016017442951b89d17861b75b1f71ead6ce94007af04286d6db02c75`).

| Workers | Simulation | Games/s | Protected wall | Created/peak |
|---:|---:|---:|---:|---:|
| 1 | 82 s | 292.68 | 90.906 s | 0/0 |
| 2 | 37 s | 648.65 | 45.563 s | 2/2 |
| 12 | 22 s | 1090.91 | 31.516 s | 12/12 |

Against Task 4C's accepted 12-worker result (21 s, 1142.86 games/s, 30.547 s
protected wall), simulation throughput changed -4.55% and protected wall
changed +3.17%, within the 5% material-regression threshold and one-second
simulation timer granularity. The current 12-vs-1 simulation speedup was 3.727x,
so the legacy `>=4x` ratio assertion is false because serial improved from 86 s
to 82 s; multiprocessing remained materially faster. All 12 workers were
created/live and cleanly terminated. Interruption preserved and reused 74 rows,
recomputed at most 48 games, shut down in 0.281 s, left no orphan PIDs, and
resumed to byte/logical equivalence.

## Changed files

Production: `release_audit.py`, `stage_runner.py`, `run_contexts.py`,
`two_seed_pipeline.py`, `run_tournament.py`, `authenticated_graph.py`, new
`completion_files.py`, `partitioned_stage.py`, and `telemetry.py`.

Tests: `simulation_to_report_oracle.py`, `test_stage_runner.py`,
`test_seed_workflows.py`, `test_run_contexts.py`, `test_authenticated_graph.py`,
`test_partitioned_stage.py`, and `test_telemetry.py`.

Documentation: this report and machine record plus `context_prompt.md`,
`metadata.md`, and `testing_and_review_map.md`.

## Known baselines and remaining risks

Historical unrelated repository-wide baselines were not expanded: six Black
files, 28 Pyright test errors, and the stage-registry tuple/list assertion were
previously documented by Task 4B; Task 4C also recorded six pre-existing
`test_simulation_completion_mutation_matrix` failures. Changed-file gates and
the Task 4D regression selection are clean.

The preserved tree's first recovery attempt changed operational provenance
before the missing preflight was discovered; canonical samples remained exact,
but those operational bytes cannot be represented as untouched. The completion
classifier recognizes new top-level executable-stage completions structurally;
future stage-layout changes must keep the executable inventory authoritative and
extend focused namespace tests. Worker event-ID retention is deliberately
bounded (250,000 IDs); authoritative counters remain parent-owned even after
old observational IDs are evicted. Benchmark ratios remain sensitive to short
fixture timing and host load.

No commit or push was performed, no result tree was deleted or force-regenerated,
and no fresh full production pipeline or Task 5 work was started.
