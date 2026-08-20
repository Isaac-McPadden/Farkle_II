# Task 4B authenticated graph snapshots and final-audit deduplication

## Verdict

Task 4B is implemented and meets the production-change decision rule. Each
completed root or pair context now produces one immutable, explicitly owned,
process-local authenticated graph snapshot. Later same-process lifecycle and
stage-state queries reuse that snapshot without opening or hashing artifact
bytes. Pipeline success remains subordinate to one top-level phase named
`final_byte_deep_release_audit`, which freshly reads all three current graphs.

At 512 artifacts across a root/root/pair topology, median repeated finalization
fell from 12.714 s to 3.844 s, a 69.8% reduction. Conservatively adding the
one-time 1.470 s snapshot inventory construction still yields a 58.2%
reduction. At 128 artifacts the corresponding reductions were 52.6% and 39.9%.
The 9-artifact fixture regressed because fixed current-code and strict
three-context audit overhead dominates sub-second graphs; it is retained as the
small correctness boundary, not used to conceal the crossover.

The machine-readable summary is
[`task4b_authenticated_graph.json`](task4b_authenticated_graph.json). Raw
repetitions are preserved under
`data/farkle-task4b-authenticated-graph-v1/task4b_authenticated_graph.json`.
The resumable benchmark driver is
[`benchmark_task4b_authenticated_graph.py`](../../scripts/benchmark_task4b_authenticated_graph.py).

## Snapshot design and lifetime

`SnapshotGeneration` is an explicit owner created only after its context has
stopped publishing. `AuthenticatedGraphSnapshot` is a frozen typed value bound
to:

- resolved graph, analysis, run-context, and active-config paths;
- exact run-context and active-config byte hashes plus authenticated
  run-context identity;
- repository code identity and policy;
- public/statistical configuration SHA, lineage, game-profile identity, and
  root or root-pair scope;
- canonical stage states, completion paths and byte hashes, completion output
  identities, stage identity hashes, and established lifecycle hash;
- a canonical sidecar-declared inventory of every ordinary artifact and
  immutable manifest in the graph; and
- process ID, explicit generation number, an unforgeable in-process object
  identity, complete construction status, and all-stages-complete status.

The snapshot and generation owner deliberately reject serialization. Reuse
requires the same owner object, process, generation, scope, roots, and resolved
run-context path. A process restart, copy for another root, force-rerun
invalidation, changed generation, incomplete state, missing sidecar, missing
completion, or interrupted construction cannot produce a usable snapshot.

The lifetime is intentionally narrow. Root snapshots are built after root
analysis becomes quiescent. Pair publication does not write either root graph.
The pair snapshot is built only after pair analysis becomes quiescent. All
three live only through finalization. This ownership boundary avoids global
filesystem hooks or a path/mtime cache. Any future writer must invalidate its
context generation before publication; current production writers all precede
snapshot construction.

## Removed and retained authentication work

Removed:

- the final deep lifecycle reconstruction for each already quiescent root;
- the redundant final pair-state reconstruction after the pair snapshot has
  been built;
- three separately timed final run-context reload/audit success gates; and
- byte hashing during same-process root lifecycle reuse.

Retained:

- StageRunner authentication after every stage action;
- one complete-context authentication before snapshot capture, including
  resumed/skipped outputs;
- exact existing root lifecycle computation from ordered completion-stamp
  bytes;
- fresh current-code, config, lineage, run-context, and active-config checks;
- three internal root traversals because the graphs are disjoint; and
- fresh byte/schema/metadata validation of every canonical graph member in the
  single final release phase.

The final audit uses snapshots only as expected inventories and identities. It
does not trust previously calculated data hashes. It rejects additions,
omissions, duplicates, orphan or missing sidecars, changed completions,
cross-context source/manifest mismatches, mixed identities, and current bytes
that differ from the snapshot.

Standalone `audit_sidecar_completeness` and
`scripts/check_structure_release.py` remain independently byte-deep and accept
no process snapshot.

## Bounded benchmark

The benchmark uses production v3 Parquet publication, completion, run-context,
stage-state, release-audit, and `_final_release_gate` code. Each fixture has two
root contexts and one pair context. One warm-up precedes three alternating
baseline/optimized repetitions.

| Artifacts | Baseline median | Optimized median | Snapshot build | Finalization reduction | Reduction incl. snapshot |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 0.323 s | 0.648 s | 0.111 s | -100.8% | negative |
| 128 | 3.130 s | 1.484 s | 0.396 s | 52.6% | 39.9% |
| 512 | 12.714 s | 3.844 s | 1.470 s | 69.8% | 58.2% |

At 512 artifacts:

| Metric | Task 4A baseline behavior | Snapshot/final audit | Reduction |
| --- | ---: | ---: | ---: |
| Opens | 8,711 | 2,576 | 70.4% |
| SHA-256 calls | 4,099 | 1,037 | 74.7% |
| Bytes hashed | 10,859,241 | 3,127,274 | 71.2% |
| Schema validations | 2,048 | 512 | 75.0% |
| Metadata validations | 2,560 | 512 | 80.0% |
| Stage-state resolutions | 3 | 0 | 100% |
| Top-level graph-audit invocations | 3 | 1 | 66.7% |
| Internal disjoint-root traversals | 3 | 3 | 0% |
| Peak observed RSS | 235,257,856 B | 235,253,760 B | -4 KiB |

The optimized path records six snapshot hits in the benchmark: three explicit
finalization reuses and three release-gate ownership checks. Canonical output
digests were unchanged. A post-snapshot artifact tamper failed closed under
both baseline and optimized audits.

The combined final audit itself is somewhat slower than the old three shallow
calls in small fixtures because it additionally authenticates expected
completion inventory, exact active-config bytes, current repository identity,
and cross-context provenance. The net improvement comes from removing the much
larger repeated lifecycle pass. This is an intentional security/performance
trade: no final byte-deep check was removed.

## Adversarial and interruption evidence

The new suite independently mutates 28 properties after snapshot construction:
artifact bytes and schema; artifact sidecar bytes, fields, and contract hash;
manifest bytes, sidecar, and summary; completion bytes, ordering, omission,
duplication, and inventory additions; run-context and active-config bytes; statistical config, lineage,
and code identity; upstream source data and sidecar; manifest-root binding;
canonical relative path, scope, and player count; orphan and missing sidecars;
duplicate canonical location; and mixed contract/global release identity.
Every case is rejected by the final audit.

Five construction interruption boundaries return no snapshot. All four
non-complete lifecycle states are rejected. Data without sidecar and sidecar
without completion cannot enter a complete snapshot. Cross-root use, simulated
process restart, explicit generation invalidation/force rerun, and serialization
are rejected. Snapshot reuse increments a hit without any SHA call. An
interrupted final audit leaves only the initial `running` health state and no
run-end success event. Failed root preconditions execute zero final audits.

The existing release-audit tamper tests were not modified and pass. The real
structural toy oracle passes, including standalone byte-deep audit and
interruption/resume.

## Identity and version review

- Persisted snapshot: no.
- Canonical data product: no.
- Resumable checkpoint or cross-process trust source: no.
- Accepted authentication semantics: unchanged; the final gate is stricter
  about matching the just-captured inventory but does not accept anything the
  old v3 contract rejected.
- Artifact contract: remains v3.
- Accepted global identity: remains `[3, 2, 2, 2, 2, 2]`.
- Sidecar, manifest, completion, and run-context schemas: unchanged.
- Canonical paths and artifact inventory: unchanged.
- Statistical and method identities: unchanged.
- Lifecycle hash: exact established ordered completion-byte formula, unchanged.

All artifacts produced under prior repository code identities, including the
historical seed-48/49 tree and Task 4A roots, remain valid read-only evidence
but cannot resume solely because this source-changing HEAD has a new whole-repo
code identity after commit. No migration, reseal, or compatibility backfill was
performed.

## Projection, limitations, and decision

The 128- and 512-artifact fixtures exceed the 20% acceptance threshold. The
512-artifact structure reduces repeated hashing/open work by roughly 70-80%
with immaterial memory change. Applying the measured structural ratio to the
historical 23-minute final tail suggests roughly 10-18 minutes of avoidable
repeated lifecycle authentication, but phase attribution in that historical
run is incomplete. This is low-confidence. The revised plan's 45-75 minute
post-Task-4B integration range is retained rather than tightened.

The benchmark uses tiny Parquet payloads and does not represent production row
width or record volume. It is an authentication/finalization benchmark, not
Task 5A capacity qualification. No full integration pair or production run was
performed.

## Verification

The final focused compatibility suite passed 376 tests in 587.4 seconds. The
directly affected snapshot/release subset also passed after the final inventory
hardening. The real structural oracle passed twice in 210.1 seconds.

Task 4B's static gates are clean: repository-wide Ruff, mypy across all 86
source files, Black on all 12 changed Python files, Pyright on every changed
Python file, `git diff --check`, and JSON parsing all pass.

Established broader baselines are reported separately and were not expanded:
whole-repository Black still identifies the same six unrelated files, whole-
repository Pyright still reports 28 unrelated test errors, and the stage-
registry test file retains its known tuple-versus-list spawn-boundary failure
(the other eight tests in that file pass). These are outside Task 4B scope.

## Disposable roots

Task 4B created and preserves only these two disposable paths. The second is
the canonical `IOConfig` data prefix used by the two root contexts; pair and
summary evidence live in the first:

- `data/farkle-task4b-authenticated-graph-v1`.
- `data/data/farkle-task4b-authenticated-graph-v1`.

The four preserved Task 4A roots were not opened as benchmark inputs, modified,
deleted, or included in Task 4B evidence.
