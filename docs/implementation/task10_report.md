# Task 10: integrated first-release qualification

September 24 first-release backlog, qualified October 7, 2026.
Status: **complete at the Task 10 validation/documentation boundary**.
All **1,584 collected functional cases** have accepted passing evidence across
isolated runs; static gates and both final audit routes passed. No production
run has been started or authorized.

## Scope and preserved work

Task 10 joins the implemented Task 0/9/1–8 contracts through validation,
operating review and documentation. Its boundary is a bounded, authenticated
two-root qualification and a reviewable future operating configuration.
It does not authorize the full run, raise its H2H cap, prescribe strategy
rankings or reopen the excluded scientific experiments.

The starting working tree was already dirty. The initial tracked/untracked
file inventory and hashes are in `data/task10/start_inventory.json`; initial
HEAD was `7db67ccc94a143ebd16c335563513e89a84a62aa`. No reset, stash or commit
was made in that working tree. The existing mega configuration and historical
reports remain preserved. The default production results directory exists;
its contents were not used as qualification inputs.

Execution used the root Python 3.12 virtual environment and independent,
clean, committed temporary snapshots containing the dirty tree's intended
contents. `PYTHONPATH` explicitly selects each snapshot's source, including
spawned launcher children, rather than the venv's editable working-tree path.
Snapshot identities and file hashes are recorded in `data/task10`.
Generated evidence and documentation are distinct from the authenticated
execution identity; no sidecars were rewritten to pretend an old execution
used new code.

## Changes required by qualification

The four inherited terminology findings are resolved in
`outcome_publication.py`: the three messages now say “all-batch”, and the
internal merge function has that name. Statistical formulas, estimands,
published schemas and RNG namespaces are unchanged. Code freshness still
invalidates incompatible historical plans/checkpoints.

Broad static checks exposed 35 test/helper typing findings and three formatting
failures. The changes annotate wrappers and fixtures accurately, narrow
optional results, correct a two-context tuple, and apply formatting. The two
formatted runtime modules have unchanged Python ASTs. These repairs permit
the repository-wide gates rather than suppressing their diagnostics.

A dense precision benchmark exposed an ingest execution blocker. A Parquet
row group per shuffle grew the high-k footer beyond the byte-aware reader's
metadata admission. Ingest now coalesces adjacent Arrow inputs within one
quarter of its existing byte allowance and a conservative schema-derived row
limit. It compacts chunk metadata periodically, preserves order/types/nulls,
and retains atomic partition publication and exact durable-unit recovery.
The allowance reserves input, old/new compaction buffers and writer headroom;
no resource ceiling was raised.

Ingest physical layout is now **3**, its stage cache version is **4**, and the
physical-layout version is included in shared method identity. Existing
incompatible units must be rejected/recomputed through normal freshness,
not accepted by refreshing their sidecars. A streamed synthetic physical
fixture exercised all **4,300 × 430 = 1,849,000** high-k coordinates through
the actual writer: **205 row groups**, **3,474,776 footer bytes**, and maximum
input **197,864 bytes**. An independent bounded reader checked every ordered
coordinate. This fixture tests storage layout; it is not production game data.

The new Windows integration test uses the normal public launcher and game
rules; the raw-row helper now also accepts contexts without an injected game
profile. New benchmark drivers measure added outcome/behavior work and
refresh conditional capacity. The existing precision driver can exercise a
dense, actually resampling fixture. No broader storage optimization was added.

## Functional, static and integrated evidence

Final counts, commands, hashes, timings and audit results are recorded in
[task10_validation.json](task10_validation.json). The coverage collector checks
set equality against all **1,584 collected cases**, including the Windows case.
All passed; none are omitted or waived. Ruff/Black cover maintained source,
tests and scripts; Mypy covers all 100 runtime modules; full Pyright and the
terminology gate passed. The raw workflow oracle took 1,066.585 seconds,
including 374.588 seconds of required preflight work, within its 3,840-second
ceiling. The inherited xunit2 `record_property` warning remains nonfatal;
properties are present in the XML evidence.

The existing raw simulation oracle exercises required preflight, root/root/
pair publication, safety/tie conditioning, independent numerical expectations
and no-force replay. Existing focused fixtures additionally cover interruption,
worker-count invariance and corrupted/incomplete units. Accepted evidence uses
the 55 integration/top-level cases, a broad unit pass with complete affected-
module followups, and the Windows launcher. All execution snapshots have the
same final **100 runtime source files**; validation records the precise test
differences and clean Git state rather than claiming one identical checkout.

The combined process passed integration/top-level cases before reaching a
fixture's 768 MiB worker threshold. The broad unit process then passed 1,518
cases and exposed ten issues: six obsolete unsealed simulation controls,
another worker threshold, two absent historical-data inputs and a negative
Git-identity check whose temporary directory was inside Git. Complete module
followups passed **54 cases**. They preserve memory limits, run in fresh
processes/outside-Git temporary directories, use the approved authenticated
simulation fixture, and provide the declared historical capacity dependency
through an ignored junction used only for reading. That dependency is engineering
baseline evidence, not a current scientific workflow input. Failed commands
remain recorded; their failing cases are replaced by successful module followups.

The Windows workload retains production resource controls but deliberately
uses an integration profile and small scientific dimensions: roots 11/22,
k=2/4, four strategies, 20 deterministic batches and 600 shuffles per root/k,
3,600 normal tournament games overall. It uses two requested workers, spawn,
an absolute S: results prefix and C: scratch. Integration precision and H2H
planning settings are explicitly reduced; the products cannot become
production or release eligible merely by passing an audit.

Its accepted sequence was:

1. Noninteractive cap stop: configured 23 attempts, frozen plan 24, exit 3,
   `blocked_by_cap`, and no H2H block results (**991.27 s**).
2. Exact-plan explicit authorization and resume: cap stays 23; matching
   authorization grants 24. **3,252** root Parquet hashes and modification times
   persist (**381.29 s**).
3. Successful final root/root/pair deep audit, generated guide validation,
   independent raw-row reconciliation and mandatory preflight/monitor evidence.
   Deliberately omitted exhaustive diagnostics are recognized as omitted.
4. No-force replay preserves all **8,964** stable product hashes (**365.80 s**).
5. Remove the final guide Markdown while retaining its stale sidecar; ordinary
   resume repairs it and preserves guide JSON plus all durable Parquet hashes
   and modification times for **3,288** files. The fresh final audit passed
   again (**379.97 s**).

Actual context evidence shows strict, enforced **12 GiB** Windows Job Objects
with no fallback, **8 GiB** admission/warning/reserve controls and actual
cross-volume publication between distinct NTFS volumes. Each successful health
result reports one top-level final audit with three internal roots, clean code
identity and integration/release ineligibility. The standalone structural
release check also passed on the fresh canonical analysis roots (**23.48 s**).
The final deep audit authenticated **3,838 artifacts and adjacent sidecars**.

The whole launcher test took **2,153.02 s**; its driver recorded 2,151.14 s.
The bounded results occupy **141,674,165 bytes**; disposable scratch was empty
at completion, which does not measure its peak occupancy. Final recovery
telemetry recorded 718,221,312 bytes peak process-tree RSS and 527,339,520 bytes
sampled aggregate memory, with no warning crossings or monitoring error. The
separate Windows committed-memory peak field is unavailable (`null`), not zero;
these last-command observations do not bound the first launch's peak.

Development failures remain in `data/task10`: the initial launcher fixture
violated the minimum integration batching rule; an overbroad test glob then
mistook the H2H manifest for a block result; the first dense benchmark exposed
the footer blocker. The first replay comparison also counted the CLI's append-
only log as stable: all 8,964 other files matched. Only that operational log was
added to the existing run-view exclusions. Each new launcher phase now saves
atomic progress evidence. Superseded runs were stopped or failed and are not
counted as passed acceptance commands; the corrected launcher passed fresh.

## Added cost measurements and conditional capacity

Synthetic producer measurements use 5,160 strategies at k=2 and k=12,
four shuffles, two batches and the existing 32 MiB analysis allowance. They
authenticate upstream fixtures, measure actual producers by phase, retain
canonical units, and prove no-force hash/mtime reuse. They do not provide
strategy-performance findings or empirical production confidence bounds.

| Added producer | k=2 elapsed | k=12 elapsed |
| --- | ---: | ---: |
| Outcomes | 17.22 s | 27.14 s |
| Seat behavior | 33.51 s | 259.23 s |

Peak incremental parent RSS across these fixtures was below 48 MB. The dense
precision fixture used 120 strategies, k=12, 40 batches, 4,800 games and
2,000 replicates; **4,800 primary cells actually resampled**. It finished in
69.93 s, reused valid outputs in 3.78 s, and retained 17,573,313 checkpoint
bytes. Parent incremental RSS was 14,295,040 bytes. Preparation and baseline
RSS are recorded separately; these are not process-tree peak guarantees.

[task10_capacity.json](task10_capacity.json) combines the historical Task 7A
base with these measured coefficients. It scales source exposures, dense
batch/seat grids, strategy tiles, empirical-frequency bins and activated
precision work separately. It does not credit unmeasured worker speedup or
ingest-coalescing speedup. Source-vs-grid mix, merge generations, frequency
support, host/I/O factors and extra monitor/parameter/guide/audit overhead are
explicit engineering sensitivities.

| Scenario | Total runtime | Durable storage |
| --- | ---: | ---: |
| Lower cost | 175.3 h / 7.3 days | 200.2 GiB |
| Central serial coefficients | 259.7 h / 10.8 days | 303.6 GiB |
| High cost | 543.8 h / 22.7 days | 570.8 GiB |

These are conditional estimates, not calibrated intervals, guaranteed lower/
upper bounds, readiness deadlines or observed production runs. The historical
base used synchronized S: storage; new synthetic benchmarks used C: and ran
alongside validation. Compression, empirical support, conditional support,
merge shrinkage, concurrency and disk performance can materially change costs.
The model assumes every primary cell can resample; sparse/degenerate cells
may do less work. Its additional overhead allowance is disclosed, not measured
at production scale.

Durable estimates include retained canonical checkpoints. Disposable scratch,
active destination-local cross-volume bridges, and other users' disk activity
need additional headroom. The model alone does not prove a storage upper bound
or that a currently free drive will remain adequate throughout execution.
Review both volumes immediately before a future authorized launch.

## Operating review and completion boundary

[task10_operations.md](task10_operations.md) records fixed scientific settings,
resource meanings, the H2H envelope versus exact-plan cap off-ramp, clean
execution identity, explicit output/scratch paths, launch and no-force resume.
The unchanged production design is roots 102/103, eight player counts,
5,160 strategies, 100 batches, 95% intervals and 2,000 bootstrap replicates.
It derives 39,013,900 tournament games and 177,504,000 exposures per root.
At the 150-candidate envelope H2H can require 195,875,400 attempts against
the unchanged 100,000,000 cap. Exact frozen-plan consent is the intended
operational off-ramp; statistical scope is not reduced to fit the cap.

Read-only planning also checked all eight player counts' source/precision
inventory reservations and serialized outcome/precision descriptor shapes
using the documented absolute result prefix. Maximum four-times-serialized
reservations were **22,098,456 bytes** for outcomes and **18,107,028 bytes** for
precision, each within the existing **32 MiB** allowance. The probe uses exact
getters/unit constructors/merge topology with fixed-length hash placeholders
and coalesced source geometry; it calls no artifact publishers. Actual source
footers, empirical support and runtime allocation still require execution's
fail-closed checks. It is not an authenticated production artifact or a memory
upper-bound proof.

Qualification supports correctness and recovery of the bounded implementation.
It does not establish comprehensive RNG independence, guarantee informative
intervals for sparse/degenerate cells, prove equivalence from detection power,
or identify causal parameter/opponent effects. The guide retains explicit
exclusions and unsupported/unestimable statuses.

The completion boundary is final accepted test/static evidence, fresh bounded
audit success, recovered/replayed durable units, refreshed conditional costs
and reviewable operations. Finalizing a release checkout and authorizing any
production execution are subsequent user decisions. No production run or
automatic next task is part of Task 10.
