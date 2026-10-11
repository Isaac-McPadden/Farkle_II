# Task 8 - automatic analysis guide

Contract: September 24 `first_release_task0_v1`, first-release implementation backlog.
Status: complete through validation and documentation.

## Products and interpretation

The existing reporting stage publishes authenticated `analysis_guide.json` and
`analysis_guide.md` in `pair_cfg.scope_dir("reporting", ArtifactScope.DIAGNOSTICS)`.
The same getters work for an explicitly labeled single-root analysis. The JSON
has exactly the eight Task 0 envelope fields. It preserves all ten workbook
groups, 62 targets, ten substantive B/C notes, five framing statements and the
heading/label cells. The frozen mapping contains definitions and claim limits,
not expected scientific findings.

Each target carries coverage, computation, estimand, conditioning, denominator,
weighting, source-specific precision, actual published values/statuses, counts,
paths, actual schemas and limits. Per-question excerpts are deterministic source
examples, not selected favorable results. Full distributions, parameter tables,
exact matched IDs/folds and retained sufficient statistics remain at their linked
canonical paths. Source row counts do not substitute for exposure/game support.

Completed score/rank/margin products and stopped scores retain their different
support. U exposes the five selected completed-outcome summaries, null/withheld
precision, full status counts and published support/precision ranges. Behavior
uses existing completed/safety/all-attempt sums and actual-turn denominators;
round proxies remain separate. Opportunity counts games once and retains the
fixed equal-k root-specific headline. Per-k/equal-k, specialist, Pareto and
maximin summaries retain their descriptive role.

H2H presentation starts with fronts, cycles, unresolved comparisons and explicit
pairwise equivalence. The guide separates the immutable planning alternative and
power, attained attempted/completed/safety support, ordinary/simultaneous bounds
and widths, and independent statistical/practical/equivalence flags. Fronts and
unresolved groups are not equivalence classes. Target detection power does not
promise decisive fronts. The authorization artifact records the exact approved
plan separately from the configured cap and statistical targets.

Parameter headlines also retain the full published matched-support status counts,
held-out folds and importance, independently of bounded row excerpts. Windows
paths containing spaces use valid angle-wrapped Markdown link targets. Diagnostic
check inventories render as compact tables of existing fields.

The dictionary carries published columns/types, byte and sidecar hashes,
code/config/method identities, support/conditioning/weighting metadata and fields
suitable for manual visuals. Diagnostic evidence retains fault-to-check records,
minimum support, clean/fault family counts and detections/misses, preflight scope,
measured blind spots and actual production-monitor results. Exhaustive omissions
are explicit. Correctness preflight, stable batches and root agreement do not
establish comprehensive independence or calibrated false-alert control.

Methodological resolutions preserve the difference between outcome spread and
win-probability uncertainty, TrueSkill disagreement and identified opponent
composition effects, and absolute equal-k probability-point advantage and
relative lift. Parameter responses/nonadditivity, held-out fold metrics and
permutation importance remain exploratory/predictive finite-grid associations.
Broad composition sensitivity, fixed-matchup entropy, state-level decisions and
prescribed strategy-performance ranking controls remain excluded/deferred.

## Publication and boundedness

`analysis_guide.py` extends `structure_reporting.py`; it is a consumer, not a new
pipeline or statistical producer. All sources use current configured root/pair
stage ownership. Required missing/tampered artifacts or empty required
support/summary/precision/population tables fail publication. They cannot become
zeros or an empty successful guide. Valid empty sparse histograms and unsupported
parameter products retain their source support/status evidence.

Reads use the existing byte-bounded Arrow iterator. Large F/B/B7/RI inventories
retain metadata counts and bounded examples; Q7 retains a bounded prefix with
explicit prefix-only status counts, while existing Task 7 reporting supplies full
behavior totals. Q/U and bounded parameter products retain complete status counts;
U ranges reduce published values, without recalculating uncertainty. Up to six
canonical initial rows and 128 metric/conditioning/status representatives are
retained per product. No accumulator grows with games or histogram bins.
Reporting reads serially within its owning process to retain the shared I/O/byte
budget; existing producers retain their parallel policies and checkpoints.

The entire guide model is normalized into JSON types before rendering. Markdown
uses only that JSON model and deterministic column/key order; it performs no
independent result calculation. Existing atomic artifact writers publish data
then sidecar. Markdown's only source is the authenticated guide JSON. Both files
are required stage outputs and final-audit inventory members. Audit validates
question/reference coverage, the exact JSON rendering and required completion
ownership. Stale-parent recovery preserves a compatible valid guide child after
checking current code/config/method and exact captured upstream identities.

Published nonfinite IEEE values have a lossless JSON representation using
`{"published_nonfinite":"Infinity"}`, `"-Infinity"` or `"NaN"`. Null and zero
remain separate. The raw oracle exposed a signed infinite standardized root
discrepancy: the raw difference was nonzero while estimated reference MCSE was
zero. This is a diagnostic scale edge case, not an infinite probability effect.
The guide retains the source columns, denominators and statuses and explains
the encoding; it does not repair the producer's statistic or invent uncertainty.

Reporting contract advances from 8 to 9 and adds `analysis_guide_method_version=1`.
Metrics cache remains 6, HGB cache remains 2, reporting registry cache remains 2;
the reporting freshness and method identities own this change. No RNG, simulation,
bootstrap, HGB fitting, H2H planning or production configuration identity is
changed independently of the repository-wide code fingerprint.

## Validation

The implementation changes the existing reporting owner, AppConfig guide-path
getters, reporting identities, plan-required outputs and final-audit checks.
It adds two runtime modules (frozen questions and extraction/rendering), three
test/helper modules and this report/validation record. Existing structural and
raw oracles include the new products; five context maps record their ownership
and limits. The validation JSON lists every changed existing/new implementation
file against the starting inventory rather than attributing unrelated dirty
workspace changes to Task 8.

The final isolated snapshot is `6b483974d388f7e66a6a028fba94d777f186e2a3` at
`C:\Users\imcpa\AppData\Local\Temp\farkle-task8-final-x5uw_lcd`. Its ignored
venv junction uses the root `.venv` (Python 3.12.10); no main-worktree commit or
reset was made. All 100 runtime Python files match the working implementation.
The source inventory SHA-256 is
`1870c33178e78eed438ca9fcb7507ca05def8a067c2ddad8f812c906fdfcf5fd`.

The final snapshot passes **153 focused tests in 451.359 seconds**
(JUnit suite time), comprising the 136-test Task 7 baseline and 17 guide tests.
These cover coverage,
extraction/denominators, source statuses, claim boundaries, actual column
references, pure rendering, authenticated outputs, missing/corrupt required
inputs, unchanged no-force replay and missing-Markdown recovery. The retained
acceptance JUnit is `data/task8-focused-completion.xml`; its one inherited
warning concerns pytest's `record_property` compatibility with xunit2. The
pre-finite snapshot passed 152 focused tests but failed its raw oracle on the
signed infinite discrepancy; it is retained as failed development evidence.
The next raw run completed the pipeline and passed final audit but exposed an
incorrect new oracle assertion comparing an artifact identity digest to its
byte digest. The corrected assertion compares `content_sha256` to file bytes
and the entire authenticated artifact identity separately. It passes against
that completed producer run as a development check. Final acceptance uses a
fresh final-snapshot run; neither failed test is credited as completion.

Ruff over `src`, `tests` and `scripts`, Mypy over all 100 source files, Pyright
over seven affected runtime modules, and Black over 15 touched/new Python files
pass. `git diff --check` passes. The full raw simulation-to-report oracle
passes in **984.954 seconds** (986.980 seconds JUnit suite time), including
**358.576 seconds** across both required preflights. Each preflight stayed
within its declared 1,800-second ceiling, and the complete oracle stayed within
its 3,840-second ceiling. This is a bounded implementation measurement, not
a production estimate or a controlled overhead comparison with Task 7.

The raw oracle completes root/root/pair byte-deep final audits, reconciles
actual guide values, columns, counts/statuses, authorization and diagnostic
coverage, and preserves every stable workflow file hash on no-force replay.
The original plan approval is reused without prompting again or adding H2H
blocks. Final health is `complete_success`, audit is `passed`, failures are
empty and `release_eligible` is false for this development validation. The
retained JUnit is `data/task8-full-oracle-completion.xml`. Its one warning is
the same inherited pytest xunit2/property warning. Guide unit tests also check
unchanged data/sidecar modification times and missing-Markdown recovery.
Structural reporting fixtures are explicitly synthetic and do not replace real
producer/preflight evidence. The raw oracle independently reconciles guide values,
counts/schemas/statuses, diagnostic coverage and authorization to current producers.

Development found and fixed dictionary-order differences after JSON publication,
JSON type normalization needed for valid-child reuse, duplicate source inventory
entries, and a structural toy manifest rewrite that left a stale sidecar. The toy
now publishes its replacement atomically; no incompatible production artifact was
re-sidecarred. An early snapshot Pyright invocation could not resolve the root
venv; the final isolated snapshot points its ignored `.venv` junction at the
repository Python 3.12 environment.

All pre-existing working files remain; the main repository was not committed or
reset. Production configuration and scientific budgets are unchanged. The separate
terminology gate continues to report the four inherited `pooled` uses in unchanged
`outcome_publication.py`, including its durable directory name. That gate does not
pass. New guide wording introduces no additional terminology findings.

## Accepted G products

These are authenticated current-code products from the fresh bounded fixture,
using its configured Task 0 scope. They are not production-run findings.

- [analysis_guide.json](../../data/task8-oracle-completion/test_raw_simulation_to_authent0/oracle_results_seed_pair_11_22/seed_pair_analysis/08_reporting/diagnostics/analysis_guide.json)
  SHA-256: `d9ef2ade004ff1f80931ef4d0110db15a4002edf1d3c2ea566d5c1e9a0f6e2ea`; 9,268,913 bytes.
- [analysis_guide.md](../../data/task8-oracle-completion/test_raw_simulation_to_authent0/oracle_results_seed_pair_11_22/seed_pair_analysis/08_reporting/diagnostics/analysis_guide.md)
  SHA-256: `973117690ec34dbd91c8cff77bad0a819c586cb6ade4051db6d602f76f297f4c`; 2,915,711 bytes.

The JSON covers ten groups and 62 workbook targets with 128 actual source
products: 55 available, three partial, three deferred and one excluded question.
Those statuses describe question coverage, not universal estimability; source
support and precision statuses remain attached to each quantity. Nine note
resolution records cover all ten substantive B/C cells. The Markdown is the
exact JSON rendering, and all 1,430 parsed local file links resolve, including
paths with spaces. Product/sidecar/completion hashes, final audit evidence,
test properties, commands, preservation inventory and changed files are in
[task8_validation.json](task8_validation.json).

## Remaining boundary

No production simulation, dashboard/figure suite, new statistical analysis,
automatic extra sampling or production cap override is performed. These results
qualify the bounded implementation, not production capacity. Task 10 integrated
qualification, launcher/storage checks and capacity refresh is the next dependency.
Task 8 stops at an evidence-locatable guide and explicit limits.
