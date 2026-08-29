# Repository terminology policy

Project-owned names must describe the operation they perform. Bare or
data/statistical uses of the ambiguous word family are not permitted in source
identifiers, configuration, artifact paths, logs, reports, tests, or ordinary
documentation because they have several incompatible meanings in this project.

Use one of these terms instead:

| Operation | Required term |
|---|---|
| Preserve rows from several player counts | `concat_ks` or `concatenate` |
| Compute an equal-weight player-count summary | `equal_k_mean` |
| Compute a configured player-count summary | `declared_k_weighted_mean` |
| Combine root observations within one player count | `within_k_exposure_combination` |
| Join inputs without implying an estimator | `combine` |
| Reduce observations by declared grouping keys | `aggregate` |
| Run work concurrently | `process executor`, `thread executor`, or `workers` |
| Name executor lifecycle events and timing | `executor_start`, `executor_complete`, or another explicit `executor_*` term |

The terminology checker permits unavoidable external API spellings, such as
Python's `multiprocessing.Pool` attribute, and explicitly qualified process
concurrency terms such as “process pool” and `process_pool`. Unqualified
lifecycle or data-estimator names must use the precise executor, worker,
concatenation, combination, or aggregation term instead.

Maintained project prose, help text, docstrings, and test names use American
English. Historical reviews, remediation evidence, generated artifacts,
third-party symbols, and stable protocol values retain their original bytes.

Legacy names may appear only in an explicit migration map that rejects them;
they must never be accepted as aliases or used to locate artifacts.
