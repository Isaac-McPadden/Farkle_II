# Repository terminology policy

Project-owned names must describe the operation they perform. The prohibited
ambiguous word family is not permitted in source identifiers, configuration,
artifact paths, logs, reports, tests, or ordinary documentation because it has
several incompatible meanings in this project.

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

The terminology checker permits only unavoidable external API spellings, such
as Python's `multiprocessing.Pool` attribute. Project-owned wrappers, local
variables, comments, and user-facing messages around those APIs must still use
the precise executor/worker terminology.

Legacy names may appear only in an explicit migration map that rejects them;
they must never be accepted as aliases or used to locate artifacts.
