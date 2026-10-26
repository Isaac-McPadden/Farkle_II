# Configuration option audit

Reviewed October 22, 2026 against configuration validation and current source
consumers. This audit distinguishes actual controls from protocol descriptions;
it does not change simulation rules, streams, numerical settings, or historical
artifacts.

The redundant profile flags have been removed. `profile.purpose` declares intent;
resolution and production qualification are calculated, and release eligibility
is published only by the final audit. See [Run profiles](config_reference.md#run-profiles).

## Fixed protocol values exposed as options

| Public field | Current behavior | Appropriate ownership |
| --- | --- | --- |
| `rng.scheme_version` | Validation and authenticated publication require exactly 2; changing it does not select another implemented stream scheme. | Internal RNG protocol identity, recorded in artifacts. |
| `rng.bit_generator` | Only `PCG64DXSM` is accepted. The RNG helper implements that generator directly; this string does not select it. | Internal RNG implementation identity. |
| `artifact_contract.*` | Artifact/schema/estimand/conditioning versions are fixed on the release path. Other fields enter freshness metadata; changing a version number does not implement a new statistical method. | Version constants owned by implementations and serializers. |
| `screening.interval_confidence` | The public statistical contract accepts only 0.95, though numerical helpers accept other confidence levels. | Fixed first-release method specification, unless support for alternative confidence levels is deliberately added. |
| `head2head.seat1_advantage_scenarios` | Only `[0, 0.03, 0.06]` is accepted. | The versioned H2H planning contract. |
| `head2head.candidate_cap_policy` | Only `balanced-tail` is accepted; no alternate policy is selected. | Candidate-family implementation identity. |
| `robustness.report_pareto`, `report_maximin` | Both must be true. The summaries are required and these fields do not control whether their producers run. | Required outputs, rather than boolean switches. |
| `orchestration.parallel_seeds` | The canonical two-root workflow rejects true; roots must run sequentially. | Orchestrator capability/constraint, rather than a supported toggle. |

These remaining fields are retained by this change. Moving protocol identity out
of the public configuration needs an explicit schema/freshness migration:
constants must still be authenticated in outputs, unsupported old declarations
must be rejected, and old results must never acquire new identities. Removing
the identity from artifacts would be incorrect.

## Unused option

`analysis.results_glob` has a dataclass definition and YAML examples but no
production consumer. Canonical input paths are resolved from configured roots,
player counts, and stage layouts. Changing this field can affect configuration
identity without changing the selected inputs. Retire it with an explicit
loader error; do not introduce directory discovery to justify the option.

## Controls that remain meaningful

Seeds, finite strategy grids, player counts, resolution targets, bootstrap
replicates, candidate contribution counts/caps, H2H effects/power/alpha and
equivalence tolerance, batch construction, and declared cross-k weights change
the design or conclusions. Worker counts, memory budgets, byte limits, scratch
paths, and progress intervals change execution. RNG lags, exhaustive omission,
matchup caps, and partition counts have actual consumers. None should be removed
merely because a checked-in example usually uses its default.

`sim.seed` also remains useful: it is the active root and single-root fallback;
`seed_list` declares the roots for orchestration. The loader normalizes and
checks their relationship. They are not two independent sources of randomness.

The audit was based on field definitions, validation restrictions, and source
reads, including indirect config property access. A field appearing only in
freshness metadata is not evidence that it controls a computation.
