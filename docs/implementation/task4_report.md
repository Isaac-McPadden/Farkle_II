# Task 4 — independent H2H equivalence and superiority

Date: 2026-09-27. Contract: `first_release_task0_v1` (September 24).
This implements Task 4 of `Farkle_II_Implementation_Todos (1).md` on top of
Tasks 0, 9, 1A, 1B, 2, and 3, preserving their uncommitted work.

## Decisions and consumers

The production mega configuration now declares `delta_equivalence: 0.03`.
`practical_delta: 0.03` remains the independent planning effect and practical
dominance threshold. Neither tournament nor H2H game targets, execution,
attempt caps, RNG coordinates, or the exact power calculation changed.

`h2h_inference` publishes these independent boolean fields:

| Field | Meaning for a claim-eligible comparison |
| --- | --- |
| `statistical_advantage_a`, `statistical_advantage_b` | Holm rejection with the corresponding estimated direction |
| `practical_dominance_a`, `practical_dominance_b` | Simultaneous lower bound strictly above practical_delta, or upper bound strictly below its negative |
| `equivalent` | Both simultaneous bounds strictly inside the independently configured equivalence margin |
| `unresolved` | Eligible, with none of the above conclusions |
| `nonviable` | Claim eligibility failed; every inferential conclusion is withheld |

Equivalence does not depend on Holm non-rejection. Equality at either tolerance
boundary does not establish equivalence. Practical dominance retains the existing
strict boundary and Holm consistency check. A nonviable comparison remains
`unresolved_nonviable` for legacy display, while its separate `unresolved` flag is
false and `nonviable` is true. Legacy aggregate unresolved counts continue to
include nonviable comparisons, with a separate nonviable count.

Single-conclusion display labels are unchanged. Directional labels append
`_and_equivalent` when both conclusions hold, for example
`statistical_only_advantage_a_and_equivalent`. Independently configured margins
can also permit practical dominance and equivalence simultaneously; both remain
visible. Consumers validate that flags agree with intervals, thresholds, and
viability.

Dominance keeps directed statistical edges for equivalent small advantages and
attaches their equivalence flag and margin. Practical graphs, SCCs, fronts, and
the direct-dominance unique-best rule keep their existing meaning. Equivalence
is not a directed dominance edge and is never closed transitively: A equivalent
to B and B to C does not establish A equivalent to C. Agreement retains all flags
and the statistical direction. Reports count overlapping conclusions separately,
retain their full-precision interval values and flags in JSON, and show each
pair's completed support, effect, simultaneous interval, and combined class in
Markdown. Tolerance wording comes from configuration.

## Interpretation and numerical evidence

The declared +/-3 percentage-point reporting tolerance is a compromise between
resolving small differences and simulation cost. It is not a universal definition
of a negligible effect. All these H2H conclusions concern balanced win probability
relative to 50%, conditional on completed games and viable support.

The planner targets 80% rejection power for the zero-advantage null at a true
balanced-win deviation of `practical_delta`, over the declared seat scenarios
and required completed support. It does not target 80% power to prove that an
advantage exceeds the practical threshold, or to establish equivalence. Planned
detection power is distinct from attained support, observed precision, and
actual inferential decisions; no post-hoc power is computed from observed effects.

The current exact first-crossing planner reproduces the historical illustration:

| Quantity | Current calculation |
| --- | ---: |
| Hypothetical frozen finalists / unordered pairs | 150 / 11,175 |
| Roots / seat-advantage scenarios | 102, 103 / 0, 0.03, 0.06 |
| Completed games per root/order block | 2,191 |
| Completed games per order / per pair | 4,382 / 8,764 |
| Total planned completed games | 97,937,700 |
| Worst-scenario zero-advantage rejection power | 80.059292% |
| Synthetic seat-one wins in opposite orders | 2,322 / 2,060 |
| Estimated balanced-win deviation | 2.989503 percentage points |
| Simultaneous interval | [0.438854, 5.524466] percentage points |

For the synthetic family, every other pair has p=1. The example pair's Holm
adjusted p is 0.00024327, so it has statistical advantage. Its lower interval
bound is below 3 points and its upper bound exceeds 3 points: it establishes
neither practical dominance nor equivalence. These are illustrative integer
observations, not simulated or measured production Farkle performance.

Reproduce from the repository root with the root venv:

```powershell
.venv/Scripts/python.exe scripts/task4_h2h_example.py --output docs/implementation/task4_power_example.json --force
```

[The retained JSON](task4_power_example.json) records actual planner counts,
scenario powers, observations, decisions, and source/config hashes. The script
does no simulation, writes atomically, and retains an existing output unless
`--force` is supplied. The 150-finalist regression recomputes the allocation;
it does not hard-code historical power or interval values.

Smaller planning effects generally cost approximately inverse-square more
games under comparable conditions; workload claims must use the actual planner.
Changing only the equivalence margin changes classification, not sample targets.
The regression compares actual plans and schedule manifests with disabled,
0.015, and 0.03 margins: block requirements, total completed requirements,
maximum attempts, schedule hashes, and schedule rows remain identical. Code or
configuration identity changes can still invalidate saved artifacts. No reuse
bypass or extra games to force equivalence were introduced.

The 2% family-wise superiority target uses Holm. Equivalence uses the retained
Bonferroni simultaneous intervals, independently of Holm rejection. No single
joint 2% guarantee over superiority and equivalence declarations is asserted.

## Identity and validation

Inference/digest/agreement/report cache versions advance to 6/5/2/2 for both
single-root and pair stages. Their method identities are 3/2/2/2, and the
structure report contract is 6. H2H execution method and RNG scheme stay at 2.
Old exclusive-classification artifacts are not silently migrated or accepted as
new outputs. No canonical artifact paths were added.

Validation evidence is retained in [task4_validation.json](task4_validation.json).
The focused suite passed **119 tests**, and the clean-snapshot structural oracle
passed **2 tests**. Repository Ruff, Mypy over all 90 source files, targeted
Black/Pyright, and diff whitespace checks passed. The focused checks cover
significant small advantage plus equivalence in both
directions, practical dominance, wide unresolved intervals, exact boundaries,
nonviability, disabled/nondefault margins, authenticated inference resume,
contradictory flags, graph direction, nontransitive equivalence chains, configured
JSON/Markdown output, and unchanged schedules.

The full simulation-to-report oracle passed every correctness, authentication,
artifact inventory, reporting, and exact no-force resume assertion in its first
run. Its only failure was the final wall-clock check: **253.380 seconds** against
the unchanged **240-second** limit. The unchanged rerun passed every assertion
in **224.557 seconds**, giving **122 distinct passing checks** overall. Both
timings are retained because the earlier miss shows runtime variability; no
runtime assertion or production target was relaxed.

Integration used the isolated clean snapshot
`C:/Users/imcpa/AppData/Local/Temp/farkle-task4-verify-5anb_zgw`, commit
`b3245b6bc70ac8302258f4d79e1fb40753e956c7`, to satisfy the existing authenticated
code-identity contract. Implementation/config/example files match the working
copy byte for byte; the oracle helper matches after newline normalization.
The working branch was not committed or reset. The oracle now checks report
contract 6 and restricts its rejection-language ban to observed claims, allowing
the required methodological explanation of planned rejection power. It also
asserts zero statistical/practical/mixed claims for its nonviable comparisons.

Next is first-release Task 5, beginning with its bounded RNG preflight workload
and result contract. No production run, capacity qualification, or historical
artifact migration was performed for Task 4.
