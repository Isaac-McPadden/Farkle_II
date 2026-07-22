# Adversarial post-remediation review: artifact contracts

## Scope, evidence, and verdict

Reviewed commit `6be5f5fa11df77155621bfc81188c7515f38f8de` and the completed
`configs/fast_config.yaml` run at `data/results_seed_pair_32_33`. This was a
read-only review of canonical paths, schemas, adjacent sidecars, cache and
completion logic, migration inventory, report inputs, and produced artifact
identities. No source, test, configuration, or existing run artifact was
changed. Negative checks used system temporary directories.

**Verdict: unsound within this review's scope.** The run has complete adjacent
sidecar coverage, atomic byte binding, correct equal-k arithmetic, a genuinely
row-preserving HGB concatenation, and internally consistent H2H family/schedule
identities. Those are useful properties. They do not rescue the artifact
system's semantic contract: current validators accept artifacts in the wrong
canonical scope, do not validate declared columns or dtypes, do not bind
derived artifacts to source content, and can call changed same-size inputs
`complete_valid`. The pair-run configuration identity is also unreloadable and
internally contradictory. Consequently a successful release audit or
completion stamp does not establish that the artifact has the estimand,
schema, sources, or code/configuration identity claimed by its sidecar.

## Findings

### 1. Canonical scopes are violated in produced artifacts and the release audit accepts them

- **Severity:** High
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** `ArtifactScope` and the path API define only the six canonical
  scopes and describe `concat_ks` as row-preserving
  (`src/farkle/config.py:44-58,536-659`). However, a scan of the 186 non-block
  sidecars in both completed root analyses found 20 cases where the physical
  canonical directory disagrees with `sidecar.scope`:
  - six combine partitions are under
    `02_combine/concat_ks/all_ingested_rows_partitioned/` but declare `by_k`.
    The placement is constructed at `src/farkle/analysis/combine.py:120-136`,
    while the sidecar declares `BY_K` at lines 207-220.
  - twelve per-k rare-event shard/stat artifacts are under
    `04_game_stats/across_k/rare_events_shards/` but declare `by_k`. The shard
    path is derived from the across-k output at
    `src/farkle/analysis/game_stats.py:1869-1883`, and the sidecars declare
    `BY_K` at lines 1901-1957.
  - both roots store `feature_importance_long.parquet` under
    `07_hgb/across_k/`, while its sidecar declares `operation=concatenate` and
    `scope=concat_ks` (for example,
    `.../results_seed_32/analysis/07_hgb/across_k/feature_importance_long.parquet.sidecar.json:34,49`).
    The code selects the across-k directory at
    `src/farkle/analysis/run_hgb.py:480-481` and writes the concat there at
    lines 651-683.
- **Further evidence:** Scope mixing is not limited to directory/sidecar
  disagreement. `hgb_importance.json` is labelled `across_k/equal_k_mean` but
  contains the keys `2p`, `4p`, `5p`, and `overall`; the producer deliberately
  adds the per-k keys at `src/farkle/analysis/run_hgb.py:620-624` and the
  across-k key at lines 684-696. `future_simulation_proposals.parquet` is a
  row concatenation of per-k proposal frames (lines 698-726) but is also
  labelled `across_k`. `rare_events.parquet` is an unaggregated combination of
  per-k shards retaining `n_players`, yet its sidecar declares `across_k`
  (`src/farkle/analysis/game_stats.py:1978-2006`). These artifacts obscure the
  difference between a row union and a complete-support cross-k estimate.
- **Validator evidence:** `audit_sidecar_completeness` merely detects whether
  any canonical-scope component is present, then calls the generic byte
  validator without an expected scope (`src/farkle/analysis/release_audit.py:52-89`).
  Direct validation of the seed-32 HGB concat returned success while reporting
  `scope=concat_ks` and physical parent `across_k`. The release audit returned
  exit 0 for both root analyses and the pair analysis.
- **Consequence:** Consumers and reviewers cannot use the directory or sidecar
  scope as a trustworthy estimand boundary. A row-preserving union can be
  mistaken for an across-k calculation, and per-k state is interleaved with
  cross-k outputs.
- **Smallest reasonable remediation:** Move every artifact to the directory
  matching its output scope; split mixed JSON into by-k and across-k artifacts;
  put row unions in `concat_ks` (or non-estimand proposal material in
  `diagnostics`); and make validation derive the expected scope from the
  canonical path and reject any disagreement. Add a failure-path test for each
  scope, including nested artifacts.

### 2. Sidecar validation does not validate the declared schema or its sources

- **Severity:** High
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** `ArtifactSidecar` records only a global integer
  `schema_version` and a list of `consistency_columns`; it has no schema hash or
  dtype declaration (`src/farkle/utils/artifact_contract.py:203-235`). The
  validator checks that sidecar text fields and enum-like values are well
  formed (`:371-465`) and that artifact name, size, and SHA-256 match
  (`:587-619`). It never opens Parquet to compare columns, column order, nullability,
  or dtypes, and never compares `consistency_columns` to the artifact.
- **Negative check:** A temporary Parquet file containing only
  `wrong_column: int8` was written with a valid sidecar claiming
  `consistency_columns=["required_but_absent"]`; `validate_artifact_sidecar`
  accepted it. Thus the generic contract cannot catch either missing columns
  or wrong dtypes. Some individual consumers add required-column checks, but
  those are inconsistent and generally do not enforce exact dtypes.
- **Source evidence:** `source_artifacts` stores path strings only
  (`src/farkle/utils/artifact_contract.py:288-340`). Validation never checks
  their existence, current hashes, sidecars, operations, scopes, or schemas.
  `input_manifest_hashes` can bind explicitly supplied manifests, but 174 of
  the 186 non-block fast-run sidecars had an empty list. A sidecar therefore
  remains valid after an upstream artifact is replaced.
- **Consequence:** A hash-valid artifact can be structurally unusable or can
  describe calculations over sources different from the ones that produced
  it. Stale-source and schema failures surface only if a particular downstream
  module happens to perform an additional check.
- **Smallest reasonable remediation:** Persist a canonical Arrow schema
  fingerprint (and, where appropriate, an exact schema declaration), bind each
  source to its artifact and sidecar SHA-256 plus expected operation/scope, and
  make the shared validator enforce these identities. Keep artifact-type
  specific semantic validators for support and family/schedule invariants.

### 3. Cache and lifecycle freshness can reuse semantically stale outputs

- **Severity:** High
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** Stage input fingerprints consist only of path, size, and mtime
  (`src/farkle/utils/stage_completion.py:126-154`); freshness compares those
  values and input-newer-than-stamp status (`:236-245`). In a temporary negative
  check, replacing `AAAA` by `BBBB` and restoring the same mtime left the stage
  as `complete_valid`.
- **Lifecycle evidence:** Sidecars are checked only when the caller supplies
  `sidecar_artifacts` (`src/farkle/utils/stage_completion.py:246-251`). The
  stamp does not record which outputs require sidecars (`:267-333`), and the
  top-level `StageRunner` resolves completion without that argument
  (`src/farkle/analysis/stage_runner.py:156-179`). A temporary valid stage whose
  sidecar was then removed resolved as `complete_valid` without the optional
  argument and `complete_stale` with it.
- **HGB evidence:** HGB has no top-level completion stamp in the root stage
  plan (`src/farkle/analysis/__init__.py:111-116`). Its local reuse test checks
  only existence, adjacent-sidecar existence, output mtime, and generic sidecar
  byte validation (`src/farkle/analysis/hgb_feat.py:62-91`). It does not compare
  HGB hyperparameters, config hash, source identity, method, or scope, and the
  root plan does not pass `force` to it. A changed HGB configuration can
  therefore invoke the stage and still reuse old outputs.
- **Consequence:** `complete_valid` and cache hits are not semantic guarantees.
  Changed inputs, missing sidecars, or changed model settings can silently
  preserve obsolete derived results.
- **Smallest reasonable remediation:** Store content identities for every
  input and required output sidecar in the completion stamp, make those checks
  mandatory rather than caller-optional, and give HGB the standard
  stage-config/freshness/`--force` lifecycle. A low-I/O manifest tree hash is
  appropriate for large immutable shard sets.

### 4. The pair-run configuration and configuration hashes are not coherent or reproducible

- **Severity:** High
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** `RootPairRunContext` creates `pair_base` with
  `dataclasses.replace(first.config, sim=pair_sim)`
  (`src/farkle/orchestration/run_contexts.py:137-159`). Because `config_sha` is
  an `init=False` field, that replacement loses the root's hash; `from_base`
  then copies the resulting `None` at lines 75-84. The pair active config is
  written without reassigning a hash (`src/farkle/orchestration/two_seed_pipeline.py:327-338`).
- **Produced evidence:** `data/results_seed_pair_32_33/active_config.done.json:3`
  records `config_sha: null`, while `pipeline_health.json:2` records
  `1909cc...`. All 152 examined root sidecars use `1909cc...`, while all 34
  non-block pair sidecars use `134bbb...`; for example,
  `pairwise_inference.parquet.sidecar.json:9`. The fallback hash differs because
  `effective_config_dict` serializes all dataclass fields except `config_sha`
  and `_stage_layout` (`src/farkle/config.py:1591-1611`), including the
  `RunContextConfig` private runtime fields.
- **Reload evidence:** The persisted pair `active_config.yaml` begins with
  `_analysis_root_override` and `_root_input_layout_override`. Passing it to
  `load_app_config` fails with `ValueError: Unknown top-level config section(s)`.
  It is therefore not a reproducible active configuration.
- **Consequence:** Pair artifacts cannot be unambiguously tied to the published
  run configuration or regenerated from the persisted config. Cache and
  provenance comparisons disagree depending on whether they use pipeline,
  root, pair-sidecar, or completion metadata.
- **Smallest reasonable remediation:** Preserve one logical run config hash
  across root/pair path wrappers (or explicitly publish a parent hash plus a
  separately named context hash), exclude runtime-only private fields from the
  persisted/hashable application config, and require the written active config
  to reload and reproduce its recorded hash in a test.

### 5. Method sidecars are too generic to carry the required inferential provenance

- **Severity:** Medium
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** The sidecar type has no first-class multiplicity, candidate
  family identity, schedule identity, equivalence margin, or schema identity
  fields (`src/farkle/utils/artifact_contract.py:203-235`). The default method
  contract usually contains only `kind` and a copy of `operation`
  (`:345-364`); `parameters` is optional and is absent from the inspected
  artifacts. `code_revision` defaults to the accepted nonblank string
  `unknown` (`:272-297`). All 186 inspected non-block fast-run sidecars record
  `code_revision=unknown`.
- **H2H evidence:** The pairwise artifact's table has separate ordinary score
  bounds, Holm fields, Bonferroni simultaneous bounds, equivalence fields, and
  `family_hash`, but its adjacent sidecar reduces this to
  `operation=seat_adjusted_score_inference` and
  `uncertainty_method=independent_two_proportion_score_v1_holm`
  (`src/farkle/analysis/h2h_inference.py:645-681,722-730` and produced sidecar
  lines 56-81). It contains neither the family-hash value nor schedule-hash
  value, family alpha, per-pair Bonferroni alpha, ordinary alpha, practical
  delta, equivalence margin, nor an explicit statement separating Holm
  decisions from simultaneous practical bounds. Planning/execution sidecars
  similarly list `family_hash`/`schedule_hash` only as grouping or consistency
  column names, not identity values (`src/farkle/analysis/h2h_schedule.py:505-525,650-700,730-764`).
- **Mixed-source evidence:** `candidate_family` consumes one `cross_seed` and
  one `across_k` artifact, but its singular `source_scope` is copied only from
  the win-rate source (`src/farkle/analysis/candidate_family.py:560-593`). It
  cannot accurately describe both source scopes.
- **Other methods:** The per-k performance sidecar labels uncertainty
  `wilson_and_batch_t_interval` but does not encode that Wilson is only a
  workload-resolution check, the t confidence/df, or separate method objects
  (`src/farkle/analysis/performance.py:664-675`). Across-k analytic and joint
  resampling outputs do have distinguishable strings
  (`independent_k_variance_sum` versus
  `joint_deterministic_batch_resampling`), and TrueSkill/HGB/two-root
  conditioning strings correctly constrain claims. The defect is inadequacy,
  not that every label is wrong.
- **Consequence:** The artifact bytes may contain enough columns to reconstruct
  a decision, but the adjacent provenance contract does not independently bind
  which family, schedule, interval, multiplicity rule, equivalence rule, or
  code revision produced it. A sidecar cannot prevent cross-family or
  cross-schedule substitution by itself.
- **Smallest reasonable remediation:** Version typed method contracts with
  required parameters for each method family. H2H contracts should contain
  concrete family/schedule hashes, score-test and interval IDs, ordinary and
  simultaneous alpha rules, Holm/Bonferroni roles, practical/equivalence
  margins, and replication design. Require an actual commit/build identity and
  support a list of source scopes/identities.

### 6. The pair migration report inventories only the first root

- **Severity:** Medium
- **Confidence:** High
- **Classification:** Confirmed coverage defect
- **Evidence:** `migration_audit.inventory` always scans `cfg.results_root`
  (`src/farkle/analysis/migration_audit.py:50-76`). The pair context overrides
  only `analysis_dir`, not `results_root`
  (`src/farkle/orchestration/run_contexts.py:87-115,148-159`). Accordingly, the
  pair report at
  `seed_pair_analysis/08_reporting/diagnostics/migration_report.json:7` says
  `scan_root=data/results_seed_pair_32_33/results_seed_32`; it does not scan
  seed 33 or the pair root. It nevertheless reports zero ignored artifacts.
- **Observed qualification:** A separate filename search found no retired
  patterns anywhere under this completed pair, so this fast run's zero happens
  to be correct. The implemented inventory coverage is still incomplete.
- **Consequence:** Retired artifacts placed under the second root or pair
  analysis can be missed while the canonical pair report presents a clean
  inventory.
- **Smallest reasonable remediation:** Give `RootPairRunContext` an explicit
  migration scan root equal to `pair_root` (or explicitly scan and report all
  three roots) and test retired files in the second root and pair analysis.

## Confirmed properties and successful checks

- The release sidecar audit returned exit 0 for both root analyses and the full
  pair analysis. The pair analysis contains 11,434 adjacent sidecars, including
  one for each of 11,400 H2H block Parquets. No orphan, missing, hash-mismatched,
  `.tmp`, `.partial`, or staged artifact was observed. Atomic publication and
  interruption behavior are also directly tested in
  `tests/unit/utils/test_artifact_contract.py`.
- The main `concat_ks` output has 326,800 rows in each root, exactly
  `172,000 + 86,000 + 68,800` from k=2/4/5 partitions. Creation performs a
  full streaming row-order/value equality check, not merely a row-count check
  (`src/farkle/analysis/combine.py:64-101,254-325`). The smaller HGB concat was
  independently recalculated and exactly equalled the 30-row ordered union of
  its three per-k importance tables in both roots. Its computation is sound;
  its placement/name contract is not.
- Each root's 80 canonical performance rows had complete support at all three
  configured k values, positive exposure (minimum 4,300), exact
  `mean_k(win_rate - 1/k)`, and analytic MCSE equal to
  `sqrt(sum_k MCSE_k^2)/3` to floating-point precision. Raw exposure counts did
  not enter the equal-k mean.
- TrueSkill rating sidecars accurately identify sequential per-root/k ratings,
  ordered-game replication, and model sigma as screening-only. Candidate
  contribution sidecars identify within-root/k mu percentiles, complete
  support, and descriptive screening. HGB artifacts consistently use
  `predictive_association_not_causal`, and cross-seed artifacts use
  `unconditional_fixed_simulation_design`/`root_pair_stability` rather than a
  root-population model.
- The H2H payloads themselves are internally coherent: 76 frozen candidates
  imply 2,850 unordered pairs; the block manifest and execution counts contain
  11,400 pair/root/order rows over roots 32/33 and orders 0/1; all schedule rows
  have the published family hash `07a2...1733` and schedule hash
  `b0fd...3c09`; inference contains exactly 2,850 decisions and retains 535
  unresolved comparisons. These identities are in the data/JSON payloads even
  though the sidecars inadequately bind them.
- `structure_reporting` selects the combined-root canonical cross-seed
  performance artifact for a two-root report, rejects the wrong scope or
  operation, filters `estimate_scope=combined_roots`, and refuses incomplete k
  support (`src/farkle/analysis/structure_reporting.py:23-75`). No report was
  found consuming a retired performance or rating artifact.
- Focused tests passed: 73 tests across artifact contract, stage completion,
  combine, HGB, release audit, structure reporting, H2H schedule, and H2H
  inference. Passing tests demonstrate implemented behavior; the temporary
  counterexamples above show important contracts those tests do not cover.

## Final verdict

**Unsound within this review's scope.** The current produced tables include
several correctly wired and recalculable results, and atomic adjacent-sidecar
coverage is excellent. But the artifact contract is not a semantic validator:
it permits real scope violations, fictitious schemas, stale sources and cache
reuse, incomplete inference provenance, and contradictory pair configuration
identity. Those are core release-contract failures, not documentation-only
issues.
