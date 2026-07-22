# Architecture and conceptual coherence review

## Review scope and evidence

Reviewed commit `6be5f5fa11df77155621bfc81188c7515f38f8de` and the completed
`configs/fast_config.yaml` run at
`data/results_seed_pair_32_33`. This was a review-only inspection of the
current workflow, configuration and path contracts, representative sidecars,
the pair manifest, and generated reports. No statistical formula was treated
as correct merely because its stage completed.

The completed run reports `complete_success` in
`data/results_seed_pair_32_33/pipeline_health.json`. Representative artifacts
use the intended evidence labels: root-combined performance is `cross_seed`,
H2H inference is `h2h_2p`, reporting is `diagnostics`, TrueSkill contribution
is screening-only, and HGB is labelled predictive association rather than
causal evidence. The generated report also identifies `execution_scope` as
`root_pair`, describes root agreement as fixed-root reproducibility, retains
535 unresolved comparisons, and permits no unique-best claim
(`data/results_seed_pair_32_33/seed_pair_analysis/08_reporting/diagnostics/structure_report.json:13,37,46,35086-35096`).

Focused validation ran:

- `python -m pytest -q -p no:cacheprovider` over CLI, seed workflow, stage
  registry, HGB, and structure-reporting tests: 69 passed.
- `python scripts/check_terminology.py`: passed.
- Read-only probes of public argument parsing, pair-config loading, HGB cache
  invalidation, and stage-definition resolution.
- SHA-256 comparison of the two root strategy manifests; they are identical in
  this fast run, which limits the current-run consequence of the asymmetric
  manifest lookup discussed below.

Passing tests establish execution coverage only. Several adversarial cases in
the findings are absent from those tests.

## Workflow trace and boundary assessment

| Phase | Active owner and artifact scope | Assessment |
| --- | --- | --- |
| Config and simulation | `AppConfig`, CLI, simulation runner; per-root simulation directories | Explicit root configs and coordinate-owned simulation are structurally separated from analysis. Public CLI parsing and resolved worker ownership are not coherent; see A1 and A3. |
| Ingest and curate | Root-owned `00_ingest/by_k` and `01_curate/by_k` | Clear row-preserving root/k ownership. |
| Concatenation | Root-owned `02_combine/concat_ks` | Actual artifact `all_ingested_rows.parquet` is in `concat_ks`; no aggregation semantics were observed in this stage. |
| Metrics and diagnostics | Root-owned `03_metrics/{by_k,across_k,diagnostics}`, then game/RNG stages | Statistical transforms are mostly isolated from the stage runner and carry explicit sidecars. |
| TrueSkill, HGB, screening | Root-owned model/screening stages; pair-owned TrueSkill percentile contribution | TrueSkill remains root/k before complete-cell percentile aggregation. HGB has lifecycle and path/sidecar contract defects; see A4 and A5. Screening is visibly descriptive. |
| Two-root stability and family freeze | One `seed_pair_analysis` workflow using both root contexts | Root stability, pair TrueSkill contribution, and candidate freeze execute in the pair tail. The pair configuration still inherits first-root path identity; see A2. |
| H2H plan, execution, inference, digestion | Pair-owned `h2h_2p` stages | Power plan, execution state, inference, dominance, and unresolved comparisons are separate artifacts. No H2H stage directory exists under either fast-run root. |
| Agreement and reporting | Pair-owned agreement and diagnostics report | Descriptive screening, predictive/model evidence, fixed-root reproducibility, and formal H2H decisions remain visibly distinct. Claim language is conservative and preserves incomparability. |

No current consumer of `ratings_k_weighted` or `ratings_pooled`, no seed
random-effects inference, and no forced H2H total order was found. The current
fast-run migration report found no retired artifacts in the directory it
scanned, but that scan did not cover the full pair run (A2). A reachable
`players = 0` fallback and active `long` filename remain in HGB (A5).

## Findings

### A1 — The installed CLI silently discards documented seed overrides and other unknown arguments

- **Severity:** High
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** The installed entry point is `farkle.cli.main:main`
  (`pyproject.toml:49-50`). It calls `parse_known_args` and discards the unknown
  list (`src/farkle/cli/main.py:298-303`). Seed overrides are global parser
  options (`src/farkle/cli/main.py:58-73`), while the `two-seed-pipeline`
  subparser itself declares only `--force` (`src/farkle/cli/main.py:242-251`).
  The documented command places the override after the subcommand:
  `farkle ... two-seed-pipeline --seed-pair 42 43`
  (`cli_args.md:130-142`). A read-only parser probe returned
  `seed_pair=None` and unknown arguments `['--seed-pair', '42', '43']` for
  exactly that ordering. Placing the option before the subcommand produced
  `[42, 43]`. A second, non-installed parser accepts these options and uses
  strict `parse_args` (`src/farkle/orchestration/two_seed_pipeline.py:368-393`),
  so the two entry points already differ.
- **Consequence:** A user can request roots 42/43, receive no error, and run the
  potentially very expensive configured roots instead. Typos in other options
  are likewise silently accepted. Artifact paths and scientific conditioning
  can therefore disagree with the user's command even though the run succeeds.
- **Smallest reasonable remediation:** Make the installed CLI use strict
  parsing. Put two-seed controls on the `two-seed-pipeline` subparser (or use a
  shared parent parser that intentionally supports both positions), remove or
  delegate the duplicate parser, and add tests using the documented argument
  order plus a test that arbitrary unknown options fail.

### A2 — Pair configuration inherits first-root identity, is not reloadable, and produces an incomplete migration claim

- **Severity:** Medium
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** `RootPairRunContext` clones the first root's config and changes
  `sim.seed_list` plus only an `analysis_dir` override
  (`src/farkle/orchestration/run_contexts.py:137-160`). `RunContextConfig` is an
  `AppConfig` subclass with private dataclass fields
  (`src/farkle/orchestration/run_contexts.py:46-52`), while
  `effective_config_dict` removes only `config_sha` and `_stage_layout` after
  `dataclasses.asdict` (`src/farkle/config.py:1605-1611`). The persisted pair
  config consequently contains `_analysis_root_override` and
  `_root_input_layout_override` and still has
  `io.results_dir_prefix: results_seed_pair_32_33\results_seed_32`
  (`data/results_seed_pair_32_33/active_config.yaml:1-2,83`). Loading this file
  through the supported `load_app_config` raises `ValueError: Unknown top-level
  config section(s)`. The reporting-stage migration audit scans
  `cfg.results_root` (`src/farkle/analysis/migration_audit.py:50-58,79-90`), and
  its pair-owned output says it scanned only
  `data\results_seed_pair_32_33\results_seed_32`
  (`data/results_seed_pair_32_33/seed_pair_analysis/08_reporting/diagnostics/migration_report.json:3-7`). H2H execution also
  obtains its strategy manifest through the pair config's singular
  `cfg.strategy_manifest_root_path()` (`src/farkle/analysis/h2h_schedule.py:1083-1085`),
  which resolves to root 32. The root-32 and root-33 manifests happen to be
  byte-identical in this run, so no fast-run H2H mismatch was observed.
- **Consequence:** The advertised active configuration cannot reproduce or
  even reload the pair run. Pair-owned code can accidentally operate on the
  first root only. The generated `ignored_artifact_count: 0` is not evidence
  about root 33 or the pair-analysis tree, despite being embedded in the pair
  report.
- **Smallest reasonable remediation:** Represent pair paths in a dedicated
  context rather than by subclassing and partially overriding `AppConfig`.
  Persist a reloadable user/effective config without runtime-only fields, give
  pair operations an explicit `pair_root`, make migration inventory accept all
  owned roots, and validate both root strategy manifests before selecting one
  immutable manifest for H2H.

### A3 — Two-seed orchestration overrides the documented analysis and H2H worker owners

- **Severity:** Medium
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** Fast config declares `sim.n_jobs: 12`, `analysis.n_jobs: 4`,
  `ingest.n_jobs: 3`, and `head2head.n_jobs: 0`
  (`configs/fast_config.yaml:15,40-47,57-58`). Two-seed budgeting derives both
  simulation and analysis process counts from `cfg.sim.n_jobs`, passing the
  full per-root budget as the analysis override
  (`src/farkle/orchestration/two_seed_pipeline.py:92-112`), then mutates the
  root configs (`src/farkle/orchestration/two_seed_pipeline.py:125-140`). The
  persisted root and pair configs therefore record `analysis.n_jobs: 12`
  (`data/results_seed_pair_32_33/active_config.yaml:19-26`) even though the
  source config says 4. Canonical H2H orchestration explicitly passes
  `inner.analysis.n_jobs` (`src/farkle/analysis/__init__.py:139-143`), bypassing
  the `head2head.n_jobs` fallback implemented by
  `execute_h2h_schedule` (`src/farkle/analysis/h2h_schedule.py:1087`). This
  conflicts with the documentation that `sim` owns simulation workers and
  `analysis` owns analysis workers (`docs/config_reference.md:91-97`).
- **Consequence:** Configured resource limits do not mean what their section
  ownership says, and `head2head.n_jobs` is dead in the canonical workflow.
  This can oversubscribe CPU/RAM and makes provenance confusing because the
  original config hash is retained while resolved operational values are
  mutated.
- **Smallest reasonable remediation:** Treat each section's `n_jobs` as that
  stage's requested ceiling, apply only an explicit global concurrency cap,
  and pass `None` or `cfg.head2head.n_jobs` to H2H execution. Record resolved
  worker policies separately from the immutable statistical config and bind
  both clearly in run metadata.

### A4 — HGB uses an mtime cache outside the canonical lifecycle and can certify stale model outputs

- **Severity:** Medium
- **Confidence:** High
- **Classification:** Confirmed defect
- **Evidence:** `hgb_feat.run` treats outputs as fresh solely when they and
  their sidecars exist, validate internally, and are newer than inputs
  (`src/farkle/analysis/hgb_feat.py:62-91`). It does not compare HGB parameters,
  stage-scoped config hash, freshness key, or a completion stamp. It can also
  log a missing input and return successfully (`src/farkle/analysis/hgb_feat.py:43-52`).
  The root stage plan declares HGB without required outputs or a completion
  stamp (`src/farkle/analysis/__init__.py:111-115`), so `StageRunner` records a
  healthy stage after either return. Both fast-run root HGB directories lack a
  `*.done.json`, while their manifests record HGB as healthy. In a read-only
  probe, loading root 32's config, changing `hgb.max_depth` from 6 to 99, and
  replacing the model runner with a mock still resulted in
  `run_hgb_called=False`.
- **Consequence:** A change to `max_depth`, estimator count, folds, permutation
  repeats, or proposal limit can silently reuse outputs from a different model
  contract. This does not contaminate candidate freezing in the current design,
  but it makes the canonical predictive/model-based evidence stale while the
  workflow reports success.
- **Smallest reasonable remediation:** Give HGB the standard stage completion
  contract keyed to its inputs, sidecars, and `hgb` config scope; declare its
  required outputs and stamp in the root plan; and fail closed on missing
  canonical performance or strategy-manifest inputs.

### A5 — HGB directory, sidecar, and retired-sentinel semantics disagree

- **Severity:** Medium
- **Confidence:** High
- **Classification:** Confirmed path/metadata defects plus a plausible retired-semantics risk
- **Evidence:** `run_hgb` creates one `combined_dir` with
  `cfg.across_k_dir("hgb")` (`src/farkle/analysis/run_hgb.py:478-481`), then
  writes `feature_importance_long.parquet` there while labelling it
  `scope="concat_ks"` and `operation="concatenate"`
  (`src/farkle/analysis/run_hgb.py:651-683`). The produced artifact confirms
  the mismatch: its path is
  `results_seed_32/analysis/07_hgb/across_k/feature_importance_long.parquet`,
  while its sidecar says `scope: concat_ks`
  (`data/results_seed_pair_32_33/results_seed_32/analysis/07_hgb/across_k/feature_importance_long.parquet.sidecar.json:34,49`). The same generic
  writer hard-codes `weighted_quantity="win_rate"` for every HGB table
  (`src/farkle/analysis/run_hgb.py:141-176`). Thus the produced across-k feature
  importance table, whose schema contains `association_importance_mean`, has
  `weighted_quantity: win_rate`
  (`data/results_seed_pair_32_33/results_seed_32/analysis/07_hgb/across_k/feature_importance_overall.parquet.sidecar.json:31,46,56`). The JSON
  representation correctly calls the quantity
  `heldout_permutation_association_importance`
  (`src/farkle/analysis/run_hgb.py:728-750`). Finally, malformed inputs without
  a player-count column are converted to `players = 0`, and nulls are also
  filled with zero (`src/farkle/analysis/run_hgb.py:511-521`), rather than
  failing closed. Canonical inputs prevented an observed zero-player artifact
  in the reviewed path, but the retired sentinel remains reachable in the
  current implementation.
- **Consequence:** Directory location cannot be trusted to identify artifact
  scope, and sidecar consumers are told the wrong scientific quantity. The
  active `long` filename plus zero-player fallback also preserves terminology
  and semantics the governing design explicitly retired, increasing the chance
  that future consumers treat a shape label or sentinel as a statistical
  scope.
- **Smallest reasonable remediation:** Write the concatenated importance table
  through `cfg.concat_ks_dir("hgb")`, rename it to a scope-neutral descriptive
  name, pass the actual weighted quantity per artifact, and raise on absent or
  null player-count coordinates. Add a test asserting that physical scope
  directories, sidecar scope, and table coordinates agree.

### A6 — Root and root-pair stage definitions collide in one unscoped lookup

- **Severity:** Low
- **Confidence:** High
- **Classification:** Plausible divergence risk
- **Evidence:** Root/single-root-tail and root-pair registries repeat keys such
  as `trueskill`, `candidate_freeze`, every H2H phase, agreement, and reporting
  (`src/farkle/analysis/stage_registry.py:187-225`). `_DEFINITION_LOOKUP` is a
  dict comprehension over root definitions followed by pair definitions, so
  pair entries overwrite root entries (`src/farkle/analysis/stage_registry.py:228-230`).
  `resolve_stage_definition` has no workflow-scope argument
  (`src/farkle/analysis/stage_registry.py:283-289`), yet stage config hashes use
  that lookup (`src/farkle/config.py:1822-1837`). A read-only probe showed that
  `resolve_stage_definition("trueskill")` returns group `root_pair` with cache
  scope `('sim.seed_list', 'trueskill')`; changing `io.results_dir_prefix` did
  not change the root TrueSkill stage hash, even though the root definition at
  `stage_registry.py:163-173` includes `io`.
- **Consequence:** The source presents two cache contracts, but only the pair
  contract is reachable by key. Current input fingerprints and separate paths
  mitigate observed staleness, but future changes to one definition can alter
  both workflows unexpectedly or leave the intended root policy dead.
- **Smallest reasonable remediation:** Key definitions by `(workflow_scope,
  stage_key)`, or separate placement from a single shared definition when the
  cache contract truly is common. Require callers to state root,
  single-root-tail, or root-pair scope and test distinct hashes.

### A7 — The maintained CLI document still exposes retired terminology and removed topology

- **Severity:** Low
- **Confidence:** High
- **Classification:** Documentation defect
- **Evidence:** `cli_args.md` describes combine and metrics as “pooled” and
  lists a removed `analyze variance` command (`cli_args.md:78-98`), while a unit
  test explicitly confirms that variance is rejected
  (`tests/unit/cli/test_main_cli.py:217-223`). It also calls `analytics` a
  downstream-only tail, although `analysis.run_all` calls
  `run_single_root_analysis`, which runs the entire root plan before H2H
  (`src/farkle/analysis/__init__.py:335-380,402-420`). The terminology
  checker passed because its search roots omit repository-root Markdown files
  such as `cli_args.md` (`scripts/check_terminology.py:9-10,25-31`).
- **Consequence:** A maintainer cannot reconstruct the current CLI and scope
  semantics from the nominal CLI reference, and the release terminology gate
  gives false reassurance about that document.
- **Smallest reasonable remediation:** Regenerate `cli_args.md` from the actual
  parser/help or update it manually, remove retired commands and aggregation
  language, and include root-level maintained Markdown in terminology checks.

## Overall assessment

The statistical evidence layers are substantially better separated than the
legacy terminology suggests: row concatenation is physically distinct from
across-k estimation, screening is descriptive, HGB is predictive association,
two-root outputs are fixed-design reproducibility diagnostics, and H2H retains
unresolved comparisons and constrained claim language. The actual fast-run
H2H tail is pair-owned and did not reappear under either root.

The architecture is nevertheless not self-consistent at important public and
provenance boundaries. The installed CLI can silently run the wrong roots, the
pair active config is invalid and first-root-biased, configured worker owners
are bypassed, and HGB does not participate in the canonical lifecycle or scope
contract. These are not formula objections, and the completed fast-run report's
conservative claims remain useful evidence, but maintainers currently need
source-history knowledge and adversarial probes to discover which entry point,
path property, and cache mechanism actually governs a phase.

**Verdict: questionable within this review's scope.**
