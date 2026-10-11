**Farkle Mk II — project status and roadmap, 8 September 2026**

Reviewed source: `7db67ccc94a143ebd16c335563513e89a84a62aa`. The worktree was clean at review start. This report compares current source and configuration with the ten supplied research-question groups and retained remediation evidence. It is a source review with focused regression testing, not a fresh production run, exhaustive statistical proof, or current-machine capacity certification. Historical measurements below remain attached to their original experiments.

**Assessment.** The project has a substantial, coherent statistical foundation and a much improved execution system. It is ready for a bounded acceptance run and targeted completion work. It has not demonstrated production capacity on the latest commit, and it does not yet deliver every requested scientific answer. The remaining work includes scientific outputs and practical operating decisions, not just performance tuning.

The strongest coverage is tournament win performance, explicit cross-player-count comparisons, two-root descriptive stability, finalist selection, and seat-adjusted H2H inference and dominance. The weakest coverage is full outcome distributions, opponent-composition sensitivity, fixed-matchup winner entropy, and seat-specific gameplay outcomes. Some requisite observations already exist in game rows; the canonical summaries and reports are incomplete.

**What the latest commit actually changes.**

`7db67ccc` adds the global CLI option `--resource-limiters {enabled,memory-only,disabled}` and optional absolute `io.temp_dir`. It also touches publication helpers, partitioned work, runtime provenance, and several analysis readers/writers, including a substantial TrueSkill edit. It deserves integrated verification, even though its intended change is operational.

| Setting | Current behavior | Practical implication |
| --- | --- | --- |
| `enabled`, the default | Job/cgroup boundary, admission and reserve checks, RSS backpressure, CPU and native-thread policy | Existing host-resource interruptions remain the default behavior. |
| `memory-only` | Retains memory controls; removes the configured CPU/native-thread restrictions | Useful for measuring whether thread restrictions hurt throughput. It does not remove memory-related stops. |
| `disabled` | Bypasses the new host-resource boundary and cooperative host ceilings | Explicit worker counts, bounded submission, nesting suppression, and byte-bounded processing remain. This cannot override an enclosing OS limit. |
| `io.temp_dir` | Moves configured staging/workspaces to scratch; publishes atomically into canonical destinations | Can reduce synchronized-storage churn. Cross-filesystem publication still copies bytes through a destination-local bridge. |

The mega config still has `io.temp_dir: null`; the CLI defaults to `enabled`. Therefore the commit by itself does not change the default production operating conditions. It does not remove the H2H authorization cap, RNG completeness cap, release-profile requirements, code-identity checks, or serial root orchestration. It does not demonstrate that scratch or disabled limiters improve end-to-end runtime.

Sources: [resource policy](os_memory_enforcement.md), [production config](../configs/farkle_mega_config.yaml), [CLI launcher](../src/farkle/cli/launcher.py), [atomic publisher](../src/farkle/utils/artifact_contract.py).

**Where the pipeline stands.**

Each root runs simulation, ingest, curate, combine, metrics, game statistics, optional RNG diagnostics, TrueSkill, HGB, and screening. Metrics owns all-player sufficient statistics, canonical performance, and seat analysis. The pair workflow combines root evidence and stability, adds TrueSkill candidates, freezes the family, plans and executes H2H, performs inference, builds dominance structures, compares methods, and reports. A final audit authenticates the complete root/root/pair graph.

Useful architectural improvements already present include coordinate-owned PCG64DXSM streams; explicit safety-limit outcomes; parent-owned upstream identity snapshots; producer-owned simulation publication; logical concatenation rather than another full combined data copy; byte-planned RNG routing and compiled reduction kernels; bootstrap range checkpoints; immutable H2H planning separated from execution state; metadata-fast routine completion; and one final fresh deep audit. These address real prior costs. Replacing them wholesale would discard useful work.

The latest retained capacity model remains [Task 7A](remediation/task7a_production_capacity.json):

| Quantity | Retained projection, not a latest-commit measurement |
| --- | ---: |
| Strategies | 5,160 |
| Tournament games | 39,013,900 per root; 78,027,800 across both |
| Player exposures | 177,504,000 per root |
| H2H envelope | 150 candidates; 11,175 pairs; 44,700 root/order blocks |
| Planned completed H2H games | 97,937,700 |
| Total elapsed time | 127.30 hours central; 90.77–200.36 plausible; 240.44 planning upper |
| Simulation / RNG diagnostics / H2H | Approximately 63.8 / 45.9 / 13.9 hours |
| Durable storage / final files | Approximately 252 GB / 231,857 files |
| Temporary storage | Approximately 25.8 GB central; 51.5 GB upper |

The actual finalist union can be smaller than 150, so this is an envelope, not the known eventual H2H workload. These forecasts are sensitive to hardware, storage, completion rates, and candidate count. They are not measured guarantees or statistical confidence intervals. The historical `not_capacity_ready` verdict means the forecast missed its scheduling target; it is not itself a runtime gate or proof that the estimators are invalid.

The [Task 7B experiment](remediation/task7b_h2h_bundling_report.md) preserved logical equivalence and improved median publication time by 66.64% for 180 blocks and 79.70% for 760 blocks. It missed its primary 70% requirement and worker-regression criterion, so its implementation was rolled back. That establishes failure against that experiment's acceptance rule. It does not establish that bundling has no practical value. The present production layout remains one Parquet and sidecar per H2H block.

**Coverage of the attached research questions.**

“Implemented” here means an identifiable canonical computation exists and its declared claim broadly fits the question. It does not mean a production result has been generated and independently validated at this revision.

| Question group | Coverage | What is available and what remains |
| --- | --- | --- |
| 1. Strategy performance | Substantially implemented | Per-k attempted-exposure win rates, `1/k` deltas, batch MCSE, bootstrap selection evidence, screening, and TrueSkill comparisons. Rankings remain descriptive; close estimates do not establish superiority. |
| 2. Across player counts | Substantially implemented | Complete-support equal-k chance deltas, player-count diagnostics, Pareto and maximin summaries. The canonical equal-k score is the mean of `p_k - 1/k`, an absolute probability-point advantage, not relative lift. Specialist/robust strategy interpretation needs a clearer user report. |
| 3. Risk and full distributions | Partial; important delivery gap | All-player score, rank, loss-margin and behavior observations feed counts, sums, and squared sums. These recover moments, not full score CDFs, finish-rank frequencies, tail risk, or the shape of losses. Game-level winner/runner-up margins are different quantities. Add per-strategy/k distributions, completed-only ranks and loser-conditioned margins with explicit support. |
| 4. Opponent composition | Partial; substantial analysis gap | Tournament rates estimate performance against the configured opponent mixture; finalist H2H answers a different direct-comparison question. Current matchup RNG diagnostics analyze rounds in repeated participant multisets. They do not estimate strategy-specific composition sensitivity or fixed-matchup winner probabilities. A support audit must determine whether existing repeated matchups suffice or a dedicated repeated-matchup experiment is needed. |
| 5. H2H superiority/dominance | Substantially implemented; execution/config caveats | Balanced root/order completed support, seat-adjusted effects, score inference, Holm decisions, simultaneous practical bounds, power information, unresolved outcomes, SCC cycles and dominance fronts. Equivalence machinery exists, but production sets `delta_equivalence: null`; the current configuration cannot answer affirmative practical-equivalence questions. |
| 6. Seat and opportunity effects | Partial | Tournament seat win rates and standardized diagnostics exist; H2H stays seat-stratified. Exact score/turn aggregates and round-proxy errors exist at strategy/k/batch level. Canonical seat counts contain wins/exposures, not score/rank/margin/turn distributions. Add those seat-level outcomes and a global opportunity-imbalance summary. Balancing allocations alone does not justify dropping seat adjustment. |
| 7. Gameplay behavior/efficiency | Partial | Exact ratio-of-sums and mean-of-game-ratios returns, round-proxy diagnostics, game lengths, and behavioral moments exist. They support descriptive efficiency comparisons. State-specific explanations of risk-taking or decision mechanisms require additional analysis and potentially event/state observations beyond terminal aggregates. |
| 8. Randomness/reproducibility | Strong core, incomplete matchup view | Independent-root comparisons, rank/top-N movement, convergence/drift diagnostics, H2H structure agreement, and lag diagnostics exist. Two roots appropriately support a fixed-design reproducibility challenge. Fixed-matchup winner entropy is absent from the canonical analysis. Lag-1 descriptive bands do not certify independence or rule out every execution-order effect. |
| 9. Explaining strategy parameters | Substantially implemented for exploration | HGB uses held-out strategy configurations, permutation association importance and future proposals. This supports predictive associations and candidate generation. Interactions/nonlinear response summaries need clearer presentation; individual fold outcomes must establish whether prediction is useful. Proposed candidates still need independent simulation. |
| 10. Controls/validity/claims | Strong framework; configuration and presentation gaps | Explicit support, conditioning, provenance, safety outcomes, claim constraints and oracles exist. Mega config has `controls: []` and `mandatory_diagnostics: []`; therefore it does not instantiate the requested known-control comparison program. Define control expectations and ensure those strategies are actually included and retained. |

Evidence: [all-player schema](../src/farkle/analysis/all_player_metrics.py), [performance](../src/farkle/analysis/performance.py), [seat schema and aggregation](../src/farkle/analysis/seat_analysis.py), [game statistics](../src/farkle/analysis/game_stats.py), [RNG diagnostics](../src/farkle/analysis/rng_diagnostics.py), [H2H inference](../src/farkle/analysis/h2h_inference.py), [HGB](../src/farkle/analysis/run_hgb.py), [reporting](../src/farkle/analysis/structure_reporting.py). A source search found no canonical outcome-entropy implementation; RNG seed entropy is unrelated.

The generated structure report concentrates on screening, H2H, robustness, agreement and claim boundaries. It is not yet an integrated answer book for these ten groups. Completion stamps cannot substitute for that delivery layer.

**Problems that remain, in practical priority order.**

1. **Production configuration can authorize less work than its own plan requires.** In `h2h_schedule.py`, authorization compares the maximum planned attempts with `total_game_cap` before execution. At the 150-candidate envelope, 97,937,700 completed targets imply 195,875,400 maximum attempts, exceeding the 100,000,000 cap. This can block even if actual games would all complete quickly. Resolve the envelope/cap explicitly before committing to a long run; a smaller actual union may fit. The cap is execution authorization, not a power criterion.

2. **RNG completeness remains a data-dependent stop.** The mega cap is 100,000 eligible matchup groups. Current eligibility logic marks excess as `blocked_by_cap`; this is not simply a warning about unexamined rows. Establish expected eligible support and storage cost, then choose sufficient capacity or explicitly redesign the diagnostic's scope. Do not describe capped diagnostics as exhaustive. Disabling the diagnostic changes evidence coverage even if tournament estimates remain computable.

3. **Repository-wide identity makes routine development expensive.** `resolve_code_identity` uses Git HEAD and whole-worktree status/diffs; stage freshness incorporates the resulting identity. An unrelated committed documentation change changes HEAD, and an untracked report can make the tree dirty for release eligibility. Dirty execution may proceed but cannot establish a clean release. This is conservative and understandable, but poor ergonomics for multi-day runs. Use a pinned clean checkout operationally now. A future stage dependency identity should bind relevant code, methods, inputs and runtime dependencies, retain full Git provenance, and require explicit migration tests. Simply dropping checks would permit stale reuse.

4. **Some memory and resume paths still contradict the broad design aspirations.** `seat_analysis.build_canonical_seat_analysis` reads the entire per-k batch-count Parquet into pandas before grouping. With 5,160 strategies, 100 batches and 12 seats, a fully populated cell can contain 6,192,000 rows, with additional grouping/copy memory. This is a source-confirmed materialization risk; its actual latest-commit peak was not measured. `_iter_seat_count_tables` also uses row-count batches and Python dictionaries rather than the shared byte-bounded reader. Further, `metrics.run` calls seat analysis with `force=True` whenever the enclosing metrics stage is stale, even though seat analysis owns its own completion. This can republish valid seat summaries after an unrelated metrics interruption. Its subordinate caches can still help, so it is not a claim that every seat input is replayed. Measure and narrow this forced invalidation; stream aggregate seat counts and retain compact batch vectors only where uncertainty requires them.

5. **Storage costs remain structural.** Per-block H2H files and sidecars, simulation shards, and durable intermediate rows cause many opens and publications. Scratch reduces growing-file synchronization but does not reduce final file count, eliminate cross-volume copy costs, or remove the final deep audit. Revisit bundling only against total elapsed time, storage, recovery work and variance on representative workloads. A fixed percentage threshold for one phase can reject a worthwhile end-to-end improvement.

6. **The authentication framework has a high maintenance cost.** Artifact contracts, authenticated contracts, release identities, stage completions, partition units, lifecycle graphs and final audits distribute related responsibilities across several large modules. Some layering is necessary; the older additive-v2 docstring in `authenticated_contract.py` illustrates how easily explanations lag current v3 behavior. Consolidate publication/validation interfaces incrementally, guided by actual repeated work and defects. Do not assume every sidecar or deep check is wasted overhead: authentication already caught real problems and routine graph reuse has been optimized.

7. **Small representation defects remain.** Task 7A documented null-typed empty pandas-derived Parquets and curate `manifest.jsonl` files that are actually single JSON objects. Current seat/dominance writers still infer Arrow schemas from pandas, and curate still writes its schema manifest as JSON. These are schema/format consistency issues, not major capacity problems. Give empty outputs explicit schemas and correct the manifest contract through a coordinated producer/consumer change. The historical count of nine empty files describes that inspected tree, not a new inventory.

8. **Scientific delivery is incomplete.** Distribution, matchup, seat-outcome and control work above needs explicit outputs and claim contracts. Adding more tournament games does not automatically answer a question whose summary or experimental design is absent.

Serial roots and player counts are not automatically bad architecture: an active tournament owns the available simulation process budget, avoiding nested oversubscription. Independent analysis cells and bootstrap ranges do use parallelism. Concurrent roots may help during underutilized phases, but there is no current evidence that another orchestration layer is the highest-value fix. The current code explicitly rejects `parallel_seeds: true`.

**Which rules should stay, and which should become policy.**

| Keep as correctness/scientific contracts | Treat as explicit operating or experiment policy |
| --- | --- |
| Coordinate-owned RNG; recorded rules/grid/opponent mixture | Worker counts, CPU/native-thread limits, host reserve and RSS thresholds |
| Authenticated durable units and source-bound completion | Scratch location, publication bundle size and checkpoint work budget |
| Exact resume without duplicate or skipped coordinates | Serial versus concurrent scheduling, subject to ownership and resource accounting |
| Complete support or explicit missingness; declared cross-k weights | 48-hour versus 108-hour scheduling windows and infrastructure choice |
| Frozen H2H family, declared conditioning, multiplicity and honest unresolved results | Engineering acceptance percentages and allowed worker regression |
| No publication of partial data as complete | Diagnostic capacity and attempt authorization, explicitly reconciled with the planned evidence |

Alpha, target power, practical delta, equivalence margin and the safety-completion policy are chosen scientific design parameters. They can be changed prospectively, with revised claims and a new plan; changing them merely to make a run faster changes the intended experiment. Resource limits generally should not change the estimand. The latest commit moves that separation in the right direction.

**Documentation: keep the evidence, reduce competing instructions.**

| Material | Assessment and recommended action |
| --- | --- |
| `README.md`, `cli_args.md`, config/artifact/RNG/turn documentation | Retain as current user/maintainer references. README's “current top-level sections” omits `profile` and `resources`; correct it. Separate conceptual workflow summaries from the exact executable stage list. |
| `docs/codex_context/*` | Useful navigation, but the 15 KB context prompt and 22 KB metadata file repeat extensive contracts and historical task decisions. Reduce the prompt to durable rules and links; keep dates/commit identities on evidence and maps. |
| `docs/remediation/PLAN Revised.md` | Historical plan rooted at `fb083fa`, with obsolete descriptions such as the old fast profile and RNG route counts. Label it historical/superseded for operating instructions. Preserve it as rationale. |
| Task 3–7 reports, benchmark plans and JSON | Valuable measurements and reproducibility records. Add one index recording commit, fixture, verdict and supersession. Task 7A remains the retained capacity estimate; Task 7B is a closed rejected experiment. |
| Earlier review reports and task inventories | Historical findings, not a current blocker checklist. Link resolved findings to fixes/evidence and carry only independently current issues into a live backlog. |
| OS-memory documentation | Mode descriptions reflect the new commit. Its “two-concurrent-root canary” wording needs reconciliation with the supported serial-root orchestrator; state precisely whether this is a special harness or revise it. |
| Source docstrings and incidental comments | Include them in the cleanup: the additive-v2 wording in `authenticated_contract.py` and HGB's “single cell in memory” comment are stale/misleading. HGB actually retains a list of compact per-k frames; those tables are small enough that this is not itself a production-memory problem. |

No historical evidence should be deleted merely for being old. The clutter comes from presenting retrospective benchmark restrictions, current code contracts, and future authorization steps together as timeless instructions. A short live status page plus an evidence index would substantially improve orientation. Avoid mass-moving files until inbound links are accounted for.

**Recommended roadmap and completion criteria.**

| Order | Work | Completion criterion |
| --- | --- | --- |
| 1 — Make the intended run explicit | Publish the live issue/evidence index; decide whether all ten question groups are required for the first scientific release. Reconcile H2H cap, diagnostic coverage, controls and optional equivalence with that scope. | One production specification separates scientific settings, machine settings, and a human scheduling budget; each requested answer maps to an output or named gap. |
| 2 — Validate the latest operational changes | Use a clean pinned source checkout and new bounded roots. Exercise default mode and the intended alternate mode/scratch setup through the real Windows launcher, including interruption/resume and final deep audit. Include same-volume and cross-volume publication cases. | Matching logical coordinates/counts/decisions, valid provenance and completion, successful resume, measured peak memory and temporary storage. Focused unit tests alone do not satisfy this step. |
| 3 — Repair boundedness and avoidable recomputation | Stream seat aggregation, review the stale-parent `force=True` path, and fix empty-schema/manifest-format inconsistencies in a narrow migration. | Production-shaped seat support fits a stated working-memory budget; an interrupted parent stage reuses valid children; empty/nonempty schema and format tests pass. |
| 4 — Fill scientific output gaps | Add score/rank/loss distributions and seat outcomes from existing rows; aggregate behavior/turn evidence into usable reports. Audit exact-matchup repetition; design dedicated repeated matchups if needed. | Hand-checkable distribution totals, explicit denominators/tie/safety rules, bounded output sizes, and each question answered or marked unidentifiable. Matchup entropy distinguishes Monte Carlo uncertainty from seat/composition variation. |
| 5 — Refresh capacity and choose optimization | Benchmark production-k mix and intended storage/resource mode; measure simulation, RNG, publication, audit and recovery. Reassess H2H bundles only if material to total time. | A dated forecast tied to the current commit and actual finalist envelope, with measured component rates and uncertainty; a deliberate decision to accept runtime, improve infrastructure, optimize, or revise scope. |
| 6 — Qualify and run the scientific release | Run required functional/static checks and fresh end-to-end/recovery oracles, then freeze code/config/environment and execute the agreed experiment. | Root/root/pair audit passes, diagnostics have declared coverage, caps permit the plan, and reports state unresolved results and claim boundaries. |
| 7 — Improve maintainability after the first usable release | Narrow stage code identity and consolidate authentication/publication APIs incrementally; archive historical instructions behind the index. | Unrelated documentation edits no longer force scientific recomputation, while relevant code/input/environment changes reliably invalidate affected outputs. |

Distribution and seat summaries may be developed against existing valid row fixtures before another large simulation. A dedicated matchup experiment or state-level behavioral instrumentation could require new data; settle that distinction before paying the production cost. Avoid attaching all proposed architectural cleanup to the first scientific release.

**Verification performed for this report.**

Read the supplied questions, repository instructions/context maps, current production config, recent commit diff summary, retained capacity/bundling evidence, and relevant implementation paths. Ran 199 focused tests in the repository `.venv`; all passed: 130 covering resource config, launcher, main CLI, atomic writer, parallel execution and run contexts; 69 covering all-player metrics, performance, seat analysis, H2H inference, dominance and root stability. These were the corresponding `tests/unit/config/test_resources_config.py`, `tests/unit/cli/test_launcher.py`, `tests/unit/cli/test_main_cli.py`, `tests/unit/utils/test_writer.py`, `tests/unit/utils/test_parallel_files.py`, `tests/unit/test_run_contexts.py`, and six named `tests/unit/analysis/test_*.py` modules. The full suite, static gates, fresh end-to-end oracle and production-scale benchmarking were not rerun. No source/configuration changes or production execution were performed. This report itself is a new untracked documentation file until committed; because of the current whole-worktree policy, its presence affects clean-release eligibility.
