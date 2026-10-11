# Testing and review map

Use the root `.venv` and checks appropriate to the change. Regression success
alone does not establish a scientific claim. Historical acceptance counts and
timings are in [development history](../development_history.md).

## Focused targets

| Concern | Tests under `tests/` |
| --- | --- |
| Workload/RNG/resume | `unit/simulation/test_workload_planner.py`, `test_run_tournament*.py`; `unit/utils/test_random_utils.py`, `unit/test_rng_contract.py` |
| Topology/publication | `unit/utils/test_parallel_files.py`, `unit/orchestration/test_seed_workflows.py`, `unit/simulation/test_simulation_publication_v3.py` |
| Rows/turns/scoring | `unit/game/test_engine*.py`, `test_scoring*.py`; `unit/analysis/test_all_player_metrics.py` |
| Authentication/recovery | `unit/utils/test_artifact_contract.py`, `test_partitioned_stage.py`, `test_authenticated_graph.py`; `unit/analysis/test_release_audit.py` |
| Outcomes/precision | `unit/analysis/test_outcome_distributions.py`, `test_outcome_publication.py`, `test_outcome_precision.py` |
| RNG preflight/monitoring | `unit/analysis/test_batch_monitor.py`, `test_rng_monitor.py`; the raw integration oracle exercises full preflight |
| Performance/behavior/models | `unit/analysis/test_performance.py`, `test_seat_analysis.py`, `test_seat_behavior.py`, `test_parameter_responses.py`, `test_run_trueskill*.py`, `test_hgb_feat.py` |
| H2H decisions/consent | `unit/analysis/test_h2h_schedule.py`, `test_h2h_inference.py`, `test_h2h_conclusions.py`, `test_h2h_authorization.py`, `test_dominance.py` |
| Reporting/guide | `unit/analysis/test_structure_reporting.py`, `test_analysis_guide.py` |
| Maintenance/benchmarks | `unit/scripts/`, `unit/test_terminology.py` |

Entries can denote patterns within the named directory. Confirm current names
with `rg --files tests`. Synthetic rendering/benchmark fixtures test arithmetic
and contracts, not independent producer evidence. Capacity tests require the
ignored historical seeds-54/55 dataset; retain it without regenerating results.

## Integration qualification

- `tests/integration/test_structure_toy_oracle.py`: interrupted/resumed two-root
  structure, changed workers, H2H decisions, sidecars and report claims.
- `tests/integration/test_simulation_to_report_oracle.py`: raw workflow,
  independent numerical checks, preflight, fresh audit and stable replay.
- `tests/integration/test_windows_launcher_qualification.py`: public Windows
  launcher, strict Job Object, cross-volume scratch, cap stop/approval and recovery.

These execute bounded simulations. Run them for implementation/release changes,
not merely documentation cleanup.

## Static and release checks

```powershell
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m black --check .
.\.venv\Scripts\python -m mypy src
.\.venv\Scripts\python -m pyright
.\.venv\Scripts\python scripts/check_rng_contract.py
.\.venv\Scripts\python scripts/check_terminology.py
.\.venv\Scripts\python scripts/check_structure_release.py --artifact-root <fresh-owned-analysis-root>
```

The structural check needs an explicit fresh root; repeat `--artifact-root`
for independent roots. Old artifacts do not acquire current code identity
through documentation edits. Historical notebooks are excluded from Ruff/Black.

The reusable documentation auditor is `scripts/doc_audit.py`:

```powershell
.\.venv\Scripts\python scripts/doc_audit.py --scope src --top-files 10
```

Keep routine reports on stdout or at an explicit temporary output path.

## Review questions

Check estimand/conditioning, direct arithmetic, complete support and weights,
replication/dependence assumptions, multiplicity, sidecar compatibility and
permitted claims. Recovery changes must preserve coordinates and logical
content across workers/interruptions; require deterministic byte identity where
the format supports it.
