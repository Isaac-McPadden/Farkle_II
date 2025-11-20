# Failure Notes

Audit date: 2025-11-19T06:47:10Z

Pytest command: `pytest -q` (full suite)
Log: `tmp/test_audit.log`

## Broken imports / fixtures
| Test(s) | Symptom | Owner |
| --- | --- | --- |
| `tests/unit/cli/test_main_cli.py`, `tests/unit/cli/test_main_module.py`, `tests/unit/simulation/test_run_tournament.py`, `tests/unit/utils/test_utilities.py` | Skipped: `ModuleNotFoundError: No module named 'pydantic'` | CLI / Simulation Platform |
| `tests/unit/game/test_scoring.py` | Skipped: `ModuleNotFoundError: No module named 'hypothesis'` | Game Logic |

## API signature mismatches / missing call points
| Test(s) | Symptom | Owner |
| --- | --- | --- |
| `tests/integration/test_farkle_integration.py::test_cli_smoke`, `tests/integration/test_run_tournament_integration.py::test_run_tournament_cli` | `farkle.cli.main` missing `run_tournament_cli` entrypoint | CLI |
| `tests/integration/test_metrics_stage.py::test_metrics_run_short_circuits_when_outputs_current`, `tests/unit/analysis/test_metrics.py::*`, `tests/unit/analysis/test_run_bonferroni_head2head.py::*` | `farkle.analysis.metrics` lacks helpers such as `_update_batch_counters`, `_write_parquet`, `_run_batches`, etc. | Analysis / Metrics |
| `tests/unit/simulation/test_stats.py::*` (multiple) | `games_for_power()` no longer accepts `delta`, `base_p`, `alpha`, etc., breaking validation tests | Simulation Statistics |

## Logic regressions / missing data
| Test(s) | Symptom | Owner |
| --- | --- | --- |
| `tests/integration/test_metrics_stage.py::test_metrics_run_creates_outputs_and_stamp`, `tests/unit/analysis_light/test_pipeline_stabilizers.py::test_metrics_golden_dataset` | RuntimeError: "metrics: no isolated metric files generated" | Metrics Pipeline |
| `tests/unit/analysis/test_analytics_run_all.py::*` | Either `FileNotFoundError` for pooled ratings parquet, or logging `extra` uses reserved `module` key causing `KeyError` | Analysis Orchestration |
| `tests/unit/game/test_engine.py::test_smart_discard_counters_non_negative` | Smart discard counter underflows | Game Engine |
| `tests/unit/game/test_numba_modifications.py::test_decide_final_round_ignores_other_flags` | `decide_final_round` returns `True` unexpectedly | Game Engine / Numba |
| `tests/unit/simulation/test_run_tournament_metrics.py::*` | Metric chunk counts, checkpoint cadence, and wins-only checkpoints all off from expectations | Simulation Metrics |
| `tests/unit/simulation/test_runner_wrapper.py::*` | Runner output paths and normalized counters differ from expected fixtures | Simulation Runner |
| `tests/unit/simulation/test_simulation.py::*` | Simulation sizing/grid calculations yield different job counts | Simulation Engine |
| `tests/unit/simulation/test_stats.py::test_bh_vs_bonferroni` | Benjamini-Hochberg comparison expects strict inequality but observed tie | Simulation Statistics |

## Flaky / timing issues
None observed in this run.

## Obsolete tests
None identified.

---
Future runs should be appended below with timestamps to build a history of failure surfaces and ownership.
