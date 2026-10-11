# Contributor context

Farkle Mk II compares a finite strategy grid using deterministic Monte Carlo
simulation and authenticated statistical analysis. Read [AGENTS.md](../../AGENTS.md),
this orientation pack and relevant source before editing. Maps are navigation
aids, not source authority.

## Find the relevant contract

- [Repository map](repo_map.md): source ownership and entry points.
- [Pipeline map](analysis_pipeline_map.md): order, scopes and lifecycle.
- [Statistical review map](statistical_methods_map.md): methods and claim limits.
- [Testing map](testing_and_review_map.md): focused checks and release gates.
- [Metadata](metadata.md): maintenance boundary and protected dependencies.
- [Documentation index](../index.md): user and operating references.

## Preserve these invariants

- Explicit full-coordinate PCG64DXSM streams; no hidden RNG, scalar-seed
  substitution or worker/scheduling-dependent randomness.
- Atomic, idempotent publication and useful exact resume. Large inputs use
  byte-bounded streaming or partitioned durable units; check existing utils first.
- AppConfig-owned paths and six canonical scopes. Each derived artifact has
  one compatible sidecar; completion follows authenticated outputs. Never
  promote historical bytes by replacing metadata.
- Safety-limit attempts have no official winner/rank. Primary tournament rates
  use attempted player exposures; completed distributions, TrueSkill and H2H
  score tests retain their distinct conditioning.
- Complete configured root/k support and explicit cross-k weights. Missing
  required inputs cannot become zero. Verified zero, undefined, insufficient
  and deliberately omitted statuses remain distinct.
- Screening, prediction, inference, dominance and display order remain separate.
  Two-root stability is descriptive fixed-design reproducibility; diagnostic
  bands/monitors make no independence guarantee.
- Roots and player counts are sequential; parallelize independent admitted stage
  units. Preserve bounded queues, nesting suppression, aggregate resource controls
  and authenticated execution provenance.
- H2H plans are immutable. Above-cap consent binds the exact plan and maximum
  attempts; approval/progress remain outside statistical planning.
- Process-local authenticated snapshots do not replace the final fresh byte-deep
  root/root/pair audit. Retain canonical resume units and input lineage.

Before a statistical change identify estimand, conditioning, baseline,
replication unit, uncertainty, multiplicity and permitted claim. Before a
recovery change compare coordinates/logical outputs across interruption,
resume and workers.

Use the root Python 3.12 `.venv`, inspect Git status and select focused checks.
Do not force old results through freshness errors. Historical task narratives
and measurements live in [development history](../development_history.md).
