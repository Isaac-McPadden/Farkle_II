# Contributor context

Use the references and contracts relevant to the change. Source and configuration
are authoritative for implemented behavior. Repository-wide guidance is in
[AGENTS.md](../../AGENTS.md).

## Find the relevant contract

- To locate source ownership or entry points, use the [repository map](repo_map.md).
- For stage order, artifact scopes, or recovery, use the [pipeline map](analysis_pipeline_map.md).
- For estimands, methods, or report claims, use the [statistical review map](statistical_methods_map.md).
- For focused tests, recovery checks, or release gates, use the [testing map](testing_and_review_map.md).
- For map maintenance or protected dependencies, see [orientation metadata](metadata.md).
- For CLI, configuration, or operating guidance, use the [documentation index](../index.md).

## Preserve these invariants

- Use the six canonical artifact scopes. Publish each derived artifact with
  one compatible sidecar and record completion after authenticating outputs.
  Preserve the original identity of historical artifacts.
- Safety-limit attempts have no official winner/rank. Primary tournament rates
  use attempted player exposures; completed distributions, TrueSkill and H2H
  score tests retain their distinct conditioning.
- Complete configured root/k support and explicit cross-k weights. Missing
  required inputs cannot become zero. Verified zero, undefined, insufficient
  and deliberately omitted statuses remain distinct.
- Screening, prediction, inference, dominance and display order remain separate.
  Two-root stability is descriptive fixed-design reproducibility; diagnostic
  bands/monitors make no independence guarantee.
- Preserve bounded queues, nesting suppression, aggregate resource controls,
  and authenticated execution provenance.
- H2H plans are immutable. Above-cap consent binds the exact plan and maximum
  attempts; approval/progress remain outside statistical planning.
- Process-local authenticated snapshots do not replace the final fresh byte-deep
  root/root/pair audit. Retain canonical resume units and input lineage.

For earlier decisions, task narratives, or measurements, see the
[development history](../development_history.md).
