# Orientation metadata

Consolidated October 11, 2026. This pack describes implemented behavior through
October 7 first-release qualification; it is not a backlog, a scientific result
or a production-capacity certificate. Source/configuration remain authoritative.

Update relevant maps when ownership, statistical contracts, paths, RNG or
release gates change. Chronological evidence belongs in
[development history](../development_history.md).

## Protected dependencies

- RNG preflight reads `docs/implementation/task5a_recipe.json` and
  `docs/implementation/task5b_protocol_lock.json`. The lock authenticates the
  capability protocol and `configs/validation/task5b_capability.json`.
  Preserve exact paths and locked bytes.
- Guide tests read `docs/implementation/task0_question_map.md`; its 62 target
  rows are a contract.
- Capacity tests read historical seeds 54/55. The density driver compares that
  tree with seeds 56/57. Neither is a fallback for current scientific inputs.
- Final qualification evidence under `data/task10/`, preflight/capability
  evidence and research results remain protected. Documentation maintenance
  does not rewrite authenticated scientific bytes.

Inspect current version constants before compatibility changes; retained
validation JSON records historical identities. Old version announcements and
completed handoffs have been removed from active context.
