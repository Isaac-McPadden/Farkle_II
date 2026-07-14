# Turn accounting and game-row provenance

`FarklePlayer.n_turns` increments once on entry to every `take_turn` call. A
farkle, a zero-point entry attempt, or a roll-limit exception still counts as
one attempted turn. The counter is copied to `PlayerStats.n_turns` and flattened
as `P#_n_turns`; `n_rounds` is not a substitute because players seated before a
later final-round trigger can receive an additional closing turn.

Tournament rows carry the stable simulation coordinates used to produce them:

- `root_seed`
- `k`
- zero-based `shuffle_index`
- zero-based `game_index` within the shuffle
- zero-based `deterministic_batch_id`
- `shuffle_seed` and `game_seed`
- `rng_scheme_version` and `rng_purpose_namespace`

The deterministic batch identifier is `shuffle_index // shuffles_per_batch`.
Production screening plans use 100 equal contiguous batches. Those batches are
also immutable process-recovery blocks; measured throughput is used only for
runtime projection and cannot change their boundaries.

Checkpoints own completed zero-based shuffle indices and one-based process-block
indices even when row and metric outputs are disabled. Row manifests own one
shuffle coordinate apiece. Metric manifests own an explicit ordered list of
shuffle indices and their coordinate-derived seeds, allowing an interrupted
row-producing block to resume from its unfinished suffix. Completion markers
record the full shuffle range, batch count, batch size, root, k, and RNG scheme.
Changing process-executor worker counts, interrupting, or resuming therefore
does not change coordinate identity or regenerate checkpointed work.

Ingest and curate retain these coordinates as typed columns rather than
reconstructing them from filenames or row order. The canonical row schema also
retains `shuffle_seed`, `game_seed`, RNG contract fields, `P#_n_turns`, and
`P#_hit_max_rounds`. The combine stage aligns only missing later-seat columns;
it verifies every normalized source row against its `concat_ks` output in a
bounded streaming comparison. That check covers row order, values, coordinate
keys, and total count, and performs no aggregation.

Legacy concatenations are never selected as inputs. Existing retired paths are
listed with their canonical replacement in
`combine/diagnostics/migration_report.json`.
