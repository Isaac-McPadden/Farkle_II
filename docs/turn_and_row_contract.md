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
Production screening plans use 100 equal contiguous batches, so coordinate
identity is independent of process-executor chunking and worker assignment.
