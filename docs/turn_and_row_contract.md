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
- diagnostic `shuffle_seed` and `game_seed` fingerprints
- `rng_scheme_version` and `rng_purpose_namespace`

The deterministic batch identifier is `shuffle_index // shuffles_per_batch`.
Production screening plans use 100 equal contiguous batches. Those batches are
also immutable process-recovery blocks; measured throughput is used only for
runtime projection and cannot change their boundaries.

Checkpoints own completed zero-based shuffle indices and one-based process-block
indices even when row and metric outputs are disabled. Row manifests own one
shuffle coordinate apiece. Metric manifests own an explicit ordered list of
shuffle indices plus non-authoritative coordinate fingerprints, allowing an
interrupted row-producing block to resume from its unfinished suffix. Each
append record also carries the content hash, byte length, adjacent-sidecar hash,
and Arrow-schema fingerprint captured by the original bounded writer. Final
manifests are canonicalized by semantic coordinate, so worker completion order,
process IDs, and timestamps cannot change their bytes or root. Completion binds
the immutable manifest and standalone producer-owned outputs rather than
rehashing an expanded shard tree. Changing process-executor worker counts,
interrupting, or resuming therefore does not change coordinate identity or
regenerate authenticated work.

RNG scheme v2 constructs every tournament seat generator directly from
`(root_seed, k, shuffle_index, game_index, seat_index)` in namespace 103.
Scalar fingerprints may collide and are never used for ownership, replay, or
as roots for child generators.

Ingest and curate retain these coordinates as typed columns rather than
reconstructing them from filenames or row order. The canonical row schema also
retains `shuffle_seed`, `game_seed`, RNG contract fields, `P#_n_turns`, and
`P#_hit_max_rounds`. The combine stage aligns only missing later-seat columns
inside independently resumable by-k Parquet partitions. Its authenticated
manifest binds deterministic partition order, per-partition row counts,
schemas, source identities, output hashes, and complete repository code
identity. The logical `concat_ks` scanner visits partitions by increasing k and
preserves row order within each source. It performs concatenation only, never
statistical aggregation. Routine validation uses this evidence; explicit deep
verification rereads all logical rows and hashes every partition.

Legacy concatenations are never selected as inputs. Existing retired paths are
listed with their canonical replacement in
`combine/diagnostics/migration_report.json`.
