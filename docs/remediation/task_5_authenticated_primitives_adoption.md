# Task 5 authenticated primitives: adoption guide for Tasks 6 and 7

## Boundary established by Task 5

`farkle.utils.authenticated_contract` is the version-3 provenance and lifecycle
API. It is additive: existing producers still use the version-2 API in
`farkle.utils.artifact_contract`. A producer is not version-3 compliant until
it publishes and validates with the new API. Do not change the global
`ArtifactContractConfig` default to 3 while any release-path producer still
emits a version-2 sidecar.

Existing artifacts are not migration inputs. In particular, neither
`ensure_artifact_sidecar_atomic` nor a newly constructed v3 sidecar may be used
to promote existing bytes. The only permitted missing-sidecar finalization is
`finalize_missing_sidecar_atomic`, and it requires an independently published
completion output identity that already binds both the exact artifact identity
and exact canonical sidecar bytes.

## Task 6: producer and artifact-boundary adoption

Migrate one stage or one independently resumable cell at a time.

1. Define every output with `CanonicalArtifactLocation`. Resolve it through
   `location.path(cfg)` and remove any producer-owned path construction. If a
   current artifact is in the wrong scope, relocate/regenerate it; do not retain
   the old path and merely change the declared scope.
2. Declare a sorted allowlist of public, semantic config fields and build its
   `StageConfigIdentity`. Exclude worker counts, process start method, chunk
   size, checkpoint frequency, logging, temporary paths, and resolved private
   context fields. Add a focused test showing a selected field changes the
   identity and a runtime-only field does not.
3. Obtain one `CodeIdentity` before execution. Release mode uses
   `RELEASE_CLEAN`; an unavailable Git identity or any dirty state is fatal.
   Development runs require the explicit `DEVELOPMENT_DIRTY` policy and remain
   ineligible for release evidence.
4. Construct the exact `VersionIdentity` required by the remediation contract:
   artifact contract 3, RNG scheme 2, outcome schema 2, schema 2, estimand 2,
   conditioning 2, and the applicable named method versions. Equality is the
   compatibility rule.
5. Replace free-form parameter dictionaries with `MethodContract`. Populate
   every applicable field: k weights, baseline, replication unit,
   multiplicity, family hash, schedule hash, practical/equivalence margins,
   and ordinary/simultaneous alpha. A parameter that affects the result must
   also affect the stage identity.
6. For ordinary inputs, validate and capture each with
   `capture_source_artifact`; pass the exact source path and its owning config
   on every subsequent validation. For large immutable shard sets, publish one
   coordinate-sorted manifest with `publish_immutable_manifest_atomic`, capture
   its `ManifestRootIdentity`, and validate only the small manifest and sidecar
   thereafter. Never substitute a path/size/mtime inventory.
7. Build `StageIdentity` from the config, code, versions, method, source or
   manifest identities, and immutable design hashes. Upstream identities must
   appear in the same declared logical-role order used in the sidecar.
8. Publish Parquet with `publish_authenticated_parquet_atomic`. The helper
   reads the actual written Arrow schema, binds field order, names, types, and
   nullability, invalidates an old sidecar before data replacement, and
   publishes data then sidecar with bounded retry behavior.
9. After every required output validates, write one
   `AuthenticatedCompletion` containing the stage identity and the exact
   artifact and sidecar hash for each output. Publish it last with
   `write_authenticated_completion_atomic`.
10. Use `classify_authenticated_lifecycle` for skip/resume decisions. Only
    `complete_valid` skips. `complete_stale` recomputes or fails according to
    stage policy; `partial_resumable` resumes only from design-compatible
    checkpoints; `blocked_by_cap` remains an explicit substantive state.

Each migrated producer needs independent negatives for wrong physical scope,
real schema mismatch, changed source or manifest, relevant method parameters,
version mismatch, artifact mutation, sidecar mutation, and completion-output
inventory mismatch. Mixed-source producers must give each source a distinct
logical role and owning config.

## Task 7: CLI/orchestration and release-path adoption

Task 7's strict CLI parsing remains separate from the artifact estimands. After
parsing succeeds and before any output is created, orchestration should:

1. materialize and round-trip only the public configuration;
2. record CLI overrides in the separate run context, not the public stage
   configuration identity;
3. resolve the release/development `CodeIdentity` once and pass it to all stage
   identities;
4. reject release mode before creating output if Git identity is unavailable
   or dirty;
5. preserve runtime controls in execution context only; and
6. call only migrated v3 stage entry points when producing release-valid work.

Unknown CLI options must still fail before path creation. A post-subcommand seed
override changes the public configuration and therefore the identities of every
stage whose allowlist includes the seed fields. Worker-count overrides change
execution context but must not stale logical results.

## Rollout and version switch

During staged adoption, release output is disabled because a mixed v2/v3 graph
is not a complete authenticated lifecycle. Once every release-path producer,
consumer, completion check, and release audit uses v3, update the public
configuration defaults/files and locked validation to
`artifact_contract_version: 3`, `schema_version: 2`, `estimand_version: 2`, and
`conditioning_version: 2`. That switch intentionally stales all v2 artifacts.
Run the clean fast oracle only in a new output root; leave the known-bad fast
tree untouched.
