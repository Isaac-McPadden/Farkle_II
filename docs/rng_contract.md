# RNG contract

RNG scheme version 2 initializes NumPy `PCG64DXSM` directly from a
`SeedSequence` containing the scheme version, a permanent purpose namespace,
and every full-width semantic coordinate. Generated scalar fingerprints never
become RNG roots or recovery identities. Worker assignment, chunking,
execution order, prior draws, interruption, and resume timing are not stream
coordinates.

Tournament identities are:

| Integer | Purpose | Semantic coordinates |
| ---: | --- | --- |
| 100 | tournament shuffle ownership | root, k, shuffle index |
| 101 | shuffle permutation | root, k, shuffle index |
| 102 | tournament game fingerprint | root, k, shuffle index, game index |
| 103 | tournament player dice | root, k, shuffle index, game index, seat index |

H2H identities are:

| Integer | Purpose | Semantic coordinates |
| ---: | --- | --- |
| 200 | pair design | root, pair id |
| 201 | seat order | root, pair id, order |
| 202 | attempt fingerprint | root, pair id, order, attempt index |
| 203 | H2H player dice | root, pair id, order, attempt index, seat index |

Other permanent namespaces remain reserved:

| Integer | Purpose |
| ---: | --- |
| 1 | indexed external-boundary seed |
| 10 | legacy/direct player dice |
| 11 | strategy generation |
| 300 | TrueSkill diagnostic |
| 400 | bootstrap |
| 401 | root-stability bootstrap |
| 500 | deterministic display tie-break |
| 600 | HGB |
| 700 | root selection diagnostic |

`shuffle_seed` and `game_seed` are collision-tolerant diagnostic fingerprints.
Canonical ownership and replay use the persisted semantic coordinates. Existing
namespace integers must never be renumbered or reused. Changing the mapping or
coordinate encoding requires another scheme-version increment and makes prior
simulation, checkpoint, H2H, and descendant artifact identities stale.
