# RNG contract

RNG scheme version 1 uses NumPy `SeedSequence` and an explicitly constructed
`PCG64DXSM` generator. Stream identity is a pure function of the root seed,
permanent purpose namespace, player count, shuffle index, pair index, seat
order, game index, seat index, and replicate index. Worker assignment,
chunking, execution order, prior draws, interruption, and resume timing are not
stream coordinates.

Purpose namespace integers are permanent:

| Integer | Purpose |
| ---: | --- |
| 1 | indexed compatibility seed |
| 10 | player dice |
| 11 | strategy generation |
| 100 | tournament shuffle |
| 101 | shuffle permutation |
| 102 | tournament game |
| 200 | H2H pair |
| 201 | H2H seat order |
| 202 | H2H game |
| 300 | TrueSkill diagnostic |
| 400 | bootstrap |
| 500 | deterministic display tie-break |
| 600 | HGB |
| 700 | root selection diagnostic |

Existing integers must never be renumbered or reused. A new experiment-wide
mapping requires incrementing `rng.scheme_version`, which intentionally makes
existing artifacts and completion state stale.
