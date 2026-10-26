# Statistical review map

The authoritative methods reference is
[Statistical methods](../statistical_methods.md). It covers workload/RNG,
attempted performance, completed outcomes/precision, seat/game behavior,
preflight/monitoring, TrueSkill/HGB, root stability, H2H and reporting.

Before editing identify:

1. Estimand, conditioning and observational unit.
2. Complete support, missing/zero semantics and cross-k weights.
3. Replication unit, uncertainty and dependence assumptions.
4. Multiplicity, decision rule and permitted report claim.
5. Source identity, sidecars and relevant arithmetic/recovery oracles.

Use the [testing map](testing_and_review_map.md) for check selection and
[development history](../development_history.md) for acceptance evidence.

Run purpose is declared; `orchestration/profile_metadata.py` computes versioned
production configuration qualification and planned resolution. Only the final
`orchestration/two_seed_pipeline.py` gate determines release eligibility from
the fresh audit, completed roots, code provenance and authenticated RNG coverage.
