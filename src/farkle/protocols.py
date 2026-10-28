"""Implementation-owned identities and fixed first-release method choices.

These describe implemented behavior, not user-selectable algorithms. Changes
require corresponding implementation, compatibility and qualification review.
"""

from __future__ import annotations

from typing import Final

PUBLIC_CONFIG_SCHEMA_VERSION: Final = 2
RUN_CONTEXT_CONTRACT_VERSION: Final = 4
RNG_SCHEME_VERSION: Final = 2
RNG_BIT_GENERATOR: Final = "PCG64DXSM"
ARTIFACT_CONTRACT_VERSION: Final = 3
DERIVED_SCHEMA_VERSION: Final = 2
ESTIMAND_VERSION: Final = 2
CONDITIONING_VERSION: Final = 2
BASELINE_VERSION: Final = 1
K_SUPPORT_VERSION: Final = 1
WEIGHTING_VERSION: Final = 1
MULTIPLICITY_VERSION: Final = 1
CANDIDATE_FAMILY_VERSION: Final = 1
SCREENING_INTERVAL_CONFIDENCE: Final = 0.95
H2H_SEAT1_ADVANTAGE_SCENARIOS: Final = (0.0, 0.03, 0.06)
CANDIDATE_CAP_POLICY: Final = "balanced-tail"


def artifact_versions() -> dict[str, int]:
    """Return a fresh metadata mapping of the implemented artifact identities."""

    return {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "schema_version": DERIVED_SCHEMA_VERSION,
        "estimand_version": ESTIMAND_VERSION,
        "conditioning_version": CONDITIONING_VERSION,
        "baseline_version": BASELINE_VERSION,
        "k_support_version": K_SUPPORT_VERSION,
        "weighting_version": WEIGHTING_VERSION,
        "multiplicity_version": MULTIPLICITY_VERSION,
        "candidate_family_version": CANDIDATE_FAMILY_VERSION,
    }


def implementation_contract() -> dict[str, object]:
    """Return JSON-compatible metadata without exposing mutable shared state."""

    return {
        "configuration_schema_version": PUBLIC_CONFIG_SCHEMA_VERSION,
        "rng": {"scheme_version": RNG_SCHEME_VERSION, "bit_generator": RNG_BIT_GENERATOR},
        "artifact_contract": artifact_versions(),
        "methods": {
            "interval_confidence": SCREENING_INTERVAL_CONFIDENCE,
            "seat1_advantage_scenarios": list(H2H_SEAT1_ADVANTAGE_SCENARIOS),
            "candidate_cap_policy": CANDIDATE_CAP_POLICY,
            "required_robustness_reports": ["pareto", "maximin"],
            "root_execution": "sequential",
        },
    }
