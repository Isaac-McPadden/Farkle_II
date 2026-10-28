from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import yaml

from farkle.config import AppConfig, apply_dot_overrides, effective_config_dict, load_app_config
from farkle.protocols import artifact_versions, implementation_contract

RETIRED_VALUES = {
    "analysis.results_glob": "*_players",
    "rng.scheme_version": 2,
    "rng.bit_generator": "PCG64DXSM",
    **{f"artifact_contract.{key}": value for key, value in artifact_versions().items()},
    "screening.interval_confidence": 0.95,
    "head2head.seat1_advantage_scenarios": [0, 0.03, 0.06],
    "head2head.candidate_cap_policy": "balanced-tail",
    "robustness.report_pareto": True,
    "robustness.report_maximin": True,
    "orchestration.parallel_seeds": False,
}


@pytest.mark.parametrize("key,value", RETIRED_VALUES.items())
@pytest.mark.parametrize("dotted", [False, True])
def test_retired_keys_reject_even_formerly_accepted_values(tmp_path, key, value, dotted):
    section, field = key.split(".", 1)
    payload = {key: value} if dotted else {section: {field: value}}
    path = tmp_path / "retired.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=f"Retired config key '{key}'") as error:
        load_app_config(path)
    assert "remove" in str(error.value)
    with pytest.raises(ValueError, match=f"Retired config key '{key}'"):
        apply_dot_overrides(AppConfig(), [f"{key}={json.dumps(value)}"])


@pytest.mark.parametrize("section", ["rng", "artifact_contract", "orchestration"])
def test_empty_retired_sections_and_unknown_overrides_explain_migration(tmp_path, section):
    path = tmp_path / "retired.yaml"
    path.write_text(yaml.safe_dump({section: {}}), encoding="utf-8")
    with pytest.raises(ValueError, match=f"Retired config section '{section}'"):
        load_app_config(path)
    with pytest.raises(ValueError, match=f"Retired config section '{section}'"):
        apply_dot_overrides(AppConfig(), [f"{section}.unknown=1"])


def test_protocol_metadata_remains_complete_and_is_not_public_configuration():
    cfg = AppConfig()
    public = effective_config_dict(cfg)
    assert (
        not {"rng", "artifact_contract", "orchestration", "implementation_contract"} & public.keys()
    )
    for key in RETIRED_VALUES:
        section, field = key.split(".", 1)
        assert section not in public or field not in public[section]
    contract = implementation_contract()
    assert contract["rng"] == {"scheme_version": 2, "bit_generator": "PCG64DXSM"}
    assert contract["artifact_contract"] == artifact_versions()
    assert cfg.freshness_key()["rng_scheme_version"] == 2
    assert cfg.freshness_key()["rng_bit_generator"] == "PCG64DXSM"
    assert cfg.freshness_key()["configuration_schema_version"] == 2
    assert all(cfg.freshness_key()[key] == value for key, value in artifact_versions().items())
    cast(dict[str, object], contract["rng"])["scheme_version"] = 99
    assert implementation_contract()["rng"] == {"scheme_version": 2, "bit_generator": "PCG64DXSM"}


def test_current_run_context_authenticates_implementation_metadata(tmp_path: Path):
    from tests.helpers.artifact_sidecars import (
        clean_test_code_identity,
        make_authenticated_v3_config,
    )

    from farkle.orchestration.run_contexts import (
        SeedRunContext,
        load_run_context,
        write_run_context_atomic,
    )
    from farkle.utils.authenticated_contract import identity_sha256

    cfg = make_authenticated_v3_config(tmp_path)
    context = SeedRunContext.from_config(cfg)
    write_run_context_atomic(context, code_identity=clean_test_code_identity())
    payload = load_run_context(context.run_context_path)
    assert payload["run_context_contract_version"] == 4
    assert payload["implementation_contract"] == implementation_contract()
    payload["implementation_contract"]["rng"]["scheme_version"] = 99
    payload.pop("run_context_sha256")
    payload["run_context_sha256"] = identity_sha256(payload)
    context.run_context_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="implementation contract"):
        load_run_context(context.run_context_path)
