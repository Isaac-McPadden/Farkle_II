"""Verify the preflight interpreter before and after running the declared correctness nodes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from farkle.config import load_app_config
from farkle.utils.authenticated_contract import identity_sha256
from farkle.utils.scientific_identity import producer_scope
from farkle.utils.scientific_invocation import (
    ROOT,
    difference,
    preflight_request,
    scientific_invocation,
    separate_execution,
)


def main(args: list[str]) -> int:
    envelope = Path(args[0])
    request = json.loads(envelope.read_bytes())
    if request.get("version") != 1 or identity_sha256(request) != args[1]:
        raise ValueError("Preflight launch request authentication failed")
    # Materialize only the authenticated payload in the parent's private fixture directory.
    overlay = envelope.with_suffix(".yaml")
    overlay.write_text(request["config_yaml"], encoding="utf-8")
    cfg = load_app_config(overlay, seed_list_len=request["root_count"])
    cfg._game_profile_sha256 = request["profile"]
    cfg.set_resource_limiter_mode(request["mode"])
    stage = "rng_diagnostics"
    with scientific_invocation([(cfg, stage)]), producer_scope(cfg, stage):

        def verify() -> None:
            live = preflight_request(cfg, request["nodes"])
            if live["placement"] != request["placement"]:
                raise ValueError(
                    f"Preflight interpreter or installed package placement differs at {ROOT}"
                )
            if live["expected"] != request["expected"]:
                detail = difference(stage, request["components"], live["components"])
                raise ValueError(f"Preflight scientific dependencies differ from launch: {detail}")

        verify()
        import pytest

        # Test fixtures own independent temporary configs. Keep the launch files
        # protected while the fixture producers evaluate those local designs.
        with separate_execution():
            report = envelope.with_name("tests.xml")
            result = int(
                pytest.main(
                    [
                        "-q",
                        "-c",
                        str(ROOT / "pytest.ini"),
                        f"--confcutdir={ROOT / 'tests'}",
                        *request["nodes"],
                        f"--junitxml={report}",
                    ]
                )
            )
        verify()
        if result == 0:
            envelope.with_suffix(".verified").write_text(args[1], encoding="ascii")
        return result


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
