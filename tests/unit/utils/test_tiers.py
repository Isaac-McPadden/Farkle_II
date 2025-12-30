from __future__ import annotations

import json
from pathlib import Path

from farkle.utils.tiers import (
    _extract_section,
    _normalize,
    load_tier_payload,
    tier_mapping_from_payload,
    write_tier_payload,
)


def test_normalize_best_effort_parsing() -> None:
    payload = {"A": "1", "B": 2.3, "C": "bad", "D": object()}
    assert _normalize(payload) == {"A": 1, "B": 2}


def test_extract_section_handles_missing_and_present() -> None:
    payload = {"trueskill": {"tiers": {"A": 1, "B": "2"}}}
    assert _extract_section(payload, "trueskill") == {"A": 1, "B": 2}
    assert _extract_section(payload, "frequentist") == {}


def test_tier_mapping_from_payload_prefers_sections() -> None:
    payload = {
        "frequentist": {"tiers": {"X": 1}},
        "legacy": {"tiers": {"Y": "2"}},
    }
    assert tier_mapping_from_payload(payload, prefer="frequentist") == {"X": 1}
    assert tier_mapping_from_payload(payload, prefer="missing") == {"X": 1}


def test_tier_mapping_from_payload_legacy_mapping() -> None:
    payload = {"A": 1.0, "B": "2"}
    assert tier_mapping_from_payload(payload) == {"A": 1, "B": 2}


def test_load_tier_payload_handles_missing_and_invalid(tmp_path: Path) -> None:
    path = tmp_path / "tiers.json"
    assert load_tier_payload(path) == {}
    path.write_text("{not json}")
    assert load_tier_payload(path) == {}
    path.write_text(json.dumps({"ok": True}))
    assert load_tier_payload(path) == {"ok": True}


def test_write_tier_payload_merges_sections(tmp_path: Path) -> None:
    path = tmp_path / "tiers.json"
    payload = write_tier_payload(
        path,
        trueskill={"tiers": {"A": 1}},
        active="trueskill",
        meta={"seed": 7},
    )
    assert payload["trueskill"]["tiers"] == {"A": 1}
    reloaded = json.loads(path.read_text())
    assert reloaded["meta"]["seed"] == 7

    updated = write_tier_payload(path, frequentist={"tiers": {"B": 2}}, meta={"extra": True})
    assert updated["frequentist"]["tiers"] == {"B": 2}
    final_payload = json.loads(path.read_text())
    assert final_payload["meta"] == {"seed": 7, "extra": True}
