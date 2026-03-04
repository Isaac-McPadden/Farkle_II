from __future__ import annotations

from pathlib import Path

import pytest

from farkle.utils.pooling import normalize_pooling_scheme
from farkle.utils.stage_io import (
    discover_per_k_artifacts,
    resolve_worker_count,
    select_preferred_or_legacy,
)


def test_normalize_pooling_scheme_aliases_and_invalid() -> None:
    assert normalize_pooling_scheme("game-count") == "game-count"
    assert normalize_pooling_scheme("GameCount") == "game-count"
    assert normalize_pooling_scheme("equal_k") == "equal-k"
    assert normalize_pooling_scheme("custom") == "config"
    with pytest.raises(ValueError, match="Unknown pooling scheme"):
        normalize_pooling_scheme("mystery")


def test_resolve_worker_count_bounds_to_item_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    assert resolve_worker_count(16, 2, item_count=3) == 3
    assert resolve_worker_count(None, 0, item_count=5) == 5
    assert resolve_worker_count(None, None, item_count=0) == 1


def test_select_preferred_or_legacy_and_discover_order(tmp_path: Path) -> None:
    p2 = tmp_path / "2p" / "curated.parquet"
    l3 = tmp_path / "3p" / "3p_ingested_rows.parquet"
    p2.parent.mkdir(parents=True)
    l3.parent.mkdir(parents=True)
    p2.write_text("p2")
    l3.write_text("l3")

    preferred = select_preferred_or_legacy(p2, l3)
    assert preferred is not None
    assert preferred.path == p2
    assert not preferred.used_legacy

    missing_preferred = tmp_path / "9p" / "curated.parquet"
    selected = select_preferred_or_legacy(missing_preferred, l3)
    assert selected is not None
    assert selected.path == l3
    assert selected.used_legacy

    discovered = discover_per_k_artifacts(
        [3, 2, 9],
        preferred_path=lambda k: tmp_path / f"{k}p" / "curated.parquet",
        legacy_path=lambda k: tmp_path / f"{k}p" / f"{k}p_ingested_rows.parquet",
    )
    assert [(k, sel.path.name, sel.used_legacy) for k, sel in discovered] == [
        (3, "3p_ingested_rows.parquet", True),
        (2, "curated.parquet", False),
    ]
