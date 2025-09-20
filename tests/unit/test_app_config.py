from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pytest
import yaml

from farkle.config import (
    AppConfig,
    _annotation_contains,
    _coerce,
    apply_dot_overrides,
    load_app_config,
)


BASE_CFG = Path("configs/base.yaml")


def test_load_app_config_overlay(tmp_path: Path) -> None:
    overlay = tmp_path / "local.yaml"
    overlay.write_text(
        yaml.safe_dump(
            {
                "sim": {
                    "n_players": 3,
                    "collect_metrics": True,
                    "row_dir": str(tmp_path / "rows"),
                },
                "analysis": {"run_trueskill": False},
                "io": {"results_dir": str(tmp_path / "out")},
            }
        )
    )
    cfg = load_app_config(BASE_CFG, overlay)
    assert cfg.sim.n_players == 3
    assert cfg.sim.collect_metrics is True
    assert cfg.sim.row_dir == tmp_path / "rows"
    assert cfg.analysis.run_trueskill is False
    assert cfg.io.results_dir == tmp_path / "out"
    # Deep merge preserves unspecified keys
    assert cfg.sim.num_shuffles == 100


def test_apply_dot_overrides(tmp_path: Path) -> None:
    cfg = load_app_config(BASE_CFG)
    pairs = [
        "sim.n_players=7",
        "analysis.run_trueskill=false",
        f"io.results_dir={tmp_path / 'results'}",
        "analysis.trueskill_beta=3.5",
        "analysis.n_jobs=4",
        "analysis.log_level=DEBUG",
        "sim.collect_metrics=true",
        f"sim.row_dir={tmp_path / 'rows'}",
    ]
    apply_dot_overrides(cfg, pairs)
    assert cfg.sim.n_players == 7
    assert cfg.sim.collect_metrics is True
    assert cfg.sim.row_dir == tmp_path / "rows"
    assert cfg.analysis.run_trueskill is False
    assert cfg.io.results_dir == tmp_path / "results"
    assert cfg.analysis.trueskill_beta == 3.5
    assert cfg.analysis.n_jobs == 4
    assert cfg.analysis.log_level == "DEBUG"


@pytest.mark.parametrize(
    ("annotation", "target", "expected"),
    [
        (Optional[Union[int, Path]], Path, True),
        (Union[str, Optional[Union[int, Path]]], Path, True),
        (Optional[Union[int, str]], Path, False),
        (Union[Optional[int], Union[str, Optional[Path]]], Path, True),
    ],
)
def test_annotation_contains_nested(annotation: object, target: type, expected: bool) -> None:
    assert _annotation_contains(annotation, target) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("false", False),
        ("YES", True),
        ("0", False),
    ],
)
def test_coerce_boolean_like_strings(raw: str, expected: bool) -> None:
    assert _coerce(raw, current=True) is expected


@pytest.mark.parametrize("raw", ["1", "off"])
def test_coerce_boolean_with_annotation(raw: str) -> None:
    expected = raw.lower() in {"1", "true", "yes", "on"}
    assert _coerce(raw, current="", annotation=Optional[bool]) is expected


def test_coerce_boolean_invalid() -> None:
    with pytest.raises(ValueError):
        _coerce("maybe", current=False)


def test_coerce_int_from_current() -> None:
    assert _coerce("42", current=1) == 42


def test_coerce_int_from_annotation() -> None:
    result = _coerce("7", current=None, annotation=Optional[int])
    assert isinstance(result, int)
    assert result == 7


def test_coerce_float_from_current() -> None:
    assert _coerce("3.14", current=0.0) == pytest.approx(3.14)


def test_coerce_float_from_annotation() -> None:
    result = _coerce("2.5", current=None, annotation=Optional[float])
    assert isinstance(result, float)
    assert result == pytest.approx(2.5)


def test_coerce_path_from_current() -> None:
    result = _coerce("foo/bar", current=Path("baz"))
    assert isinstance(result, Path)
    assert result == Path("foo/bar")


def test_coerce_path_from_annotation() -> None:
    result = _coerce("nested/data", current=None, annotation=Optional[Path])
    assert isinstance(result, Path)
    assert result == Path("nested/data")


def test_coerce_passthrough_string() -> None:
    assert _coerce("value", current="existing") == "value"


def test_apply_dot_overrides_missing_equals() -> None:
    cfg = load_app_config(BASE_CFG)
    with pytest.raises(ValueError, match="Invalid override"):
        apply_dot_overrides(cfg, ["sim.n_players"])


def test_apply_dot_overrides_missing_section_separator() -> None:
    cfg = load_app_config(BASE_CFG)
    with pytest.raises(ValueError, match="Invalid override"):
        apply_dot_overrides(cfg, ["n_players=3"])


def test_apply_dot_overrides_unknown_option() -> None:
    cfg = load_app_config(BASE_CFG)
    with pytest.raises(AttributeError, match="Unknown option"):
        apply_dot_overrides(cfg, ["sim.unknown=1"])


def test_apply_dot_overrides_relative_path_override() -> None:
    cfg = load_app_config(BASE_CFG)
    apply_dot_overrides(cfg, ["sim.row_dir=rows/output"])
    assert isinstance(cfg.sim.row_dir, Path)
    assert cfg.sim.row_dir == Path("rows/output")
