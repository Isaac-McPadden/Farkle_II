# src/farkle/__init__.py
"""Farkle Mk II - fast Monte-Carlo engine & strategy tools.

Note
----
At import time :class:`pathlib.Path.unlink` is monkey patched with a helper
that safely suppresses transient ``PermissionError`` on Windows.
"""

from __future__ import annotations

import pathlib
import tomllib
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v
from pathlib import Path

# Path to the project's pyproject.toml for local version fallback
PYPROJECT_TOML = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"

# Diagnostic message for fallback version retrieval
NO_PKG_MSG = "__package__ not detected, loading version from pyproject.toml"

# --------------------------------------------------------------------------- #
# Robust Windows delete helper
# OneDrive/AV software may hold a handle on newly written files,
# making Path.unlink() raise PermissionError.
# The test-suite calls unlink() many times and assumes it will *never* fail.
# We patch pathlib.Path.unlink at import time so ONLY PermissionError is
# suppressed; other OSErrors still propagate.
# --------------------------------------------------------------------------- #

_ORIG_UNLINK = pathlib.Path.unlink


def _safe_unlink(self: pathlib.Path, *, missing_ok: bool = False):
    """Wrapper around ``Path.unlink`` that squashes the WinError 32 race.

    Deletes ``self`` while ignoring transient permission issues.

    Parameters
    ----------
    missing_ok : bool, optional
        Forwarded to :meth:`pathlib.Path.unlink`.

    Returns
    -------
    None

    Notes
    -----
    Only ``PermissionError`` is suppressed. Any other :class:`OSError` will be
    re-raised.
    """
    try:
        _ORIG_UNLINK(self, missing_ok=missing_ok)
    except PermissionError as exc:
        if getattr(exc, "winerror", None) == 32:
            return None
        raise


# Patch globally (harmless on POSIX; vital on Windows)
pathlib.Path.unlink = _safe_unlink  # type: ignore[assignment]

# --------------------------------------------------------------------------- #
# Lazily expose the "friendly" surface
# Heavy modules (numba, etc.) are imported only when accessed, dramatically
# reducing import-time dependencies for light utilities such as
# ``run_trueskill``.
# --------------------------------------------------------------------------- #

__all__ = [  # loads lazily, that's why reportUnsupportedDunderAll is triggered
    "FarklePlayer",  # pyright: ignore[reportUnsupportedDunderAll]
    "GameMetrics",  # pyright: ignore[reportUnsupportedDunderAll]
    "FavorDiceOrScore",  # pyright: ignore[reportUnsupportedDunderAll]
    "ThresholdStrategy",  # pyright: ignore[reportUnsupportedDunderAll]
    "generate_strategy_grid",  # pyright: ignore[reportUnsupportedDunderAll]
    "simulate_many_games_stream",  # pyright: ignore[reportUnsupportedDunderAll]
    "simulate_many_games_from_seeds",  # pyright: ignore[reportUnsupportedDunderAll]
    "games_for_power",  # pyright: ignore[reportUnsupportedDunderAll]
]

_LAZY_IMPORTS = {
    "FarklePlayer": "farkle.engine",
    "GameMetrics": "farkle.engine",
    "FavorDiceOrScore": "farkle.strategies",
    "ThresholdStrategy": "farkle.strategies",
    "generate_strategy_grid": "farkle.simulation",
    "simulate_many_games_stream": "farkle.utils.parallel",
    "simulate_many_games_from_seeds": "farkle.simulation",
    "games_for_power": "farkle.stats",
}


def __getattr__(name: str):  # pragma: no cover - simple dynamic loader
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


def _read_version_from_toml() -> str:
    """Return the package version declared in ``pyproject.toml``.

    The file is expected to reside at the repository root three directories
    above this module. If the ``[project]`` table or the ``version`` entry is
    missing a :class:`KeyError` will be raised by :mod:`tomllib`.
    """
    with PYPROJECT_TOML.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["version"]


try:
    assert __package__ is not None, NO_PKG_MSG
    __version__ = _v(__package__)  # importlib.metadata
except PackageNotFoundError:
    __version__ = _read_version_from_toml()
