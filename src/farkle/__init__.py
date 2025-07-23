"""Farkle Mk II - fast Monte-Carlo engine & strategy tools.

Note
----
At import time :class:`pathlib.Path.unlink` is monkey patched with a helper
that safely suppresses transient ``PermissionError`` on Windows.
"""

from __future__ import annotations

import pathlib
import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v
from pathlib import Path

# Re-export the "friendly" surface
from farkle.engine import FarklePlayer, GameMetrics  # noqa: E402
from farkle.farkle_io import simulate_many_games_stream  # noqa: E402
from farkle.simulation import generate_strategy_grid, simulate_many_games_from_seeds  # noqa: E402
from farkle.stats import games_for_power  # noqa: E402
from farkle.strategies import FavorDiceOrScore, ThresholdStrategy  # noqa: E402

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

# The following identifiers are re-exported for user convenience
__all__ = [
    "FarklePlayer",
    "GameMetrics",
    "FavorDiceOrScore",
    "ThresholdStrategy",
    "generate_strategy_grid",
    "simulate_many_games_stream",
    "simulate_many_games_from_seeds",
    "games_for_power",
]


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
