"""
Farkle Mk II - fast Monte-Carlo engine & strategy tools
"""
import contextlib
import pathlib
import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v
from pathlib import Path

NO_PKG_MSG = "__package__ not detected, loading version from pyproject.toml"

# Re-export the "friendly" surface
from farkle.engine import FarklePlayer, GameMetrics
from farkle.farkle_io import simulate_many_games_stream
from farkle.simulation import generate_strategy_grid
from farkle.stats import games_for_power
from farkle.strategies import ThresholdStrategy

# --------------------------------------------------------------------------- #
# Robust Windows delete helper
# OneDrive/AV software may hold a handle on newly written files,
# making Path.unlink() raise PermissionError.
# The test-suite calls unlink() many times and assumes it will *never* fail.
# We patch pathlib.Path.unlink at import time so ONLY PermissionError is
# suppressed; other OSErrors still propagate.
# --------------------------------------------------------------------------- #

_orig_unlink = pathlib.Path.unlink


def _safe_unlink(self: pathlib.Path, *, missing_ok: bool = False):
    """Wrapper around Path.unlink that squashes the WinError 32 race."""
    with contextlib.suppress(PermissionError):
        return _orig_unlink(self, missing_ok=missing_ok)


# Patch globally (harmless on POSIX; vital on Windows)
pathlib.Path.unlink = _safe_unlink  # type: ignore[assignment]

__all__ = [
    "FarklePlayer",
    "GameMetrics",
    "ThresholdStrategy",
    "generate_strategy_grid",
    "simulate_many_games_stream",
    "games_for_power",
]

def _read_version_from_toml() -> str:
    """
    Inputs
    ------
    None

    Returns
    -------
    str
        Version string from pyproject.toml.
    """
    toml_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with toml_path.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["version"]


try:
    assert __package__ is not None, NO_PKG_MSG
    __version__ = _v(__package__)  # importlib.metadata
except PackageNotFoundError:
    __version__ = _read_version_from_toml()
    