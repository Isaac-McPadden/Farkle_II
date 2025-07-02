"""
Farkle Mk II - fast Monte-Carlo engine & strategy tools
"""
import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _v
from pathlib import Path

# Re-export the "friendly" surface
from farkle.engine import FarklePlayer, GameMetrics
from farkle.farkle_io import simulate_many_games_stream
from farkle.simulation import generate_strategy_grid
from farkle.stats import games_for_power
from farkle.strategies import ThresholdStrategy

__all__ = [
    "FarklePlayer",
    "GameMetrics",
    "ThresholdStrategy",
    "generate_strategy_grid",
    "simulate_many_games_stream",
    "games_for_power"
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
    toml_path = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    with toml_path.open("rb") as fh:
        data = tomllib.load(fh)
    return data["project"]["version"]

try:
    assert __package__ is not None, "__package__ not detected, loading version from pyproject.toml"
    __version__ = _v(__package__)      # importlib.metadata
except PackageNotFoundError:
    __version__ = _read_version_from_toml()