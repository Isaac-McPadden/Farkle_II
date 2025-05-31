"""
Farkle Mk II - fast Monte-Carlo engine & strategy tools
"""
from importlib.metadata import version as _v

# Re-export the "friendly" surface
from .engine import FarklePlayer, GameMetrics
from .farkle_io import simulate_many_games_stream
from .simulation import generate_strategy_grid
from .strategies import ThresholdStrategy

__all__ = [
    "FarklePlayer",
    "GameMetrics",
    "ThresholdStrategy",
    "generate_strategy_grid",
    "simulate_many_games_stream",
]

# Narrow __package__ to str for Pylance
assert __package__ is not None, "__package__ must be defined before calling version"
# Package version (reads from pyproject.toml)
__version__ = _v(__package__)