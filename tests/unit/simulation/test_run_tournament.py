"""
Fast, side-effect-free sanity checks for run_tournament.py
(all run in < 1 s).

We monkey-patch the heavy helpers so no real games are played.
"""

from __future__ import annotations

import logging
import types  # noqa: F401
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import numpy as np  # noqa: F401 | Potentially imports something that needs it
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pyarrow")

import farkle.simulation.run_tournament as rt
from farkle.cli import main as cli_main
from farkle.simulation.strategies import ThresholdStrategy

# --------------------------------------------------------------------------- #
# Mini test doubles ? replace expensive pieces with cheap determinism
# --------------------------------------------------------------------------- #


def _mini_strats(n: int = 6):
    """Return deterministic Strategy objects with distinct __str__()."""
    return [ThresholdStrategy(50 + 50 * i, i % 3, True, True) for i in range(n)]


def fake_play_shuffle(seed: int) -> Counter[str]:
        # pretend player at index (seed % len(strats)) always wins
        strats = _mini_strats(12)
        return Counter({str(strats[seed % len(strats)]): 1})


@pytest.fixture(autouse=True)
def fast_helpers(monkeypatch):
    """
    Keep tests lightning-fast **and** remember the real `_play_shuffle` so
    individual tests can restore it when they *need* the real behavior.
    """
    strats = _mini_strats(12)
    monkeypatch.setattr(
        rt, "generate_strategy_grid", lambda *a, **kw: (strats, None), raising=True  # noqa: ARG005
    )

    # save the genuine function on the module *once*
    if not hasattr(rt, "_ORIG_PLAY_SHUFFLE"):
        rt._ORIG_PLAY_SHUFFLE = rt._play_shuffle # type: ignore

    monkeypatch.setattr(rt, "_play_shuffle", fake_play_shuffle, raising=True)

