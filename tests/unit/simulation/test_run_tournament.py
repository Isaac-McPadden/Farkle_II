"""
Fast, side-effect-free sanity checks for run_tournament.py
(all run in < 1 s).

We monkey-patch the heavy helpers so no real games are played.
"""

from __future__ import annotations

import logging
import types  # noqa: F401
from collections import Counter

import numpy as np  # noqa: F401 | Potentially imports something that needs it
import pytest

pytest.importorskip("pydantic")
pytest.importorskip("pyarrow")

import farkle.simulation.run_tournament as rt
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
        rt._ORIG_PLAY_SHUFFLE = rt._play_shuffle  # type: ignore

    monkeypatch.setattr(rt, "_play_shuffle", fake_play_shuffle, raising=True)


@pytest.mark.parametrize(
    ("n_players", "expected"),
    [
        (1, "n_players must be ≥2"),
        (7, "n_players must divide 8,160"),
    ],
)
@pytest.mark.xfail(
    reason=(
        "Validation messaging changed with reduced strategy roster; "
        "see https://github.com/Isaac-McPadden/Farkle_II/issues/202"
    ),
    strict=False,
)
def test_run_tournament_invalid_player_counts(n_players: int, expected: str) -> None:
    cfg = rt.TournamentConfig()
    cfg.num_shuffles = 1

    with pytest.raises(ValueError) as excinfo:
        rt.run_tournament(config=cfg, n_players=n_players)

    assert expected in str(excinfo.value)


@pytest.mark.xfail(
    reason=(
        "Validation messaging changed with reduced strategy roster; "
        "see https://github.com/Isaac-McPadden/Farkle_II/issues/202"
    ),
    strict=False,
)
def test_init_worker_rejects_bad_player_counts(monkeypatch) -> None:
    strats = _mini_strats(3)
    cfg = rt.TournamentConfig(n_players=7)
    monkeypatch.setattr(rt, "_STATE", None, raising=False)

    with pytest.raises(ValueError, match="n_players must divide 8,160"):
        rt._init_worker(strats, cfg)


def test_run_chunk_logs_and_propagates(monkeypatch, caplog) -> None:
    class BoomError(RuntimeError):
        pass

    def boom(_seed: int):
        raise BoomError("boom")

    monkeypatch.setattr(rt, "_play_shuffle", boom, raising=True)

    with caplog.at_level(logging.ERROR, logger=rt.LOGGER.name), pytest.raises(BoomError):
        rt._run_chunk([123])

    assert any("Shuffle failed" in rec.getMessage() for rec in caplog.records)
    logged = [rec for rec in caplog.records if "Shuffle failed" in rec.getMessage()][0]
    assert logged.stage == "simulation"
    assert logged.shuffle_seed == 123
