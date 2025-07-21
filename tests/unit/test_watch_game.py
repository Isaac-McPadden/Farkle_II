import importlib
import numpy as np

import pytest

import farkle.scoring as scoring

wg = pytest.importorskip("farkle.watch_game")

def test_default_score_patch_handles_discards(monkeypatch):  # noqa: ARG001
    orig = scoring.default_score
    wg._patch_default_score()
    try:
        res = scoring.default_score([1, 1], turn_score_pre=0, return_discards=True)
        assert len(res) == 5
    finally:
        importlib.reload(scoring)
        wg.default_score = orig


def test_patched_score_used_in_turn(monkeypatch):  # noqa: ARG001
    """FarklePlayer.take_turn should call the monkey‑patched default_score."""
    orig = scoring.default_score
    wg._patch_default_score()
    calls: list[str] = []

    monkeypatch.setattr(wg.log, "info", lambda msg, *a, **k: calls.append(msg))

    class FixedGen(np.random.Generator):
        def __init__(self) -> None:
            super().__init__(np.random.PCG64())

        def integers(self, *a, size=None, **k):  # noqa: ARG002, D401
            return np.array([1, 1, 1, 2, 2, 2][: size or 1])

    player = wg.FarklePlayer(
        "P",
        wg.ThresholdStrategy(score_threshold=0, dice_threshold=6),
        rng=FixedGen(),
    )
    wg.FarkleGame([player])  # minimal instantiation
    player.take_turn(target_score=1000)

    try:
        assert any(msg.startswith("score(") for msg in calls)
    finally:
        importlib.reload(scoring)
        wg.default_score = orig
