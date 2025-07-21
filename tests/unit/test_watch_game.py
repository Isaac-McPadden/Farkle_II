import importlib
import logging

import numpy as np
import pytest

import farkle.engine as engine
import farkle.scoring as scoring

wg = pytest.importorskip("farkle.watch_game")


def test_default_score_patch_handles_discards(monkeypatch):  # noqa: ARG001
    orig_s = scoring.default_score
    orig_e = engine.default_score
    wg._patch_default_score()
    try:
        res = scoring.default_score([1, 1], turn_score_pre=0, return_discards=True)
        assert len(res) == 5
    finally:
        importlib.reload(scoring)
        engine.default_score = orig_e
        wg.default_score = orig_s


def test_patched_score_traces_take_turn(caplog):  # noqa: D103
    orig_s = scoring.default_score
    orig_e = engine.default_score
    wg._patch_default_score()
    try:
        caplog.set_level(logging.INFO, logger="watch")

        class StubGen(np.random.Generator):
            def __init__(self):
                super().__init__(np.random.PCG64())

            def integers(self, low, high=None, size=None, **kwargs):  # noqa: ARG002
                if size is None:
                    size = 6
                return np.array([1, 1, 1, 5, 5, 5][:size])

        p = engine.FarklePlayer(
            "T",
            wg.ThresholdStrategy(score_threshold=0, dice_threshold=6),
            rng=StubGen(),
        )
        p.take_turn(target_score=10_000)
        assert any("score([" in rec.message for rec in caplog.records)
    finally:
        engine.default_score = orig_e
        importlib.reload(scoring)
        wg.default_score = orig_s
