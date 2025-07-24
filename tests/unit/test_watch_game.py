import importlib
import logging
import sys
import types
from contextlib import contextmanager
from types import MethodType

import numpy as np
import pytest

import farkle.engine as engine
import farkle.scoring as scoring

# ---------------------------------------------------------------------------
# Some modules depend on ``scipy`` at import time.  The real package is heavy
# and pulls in compiled extensions which cause issues when loaded under the
# coverage tracer.  Tests only need the ``stats.norm.ppf`` function, so provide
# a very small stub before importing the rest of the project.  ``setdefault``
# ensures that if a real SciPy installation is already present we leave it
# untouched.
stats_stub = types.ModuleType("scipy.stats")


class _Norm:
    @staticmethod
    def ppf(x):
        return x


stats_stub.norm = _Norm()  # type: ignore[attr-defined]
scipy_stub = types.ModuleType("scipy")
scipy_stub.stats = stats_stub  # type: ignore[attr-defined]
sys.modules.setdefault("scipy", scipy_stub)
sys.modules.setdefault("scipy.stats", stats_stub)



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


def test_patched_score_used_in_turn(monkeypatch):  # noqa: ARG001
    """FarklePlayer.take_turn should call the monkey‑patched default_score."""
    orig = scoring.default_score
    wg._patch_default_score()
    calls: list[str] = []

    monkeypatch.setattr(wg.log, "info", lambda msg, *a, **k: calls.append(msg))  # noqa: ARG005

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


def test_strategy_yaml_and_type_error():
    """``strategy_yaml`` should format dataclass fields and validate type."""
    strat = wg.ThresholdStrategy(
        score_threshold=600,
        dice_threshold=3,
        smart_five=True,
        smart_one=False,
        consider_score=True,
        consider_dice=False,
        require_both=False,
        auto_hot_dice=False,
        run_up_score=True,
        favor_dice_or_score=True,
    )

    yaml = wg.strategy_yaml(strat)
    lines = yaml.splitlines()

    assert lines[0].startswith("score_threshold")
    assert lines[1].startswith("dice_threshold")
    assert "smart_five" in lines[2] and "true" in lines[2]
    assert "smart_one" in lines[3] and "false" in lines[3]

    with pytest.raises(TypeError):
        wg.strategy_yaml("bad")  # type: ignore[arg-type]


def test_trace_decide_logs_and_returns(caplog):
    """Monkey patched ``decide`` should log calls and preserve return value."""
    strat = wg.ThresholdStrategy(score_threshold=0, dice_threshold=6)

    def dummy(self, *, turn_score, dice_left):  # noqa: D401 ARG001
        return turn_score < 5

    strat.decide = MethodType(dummy, strat)
    wg._trace_decide(strat, "DBG")

    caplog.set_level(logging.INFO, logger="watch")
    assert strat.decide(turn_score=1, dice_left=6) is True
    assert strat.decide(turn_score=10, dice_left=2) is False

    messages = [rec.message for rec in caplog.records]
    assert any("DBG decide():" in m and "ROLL" in m for m in messages)
    assert any("dice_left=2" in m and "BANK" in m for m in messages)


def test_patch_scoring_logs_and_restores(caplog):
    """The ``patch_scoring`` context manager should log and clean up."""
    roll = [1, 5]

    orig = scoring.default_score
    caplog.set_level(logging.INFO, logger="watch")
    with wg.patch_scoring():
        res = scoring.default_score(roll, turn_score_pre=0)
        assert res[0] > 0
        assert any("score([1, 5]" in r.message for r in caplog.records)
    # after context manager, original function restored
    assert scoring.default_score is orig


def test_traceplayer_roll_logs(caplog):
    """``TracePlayer`` should log every roll produced."""

    class FixedRng(np.random.Generator):
        def __init__(self):
            super().__init__(np.random.PCG64())

        def integers(self, low, high=None, size=None, **kwargs):  # noqa: ARG002
            return np.array([3, 3, 3][: size or 1])

    p = wg.TracePlayer(
        "X",
        wg.ThresholdStrategy(score_threshold=0, dice_threshold=6),
        rng=FixedRng(),
    )

    caplog.set_level(logging.INFO, logger="watch")
    faces = p._roll(3)
    assert faces == [3, 3, 3]
    assert any("X rolls [3, 3, 3]" in rec.message for rec in caplog.records)


def test_watch_game_runs_with_dummies(monkeypatch, caplog):
    """Exercise ``watch_game`` end-to-end with lightweight patches."""

    metrics = types.SimpleNamespace(winner="P1", winning_score=42, n_rounds=2)

    class DummyGame:
        def __init__(self, players, target_score=0):
            self.players = players
            self.target_score = target_score

        def play(self):
            return metrics

    def dummy_strategy(_rng):
        return wg.ThresholdStrategy(score_threshold=0, dice_threshold=6)

    @contextmanager
    def dummy_patch_scoring():
        yield

    monkeypatch.setattr(wg, "FarkleGame", DummyGame)
    monkeypatch.setattr(wg, "random_threshold_strategy", dummy_strategy)
    monkeypatch.setattr(wg, "patch_scoring", dummy_patch_scoring)
    monkeypatch.setattr(wg, "TracePlayer", wg.FarklePlayer)
    monkeypatch.setattr(wg, "_trace_decide", lambda *a, **k: None)  # noqa: ARG005

    caplog.set_level(logging.INFO, logger="watch")
    wg.watch_game(seed=1)

    msgs = "\n".join(rec.message for rec in caplog.records)
    assert "===== final result =====" in msgs
    assert "Winner: P1" in msgs
