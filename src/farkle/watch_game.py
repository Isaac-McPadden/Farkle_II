# src/farkle/watch_game.py
"""
watch_game.py - run a *single* Farkle game with very chatty logging.

It
 • prints every dice throw,
 • prints every Smart-discard scoring call,
 • prints every keep-rolling / bank decision, and
 • finishes with a tiny summary.

No game-logic is duplicated - we only *wrap* the real engine.
"""

from __future__ import annotations

import contextlib
import logging
import random
from dataclasses import asdict
from types import MethodType
from typing import Sequence

import numpy as np

from farkle.engine import FarkleGame, FarklePlayer
from farkle.scoring import default_score
from farkle.strategies import (
    ThresholdStrategy,
    random_threshold_strategy,
)

# ── 1.  Plain-text logger ----------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])
log = logging.getLogger("watch")


# ── 2.  Tiny helpers ---------------------------------------------------------
# ── helper: dump a ThresholdStrategy in YAML-ish format ─────────────────────
def strategy_yaml(strategy: ThresholdStrategy) -> str:
    """Return a human friendly description of ``strategy``.

    The function formats the dataclass fields of ``ThresholdStrategy`` in the
    order they were declared so that the resulting string can be logged or
    printed.  The output intentionally resembles YAML but no YAML library is
    used.  Example output::

        score_threshold : 600
        dice_threshold  : 3
        smart_five      : true
        smart_one       : false
        consider_score  : true
        consider_dice   : true
        require_both    : false
        auto_hot_dice   : false
        run_up_score    : false
        favor_dice_or_score    : true
    """

    # dataclass → plain dict (keeps declared order)
    if not isinstance(strategy, ThresholdStrategy):
        raise TypeError("strategy_yaml expects a ThresholdStrategy")
    d = asdict(strategy)

    # YAML-friendly booleans (lowercase)
    def format_bool(v):
        return str(v).lower() if isinstance(v, bool) else v

    lines = [f"{k:<15}: {format_bool(v)}" for k, v in d.items()]
    return "\n".join(lines)


def _trace_decide(strategy: ThresholdStrategy, label: str) -> None:
    """Log calls to ``strategy.decide``.

    The function replaces the ``decide`` method of the provided strategy with a
    wrapper that logs the input state (turn score and dice left) and the
    resulting decision using :mod:`logging`.  The replacement happens in place
    (a classic monkey patch) so further calls to ``decide`` are intercepted
    until the program exits or the method is reset.
    """
    original = strategy.decide

    def traced_decide(self: ThresholdStrategy, **kw):  # same signature  # noqa: ARG001
        keep = original(**kw)
        log.info(
            f"{label} decide(): turn={kw['turn_score']:<4}  "
            f"dice_left={kw['dice_left']}  →  "
            f"{'ROLL' if keep else 'BANK'}"
        )
        return keep

    # bind as *method* so `self` is passed correctly
    strategy.decide = MethodType(traced_decide, strategy)  # type: ignore[attr-defined, method-assign]


@contextlib.contextmanager
def patch_scoring():
    """Patch :func:`farkle.scoring.default_score` to emit log messages.

    The patched function behaves identically to the original but logs the
    points scored, which dice were used and whether the player gets to reroll.
    Both the local ``default_score`` symbol in this module and the one in
    :mod:`farkle.scoring` are replaced.  This is a global side effect and should
    be undone after tests if necessary.
    """
    global default_score  # module alias
    import farkle.engine as _engine_mod
    import farkle.scoring as _scoring_mod

    orig_local = default_score
    orig_mod = _scoring_mod.default_score
    orig_engine = _engine_mod.default_score

    def traced_default_score(*args, **kw):
        res = orig_mod(*args, **kw)  # type: ignore[arg-type]
        pts, used, reroll = res[:3]
        roll = args[0] if args else kw.get("dice_roll")
        log.info(f"score({roll}) -> pts={pts:<4} used={used} reroll={reroll}")
        return res

    _scoring_mod.default_score = traced_default_score  # type: ignore[assignment]
    _engine_mod.default_score = traced_default_score  # type: ignore[assignment]
    default_score = traced_default_score
    try:
        yield
    finally:
        _scoring_mod.default_score = orig_mod  # type: ignore[assignment]
        _engine_mod.default_score = orig_engine  # type: ignore[assignment]
        default_score = orig_local


def _patch_default_score() -> None:
    """Backward compatible patch for unit tests."""

    global default_score
    import farkle.engine as _engine_mod
    import farkle.scoring as _scoring_mod

    orig_local = default_score
    orig_mod = _scoring_mod.default_score
    orig_engine = _engine_mod.default_score

    def traced_default_score(*args, **kw):
        res = orig_mod(*args, **kw)  # type: ignore[arg-type]
        pts, used, reroll = res[:3]
        roll = args[0] if args else kw.get("dice_roll")
        log.info(f"score({roll}) -> pts={pts:<4} used={used} reroll={reroll}")
        return res

    _scoring_mod.default_score = traced_default_score  # type: ignore[assignment]
    _engine_mod.default_score = traced_default_score  # type: ignore[assignment]
    default_score = traced_default_score

    _patch_default_score._orig_local = orig_local  # type: ignore[attr-defined]
    _patch_default_score._orig_mod = orig_mod  # type: ignore[attr-defined]
    _patch_default_score._orig_engine = orig_engine  # type: ignore[attr-defined]


class TracePlayer(FarklePlayer):
    """Subclass of :class:`FarklePlayer` that logs every dice roll."""

    def _roll(self, n: int) -> Sequence[int]:
        """Return ``n`` dice faces and log the result."""
        faces = super()._roll(n)
        log.info(f"{self.name} rolls {faces}")
        return faces


# ── 3.  High-level entry-point ----------------------------------------------
def watch_game(seed: int | None = None) -> None:
    """Run a single game with very verbose output.

    Two :class:`ThresholdStrategy` instances are created with random
    parameters.  Their ``decide`` methods and the global scoring function are
    monkey patched so that every decision and scoring call is logged.  Each
    player is wrapped in :class:`TracePlayer` so that all dice rolls are also
    printed.

    Parameters
    ----------
    seed:
        Optional seed forwarded to ``numpy.random.default_rng`` to make the game
        deterministic.
    """
    rng = np.random.default_rng(seed)

    # --- make two random strategies -------------------------------------
    strategy1 = random_threshold_strategy(random.Random(int(rng.integers(2**32))))
    strategy2 = random_threshold_strategy(random.Random(int(rng.integers(2**32))))
    log.info("P1 strategy\n%s\n", strategy_yaml(strategy1))
    log.info("P2 strategy\n%s\n", strategy_yaml(strategy2))

    # monkey-patch the *strategy* layer so we can see decide()
    _trace_decide(strategy1, "P1")
    _trace_decide(strategy2, "P2")

    # monkey-patch the *scoring* layer so we can see Smart-discard results
    with patch_scoring():
        # --- wrap players so we can see every dice throw --------------------
        p1 = TracePlayer(
            "P1",
            strategy1,
            rng=np.random.default_rng(rng.integers(2**32)),
        )
        p2 = TracePlayer(
            "P2",
            strategy2,
            rng=np.random.default_rng(rng.integers(2**32)),
        )

        game = FarkleGame([p1, p2], target_score=10_000)
        metrics = game.play()

    log.info("\n===== final result =====")
    log.info(
        f"Winner: {metrics.winner}  "
        f"score={metrics.winning_score}  "
        f"rounds={metrics.n_rounds}"
    )


if __name__ == "__main__":
    # run:  python -m farkle.watch_game  (optionally pass a seed)
    watch_game()
