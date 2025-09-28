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

from farkle.game.engine import FarkleGame, FarklePlayer
from farkle.game.scoring import default_score
from farkle.simulation.strategies import (
    ThresholdStrategy,
    random_threshold_strategy,
)
from farkle.utils.random import make_rng, spawn_seeds

# ── 1.  Plain-text logger ----------------------------------------------------
LOGGER = logging.getLogger(__name__)


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
        LOGGER.info(
            "%s decide(): turn=%d dice_left=%d -> %s",
            label,
            kw.get("turn_score", 0),
            kw.get("dice_left", 0),
            "ROLL" if keep else "BANK",
            extra={"stage": "watch"},
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
    import farkle.game.engine as _engine_mod
    import farkle.game.scoring as _scoring_mod

    orig_local = default_score
    orig_mod = _scoring_mod.default_score
    orig_engine = _engine_mod.default_score

    def traced_default_score(*args, **kw):
        res = orig_mod(*args, **kw)  # type: ignore[arg-type]
        pts, used, reroll = res[:3]
        roll = args[0] if args else kw.get("dice_roll")
        LOGGER.info(
            "score(%s) -> pts=%s used=%s reroll=%s",
            roll,
            f"{pts:<4}",
            used,
            reroll,
            extra={"stage": "watch"},
        )
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
    import farkle.game.engine as _engine_mod
    import farkle.game.scoring as _scoring_mod

    orig_local = default_score
    orig_mod = _scoring_mod.default_score
    orig_engine = _engine_mod.default_score

    def traced_default_score(*args, **kw):
        res = orig_mod(*args, **kw)  # type: ignore[arg-type]
        pts, used, reroll = res[:3]
        roll = args[0] if args else kw.get("dice_roll")
        LOGGER.info(
            "score(%s) -> pts=%s used=%s reroll=%s",
            roll,
            f"{pts:<4}",
            used,
            reroll,
            extra={"stage": "watch"},
        )
        return res

    _scoring_mod.default_score = traced_default_score  # type: ignore[assignment]
    _engine_mod.default_score = traced_default_score  # type: ignore[assignment]
    default_score = traced_default_score

    _patch_default_score._orig_local = orig_local  # type: ignore[attr-defined]
    _patch_default_score._orig_mod = orig_mod  # type: ignore[attr-defined]
    _patch_default_score._orig_engine = orig_engine  # type: ignore[attr-defined]


class TracePlayer(FarklePlayer):
    """Subclass of :class:`FarklePlayer` that logs every dice roll."""

    def _roll(self, n: int) -> list[int]:
        """Return ``n`` dice faces and log the result."""
        faces = super()._roll(n)
        LOGGER.info(
            "%s rolls %s",
            self.name,
            faces,
            extra={"stage": "watch"},
        )
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
        Optional seed forwarded to :func:`farkle.utils.random.make_rng` to make
        the game deterministic.
    """
    strategy_seed1, strategy_seed2, player_seed1, player_seed2 = spawn_seeds(4, seed=seed)

    # --- make two random strategies -------------------------------------
    strategy1 = random_threshold_strategy(random.Random(strategy_seed1))
    strategy2 = random_threshold_strategy(random.Random(strategy_seed2))
    LOGGER.info(
        "P1 strategy\n%s\n",
        strategy_yaml(strategy1),
        extra={"stage": "watch"},
    )
    LOGGER.info(
        "P2 strategy\n%s\n",
        strategy_yaml(strategy2),
        extra={"stage": "watch"},
    )

    # monkey-patch the *strategy* layer so we can see decide()
    _trace_decide(strategy1, "P1")
    _trace_decide(strategy2, "P2")

    # monkey-patch the *scoring* layer so we can see Smart-discard results
    with patch_scoring():
        # --- wrap players so we can see every dice throw --------------------
        p1 = TracePlayer(
            "P1",
            strategy1,
            rng=make_rng(player_seed1),
        )
        p2 = TracePlayer(
            "P2",
            strategy2,
            rng=make_rng(player_seed2),
        )

        game = FarkleGame([p1, p2], target_score=10_000)
        metrics = game.play()

    LOGGER.info("\n===== final result =====", extra={"stage": "watch"})
    LOGGER.info(
        "Winner: %s  score=%d  rounds=%d",
        metrics.winner,
        metrics.winning_score,
        metrics.n_rounds,
        extra={"stage": "watch"},
    )


if __name__ == "__main__":
    # run:  python -m farkle.watch_game  (optionally pass a seed)
    watch_game()
