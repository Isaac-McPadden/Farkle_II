# src/farkle/strategies.py
"""strategies.py
================
Strategy abstractions used by the Farkle simulation engine.  A *strategy*
object decides **whether to continue rolling** based on the current turn
context, independent of dice-scoring rules.

Classes
-------
`ThresholdStrategy` - simple heuristic combining score & dice limits plus
an optional Smart-5 flag.

Functions
---------
`random_threshold_strategy` - convenience generator for Monte-Carlo
sweeps.
"""
from __future__ import annotations

import pickle
import random
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numba as nb
import pandas as pd

from farkle.types import DiceRoll  # noqa: F401 Likely needed for decide(*)

__all__: list[str] = [
    "FavorDiceOrScore",
    "ThresholdStrategy",
    "random_threshold_strategy",
]


class FavorDiceOrScore(Enum):
    """Tie-break preference when both score and dice targets are hit."""

    SCORE = "score"
    DICE = "dice"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


_STRAT_RE = re.compile(
    r"""
    \A
    Strat\(\s*(?P<score>\d+)\s*,\s*(?P<dice>\d+)\s*\)  # thresholds
    \[
        (?P<cs>[S\-])(?P<cd>[D\-])
    \]
    \[
        (?P<sf>[F\-])  # smart_five block
        (?P<so>[O\-])  # smart_one block
        (?P<fs>FS|FD)
    \]
    \[
        (?P<rb>AND|OR)
    \]
    \[
        (?P<hd>[H\-])(?P<rs>[R\-])
    \]
    \Z
    """,
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------


@nb.njit(cache=True)
def _decide_continue(
    turn_score,
    dice_left,
    score_threshold,
    dice_threshold,
    consider_score,
    consider_dice,
    require_both,
) -> bool:
    """Return ``True`` to keep rolling based on score/dice thresholds.

    Parameters
    ----------
    turn_score, dice_left
        Current turn score and dice remaining.
    score_threshold, dice_threshold
        Limits governing when to stop rolling.
    consider_score, consider_dice
        Flags enabling the above limits.
    require_both
        When both consider_score and consider_dice flags are set to ``True``, 
        if ``require_both`` is ``True`` the function continues when *either* 
        limit is still unmet (``OR``); 
        if ``False`` it continues only when *both* limits are unmet (``AND``).
    """
    # The function asks if we continue while the thresholds dictate when to stop.
    # As a result, the logic here looks incorrect at first glance but it is correct.
    # want_s and want_d are checking if limits are NOT hit for easier boolean logic.
    want_s = consider_score and turn_score < score_threshold  # want higher score
    want_d = consider_dice and dice_left > dice_threshold  # want to spend more dice
    if consider_score and consider_dice:  # the booleans here are counterintuitive but correct
        return (want_s or want_d) if require_both else (want_s and want_d)  
    if consider_score:
        return want_s
    if consider_dice:
        return want_d
    return False


@dataclass
class ThresholdStrategy:
    """Threshold-based decision rule.

    Parameters
    ----------
    score_threshold
        Keep rolling until turn score reaches this number (subject to
        *consider_score*).
    dice_threshold
        Bank when dice-left drop to this value or below (subject to
        *consider_dice*).
    smart_five, smart_one
        Enables/Disables Smart-5 and Smart-1 heuristics during scoring.
    consider_score, consider_dice
        Toggle the two conditions to reproduce *score-only*, *dice-only*,
        or *balanced* play styles.
    """

    score_threshold: int = 300
    dice_threshold: int = 2
    smart_five: bool = False  # “throw back lone 5’s first”
    smart_one: bool = False
    consider_score: bool = True
    consider_dice: bool = True
    require_both: bool = False
    auto_hot_dice: bool = False
    run_up_score: bool = False
    favor_dice_or_score: FavorDiceOrScore = FavorDiceOrScore.SCORE
    
    def __post_init__(self):
        # 1) smart_one may never be True if smart_five is False
        if self.smart_one and not self.smart_five:
            raise ValueError("ThresholdStrategy: smart_one=True requires smart_five=True")

        # 2) require_both may only be True if both consider_score and consider_dice are True
        if self.require_both and not (self.consider_score and self.consider_dice):
            raise ValueError(
                "ThresholdStrategy: require_both=True requires both "
                "consider_score=True and consider_dice=True"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        *,
        turn_score: int,
        dice_left: int,
        has_scored: bool,
        score_needed: int,  # noqa: ARG002 (vestigial/reserved for richer strats)
        final_round: bool = False,  # (new end-game hooks – harmless defaults)
        score_to_beat: int = 0,
        running_total: int = 0,
    ) -> bool:  # noqa: D401 – imperative name
        """
        Return **True** to keep rolling, *False** to bank.

        Counterintuitively, require_both = True is riskier play
        
        Outcomes of combinations of consider_score = True, consider_dice = True,
        require_both = [True, False] for score_threshold = 300 and dice_threshold = 3:

        cs and cd are True, require_both = True (AND logic)
        (400, 4, True),  # Enough dice but too many points
        (200, 2, True),  # Low enough points but not enough dice
        (200, 4, True),   # Low enough points and enough dice available
        (400, 2, False),  # # Too many points and not enough dice available

        cs and cd are True, require_both = False (OR logic)
        (400, 4, False),  # Enough dice but too many points
        (200, 2, False),  # Low enough points but not enough dice
        (200, 4, True),   # Low enough points and enough dice available
        (400, 2, False),  # # Too many points and not enough dice available
        """

        # --------------------------------- fast exits ---------------------------------
        if not has_scored and turn_score < 500:
            return True  # must cross the 500-pt entry gate

        # final-round catch-up rule
        if final_round:
            return running_total <= score_to_beat

        # -----------------------------------------------------------------------------
        keep_rolling = _decide_continue(
            turn_score,
            dice_left,
            self.score_threshold,
            self.dice_threshold,
            self.consider_score,
            self.consider_dice,
            self.require_both,
        )

        return keep_rolling

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------


    def __str__(self) -> str:  # noqa: D401 - magic method
        cs = "S" if self.consider_score else "-"
        cd = "D" if self.consider_dice else "-"
        sf = "F" if self.smart_five else "-"
        so = "O" if self.smart_one else "-"
        rb = "AND" if self.require_both else "OR"
        hd = "H" if self.auto_hot_dice else "-"
        rs = "R" if self.run_up_score else "-"
        fs = "FS" if self.favor_dice_or_score is FavorDiceOrScore.SCORE else "FD"
        return f"Strat({self.score_threshold},{self.dice_threshold})[{cs}{cd}][{sf}{so}{fs}][{rb}][{hd}{rs}]"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def _sample_favor_score(cs: bool, cd: bool, rng: random.Random) -> FavorDiceOrScore:
    """
    Return the *only* legal value(s) for `favor_dice_or_score`
    given the (consider_score, consider_dice) pair.

        cs  cd   →  favor_dice_or_score
        ─────────────────────────
        T   F       True    (always favour score)
        F   T       False   (always favour dice)
        T   T       rng     (tie-break random)
        F   F       rng     (doesn't matter - random)

    Much easier to read than a stacked ternary.
    """
    if cs == cd:  # (T,T) or (F,F)   →  free choice
        return rng.choice([FavorDiceOrScore.SCORE, FavorDiceOrScore.DICE])
    return FavorDiceOrScore.SCORE if cs else FavorDiceOrScore.DICE


def random_threshold_strategy(rng: random.Random | None = None) -> ThresholdStrategy:
    """Return a random ThresholdStrategy that always satisfies the two constraints."""

    rng_inst = rng if rng is not None else random.Random()

    # pick smart_five first; if it’s False, force smart_one=False
    sf = rng_inst.choice([True, False])
    so = rng_inst.choice([True, False]) if sf else False

    # pick consider_score/dice; if either is False, force require_both=False
    cs = rng_inst.choice([True, False])
    cd = rng_inst.choice([True, False])
    rb = rng_inst.choice([True, False]) if (cs and cd) else False
    fs = _sample_favor_score(cs, cd, rng_inst)

    return ThresholdStrategy(
        score_threshold=rng_inst.randrange(50, 1000, 50),
        dice_threshold=rng_inst.randint(0, 4),
        smart_five=sf,
        smart_one=so,
        consider_score=cs,
        consider_dice=cd,
        require_both=rb,
        favor_dice_or_score=fs,
    )


def _parse_strategy_flags(s: str) -> dict[str, Any]:
    """Return a mapping of strategy fields parsed from ``s``."""

    m = _STRAT_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse strategy string: {s!r}")

    return {
        "score_threshold": int(m.group("score")),
        "dice_threshold": int(m.group("dice")),
        "smart_five": m.group("sf") == "F",
        "smart_one": m.group("so") == "O",
        "consider_score": m.group("cs") == "S",
        "consider_dice": m.group("cd") == "D",
        "require_both": m.group("rb") == "AND",
        "auto_hot_dice": m.group("hd") == "H",
        "run_up_score": m.group("rs") == "R",
        "favor_dice_or_score": (
            FavorDiceOrScore.SCORE if m.group("fs") == "FS" else FavorDiceOrScore.DICE
        ),
    }


def parse_strategy(s: str) -> ThresholdStrategy:
    """
    Reverse of ThresholdStrategy.__str__.

    Accepts strings of the form:
      Strat(300,2)[SD][FO][AND][H-]
    or (for example):
      Strat(500,1)[S-][-O][OR][-R]

    Returns a ThresholdStrategy(...) with all booleans parsed.

    NOTE: this is meant for analysis/log-parsing, not the hot path.
    """
    # 1) Top-level pattern:  Strat(score_threshold, dice_threshold)
    # 2) Four bracketed blocks:
    #    [<consider_score><consider_dice>]
    #    [<smart_five><smart_one>]
    #    [AND|OR]
    #    [<auto_hot><run_up>]
    #
    #   where:
    #     consider_score = "S" or "-"
    #     consider_dice  = "D" or "-"
    #     smart_five     = "F" or "-"
    #     smart_one      = "O" or "-"
    #     auto_hot       = "H" or "-"
    #     run_up_score   = "R" or "-"
    #     require_both   = "AND" or "OR"
    #     favor_dice_or_score   = "FS" or "FD"
    #
    # Example literal: "Strat(300,2)[SD][FOFD][AND][H-]"

    flags = _parse_strategy_flags(s)
    return ThresholdStrategy(**flags)


def parse_strategy_for_df(s: str) -> dict:
    """
    Reverse of ThresholdStrategy.__str__.

    Accepts strings of the form:
      Strat(300,2)[SD][FO][AND][H-]
    or (for example):
      Strat(500,1)[S-][-O][OR][-R]

    Returns a dictionary with all booleans parsed.

    NOTE: this is meant for analysis/log-parsing, not the hot path.
    """
    # 1) Top-level pattern:  Strat(score_threshold, dice_threshold)
    # 2) Four bracketed blocks:
    #    [<consider_score><consider_dice>]
    #    [<smart_five><smart_one>]
    #    [AND|OR]
    #    [<auto_hot><run_up>]
    #
    #   where:
    #     consider_score = "S" or "-"
    #     consider_dice  = "D" or "-"
    #     smart_five     = "F" or "-"
    #     smart_one      = "O" or "-"
    #     auto_hot       = "H" or "-"
    #     run_up_score   = "R" or "-"
    #     require_both   = "AND" or "OR"
    #     favor_dice_or_score   = "FS" or "FD"
    #
    # Example literal: "Strat(300,2)[SD][FOFD][AND][H-]"

    return _parse_strategy_flags(s)


def load_farkle_results(
    pkl_path: str | Path,
    *,
    parse_strategy: Callable[[str], dict] = parse_strategy_for_df,
    ordered: bool = True,
) -> pd.DataFrame:
    """
    Load a pickled Counter {strategy_str: wins} and explode it into a
    “full-fat” DataFrame with every strategy flag broken out.

    Warning
    -------
    Unpickling data from untrusted sources can execute arbitrary code.
    Only load pickle files from trusted sources.

    Parameters
    ----------
    pkl_path : str | Path
        Path to the pickle file produced by `run_tournament_2.py`.
        The pickle must contain a `collections.Counter` or plain dict
        whose keys are strategy strings and whose values are win counts.
    parse_strategy : callable, default ``parse_strategy_for_df``
        Function that converts one strategy string to a dict of columns.
        Override only if you have a different parser.
    ordered : bool, default True
        Whether to return the columns in a logical, pre-defined order.

    Returns
    -------
    pd.DataFrame
        Columns:
          strategy, wins,
          score_threshold, dice_threshold,
          consider_score, consider_dice, require_both, favor_dice_or_score,
          smart_five, smart_one, auto_hot_dice, run_up_score
    """
    # ------------------------------------------------------------------
    # 1) Load the Counter
    # ------------------------------------------------------------------
    pkl_path = Path(pkl_path).expanduser().resolve()
    counter = pickle.loads(pkl_path.read_bytes())

    # ------------------------------------------------------------------
    # 2) Counter → two-column DataFrame
    # ------------------------------------------------------------------
    base_df = (
        pd.Series(counter, name="wins")
        .reset_index(drop=False)
        .rename(columns={"index": "strategy"})
    )

    # ------------------------------------------------------------------
    # 3) Explode strategy strings into individual columns
    # ------------------------------------------------------------------
    flags_df = (
        base_df["strategy"].apply(parse_strategy).apply(pd.Series)  # str → dict  # dict → DataFrame
    )

    full_df = pd.concat([base_df, flags_df], axis=1)

    # ------------------------------------------------------------------
    # 4) Nice-to-have column ordering
    # ------------------------------------------------------------------
    if ordered:
        col_order = [
            "strategy",
            "wins",
            "score_threshold",
            "dice_threshold",
            "consider_score",
            "consider_dice",
            "require_both",
            "smart_five",
            "smart_one",
            "favor_dice_or_score",
            "auto_hot_dice",
            "run_up_score",
        ]
        full_df = full_df[col_order].sort_values(by="wins", ascending=False, ignore_index=True)

    return full_df
