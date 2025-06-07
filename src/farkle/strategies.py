from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List

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



__all__: list[str] = [
    "ThresholdStrategy",
    "random_threshold_strategy",
]

DiceRoll = List[int]
"""A list of integers 1-6 representing a single dice roll."""

# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------

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
    smart_five: bool = False   # “throw back lone 5’s first”
    smart_one:  bool = False
    consider_score: bool = True
    consider_dice: bool = True
    require_both: bool = False
    auto_hot_dice: bool = False
    run_up_score: bool = False
    prefer_score: bool = True
    
    def __post_init__(self):
        # 1) smart_one may never be True if smart_five is False
        if self.smart_one and not self.smart_five:
            raise ValueError(
                "ThresholdStrategy: smart_one=True requires smart_five=True"
            )

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
        score_needed: int,          # noqa: ARG002 (vestigial/reserved for richer strats)
        final_round: bool = False,  # (new end-game hooks – harmless defaults)
        score_to_beat: int = 0,
        running_total: int = 0,
    ) -> bool:  # noqa: D401 – imperative name
        """
        Return **True** to keep rolling, *False** to bank.
        
        Counterintuitively, require_both = True is riskier play
        
        Outcomes of cominations of consider_score = True, consider_dice = True, 
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
            return True                     # must cross the 500-pt entry gate

        # final-round catch-up rule
        if final_round:
            return running_total <= score_to_beat

        # -----------------------------------------------------------------------------  
        want_score = self.consider_score and turn_score < self.score_threshold
        want_dice  = self.consider_dice and dice_left  > self.dice_threshold

        # both score & dice active ----------------------------------------------------
        if self.consider_score and self.consider_dice:
            if self.require_both:
                # stop only when **both** limits are satisfied
                return want_score or want_dice
            else:
                # stop as soon as **either** limit is satisfied
                return want_score and want_dice

        # single-axis strategies ------------------------------------------------------
        if self.consider_score:
            return want_score
        if self.consider_dice:
            return want_dice

        # nothing left to consider → always bank
        return False

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __str__(self) -> str:  # noqa: D401 - magics method
        cs = "S" if self.consider_score else "-"
        cd = "D" if self.consider_dice else "-"
        sf = "F" if self.smart_five else "-"
        so = "O" if self.smart_one else "-"
        rb = "AND" if self.require_both else "OR"
        hd = "H" if self.auto_hot_dice else "-"
        rs = "R" if self.run_up_score else "-"
        ps = "PS" if self.prefer_score else "PD"
        return f"Strat({self.score_threshold},{self.dice_threshold})[{cs}{cd}][{sf}{so}{ps}][{rb}][{hd}{rs}]"


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
# strategies.py
def _sample_prefer_score(cs: bool, cd: bool, rng: random.Random) -> bool:
    """
    Return the *only* legal value(s) for `prefer_score`
    given the (consider_score, consider_dice) pair.

        cs  cd   →  prefer_score
        ─────────────────────────
        T   F       True    (always favour score)
        F   T       False   (always favour dice)
        T   T       rng     (tie-break random)
        F   F       rng     (doesn’t matter – random)

    Much easier to read than a stacked ternary.
    """
    if cs == cd:                 # (T,T) or (F,F)   →  free choice
        return rng.choice([True, False])
    return cs                    # (T,F) or (F,T)



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
    ps = _sample_prefer_score(cs, cd, rng_inst)

    return ThresholdStrategy(
        score_threshold=rng_inst.randrange(50, 1000, 50),
        dice_threshold=rng_inst.randint(0, 4),
        smart_five=sf,
        smart_one=so,
        consider_score=cs,
        consider_dice=cd,
        require_both=rb,
        prefer_score=ps,
    )


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
    #     prefer_score   = "P" or "-"
    #
    # Example literal: "Strat(300,2)[SD][F-O][AND][H-]"

    pattern = re.compile(
        r"""
        \A
        Strat\(\s*(?P<score>\d+)\s*,\s*(?P<dice>\d+)\s*\)   # thresholds
        \[
            (?P<cs>[S\-])(?P<cd>[D\-])
        \]
        \[
            (?P<sf>[F\-])  # smart_five block
            (?P<so>[O\-])  # smart_one block
            (?P<ps>PS|PD)
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

    m = pattern.match(s)
    if not m:
        raise ValueError(f"Cannot parse strategy string: {s!r}")

    score_threshold = int(m.group("score"))
    dice_threshold  = int(m.group("dice"))

    cs_flag = m.group("cs") == "S"
    cd_flag = m.group("cd") == "D"

    sf_token = m.group("sf")  # "F", "-F", or "--"
    sf_flag = bool(sf_token.startswith("F"))

    so_token = m.group("so")  # "O", "-O", or "--"
    so_flag = bool(so_token.startswith("O"))
    
    ps_flag = m.group("ps") == "PS"

    rb_token = m.group("rb")  # "AND" or "OR"
    require_both = rb_token == "AND"

    hd_flag = m.group("hd") == "H"
    rs_flag = m.group("rs") == "R"

    return ThresholdStrategy(
        score_threshold=score_threshold,
        dice_threshold=dice_threshold,
        smart_five=sf_flag,
        smart_one=so_flag,
        consider_score=cs_flag,
        consider_dice=cd_flag,
        require_both=require_both,
        auto_hot_dice=hd_flag,
        run_up_score=rs_flag,
        prefer_score=ps_flag,
    )
