# Farkle rules

The simulation uses the scoring and final-round rules below. The authoritative
implementation is in [scoring](../src/farkle/game/scoring.py), the
[game engine](../src/farkle/game/engine.py), and the
[threshold strategy](../src/farkle/simulation/strategies.py).

## Scoring

Combinations use dice from the same roll; dice saved on earlier rolls cannot
complete a new combination.

| Dice | Points |
| --- | ---: |
| Single 1 | 100 |
| Single 5 | 50 |
| Three 1s | 300 |
| Three 2s / 3s / 4s / 5s / 6s | 200 / 300 / 400 / 500 / 600 |
| Four of a kind | 1,000 |
| Five of a kind | 2,000 |
| Six of a kind | 3,000 |
| Straight 1-6 | 1,500 |
| Three pairs | 1,500 |
| Four of a kind plus a pair | 1,500 |
| Two triplets | 2,500 |

## Turn flow

Start with six dice. A roll without scoring dice is a **Farkle** and immediately
ends the turn. The player loses all unbanked points and keeps their previous
banked score. On a scoring roll, keep at least one scoring die. If this uses
all dice available on that roll, **hot dice** resets the next roll to six dice.

Apply the following decisions in order after selecting scoring dice.

1. A player's first bank requires at least 500 points in one turn. Until then
   they must roll again. After entry they can bank smaller amounts.
2. A final challenger must roll again while their banked score plus turn points
   is tied with or below the current leader.
3. A final challenger who can legally bank a strict lead must do so immediately
   when `run_up_score=False`, even on hot dice. With `run_up_score=True`, apply
   the following strategy rules. An ordinary player also uses these rules,
   including on the roll that reaches or crosses the target.
4. On hot dice, `auto_hot_dice=True` requires another roll. Otherwise evaluate
   the enabled banking criteria. The score criterion is met at or above the
   turn-score threshold. The dice criterion is met when the number available
   for the next roll is at or below the dice threshold, counting six after a
   reset. A single enabled criterion governs banking on its own. With both
   enabled, `require_both=True` requires both to bank and `require_both=False`
   accepts either. With neither enabled, the threshold strategy banks.

For example, keep three 1s and a 5 for 350 points, leaving two dice. Rolling
a 1 and a 5 adds 150, reaching 500 and hot dice. If the player rolls again,
three 2s, a 1 and a 5 add 350. Banking now records 850 points.

## Finish and safety outcomes

Banking a total at or above the target (normally 10,000) gives every opponent
exactly one challenge turn in normal cyclic order. For example, when P2
triggers in a four-player game, P3 challenges first, followed by P4 and P1.
Each challenger who banks a strict lead sets the score the next challenger
must beat. A Farkle leaves the leading banked score in place. The highest
final banked score wins. Seat and challenge-order advantages are intentional.
The game records each player's actual turn count and assigns an official
finishing rank. Players with equal losing scores are ordered by original seat.

If nobody triggers by the end of the last permitted round, the attempt has no
official winner or ranks. It counts as an attempted loss for every participant
and is excluded from winner-conditioned analyses.
A trigger in the last permitted round still grants all challenge turns.
The per-turn roll limit is a separate guard. If another roll would exceed it,
the engine raises an error before rolling. See the
[turn and row contract](turn_and_row_contract.md) for persisted semantics.

## Historical results

Games produced at the Phase 6C-A baseline (`a08bc7a3`) visited opponents from
original seat one, skipping the trigger. Automatic hot dice also took priority
over banking a final lead. Historical artifacts retain their producer identity
and describe play under those rules. The revised rules change simulation
identity and require fresh tournament and head-to-head observations.
