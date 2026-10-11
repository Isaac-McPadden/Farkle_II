# Farkle rules

The simulation uses the scoring and final-round rules below. The authoritative
implementation is [scoring](../src/farkle/game/scoring.py) and the
[game engine](../src/farkle/game/engine.py); strategies decide which available
scoring dice to keep and when to stop.

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

1. Start with six dice and keep at least one scoring die after each roll.
2. A roll without scoring dice is a **Farkle**: lose this turn's accumulated
   points and end the turn.
3. Otherwise stop to bank, or roll the remaining dice. A player must accumulate
   at least 500 in one turn to enter the game; later banked turns can be smaller.
4. If all six dice score across a roll or series of rolls, **hot dice** allows
   another roll of all six. Already banked points cannot be lost.

For example, keep three 1s and a 5 for 350 points, leaving two dice. Rolling
a 1 and a 5 adds 150, reaching 500 and hot dice. Roll six again; three 2s,
a 1 and a 5 add 350. Banking now records 850 points.

## Finish and safety outcomes

Reaching the target score (normally 10,000) triggers a final turn for each other
player. Challengers must exceed the current leading total; the highest final
score wins. Exact turn opportunities and tied score ranks are recorded separately
from the single official winner.

The engine also enforces roll/round safety limits. A safety-limit attempt has
no official winner or ranks. It remains an attempted loss for each tournament
participant and is excluded from winner-conditioned analyses. See the
[turn and row contract](turn_and_row_contract.md) for persisted semantics.
