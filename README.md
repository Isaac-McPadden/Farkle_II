# Farkle_II
The fully tested, software packaged, upgraded version of my FarkleProject
Efficient packaged version of my Farkle Monte Carlo simulation

import farkle as far

Any of these work:
python -m farkle run cfg.yml
python -m farkle.cli run cfg.yml
farkle run cfg.yml               # installed entry-point


Dice threshold -> I must have at least n dice to keep rolling (Inclusive Down)

Score threshold -> I stop at this number or higher (Inclusive Up)

5.3 Stat-power vs player-count
n_games ≥ 2·(z_α + z_β)² · p(1-p) / δ²
